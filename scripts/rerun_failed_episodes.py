#!/usr/bin/env python3
"""
Rerun failed episodes from a previous experiment run.

Parses failed_episodes.json, extracts configuration info, and reruns
those specific episodes.
"""

import json
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import time
import random

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.runner import ExperimentRunner
from agents.observer import ObserverAgent
from agents.actor import ActorAgent
from agents.ace import ACEAgent


def parse_failed_episode(failed_json_str):
    """Parse failed episode JSON and extract configuration."""
    try:
        data = json.loads(failed_json_str)
        episode_id = data.get('episode_id', '')

        # Parse environment, agent, and episode number from ID
        # Format: {env}_{agent}_ep{num}
        if 'observer' in episode_id:
            agent = 'observer'
            env_part = episode_id.split('_observer')[0]
        elif 'actor' in episode_id:
            agent = 'actor'
            env_part = episode_id.split('_actor')[0]
        elif 'a_c_e' in episode_id:
            agent = 'a_c_e'
            env_part = episode_id.split('_a_c_e')[0]
        else:
            return None

        # Extract episode number
        ep_num = int(episode_id.split('_ep')[-1])

        return {
            'episode_id': episode_id,
            'environment': env_part,
            'agent': agent,
            'episode_num': ep_num,
            'original_error': data.get('error_type', 'Unknown')
        }
    except Exception as e:
        print(f"Warning: Failed to parse episode: {e}")
        return None


def load_failed_episodes(failed_file):
    """Load and parse all failed episodes."""
    with open(failed_file) as f:
        failed_list = json.load(f)

    parsed = []
    for failed_json in failed_list:
        ep = parse_failed_episode(failed_json)
        if ep:
            parsed.append(ep)

    return parsed


def get_seed_for_episode(config, environment, episode_num):
    """Get the seed for a specific episode number in an environment."""
    env_config = config['environments'].get(environment)
    if not env_config:
        return None

    seeds = env_config.get('seeds', [])
    if episode_num > len(seeds):
        return None

    # Episode numbers are 1-indexed, seeds list is 0-indexed
    return seeds[episode_num - 1]


def run_single_episode_with_retry(args):
    """Run a single episode with retry logic for API errors."""
    episode_id, environment, agent_type, seed, config, output_dir, max_retries = args

    for attempt in range(max_retries):
        try:
            # Add jitter to avoid thundering herd
            if attempt > 0:
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"  Retry {attempt}/{max_retries} for {episode_id} after {sleep_time:.1f}s...")
                time.sleep(sleep_time)

            # Create runner
            runner = ExperimentRunner(config)

            # Create agent
            if agent_type == 'observer':
                agent = ObserverAgent(
                    model_config=config['models']['observer'],
                    token_budget=config['budgets']['tokens_per_call']
                )
            elif agent_type == 'actor':
                agent = ActorAgent(
                    model_config=config['models']['actor'],
                    token_budget=config['budgets']['tokens_per_call']
                )
            elif agent_type == 'a_c_e':
                agent = ACEAgent(
                    model_config=config['models']['a_c_e'],
                    token_budget=config['budgets']['tokens_per_call']
                )
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

            # Run episode
            result = runner.run_episode(
                agent=agent,
                environment_name=environment,
                seed=seed,
                episode_id=episode_id,
                actions_budget=config['budgets']['actions_per_episode']
            )

            # Save result
            output_file = Path(output_dir) / 'raw' / f"{episode_id}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"✅ Completed: {episode_id} (attempt {attempt + 1})")
            return {'episode_id': episode_id, 'status': 'success', 'attempts': attempt + 1}

        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1 and ('Overloaded' in error_msg or '500' in error_msg or 'overloaded' in error_msg.lower()):
                # Retry on API overload errors
                continue
            else:
                # Final failure or non-retryable error
                print(f"❌ Failed: {episode_id} after {attempt + 1} attempts - {error_msg[:100]}")
                return {
                    'episode_id': episode_id,
                    'status': 'failed',
                    'error': error_msg,
                    'attempts': attempt + 1
                }

    return {'episode_id': episode_id, 'status': 'failed', 'error': 'Max retries exceeded'}


def main():
    parser = argparse.ArgumentParser(description='Rerun failed episodes')
    parser.add_argument('--results-dir', required=True, help='Results directory containing failed_episodes.json')
    parser.add_argument('--config', required=True, help='Config file used for original run')
    parser.add_argument('--workers', type=int, default=3, help='Number of parallel workers (default: 3)')
    parser.add_argument('--max-retries', type=int, default=5, help='Max retry attempts per episode (default: 5)')
    parser.add_argument('--dry-run', action='store_true', help='Print episodes to rerun without executing')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    failed_file = results_dir / 'failed_episodes.json'

    if not failed_file.exists():
        print(f"Error: {failed_file} not found")
        sys.exit(1)

    # Load config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load failed episodes
    print("Loading failed episodes...")
    failed_episodes = load_failed_episodes(failed_file)
    print(f"Found {len(failed_episodes)} failed episodes to rerun")

    # Map environment names
    env_name_map = {
        'hot_pot': 'HotPotLab',
        'switch_light': 'SwitchLight',
        'chem_tile': 'ChemTile'
    }

    # Prepare episodes for rerun
    episodes_to_run = []
    for ep in failed_episodes:
        # Map environment name to class name
        env_name = env_name_map.get(ep['environment'], ep['environment'])

        # Get seed from config
        seed = get_seed_for_episode(config, ep['environment'], ep['episode_num'])
        if seed is None:
            print(f"Warning: Could not find seed for {ep['episode_id']}")
            continue

        episodes_to_run.append({
            'episode_id': ep['episode_id'],
            'environment': env_name,
            'agent': ep['agent'],
            'seed': seed,
            'original_error': ep['original_error']
        })

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Would rerun {len(episodes_to_run)} episodes:")
        for ep in episodes_to_run[:10]:
            print(f"  {ep['episode_id']} - {ep['environment']} - {ep['agent']} - seed={ep['seed']}")
        if len(episodes_to_run) > 10:
            print(f"  ... and {len(episodes_to_run) - 10} more")
        return

    print(f"\n=== RERUNNING {len(episodes_to_run)} EPISODES ===")
    print(f"Workers: {args.workers}")
    print(f"Max retries per episode: {args.max_retries}")
    print()

    # Prepare arguments for parallel execution
    run_args = [
        (
            ep['episode_id'],
            ep['environment'],
            ep['agent'],
            ep['seed'],
            config,
            results_dir,
            args.max_retries
        )
        for ep in episodes_to_run
    ]

    # Run episodes in parallel
    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_episode_with_retry, arg): arg for arg in run_args}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            completed = len(results)
            total = len(run_args)
            print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")

    # Summary
    elapsed = time.time() - start_time
    successes = sum(1 for r in results if r['status'] == 'success')
    failures = len(results) - successes

    print("\n" + "="*70)
    print("RERUN SUMMARY")
    print("="*70)
    print(f"Total episodes: {len(results)}")
    print(f"Successful: {successes}")
    print(f"Failed: {failures}")
    print(f"Success rate: {successes/len(results)*100:.1f}%")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print()

    if failures > 0:
        print("Still-failed episodes:")
        for r in results:
            if r['status'] == 'failed':
                print(f"  {r['episode_id']}: {r.get('error', 'Unknown')[:80]}")

        # Update failed_episodes.json with only the still-failed ones
        still_failed = []
        with open(failed_file) as f:
            original_failed = json.load(f)

        failed_ids = {r['episode_id'] for r in results if r['status'] == 'failed'}
        for failed_json in original_failed:
            try:
                data = json.loads(failed_json)
                if data.get('episode_id') in failed_ids:
                    # Update timestamp
                    data['timestamp'] = datetime.now().isoformat()
                    data['rerun_attempted'] = True
                    still_failed.append(json.dumps(data))
            except:
                pass

        # Save updated failed list
        with open(failed_file, 'w') as f:
            json.dump(still_failed, f, indent=2)

        print(f"\nUpdated {failed_file} with {len(still_failed)} still-failed episodes")
    else:
        # All succeeded - clear failed_episodes.json
        with open(failed_file, 'w') as f:
            json.dump([], f, indent=2)
        print(f"\n✅ All episodes succeeded! Cleared {failed_file}")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
