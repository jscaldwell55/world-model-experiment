#!/usr/bin/env python3
"""
Parallel experiment runner with intelligent rate limiting.

Runs multiple episodes concurrently while respecting Anthropic API rate limits:
- Requests per minute: 1,000
- Input tokens per minute: 450,000
- Output tokens per minute: 90,000

Uses ThreadPoolExecutor with configurable workers and real-time progress tracking.
"""

import argparse
import json
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import load_config
from experiments.runner import ExperimentRunner
from experiments.rate_limiter import RateLimiter, get_token_estimate
from environments.hot_pot import HotPotLab
from environments.switch_light import SwitchLight
from environments.chem_tile import ChemTile
from agents.observer import ObserverAgent
from agents.actor import ActorAgent
from agents.text_reader import TextReaderAgent
from agents.ace import ACEAgent
from agents.simple_world_model import SimpleWorldModel


# Global shutdown flag for graceful termination
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    print("\n\n⚠️  Shutdown requested... waiting for running episodes to complete...")
    shutdown_requested = True


def run_single_episode_with_retry(
    runner: ExperimentRunner,
    episode_id: str,
    seed: int,
    save_dir: Path,
    rate_limiter: RateLimiter,
    max_retries: int = 3
) -> Tuple[bool, Optional[dict], Optional[str]]:
    """
    Run a single episode with retry logic.

    Args:
        runner: ExperimentRunner instance
        episode_id: Episode identifier
        seed: Random seed
        save_dir: Directory to save results
        rate_limiter: RateLimiter instance
        max_retries: Maximum retry attempts

    Returns:
        Tuple of (success, result_dict, error_message)
    """
    for attempt in range(max_retries):
        try:
            result = runner.run_episode(
                episode_id=episode_id,
                seed=seed,
                save_dir=save_dir,
                rate_limiter=rate_limiter
            )
            return (True, result, None)

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)

            # Check if this is a rate limit error
            is_rate_limit = 'rate' in error_msg.lower() or 'RateLimitError' in error_type

            # Check if this is a network error
            is_network = 'ConnectionError' in error_type or 'Timeout' in error_type

            # Determine if we should retry
            should_retry = attempt < max_retries - 1 and (is_rate_limit or is_network)

            if should_retry:
                if is_rate_limit:
                    # Exponential backoff for rate limits
                    wait_time = 60 * (2 ** attempt)  # 60s, 120s, 240s
                    print(f"⚠️  Rate limit hit on {episode_id}, retry {attempt+1}/{max_retries} after {wait_time}s")
                    time.sleep(wait_time)
                elif is_network:
                    # Quick retry for network errors
                    wait_time = 5
                    print(f"⚠️  Network error on {episode_id}, retry {attempt+1}/{max_retries} after {wait_time}s")
                    time.sleep(wait_time)
            else:
                # Final failure - return error
                error_details = {
                    'episode_id': episode_id,
                    'error_type': error_type,
                    'error_message': error_msg,
                    'traceback': traceback.format_exc(),
                    'timestamp': datetime.now().isoformat()
                }
                return (False, None, json.dumps(error_details, indent=2))

    # Should not reach here
    return (False, None, "Max retries exceeded")


def run_parallel_experiment(
    config_path: str,
    preregistration_path: str,
    output_dir: Optional[str] = None,
    workers: int = 10,
    resume_from: Optional[str] = None
):
    """
    Run experiment with parallel execution and rate limiting.

    Args:
        config_path: Path to config.yaml
        preregistration_path: Path to preregistration.yaml
        output_dir: Output directory (default: timestamped)
        workers: Number of parallel workers
        resume_from: Directory to resume from (skip completed episodes)
    """
    global shutdown_requested

    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Load config
    config = load_config(config_path)

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('results') / f'parallel_run_{timestamp}'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = output_dir / 'raw'
    save_dir.mkdir(exist_ok=True)

    # Create rate limiter (use 90% of limits as safety buffer)
    rate_limiter = RateLimiter(
        rpm=1000,
        input_tpm=450000,
        output_tpm=90000,
        safety_factor=0.9
    )

    # Print header
    print("=" * 70)
    print("PARALLEL EXECUTION WITH RATE LIMITING")
    print("=" * 70)
    print(f"Workers: {workers}")
    print("Rate limits:")
    print(f"  - Requests: 1,000/min (using ≤900/min)")
    print(f"  - Input tokens: 450,000/min (using ≤405,000/min)")
    print(f"  - Output tokens: 90,000/min (using ≤81,000/min)")
    print("=" * 70)
    print()

    # Build episode list
    episodes = []
    environment_mapping = {
        'hot_pot': HotPotLab,
        'switch_light': SwitchLight,
        'chem_tile': ChemTile
    }
    agent_mapping = {
        'observer': ObserverAgent,
        'actor': ActorAgent,
        'text_reader': TextReaderAgent,
        'a_c_e': ACEAgent,
        'simple_world_model': SimpleWorldModel
    }

    # Get number of epochs (default 1 for non-ACE agents, or from config for ACE)
    ace_config = config.get('ace_config', {})
    max_epochs = ace_config.get('max_epochs', 1)

    for env_name, env_cls in environment_mapping.items():
        if env_name not in config.get('environments', {}):
            continue

        env_config = config['environments'][env_name]
        num_episodes = env_config.get('num_episodes', 0)
        seeds = env_config.get('seeds', list(range(42, 42 + num_episodes)))

        for agent_name in config.get('agents', agent_mapping.keys()):
            agent_cls = agent_mapping[agent_name]

            # Determine epochs for this agent (only ACE uses multi-epoch)
            epochs = max_epochs if agent_name == 'a_c_e' else 1

            for epoch in range(epochs):
                for i, seed in enumerate(seeds):
                    # Include epoch in ID if multi-epoch
                    if epochs > 1:
                        episode_id = f"{env_name}_{agent_name}_epoch{epoch+1}_ep{str(i+1).zfill(3)}"
                    else:
                        episode_id = f"{env_name}_{agent_name}_ep{str(i+1).zfill(3)}"

                    episodes.append({
                        'episode_id': episode_id,
                        'env_name': env_name,
                        'env_cls': env_cls,
                        'agent_name': agent_name,
                        'agent_cls': agent_cls,
                        'seed': seed,
                        'epoch': epoch,
                        'max_epochs': epochs
                    })

    # Check for resume
    completed_ids = set()
    if resume_from:
        resume_dir = Path(resume_from) / 'raw'
        if resume_dir.exists():
            for log_file in resume_dir.glob('*.json'):
                completed_ids.add(log_file.stem)
            print(f"Resuming: skipping {len(completed_ids)} completed episodes")
            episodes = [ep for ep in episodes if ep['episode_id'] not in completed_ids]

    total_episodes = len(episodes)
    print(f"Total episodes to run: {total_episodes}")

    # Estimate time
    avg_time_per_episode = 120  # seconds (estimate)
    est_sequential_time = total_episodes * avg_time_per_episode / 60
    est_parallel_time_min = est_sequential_time / workers
    est_parallel_time_max = est_sequential_time / (workers * 0.5)  # Account for rate limiting
    print(f"Estimated time: {est_parallel_time_min:.0f}-{est_parallel_time_max:.0f} minutes with {workers} workers")
    print()

    # Track progress
    completed_count = 0
    failed_count = 0
    start_time = time.time()
    lock = threading.Lock()
    failed_episodes = []

    # For ACE agents, maintain shared agent instances per (env, agent) pair to persist playbooks
    ace_agent_cache = {}

    def run_episode_wrapper(episode_info):
        """Wrapper to run episode and track progress."""
        nonlocal completed_count, failed_count

        # Check shutdown flag
        if shutdown_requested:
            return None

        # For ACE multi-epoch, use a shared agent instance
        shared_agent = None
        if episode_info['agent_name'] == 'a_c_e' and episode_info['max_epochs'] > 1:
            cache_key = f"{episode_info['env_name']}_{episode_info['agent_name']}"
            shared_agent = ace_agent_cache.get(cache_key)

        # Create runner for this episode
        runner = ExperimentRunner(
            config=config,
            environment_cls=episode_info['env_cls'],
            agent_cls=episode_info['agent_cls'],
            shared_agent=shared_agent
        )

        # Run episode with retry
        success, result, error = run_single_episode_with_retry(
            runner=runner,
            episode_id=episode_info['episode_id'],
            seed=episode_info['seed'],
            save_dir=save_dir,
            rate_limiter=rate_limiter
        )

        # Cache the agent for future episodes if ACE multi-epoch
        if episode_info['agent_name'] == 'a_c_e' and episode_info['max_epochs'] > 1 and success:
            cache_key = f"{episode_info['env_name']}_{episode_info['agent_name']}"
            # Get the agent from the runner's last episode
            if hasattr(runner, '_last_agent'):
                ace_agent_cache[cache_key] = runner._last_agent

        # Update progress
        with lock:
            if success:
                completed_count += 1
                status = "✓"
            else:
                failed_count += 1
                status = "✗"
                failed_episodes.append(error)

            # Calculate stats
            elapsed = time.time() - start_time
            rate = completed_count / (elapsed / 60) if elapsed > 0 else 0
            remaining = total_episodes - (completed_count + failed_count)
            eta_min = remaining / rate if rate > 0 else 0

            # Print progress
            print(f"{status} [{completed_count + failed_count}/{total_episodes}] "
                  f"{episode_info['episode_id']} "
                  f"({rate:.1f} eps/min, ETA: {eta_min:.1f}min)")

        return result

    # Run episodes in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_episode_wrapper, ep): ep for ep in episodes}

        for future in as_completed(futures):
            if shutdown_requested:
                print("Cancelling remaining episodes...")
                executor.shutdown(wait=False, cancel_futures=True)
                break

            try:
                future.result()
            except Exception as e:
                print(f"Unexpected error: {e}")
                traceback.print_exc()

    # Save failed episodes
    if failed_episodes:
        failed_path = output_dir / 'failed_episodes.json'
        with open(failed_path, 'w') as f:
            json.dump(failed_episodes, f, indent=2)
        print(f"\n⚠️  Saved {len(failed_episodes)} failed episodes to {failed_path}")

    # Print summary
    total_time = time.time() - start_time
    rate_limiter_stats = rate_limiter.get_total_stats()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE" if not shutdown_requested else "EXPERIMENT INTERRUPTED")
    print("=" * 70)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Completed: {completed_count}/{total_episodes}")
    print(f"Failed: {failed_count}/{total_episodes}")
    if completed_count > 0:
        print(f"Average rate: {completed_count / (total_time / 60):.1f} episodes/min")
    print()
    print("Rate limiter stats:")
    print(f"  Total requests: {rate_limiter_stats['total_requests']}")
    print(f"  Total input tokens: {rate_limiter_stats['total_input_tokens']:,}")
    print(f"  Total output tokens: {rate_limiter_stats['total_output_tokens']:,}")
    print(f"  Wait count: {rate_limiter_stats['wait_count']}")
    print(f"  Total wait time: {rate_limiter_stats['total_wait_time']:.1f}s")
    if rate_limiter_stats['wait_count'] > 0:
        print(f"  Avg wait time: {rate_limiter_stats['avg_wait_time']:.1f}s")
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run world-model experiment with parallel execution and rate limiting"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    parser.add_argument(
        '--preregistration',
        type=str,
        default='preregistration.yaml',
        help='Path to preregistration file (default: preregistration.yaml)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: results/parallel_run_TIMESTAMP)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=10,
        help='Number of parallel workers (default: 10)'
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Resume from previous run (path to output directory)'
    )

    args = parser.parse_args()

    run_parallel_experiment(
        config_path=args.config,
        preregistration_path=args.preregistration,
        output_dir=args.output_dir,
        workers=args.workers,
        resume_from=args.resume_from
    )


if __name__ == '__main__':
    main()
