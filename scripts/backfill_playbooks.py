#!/usr/bin/env python3
"""
Backfill playbooks from historical experiment results.

This script processes episodes from results directories that were run with
agents OTHER than SimpleWorldModel (which auto-saves to playbooks).

Supports:
- hot_pot (HotPotLab)
- chem_tile (ChemTile)
- switch_light (SwitchLight)

Each domain has specific reliability classification logic based on
methodology analysis.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict


# Environment name mapping
ENV_TO_DOMAIN = {
    'HotPotLab': 'hot_pot',
    'ChemTile': 'chem_tile',
    'SwitchLight': 'switch_light'
}


def analyze_methodology_hot_pot(episode: dict) -> Tuple[str, str]:
    """
    Analyze HotPot methodology for reliability classification.

    HIGH reliability if:
    - Single power setting used
    - Multiple temperature measurements taken

    Returns:
        (reliability, reason)
    """
    ground_truth = episode.get('ground_truth', {})
    stove_power = ground_truth.get('stove_power', '').lower()

    # Count temperature measurements in steps
    steps = episode.get('steps', [])
    temp_measurements = 0
    actions = []

    for step in steps:
        action = step.get('action', '')
        if action:
            actions.append(action)
            if 'measure_temp' in action or 'read_temp' in action:
                temp_measurements += 1

    # Reliability logic
    if stove_power in ['high', 'low', 'off']:
        if temp_measurements >= 2:
            return 'HIGH', f"Consistent {stove_power} power with {temp_measurements} measurements"
        else:
            return 'MEDIUM', f"Single {stove_power} power but limited measurements ({temp_measurements})"
    else:
        return 'LOW', "Unknown or mixed power settings"


def analyze_methodology_chem_tile(episode: dict) -> Tuple[str, str]:
    """
    Analyze ChemTile methodology for reliability classification.

    HIGH reliability if:
    - Multiple reactions observed
    - Systematic exploration of compound combinations

    Returns:
        (reliability, reason)
    """
    steps = episode.get('steps', [])

    # Count reaction-related actions
    mix_count = 0
    inspect_count = 0
    unique_compounds = set()

    for step in steps:
        action = step.get('action', '')
        if action:
            if 'mix' in action.lower():
                mix_count += 1
            if 'inspect' in action.lower():
                inspect_count += 1
            # Try to extract compound names
            if '(' in action:
                try:
                    args = action.split('(')[1].rstrip(')')
                    for arg in args.split(','):
                        unique_compounds.add(arg.strip().strip('"\''))
                except:
                    pass

    # Reliability logic
    total_actions = mix_count + inspect_count
    if mix_count >= 3 and len(unique_compounds) >= 2:
        return 'HIGH', f"Good exploration: {mix_count} mixes, {len(unique_compounds)} compounds"
    elif total_actions >= 2:
        return 'MEDIUM', f"Limited exploration: {mix_count} mixes"
    else:
        return 'LOW', "Few reactions - minimal exploration"


def analyze_methodology_switch_light(episode: dict) -> Tuple[str, str]:
    """
    Analyze SwitchLight methodology for reliability classification.

    HIGH reliability if:
    - Multiple switch flips observed
    - Systematic testing pattern

    Returns:
        (reliability, reason)
    """
    steps = episode.get('steps', [])

    flip_count = 0
    observe_count = 0
    inspect_count = 0

    for step in steps:
        action = step.get('action', '')
        if action:
            if 'flip' in action.lower():
                flip_count += 1
            if 'observe' in action.lower():
                observe_count += 1
            if 'inspect' in action.lower():
                inspect_count += 1

    # Reliability logic
    if flip_count >= 4:
        return 'HIGH', f"Systematic exploration ({flip_count} flips) - good coverage"
    elif flip_count >= 2:
        return 'MEDIUM', f"Moderate exploration ({flip_count} flips)"
    else:
        return 'LOW', f"Minimal testing ({flip_count} flips)"


def extract_beliefs_hot_pot(episode: dict) -> Dict:
    """Extract world model beliefs for HotPot domain."""
    ground_truth = episode.get('ground_truth', {})
    heating_rate = ground_truth.get('heating_rate', 1.2)
    power = ground_truth.get('stove_power', 'unknown').upper()

    beliefs = {
        'heating_rate_mean': {'value': heating_rate},
        'heating_rate_std': {'value': 0.2},
        'measurement_noise': {'value': 2.0},
        'base_temp': {'value': 20.0}
    }

    context = {'power_setting': power}
    return beliefs, context


def extract_beliefs_chem_tile(episode: dict) -> Dict:
    """
    Extract world model beliefs for ChemTile domain.

    Uses ground_truth to extract actual temperature and infers reaction
    success rates from episode outcomes (explosions, products created).
    """
    ground_truth = episode.get('ground_truth', {})

    # Extract actual temperature from ground_truth
    temperature = ground_truth.get('temperature', 'medium')

    # Infer reaction outcomes from episode results
    explosion_count = ground_truth.get('explosion_count', 0)
    available_compounds = ground_truth.get('available_compounds', [])
    last_reaction = ground_truth.get('last_reaction')

    # Count reaction attempts from steps
    steps = episode.get('steps', [])
    mix_count = sum(1 for s in steps if s.get('action') and 'mix' in s.get('action', '').lower())

    # Base reaction probabilities - vary based on observed outcomes
    # If explosions occurred, increase explosion probability estimate
    # If products were created, reaction was successful

    # Calculate success indicators
    has_c = 'C' in available_compounds or 'D' in available_compounds
    has_d = 'D' in available_compounds
    explosion_rate = explosion_count / max(mix_count, 1) if mix_count > 0 else 0.1

    # Adjust probabilities based on temperature and outcomes
    # Higher temps = more explosions, lower success
    temp_modifier = {'low': 0.05, 'medium': 0.0, 'high': -0.1}.get(temperature, 0.0)

    # A+B reaction probabilities
    ab_explode = min(0.3, max(0.05, 0.1 + explosion_rate * 0.2))
    ab_success = 0.8 + temp_modifier if has_c or has_d else 0.7 + temp_modifier
    ab_nothing = 1.0 - ab_success - ab_explode

    # C+B reaction probabilities
    cb_explode = min(0.35, max(0.1, 0.2 + explosion_rate * 0.2))
    cb_success = 0.7 + temp_modifier if has_d else 0.6 + temp_modifier
    cb_nothing = 1.0 - cb_success - cb_explode

    # Ensure probabilities are valid
    ab_success = max(0.5, min(0.9, ab_success))
    cb_success = max(0.4, min(0.85, cb_success))
    ab_nothing = max(0.05, 1.0 - ab_success - ab_explode)
    cb_nothing = max(0.05, 1.0 - cb_success - cb_explode)

    # Confidence based on exploration depth
    confidence = min(0.8, 0.3 + mix_count * 0.1)

    beliefs = {
        'reaction_probs': {
            'value': {
                'A+B': {
                    'C': round(ab_success, 2),
                    'explode': round(ab_explode, 2),
                    'nothing': round(ab_nothing, 2)
                },
                'C+B': {
                    'D': round(cb_success, 2),
                    'explode': round(cb_explode, 2),
                    'nothing': round(cb_nothing, 2)
                }
            },
            'confidence': confidence,
            'observation_count': mix_count,
            'episode_count': 1
        },
        'temperature': {
            'value': temperature,
            'confidence': 0.8,  # Temperature is directly observed
            'observation_count': 1,
            'episode_count': 1
        }
    }

    context = {
        'explosion_count': explosion_count,
        'products_created': available_compounds,
        'last_reaction': last_reaction
    }
    return beliefs, context


def extract_beliefs_switch_light(episode: dict) -> Dict:
    """
    Extract world model beliefs for SwitchLight domain.

    Uses ground_truth.wire_layout to determine the actual wiring configuration.
    Infers confidence from episode exploration depth (flip count).
    """
    ground_truth = episode.get('ground_truth', {})

    # Extract actual wire layout from ground_truth
    wire_layout = ground_truth.get('wire_layout', 'layout_A')
    faulty_relay = ground_truth.get('faulty_relay', False)

    # Count switch flips to determine confidence
    steps = episode.get('steps', [])
    flip_count = sum(1 for s in steps if s.get('action') and 'flip' in s.get('action', '').lower())
    observe_count = sum(1 for s in steps if s.get('action') and 'observe' in s.get('action', '').lower())

    # Calculate confidence based on exploration
    # More flips = higher confidence in layout determination
    base_confidence = min(0.95, 0.5 + flip_count * 0.08)

    # Set wiring probabilities based on ground truth
    # The actual layout gets high probability, other layout gets low
    if wire_layout == 'layout_A':
        layout_a_prob = base_confidence
        layout_b_prob = 1.0 - base_confidence
    elif wire_layout == 'layout_B':
        layout_a_prob = 1.0 - base_confidence
        layout_b_prob = base_confidence
    else:
        # Unknown layout - stay uncertain
        layout_a_prob = 0.5
        layout_b_prob = 0.5

    # Failure probability estimation
    # If faulty relay, estimate higher failure rate
    if faulty_relay:
        failure_prob = 0.08 + (0.02 * flip_count)  # Varies 0.08-0.18
    else:
        # Normal failure rate with small variation based on observations
        failure_prob = 0.01 + (0.005 * (flip_count % 3))  # Varies 0.01-0.02

    failure_prob = round(min(0.2, max(0.005, failure_prob)), 3)

    beliefs = {
        'wiring_probs': {
            'value': {
                'layout_A': round(layout_a_prob, 2),
                'layout_B': round(layout_b_prob, 2)
            },
            'confidence': round(base_confidence, 2),
            'observation_count': flip_count + observe_count,
            'episode_count': 1
        },
        'failure_prob': {
            'value': failure_prob,
            'confidence': round(min(0.8, 0.3 + flip_count * 0.1), 2),
            'observation_count': flip_count,
            'episode_count': 1
        }
    }

    context = {
        'true_layout': wire_layout,
        'faulty_relay': faulty_relay,
        'flip_count': flip_count
    }
    return beliefs, context


def convert_episode_to_observation(episode: dict, domain: str) -> Optional[Dict]:
    """
    Convert a raw episode to playbook observation format.

    Args:
        episode: Raw episode dictionary
        domain: Domain name (hot_pot, chem_tile, switch_light)

    Returns:
        Observation dictionary or None if invalid
    """
    # Determine reliability based on domain
    if domain == 'hot_pot':
        reliability, reason = analyze_methodology_hot_pot(episode)
        beliefs, context = extract_beliefs_hot_pot(episode)
    elif domain == 'chem_tile':
        reliability, reason = analyze_methodology_chem_tile(episode)
        beliefs, context = extract_beliefs_chem_tile(episode)
    elif domain == 'switch_light':
        reliability, reason = analyze_methodology_switch_light(episode)
        beliefs, context = extract_beliefs_switch_light(episode)
    else:
        return None

    # Calculate score from test results
    test_results = episode.get('test_results', [])
    if isinstance(test_results, list) and len(test_results) > 0:
        total_score = sum(r.get('score', 0) for r in test_results)
        score = total_score / len(test_results)
    else:
        score = 0.5

    # Build observation
    observation = {
        'episode_id': episode.get('episode_id', f"backfill_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        'timestamp': episode.get('timestamp', datetime.now().timestamp()),
        'score': score,
        'beliefs': beliefs,
        'context': context,
        'reliability': reliability,
        'reason': reason,
        'metadata': {
            'patterns': [f"Achieved {score*100:.0f}% accuracy"] if score >= 0.7 else [],
            'failures': [f"Low accuracy: {score*100:.0f}%"] if score < 0.7 else [],
            'source_agent': episode.get('agent_type', 'unknown'),
            'backfilled': True,
            'backfill_timestamp': datetime.now().isoformat()
        }
    }

    return observation


def load_existing_playbook(playbook_path: Path) -> Dict:
    """Load existing playbook or return empty structure."""
    if playbook_path.exists():
        with open(playbook_path) as f:
            return json.load(f)
    return {
        'observations': [],
        'strategies_and_rules': [],
        'troubleshooting': [],
        'context_patterns': []
    }


def backfill_from_results(
    results_dirs: List[str],
    output_base: str = "memory/domains",
    merge: bool = True,
    agent_filter: Optional[List[str]] = None,
    dry_run: bool = False
):
    """
    Backfill playbooks from multiple results directories.

    Args:
        results_dirs: List of results directories to process
        output_base: Base path for output playbooks
        merge: If True, merge with existing playbooks; if False, replace
        agent_filter: Only process episodes from these agent types (None = all)
        dry_run: If True, don't write files, just report what would be done
    """
    output_base = Path(output_base)

    # Collect episodes by domain
    episodes_by_domain = defaultdict(list)

    for results_dir in results_dirs:
        raw_dir = Path(results_dir) / 'raw'
        if not raw_dir.exists():
            print(f"⚠️  Skipping {results_dir}: no 'raw' subdirectory")
            continue

        print(f"\nProcessing {results_dir}...")

        for ep_file in sorted(raw_dir.glob('*.json')):
            try:
                with open(ep_file) as f:
                    episode = json.load(f)

                # Filter by agent type if specified
                agent_type = episode.get('agent_type', 'unknown')
                if agent_filter and agent_type not in agent_filter:
                    continue

                # Map environment to domain
                env_name = episode.get('environment', '')
                domain = ENV_TO_DOMAIN.get(env_name)

                if not domain:
                    continue

                episodes_by_domain[domain].append(episode)

            except Exception as e:
                print(f"  ⚠️  Error reading {ep_file.name}: {e}")

    # Process each domain
    for domain, episodes in episodes_by_domain.items():
        print(f"\n{'='*60}")
        print(f"Domain: {domain}")
        print(f"{'='*60}")
        print(f"Episodes to process: {len(episodes)}")

        # Load existing playbook if merging
        playbook_path = output_base / domain / 'playbook.json'
        if merge:
            playbook = load_existing_playbook(playbook_path)
            existing_ids = {o.get('episode_id') for o in playbook.get('observations', [])}
        else:
            playbook = {
                'observations': [],
                'strategies_and_rules': [],
                'troubleshooting': [],
                'context_patterns': []
            }
            existing_ids = set()

        # Convert episodes to observations
        new_observations = []
        reliability_counts = defaultdict(int)
        skipped_duplicates = 0

        for episode in episodes:
            ep_id = episode.get('episode_id', '')

            # Skip duplicates
            if ep_id in existing_ids:
                skipped_duplicates += 1
                continue

            obs = convert_episode_to_observation(episode, domain)
            if obs:
                new_observations.append(obs)
                reliability_counts[obs['reliability']] += 1
                existing_ids.add(ep_id)

        print(f"\nResults:")
        print(f"  New observations: {len(new_observations)}")
        print(f"  Skipped duplicates: {skipped_duplicates}")
        print(f"  Reliability breakdown:")
        for rel in ['HIGH', 'MEDIUM', 'LOW']:
            if reliability_counts[rel] > 0:
                print(f"    {rel}: {reliability_counts[rel]}")

        # Merge and save
        if not dry_run:
            playbook['observations'].extend(new_observations)

            # Ensure directory exists
            playbook_path.parent.mkdir(parents=True, exist_ok=True)

            with open(playbook_path, 'w') as f:
                json.dump(playbook, f, indent=2)

            print(f"\n✅ Saved: {playbook_path}")
            print(f"   Total observations: {len(playbook['observations'])}")
        else:
            print(f"\n[DRY RUN] Would save {len(new_observations)} new observations to {playbook_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill playbooks from historical experiment results"
    )
    parser.add_argument(
        '--results-dirs',
        nargs='+',
        default=['results/full_study_v2', 'results/full_study_final'],
        help='Results directories to process'
    )
    parser.add_argument(
        '--output-base',
        default='memory/domains',
        help='Base path for output playbooks'
    )
    parser.add_argument(
        '--no-merge',
        action='store_true',
        help='Replace existing playbooks instead of merging'
    )
    parser.add_argument(
        '--agent-filter',
        nargs='+',
        help='Only process episodes from these agent types'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Report what would be done without writing files'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PLAYBOOK BACKFILL")
    print("=" * 70)
    print(f"Results dirs: {args.results_dirs}")
    print(f"Output base: {args.output_base}")
    print(f"Merge mode: {'merge' if not args.no_merge else 'replace'}")
    print(f"Agent filter: {args.agent_filter or 'all'}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 70)

    backfill_from_results(
        results_dirs=args.results_dirs,
        output_base=args.output_base,
        merge=not args.no_merge,
        agent_filter=args.agent_filter,
        dry_run=args.dry_run
    )

    print("\n✅ Backfill complete!")


if __name__ == '__main__':
    main()
