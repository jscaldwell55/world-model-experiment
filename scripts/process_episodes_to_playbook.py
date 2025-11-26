#!/usr/bin/env python3
"""
Process raw episode data into playbook observation format.

Converts episodes from results directory into ACE playbook observations
with reliability classification based on power setting consistency.
"""

import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime


def determine_reliability(episode: dict) -> str:
    """
    Determine reliability based on power setting consistency.

    For HotPot, reliability is determined by whether a single power
    setting was used throughout the episode.

    Returns:
        'HIGH' - Single consistent power setting (reliable)
        'LOW' - Multiple power settings or inconsistent (unreliable)
    """
    # Get ground truth power setting
    ground_truth = episode.get('ground_truth', {})
    stove_power = ground_truth.get('stove_power', '').lower()

    # In HotPot, each episode has a single power setting in ground truth
    # HIGH or LOW power = reliable single-setting episode
    # Treat all as HIGH reliability since each episode has consistent power
    # (The environment generates episodes with single power setting)

    return 'HIGH' if stove_power in ['high', 'low'] else 'LOW'


def extract_beliefs_from_episode(episode: dict, reliability: str) -> Dict:
    """
    Extract world model beliefs from episode.

    For HotPot:
    - heating_rate: from ground truth
    - measurement_noise: standard value
    - base_temp: standard value
    """
    ground_truth = episode.get('ground_truth', {})
    heating_rate = ground_truth.get('heating_rate', 1.2)

    # Get power setting for context
    power = ground_truth.get('stove_power', 'unknown')

    beliefs = {
        'heating_rate_mean': {'value': heating_rate},
        'heating_rate_std': {'value': 0.2},
        'measurement_noise': {'value': 2.0},
        'base_temp': {'value': 20.0}
    }

    return beliefs, power.upper()


def convert_episode_to_observation(episode: dict) -> Dict:
    """Convert raw episode to playbook observation format"""

    # Determine reliability
    reliability = determine_reliability(episode)

    # Extract beliefs and context
    beliefs, power_setting = extract_beliefs_from_episode(episode, reliability)

    # Calculate test accuracy from test results list
    test_results = episode.get('test_results', [])
    if isinstance(test_results, list) and len(test_results) > 0:
        # Calculate average score from all test queries
        total_score = sum(result.get('score', 0) for result in test_results)
        score = total_score / len(test_results)
    else:
        score = 0.5  # Default if no test results

    # Create observation
    observation = {
        'episode_id': episode.get('episode_id', 'unknown'),
        'timestamp': episode.get('timestamp', datetime.now().isoformat()),
        'score': score,
        'beliefs': beliefs,
        'context': {
            'power_setting': power_setting
        },
        'reliability': reliability,
        'reason': f"{'Consistent' if reliability == 'HIGH' else 'Mixed'} power setting - {reliability.lower()} reliability",
        'metadata': {
            'patterns': [f"Achieved {score*100:.0f}% accuracy"] if score >= 0.7 else [],
            'failures': [f"Low accuracy: {score*100:.0f}%"] if score < 0.7 else []
        }
    }

    return observation


def process_episodes_to_playbook(
    episodes_dir: Path,
    output_path: Path,
    domain: str = 'hot_pot'
):
    """
    Process all episodes in directory to playbook format.

    Args:
        episodes_dir: Directory containing raw episode JSON files
        output_path: Path to output playbook JSON
        domain: Domain name
    """
    print(f"Processing episodes from {episodes_dir}")
    print(f"Output playbook: {output_path}")
    print()

    # Load all episodes
    episode_files = sorted(episodes_dir.glob('*.json'))
    print(f"Found {len(episode_files)} episode files")

    observations = []
    reliability_counts = {'HIGH': 0, 'LOW': 0}

    for episode_file in episode_files:
        try:
            with open(episode_file, 'r') as f:
                episode = json.load(f)

            observation = convert_episode_to_observation(episode)
            observations.append(observation)

            rel = observation['reliability']
            reliability_counts[rel] += 1

        except Exception as e:
            print(f"⚠️  Error processing {episode_file.name}: {e}")
            continue

    print(f"\nProcessed {len(observations)} episodes:")
    print(f"  HIGH reliability: {reliability_counts['HIGH']}")
    print(f"  LOW reliability: {reliability_counts['LOW']}")

    # Create playbook structure
    playbook = {
        'domain': domain,
        'created': datetime.now().isoformat(),
        'observations': observations,
        'metadata': {
            'total_observations': len(observations),
            'high_reliability_count': reliability_counts['HIGH'],
            'low_reliability_count': reliability_counts['LOW'],
            'source': 'fidelity_validation_30ep'
        }
    }

    # Save playbook
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(playbook, f, indent=2)

    print(f"\n✓ Playbook saved to {output_path}")
    print(f"  Total observations: {len(observations)}")
    if len(observations) > 0:
        print(f"  HIGH reliability: {reliability_counts['HIGH']} ({reliability_counts['HIGH']/len(observations)*100:.1f}%)")
    else:
        print("  No observations processed!")

    return playbook


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Process raw episodes to playbook format")
    parser.add_argument('--episodes-dir', type=str,
                       default='results/fidelity_data_30ep/raw',
                       help='Directory containing raw episode files')
    parser.add_argument('--output', type=str,
                       default='memory/domains/hot_pot/playbook.json',
                       help='Output playbook path')
    parser.add_argument('--domain', type=str, default='hot_pot',
                       help='Domain name')

    args = parser.parse_args()

    episodes_dir = Path(args.episodes_dir)
    output_path = Path(args.output)

    if not episodes_dir.exists():
        print(f"Error: Episodes directory not found: {episodes_dir}")
        exit(1)

    process_episodes_to_playbook(episodes_dir, output_path, args.domain)
