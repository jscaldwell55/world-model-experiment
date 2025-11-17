#!/usr/bin/env python3
"""
Analyze domain-specific memory to understand what the agent has learned.

Usage:
    python memory/analyze_memory.py
    python memory/analyze_memory.py --domain hot_pot
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List


def analyze_domain_memory(domain: str, base_path: str = "memory/domains") -> Dict:
    """
    Analyze consolidated beliefs for a domain.

    Args:
        domain: Domain name (hot_pot, chem_tile, switch_light)
        base_path: Base path to memory directories

    Returns:
        Dictionary with analysis results
    """
    consolidated_path = Path(base_path) / domain / 'consolidated' / 'beliefs.json'
    episodes_path = Path(base_path) / domain / 'episodes'

    analysis = {
        'domain': domain,
        'has_memory': False,
        'num_episodes': 0,
        'avg_score': 0.0,
        'confidence': 0.0,
        'beliefs': {},
        'episode_scores': []
    }

    # Check for consolidated memory
    if consolidated_path.exists():
        with open(consolidated_path, 'r') as f:
            data = json.load(f)

        analysis['has_memory'] = True
        analysis['num_episodes'] = data.get('num_episodes', 0)
        analysis['avg_score'] = data.get('avg_score', 0.0)
        analysis['confidence'] = data.get('confidence', 0.0)
        analysis['beliefs'] = data.get('beliefs', {})

        print(f"\n{'='*70}")
        print(f"{domain.upper().replace('_', ' ')} DOMAIN MEMORY")
        print(f"{'='*70}")
        print(f"Episodes completed: {analysis['num_episodes']}")
        print(f"Average score:      {analysis['avg_score']:.1f}%")
        print(f"Confidence level:   {analysis['confidence']:.2f}")
        print(f"Prior strength:     {min(0.3, analysis['confidence']):.2f} (capped at 0.3)")
        print(f"\nKey Beliefs:")
        print(json.dumps(analysis['beliefs'], indent=2))

    else:
        print(f"\n{'='*70}")
        print(f"{domain.upper().replace('_', ' ')} DOMAIN MEMORY")
        print(f"{'='*70}")
        print(f"Status: No memory found (first episode not yet run)")

    # Analyze individual episodes if they exist
    if episodes_path.exists():
        episode_files = sorted(episodes_path.glob('*.json'))
        if episode_files:
            print(f"\nEpisode History ({len(episode_files)} episodes):")
            print(f"{'Episode':<40} {'Score':>8} {'Timestamp':<20}")
            print(f"{'-'*70}")

            for ep_file in episode_files:
                with open(ep_file, 'r') as f:
                    ep_data = json.load(f)
                    ep_id = ep_data.get('episode_id', ep_file.stem)
                    score = ep_data.get('score', 0.0)
                    timestamp = ep_data.get('timestamp', '')[:19]  # Trim to date+time
                    analysis['episode_scores'].append(score)
                    print(f"{ep_id:<40} {score:>7.1f}% {timestamp:<20}")

            # Show score trend
            if len(analysis['episode_scores']) > 1:
                first_5_avg = sum(analysis['episode_scores'][:5]) / min(5, len(analysis['episode_scores']))
                last_5_avg = sum(analysis['episode_scores'][-5:]) / min(5, len(analysis['episode_scores'][-5:]))
                improvement = last_5_avg - first_5_avg

                print(f"\nScore Trend:")
                print(f"  First 5 episodes avg: {first_5_avg:.1f}%")
                print(f"  Last 5 episodes avg:  {last_5_avg:.1f}%")
                print(f"  Improvement:          {improvement:+.1f}%")

    print(f"{'='*70}\n")

    return analysis


def compare_domains(base_path: str = "memory/domains") -> None:
    """
    Compare memory across all domains.

    Args:
        base_path: Base path to memory directories
    """
    domains = ['chem_tile', 'hot_pot', 'switch_light']
    all_analysis = {}

    print(f"\n{'='*70}")
    print(f"CROSS-DOMAIN MEMORY COMPARISON")
    print(f"{'='*70}\n")

    for domain in domains:
        all_analysis[domain] = analyze_domain_memory(domain, base_path)

    # Summary comparison
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Domain':<15} {'Episodes':>10} {'Avg Score':>12} {'Confidence':>12}")
    print(f"{'-'*70}")

    for domain in domains:
        analysis = all_analysis[domain]
        if analysis['has_memory']:
            print(f"{domain:<15} {analysis['num_episodes']:>10} "
                  f"{analysis['avg_score']:>11.1f}% {analysis['confidence']:>12.2f}")
        else:
            print(f"{domain:<15} {'--':>10} {'--':>12} {'--':>12}")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze domain-specific memory")
    parser.add_argument(
        '--domain',
        choices=['hot_pot', 'chem_tile', 'switch_light', 'all'],
        default='all',
        help='Domain to analyze (default: all)'
    )
    parser.add_argument(
        '--base-path',
        default='memory/domains',
        help='Base path to memory directories'
    )
    args = parser.parse_args()

    if args.domain == 'all':
        compare_domains(args.base_path)
    else:
        analyze_domain_memory(args.domain, args.base_path)


if __name__ == "__main__":
    main()
