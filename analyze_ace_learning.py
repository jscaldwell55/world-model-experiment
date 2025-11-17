#!/usr/bin/env python3
"""
Analyze ACE Learning Progression

Visualizes how the ACE memory system learns across episodes:
- Reliability distribution
- Score progression
- Belief evolution
- Methodology quality trends
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def load_episode_results(results_dir: Path) -> Dict[str, List[Dict]]:
    """
    Load episode results grouped by domain.

    Args:
        results_dir: Directory containing episode JSON files

    Returns:
        Dict mapping domain name to list of episode results
    """
    raw_dir = results_dir / 'raw'
    if not raw_dir.exists():
        print(f"Error: {raw_dir} does not exist")
        return {}

    episodes_by_domain = defaultdict(list)

    for episode_file in sorted(raw_dir.glob('*.json')):
        try:
            with open(episode_file, 'r') as f:
                episode_data = json.load(f)

            environment = episode_data.get('environment', 'unknown')

            # Map environment names to domains
            domain_map = {
                'HotPotLab': 'hot_pot',
                'ChemTile': 'chem_tile',
                'SwitchLight': 'switch_light'
            }
            domain = domain_map.get(environment, environment.lower())

            episodes_by_domain[domain].append(episode_data)
        except Exception as e:
            print(f"Warning: Failed to load {episode_file}: {e}")

    return episodes_by_domain


def load_playbook(domain: str) -> Dict:
    """
    Load ACE playbook for a domain.

    Args:
        domain: Domain name (hot_pot, chem_tile, switch_light)

    Returns:
        Playbook dictionary
    """
    playbook_path = Path(f'memory/domains/{domain}/playbook.json')
    if playbook_path.exists():
        with open(playbook_path, 'r') as f:
            return json.load(f)
    return {}


def analyze_domain(domain: str, episodes: List[Dict]):
    """
    Analyze ACE learning progression for a single domain.

    Args:
        domain: Domain name
        episodes: List of episode data dictionaries
    """
    print(f"\n{'='*80}")
    print(f"DOMAIN: {domain.upper().replace('_', ' ')}")
    print(f"{'='*80}")

    # Load playbook
    playbook = load_playbook(domain)

    if not playbook:
        print(f"⚠️  No playbook found for {domain}")
        return

    print(f"\n📊 EPISODE PROGRESSION ({len(episodes)} episodes)")
    print(f"{'─'*80}")

    # Analyze each episode
    for i, episode in enumerate(episodes, 1):
        episode_id = episode.get('episode_id', f'episode_{i}')

        # Extract test scores
        test_results = episode.get('test_results', [])
        if test_results:
            total_score = sum(r['score'] for r in test_results) / len(test_results)
            correct_count = sum(1 for r in test_results if r.get('correct', False))
            accuracy = (correct_count / len(test_results)) * 100 if test_results else 0
        else:
            total_score = 0
            accuracy = 0

        print(f"\nEpisode {i}: {episode_id}")
        print(f"  Score: {total_score*100:.1f}% | Accuracy: {accuracy:.0f}% ({correct_count}/{len(test_results)})")

        # Show actions taken
        steps = episode.get('steps', [])
        actions = [s.get('action') for s in steps if s.get('action')]
        print(f"  Actions: {', '.join(actions[:5])}" + ("..." if len(actions) > 5 else ""))

    # Analyze playbook observations
    observations = playbook.get('observations', [])

    if observations:
        print(f"\n🔍 PLAYBOOK ANALYSIS ({len(observations)} observations)")
        print(f"{'─'*80}")

        # Reliability distribution
        reliability_counts = defaultdict(int)
        for obs in observations:
            reliability = obs.get('reliability', 'UNKNOWN')
            reliability_counts[reliability] += 1

        print(f"\nReliability Distribution:")
        for level in ['HIGH', 'MEDIUM', 'LOW']:
            count = reliability_counts.get(level, 0)
            bar = '█' * count + '░' * (len(observations) - count)
            print(f"  {level:8s}: {bar} ({count}/{len(observations)})")

        # Methodology reasons
        print(f"\nMethodology Analysis:")
        for obs in observations:
            ep_id = obs.get('episode_id', 'unknown')
            reliability = obs.get('reliability', 'UNKNOWN')
            reason = obs.get('reason', 'No reason provided')
            score = obs.get('score', 0) * 100

            # Reliability symbol
            symbol = {
                'HIGH': '✓',
                'MEDIUM': '○',
                'LOW': '⚠️'
            }.get(reliability, '?')

            print(f"  {symbol} {ep_id}: {reason} (score: {score:.0f}%)")

        # Domain-specific analysis
        if domain == 'hot_pot':
            print(f"\n📈 BELIEF EVOLUTION (HotPot)")
            print(f"{'─'*80}")

            for obs in observations:
                ep_id = obs.get('episode_id', 'unknown')
                beliefs = obs.get('beliefs', {})
                context = obs.get('context', {})

                # Extract heating rate
                heating_rate = extract_value(beliefs.get('heating_rate_mean', {}))
                power = context.get('power_setting', 'UNKNOWN')
                reliability = obs.get('reliability', 'UNKNOWN')

                if isinstance(heating_rate, (int, float)):
                    print(f"  {ep_id}: {heating_rate:.2f}°C/s [power: {power:6s}] ({reliability})")

        # Strategies learned
        strategies = playbook.get('strategies_and_rules', [])
        if strategies:
            print(f"\n📋 STRATEGIES LEARNED ({len(strategies)})")
            print(f"{'─'*80}")
            for strategy in strategies:
                content = strategy.get('content', '')
                print(f"  • {content}")

        # Troubleshooting patterns
        troubleshooting = playbook.get('troubleshooting', [])
        if troubleshooting:
            print(f"\n⚠️  TROUBLESHOOTING PATTERNS ({len(troubleshooting)})")
            print(f"{'─'*80}")
            for issue in troubleshooting:
                content = issue.get('content', '')
                print(f"  • {content}")
    else:
        print(f"\n⚠️  No observations in playbook yet")


def extract_value(belief_data):
    """Extract value from structured belief format or return raw value."""
    if isinstance(belief_data, dict) and 'value' in belief_data:
        return belief_data['value']
    return belief_data


def print_summary(episodes_by_domain: Dict[str, List[Dict]]):
    """
    Print overall summary statistics.

    Args:
        episodes_by_domain: Episodes grouped by domain
    """
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*80}")

    total_episodes = sum(len(eps) for eps in episodes_by_domain.values())
    print(f"\nTotal episodes: {total_episodes}")
    print(f"Domains: {', '.join(episodes_by_domain.keys())}")

    # Average scores by domain
    print(f"\nAverage Scores by Domain:")
    for domain, episodes in episodes_by_domain.items():
        scores = []
        for episode in episodes:
            test_results = episode.get('test_results', [])
            if test_results:
                score = sum(r['score'] for r in test_results) / len(test_results)
                scores.append(score * 100)

        if scores:
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            print(f"  {domain:12s}: {avg_score:.1f}% (range: {min_score:.1f}%-{max_score:.1f}%)")

    # ACE playbook statistics
    print(f"\nACE Playbook Statistics:")
    for domain in episodes_by_domain.keys():
        playbook = load_playbook(domain)
        observations = playbook.get('observations', [])

        if observations:
            high_rel = sum(1 for o in observations if o.get('reliability') == 'HIGH')
            low_rel = sum(1 for o in observations if o.get('reliability') == 'LOW')

            print(f"  {domain:12s}: {len(observations)} obs "
                  f"({high_rel} HIGH, {low_rel} LOW)")
        else:
            print(f"  {domain:12s}: No observations yet")


def check_belief_trap_prevention(episodes_by_domain: Dict[str, List[Dict]]):
    """
    Check if ACE prevented belief traps (specific to HotPot).

    Args:
        episodes_by_domain: Episodes grouped by domain
    """
    if 'hot_pot' not in episodes_by_domain:
        return

    print(f"\n{'='*80}")
    print(f"BELIEF TRAP PREVENTION CHECK (HotPot)")
    print(f"{'='*80}")

    playbook = load_playbook('hot_pot')
    observations = playbook.get('observations', [])

    if not observations:
        print("\n⚠️  No observations to analyze")
        return

    # Check for mixed power observations with LOW reliability
    low_rel_mixed = []
    high_rel_consistent = []

    for obs in observations:
        reliability = obs.get('reliability', 'UNKNOWN')
        context = obs.get('context', {})
        power = context.get('power_setting', 'UNKNOWN')

        if reliability == 'LOW' and power == 'MIXED':
            low_rel_mixed.append(obs)
        elif reliability == 'HIGH' and power != 'MIXED':
            high_rel_consistent.append(obs)

    print(f"\nMixed Power Observations (LOW reliability): {len(low_rel_mixed)}")
    for obs in low_rel_mixed:
        ep_id = obs.get('episode_id', 'unknown')
        reason = obs.get('reason', '')
        print(f"  ⚠️  {ep_id}: {reason}")

    print(f"\nConsistent Power Observations (HIGH reliability): {len(high_rel_consistent)}")
    for obs in high_rel_consistent:
        ep_id = obs.get('episode_id', 'unknown')
        reason = obs.get('reason', '')
        print(f"  ✓ {ep_id}: {reason}")

    # Check if any HIGH reliability observations exist
    if high_rel_consistent:
        print(f"\n✅ BELIEF TRAP PREVENTION: SUCCESS")
        print(f"   ACE stored {len(high_rel_consistent)} HIGH reliability observation(s)")
        print(f"   These would have been REJECTED as outliers in old system!")
    elif low_rel_mixed:
        print(f"\n⚠️  WARNING: Only LOW reliability observations found")
        print(f"   Need consistent methodology episodes for HIGH reliability data")
    else:
        print(f"\n⚠️  Insufficient data to assess belief trap prevention")


def main():
    """Main analysis function."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze ACE learning progression')
    parser.add_argument('--results-dir', type=str, default='results/ace_validation_9ep',
                       help='Directory containing episode results')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' does not exist")
        print(f"\nRun the experiment first:")
        print(f"  python scripts/run_experiment_parallel.py \\")
        print(f"    --config config_ace_validation_9ep.yaml \\")
        print(f"    --output-dir {results_dir} \\")
        print(f"    --workers 1")
        return 1

    print(f"\n{'='*80}")
    print(f"ACE LEARNING PROGRESSION ANALYSIS")
    print(f"{'='*80}")
    print(f"Results directory: {results_dir}")

    # Load episodes
    episodes_by_domain = load_episode_results(results_dir)

    if not episodes_by_domain:
        print(f"\nError: No episodes found in {results_dir}/raw/")
        return 1

    # Analyze each domain
    for domain, episodes in sorted(episodes_by_domain.items()):
        analyze_domain(domain, episodes)

    # Print summary
    print_summary(episodes_by_domain)

    # Check belief trap prevention
    check_belief_trap_prevention(episodes_by_domain)

    print(f"\n{'='*80}")
    print(f"✓ Analysis complete!")
    print(f"{'='*80}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
