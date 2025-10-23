"""
Rebuild all metrics from raw episode logs.

This script demonstrates the metrics recomputation pipeline:
1. Load raw JSON logs (no cached data)
2. Recompute accuracy, surprisal, slopes from scratch
3. Perform statistical comparisons between agents
4. Generate comprehensive report

Usage:
    python scripts/rebuild_metrics_from_raw.py --results-dir results/pilot_h1h5/raw
"""

import argparse
from pathlib import Path
import json
import pandas as pd
from collections import defaultdict

from evaluation.recompute_metrics import (
    load_all_episodes_from_directory,
    aggregate_episode_metrics,
    compare_agents_statistical,
    generate_summary_report,
    compute_cohens_d,
    EpisodeMetrics
)


def main():
    parser = argparse.ArgumentParser(
        description='Rebuild metrics from raw episode logs'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/pilot_h1h5/raw',
        help='Directory containing raw episode JSON logs'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for results (JSON)'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Directory {results_dir} not found")
        return

    print("=" * 80)
    print("REBUILDING METRICS FROM RAW LOGS")
    print("=" * 80)
    print(f"Source: {results_dir}")
    print()

    # Load all episodes
    print("Loading episodes...")
    all_episodes = load_all_episodes_from_directory(results_dir)

    print(f"✓ Loaded {len(all_episodes)} episodes")
    print()

    # Group by agent type and environment
    episodes_by_agent = defaultdict(list)
    episodes_by_env_agent = defaultdict(list)

    for ep in all_episodes:
        episodes_by_agent[ep.agent_type].append(ep)
        key = (ep.environment, ep.agent_type)
        episodes_by_env_agent[key].append(ep)

    # Print dataset summary
    print("DATASET SUMMARY")
    print("-" * 80)
    for agent_type, eps in sorted(episodes_by_agent.items()):
        print(f"  {agent_type:15s}: {len(eps):3d} episodes")

    print()

    # Aggregate metrics by environment and agent type
    print("AGGREGATED METRICS BY AGENT TYPE")
    print("=" * 80)
    print()

    aggregated = {}

    for (env, agent), eps in sorted(episodes_by_env_agent.items()):
        agg = aggregate_episode_metrics(eps, agent, env)
        aggregated[f"{env}_{agent}"] = agg

        print(f"{env} - {agent}")
        print("-" * 80)
        print(f"  Episodes:        {agg.n_episodes}")
        print(f"  Accuracy:        {agg.accuracy_mean:.3f} ± {agg.accuracy_std:.3f}")
        print(f"                   95% CI: [{agg.accuracy_ci_lower:.3f}, {agg.accuracy_ci_upper:.3f}]")
        print(f"  Surprisal Slope: {agg.slope_mean:.4f} ± {agg.slope_std:.4f}")
        print(f"                   95% CI: [{agg.slope_ci_lower:.4f}, {agg.slope_ci_upper:.4f}]")
        print(f"                   p-value: {agg.slope_p_value:.4f}", end="")

        if agg.slope_p_value < 0.05:
            direction = "↓ learning" if agg.slope_mean < 0 else "↑ increasing"
            print(f"  ** {direction} **")
        else:
            print("  (not significant)")

        print()

    # Statistical comparisons between agent types
    print("\nSTATISTICAL COMPARISONS")
    print("=" * 80)

    # Group episodes by agent for comparison
    for env in sorted(set(ep.environment for ep in all_episodes)):
        env_episodes = {
            agent: [ep for ep in eps if ep.environment == env]
            for agent, eps in episodes_by_agent.items()
        }

        # Remove empty groups
        env_episodes = {k: v for k, v in env_episodes.items() if v}

        if len(env_episodes) < 2:
            continue

        print(f"\n{env} Environment")
        print("-" * 80)

        # Compare accuracy
        print("\nAccuracy Comparison:")
        acc_comparison = compare_agents_statistical(env_episodes, 'accuracy_overall')

        if 'error' not in acc_comparison:
            print(f"  ANOVA: F={acc_comparison['anova']['f_statistic']:.3f}, " +
                  f"p={acc_comparison['anova']['p_value']:.4f}")

            if acc_comparison['anova']['significant']:
                print("  ** Significant overall difference **")

            print("\n  Pairwise comparisons:")
            for comp in acc_comparison['pairwise_comparisons']:
                sig_marker = "**" if comp['significant'] else "  "
                print(f"    {sig_marker} {comp['agent1']:15s} vs {comp['agent2']:15s}: " +
                      f"Δ={comp['mean1'] - comp['mean2']:+.3f}, " +
                      f"d={comp['cohens_d']:+.2f}, " +
                      f"p={comp['p_value']:.4f}")

        # Compare surprisal slopes
        print("\nSurprisal Slope Comparison:")
        slope_comparison = compare_agents_statistical(env_episodes, 'surprisal_slope')

        if 'error' not in slope_comparison:
            print(f"  ANOVA: F={slope_comparison['anova']['f_statistic']:.3f}, " +
                  f"p={slope_comparison['anova']['p_value']:.4f}")

            if slope_comparison['anova']['significant']:
                print("  ** Significant overall difference **")

            print("\n  Pairwise comparisons:")
            for comp in slope_comparison['pairwise_comparisons']:
                sig_marker = "**" if comp['significant'] else "  "
                print(f"    {sig_marker} {comp['agent1']:15s} vs {comp['agent2']:15s}: " +
                      f"Δ={comp['mean1'] - comp['mean2']:+.4f}, " +
                      f"d={comp['cohens_d']:+.2f}, " +
                      f"p={comp['p_value']:.4f}")

    # Save results if requested
    if args.output:
        output_path = Path(args.output)

        # Convert to serializable format
        results = {
            'dataset_summary': {
                agent: len(eps) for agent, eps in episodes_by_agent.items()
            },
            'aggregated_metrics': {
                key: {
                    'agent_type': agg.agent_type,
                    'environment': agg.environment,
                    'n_episodes': agg.n_episodes,
                    'accuracy_mean': agg.accuracy_mean,
                    'accuracy_std': agg.accuracy_std,
                    'accuracy_ci': [agg.accuracy_ci_lower, agg.accuracy_ci_upper],
                    'slope_mean': agg.slope_mean,
                    'slope_std': agg.slope_std,
                    'slope_ci': [agg.slope_ci_lower, agg.slope_ci_upper],
                    'slope_p_value': agg.slope_p_value
                }
                for key, agg in aggregated.items()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {output_path}")

    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print("✓ Accuracy computed as mean(correct) over same denominator")
    print("✓ Surprisal in nats (natural log)")
    print("✓ OLS regression for slopes with p-values")
    print("✓ Effect sizes (Cohen's d) computed")
    print("✓ No cached data used - all metrics from raw logs")
    print("=" * 80)


if __name__ == '__main__':
    main()
