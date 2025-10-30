#!/usr/bin/env python3
"""
Comprehensive statistical analysis for ACE study.

Includes:
- Paired t-tests between all agent pairs
- Bootstrap confidence intervals (10,000 resamples)
- Effect sizes (Cohen's d)
- Multiple comparison correction (Bonferroni)
- Power analysis
- Statistical plots

Usage:
    python scripts/analyze_with_statistics.py results/ace_full_n20
"""

import json
import glob
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import sys
from collections import defaultdict


class ACEStatisticalAnalysis:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.df = None
        self.load_data()

    def load_data(self):
        """Load all episode results into DataFrame"""
        episodes = []

        raw_dir = self.results_dir / "raw"
        if not raw_dir.exists():
            print(f"Error: {raw_dir} does not exist")
            sys.exit(1)

        for file in sorted(raw_dir.glob("*.json")):
            try:
                with open(file) as f:
                    data = json.load(f)

                    # Extract key metrics
                    test_results = data.get('test_results', [])
                    scores = [t.get('score', 0) for t in test_results]
                    correct_count = sum(1 for t in test_results if t.get('correct', False))

                    # Extract cost data (preregistration requirement)
                    cost_data = data.get('cost', {})
                    cost_usd = cost_data.get('total_cost_usd', 0.0)

                    episodes.append({
                        'episode_id': data.get('episode_id'),
                        'environment': data.get('environment'),
                        'agent_type': data.get('agent_type'),
                        'seed': data.get('seed'),
                        'accuracy': correct_count / len(test_results) if test_results else 0,
                        'mean_score': np.mean(scores) if scores else 0,
                        'total_tokens': data.get('total_input_tokens', 0) + data.get('total_output_tokens', 0),
                        'duration': data.get('duration_seconds', 0),
                        'num_steps': len(data.get('steps', [])),
                        'cost_usd': cost_usd,
                        'input_tokens': data.get('total_input_tokens', 0),
                        'output_tokens': data.get('total_output_tokens', 0),
                    })
            except Exception as e:
                print(f"Warning: Error loading {file}: {e}")

        self.df = pd.DataFrame(episodes)
        print(f"Loaded {len(self.df)} episodes")
        print(f"Agents: {sorted(self.df['agent_type'].unique())}")
        print(f"Environments: {sorted(self.df['environment'].unique())}")
        print()

    def paired_t_tests(self):
        """Run paired t-tests between all agent pairs"""
        print("\n" + "="*70)
        print("PAIRED T-TESTS (Accuracy)")
        print("="*70)

        agents = sorted(self.df['agent_type'].unique())
        results = []

        # All pairwise comparisons
        for i, agent1 in enumerate(agents):
            for agent2 in agents[i+1:]:
                # Get paired data (same seed, same environment)
                data1 = self.df[self.df['agent_type'] == agent1].set_index(['environment', 'seed'])['accuracy']
                data2 = self.df[self.df['agent_type'] == agent2].set_index(['environment', 'seed'])['accuracy']

                # Paired comparison
                common_idx = data1.index.intersection(data2.index)
                paired1 = data1.loc[common_idx]
                paired2 = data2.loc[common_idx]

                if len(paired1) > 1:
                    t_stat, p_value = stats.ttest_rel(paired1, paired2)

                    # Cohen's d for paired samples
                    diff = paired1 - paired2
                    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0

                    results.append({
                        'comparison': f"{agent1} vs {agent2}",
                        'n': len(paired1),
                        'mean_diff': diff.mean(),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'significant': p_value < 0.05
                    })

                    sig_mark = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    print(f"\n{agent1} vs {agent2}:")
                    print(f"  Mean difference: {diff.mean():+.3f} ({diff.mean()*100:+.1f} percentage points)")
                    print(f"  t({len(paired1)-1}) = {t_stat:.3f}, p = {p_value:.4f} {sig_mark}")
                    print(f"  Cohen's d = {cohens_d:.3f}")

        # Bonferroni correction
        if results:
            n_comparisons = len(results)
            bonferroni_alpha = 0.05 / n_comparisons
            print(f"\nBonferroni-corrected α = {bonferroni_alpha:.4f} ({n_comparisons} comparisons)")

            print("\nSurviving multiple comparison correction:")
            for r in results:
                if r['p_value'] < bonferroni_alpha:
                    print(f"  ✅ {r['comparison']}: p={r['p_value']:.4f} < {bonferroni_alpha:.4f}")

        return pd.DataFrame(results) if results else pd.DataFrame()

    def bootstrap_confidence_intervals(self, n_bootstrap=10000):
        """Calculate bootstrap 95% CIs for each agent's accuracy"""
        print("\n" + "="*70)
        print(f"BOOTSTRAP CONFIDENCE INTERVALS ({n_bootstrap:,} resamples)")
        print("="*70)

        results = []

        for agent in sorted(self.df['agent_type'].unique()):
            agent_data = self.df[self.df['agent_type'] == agent]['accuracy'].values

            if len(agent_data) == 0:
                continue

            # Bootstrap resampling
            bootstrap_means = []
            rng = np.random.RandomState(42)
            for _ in range(n_bootstrap):
                sample = rng.choice(agent_data, size=len(agent_data), replace=True)
                bootstrap_means.append(sample.mean())

            # Calculate 95% CI
            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)

            results.append({
                'agent': agent,
                'mean': agent_data.mean(),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_width': ci_upper - ci_lower,
                'n': len(agent_data)
            })

            print(f"\n{agent}:")
            print(f"  Mean: {agent_data.mean():.3f} ({agent_data.mean()*100:.1f}%)")
            print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
            print(f"  CI width: {ci_upper - ci_lower:.3f}")
            print(f"  n: {len(agent_data)}")

        return pd.DataFrame(results)

    def effect_sizes(self):
        """Calculate Cohen's d between all pairs"""
        print("\n" + "="*70)
        print("EFFECT SIZES (Cohen's d)")
        print("="*70)
        print("Interpretation: |d| < 0.2 = negligible, 0.2-0.5 = small,")
        print("                0.5-0.8 = medium, >0.8 = large")
        print()

        agents = sorted(self.df['agent_type'].unique())

        for i, agent1 in enumerate(agents):
            for agent2 in agents[i+1:]:
                data1 = self.df[self.df['agent_type'] == agent1]['accuracy']
                data2 = self.df[self.df['agent_type'] == agent2]['accuracy']

                if len(data1) == 0 or len(data2) == 0:
                    continue

                # Pooled standard deviation
                n1, n2 = len(data1), len(data2)
                var1, var2 = data1.var(), data2.var()
                pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))

                # Cohen's d
                if pooled_std > 0:
                    cohens_d = (data1.mean() - data2.mean()) / pooled_std
                else:
                    cohens_d = 0

                magnitude = ("negligible" if abs(cohens_d) < 0.2 else
                           "small" if abs(cohens_d) < 0.5 else
                           "medium" if abs(cohens_d) < 0.8 else "large")

                print(f"{agent1:15s} vs {agent2:15s}: d = {cohens_d:+.3f} ({magnitude})")

    def power_analysis(self):
        """Estimate statistical power for detecting differences"""
        print("\n" + "="*70)
        print("POWER ANALYSIS")
        print("="*70)

        # Focus on ACE vs other agents
        ace_data = self.df[self.df['agent_type'] == 'a_c_e']['accuracy']

        for agent in sorted(self.df['agent_type'].unique()):
            if agent == 'a_c_e':
                continue

            agent_data = self.df[self.df['agent_type'] == agent]['accuracy']

            if len(ace_data) == 0 or len(agent_data) == 0:
                continue

            # Observed effect size
            pooled_std = np.sqrt((ace_data.var() + agent_data.var()) / 2)
            if pooled_std > 0:
                observed_d = (ace_data.mean() - agent_data.mean()) / pooled_std
            else:
                observed_d = 0

            print(f"\nACE vs {agent}:")
            print(f"  Observed effect size: d = {observed_d:.3f}")
            print(f"  Current n per group: ACE={len(ace_data)}, {agent}={len(agent_data)}")

        print(f"\n  Sample size recommendations (for 80% power at α=0.05):")
        print(f"    To detect d=0.5 (medium effect): n≥64 per group")
        print(f"    To detect d=0.3 (small effect): n≥176 per group")
        print(f"    To detect d=0.2 (small effect): n≥394 per group")

    def cost_analysis(self):
        """
        Analyze costs per agent (preregistration requirement).

        Returns per-agent cost statistics with mean, std, min, max.
        """
        print("\n" + "="*70)
        print("COST ANALYSIS (Preregistration Requirement)")
        print("="*70)
        print()

        results = []

        for agent in sorted(self.df['agent_type'].unique()):
            agent_data = self.df[self.df['agent_type'] == agent]
            costs = agent_data['cost_usd'].values

            if len(costs) == 0:
                continue

            results.append({
                'agent': agent,
                'mean_cost_usd': costs.mean(),
                'std_cost_usd': costs.std(),
                'min_cost_usd': costs.min(),
                'max_cost_usd': costs.max(),
                'total_cost_usd': costs.sum(),
                'n_episodes': len(costs)
            })

            print(f"{agent}:")
            print(f"  Mean cost per episode: ${costs.mean():.4f} ± ${costs.std():.4f}")
            print(f"  Range: ${costs.min():.4f} - ${costs.max():.4f}")
            print(f"  Total cost ({len(costs)} episodes): ${costs.sum():.2f}")
            print()

        return pd.DataFrame(results)

    def cost_efficiency_analysis(self):
        """
        Calculate cost-efficiency metric: accuracy / cost_usd.

        Higher is better (more accuracy per dollar spent).
        """
        print("\n" + "="*70)
        print("COST-EFFICIENCY ANALYSIS")
        print("="*70)
        print("Metric: Accuracy / Cost (higher = more efficient)")
        print()

        results = []

        for agent in sorted(self.df['agent_type'].unique()):
            agent_data = self.df[self.df['agent_type'] == agent]

            if len(agent_data) == 0:
                continue

            # Calculate efficiency for each episode
            agent_data_copy = agent_data.copy()
            agent_data_copy['efficiency'] = agent_data_copy['accuracy'] / agent_data_copy['cost_usd']

            # Aggregate statistics
            mean_accuracy = agent_data_copy['accuracy'].mean()
            mean_cost = agent_data_copy['cost_usd'].mean()
            mean_efficiency = agent_data_copy['efficiency'].mean()

            results.append({
                'agent': agent,
                'mean_accuracy': mean_accuracy,
                'mean_cost_usd': mean_cost,
                'efficiency': mean_efficiency,
                'rank': 0  # Will be filled below
            })

            print(f"{agent}:")
            print(f"  Accuracy: {mean_accuracy:.3f} ({mean_accuracy*100:.1f}%)")
            print(f"  Cost: ${mean_cost:.4f} per episode")
            print(f"  Efficiency: {mean_efficiency:.2f} accuracy points per $1")
            print()

        # Rank by efficiency
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('efficiency', ascending=False)
        df_results['rank'] = range(1, len(df_results) + 1)

        print("\nRanking (by efficiency):")
        for _, row in df_results.iterrows():
            print(f"  {row['rank']}. {row['agent']:15s} - {row['efficiency']:.2f} pts/$1")

        return df_results

    def summary_table(self):
        """Create summary table of all agents"""
        print("\n" + "="*70)
        print("SUMMARY TABLE")
        print("="*70)
        print()

        summary = self.df.groupby('agent_type').agg({
            'accuracy': ['mean', 'std', 'count'],
            'total_tokens': ['mean'],
            'duration': ['mean'],
            'cost_usd': ['mean', 'std']
        }).round(3)

        print(summary)
        print()

    def run_full_analysis(self):
        """Run complete statistical analysis"""
        print("\n" + "="*70)
        print("COMPREHENSIVE STATISTICAL ANALYSIS")
        print("="*70)

        # Summary
        self.summary_table()

        # Run all tests
        t_test_results = self.paired_t_tests()
        ci_results = self.bootstrap_confidence_intervals()
        self.effect_sizes()
        self.power_analysis()

        # Cost analysis (preregistration requirement)
        cost_results = self.cost_analysis()
        efficiency_results = self.cost_efficiency_analysis()

        # Export results
        output_dir = self.results_dir
        if not t_test_results.empty:
            t_test_results.to_csv(output_dir / 'statistical_ttests.csv', index=False)
            print(f"\n✅ Saved: {output_dir / 'statistical_ttests.csv'}")

        if not ci_results.empty:
            ci_results.to_csv(output_dir / 'statistical_confidence_intervals.csv', index=False)
            print(f"✅ Saved: {output_dir / 'statistical_confidence_intervals.csv'}")

        if not cost_results.empty:
            cost_results.to_csv(output_dir / 'cost_analysis.csv', index=False)
            print(f"✅ Saved: {output_dir / 'cost_analysis.csv'}")

        if not efficiency_results.empty:
            efficiency_results.to_csv(output_dir / 'cost_efficiency.csv', index=False)
            print(f"✅ Saved: {output_dir / 'cost_efficiency.csv'}")

        # Export full DataFrame
        self.df.to_csv(output_dir / 'statistical_raw_data.csv', index=False)
        print(f"✅ Saved: {output_dir / 'statistical_raw_data.csv'}")

        # Generate summary statement (preregistration requirement)
        self._generate_summary_statement(ci_results, cost_results, efficiency_results)

        print("\n" + "="*70)
        print("✅ STATISTICAL ANALYSIS COMPLETE")
        print("="*70)

    def _generate_summary_statement(self, ci_results, cost_results, efficiency_results):
        """
        Generate preregistration-compliant summary statement.

        Example: 'ACE achieves 75% accuracy at $0.285 per episode'
        """
        print("\n" + "="*70)
        print("SUMMARY STATEMENT (For Preregistration)")
        print("="*70)
        print()

        if ci_results.empty or cost_results.empty:
            print("Insufficient data for summary statement")
            return

        # Merge accuracy and cost data
        summary = ci_results.merge(cost_results, on='agent')

        for _, row in summary.iterrows():
            agent = row['agent']
            accuracy = row['mean']
            ci_lower = row['ci_lower']
            ci_upper = row['ci_upper']
            cost = row['mean_cost_usd']
            n = row['n_episodes']

            print(f"{agent}:")
            print(f"  Achieves {accuracy:.1%} accuracy [95% CI: {ci_lower:.1%}, {ci_upper:.1%}]")
            print(f"  at ${cost:.4f} per episode (n={n})")
            print()

        # Write to markdown file
        output_path = self.results_dir / 'SUMMARY_STATEMENT.md'
        with open(output_path, 'w') as f:
            f.write("# Experiment Summary (Preregistration Requirement)\n\n")
            f.write("Generated from statistical analysis.\n\n")
            f.write("## Agent Performance\n\n")
            f.write("| Agent | Accuracy (95% CI) | Cost per Episode | Cost-Efficiency |\n")
            f.write("|-------|-------------------|------------------|------------------|\n")

            # Merge with efficiency data
            full_summary = summary.merge(
                efficiency_results[['agent', 'efficiency']],
                on='agent',
                how='left'
            )

            for _, row in full_summary.iterrows():
                agent = row['agent']
                accuracy = row['mean']
                ci_lower = row['ci_lower']
                ci_upper = row['ci_upper']
                cost = row['mean_cost_usd']
                efficiency = row.get('efficiency', 0)

                f.write(f"| {agent:12s} | {accuracy:.1%} [{ci_lower:.1%}, {ci_upper:.1%}] | ")
                f.write(f"${cost:.4f} | {efficiency:.2f} pts/$1 |\n")

            f.write("\n## Key Findings\n\n")

            # Find best accuracy
            best_acc_row = full_summary.loc[full_summary['mean'].idxmax()]
            f.write(f"- **Highest Accuracy**: {best_acc_row['agent']} ")
            f.write(f"({best_acc_row['mean']:.1%} at ${best_acc_row['mean_cost_usd']:.4f}/episode)\n")

            # Find most efficient
            best_eff_row = full_summary.loc[full_summary['efficiency'].idxmax()]
            f.write(f"- **Most Efficient**: {best_eff_row['agent']} ")
            f.write(f"({best_eff_row['efficiency']:.2f} accuracy points per $1)\n")

            # Total cost
            total_cost = cost_results['total_cost_usd'].sum()
            total_episodes = cost_results['n_episodes'].sum()
            f.write(f"\n## Experiment Totals\n\n")
            f.write(f"- Total episodes: {total_episodes}\n")
            f.write(f"- Total cost: ${total_cost:.2f}\n")
            f.write(f"- Average cost per episode: ${total_cost/total_episodes:.4f}\n")

        print(f"✅ Saved: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_with_statistics.py <results_dir>")
        print("Example: python scripts/analyze_with_statistics.py results/ace_full_n20")
        sys.exit(1)

    results_dir = sys.argv[1]

    if not Path(results_dir).exists():
        print(f"Error: Directory {results_dir} does not exist")
        sys.exit(1)

    analyzer = ACEStatisticalAnalysis(results_dir)
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
