#!/usr/bin/env python3
"""
Generate figures from pilot experiment results.

Creates:
1. Pareto plot (accuracy vs tokens)
2. Budget sweep (token cap vs accuracy)
3. Summary table

Usage:
    python scripts/generate_pilot_figures.py results/ace_pilot
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Try to import matplotlib, provide helpful error if missing
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


def load_episode_logs(results_dir: Path) -> pd.DataFrame:
    """Load all episode logs into DataFrame"""
    raw_dir = results_dir / "raw"

    if not raw_dir.exists():
        print(f"Error: {raw_dir} does not exist")
        return pd.DataFrame()

    episodes = []
    for log_file in raw_dir.glob("*.json"):
        try:
            with open(log_file) as f:
                episode = json.load(f)
                episodes.append(episode)
        except Exception as e:
            print(f"Warning: Failed to load {log_file}: {e}")

    if not episodes:
        print("No episode logs found")
        return pd.DataFrame()

    return pd.DataFrame(episodes)


def compute_aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate metrics per agent"""
    if df.empty:
        return pd.DataFrame()

    # Group by agent
    agent_metrics = []
    for agent in df['agent'].unique():
        agent_df = df[df['agent'] == agent]

        # Calculate metrics
        accuracy = agent_df['overall_accuracy'].mean()
        tokens_per_ep = agent_df['total_tokens'].mean()
        tokens_std = agent_df['total_tokens'].std()
        episodes = len(agent_df)

        # Efficiency: tokens per % accuracy
        tokens_per_pct = tokens_per_ep / accuracy if accuracy > 0 else float('inf')

        agent_metrics.append({
            'agent': agent,
            'accuracy': accuracy,
            'accuracy_std': agent_df['overall_accuracy'].std(),
            'tokens_per_episode': tokens_per_ep,
            'tokens_std': tokens_std,
            'tokens_per_pct_accuracy': tokens_per_pct,
            'episodes': episodes
        })

    return pd.DataFrame(agent_metrics)


def plot_pareto(df: pd.DataFrame, output_path: Path):
    """Generate Pareto frontier plot"""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping Pareto plot (matplotlib not available)")
        return

    metrics = compute_aggregate_metrics(df)
    if metrics.empty:
        print("No data for Pareto plot")
        return

    plt.figure(figsize=(10, 6))

    # Plot each agent
    agent_colors = {
        'observer': 'blue',
        'actor': 'red',
        'a_c_e': 'purple'
    }

    for _, row in metrics.iterrows():
        agent = row['agent']
        color = agent_colors.get(agent, 'gray')

        plt.scatter(
            row['tokens_per_episode'],
            row['accuracy'] * 100,
            s=200,
            color=color,
            label=agent.replace('_', ' ').title(),
            alpha=0.7,
            edgecolors='black',
            linewidth=1.5
        )

        # Add error bars
        plt.errorbar(
            row['tokens_per_episode'],
            row['accuracy'] * 100,
            xerr=row['tokens_std'],
            yerr=row['accuracy_std'] * 100,
            fmt='none',
            color=color,
            alpha=0.3
        )

    plt.xlabel("Tokens per Episode", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Pareto Frontier: Accuracy vs Cost", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add Pareto frontier line
    frontier_df = metrics.sort_values('tokens_per_episode')
    plt.plot(
        frontier_df['tokens_per_episode'],
        frontier_df['accuracy'] * 100,
        '--',
        color='gray',
        alpha=0.5,
        linewidth=1
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Pareto plot to {output_path}")
    plt.close()


def generate_summary_table(df: pd.DataFrame, output_path: Path):
    """Generate summary table CSV"""
    metrics = compute_aggregate_metrics(df)
    if metrics.empty:
        print("No data for summary table")
        return

    # Sort by accuracy descending
    metrics = metrics.sort_values('accuracy', ascending=False)

    # Round for readability
    metrics['accuracy'] = (metrics['accuracy'] * 100).round(1)
    metrics['accuracy_std'] = (metrics['accuracy_std'] * 100).round(1)
    metrics['tokens_per_episode'] = metrics['tokens_per_episode'].round(0)
    metrics['tokens_std'] = metrics['tokens_std'].round(0)
    metrics['tokens_per_pct_accuracy'] = metrics['tokens_per_pct_accuracy'].round(0)

    # Save CSV
    metrics.to_csv(output_path, index=False)
    print(f"Saved summary table to {output_path}")

    # Print to console
    print("\n" + "="*80)
    print("PILOT RESULTS SUMMARY")
    print("="*80)
    print(metrics.to_string(index=False))
    print("="*80)


def generate_summary_json(df: pd.DataFrame, output_path: Path):
    """Generate summary JSON"""
    metrics = compute_aggregate_metrics(df)
    if metrics.empty:
        print("No data for summary JSON")
        return

    summary = {
        "total_episodes": len(df),
        "agents": metrics.to_dict(orient='records'),
        "comparison": {
            "best_accuracy": metrics.loc[metrics['accuracy'].idxmax()].to_dict() if not metrics.empty else None,
            "best_efficiency": metrics.loc[metrics['tokens_per_pct_accuracy'].idxmin()].to_dict() if not metrics.empty else None
        }
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary JSON to {output_path}")


def check_pareto_position(df: pd.DataFrame):
    """Check if ACE is on Pareto frontier"""
    metrics = compute_aggregate_metrics(df)
    if metrics.empty or 'a_c_e' not in metrics['agent'].values:
        print("\nACE not found in results")
        return

    ace_row = metrics[metrics['agent'] == 'a_c_e'].iloc[0]

    # Check if dominated (another agent better on both metrics)
    dominated = False
    for _, row in metrics[metrics['agent'] != 'a_c_e'].iterrows():
        if (row['accuracy'] >= ace_row['accuracy'] and
            row['tokens_per_episode'] <= ace_row['tokens_per_episode']):
            dominated = True
            print(f"\n⚠️  ACE DOMINATED by {row['agent'].upper()}")
            print(f"   {row['agent']}: {row['accuracy']:.1f}% @ {row['tokens_per_episode']:.0f} tokens")
            print(f"   ACE: {ace_row['accuracy']:.1f}% @ {ace_row['tokens_per_episode']:.0f} tokens")

    if not dominated:
        print(f"\n✅ ACE ON PARETO FRONTIER")
        print(f"   Accuracy: {ace_row['accuracy']:.1f}%")
        print(f"   Tokens/ep: {ace_row['tokens_per_episode']:.0f}")
        print(f"   Efficiency: {ace_row['tokens_per_pct_accuracy']:.0f} tokens/%")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_pilot_figures.py results/ace_pilot")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    print(f"Loading results from {results_dir}")

    # Load episode logs
    df = load_episode_logs(results_dir)

    if df.empty:
        print("No episode data found")
        sys.exit(1)

    print(f"Loaded {len(df)} episodes")

    # Generate outputs
    plot_pareto(df, results_dir / "pareto_plot.png")
    generate_summary_table(df, results_dir / "aggregate_metrics.csv")
    generate_summary_json(df, results_dir / "summary.json")
    check_pareto_position(df)

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Review Pareto plot: ", results_dir / "pareto_plot.png")
    print("2. Check summary table: ", results_dir / "aggregate_metrics.csv")
    print("3. Decide: Proceed to full experiment (600 episodes)?")
    print("="*80)


if __name__ == "__main__":
    main()
