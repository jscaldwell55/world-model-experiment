#!/usr/bin/env python3
"""
Analyze pilot token prediction results.

Compute A1 (coupling) and A2 (surprise detection) metrics to test whether
token-level NLL correlates with belief surprisal.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_token_logs(log_dir: str) -> pd.DataFrame:
    """Load all token logs from directory into DataFrame.

    Args:
        log_dir: Directory containing *_token.json files

    Returns:
        DataFrame with columns: episode_id, environment, agent_type, step,
        token_nll, per_token_nll, belief_surprisal, accuracy
    """

    records = []
    log_files = list(Path(log_dir).glob("*_token.json"))

    if not log_files:
        print(f"Warning: No token log files found in {log_dir}")
        return pd.DataFrame()

    print(f"Found {len(log_files)} token log files")

    for log_file in log_files:
        with open(log_file) as f:
            data = json.load(f)

        episode_id = data['episode_id']

        # Parse environment and agent from episode_id
        # Format: "EnvironmentName_AgentName_ep000"
        parts = episode_id.rsplit('_', 1)[0].split('_')

        # Handle names like "HotPotLab" -> "HotPot"
        if len(parts) >= 2:
            # Last part is agent name (e.g., "ActorAgent" or "ObserverAgent")
            agent_part = parts[-1].replace('Agent', '')
            # Everything before is environment name
            env_part = '_'.join(parts[:-1]).replace('Lab', '')
        else:
            env_part = parts[0]
            agent_part = 'unknown'

        # Extract entries
        for entry in data.get('entries', []):
            records.append({
                'episode_id': episode_id,
                'environment': env_part,
                'agent_type': agent_part,
                'step': entry['step'],
                'token_nll': entry['sequence_nll'],
                'per_token_nll': entry['per_token_nll'],
                'belief_surprisal': entry.get('belief_surprisal'),
                'accuracy': entry.get('accuracy'),
            })

    return pd.DataFrame(records)


def compute_coupling(df: pd.DataFrame) -> pd.DataFrame:
    """A1: Compute correlation between token NLL and belief surprisal.

    Args:
        df: DataFrame with token_nll and belief_surprisal columns

    Returns:
        DataFrame with coupling statistics per environment
    """
    from scipy.stats import pearsonr, spearmanr

    # Filter to rows with both values
    df_valid = df.dropna(subset=['token_nll', 'belief_surprisal'])

    if len(df_valid) == 0:
        print("Warning: No valid data for coupling analysis")
        return pd.DataFrame()

    results = []

    for env in sorted(df_valid['environment'].unique()):
        env_data = df_valid[df_valid['environment'] == env]

        if len(env_data) < 2:
            print(f"Warning: Not enough data for {env} (n={len(env_data)})")
            continue

        # Pearson correlation
        r_pearson, p_pearson = pearsonr(
            env_data['token_nll'],
            env_data['belief_surprisal']
        )

        # Spearman correlation
        r_spearman, p_spearman = spearmanr(
            env_data['token_nll'],
            env_data['belief_surprisal']
        )

        results.append({
            'environment': env,
            'pearson_r': r_pearson,
            'pearson_p': p_pearson,
            'spearman_r': r_spearman,
            'spearman_p': p_spearman,
            'n_steps': len(env_data),
            'mean_token_nll': env_data['token_nll'].mean(),
            'mean_belief_surprisal': env_data['belief_surprisal'].mean(),
        })

    return pd.DataFrame(results)


def plot_coupling(df: pd.DataFrame, output_dir: str):
    """Generate coupling visualization.

    Args:
        df: DataFrame with token_nll and belief_surprisal columns
        output_dir: Directory to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plots")
        return

    from scipy.stats import pearsonr

    df_valid = df.dropna(subset=['token_nll', 'belief_surprisal'])

    if len(df_valid) == 0:
        print("Warning: No valid data for plotting")
        return

    envs = sorted(df_valid['environment'].unique())
    n_envs = len(envs)

    if n_envs == 0:
        return

    fig, axes = plt.subplots(1, min(3, n_envs), figsize=(5*min(3, n_envs), 4))

    # Handle case of single environment
    if n_envs == 1:
        axes = [axes]

    for idx, env in enumerate(envs):
        if idx >= 3:
            break

        env_data = df_valid[df_valid['environment'] == env]

        ax = axes[idx]
        ax.scatter(
            env_data['token_nll'],
            env_data['belief_surprisal'],
            alpha=0.5,
            s=30
        )

        # Add regression line
        if len(env_data) >= 2:
            z = np.polyfit(env_data['token_nll'], env_data['belief_surprisal'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(env_data['token_nll'].min(), env_data['token_nll'].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            # Compute correlation
            r, p_val = pearsonr(env_data['token_nll'], env_data['belief_surprisal'])

            ax.set_xlabel('Token NLL', fontsize=11)
            ax.set_ylabel('Belief Surprisal', fontsize=11)
            ax.set_title(f'{env}\nr={r:.3f}, p={p_val:.4f}, n={len(env_data)}', fontsize=12)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'coupling_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved coupling plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze token prediction pilot results")
    parser.add_argument('log_dir', type=str, help='Directory containing token logs')
    parser.add_argument('--no-plot', action='store_true', help='Skip generating plots')
    args = parser.parse_args()

    print("=" * 70)
    print("TOKEN PREDICTION PILOT ANALYSIS")
    print("=" * 70)
    print()

    # Load data
    print(f"Loading token logs from: {args.log_dir}")
    df = load_token_logs(args.log_dir)

    if len(df) == 0:
        print("ERROR: No data loaded")
        sys.exit(1)

    print(f"Loaded {len(df)} steps from {df['episode_id'].nunique()} episodes")
    print()

    # Summary statistics
    print("=== Summary Statistics ===")
    print()
    summary = df.groupby('environment')[['token_nll', 'belief_surprisal']].agg(['count', 'mean', 'std'])
    print(summary)
    print()

    # A1: Coupling
    print("=== A1: Coupling Analysis ===")
    print()
    coupling_results = compute_coupling(df)

    if len(coupling_results) > 0:
        print(coupling_results.to_string(index=False))
        print()

        # Expected pattern
        print("Expected pattern: HotPot > SwitchLight > ChemTile")
        sorted_by_r = coupling_results.sort_values('pearson_r', ascending=False)
        print(f"Actual order:     {' > '.join(sorted_by_r['environment'].values)}")

        # Check if pattern holds
        env_order = list(sorted_by_r['environment'].values)
        expected_order = ['HotPot', 'SwitchLight', 'ChemTile']

        # Normalize names for comparison
        env_order_norm = [e.replace('Lab', '') for e in env_order]

        if env_order_norm == expected_order[:len(env_order_norm)]:
            print("✓ Pattern matches expectation!")
        else:
            print("✗ Pattern differs from expectation")

        print()

        # Save results
        output_dir = args.log_dir
        coupling_path = os.path.join(output_dir, 'coupling_results.csv')
        coupling_results.to_csv(coupling_path, index=False)
        print(f"✓ Saved coupling results to {coupling_path}")

        # Plot
        if not args.no_plot:
            plot_coupling(df, output_dir)
    else:
        print("No coupling results (insufficient data)")

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
