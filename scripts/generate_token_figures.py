#!/usr/bin/env python3
"""
Generate publication-ready figures for token prediction analysis.
Creates: coupling plot, surprise alignment, predictive validity, calibration.
"""

import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.token_analysis import TokenAnalysis


def plot_coupling_scatter(analysis: TokenAnalysis, output_dir: str):
    """Figure 1: Token NLL vs Belief Surprisal scatter plots."""
    try:
        import matplotlib.pyplot as plt
        from scipy.stats import pearsonr
    except ImportError:
        print("⚠ matplotlib not installed, skipping coupling scatter plot")
        return

    df = analysis.df.dropna(subset=['token_nll', 'belief_surprisal'])

    if len(df) == 0:
        print("⚠ No valid data for coupling plot")
        return

    envs = sorted(df['environment'].unique())
    n_envs = len(envs)

    fig, axes = plt.subplots(1, n_envs, figsize=(5*n_envs, 4))

    if n_envs == 1:
        axes = [axes]

    for idx, env in enumerate(envs):
        env_data = df[df['environment'] == env]

        ax = axes[idx]
        ax.scatter(
            env_data['token_nll'],
            env_data['belief_surprisal'],
            alpha=0.4,
            s=20,
            color='steelblue'
        )

        # Regression line
        if len(env_data) >= 2:
            z = np.polyfit(env_data['token_nll'], env_data['belief_surprisal'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(env_data['token_nll'].min(), env_data['token_nll'].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            # Correlation
            r, p_val = pearsonr(env_data['token_nll'], env_data['belief_surprisal'])

            ax.set_xlabel('Token NLL', fontsize=11)
            ax.set_ylabel('Belief Surprisal', fontsize=11)
            ax.set_title(f'{env}\nr = {r:.3f}, p = {p_val:.4f}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'figure1_coupling_scatter.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 1 to: {fig_path}")
    plt.close()


def plot_coupling_heatmap(output_dir: str):
    """Figure 2: Coupling correlation heatmap (Environment × Agent)."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("⚠ matplotlib/seaborn not installed, skipping heatmap")
        return

    coupling_file = os.path.join(output_dir, 'coupling_by_agent.csv')

    if not os.path.exists(coupling_file):
        print("⚠ coupling_by_agent.csv not found, skipping heatmap")
        return

    df = pd.read_csv(coupling_file)

    if len(df) == 0:
        print("⚠ No data in coupling_by_agent.csv")
        return

    # Pivot to matrix format
    matrix = df.pivot(index='environment', columns='agent_type', values='pearson_r')

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0,
        vmin=-0.5,
        vmax=1.0,
        cbar_kws={'label': 'Pearson r'},
        ax=ax
    )

    ax.set_title('Token NLL - Belief Surprisal Coupling\n(Environment × Agent Type)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Agent Type', fontsize=11)
    ax.set_ylabel('Environment', fontsize=11)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'figure2_coupling_heatmap.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 2 to: {fig_path}")
    plt.close()


def plot_temporal_alignment(analysis: TokenAnalysis, output_dir: str):
    """Figure 3: Temporal alignment of token NLL and belief surprisal."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not installed, skipping temporal alignment plot")
        return

    df = analysis.df.dropna(subset=['token_nll', 'belief_surprisal'])

    if len(df) == 0:
        print("⚠ No valid data for temporal alignment plot")
        return

    # Pick one representative episode per environment
    envs = sorted(df['environment'].unique())[:3]
    n_envs = min(3, len(envs))

    fig, axes = plt.subplots(n_envs, 1, figsize=(12, 4*n_envs))

    if n_envs == 1:
        axes = [axes]

    for idx, env in enumerate(envs):
        env_data = df[df['environment'] == env]

        # Pick first episode
        if len(env_data) == 0:
            continue

        first_episode = env_data['episode_id'].iloc[0]
        ep_data = env_data[env_data['episode_id'] == first_episode].sort_values('step')

        if len(ep_data) == 0:
            continue

        ax = axes[idx]

        # Normalize both metrics to [0, 1] for visualization
        token_range = ep_data['token_nll'].max() - ep_data['token_nll'].min()
        belief_range = ep_data['belief_surprisal'].max() - ep_data['belief_surprisal'].min()

        if token_range > 0:
            token_nll_norm = (ep_data['token_nll'] - ep_data['token_nll'].min()) / token_range
        else:
            token_nll_norm = ep_data['token_nll'] * 0

        if belief_range > 0:
            belief_norm = (ep_data['belief_surprisal'] - ep_data['belief_surprisal'].min()) / belief_range
        else:
            belief_norm = ep_data['belief_surprisal'] * 0

        ax.plot(ep_data['step'], token_nll_norm, 'o-', label='Token NLL (normalized)',
                linewidth=2, markersize=6, color='steelblue')
        ax.plot(ep_data['step'], belief_norm, 's-', label='Belief Surprisal (normalized)',
                linewidth=2, markersize=6, color='coral')

        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel('Normalized Value', fontsize=11)
        ax.set_title(f'{env} - Episode {first_episode}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'figure3_temporal_alignment.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 3 to: {fig_path}")
    plt.close()


def plot_predictive_validity(output_dir: str):
    """Figure 4: Predictive validity by environment and agent."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("⚠ matplotlib/seaborn not installed, skipping predictive validity plot")
        return

    validity_file = os.path.join(output_dir, 'predictive_validity.csv')

    if not os.path.exists(validity_file):
        print("⚠ predictive_validity.csv not found, skipping plot")
        return

    df = pd.read_csv(validity_file)

    if len(df) == 0:
        print("⚠ No data in predictive_validity.csv")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Box plot by environment and agent
    df_plot = df[['environment', 'agent_type', 'correlation']].copy()

    sns.boxplot(
        data=df_plot,
        x='environment',
        y='correlation',
        hue='agent_type',
        ax=ax,
        palette='Set2'
    )

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Environment', fontsize=11)
    ax.set_ylabel('Correlation (Token NLL → Future Accuracy)', fontsize=11)
    ax.set_title('Predictive Validity: Token NLL at t predicts Accuracy at t+1',
                 fontsize=13, fontweight='bold')
    ax.legend(title='Agent Type', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'figure4_predictive_validity.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 4 to: {fig_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate token prediction figures")
    parser.add_argument('results_dir', type=str, help='Directory with analysis results')
    args = parser.parse_args()

    results_dir = args.results_dir

    print("=" * 70)
    print("GENERATING TOKEN PREDICTION FIGURES")
    print("=" * 70)
    print(f"Results directory: {results_dir}")
    print()

    # Load analysis
    print("Loading analysis data...")
    analysis = TokenAnalysis(results_dir)

    # Create figures directory
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Figures directory: {figures_dir}")
    print()

    # Generate figures
    print("Generating Figure 1: Coupling scatter...")
    plot_coupling_scatter(analysis, figures_dir)

    print("Generating Figure 2: Coupling heatmap...")
    plot_coupling_heatmap(results_dir)

    print("Generating Figure 3: Temporal alignment...")
    plot_temporal_alignment(analysis, figures_dir)

    print("Generating Figure 4: Predictive validity...")
    plot_predictive_validity(results_dir)

    print()
    print("=" * 70)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 70)
    print(f"All figures saved to: {figures_dir}")
    print("\nGenerated figures:")
    for fig_file in sorted(Path(figures_dir).glob("figure*.png")):
        print(f"  - {fig_file.name}")
    print("=" * 70)


if __name__ == '__main__':
    main()
