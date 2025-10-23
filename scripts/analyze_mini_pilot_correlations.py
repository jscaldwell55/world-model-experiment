#!/usr/bin/env python3
"""
Analyze Mini Pilot Correlations: Token NLL vs Belief Surprisal
Compute correlation metrics and compare to Pilot 2 baseline.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Pilot 2 baseline results for comparison
PILOT_2_BASELINE = {
    'HotPotLab': {'pearson_r': -0.292, 'spearman_rho': -0.201, 'n': 20},
    'SwitchLight': {'pearson_r': -0.137, 'spearman_rho': 0.215, 'n': 20},
    'ChemTile': {'pearson_r': -0.194, 'spearman_rho': -0.006, 'n': 14},
}

def load_episode_data(filepath: Path) -> Dict:
    """Load episode JSON data."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_metrics(episode_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract token NLL and belief surprisal per step.

    Returns:
        token_nll: array of per-token NLL values
        belief_surprisal: array of belief surprisal values
    """
    entries = episode_data['entries']

    token_nll = []
    belief_surprisal = []

    for entry in entries:
        # Use per_token_nll as the primary metric
        token_nll.append(entry['per_token_nll'])
        belief_surprisal.append(entry['belief_surprisal'])

    return np.array(token_nll), np.array(belief_surprisal)

def compute_correlations(x: np.ndarray, y: np.ndarray) -> Dict:
    """Compute Pearson and Spearman correlations with p-values."""
    # Remove any zero-variance data points from belief surprisal
    # (but keep them in the analysis)

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)

    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p,
        'n': len(x),
    }

def detect_outliers(x: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Detect outliers using z-score method."""
    if np.std(x) == 0:
        return np.zeros(len(x), dtype=bool)
    z_scores = np.abs(stats.zscore(x))
    return z_scores > threshold

def diagnostic_checks(token_nll: np.ndarray, belief_surprisal: np.ndarray,
                     env_name: str) -> Dict:
    """Run diagnostic checks on the data."""
    diagnostics = {
        'env': env_name,
        'n_steps': len(token_nll),
        'nll_mean': np.mean(token_nll),
        'nll_std': np.std(token_nll),
        'surprisal_mean': np.mean(belief_surprisal),
        'surprisal_std': np.std(belief_surprisal),
        'nll_outliers': detect_outliers(token_nll),
        'surprisal_outliers': detect_outliers(belief_surprisal),
    }

    # Count outliers
    diagnostics['n_nll_outliers'] = np.sum(diagnostics['nll_outliers'])
    diagnostics['n_surprisal_outliers'] = np.sum(diagnostics['surprisal_outliers'])

    # Red flags
    diagnostics['red_flags'] = []

    if diagnostics['nll_std'] < 0.1:
        diagnostics['red_flags'].append('NLL variance collapse (std < 0.1)')
    if diagnostics['surprisal_std'] < 0.1:
        diagnostics['red_flags'].append('Surprisal variance collapse (std < 0.1)')
    if diagnostics['n_nll_outliers'] > 0:
        diagnostics['red_flags'].append(f'{diagnostics["n_nll_outliers"]} NLL outliers (>3σ)')
    if diagnostics['n_surprisal_outliers'] > 0:
        diagnostics['red_flags'].append(f'{diagnostics["n_surprisal_outliers"]} Surprisal outliers (>3σ)')

    return diagnostics

def plot_scatter(token_nll: np.ndarray, belief_surprisal: np.ndarray,
                env_name: str, corr_stats: Dict, output_dir: Path):
    """Generate scatter plot with regression line."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    ax.scatter(token_nll, belief_surprisal, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

    # Add regression line
    z = np.polyfit(token_nll, belief_surprisal, 1)
    p = np.poly1d(z)
    x_line = np.linspace(token_nll.min(), token_nll.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    # Labels and title
    ax.set_xlabel('Token NLL (per-token)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Belief Surprisal', fontsize=12, fontweight='bold')
    ax.set_title(f'{env_name}: Token NLL vs Belief Surprisal\n' +
                 f'Pearson r={corr_stats["pearson_r"]:.3f} (p={corr_stats["pearson_p"]:.3f}), ' +
                 f'Spearman ρ={corr_stats["spearman_rho"]:.3f} (p={corr_stats["spearman_p"]:.3f})',
                 fontsize=14, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3)

    # Save
    plt.tight_layout()
    plt.savefig(output_dir / f'{env_name}_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_timeseries(token_nll: np.ndarray, belief_surprisal: np.ndarray,
                   env_name: str, output_dir: Path):
    """Generate time series plot for both metrics."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    steps = np.arange(len(token_nll))

    # Token NLL
    ax1.plot(steps, token_nll, marker='o', linewidth=2, markersize=8,
             color='steelblue', label='Token NLL')
    ax1.set_ylabel('Token NLL (per-token)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{env_name}: Metrics Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # Belief Surprisal
    ax2.plot(steps, belief_surprisal, marker='s', linewidth=2, markersize=8,
             color='coral', label='Belief Surprisal')
    ax2.set_xlabel('Episode Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Belief Surprisal', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # Save
    plt.tight_layout()
    plt.savefig(output_dir / f'{env_name}_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()

def compare_to_pilot2(corr_stats: Dict, env_name: str) -> Dict:
    """Compare correlation results to Pilot 2 baseline."""
    if env_name not in PILOT_2_BASELINE:
        return {'match': 'N/A', 'notes': 'No Pilot 2 baseline available'}

    pilot2 = PILOT_2_BASELINE[env_name]

    # Check sign consistency
    pearson_sign_match = np.sign(corr_stats['pearson_r']) == np.sign(pilot2['pearson_r'])
    spearman_sign_match = np.sign(corr_stats['spearman_rho']) == np.sign(pilot2['spearman_rho'])

    # Check magnitude differences
    pearson_diff = abs(corr_stats['pearson_r'] - pilot2['pearson_r'])
    spearman_diff = abs(corr_stats['spearman_rho'] - pilot2['spearman_rho'])

    # Determine match quality
    if pearson_sign_match and spearman_sign_match and pearson_diff < 0.3 and spearman_diff < 0.3:
        match = 'Strong'
    elif pearson_sign_match and spearman_sign_match:
        match = 'Moderate (signs match)'
    elif pearson_diff < 0.2 and spearman_diff < 0.2:
        match = 'Weak (magnitude similar)'
    else:
        match = 'Poor (divergent)'

    notes = []
    if not pearson_sign_match:
        notes.append(f'Pearson sign flip: {pilot2["pearson_r"]:.3f} → {corr_stats["pearson_r"]:.3f}')
    if not spearman_sign_match:
        notes.append(f'Spearman sign flip: {pilot2["spearman_rho"]:.3f} → {corr_stats["spearman_rho"]:.3f}')
    if pearson_diff > 0.3:
        notes.append(f'Large Pearson difference: Δ={pearson_diff:.3f}')
    if spearman_diff > 0.3:
        notes.append(f'Large Spearman difference: Δ={spearman_diff:.3f}')

    return {
        'match': match,
        'pearson_sign_match': pearson_sign_match,
        'spearman_sign_match': spearman_sign_match,
        'pearson_diff': pearson_diff,
        'spearman_diff': spearman_diff,
        'notes': '; '.join(notes) if notes else 'Consistent with Pilot 2',
    }

def generate_report(results: Dict, output_path: Path):
    """Generate markdown analysis report."""
    report = []

    report.append("# Mini Pilot Correlation Analysis Report")
    report.append("")
    report.append("## Executive Summary")
    report.append("")
    report.append("Analysis of 3-episode mini pilot (seed 42) examining correlation between:")
    report.append("- **Token NLL** (per-token negative log-likelihood from GPT-4o-mini)")
    report.append("- **Belief Surprisal** (explicit world-model prediction error)")
    report.append("")
    report.append("---")
    report.append("")

    # Summary table
    report.append("## Summary Table")
    report.append("")
    report.append("| Environment | Pearson r | Spearman ρ | n_steps | NLL std | Surprisal std | Match Pilot 2? |")
    report.append("|-------------|-----------|------------|---------|---------|---------------|----------------|")

    for env_name in ['HotPotLab', 'SwitchLight', 'ChemTile']:
        r = results[env_name]
        report.append(
            f"| {env_name:11} | {r['corr']['pearson_r']:>9.3f} | "
            f"{r['corr']['spearman_rho']:>10.3f} | {r['diag']['n_steps']:>7} | "
            f"{r['diag']['nll_std']:>7.3f} | {r['diag']['surprisal_std']:>13.3f} | "
            f"{r['pilot2_comp']['match']:14} |"
        )

    report.append("")
    report.append("---")
    report.append("")

    # Detailed results per environment
    report.append("## Detailed Results by Environment")
    report.append("")

    for env_name in ['HotPotLab', 'SwitchLight', 'ChemTile']:
        r = results[env_name]

        report.append(f"### {env_name}")
        report.append("")

        # Correlation results
        report.append("**Correlation Metrics:**")
        report.append(f"- Pearson r = {r['corr']['pearson_r']:.3f} (p = {r['corr']['pearson_p']:.3f})")
        report.append(f"- Spearman ρ = {r['corr']['spearman_rho']:.3f} (p = {r['corr']['spearman_p']:.3f})")
        report.append(f"- Sample size: n = {r['corr']['n']}")
        report.append("")

        # Statistical significance
        if r['corr']['pearson_p'] < 0.05:
            report.append(f"✓ **Statistically significant** (Pearson p < 0.05)")
        else:
            report.append(f"✗ Not statistically significant (Pearson p = {r['corr']['pearson_p']:.3f})")
        report.append("")

        # Diagnostics
        report.append("**Diagnostic Checks:**")
        report.append(f"- Token NLL: μ={r['diag']['nll_mean']:.3f}, σ={r['diag']['nll_std']:.3f}")
        report.append(f"- Belief Surprisal: μ={r['diag']['surprisal_mean']:.3f}, σ={r['diag']['surprisal_std']:.3f}")
        report.append(f"- Outliers: {r['diag']['n_nll_outliers']} NLL, {r['diag']['n_surprisal_outliers']} Surprisal")
        report.append("")

        # Red flags
        if r['diag']['red_flags']:
            report.append("**🚩 Red Flags:**")
            for flag in r['diag']['red_flags']:
                report.append(f"- {flag}")
            report.append("")
        else:
            report.append("**✓ No red flags detected**")
            report.append("")

        # Pilot 2 comparison
        report.append("**Comparison to Pilot 2:**")
        report.append(f"- Match quality: **{r['pilot2_comp']['match']}**")
        report.append(f"- {r['pilot2_comp']['notes']}")
        report.append("")

        # Pearson-Spearman divergence check
        divergence = abs(r['corr']['pearson_r'] - r['corr']['spearman_rho'])
        if divergence > 0.3:
            report.append(f"**⚠️ Warning:** Pearson-Spearman divergence = {divergence:.3f} (>0.3 threshold)")
            report.append("This suggests outliers may be distorting the Pearson correlation.")
            report.append("")

        report.append("---")
        report.append("")

    # Overall interpretation
    report.append("## Overall Interpretation")
    report.append("")

    # Determine scenario
    all_pearson = [results[env]['corr']['pearson_r'] for env in ['HotPotLab', 'SwitchLight', 'ChemTile']]
    all_negative = all(r < -0.1 for r in all_pearson)
    all_near_zero = all(abs(r) < 0.3 for r in all_pearson)
    all_positive = all(r > 0.1 for r in all_pearson)
    mixed = not (all_negative or all_near_zero or all_positive)

    report.append("**Correlation Pattern:**")
    if all_negative:
        report.append("- **Scenario A: Negative Correlations** - Consistent anti-coupling pattern")
    elif all_near_zero:
        report.append("- **Scenario B: Near-Zero Correlations** - No meaningful coupling")
    elif all_positive:
        report.append("- **Scenario D: Positive Correlations** - Token predictions track belief surprisal")
    else:
        report.append("- **Scenario C: Mixed Correlations** - Inconsistent patterns across environments")
    report.append("")

    # Consistency with Pilot 2
    matches = [results[env]['pilot2_comp']['match'] for env in ['HotPotLab', 'SwitchLight', 'ChemTile']]
    strong_matches = sum(1 for m in matches if m == 'Strong')

    report.append("**Consistency with Pilot 2:**")
    if strong_matches >= 2:
        report.append(f"- **High consistency** ({strong_matches}/3 strong matches)")
        report.append("- Results replicate Pilot 2 findings")
    elif strong_matches == 1:
        report.append(f"- **Moderate consistency** ({strong_matches}/3 strong matches)")
        report.append("- Some divergence from Pilot 2, worth investigating")
    else:
        report.append(f"- **Low consistency** ({strong_matches}/3 strong matches)")
        report.append("- Significant divergence from Pilot 2, may indicate methodology issues")
    report.append("")

    # Red flags summary
    all_red_flags = []
    for env in ['HotPotLab', 'SwitchLight', 'ChemTile']:
        all_red_flags.extend(results[env]['diag']['red_flags'])

    if all_red_flags:
        report.append("**Critical Issues:**")
        for flag in set(all_red_flags):
            count = sum(1 for f in all_red_flags if f == flag)
            report.append(f"- {flag} ({count} environment(s))")
        report.append("")

    # Recommendation
    report.append("## Recommendation")
    report.append("")

    if strong_matches >= 2 and not all_red_flags:
        report.append("**✓ PROCEED with full experiment**")
        report.append("- Results are consistent with Pilot 2")
        report.append("- No major red flags detected")
        report.append("- The null/negative correlation pattern is stable")
    elif strong_matches >= 2 and all_red_flags:
        report.append("**⚠️ PROCEED with CAUTION**")
        report.append("- Results consistent with Pilot 2, but red flags present")
        report.append("- Address methodological concerns before scaling up")
    elif strong_matches < 2 and not all_red_flags:
        report.append("**🔍 INVESTIGATE methodology**")
        report.append("- Results diverge from Pilot 2 despite good data quality")
        report.append("- Run additional pilot episodes to understand variance")
    else:
        report.append("**⛔ DO NOT PROCEED - PIVOT required**")
        report.append("- Significant red flags and/or inconsistency with Pilot 2")
        report.append("- Review experimental design and data collection methodology")

    report.append("")
    report.append("---")
    report.append("")

    # Methodology notes
    report.append("## Methodology Notes")
    report.append("")
    report.append("- **Data source:** `results/raw/pilot_token_20251022_182749/`")
    report.append("- **Episodes:** 3 (seed 42)")
    report.append("- **Environments:** HotPotLab, SwitchLight, ChemTile")
    report.append("- **Steps per episode:** 10")
    report.append("- **Token predictor:** GPT-4o-mini (temperature=0.0)")
    report.append("- **Metric 1:** Per-token NLL (negative log-likelihood)")
    report.append("- **Metric 2:** Belief surprisal (explicit world-model error)")
    report.append("- **Correlation methods:** Pearson r, Spearman ρ")
    report.append("- **Outlier detection:** Z-score > 3.0")
    report.append("")
    report.append("---")
    report.append("")
    report.append(f"**Report generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Report saved to: {output_path}")

def main():
    """Main analysis pipeline."""
    # Paths
    data_dir = Path("/Users/jaycaldwell/world-model-experiment/results/raw/pilot_token_20251022_182749")
    output_dir = Path("/Users/jaycaldwell/world-model-experiment/results/mini_pilot_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # File mapping
    episodes = {
        'HotPotLab': data_dir / 'HotPotLab_ActorAgent_ep042_token.json',
        'SwitchLight': data_dir / 'SwitchLight_ActorAgent_ep042_token.json',
        'ChemTile': data_dir / 'ChemTile_ActorAgent_ep042_token.json',
    }

    results = {}

    print("\n" + "="*80)
    print("MINI PILOT CORRELATION ANALYSIS")
    print("="*80 + "\n")

    for env_name, filepath in episodes.items():
        print(f"Processing {env_name}...")

        # Load data
        episode_data = load_episode_data(filepath)

        # Extract metrics
        token_nll, belief_surprisal = extract_metrics(episode_data)

        # Compute correlations
        corr_stats = compute_correlations(token_nll, belief_surprisal)

        # Diagnostics
        diagnostics = diagnostic_checks(token_nll, belief_surprisal, env_name)

        # Pilot 2 comparison
        pilot2_comp = compare_to_pilot2(corr_stats, env_name)

        # Generate plots
        plot_scatter(token_nll, belief_surprisal, env_name, corr_stats, output_dir)
        plot_timeseries(token_nll, belief_surprisal, env_name, output_dir)

        # Store results
        results[env_name] = {
            'corr': corr_stats,
            'diag': diagnostics,
            'pilot2_comp': pilot2_comp,
            'token_nll': token_nll,
            'belief_surprisal': belief_surprisal,
        }

        print(f"  Pearson r = {corr_stats['pearson_r']:.3f} (p = {corr_stats['pearson_p']:.3f})")
        print(f"  Spearman ρ = {corr_stats['spearman_rho']:.3f} (p = {corr_stats['spearman_p']:.3f})")
        print(f"  Match Pilot 2: {pilot2_comp['match']}")
        print()

    # Generate report
    report_path = output_dir / 'mini_pilot_analysis.md'
    generate_report(results, report_path)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"- Scatter plots: *_scatter.png")
    print(f"- Time series plots: *_timeseries.png")
    print(f"- Analysis report: mini_pilot_analysis.md")
    print()

    # Quick summary
    print("QUICK SUMMARY:")
    print("-" * 80)
    for env_name in ['HotPotLab', 'SwitchLight', 'ChemTile']:
        r = results[env_name]['corr']
        print(f"{env_name:12} | Pearson r={r['pearson_r']:>6.3f} | "
              f"Spearman ρ={r['spearman_rho']:>6.3f} | "
              f"Match: {results[env_name]['pilot2_comp']['match']}")
    print()

if __name__ == '__main__':
    main()
