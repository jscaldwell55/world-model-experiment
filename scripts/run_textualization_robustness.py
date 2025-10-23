#!/usr/bin/env python3
"""
Test whether anti-coupling finding is robust to textualization choices.

Runs HotPot episodes with 3 textualization variants:
1. Original: "Boiling!" label, standard phrasing
2. Neutral: No emotive labels, technical phrasing
3. Format B: Abbreviated formatting, different units

Computes correlations and tests for statistical differences.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.hot_pot import HotPotLab
from agents.base import OpenAILLM
from agents.actor import ActorAgent
from models.belief_state import HotPotBelief
from textualization.hot_pot_text import HotPotTextualization
from textualization.hot_pot_text_neutral import HotPotTextualizationNeutral
from textualization.hot_pot_text_format_b import HotPotTextualizationFormatB
from experiments.token_runner import run_episode_with_tokens, create_predictor

# Configuration
SEEDS = [100, 101, 102, 103, 104]  # New seeds to avoid overlap with pilots
N_STEPS = 10
OUTPUT_DIR = Path("results/textualization_robustness")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def extract_metrics_from_logger(logger) -> Tuple[np.ndarray, np.ndarray]:
    """Extract token NLL and belief surprisal from TokenLogger."""
    token_nlls = []
    belief_surprisals = []

    for entry in logger.entries:
        token_nlls.append(entry.per_token_nll)
        # Handle None values in belief surprisal
        if entry.belief_surprisal is not None:
            belief_surprisals.append(entry.belief_surprisal)
        else:
            belief_surprisals.append(0.0)  # Treat None as zero surprisal

    return np.array(token_nlls), np.array(belief_surprisals)


def compute_correlation(token_nll: np.ndarray, belief_surprisal: np.ndarray) -> Dict:
    """Compute correlation statistics."""
    pearson_r, pearson_p = stats.pearsonr(token_nll, belief_surprisal)
    spearman_rho, spearman_p = stats.spearmanr(token_nll, belief_surprisal)

    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p,
        'n': len(token_nll),
        'token_nll_mean': np.mean(token_nll),
        'token_nll_std': np.std(token_nll),
        'surprisal_mean': np.mean(belief_surprisal),
        'surprisal_std': np.std(belief_surprisal),
    }


def fisher_z_test(r1: float, n1: int, r2: float, n2: int) -> Tuple[float, float]:
    """Test if two correlations are significantly different using Fisher's z-transformation."""
    # Fisher z-transformation
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))

    # Standard error of difference
    se_diff = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))

    # Z-statistic
    z_stat = (z1 - z2) / se_diff

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return z_stat, p_value


def plot_variant_comparison(results: Dict, output_dir: Path):
    """Generate 3-panel scatter plot comparing variants."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    variant_names = ['original', 'neutral', 'format_b']
    titles = ['Original (Misleading Label)', 'Neutral (No Label)', 'Format B (Abbreviated)']

    for idx, (variant, title) in enumerate(zip(variant_names, titles)):
        ax = axes[idx]

        # Get data
        token_nlls = results[variant]['all_token_nlls']
        surprisals = results[variant]['all_surprisals']
        corr_stats = results[variant]['correlation']

        # Scatter plot
        ax.scatter(token_nlls, surprisals, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)

        # Regression line
        if len(token_nlls) > 1 and np.std(token_nlls) > 0:
            z = np.polyfit(token_nlls, surprisals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(token_nlls.min(), token_nlls.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        # Labels
        ax.set_xlabel('Token NLL (per-token)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Belief Surprisal', fontsize=12, fontweight='bold')
        ax.set_title(f'{title}\nPearson r={corr_stats["pearson_r"]:.3f} (p={corr_stats["pearson_p"]:.3f})',
                    fontsize=13, fontweight='bold')

        # Grid
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'textualization_robustness_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved scatter plot to: {output_dir / 'textualization_robustness_scatter.png'}")


def generate_report(results: Dict, fisher_tests: Dict, output_dir: Path):
    """Generate comprehensive markdown report."""
    report = []

    report.append("# Textualization Robustness Analysis Report")
    report.append("")
    report.append("## Executive Summary")
    report.append("")
    report.append("Testing whether the anti-coupling finding (Token NLL vs Belief Surprisal) is:")
    report.append("- **Robust**: Persists across different textualization variants")
    report.append("- **Artifact**: Only appears with misleading labels")
    report.append("")
    report.append("---")
    report.append("")

    # Comparison table
    report.append("## Comparison Table")
    report.append("")
    report.append("| Variant | Pearson r | Spearman ρ | p-value | n_steps | NLL std | Surprisal std | Sign |")
    report.append("|---------|-----------|------------|---------|---------|---------|---------------|------|")

    for variant in ['original', 'neutral', 'format_b']:
        r = results[variant]['correlation']
        sign = "Negative" if r['pearson_r'] < -0.1 else ("Positive" if r['pearson_r'] > 0.1 else "Zero")
        report.append(
            f"| {variant.capitalize():7} | {r['pearson_r']:>9.3f} | "
            f"{r['spearman_rho']:>10.3f} | {r['pearson_p']:>7.3f} | {r['n']:>7} | "
            f"{r['token_nll_std']:>7.3f} | {r['surprisal_std']:>13.3f} | {sign:4} |"
        )

    report.append("")
    report.append("---")
    report.append("")

    # Fisher z-tests
    report.append("## Statistical Comparison (Fisher Z-Tests)")
    report.append("")
    report.append("Testing if correlations differ significantly between variants:")
    report.append("")
    report.append("| Comparison | z-statistic | p-value | Significant? |")
    report.append("|------------|-------------|---------|--------------|")

    for comparison, test_result in fisher_tests.items():
        sig = "✓ YES" if test_result['p'] < 0.05 else "✗ NO"
        report.append(
            f"| {comparison:20} | {test_result['z']:>11.3f} | {test_result['p']:>7.3f} | {sig:12} |"
        )

    report.append("")
    report.append("---")
    report.append("")

    # Interpretation
    report.append("## Interpretation")
    report.append("")

    # Check if anti-coupling persists
    all_negative = all(results[v]['correlation']['pearson_r'] < -0.1 for v in ['original', 'neutral', 'format_b'])
    all_near_zero = all(abs(results[v]['correlation']['pearson_r']) < 0.1 for v in ['original', 'neutral', 'format_b'])

    report.append("**Pattern Across Variants**:")
    if all_negative:
        report.append("- ✅ **ROBUST**: Anti-coupling persists across all variants")
        report.append("- All correlations are negative (r < -0.1)")
        report.append("- Finding is NOT an artifact of misleading labels")
        report.append("- **Recommendation**: PASS - Anti-coupling is a genuine phenomenon")
    elif all_near_zero:
        report.append("- ⚠️ **ARTIFACT**: No correlation in neutral variants")
        report.append("- Original anti-coupling may be due to misleading labels")
        report.append("- **Recommendation**: FAIL - Revise hypothesis")
    else:
        report.append("- ⚠️ **MIXED**: Inconsistent patterns across variants")
        report.append("- Anti-coupling may be context-dependent")
        report.append("- **Recommendation**: INVESTIGATE - Run more trials")

    report.append("")
    report.append("---")
    report.append("")

    # Write report
    report_path = output_dir / 'textualization_robustness_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"\nReport saved to: {report_path}")


def main():
    """Main execution pipeline."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("TEXTUALIZATION ROBUSTNESS EXPERIMENT")
    print("="*80 + "\n")

    # Initialize textualization variants
    textualizations = {
        'original': HotPotTextualization(),
        'neutral': HotPotTextualizationNeutral(),
        'format_b': HotPotTextualizationFormatB(),
    }

    # Initialize predictor
    predictor = create_predictor(model='gpt-4o-mini', provider='openai')

    # Initialize LLM for agent
    llm = OpenAILLM(model='gpt-4o-mini')

    # Storage for results
    results = {}

    # Run episodes for each variant
    for variant_name, textualization in textualizations.items():
        print(f"\n{'='*80}")
        print(f"Running variant: {variant_name.upper()}")
        print(f"{'='*80}\n")

        all_token_nlls = []
        all_surprisals = []

        for seed in SEEDS:
            print(f"  Running episode with seed {seed}...")

            # Create environment
            env = HotPotLab(seed=seed)

            # Create agent with belief state
            agent = ActorAgent(llm, action_budget=N_STEPS, environment_name='HotPotLab')
            agent.set_belief_state(HotPotBelief())

            # Run episode
            test_results, token_logger = run_episode_with_tokens(
                env=env,
                agent=agent,
                textualizer=textualization,
                predictor=predictor,
                seed=seed,
                max_actions=N_STEPS,
                save_dir=str(OUTPUT_DIR / variant_name),
            )

            # Extract metrics
            token_nll, belief_surprisal = extract_metrics_from_logger(token_logger)
            all_token_nlls.extend(token_nll)
            all_surprisals.extend(belief_surprisal)

        # Convert to numpy arrays
        all_token_nlls = np.array(all_token_nlls)
        all_surprisals = np.array(all_surprisals)

        # Compute correlation
        correlation = compute_correlation(all_token_nlls, all_surprisals)

        results[variant_name] = {
            'all_token_nlls': all_token_nlls,
            'all_surprisals': all_surprisals,
            'correlation': correlation,
        }

        print(f"\n  Results for {variant_name}:")
        print(f"    Pearson r = {correlation['pearson_r']:.3f} (p = {correlation['pearson_p']:.3f})")
        print(f"    Spearman ρ = {correlation['spearman_rho']:.3f}")
        print(f"    Token NLL std = {correlation['token_nll_std']:.3f}")
        print(f"    Surprisal std = {correlation['surprisal_std']:.3f}")

    # Perform Fisher z-tests
    print(f"\n{'='*80}")
    print("STATISTICAL COMPARISON (Fisher Z-Tests)")
    print(f"{'='*80}\n")

    fisher_tests = {}

    comparisons = [
        ('original', 'neutral'),
        ('original', 'format_b'),
        ('neutral', 'format_b'),
    ]

    for variant1, variant2 in comparisons:
        r1 = results[variant1]['correlation']['pearson_r']
        n1 = results[variant1]['correlation']['n']
        r2 = results[variant2]['correlation']['pearson_r']
        n2 = results[variant2]['correlation']['n']

        z, p = fisher_z_test(r1, n1, r2, n2)

        comparison_name = f"{variant1} vs {variant2}"
        fisher_tests[comparison_name] = {'z': z, 'p': p}

        sig = "✓ Significant" if p < 0.05 else "✗ Not significant"
        print(f"{comparison_name:20} | z={z:>6.3f}, p={p:.3f} | {sig}")

    # Generate visualizations
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")

    plot_variant_comparison(results, OUTPUT_DIR)

    # Generate report
    print(f"\n{'='*80}")
    print("GENERATING REPORT")
    print(f"{'='*80}\n")

    generate_report(results, fisher_tests, OUTPUT_DIR)

    # Summary
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}\n")
    print(f"Results saved to: {OUTPUT_DIR}")
    print()


if __name__ == '__main__':
    main()
