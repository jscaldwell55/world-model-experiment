#!/usr/bin/env python3
"""
Statistical power analysis for SwitchLight actor vs observer comparison.

Analyzes whether the observed difference in performance is statistically
significant and computes the required sample size for desired power.
"""

import numpy as np
from scipy import stats
from statsmodels.stats.power import ttest_power
import argparse
import json
from pathlib import Path


def load_switchlight_results(results_dir: Path):
    """
    Load SwitchLight results from aggregated summary.

    Args:
        results_dir: Path to aggregated results directory

    Returns:
        Tuple of (actor_scores, observer_scores)
    """
    summary_file = results_dir / 'summary.csv'

    if not summary_file.exists():
        print(f"Warning: {summary_file} not found")
        return None, None

    # Parse summary CSV
    import csv
    actor_scores = []
    observer_scores = []

    with open(summary_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['environment'] == 'SwitchLight':
                if row['agent_type'] == 'actor':
                    actor_scores.append(float(row['mean_score']))
                elif row['agent_type'] == 'observer':
                    observer_scores.append(float(row['mean_score']))

    return actor_scores, observer_scores


def analyze_power(actor_scores, observer_scores, alpha=0.05, desired_power=0.8):
    """
    Perform statistical power analysis.

    Args:
        actor_scores: List of actor accuracy scores
        observer_scores: List of observer accuracy scores
        alpha: Significance level (default: 0.05)
        desired_power: Desired statistical power (default: 0.8)

    Returns:
        Dictionary with analysis results
    """
    # Convert to numpy arrays
    actor_scores = np.array(actor_scores)
    observer_scores = np.array(observer_scores)

    # Compute summary statistics
    actor_mean = np.mean(actor_scores)
    actor_std = np.std(actor_scores, ddof=1)
    observer_mean = np.mean(observer_scores)
    observer_std = np.std(observer_scores, ddof=1)
    n = len(actor_scores)

    # Difference
    diff = actor_mean - observer_mean

    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(actor_scores, observer_scores)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((actor_std**2 + observer_std**2) / 2)
    cohens_d = diff / pooled_std if pooled_std > 0 else 0.0

    # Current statistical power
    try:
        current_power = ttest_power(
            effect_size=abs(cohens_d),
            nobs=n,
            alpha=alpha,
            alternative='two-sided'
        )
    except Exception as e:
        print(f"Warning: Could not compute power: {e}")
        current_power = 0.0

    # Required sample size for desired power
    required_n = n
    if abs(cohens_d) > 0.01:  # Only compute if effect size is non-trivial
        try:
            # Binary search for required n
            for test_n in range(5, 1000):
                power = ttest_power(
                    effect_size=abs(cohens_d),
                    nobs=test_n,
                    alpha=alpha,
                    alternative='two-sided'
                )
                if power >= desired_power:
                    required_n = test_n
                    break
        except Exception as e:
            print(f"Warning: Could not compute required n: {e}")
            required_n = 999  # Large number indicating "impractical"

    # Confidence interval for difference
    se_diff = np.sqrt(actor_std**2 / n + observer_std**2 / n)
    ci_lower = diff - 1.96 * se_diff
    ci_upper = diff + 1.96 * se_diff

    return {
        'n': n,
        'actor_mean': actor_mean,
        'actor_std': actor_std,
        'observer_mean': observer_mean,
        'observer_std': observer_std,
        'difference': diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'current_power': current_power,
        'required_n': required_n,
        'alpha': alpha,
        'desired_power': desired_power
    }


def print_report(results):
    """Print formatted power analysis report."""
    print("=" * 70)
    print("SWITCHLIGHT STATISTICAL POWER ANALYSIS")
    print("=" * 70)
    print()

    print("SAMPLE STATISTICS")
    print("-" * 70)
    print(f"Sample size (n):              {results['n']}")
    print(f"Actor mean:                   {results['actor_mean']:.3f} ± {results['actor_std']:.3f}")
    print(f"Observer mean:                {results['observer_mean']:.3f} ± {results['observer_std']:.3f}")
    print(f"Difference (Actor - Observer): {results['difference']:.3f}")
    print(f"95% CI for difference:        [{results['ci_lower']:.3f}, {results['ci_upper']:.3f}]")
    print()

    print("STATISTICAL TESTS")
    print("-" * 70)
    print(f"t-statistic:                  {results['t_statistic']:.3f}")
    print(f"p-value (two-sided):          {results['p_value']:.4f}")

    if results['p_value'] < 0.001:
        sig_str = "*** (highly significant)"
    elif results['p_value'] < 0.01:
        sig_str = "** (very significant)"
    elif results['p_value'] < 0.05:
        sig_str = "* (significant)"
    elif results['p_value'] < 0.10:
        sig_str = "† (marginally significant)"
    else:
        sig_str = "ns (not significant)"

    print(f"Significance:                 {sig_str}")
    print()

    print("EFFECT SIZE")
    print("-" * 70)
    print(f"Cohen's d:                    {results['cohens_d']:.3f}")

    if abs(results['cohens_d']) < 0.2:
        effect_str = "negligible"
    elif abs(results['cohens_d']) < 0.5:
        effect_str = "small"
    elif abs(results['cohens_d']) < 0.8:
        effect_str = "medium"
    else:
        effect_str = "large"

    print(f"Effect size interpretation:   {effect_str}")
    print()

    print("POWER ANALYSIS")
    print("-" * 70)
    print(f"Current power (n={results['n']}):         {results['current_power']:.3f}")
    print(f"Desired power:                {results['desired_power']:.2f}")
    print(f"Required n for {results['desired_power']:.0%} power:       {results['required_n']}")
    print()

    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print()

    if results['p_value'] > 0.3:
        print("✗ NOT STATISTICALLY SIGNIFICANT")
        print()
        print("The observed difference is not statistically significant (p > 0.30).")
        print("The effect size is negligible (Cohen's d ≈ 0).")
        print()
        print("RECOMMENDATION:")
        print("  Accept that Actor and Observer perform equivalently on SwitchLight.")
        print("  No clear winner - both strategies work equally well.")
        print()
        if results['required_n'] > 100:
            print(f"  To detect this small effect would require n={results['required_n']} episodes,")
            print("  which is impractical for such a small effect size.")

    elif results['p_value'] > 0.1:
        print("⚠ BORDERLINE SIGNIFICANCE")
        print()
        print(f"The observed difference is not quite significant (p = {results['p_value']:.3f}).")
        print(f"Effect size is {effect_str} (Cohen's d = {results['cohens_d']:.3f}).")
        print()
        print("RECOMMENDATION:")
        if results['required_n'] < 20:
            print(f"  Consider running {results['required_n']} episodes total to confirm.")
            print("  This would provide 80% power to detect the observed effect.")
        else:
            print(f"  Required sample size (n={results['required_n']}) may be too large.")
            print("  Consider accepting 'no clear winner' or running a smaller study.")

    else:
        print("✓ POTENTIALLY SIGNIFICANT")
        print()
        print(f"The observed difference is statistically significant (p = {results['p_value']:.3f}).")
        print(f"Effect size is {effect_str} (Cohen's d = {results['cohens_d']:.3f}).")
        print()
        print("RECOMMENDATION:")
        if results['current_power'] < 0.8:
            print(f"  Current power is low ({results['current_power']:.2f}).")
            print(f"  Run {results['required_n']} episodes total to confirm with 80% power.")
        else:
            print("  Result is well-powered. Difference is likely real.")
            if results['difference'] > 0:
                print("  Actor outperforms Observer on SwitchLight.")
            else:
                print("  Observer outperforms Actor on SwitchLight.")

    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Statistical power analysis for SwitchLight results'
    )
    parser.add_argument(
        '--results',
        type=Path,
        help='Path to aggregated results directory (or will use default data)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level (default: 0.05)'
    )
    parser.add_argument(
        '--power',
        type=float,
        default=0.8,
        help='Desired statistical power (default: 0.8)'
    )

    args = parser.parse_args()

    # Load results if provided
    actor_scores = None
    observer_scores = None

    if args.results:
        actor_scores, observer_scores = load_switchlight_results(args.results)

    # Use default data if not found or not provided
    if actor_scores is None or observer_scores is None:
        print("Using default SwitchLight data from user's report:")
        print("  Actor:    [0.78, 0.68, 0.68, 0.692, 0.4]")
        print("  Observer: [0.5, 0.6, 0.9, 0.66, 0.68]")
        print()

        actor_scores = [0.78, 0.68, 0.68, 0.692, 0.4]
        observer_scores = [0.5, 0.6, 0.9, 0.66, 0.68]

    # Perform power analysis
    results = analyze_power(
        actor_scores,
        observer_scores,
        alpha=args.alpha,
        desired_power=args.power
    )

    # Print report
    print_report(results)


if __name__ == '__main__':
    main()
