"""
Identify systematic biases in ESOL baseline predictions.

Analyzes prediction errors to find molecular features that correlate
with high errors. These biased regions are targets for synthetic data.

Example biases:
- High MW molecules → underpredicted solubility
- High LogP molecules → overpredicted solubility
- Many H-bond donors → high error variance

Strategy: Generate synthetics in these biased regions to fix weaknesses.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.bias_analysis import BiasAnalyzer
import pandas as pd
import json


def main():
    """Run bias analysis on ESOL baseline predictions."""

    print("=" * 70)
    print("ESOL BIAS ANALYSIS")
    print("=" * 70)

    # Load baseline predictions
    baseline_df = pd.read_csv('memory/esol_baseline_predictions.csv')

    print(f"\nLoaded {len(baseline_df)} test predictions")
    print(f"Overall MAE: {baseline_df['error'].mean():.3f}")

    # Run bias analysis
    print(f"\nAnalyzing biases...")
    analyzer = BiasAnalyzer(threshold_percentile=75)
    biases = analyzer.analyze_biases(
        baseline_df,
        min_count=15,  # Need at least 15 molecules to flag bias
        min_error_increase=0.05  # Error must increase by 0.05+ to flag
    )

    # Visualize
    analyzer.visualize_biases(baseline_df, 'memory/esol_bias_visualization.png')

    # Save detailed bias info
    bias_summary = {
        'n_biases_found': len(biases),
        'analysis_settings': {
            'threshold_percentile': 75,
            'min_count': 15,
            'min_error_increase': 0.05
        },
        'biases': {}
    }

    for feature, info in biases.items():
        bias_summary['biases'][feature] = {
            'threshold': float(info['threshold']),
            'error_increase': float(info['error_increase']),
            'high_error': float(info['high_error']),
            'low_error': float(info['low_error']),
            'bias_direction': float(info['bias_direction']),
            'count_high': int(info['count_high']),
            'count_low': int(info['count_low']),
            'interpretation': _interpret_bias(feature, info)
        }

    with open('memory/esol_bias_analysis.json', 'w') as f:
        json.dump(bias_summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print("BIAS ANALYSIS RESULTS")
    print(f"{'=' * 70}")

    if len(biases) > 0:
        print(f"\n✅ Found {len(biases)} systematic biases:")

        for feature, info in biases.items():
            print(f"\n  {feature}:")
            print(f"    Threshold: {info['threshold']:.2f}")
            print(f"    Error increase: {info['error_increase']:.3f}")
            print(f"    High-value MAE: {info['high_error']:.3f}")
            print(f"    Low-value MAE: {info['low_error']:.3f}")
            print(f"    Bias direction: {'over' if info['bias_direction'] > 0 else 'under'}prediction")
            print(f"    Interpretation: {bias_summary['biases'][feature]['interpretation']}")

        print(f"\n🎯 SYNTHETIC GENERATION STRATEGY:")
        print(f"   Target molecules with:")
        for feature in biases.keys():
            print(f"     - High {feature} (>{biases[feature]['threshold']:.1f})")

        print(f"\n   This will fix model weaknesses in these regions")

    else:
        print("\n⚠️  No major systematic biases found")
        print("   Model performs uniformly across feature space")
        print("   Standard augmentation (uniform sampling) will be used")

    print(f"\n{'=' * 70}")
    print("OUTPUT FILES")
    print(f"{'=' * 70}")
    print(f"  Bias data: memory/esol_bias_analysis.json")
    print(f"  Visualization: memory/esol_bias_visualization.png")

    print(f"\n{'=' * 70}")
    print("✅ Bias analysis complete!")
    print(f"{'=' * 70}")

    return biases


def _interpret_bias(feature: str, info: dict) -> str:
    """Generate human-readable interpretation of bias."""

    direction = "overpredicts" if info['bias_direction'] > 0 else "underpredicts"

    interpretation = (f"Model {direction} solubility for molecules with "
                     f"high {feature} (>{info['threshold']:.2f}). "
                     f"Error increases by {info['error_increase']:.3f} ({info['error_increase']/info['low_error']*100:.0f}%).")

    return interpretation


if __name__ == "__main__":
    main()
