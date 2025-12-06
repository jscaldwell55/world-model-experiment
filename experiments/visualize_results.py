"""
Visualization for Phase 1 Experiment Results.

Generates learning curves and comparison plots.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plots will be skipped.")


def load_results(filepath: str) -> Dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_learning_curves(results: Dict) -> Dict[str, Dict[int, List[float]]]:
    """Extract learning curves (MAE at each step) for each condition."""
    curves = {}

    for condition, runs in results['results'].items():
        curves[condition] = {}

        for run in runs:
            for step_metric in run['step_metrics']:
                step = step_metric['step']
                mae = step_metric['test_mae']

                if step not in curves[condition]:
                    curves[condition][step] = []
                curves[condition][step].append(mae)

    return curves


def plot_learning_curves(curves: Dict, output_path: str = 'results/phase1/learning_curves.png'):
    """Plot learning curves for all conditions."""
    if not HAS_MATPLOTLIB:
        print("Skipping learning curves plot (matplotlib not available)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'static_random': '#1f77b4',
        'static_uncertainty': '#ff7f0e',
        'online_no_oc': '#2ca02c',
        'full_stack_oc': '#d62728',
        'oracle': '#9467bd'
    }

    labels = {
        'static_random': 'Static + Random',
        'static_uncertainty': 'Static + Uncertainty',
        'online_no_oc': 'Online (no OC)',
        'full_stack_oc': 'Full Stack (OC+FTB)',
        'oracle': 'Oracle'
    }

    for condition, step_data in curves.items():
        if not step_data:
            continue

        steps = sorted(step_data.keys())
        means = [np.mean(step_data[s]) for s in steps]
        stds = [np.std(step_data[s]) for s in steps]

        color = colors.get(condition, '#333333')
        label = labels.get(condition, condition)

        ax.plot(steps, means, color=color, label=label, marker='o', linewidth=2)
        ax.fill_between(steps,
                       [m - s for m, s in zip(means, stds)],
                       [m + s for m, s in zip(means, stds)],
                       color=color, alpha=0.2)

    ax.set_xlabel('Queries', fontsize=12)
    ax.set_ylabel('Test MAE', fontsize=12)
    ax.set_title('Learning Curves: Test MAE vs Queries', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Learning curves saved to {output_path}")


def plot_final_comparison(results: Dict, output_path: str = 'results/phase1/final_comparison.png'):
    """Plot bar chart comparing final MAE across conditions."""
    if not HAS_MATPLOTLIB:
        print("Skipping final comparison plot (matplotlib not available)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = []
    means = []
    stds = []

    labels = {
        'static_random': 'Static\n+Random',
        'static_uncertainty': 'Static\n+Uncertainty',
        'online_no_oc': 'Online\n(no OC)',
        'full_stack_oc': 'Full Stack\n(OC+FTB)',
        'oracle': 'Oracle'
    }

    colors = {
        'static_random': '#1f77b4',
        'static_uncertainty': '#ff7f0e',
        'online_no_oc': '#2ca02c',
        'full_stack_oc': '#d62728',
        'oracle': '#9467bd'
    }

    order = ['static_random', 'static_uncertainty', 'online_no_oc', 'full_stack_oc', 'oracle']

    for condition in order:
        if condition not in results['results']:
            continue

        runs = results['results'][condition]
        maes = [r['final_metrics']['test_mae'] for r in runs]

        conditions.append(labels.get(condition, condition))
        means.append(np.mean(maes))
        stds.append(np.std(maes))

    x = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[colors.get(c, '#333333') for c in order[:len(conditions)]],
                  edgecolor='black', linewidth=1)

    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Final Test MAE', fontsize=12)
    ax.set_title('Final Test MAE Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Final comparison saved to {output_path}")


def plot_calibration(results: Dict, output_path: str = 'results/phase1/calibration.png'):
    """Plot calibration comparison."""
    if not HAS_MATPLOTLIB:
        print("Skipping calibration plot (matplotlib not available)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = {
        'static_random': 'Static+Random',
        'static_uncertainty': 'Static+Uncertainty',
        'online_no_oc': 'Online (no OC)',
        'full_stack_oc': 'Full Stack',
        'oracle': 'Oracle'
    }

    conditions = []
    calibrations = []
    errors = []

    order = ['static_random', 'static_uncertainty', 'online_no_oc', 'full_stack_oc', 'oracle']

    for condition in order:
        if condition not in results['results']:
            continue

        runs = results['results'][condition]
        cals = [r['final_metrics'].get('calibration_corr', 0) for r in runs]

        conditions.append(labels.get(condition, condition))
        calibrations.append(np.mean(cals))
        errors.append(np.std(cals))

    x = np.arange(len(conditions))
    ax.bar(x, calibrations, yerr=errors, capsize=5, color='steelblue', edgecolor='black')

    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Uncertainty-Error Correlation', fontsize=12)
    ax.set_title('Calibration: Uncertainty vs Prediction Error', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Calibration plot saved to {output_path}")


def plot_all_metrics(results: Dict, output_path: str = 'results/phase1/all_metrics.png'):
    """Plot all metrics in subplots."""
    if not HAS_MATPLOTLIB:
        print("Skipping all metrics plot (matplotlib not available)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    labels = {
        'static_random': 'Static+Random',
        'static_uncertainty': 'Static+Unc',
        'online_no_oc': 'Online',
        'full_stack_oc': 'Full Stack',
        'oracle': 'Oracle'
    }

    order = ['static_random', 'static_uncertainty', 'online_no_oc', 'full_stack_oc', 'oracle']

    metrics = [
        ('test_mae', 'MAE (lower is better)'),
        ('test_rmse', 'RMSE (lower is better)'),
        ('test_r2', 'R² (higher is better)'),
        ('calibration_corr', 'Calibration (higher is better)')
    ]

    for ax, (metric_key, metric_label) in zip(axes.flatten(), metrics):
        conditions = []
        values = []
        errors = []

        for condition in order:
            if condition not in results['results']:
                continue

            runs = results['results'][condition]
            vals = [r['final_metrics'].get(metric_key, 0) for r in runs]

            conditions.append(labels.get(condition, condition))
            values.append(np.mean(vals))
            errors.append(np.std(vals))

        x = np.arange(len(conditions))
        ax.bar(x, values, yerr=errors, capsize=5, color='steelblue', edgecolor='black')
        ax.set_xlabel('Condition')
        ax.set_ylabel(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=15)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Phase 1 Experiment: All Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"All metrics plot saved to {output_path}")


def print_text_summary(results: Dict):
    """Print text-based summary for environments without matplotlib."""
    print("\n" + "="*70)
    print("PHASE 1 EXPERIMENT RESULTS (TEXT SUMMARY)")
    print("="*70)

    labels = {
        'static_random': 'Static+Random',
        'static_uncertainty': 'Static+Uncertainty',
        'online_no_oc': 'Online (no OC)',
        'full_stack_oc': 'Full Stack (OC+FTB)',
        'oracle': 'Oracle'
    }

    order = ['static_random', 'static_uncertainty', 'online_no_oc', 'full_stack_oc', 'oracle']

    print(f"\n{'Condition':<25} {'MAE':>15} {'RMSE':>15} {'R²':>15} {'Calibration':>15}")
    print("-"*85)

    for condition in order:
        if condition not in results['results']:
            continue

        runs = results['results'][condition]
        maes = [r['final_metrics']['test_mae'] for r in runs]
        rmses = [r['final_metrics']['test_rmse'] for r in runs]
        r2s = [r['final_metrics']['test_r2'] for r in runs]
        cals = [r['final_metrics'].get('calibration_corr', 0) for r in runs]

        label = labels.get(condition, condition)
        print(f"{label:<25} "
              f"{np.mean(maes):.4f}±{np.std(maes):.4f}  "
              f"{np.mean(rmses):.4f}±{np.std(rmses):.4f}  "
              f"{np.mean(r2s):.4f}±{np.std(r2s):.4f}  "
              f"{np.mean(cals):.4f}±{np.std(cals):.4f}")

    # Learning curve summary
    print("\n" + "-"*70)
    print("LEARNING CURVES (MAE at each checkpoint)")
    print("-"*70)

    curves = extract_learning_curves(results)

    # Get all steps
    all_steps = set()
    for cond_curves in curves.values():
        all_steps.update(cond_curves.keys())
    all_steps = sorted(all_steps)

    header = f"{'Condition':<25}"
    for step in all_steps:
        header += f"{step:>10}"
    print(header)
    print("-"*85)

    for condition in order:
        if condition not in curves:
            continue

        label = labels.get(condition, condition)
        row = f"{label:<25}"

        for step in all_steps:
            if step in curves[condition]:
                mae = np.mean(curves[condition][step])
                row += f"{mae:>10.4f}"
            else:
                row += f"{'N/A':>10}"
        print(row)


def generate_all_plots(results_path: str = 'results/phase1/results.json'):
    """Generate all plots from results file."""
    results = load_results(results_path)

    output_dir = Path(results_path).parent

    # Extract curves
    curves = extract_learning_curves(results)

    # Generate plots
    plot_learning_curves(curves, str(output_dir / 'learning_curves.png'))
    plot_final_comparison(results, str(output_dir / 'final_comparison.png'))
    plot_calibration(results, str(output_dir / 'calibration.png'))
    plot_all_metrics(results, str(output_dir / 'all_metrics.png'))

    # Always print text summary
    print_text_summary(results)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize Phase 1 experiment results')
    parser.add_argument('--results', type=str, default='results/phase1/results.json',
                       help='Path to results JSON file')
    args = parser.parse_args()

    generate_all_plots(args.results)


if __name__ == '__main__':
    main()
