"""
Uncertainty Calibration Check for MolecularWorldModel.

Verifies if the World Model's uncertainty estimates are correlated with
prediction errors before trusting it for synthetic data generation.

Logic:
1. Load ESOL data and train MolecularWorldModel on clean/noisy conditions
2. Collect raw predictions, uncertainties, and true labels on test set
3. Calculate:
   - Pearson/Spearman correlation between uncertainty and |error|
   - Binned MAE analysis (Low/Med/High uncertainty bins)

Decision Criteria:
- If Correlation > 0.3: Proceed to Phase 2 (Dream State)
- If Correlation < 0.3: Implement Platt Scaling or Isotonic Regression

Usage:
    python experiments/check_calibration.py
"""

import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from molecular_world_model import MolecularWorldModel


# =============================================================================
# DATA LOADING
# =============================================================================

def load_esol_data(data_path: str = 'data/esol_processed.pkl') -> Dict:
    """Load processed ESOL data."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Run scripts/esol_01_process_data.py first."
        )

    with open(data_path, 'rb') as f:
        return pickle.load(f)


def prepare_data_with_condition(
    condition: str,
    seed: int = 42,
    noise_level: float = 0.15
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """
    Prepare train/test data with specified condition.

    Args:
        condition: 'clean' or 'noisy_15pct'
        seed: Random seed
        noise_level: Noise std multiplier for noisy condition

    Returns:
        train_smiles, train_labels, test_smiles, test_labels
    """
    rng = np.random.RandomState(seed)
    data = load_esol_data()

    # Combine candidate and test sets
    candidate_df = data['candidate_pool']
    test_df = data['test_set']

    all_smiles = candidate_df['smiles'].tolist() + test_df['smiles'].tolist()
    all_labels = np.array(candidate_df['logS'].tolist() + test_df['logS'].tolist())

    # Shuffle
    indices = np.arange(len(all_smiles))
    rng.shuffle(indices)
    all_smiles = [all_smiles[i] for i in indices]
    all_labels = all_labels[indices]

    # Apply noise if needed
    if condition == 'noisy_15pct':
        label_std = np.std(all_labels)
        noise = rng.normal(0, label_std * noise_level, size=len(all_labels))
        all_labels = all_labels + noise

    # Split: 80% train, 20% test
    split_idx = int(0.8 * len(all_smiles))

    train_smiles = all_smiles[:split_idx]
    train_labels = all_labels[:split_idx]
    test_smiles = all_smiles[split_idx:]
    test_labels = all_labels[split_idx:]

    return train_smiles, train_labels, test_smiles, test_labels


# =============================================================================
# CALIBRATION ANALYSIS
# =============================================================================

def collect_predictions(
    model: MolecularWorldModel,
    smiles_list: List[str],
    true_labels: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Collect predictions, uncertainties, and compute errors.

    Returns dict with:
        - predictions: Model predictions
        - uncertainties: Uncertainty estimates (RF tree std)
        - true_labels: Ground truth
        - errors: Absolute errors |pred - true|
    """
    predictions, uncertainties = model.predict(smiles_list, return_uncertainty=True)

    # Handle any NaN values (invalid molecules)
    valid_mask = ~np.isnan(predictions)

    predictions = predictions[valid_mask]
    uncertainties = uncertainties[valid_mask]
    true_labels = true_labels[valid_mask]
    errors = np.abs(predictions - true_labels)

    return {
        'predictions': predictions,
        'uncertainties': uncertainties,
        'true_labels': true_labels,
        'errors': errors,
        'n_valid': int(np.sum(valid_mask)),
        'n_total': len(smiles_list)
    }


def compute_correlation_metrics(
    uncertainties: np.ndarray,
    errors: np.ndarray
) -> Dict[str, float]:
    """
    Compute Pearson and Spearman correlation between uncertainty and error.

    Returns:
        pearson_r: Pearson correlation coefficient
        pearson_p: Pearson p-value
        spearman_r: Spearman rank correlation
        spearman_p: Spearman p-value
    """
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(uncertainties, errors)

    # Spearman rank correlation (more robust to non-linear relationships)
    spearman_r, spearman_p = stats.spearmanr(uncertainties, errors)

    return {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p)
    }


def compute_binned_mae(
    uncertainties: np.ndarray,
    errors: np.ndarray,
    n_bins: int = 3
) -> List[Dict]:
    """
    Divide data into uncertainty bins and compute MAE for each.

    Args:
        uncertainties: Uncertainty estimates
        errors: Absolute errors
        n_bins: Number of bins (default 3: Low, Med, High)

    Returns:
        List of dicts with bin statistics
    """
    # Use quantile-based binning for equal sample sizes
    bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))

    bin_labels = ['Low', 'Med', 'High'] if n_bins == 3 else [f'Bin_{i}' for i in range(n_bins)]

    bin_results = []

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i+1])
        else:
            # Last bin includes upper bound
            mask = (uncertainties >= bin_edges[i]) & (uncertainties <= bin_edges[i+1])

        bin_uncerts = uncertainties[mask]
        bin_errors = errors[mask]

        bin_results.append({
            'bin': bin_labels[i],
            'n_samples': int(np.sum(mask)),
            'uncertainty_range': (float(bin_edges[i]), float(bin_edges[i+1])),
            'avg_uncertainty': float(np.mean(bin_uncerts)) if len(bin_uncerts) > 0 else 0,
            'mae': float(np.mean(bin_errors)) if len(bin_errors) > 0 else 0,
            'std_error': float(np.std(bin_errors)) if len(bin_errors) > 0 else 0
        })

    return bin_results


def print_calibration_report(
    condition: str,
    correlation_metrics: Dict[str, float],
    binned_mae: List[Dict],
    data_stats: Dict
) -> None:
    """Print formatted calibration analysis report."""

    print(f"\n{'='*70}")
    print(f"CALIBRATION CHECK: {condition.upper()} CONDITION")
    print(f"{'='*70}")

    print(f"\nData Statistics:")
    print(f"  Valid samples: {data_stats['n_valid']}/{data_stats['n_total']}")
    print(f"  Overall MAE: {np.mean(data_stats['errors']):.4f}")
    print(f"  Mean Uncertainty: {np.mean(data_stats['uncertainties']):.4f}")

    print(f"\n{'─'*70}")
    print("CORRELATION ANALYSIS")
    print(f"{'─'*70}")
    print(f"  Pearson r:  {correlation_metrics['pearson_r']:+.4f} (p={correlation_metrics['pearson_p']:.2e})")
    print(f"  Spearman r: {correlation_metrics['spearman_r']:+.4f} (p={correlation_metrics['spearman_p']:.2e})")

    print(f"\n{'─'*70}")
    print("UNCERTAINTY BIN ANALYSIS")
    print(f"{'─'*70}")
    print(f"{'Bin':<8} {'N':<8} {'Avg Uncertainty':<18} {'Actual MAE':<15} {'Std':<10}")
    print(f"{'-'*70}")

    for bin_data in binned_mae:
        print(f"{bin_data['bin']:<8} "
              f"{bin_data['n_samples']:<8} "
              f"{bin_data['avg_uncertainty']:<18.4f} "
              f"{bin_data['mae']:<15.4f} "
              f"{bin_data['std_error']:<10.4f}")

    # Check if high uncertainty bin has higher MAE (expected behavior)
    if len(binned_mae) >= 2:
        low_mae = binned_mae[0]['mae']
        high_mae = binned_mae[-1]['mae']
        mae_ratio = high_mae / low_mae if low_mae > 0 else float('inf')

        print(f"\nHigh/Low MAE Ratio: {mae_ratio:.2f}x")
        if mae_ratio > 1.5:
            print("  -> Uncertainty correctly identifies harder predictions")
        elif mae_ratio > 1.0:
            print("  -> Weak separation between uncertainty bins")
        else:
            print("  -> WARNING: Uncertainty may be miscalibrated")


def make_calibration_decision(
    correlation_metrics: Dict[str, float],
    binned_mae: List[Dict]
) -> Dict:
    """
    Make go/no-go decision for Phase 2.

    Criteria:
    - Spearman r > 0.3: Proceed to Phase 2
    - Spearman r < 0.3: Need calibration (Platt/Isotonic)

    Returns:
        decision: Dict with verdict and reasoning
    """
    spearman_r = correlation_metrics['spearman_r']

    # Check bin monotonicity
    mae_values = [b['mae'] for b in binned_mae]
    is_monotonic = all(mae_values[i] <= mae_values[i+1] for i in range(len(mae_values)-1))

    # High/Low ratio
    if len(binned_mae) >= 2 and binned_mae[0]['mae'] > 0:
        mae_ratio = binned_mae[-1]['mae'] / binned_mae[0]['mae']
    else:
        mae_ratio = 1.0

    # Decision logic
    if spearman_r >= 0.3 and is_monotonic and mae_ratio >= 1.2:
        verdict = 'PROCEED'
        confidence = 'HIGH'
        reason = (f"Strong uncertainty-error correlation (r={spearman_r:.3f}), "
                  f"monotonic bin MAE, good separation ({mae_ratio:.2f}x)")
    elif spearman_r >= 0.3:
        verdict = 'PROCEED'
        confidence = 'MEDIUM'
        reason = (f"Adequate correlation (r={spearman_r:.3f}), "
                  f"but bin separation could be stronger")
    elif spearman_r >= 0.2:
        verdict = 'CAUTION'
        confidence = 'LOW'
        reason = (f"Weak correlation (r={spearman_r:.3f}). "
                  f"Consider Platt scaling for better calibration.")
    else:
        verdict = 'CALIBRATE'
        confidence = 'N/A'
        reason = (f"Poor correlation (r={spearman_r:.3f}). "
                  f"MUST implement Platt Scaling or Isotonic Regression before Phase 2.")

    return {
        'verdict': verdict,
        'confidence': confidence,
        'reason': reason,
        'spearman_r': spearman_r,
        'is_monotonic': is_monotonic,
        'mae_ratio': mae_ratio
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_calibration_check(
    conditions: List[str] = ['clean', 'noisy_15pct'],
    seeds: List[int] = [42, 123, 456],
    save_results: bool = True
) -> Dict:
    """
    Run full calibration analysis across conditions and seeds.

    Args:
        conditions: Data conditions to test
        seeds: Random seeds for multiple trials
        save_results: Whether to save results to JSON

    Returns:
        Full results dictionary
    """
    print("\n" + "="*70)
    print("WORLD MODEL UNCERTAINTY CALIBRATION CHECK")
    print("="*70)
    print(f"Testing conditions: {conditions}")
    print(f"Seeds: {seeds}")

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'conditions': conditions,
        'seeds': seeds,
        'per_condition': {},
        'aggregated': {}
    }

    for condition in conditions:
        condition_results = []

        for seed in seeds:
            print(f"\n[{condition}] Seed {seed}...")

            # Prepare data
            train_smiles, train_labels, test_smiles, test_labels = \
                prepare_data_with_condition(condition, seed=seed)

            # Train model
            model = MolecularWorldModel(
                n_scaffold_clusters=20,
                n_estimators=100,
                random_state=seed
            )
            model.fit(train_smiles, train_labels)

            # Collect predictions
            data_stats = collect_predictions(model, test_smiles, test_labels)

            # Compute metrics
            correlation = compute_correlation_metrics(
                data_stats['uncertainties'],
                data_stats['errors']
            )

            binned = compute_binned_mae(
                data_stats['uncertainties'],
                data_stats['errors'],
                n_bins=3
            )

            # Store results
            condition_results.append({
                'seed': seed,
                'n_train': len(train_smiles),
                'n_test': data_stats['n_valid'],
                'overall_mae': float(np.mean(data_stats['errors'])),
                'mean_uncertainty': float(np.mean(data_stats['uncertainties'])),
                'correlation': correlation,
                'binned_mae': binned
            })

        # Aggregate across seeds
        all_spearman = [r['correlation']['spearman_r'] for r in condition_results]
        all_pearson = [r['correlation']['pearson_r'] for r in condition_results]
        all_mae = [r['overall_mae'] for r in condition_results]

        # Average binned results
        avg_binned = []
        for bin_idx in range(3):
            bin_maes = [r['binned_mae'][bin_idx]['mae'] for r in condition_results]
            bin_uncerts = [r['binned_mae'][bin_idx]['avg_uncertainty'] for r in condition_results]
            bin_stds = [r['binned_mae'][bin_idx]['std_error'] for r in condition_results]
            avg_binned.append({
                'bin': condition_results[0]['binned_mae'][bin_idx]['bin'],
                'n_samples': sum(r['binned_mae'][bin_idx]['n_samples'] for r in condition_results),
                'avg_uncertainty': float(np.mean(bin_uncerts)),
                'mae': float(np.mean(bin_maes)),
                'mae_std': float(np.std(bin_maes)),
                'std_error': float(np.mean(bin_stds))  # Average std across seeds
            })

        aggregated = {
            'mean_spearman_r': float(np.mean(all_spearman)),
            'std_spearman_r': float(np.std(all_spearman)),
            'mean_pearson_r': float(np.mean(all_pearson)),
            'std_pearson_r': float(np.std(all_pearson)),
            'mean_mae': float(np.mean(all_mae)),
            'std_mae': float(np.std(all_mae)),
            'binned_mae': avg_binned
        }

        # Print report
        print_calibration_report(
            condition,
            {'spearman_r': aggregated['mean_spearman_r'],
             'spearman_p': 0.0,  # Aggregated, no single p-value
             'pearson_r': aggregated['mean_pearson_r'],
             'pearson_p': 0.0},
            avg_binned,
            {'n_valid': sum(r['n_test'] for r in condition_results),
             'n_total': sum(r['n_test'] for r in condition_results),
             'errors': np.concatenate([np.array([r['overall_mae']]) for r in condition_results]),
             'uncertainties': np.concatenate([np.array([r['mean_uncertainty']]) for r in condition_results])}
        )

        # Make decision
        decision = make_calibration_decision(
            {'spearman_r': aggregated['mean_spearman_r'],
             'pearson_r': aggregated['mean_pearson_r'],
             'pearson_p': 0.0, 'spearman_p': 0.0},
            avg_binned
        )

        all_results['per_condition'][condition] = {
            'per_seed': condition_results,
            'aggregated': aggregated,
            'decision': decision
        }

    # Overall decision (across conditions)
    all_spearman = [
        all_results['per_condition'][c]['aggregated']['mean_spearman_r']
        for c in conditions
    ]

    overall_spearman = np.mean(all_spearman)

    if overall_spearman >= 0.3:
        overall_verdict = 'PROCEED TO PHASE 2'
        action = 'Uncertainty estimates are reliable. Safe to generate synthetic data.'
    elif overall_spearman >= 0.2:
        overall_verdict = 'PROCEED WITH CAUTION'
        action = 'Consider implementing Platt Scaling for improved calibration.'
    else:
        overall_verdict = 'CALIBRATION REQUIRED'
        action = 'MUST implement Platt Scaling or Isotonic Regression before Phase 2.'

    all_results['aggregated'] = {
        'mean_spearman_r': float(overall_spearman),
        'verdict': overall_verdict,
        'action': action
    }

    # Print final verdict
    print(f"\n{'='*70}")
    print("FINAL VERDICT")
    print(f"{'='*70}")
    print(f"Overall Spearman r: {overall_spearman:.4f}")
    print(f"Decision: {overall_verdict}")
    print(f"Action: {action}")
    print(f"{'='*70}\n")

    # Save results
    if save_results:
        results_path = Path('results/calibration_check.json')
        results_path.parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"Results saved to: {results_path}")

    return all_results


if __name__ == '__main__':
    results = run_calibration_check(
        conditions=['clean', 'noisy_15pct'],
        seeds=[42, 123, 456],
        save_results=True
    )
