"""
Model Recalibration: Fix uncertainty estimates via isotonic regression.

The current model has poor calibration (ECE ~0.28-0.32) which means
uncertainty estimates are unreliable. This module implements:

1. Expected Calibration Error (ECE) measurement
2. Reliability diagrams (calibration plots)
3. Multiple calibration methods:
   - Isotonic regression (non-parametric)
   - Platt scaling (logistic regression)
   - Temperature scaling (single parameter)
4. Integration with MolecularWorldModel

Success criteria:
- ECE < 0.1 after recalibration (down from ~0.3)
- Calibration slope near 1.0 on reliability diagram
- Sharpness maintained (std of uncertainties > 0.1)
"""

import json
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from molecular_world_model import MolecularWorldModel


@dataclass
class CalibrationMetrics:
    """Container for calibration metrics."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    avg_confidence: float  # Average predicted confidence
    avg_accuracy: float  # Average actual accuracy
    calibration_slope: float  # Slope of calibration curve
    sharpness: float  # Std of confidence predictions
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]

    def to_dict(self) -> Dict:
        return {
            'ece': float(self.ece),
            'mce': float(self.mce),
            'avg_confidence': float(self.avg_confidence),
            'avg_accuracy': float(self.avg_accuracy),
            'calibration_slope': float(self.calibration_slope),
            'sharpness': float(self.sharpness),
            'bin_accuracies': [float(x) for x in self.bin_accuracies],
            'bin_confidences': [float(x) for x in self.bin_confidences],
            'bin_counts': [int(x) for x in self.bin_counts]
        }


def compute_expected_calibration_error(
    uncertainties: np.ndarray,
    errors: np.ndarray,
    n_bins: int = 10
) -> CalibrationMetrics:
    """
    Compute Expected Calibration Error and related metrics.

    For regression with uncertainty, we check if:
    - High uncertainty predictions have high errors
    - Low uncertainty predictions have low errors

    Ideal: mean_error_in_bin ≈ mean_uncertainty_in_bin

    Args:
        uncertainties: Predicted uncertainty values
        errors: Absolute prediction errors
        n_bins: Number of bins for calibration

    Returns:
        CalibrationMetrics with ECE, MCE, and per-bin stats
    """
    # Normalize errors and uncertainties to [0, 1] for comparable binning
    # We use quantile-based binning on uncertainties

    # Handle edge cases
    if len(uncertainties) == 0:
        return CalibrationMetrics(
            ece=np.nan, mce=np.nan, avg_confidence=0, avg_accuracy=0,
            calibration_slope=0, sharpness=0,
            bin_accuracies=[], bin_confidences=[], bin_counts=[]
        )

    # Compute bins using quantiles of uncertainty
    try:
        bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
        # Ensure unique edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            bin_edges = np.array([uncertainties.min(), uncertainties.max()])
    except Exception:
        bin_edges = np.linspace(uncertainties.min(), uncertainties.max(), n_bins + 1)

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    ece = 0.0
    mce = 0.0
    n_total = len(uncertainties)

    for i in range(len(bin_edges) - 1):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1])
        if i == len(bin_edges) - 2:  # Include right edge for last bin
            mask = (uncertainties >= bin_edges[i]) & (uncertainties <= bin_edges[i + 1])

        n_in_bin = np.sum(mask)
        if n_in_bin == 0:
            continue

        # Mean uncertainty in bin (predicted "error")
        mean_uncertainty = np.mean(uncertainties[mask])

        # Mean actual error in bin
        mean_error = np.mean(errors[mask])

        bin_confidences.append(mean_uncertainty)
        bin_accuracies.append(mean_error)
        bin_counts.append(n_in_bin)

        # Calibration error for this bin
        bin_error = np.abs(mean_error - mean_uncertainty)
        ece += bin_error * (n_in_bin / n_total)
        mce = max(mce, bin_error)

    # Compute calibration slope (how well uncertainty predicts error)
    if len(bin_confidences) > 1:
        slope, _, _, _, _ = stats.linregress(bin_confidences, bin_accuracies)
    else:
        slope = 0.0

    # Sharpness: spread of uncertainty predictions
    sharpness = np.std(uncertainties)

    return CalibrationMetrics(
        ece=ece,
        mce=mce,
        avg_confidence=np.mean(uncertainties),
        avg_accuracy=np.mean(errors),
        calibration_slope=slope,
        sharpness=sharpness,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts
    )


def plot_reliability_diagram(
    metrics: CalibrationMetrics,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot reliability diagram showing calibration quality.

    Perfect calibration = points on y=x diagonal.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Reliability diagram
    bin_confs = np.array(metrics.bin_confidences)
    bin_accs = np.array(metrics.bin_accuracies)
    bin_counts = np.array(metrics.bin_counts)

    if len(bin_confs) > 0:
        # Scatter points sized by count
        sizes = 100 * (bin_counts / max(bin_counts))
        ax1.scatter(bin_confs, bin_accs, s=sizes, alpha=0.7, c='steelblue',
                   edgecolors='black', linewidth=1)

        # Perfect calibration line
        max_val = max(bin_confs.max(), bin_accs.max()) * 1.1
        ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect calibration')

        # Fit line
        if len(bin_confs) > 1:
            slope, intercept, _, _, _ = stats.linregress(bin_confs, bin_accs)
            x_fit = np.linspace(0, max_val, 100)
            ax1.plot(x_fit, slope * x_fit + intercept, 'r-', alpha=0.7,
                    label=f'Fit (slope={slope:.2f})')

    ax1.set_xlabel('Mean Predicted Uncertainty')
    ax1.set_ylabel('Mean Actual Error')
    ax1.set_title(f'{title}\nECE={metrics.ece:.4f}, Slope={metrics.calibration_slope:.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Histogram of uncertainties (sharpness)
    ax2.bar(range(len(bin_counts)), bin_counts, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Uncertainty Bin')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Uncertainty Distribution\nSharpness={metrics.sharpness:.4f}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved reliability diagram to {save_path}")

    return fig


class IsotonicCalibrator:
    """
    Isotonic regression calibrator for uncertainty estimates.

    Maps raw uncertainties to calibrated uncertainties such that
    the predicted uncertainty matches the actual error distribution.
    """

    def __init__(self):
        self.isotonic = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False

    def fit(self, uncertainties: np.ndarray, errors: np.ndarray) -> 'IsotonicCalibrator':
        """
        Fit calibrator on validation data.

        Args:
            uncertainties: Raw uncertainty predictions
            errors: Actual absolute errors
        """
        # Isotonic regression: uncertainty -> expected error
        self.isotonic.fit(uncertainties, errors)
        self.is_fitted = True
        return self

    def calibrate(self, uncertainties: np.ndarray) -> np.ndarray:
        """Transform raw uncertainties to calibrated uncertainties."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        return self.isotonic.predict(uncertainties)

    def save(self, filepath: str):
        """Save calibrator to pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'IsotonicCalibrator':
        """Load calibrator from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class TemperatureScaler:
    """
    Temperature scaling for uncertainty calibration.

    Scales uncertainty by a single learned temperature parameter:
    calibrated_uncertainty = raw_uncertainty * temperature
    """

    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False

    def fit(self, uncertainties: np.ndarray, errors: np.ndarray) -> 'TemperatureScaler':
        """
        Find optimal temperature to minimize ECE.
        """
        def objective(temp):
            scaled = uncertainties * temp
            metrics = compute_expected_calibration_error(scaled, errors)
            return metrics.ece

        result = minimize_scalar(objective, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        self.is_fitted = True
        return self

    def calibrate(self, uncertainties: np.ndarray) -> np.ndarray:
        """Apply temperature scaling."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted")
        return uncertainties * self.temperature


class PlattScaler:
    """
    Platt scaling using logistic regression.

    For regression uncertainty, we convert to a classification problem:
    - "Accurate" if error < threshold
    - Map uncertainty to probability of being accurate
    """

    def __init__(self, error_threshold: float = 0.5):
        self.error_threshold = error_threshold
        self.scaler = LogisticRegression()
        self.is_fitted = False

    def fit(self, uncertainties: np.ndarray, errors: np.ndarray) -> 'PlattScaler':
        """Fit Platt scaling."""
        # Convert to binary: 1 if error > threshold (high uncertainty warranted)
        labels = (errors > self.error_threshold).astype(int)

        # Reshape for sklearn
        X = uncertainties.reshape(-1, 1)

        self.scaler.fit(X, labels)
        self.is_fitted = True
        return self

    def calibrate(self, uncertainties: np.ndarray) -> np.ndarray:
        """
        Return probability of high error (as calibrated uncertainty).

        Higher probability = higher calibrated uncertainty.
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted")
        X = uncertainties.reshape(-1, 1)
        probs = self.scaler.predict_proba(X)[:, 1]
        # Scale to match error magnitude
        return probs * self.error_threshold * 2


def compare_calibration_methods(
    train_smiles: List[str],
    train_labels: np.ndarray,
    calib_smiles: List[str],
    calib_labels: np.ndarray,
    test_smiles: List[str],
    test_labels: np.ndarray,
    random_state: int = 42
) -> Dict:
    """
    Compare multiple calibration methods.

    Args:
        train_smiles/labels: Data for training the world model
        calib_smiles/labels: Data for fitting calibrators
        test_smiles/labels: Data for evaluation

    Returns:
        Dict with metrics for each method
    """
    results = {}

    # Train world model
    print("Training world model...")
    model = MolecularWorldModel(n_estimators=100, random_state=random_state)
    model.fit(train_smiles, train_labels)

    # Get calibration set predictions
    calib_preds, calib_uncerts = model.predict(calib_smiles, return_uncertainty=True)
    valid_mask = ~np.isnan(calib_preds)
    calib_preds = calib_preds[valid_mask]
    calib_uncerts = calib_uncerts[valid_mask]
    calib_labels_valid = calib_labels[valid_mask]
    calib_errors = np.abs(calib_preds - calib_labels_valid)

    # Get test set predictions
    test_preds, test_uncerts = model.predict(test_smiles, return_uncertainty=True)
    valid_mask = ~np.isnan(test_preds)
    test_preds = test_preds[valid_mask]
    test_uncerts = test_uncerts[valid_mask]
    test_labels_valid = test_labels[valid_mask]
    test_errors = np.abs(test_preds - test_labels_valid)

    # 1. Baseline (no calibration)
    print("Evaluating baseline...")
    baseline_metrics = compute_expected_calibration_error(test_uncerts, test_errors)
    results['baseline'] = {
        'method': 'No Calibration',
        'metrics': baseline_metrics.to_dict(),
        'calibrator': None
    }

    # 2. Isotonic regression
    print("Fitting isotonic regression...")
    isotonic = IsotonicCalibrator()
    isotonic.fit(calib_uncerts, calib_errors)
    iso_uncerts = isotonic.calibrate(test_uncerts)
    iso_metrics = compute_expected_calibration_error(iso_uncerts, test_errors)
    results['isotonic'] = {
        'method': 'Isotonic Regression',
        'metrics': iso_metrics.to_dict(),
        'calibrator': isotonic
    }

    # 3. Temperature scaling
    print("Fitting temperature scaling...")
    temp_scaler = TemperatureScaler()
    temp_scaler.fit(calib_uncerts, calib_errors)
    temp_uncerts = temp_scaler.calibrate(test_uncerts)
    temp_metrics = compute_expected_calibration_error(temp_uncerts, test_errors)
    results['temperature'] = {
        'method': f'Temperature Scaling (T={temp_scaler.temperature:.3f})',
        'metrics': temp_metrics.to_dict(),
        'calibrator': temp_scaler
    }

    # 4. Platt scaling
    print("Fitting Platt scaling...")
    platt_scaler = PlattScaler(error_threshold=np.median(calib_errors))
    platt_scaler.fit(calib_uncerts, calib_errors)
    platt_uncerts = platt_scaler.calibrate(test_uncerts)
    platt_metrics = compute_expected_calibration_error(platt_uncerts, test_errors)
    results['platt'] = {
        'method': 'Platt Scaling',
        'metrics': platt_metrics.to_dict(),
        'calibrator': platt_scaler
    }

    # Summary
    print("\n" + "=" * 60)
    print("CALIBRATION COMPARISON")
    print("=" * 60)
    print(f"{'Method':<30} {'ECE':<10} {'Slope':<10} {'Sharpness':<10}")
    print("-" * 60)

    for name, result in results.items():
        m = result['metrics']
        print(f"{result['method']:<30} {m['ece']:.4f}     {m['calibration_slope']:.3f}      {m['sharpness']:.4f}")

    # Find best method
    best_method = min(results.keys(), key=lambda k: results[k]['metrics']['ece'])
    results['best_method'] = best_method
    print("-" * 60)
    print(f"Best method: {results[best_method]['method']} (ECE={results[best_method]['metrics']['ece']:.4f})")

    return results


def run_recalibration_experiment(
    data_path: str = 'data/esol_processed.pkl',
    output_dir: str = 'results/recalibration',
    random_state: int = 42
) -> Dict:
    """
    Run full recalibration experiment.

    1. Load data and split into train/calibration/test
    2. Train world model on train set
    3. Compare calibration methods on calibration set
    4. Evaluate best calibrator on test set
    5. Save calibrated model and visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'plots').mkdir(exist_ok=True)

    print("=" * 70)
    print("MODEL RECALIBRATION EXPERIMENT")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    candidate_df = data['candidate_pool']
    test_df = data['test_set']

    all_smiles = candidate_df['smiles'].tolist() + test_df['smiles'].tolist()
    all_labels = np.array(candidate_df['logS'].tolist() + test_df['logS'].tolist())

    # Split: 60% train, 20% calibration, 20% test
    train_smiles, temp_smiles, train_labels, temp_labels = train_test_split(
        all_smiles, all_labels, test_size=0.4, random_state=random_state
    )
    calib_smiles, test_smiles, calib_labels, test_labels = train_test_split(
        temp_smiles, temp_labels, test_size=0.5, random_state=random_state
    )

    print(f"Train: {len(train_smiles)}, Calibration: {len(calib_smiles)}, Test: {len(test_smiles)}")

    # Compare methods
    results = compare_calibration_methods(
        train_smiles, train_labels,
        calib_smiles, calib_labels,
        test_smiles, test_labels,
        random_state=random_state
    )

    # Add metadata
    results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'data_path': data_path,
        'n_train': len(train_smiles),
        'n_calib': len(calib_smiles),
        'n_test': len(test_smiles),
        'random_state': random_state
    }

    # Create reliability diagrams
    print("\nGenerating reliability diagrams...")

    # Train a fresh model to get predictions for plotting
    model = MolecularWorldModel(n_estimators=100, random_state=random_state)
    model.fit(train_smiles, train_labels)

    test_preds, test_uncerts = model.predict(test_smiles, return_uncertainty=True)
    valid_mask = ~np.isnan(test_preds)
    test_uncerts_valid = test_uncerts[valid_mask]
    test_errors = np.abs(test_preds[valid_mask] - test_labels[valid_mask])

    # Plot baseline
    baseline_metrics = compute_expected_calibration_error(test_uncerts_valid, test_errors)
    plot_reliability_diagram(
        baseline_metrics,
        title="Before Calibration",
        save_path=str(output_path / 'plots' / 'reliability_diagram_before.png')
    )

    # Plot best calibrated
    best_method = results['best_method']
    if results[best_method]['calibrator'] is not None:
        calibrator = results[best_method]['calibrator']
        calib_uncerts = calibrator.calibrate(test_uncerts_valid)
        calib_metrics = compute_expected_calibration_error(calib_uncerts, test_errors)
        plot_reliability_diagram(
            calib_metrics,
            title=f"After {results[best_method]['method']}",
            save_path=str(output_path / 'plots' / 'reliability_diagram_after.png')
        )

    # Save comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    methods = ['baseline', 'isotonic', 'temperature', 'platt']
    titles = ['No Calibration', 'Isotonic Regression', 'Temperature Scaling', 'Platt Scaling']

    for ax, method, title in zip(axes.flatten(), methods, titles):
        metrics = CalibrationMetrics(**{k: v if not isinstance(v, list) else v
                                       for k, v in results[method]['metrics'].items()})

        bin_confs = np.array(metrics.bin_confidences)
        bin_accs = np.array(metrics.bin_accuracies)

        if len(bin_confs) > 0:
            max_val = max(max(bin_confs), max(bin_accs)) * 1.1
            ax.scatter(bin_confs, bin_accs, s=50, alpha=0.7, c='steelblue')
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
            if len(bin_confs) > 1:
                slope, intercept, _, _, _ = stats.linregress(bin_confs, bin_accs)
                x_fit = np.linspace(0, max_val, 100)
                ax.plot(x_fit, slope * x_fit + intercept, 'r-', alpha=0.7)

        ax.set_xlabel('Predicted Uncertainty')
        ax.set_ylabel('Actual Error')
        ax.set_title(f'{title}\nECE={metrics.ece:.4f}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'plots' / 'ece_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path / 'plots' / 'ece_comparison.png'}")
    plt.close()

    # Save best calibrator
    if results[best_method]['calibrator'] is not None:
        calibrator_path = output_path / 'best_calibrator.pkl'
        with open(calibrator_path, 'wb') as f:
            pickle.dump(results[best_method]['calibrator'], f)
        print(f"\nSaved best calibrator to {calibrator_path}")

    # Save results JSON (without calibrator objects)
    results_for_json = {
        k: v if k != 'best_method' else v
        for k, v in results.items()
    }
    for method in ['baseline', 'isotonic', 'temperature', 'platt']:
        if method in results_for_json:
            results_for_json[method] = {
                'method': results_for_json[method]['method'],
                'metrics': results_for_json[method]['metrics']
            }

    with open(output_path / 'calibration_metrics.json', 'w') as f:
        json.dump(results_for_json, f, indent=2)
    print(f"Saved metrics to {output_path / 'calibration_metrics.json'}")

    # Print verdict
    print("\n" + "=" * 70)
    print("RECALIBRATION VERDICT")
    print("=" * 70)

    baseline_ece = results['baseline']['metrics']['ece']
    best_ece = results[best_method]['metrics']['ece']
    improvement = (baseline_ece - best_ece) / baseline_ece * 100

    print(f"Baseline ECE: {baseline_ece:.4f}")
    print(f"Best ECE ({results[best_method]['method']}): {best_ece:.4f}")
    print(f"Improvement: {improvement:.1f}%")

    if best_ece < 0.1:
        print("\n[PASS] ECE < 0.1 - Calibration is good!")
    elif best_ece < 0.2:
        print("\n[PARTIAL] ECE < 0.2 - Calibration is acceptable")
    else:
        print("\n[NEEDS ATTENTION] ECE >= 0.2 - Consider different approaches")

    return results


def integrate_calibrator_with_model(
    model: MolecularWorldModel,
    calibrator_path: str
) -> MolecularWorldModel:
    """
    Create a calibrated version of the world model.

    This modifies the model's predict method to apply calibration.
    """
    with open(calibrator_path, 'rb') as f:
        calibrator = pickle.load(f)

    # Store original predict
    original_predict = model.predict

    def calibrated_predict(smiles_list, return_uncertainty=True):
        result = original_predict(smiles_list, return_uncertainty=return_uncertainty)
        if return_uncertainty:
            preds, uncerts = result
            # Apply calibration
            valid_mask = ~np.isnan(uncerts)
            if np.any(valid_mask):
                uncerts_valid = uncerts[valid_mask]
                calibrated = calibrator.calibrate(uncerts_valid)
                uncerts[valid_mask] = calibrated
            return preds, uncerts
        return result

    model.predict = calibrated_predict
    model._calibrator = calibrator

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run model recalibration experiment")
    parser.add_argument('--data-path', default='data/esol_processed.pkl',
                       help='Path to processed ESOL data')
    parser.add_argument('--output-dir', default='results/recalibration',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    results = run_recalibration_experiment(
        data_path=args.data_path,
        output_dir=args.output_dir,
        random_state=args.seed
    )
