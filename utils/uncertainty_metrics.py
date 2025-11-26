"""
Uncertainty Quantification and Calibration Metrics

Goal: Know when the world model's predictions are reliable.

Good calibration means: high uncertainty → high error
This allows the agent to know when to trust its world model vs. explore more.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.stats import spearmanr
import warnings


def compute_calibration_metrics(
    predictions: List[float],
    uncertainties: List[float],
    actuals: List[float]
) -> Dict:
    """
    Measure if uncertainty estimates are calibrated.

    Good calibration: high uncertainty correlates with high prediction error.

    Args:
        predictions: Predicted values
        uncertainties: Uncertainty estimates (e.g., std dev)
        actuals: Ground truth values

    Returns:
        {
            'uncertainty_correlation': float,  # Correlation between uncertainty and error
            'correlation_p_value': float,      # Statistical significance
            'calibration_plot_data': list,      # Data for calibration curve
            'mean_error': float,                # Average absolute error
            'mean_uncertainty': float           # Average uncertainty
        }
    """
    if not predictions or not uncertainties or not actuals:
        return {
            'uncertainty_correlation': 0.0,
            'correlation_p_value': 1.0,
            'calibration_plot_data': [],
            'mean_error': 0.0,
            'mean_uncertainty': 0.0
        }

    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)
    actuals = np.array(actuals)

    # Calculate prediction errors
    errors = np.abs(predictions - actuals)

    # Correlation between uncertainty and error
    # High correlation = good calibration (uncertain predictions have higher errors)
    try:
        corr, p_value = spearmanr(uncertainties, errors)
        if np.isnan(corr):
            corr = 0.0
            p_value = 1.0
    except Exception as e:
        warnings.warn(f"Failed to compute correlation: {e}")
        corr = 0.0
        p_value = 1.0

    # Calibration curve: bin by uncertainty, plot mean error per bin
    n_bins = 5
    calibration_data = []

    try:
        # Create bins based on uncertainty percentiles
        if len(uncertainties) >= n_bins:
            unc_bins = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))

            for i in range(n_bins):
                mask = (uncertainties >= unc_bins[i]) & (uncertainties < unc_bins[i+1])

                # Include last bin's upper boundary
                if i == n_bins - 1:
                    mask = uncertainties >= unc_bins[i]

                if np.sum(mask) > 0:
                    calibration_data.append({
                        'uncertainty_bin': (float(unc_bins[i]), float(unc_bins[i+1])),
                        'mean_uncertainty': float(np.mean(uncertainties[mask])),
                        'mean_error': float(np.mean(errors[mask])),
                        'count': int(np.sum(mask))
                    })
    except Exception as e:
        warnings.warn(f"Failed to compute calibration curve: {e}")

    return {
        'uncertainty_correlation': float(corr),
        'correlation_p_value': float(p_value),
        'calibration_plot_data': calibration_data,
        'mean_error': float(np.mean(errors)),
        'mean_uncertainty': float(np.mean(uncertainties))
    }


def compute_brier_score(
    predictions: List[float],
    actuals: List[float]
) -> float:
    """
    Compute Brier score for probabilistic predictions.

    Lower is better. Only works for predictions in [0, 1].

    Args:
        predictions: Predicted probabilities
        actuals: Actual binary outcomes (0 or 1)

    Returns:
        Brier score (lower is better)
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Check if predictions are probabilities
    if not np.all((predictions >= 0) & (predictions <= 1)):
        warnings.warn("Predictions not in [0, 1], cannot compute Brier score")
        return float('nan')

    # Check if actuals are binary
    if not np.all((actuals == 0) | (actuals == 1)):
        warnings.warn("Actuals not binary, cannot compute Brier score")
        return float('nan')

    # Brier score = mean squared error
    brier = np.mean((predictions - actuals) ** 2)

    return float(brier)


def add_uncertainty_to_world_model(world_model):
    """
    Augment WorldModelSimulator with uncertainty estimation.

    For Phase 1, use simple bootstrap-based uncertainty.
    Uncertainty decreases with sqrt(n) observations (standard error).

    Args:
        world_model: WorldModelSimulator instance

    Returns:
        Augmented world model with predict_with_uncertainty method
    """

    def predict_with_uncertainty(
        observation: dict,
        time: float,
        base_uncertainty: float = 5.0
    ) -> Tuple[Optional[float], float]:
        """
        Predict with uncertainty estimate.

        Args:
            observation: Observation dict with context
            time: Time parameter
            base_uncertainty: Base uncertainty (domain-specific)

        Returns:
            (prediction, uncertainty) tuple
            - prediction: Predicted value (or None if context unknown)
            - uncertainty: Uncertainty estimate (higher = less reliable)
        """
        # Extract context from observation
        context_key = world_model.context_spec.extract_context(observation)

        # Check if context is known
        if context_key not in world_model.context_models:
            # Unknown context → maximum uncertainty
            return (None, float('inf'))

        # Point prediction
        try:
            prediction = world_model.predict(observation, time)
        except Exception as e:
            warnings.warn(f"Prediction failed: {e}")
            return (None, float('inf'))

        # Uncertainty estimation
        # Simple heuristic: fewer observations → higher uncertainty
        n_observations = world_model.context_observation_counts.get(context_key, 1)

        # Uncertainty decreases with sqrt(n) (standard error formula)
        uncertainty = base_uncertainty / np.sqrt(n_observations)

        return (prediction, uncertainty)

    # Monkey-patch the method onto the world model
    world_model.predict_with_uncertainty = predict_with_uncertainty

    return world_model


def evaluate_calibration(
    world_model,
    test_observations: List[dict],
    base_uncertainty: float = 5.0
) -> Dict:
    """
    Evaluate calibration of world model's uncertainty estimates.

    Args:
        world_model: WorldModelSimulator with predict_with_uncertainty
        test_observations: List of test observations with ground truth
        base_uncertainty: Base uncertainty parameter

    Returns:
        Calibration metrics dictionary
    """
    # Add uncertainty capability if not present
    if not hasattr(world_model, 'predict_with_uncertainty'):
        world_model = add_uncertainty_to_world_model(world_model)

    predictions = []
    uncertainties = []
    actuals = []

    for obs in test_observations:
        try:
            # Get prediction with uncertainty
            time = obs.get('time', 1.0)
            pred, unc = world_model.predict_with_uncertainty(
                obs, time, base_uncertainty
            )

            if pred is None:
                continue

            # Extract actual value
            # Domain-specific: for hot_pot, use measured_temp
            actual = obs.get('measured_temp')
            if actual is None:
                actual = obs.get('temperature')
            if actual is None:
                continue

            predictions.append(pred)
            uncertainties.append(unc)
            actuals.append(actual)

        except Exception as e:
            warnings.warn(f"Failed to evaluate observation: {e}")
            continue

    if not predictions:
        return {
            'uncertainty_correlation': 0.0,
            'error': 'No valid predictions'
        }

    return compute_calibration_metrics(predictions, uncertainties, actuals)


def uncertainty_weighted_ensemble(
    predictions: List[float],
    uncertainties: List[float]
) -> Tuple[float, float]:
    """
    Combine multiple predictions weighted by inverse uncertainty.

    More certain predictions get higher weight.

    Args:
        predictions: List of predictions
        uncertainties: List of uncertainties (lower = more certain)

    Returns:
        (ensemble_prediction, ensemble_uncertainty)
    """
    if not predictions:
        return (0.0, float('inf'))

    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)

    # Avoid division by zero
    uncertainties = np.maximum(uncertainties, 1e-6)

    # Inverse uncertainty weighting
    weights = 1.0 / uncertainties
    weights = weights / np.sum(weights)  # Normalize

    # Weighted average
    ensemble_pred = np.sum(predictions * weights)

    # Ensemble uncertainty (weighted variance)
    ensemble_unc = np.sqrt(np.sum(weights * (predictions - ensemble_pred)**2))

    return (float(ensemble_pred), float(ensemble_unc))
