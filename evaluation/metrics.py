# evaluation/metrics.py
"""
Metrics for evaluating agent performance.

Key metrics:
- Interventional accuracy: Success on "what would happen if" queries
- Counterfactual accuracy: Success on "what would have happened if" queries
- Surprisal trajectory: Learning over time
- Calibration: Alignment of confidence with correctness
- Planning success: Success on goal-directed planning queries
"""

import numpy as np
from typing import List, Dict, Any, Optional


def interventional_accuracy(test_results: List[Dict]) -> float:
    """
    Accuracy on interventional queries.

    Interventional queries test: "What would happen if we DO X?"

    Args:
        test_results: List of test result dictionaries

    Returns:
        Accuracy (0.0 to 1.0)
    """
    interventional = [r for r in test_results if r.get('query_type') == 'interventional']

    if not interventional:
        return 0.0

    return float(np.mean([r.get('score', r.get('correct', 0.0)) for r in interventional]))


def counterfactual_accuracy(test_results: List[Dict]) -> float:
    """
    Accuracy on counterfactual queries.

    Counterfactual queries test: "What would have happened if we HAD DONE X?"

    Args:
        test_results: List of test result dictionaries

    Returns:
        Accuracy (0.0 to 1.0)
    """
    counterfactual = [r for r in test_results if r.get('query_type') == 'counterfactual']

    if not counterfactual:
        return 0.0

    return float(np.mean([r.get('score', r.get('correct', 0.0)) for r in counterfactual]))


def planning_success_rate(test_results: List[Dict]) -> Dict[str, float]:
    """
    Success rate on planning queries.

    Planning queries test: "How can we achieve goal Y?"

    Args:
        test_results: List of test result dictionaries

    Returns:
        Dictionary with success_rate and n_queries
    """
    planning = [r for r in test_results if r.get('query_type') == 'planning']

    if not planning:
        return {'success_rate': 0.0, 'n_queries': 0}

    return {
        'success_rate': float(np.mean([r.get('score', r.get('correct', 0.0)) for r in planning])),
        'n_queries': len(planning)
    }


def surprisal_trajectory(steps: List[Dict]) -> Dict[str, Any]:
    """
    Analyze surprisal over time.

    Actor agents should show decreasing surprisal (learning).
    Observer agents should show flat surprisal (no learning).

    Args:
        steps: List of episode steps with 'surprisal' field

    Returns:
        Dictionary with trajectory statistics
    """
    surprisals = [s.get('surprisal', 0.0) for s in steps]

    # Filter out zero surprisals (from agents that don't compute it)
    non_zero_surprisals = [s for s in surprisals if s > 0]

    # Handle case with no non-zero surprisals
    if len(non_zero_surprisals) == 0:
        return {
            'surprisals': surprisals,
            'slope': 0.0,
            'mean_surprisal': 0.0,
            'final_surprisal': 0.0,
            'max_surprisal': 0.0,
            'max_surprisal_step': 0,
            'learning_detected': False
        }

    # Handle case with single non-zero surprisal
    if len(non_zero_surprisals) == 1:
        return {
            'surprisals': surprisals,
            'slope': 0.0,  # Cannot compute slope with 1 point
            'mean_surprisal': float(non_zero_surprisals[0]),
            'final_surprisal': float(non_zero_surprisals[0]),
            'max_surprisal': float(non_zero_surprisals[0]),
            'max_surprisal_step': int(np.argmax(surprisals)),
            'initial_surprisal': float(non_zero_surprisals[0]),
            'learning_detected': False
        }

    # Use non-zero surprisals for analysis
    analysis_surprisals = non_zero_surprisals

    # Linear trend (negative slope = decreasing = learning)
    x = np.arange(len(analysis_surprisals))
    slope, intercept = np.polyfit(x, analysis_surprisals, 1)

    return {
        'surprisals': surprisals,
        'slope': float(slope),
        'mean_surprisal': float(np.mean(analysis_surprisals)),
        'final_surprisal': float(analysis_surprisals[-1]),
        'max_surprisal': float(np.max(analysis_surprisals)),
        'max_surprisal_step': int(np.argmax(surprisals)),
        'initial_surprisal': float(analysis_surprisals[0]),
        'learning_detected': slope < -0.1  # Arbitrary threshold
    }


def calibration_metrics(test_results: List[Dict]) -> Dict[str, float]:
    """
    Measure calibration of confidence scores.

    Well-calibrated agents have confidence aligned with correctness.
    Brier score measures probabilistic calibration (lower is better).

    Args:
        test_results: List of test result dictionaries with 'confidence' and 'correct'

    Returns:
        Dictionary with calibration metrics
    """
    confidences = [r.get('confidence', 0.5) for r in test_results]
    correct = [float(r.get('score', r.get('correct', 0.0))) for r in test_results]

    if not confidences or len(confidences) != len(correct):
        return {
            'brier_score': 1.0,
            'mean_confidence': 0.5,
            'confidence_std': 0.0,
            'mean_correctness': 0.0
        }

    # Brier score: mean squared error of probabilistic predictions
    brier = float(np.mean([(c - p)**2 for c, p in zip(correct, confidences)]))

    # Compute calibration curve (binned)
    calibration_curve = _compute_calibration_curve(confidences, correct)

    return {
        'brier_score': float(brier),
        'mean_confidence': float(np.mean(confidences)),
        'confidence_std': float(np.std(confidences)),
        'mean_correctness': float(np.mean(correct)),
        'calibration_curve': calibration_curve
    }


def _compute_calibration_curve(
    confidences: List[float],
    correct: List[float],
    n_bins: int = 5
) -> Dict[str, List[float]]:
    """
    Compute calibration curve by binning confidences.

    Args:
        confidences: Predicted probabilities
        correct: Actual outcomes (0 or 1, or scores 0.0 to 1.0)
        n_bins: Number of bins

    Returns:
        Dictionary with bin_edges, bin_means, bin_accuracies
    """
    if not confidences:
        return {'bin_edges': [], 'bin_confidences': [], 'bin_accuracies': []}

    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)

    bin_confidences = []
    bin_accuracies = []

    for i in range(n_bins):
        lower = bins[i]
        upper = bins[i + 1]

        # Find confidences in this bin
        in_bin = [(c, a) for c, a in zip(confidences, correct)
                  if lower <= c < upper or (i == n_bins - 1 and c == upper)]

        if in_bin:
            bin_conf = np.mean([c for c, a in in_bin])
            bin_acc = np.mean([a for c, a in in_bin])

            bin_confidences.append(float(bin_conf))
            bin_accuracies.append(float(bin_acc))
        else:
            bin_confidences.append(float((lower + upper) / 2))
            bin_accuracies.append(0.0)

    return {
        'bin_edges': bins.tolist(),
        'bin_confidences': bin_confidences,
        'bin_accuracies': bin_accuracies
    }


def overall_accuracy(test_results: List[Dict]) -> float:
    """
    Overall accuracy across all query types.

    Args:
        test_results: List of test result dictionaries

    Returns:
        Overall accuracy (0.0 to 1.0)
    """
    if not test_results:
        return 0.0

    scores = [r.get('score', r.get('correct', 0.0)) for r in test_results]
    return float(np.mean(scores))


def accuracy_by_difficulty(test_results: List[Dict]) -> Dict[str, float]:
    """
    Accuracy broken down by difficulty level.

    Args:
        test_results: List of test result dictionaries

    Returns:
        Dictionary mapping difficulty to accuracy
    """
    by_difficulty = {}

    for difficulty in ['easy', 'medium', 'hard']:
        queries = [r for r in test_results if r.get('difficulty') == difficulty]

        if queries:
            acc = float(np.mean([r.get('score', r.get('correct', 0.0)) for r in queries]))
            by_difficulty[difficulty] = acc
        else:
            by_difficulty[difficulty] = 0.0

    return by_difficulty


def action_efficiency(steps: List[Dict]) -> Dict[str, Any]:
    """
    Measure how efficiently agent used its action budget.

    Args:
        steps: List of episode steps

    Returns:
        Dictionary with efficiency metrics
    """
    total_steps = len(steps)
    actions_taken = sum(1 for s in steps if s.get('action') is not None)

    # Compute action diversity
    action_types = [s.get('action', '').split('(')[0] for s in steps if s.get('action')]
    unique_actions = len(set(action_types)) if action_types else 0

    return {
        'total_steps': total_steps,
        'actions_taken': actions_taken,
        'action_usage_rate': float(actions_taken / total_steps) if total_steps > 0 else 0.0,
        'unique_action_types': unique_actions,
        'action_diversity': float(unique_actions / actions_taken) if actions_taken > 0 else 0.0
    }


def token_usage(steps: List[Dict]) -> Dict[str, int]:
    """
    Measure LLM token usage.

    Args:
        steps: List of episode steps

    Returns:
        Dictionary with token usage statistics
    """
    total_tokens = sum(s.get('token_usage', 0) for s in steps)
    tokens_per_step = [s.get('token_usage', 0) for s in steps]

    return {
        'total_tokens': total_tokens,
        'mean_tokens_per_step': int(np.mean(tokens_per_step)) if tokens_per_step else 0,
        'max_tokens_per_step': int(np.max(tokens_per_step)) if tokens_per_step else 0
    }


def compute_all_metrics(episode_log: Dict) -> Dict[str, Any]:
    """
    Compute all metrics for an episode.

    Args:
        episode_log: Episode log dictionary with 'steps' and 'test_results'

    Returns:
        Dictionary with all computed metrics
    """
    steps = episode_log.get('steps', [])
    test_results = episode_log.get('test_results', [])

    return {
        'overall_accuracy': overall_accuracy(test_results),
        'interventional_accuracy': interventional_accuracy(test_results),
        'counterfactual_accuracy': counterfactual_accuracy(test_results),
        'planning_success': planning_success_rate(test_results),
        'accuracy_by_difficulty': accuracy_by_difficulty(test_results),
        'surprisal_trajectory': surprisal_trajectory(steps),
        'calibration': calibration_metrics(test_results),
        'action_efficiency': action_efficiency(steps),
        'token_usage': token_usage(steps),
    }


def aggregate_metrics(episode_logs: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple episodes.

    Args:
        episode_logs: List of episode log dictionaries

    Returns:
        Dictionary with aggregated metrics
    """
    if not episode_logs:
        return {}

    # Compute metrics for each episode
    all_metrics = [compute_all_metrics(ep) for ep in episode_logs]

    # Aggregate
    aggregated = {
        'n_episodes': len(episode_logs),
        'overall_accuracy_mean': float(np.mean([m['overall_accuracy'] for m in all_metrics])),
        'overall_accuracy_std': float(np.std([m['overall_accuracy'] for m in all_metrics])),
        'interventional_accuracy_mean': float(np.mean([m['interventional_accuracy'] for m in all_metrics])),
        'interventional_accuracy_std': float(np.std([m['interventional_accuracy'] for m in all_metrics])),
        'counterfactual_accuracy_mean': float(np.mean([m['counterfactual_accuracy'] for m in all_metrics])),
        'counterfactual_accuracy_std': float(np.std([m['counterfactual_accuracy'] for m in all_metrics])),
        'brier_score_mean': float(np.mean([m['calibration']['brier_score'] for m in all_metrics])),
        'brier_score_std': float(np.std([m['calibration']['brier_score'] for m in all_metrics])),
    }

    # Add surprisal slopes
    slopes = [m['surprisal_trajectory']['slope'] for m in all_metrics
              if m['surprisal_trajectory']['slope'] != 0.0]

    if slopes:
        aggregated['surprisal_slope_mean'] = float(np.mean(slopes))
        aggregated['surprisal_slope_std'] = float(np.std(slopes))
        aggregated['learning_rate'] = float(np.mean(slopes))  # Negative = learning

    return aggregated
