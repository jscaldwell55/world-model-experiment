"""
Rebuild all metrics from raw episode logs.

This module recomputes Actor/Observer/Model-Based accuracies, surprisal,
slopes, p-values, and effect sizes directly from episode JSON logs.

NO cached tensors, NO pre-aggregated CSVs - everything from raw logs only.

Key metrics:
- Accuracy: mean(correct) over same denominator for all agents
- Surprisal: -log p(o_t | belief_{t-1}) in nats
- Surprisal slope: OLS regression surprisal_t = β₀ + β₁·t + ε_t
- Effect sizes: Cohen's d for between-agent comparisons
- Statistical tests: t-tests, ANOVA, regression diagnostics
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, f_oneway
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd


@dataclass
class EpisodeMetrics:
    """Metrics computed for a single episode."""
    episode_id: str
    environment: str
    agent_type: str
    seed: int

    # Accuracy metrics
    accuracy_overall: float
    accuracy_interventional: float
    accuracy_counterfactual: float
    accuracy_planning: float
    num_test_queries: int
    num_correct: int

    # Surprisal metrics
    surprisal_values: List[float]
    surprisal_mean: float
    surprisal_std: float
    surprisal_slope: float
    surprisal_intercept: float
    surprisal_r_squared: float
    surprisal_p_value: float

    # Step-level data
    num_steps: int
    actions_taken: int

    # Raw data for verification
    test_results: List[Dict] = field(default_factory=list)
    steps: List[Dict] = field(default_factory=list)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple episodes for one agent type."""
    agent_type: str
    environment: str
    n_episodes: int

    # Accuracy statistics
    accuracy_mean: float
    accuracy_std: float
    accuracy_sem: float
    accuracy_ci_lower: float
    accuracy_ci_upper: float

    # Surprisal slope statistics
    slope_mean: float
    slope_std: float
    slope_sem: float
    slope_ci_lower: float
    slope_ci_upper: float
    slope_p_value: float

    # Effect sizes (vs baseline)
    cohens_d_accuracy: Optional[float] = None
    cohens_d_slope: Optional[float] = None

    # Individual episode metrics for verification
    episodes: List[EpisodeMetrics] = field(default_factory=list)


def load_episode_log(filepath: Path) -> Dict:
    """Load a single episode JSON log.

    Args:
        filepath: Path to JSON log file

    Returns:
        Episode log dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_accuracy_from_raw(test_results: List[Dict]) -> Dict[str, Any]:
    """Compute accuracy metrics from raw test results.

    Accuracy = mean(correct) where correct ∈ {0, 1} or score ∈ [0, 1].
    Uses the SAME denominator (all queries) for all agent types.

    Args:
        test_results: List of test result dictionaries with 'correct' or 'score'

    Returns:
        Dictionary with accuracy metrics
    """
    if not test_results:
        return {
            'overall': 0.0,
            'interventional': 0.0,
            'counterfactual': 0.0,
            'planning': 0.0,
            'num_queries': 0,
            'num_correct': 0
        }

    # Extract scores (prefer 'score' field, fallback to 'correct')
    scores = [r.get('score', float(r.get('correct', 0.0))) for r in test_results]

    # Overall accuracy
    overall = float(np.mean(scores))
    num_correct = sum(1 for s in scores if s >= 0.5)  # Count as correct if score >= 0.5

    # By query type
    interventional = [r.get('score', float(r.get('correct', 0.0)))
                      for r in test_results if r.get('query_type') == 'interventional']
    counterfactual = [r.get('score', float(r.get('correct', 0.0)))
                      for r in test_results if r.get('query_type') == 'counterfactual']
    planning = [r.get('score', float(r.get('correct', 0.0)))
                for r in test_results if r.get('query_type') == 'planning']

    return {
        'overall': overall,
        'interventional': float(np.mean(interventional)) if interventional else 0.0,
        'counterfactual': float(np.mean(counterfactual)) if counterfactual else 0.0,
        'planning': float(np.mean(planning)) if planning else 0.0,
        'num_queries': len(test_results),
        'num_correct': num_correct
    }


def compute_surprisal_from_raw(steps: List[Dict]) -> Dict[str, Any]:
    """Compute surprisal metrics from raw step data.

    Surprisal is stored in steps as 'surprisal' field.
    Units: nats (natural logarithm base e).

    Definition: surprisal_t = -log p(o_t | belief_{t-1})

    Args:
        steps: List of step dictionaries with 'surprisal' field

    Returns:
        Dictionary with surprisal statistics
    """
    # Extract surprisal values (filter out zeros from agents that don't compute it)
    surprisals = [s.get('surprisal', 0.0) for s in steps]
    non_zero_surprisals = [s for s in surprisals if s > 0]

    if not non_zero_surprisals:
        return {
            'values': surprisals,
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'n_nonzero': 0
        }

    return {
        'values': surprisals,
        'mean': float(np.mean(non_zero_surprisals)),
        'std': float(np.std(non_zero_surprisals, ddof=1)) if len(non_zero_surprisals) > 1 else 0.0,
        'min': float(np.min(non_zero_surprisals)),
        'max': float(np.max(non_zero_surprisals)),
        'n_nonzero': len(non_zero_surprisals)
    }


def fit_surprisal_slope_ols(steps: List[Dict]) -> Dict[str, Any]:
    """Fit OLS regression for surprisal over time.

    Model: surprisal_t = β₀ + β₁·t + ε_t

    Where:
    - t is the step number (time index)
    - β₁ < 0 indicates learning (decreasing surprisal)
    - β₁ ≈ 0 indicates no learning

    Args:
        steps: List of step dictionaries with 'surprisal' and 'step_num'

    Returns:
        Dictionary with regression results and diagnostics
    """
    # Extract data
    surprisals = [s.get('surprisal', 0.0) for s in steps]
    step_nums = [s.get('step_num', i) for i, s in enumerate(steps)]

    # Filter non-zero surprisals (some agents don't compute it)
    valid_pairs = [(t, surp) for t, surp in zip(step_nums, surprisals) if surp > 0]

    if len(valid_pairs) < 2:
        # Not enough data for regression
        return {
            'slope': 0.0,
            'intercept': 0.0,
            'r_squared': 0.0,
            'p_value': 1.0,
            'stderr': 0.0,
            'n_points': len(valid_pairs)
        }

    # Prepare data
    t = np.array([x[0] for x in valid_pairs]).reshape(-1, 1)
    y = np.array([x[1] for x in valid_pairs])

    # Fit OLS
    model = LinearRegression()
    model.fit(t, y)

    # Predictions
    y_pred = model.predict(t)

    # Statistics
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r_squared = float(r2_score(y, y_pred))

    # Compute p-value for slope (test H0: β₁ = 0)
    # Use t-statistic: t = slope / SE(slope)
    n = len(t)
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (n - 2) if n > 2 else 0.0

    # Standard error of slope
    t_mean = np.mean(t)
    se_slope = np.sqrt(mse / np.sum((t - t_mean)**2)) if n > 2 else 0.0

    # t-statistic and p-value
    if se_slope > 0:
        t_stat = slope / se_slope
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
    else:
        p_value = 1.0

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'p_value': float(p_value),
        'stderr': float(se_slope),
        'n_points': n,
        'residuals': residuals.tolist(),
        'predictions': y_pred.tolist()
    }


def compute_episode_metrics(log_filepath: Path) -> EpisodeMetrics:
    """Compute all metrics for a single episode from raw log.

    Args:
        log_filepath: Path to episode JSON log

    Returns:
        EpisodeMetrics object
    """
    # Load raw log
    log = load_episode_log(log_filepath)

    # Parse metadata
    episode_id = log['episode_id']
    environment = log.get('environment', 'unknown')
    agent_type = log.get('agent_type', 'unknown')
    seed = log.get('seed', 0)

    # Extract data
    steps = log.get('steps', [])
    test_results = log.get('test_results', [])

    # Compute accuracy
    acc = compute_accuracy_from_raw(test_results)

    # Compute surprisal statistics
    surp = compute_surprisal_from_raw(steps)

    # Fit surprisal slope
    slope_results = fit_surprisal_slope_ols(steps)

    # Count actions
    actions_taken = sum(1 for s in steps if s.get('action') is not None)

    return EpisodeMetrics(
        episode_id=episode_id,
        environment=environment,
        agent_type=agent_type,
        seed=seed,
        accuracy_overall=acc['overall'],
        accuracy_interventional=acc['interventional'],
        accuracy_counterfactual=acc['counterfactual'],
        accuracy_planning=acc['planning'],
        num_test_queries=acc['num_queries'],
        num_correct=acc['num_correct'],
        surprisal_values=surp['values'],
        surprisal_mean=surp['mean'],
        surprisal_std=surp['std'],
        surprisal_slope=slope_results['slope'],
        surprisal_intercept=slope_results['intercept'],
        surprisal_r_squared=slope_results['r_squared'],
        surprisal_p_value=slope_results['p_value'],
        num_steps=len(steps),
        actions_taken=actions_taken,
        test_results=test_results,
        steps=steps
    )


def aggregate_episode_metrics(
    episodes: List[EpisodeMetrics],
    agent_type: str,
    environment: str,
    confidence_level: float = 0.95
) -> AggregatedMetrics:
    """Aggregate metrics across multiple episodes for one agent type.

    Args:
        episodes: List of EpisodeMetrics
        agent_type: Agent type name
        environment: Environment name
        confidence_level: Confidence level for CIs (default 95%)

    Returns:
        AggregatedMetrics object
    """
    if not episodes:
        return AggregatedMetrics(
            agent_type=agent_type,
            environment=environment,
            n_episodes=0,
            accuracy_mean=0.0,
            accuracy_std=0.0,
            accuracy_sem=0.0,
            accuracy_ci_lower=0.0,
            accuracy_ci_upper=0.0,
            slope_mean=0.0,
            slope_std=0.0,
            slope_sem=0.0,
            slope_ci_lower=0.0,
            slope_ci_upper=0.0,
            slope_p_value=1.0,
            episodes=episodes
        )

    n = len(episodes)

    # Accuracy statistics
    accuracies = [ep.accuracy_overall for ep in episodes]
    acc_mean = float(np.mean(accuracies))
    acc_std = float(np.std(accuracies, ddof=1)) if n > 1 else 0.0
    acc_sem = acc_std / np.sqrt(n) if n > 0 else 0.0

    # Confidence interval for accuracy
    if n > 1:
        t_crit = stats.t.ppf((1 + confidence_level) / 2, df=n-1)
        acc_ci_lower = acc_mean - t_crit * acc_sem
        acc_ci_upper = acc_mean + t_crit * acc_sem
    else:
        acc_ci_lower = acc_mean
        acc_ci_upper = acc_mean

    # Surprisal slope statistics
    slopes = [ep.surprisal_slope for ep in episodes]
    slope_mean = float(np.mean(slopes))
    slope_std = float(np.std(slopes, ddof=1)) if n > 1 else 0.0
    slope_sem = slope_std / np.sqrt(n) if n > 0 else 0.0

    # Confidence interval for slope
    if n > 1:
        slope_ci_lower = slope_mean - t_crit * slope_sem
        slope_ci_upper = slope_mean + t_crit * slope_sem
    else:
        slope_ci_lower = slope_mean
        slope_ci_upper = slope_mean

    # Test if mean slope is significantly different from 0
    if n > 1:
        t_stat, p_val = stats.ttest_1samp(slopes, 0.0)
        slope_p_value = float(p_val)
    else:
        slope_p_value = 1.0

    return AggregatedMetrics(
        agent_type=agent_type,
        environment=environment,
        n_episodes=n,
        accuracy_mean=acc_mean,
        accuracy_std=acc_std,
        accuracy_sem=acc_sem,
        accuracy_ci_lower=acc_ci_lower,
        accuracy_ci_upper=acc_ci_upper,
        slope_mean=slope_mean,
        slope_std=slope_std,
        slope_sem=slope_sem,
        slope_ci_lower=slope_ci_lower,
        slope_ci_upper=slope_ci_upper,
        slope_p_value=slope_p_value,
        episodes=episodes
    )


def compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size between two groups.

    Cohen's d = (mean1 - mean2) / pooled_std

    Interpretation:
    - |d| < 0.2: small effect
    - 0.2 <= |d| < 0.8: medium effect
    - |d| >= 0.8: large effect

    Args:
        group1: Values for group 1
        group2: Values for group 2

    Returns:
        Cohen's d effect size
    """
    if not group1 or not group2:
        return 0.0

    mean1 = np.mean(group1)
    mean2 = np.mean(group2)

    n1 = len(group1)
    n2 = len(group2)

    var1 = np.var(group1, ddof=1) if n1 > 1 else 0.0
    var2 = np.var(group2, ddof=1) if n2 > 1 else 0.0

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((mean1 - mean2) / pooled_std)


def load_all_episodes_from_directory(
    log_dir: Path,
    pattern: str = "*.json"
) -> List[EpisodeMetrics]:
    """Load and compute metrics for all episodes in a directory.

    Args:
        log_dir: Directory containing episode JSON logs
        pattern: Glob pattern for log files

    Returns:
        List of EpisodeMetrics
    """
    log_files = sorted(Path(log_dir).glob(pattern))

    episodes = []
    for log_file in log_files:
        try:
            ep_metrics = compute_episode_metrics(log_file)
            episodes.append(ep_metrics)
        except Exception as e:
            print(f"Warning: Failed to process {log_file}: {e}")
            continue

    return episodes


def compare_agents_statistical(
    agent_metrics: Dict[str, List[EpisodeMetrics]],
    metric_name: str = 'accuracy_overall'
) -> Dict[str, Any]:
    """Compare agents using statistical tests.

    Performs:
    - ANOVA F-test for overall difference
    - Pairwise t-tests with Bonferroni correction
    - Effect sizes (Cohen's d)

    Args:
        agent_metrics: Dictionary mapping agent_type -> list of EpisodeMetrics
        metric_name: Name of metric to compare (e.g., 'accuracy_overall', 'surprisal_slope')

    Returns:
        Dictionary with statistical test results
    """
    # Extract groups
    groups = {}
    for agent_type, episodes in agent_metrics.items():
        values = [getattr(ep, metric_name) for ep in episodes]
        groups[agent_type] = values

    if len(groups) < 2:
        return {'error': 'Need at least 2 agent types to compare'}

    # ANOVA F-test
    group_values = list(groups.values())
    f_stat, p_anova = f_oneway(*group_values)

    # Pairwise comparisons
    agent_types = list(groups.keys())
    n_comparisons = len(agent_types) * (len(agent_types) - 1) // 2
    bonferroni_alpha = 0.05 / n_comparisons if n_comparisons > 0 else 0.05

    pairwise = []
    for i, agent1 in enumerate(agent_types):
        for j, agent2 in enumerate(agent_types):
            if i < j:
                vals1 = groups[agent1]
                vals2 = groups[agent2]

                # t-test
                t_stat, p_val = ttest_ind(vals1, vals2)

                # Effect size
                d = compute_cohens_d(vals1, vals2)

                pairwise.append({
                    'agent1': agent1,
                    'agent2': agent2,
                    'mean1': float(np.mean(vals1)),
                    'mean2': float(np.mean(vals2)),
                    't_statistic': float(t_stat),
                    'p_value': float(p_val),
                    'significant': p_val < bonferroni_alpha,
                    'cohens_d': float(d)
                })

    return {
        'metric': metric_name,
        'anova': {
            'f_statistic': float(f_stat),
            'p_value': float(p_anova),
            'significant': p_anova < 0.05
        },
        'pairwise_comparisons': pairwise,
        'bonferroni_alpha': bonferroni_alpha
    }


def generate_summary_report(
    aggregated: Dict[str, AggregatedMetrics]
) -> str:
    """Generate text summary report of metrics.

    Args:
        aggregated: Dictionary mapping agent_type -> AggregatedMetrics

    Returns:
        Formatted text report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("METRICS RECOMPUTATION SUMMARY (from raw logs)")
    lines.append("=" * 80)
    lines.append("")

    for agent_type, metrics in aggregated.items():
        lines.append(f"Agent: {agent_type} ({metrics.environment})")
        lines.append("-" * 80)
        lines.append(f"  Episodes: {metrics.n_episodes}")
        lines.append(f"  Accuracy: {metrics.accuracy_mean:.3f} ± {metrics.accuracy_std:.3f}")
        lines.append(f"            95% CI: [{metrics.accuracy_ci_lower:.3f}, {metrics.accuracy_ci_upper:.3f}]")
        lines.append(f"  Surprisal Slope: {metrics.slope_mean:.4f} ± {metrics.slope_std:.4f}")
        lines.append(f"                   95% CI: [{metrics.slope_ci_lower:.4f}, {metrics.slope_ci_upper:.4f}]")
        lines.append(f"                   p-value: {metrics.slope_p_value:.4f}")

        if metrics.slope_p_value < 0.05:
            direction = "decreasing (learning)" if metrics.slope_mean < 0 else "increasing"
            lines.append(f"                   ** Significant {direction} trend **")

        lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)
