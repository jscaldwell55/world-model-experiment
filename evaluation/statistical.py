# evaluation/statistical.py
import numpy as np
import scipy.stats as stats
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class StatisticalResult:
    """Container for statistical test results"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    ci_lower: float
    ci_upper: float
    power: float
    interpretation: str

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def bootstrap_ci(
    data: np.ndarray, 
    statistic_func=np.mean, 
    n_bootstrap: int = 10000,
    confidence: float = 0.95
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for any statistic.
    """
    bootstrap_stats = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, alpha/2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
    
    return lower, upper

def power_analysis(
    effect_size: float,
    n_per_group: int,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> float:
    """
    Calculate statistical power for independent t-test.
    
    Uses statsmodels if available, otherwise approximation.
    """
    try:
        from statsmodels.stats.power import TTestIndPower
        power_calc = TTestIndPower()
        return power_calc.power(
            effect_size=effect_size,
            nobs1=n_per_group,
            alpha=alpha,
            alternative=alternative
        )
    except ImportError:
        # Approximation using non-central t distribution
        from scipy.stats import nct
        df = 2 * n_per_group - 2
        ncp = effect_size * np.sqrt(n_per_group / 2)
        
        if alternative == 'two-sided':
            t_crit = stats.t.ppf(1 - alpha/2, df)
            power = 1 - nct.cdf(t_crit, df, ncp) + nct.cdf(-t_crit, df, ncp)
        else:
            t_crit = stats.t.ppf(1 - alpha, df)
            power = 1 - nct.cdf(t_crit, df, ncp)
        
        return power

def compare_conditions(
    condition_a: np.ndarray,
    condition_b: np.ndarray,
    condition_a_name: str = "Actor",
    condition_b_name: str = "Observer",
    alpha: float = 0.05
) -> StatisticalResult:
    """
    Compare two conditions with full statistical analysis.
    
    Returns effect size, p-value, CI, and power.
    """
    # T-test
    t_stat, p_value = stats.ttest_ind(condition_a, condition_b)
    
    # Effect size
    effect_size = cohens_d(condition_a, condition_b)
    
    # Bootstrap CI for effect size
    combined = np.concatenate([condition_a, condition_b])
    n_a = len(condition_a)
    
    def effect_size_stat(data):
        return cohens_d(data[:n_a], data[n_a:])
    
    ci_lower, ci_upper = bootstrap_ci(combined, effect_size_stat)
    
    # Post-hoc power
    power = power_analysis(
        effect_size=abs(effect_size),
        n_per_group=min(len(condition_a), len(condition_b)),
        alpha=alpha
    )
    
    # Interpretation
    if p_value < alpha:
        if effect_size > 0:
            interpretation = f"{condition_a_name} significantly outperforms {condition_b_name}"
        else:
            interpretation = f"{condition_b_name} significantly outperforms {condition_a_name}"
    else:
        interpretation = "No significant difference detected"
    
    interpretation += f" (power={power:.2f})"
    
    return StatisticalResult(
        test_name="Independent t-test",
        statistic=t_stat,
        p_value=p_value,
        effect_size=effect_size,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        power=power,
        interpretation=interpretation
    )

def check_preregistration(
    result: StatisticalResult,
    preregistered_effect_size: float,
    preregistered_direction: str
) -> Dict[str, bool]:
    """
    Check if results align with preregistered hypotheses.
    
    Returns:
        - hypothesis_supported: Effect in predicted direction?
        - meets_minimum_effect: Effect size >= preregistered minimum?
        - significant: p < alpha?
    """
    if preregistered_direction == "positive":
        direction_match = result.effect_size > 0
    elif preregistered_direction == "negative":
        direction_match = result.effect_size < 0
    else:
        direction_match = True
    
    return {
        'hypothesis_supported': direction_match and result.p_value < 0.05,
        'meets_minimum_effect': abs(result.effect_size) >= preregistered_effect_size,
        'significant': result.p_value < 0.05,
        'adequate_power': result.power >= 0.80
    }
