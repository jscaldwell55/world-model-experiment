"""
Phase 3 Statistical Validation with Multiple Seeds.

Properly tests significance of Dream State improvements using:
1. Multi-seed evaluation (10 seeds minimum)
2. Paired t-tests with Bonferroni correction
3. Effect size calculation (Cohen's d)
4. Variance decomposition
5. Publication-quality visualizations

This replaces ad-hoc comparisons with rigorous statistical testing.
"""

import json
import logging
import pickle
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from molecular_world_model import MolecularWorldModel
from molecular_consolidation_pipeline import SimplifiedFTB
from dream_state import AnalogGenerator, SARExtractor, DreamPipeline
from nesy_bridge import SemanticMemory, ConsistencyChecker, HybridPredictor

from rdkit import Chem
from rdkit.Chem import Descriptors

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# =============================================================================
# STATISTICAL UTILITIES
# =============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size for paired samples.

    Effect size interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    diff = group1 - group2
    return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0


def paired_ttest_with_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05
) -> Dict:
    """
    Perform paired t-test with 95% CI and effect size.

    Args:
        group1: First group (e.g., FTB MAEs across seeds)
        group2: Second group (e.g., DreamFTB MAEs across seeds)
        alpha: Significance level

    Returns:
        Dict with statistics
    """
    diff = group1 - group2
    n = len(diff)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(group1, group2)

    # Mean difference and CI
    mean_diff = np.mean(diff)
    se = stats.sem(diff)
    ci = stats.t.interval(1 - alpha, n - 1, loc=mean_diff, scale=se)

    # Effect size
    d = cohens_d(group1, group2)

    return {
        'mean_diff': float(mean_diff),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'ci_95': [float(ci[0]), float(ci[1])],
        'cohens_d': float(d),
        'n_pairs': n
    }


def bonferroni_correction(p_values: List[float]) -> List[float]:
    """Apply Bonferroni correction for multiple comparisons."""
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def interpret_effect_size(d: float, p: float, alpha: float = 0.05) -> str:
    """Human-readable interpretation of effect size."""
    if p >= alpha:
        return "no significant difference"

    magnitude = "negligible" if abs(d) < 0.2 else \
                "small" if abs(d) < 0.5 else \
                "medium" if abs(d) < 0.8 else "large"

    # Positive d means group1 > group2
    # For MAE comparison: positive = group1 worse, negative = group1 better
    direction = "improvement" if d > 0 else "degradation"

    return f"significant {direction}, {magnitude} effect"


# =============================================================================
# DATA LOADING
# =============================================================================

def compute_molecular_weight(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol) if mol else 0.0


def load_data_with_condition(
    condition: str,
    data_path: str = 'data/esol_processed.pkl',
    seed: int = 42,
    noise_level: float = 0.15
) -> Dict:
    """Load data with condition-specific preprocessing."""
    rng = np.random.RandomState(seed)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    candidate_df = data['candidate_pool'].copy()
    test_df = data['test_set'].copy()

    all_smiles = candidate_df['smiles'].tolist() + test_df['smiles'].tolist()
    all_labels = candidate_df['logS'].tolist() + test_df['logS'].tolist()

    if condition == 'clean':
        indices = list(range(len(all_smiles)))
        rng.shuffle(indices)
        all_smiles = [all_smiles[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]

    elif condition == 'noisy_15pct':
        indices = list(range(len(all_smiles)))
        rng.shuffle(indices)
        all_smiles = [all_smiles[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]
        label_std = np.std(all_labels)
        noise = rng.normal(0, label_std * noise_level, size=len(all_labels))
        all_labels = [l + n for l, n in zip(all_labels, noise)]

    elif condition == 'distribution_shift':
        mol_weights = [compute_molecular_weight(s) for s in all_smiles]
        sorted_indices = np.argsort(mol_weights)
        all_smiles = [all_smiles[i] for i in sorted_indices]
        all_labels = [all_labels[i] for i in sorted_indices]

    n_candidates = int(0.7 * len(all_smiles))

    return {
        'candidate_pool': pd.DataFrame({
            'smiles': all_smiles[:n_candidates],
            'logS': all_labels[:n_candidates]
        }),
        'test_set': pd.DataFrame({
            'smiles': all_smiles[n_candidates:],
            'logS': all_labels[n_candidates:]
        })
    }


# =============================================================================
# STRATEGY IMPLEMENTATIONS
# =============================================================================

@dataclass
class ExperimentMetrics:
    """Metrics for a single experiment run."""
    final_mae: float = 0.0
    updates_performed: int = 0
    repair_count: int = 0
    total_synthetics: int = 0
    acceptance_rate: float = 0.0
    n_rules: int = 0
    final_consistency: float = 0.0
    mae_history: List[float] = field(default_factory=list)


class BaseStrategy:
    """Base class for learning strategies."""

    def __init__(self, update_interval: int = 10, retention_threshold: float = 0.25):
        self.update_interval = update_interval
        self.retention_threshold = retention_threshold

    def run(self, data: Dict, seed: int, n_steps: int = 50) -> ExperimentMetrics:
        """Run the strategy and return metrics."""
        raise NotImplementedError


class FTBStrategy(BaseStrategy):
    """FTB baseline - no dreams, no memory."""

    name = "FTB"

    def run(self, data: Dict, seed: int, n_steps: int = 50) -> ExperimentMetrics:
        metrics = ExperimentMetrics()

        candidate_df = data['candidate_pool']
        test_df = data['test_set']

        seed_size = 20
        seed_smiles = candidate_df['smiles'].tolist()[:seed_size]
        seed_labels = candidate_df['logS'].tolist()[:seed_size]

        test_smiles = test_df['smiles'].tolist()
        test_labels = np.array(test_df['logS'].tolist())

        probe_smiles = test_smiles[:30]
        probe_labels = test_labels[:30].tolist()

        query_smiles = candidate_df['smiles'].tolist()[seed_size:seed_size + n_steps]
        query_labels = candidate_df['logS'].tolist()[seed_size:seed_size + n_steps]

        # Initialize
        model = MolecularWorldModel(n_estimators=50, random_state=seed)
        accumulated_smiles = list(seed_smiles)
        accumulated_labels = list(seed_labels)
        model.fit(accumulated_smiles, accumulated_labels)

        ftb = SimplifiedFTB(
            world_model=model,
            probe_smiles=probe_smiles,
            probe_labels=probe_labels,
            retention_threshold=self.retention_threshold,
            random_state=seed
        )

        metrics.updates_performed = 1
        batch_smiles = []
        batch_labels = []

        # Record initial MAE
        preds, _ = model.predict(test_smiles)
        valid_mask = ~np.isnan(preds)
        initial_mae = np.mean(np.abs(preds[valid_mask] - test_labels[valid_mask]))
        metrics.mae_history.append(initial_mae)

        # Run steps
        for step, (smiles, label) in enumerate(zip(query_smiles, query_labels)):
            batch_smiles.append(smiles)
            batch_labels.append(label)

            if len(batch_smiles) >= self.update_interval:
                accumulated_smiles.extend(batch_smiles)
                accumulated_labels.extend(batch_labels)

                result = ftb.update(accumulated_smiles, accumulated_labels)
                metrics.updates_performed += 1
                if result['was_repaired']:
                    metrics.repair_count += result['repair_attempts']

                batch_smiles = []
                batch_labels = []

                # Record MAE
                preds, _ = model.predict(test_smiles)
                valid_mask = ~np.isnan(preds)
                mae = np.mean(np.abs(preds[valid_mask] - test_labels[valid_mask]))
                metrics.mae_history.append(mae)

        # Final update
        if batch_smiles:
            accumulated_smiles.extend(batch_smiles)
            accumulated_labels.extend(batch_labels)
            ftb.update(accumulated_smiles, accumulated_labels)
            metrics.updates_performed += 1

        # Final MAE
        preds, _ = model.predict(test_smiles)
        valid_mask = ~np.isnan(preds)
        metrics.final_mae = np.mean(np.abs(preds[valid_mask] - test_labels[valid_mask]))

        return metrics


class DreamFTBStrategy(BaseStrategy):
    """FTB + Dreams strategy."""

    name = "DreamFTB"

    def __init__(self, update_interval: int = 10, retention_threshold: float = 0.25,
                 confidence_threshold: float = 0.85, max_synthetics_ratio: float = 0.3):
        super().__init__(update_interval, retention_threshold)
        self.confidence_threshold = confidence_threshold
        self.max_synthetics_ratio = max_synthetics_ratio

    def run(self, data: Dict, seed: int, n_steps: int = 50, condition: str = 'clean') -> ExperimentMetrics:
        metrics = ExperimentMetrics()

        candidate_df = data['candidate_pool']
        test_df = data['test_set']

        seed_size = 20
        seed_smiles = candidate_df['smiles'].tolist()[:seed_size]
        seed_labels = candidate_df['logS'].tolist()[:seed_size]

        test_smiles = test_df['smiles'].tolist()
        test_labels = np.array(test_df['logS'].tolist())

        probe_smiles = test_smiles[:30]
        probe_labels = test_labels[:30].tolist()

        query_smiles = candidate_df['smiles'].tolist()[seed_size:seed_size + n_steps]
        query_labels = candidate_df['logS'].tolist()[seed_size:seed_size + n_steps]

        # Initialize
        model = MolecularWorldModel(n_estimators=50, random_state=seed)
        accumulated_smiles = list(seed_smiles)
        accumulated_labels = list(seed_labels)
        model.fit(accumulated_smiles, accumulated_labels)

        ftb = SimplifiedFTB(
            world_model=model,
            probe_smiles=probe_smiles,
            probe_labels=probe_labels,
            retention_threshold=self.retention_threshold,
            random_state=seed
        )

        analog_gen = AnalogGenerator(random_state=seed)
        sar_ext = SARExtractor()
        dream_pipeline = DreamPipeline(
            world_model=model,
            analog_generator=analog_gen,
            sar_extractor=sar_ext,
            confidence_threshold=self.confidence_threshold,
            max_synthetics_ratio=self.max_synthetics_ratio,
            random_state=seed
        )

        metrics.updates_performed = 1
        batch_smiles = []
        batch_labels = []
        total_synthetics = 0
        acceptance_rates = []

        # Record initial MAE
        preds, _ = model.predict(test_smiles)
        valid_mask = ~np.isnan(preds)
        initial_mae = np.mean(np.abs(preds[valid_mask] - test_labels[valid_mask]))
        metrics.mae_history.append(initial_mae)

        # Run steps
        for step, (smiles, label) in enumerate(zip(query_smiles, query_labels)):
            batch_smiles.append(smiles)
            batch_labels.append(label)

            if len(batch_smiles) >= self.update_interval:
                accumulated_smiles.extend(batch_smiles)
                accumulated_labels.extend(batch_labels)

                # Dream
                dream_result = dream_pipeline.dream(
                    accumulated_smiles, accumulated_labels, condition
                )
                total_synthetics += dream_result['n_after_cap']
                acceptance_rates.append(dream_result['acceptance_rate'])

                # Combined update
                combined_smiles = accumulated_smiles + dream_result['synthetic_smiles']
                combined_labels = accumulated_labels + dream_result['synthetic_labels']
                combined_weights = [1.0] * len(accumulated_smiles) + dream_result['synthetic_weights']

                result = ftb.update(combined_smiles, combined_labels, combined_weights)
                metrics.updates_performed += 1
                if result['was_repaired']:
                    metrics.repair_count += result['repair_attempts']

                batch_smiles = []
                batch_labels = []

                # Record MAE
                preds, _ = model.predict(test_smiles)
                valid_mask = ~np.isnan(preds)
                mae = np.mean(np.abs(preds[valid_mask] - test_labels[valid_mask]))
                metrics.mae_history.append(mae)

        # Final update
        if batch_smiles:
            accumulated_smiles.extend(batch_smiles)
            accumulated_labels.extend(batch_labels)
            dream_result = dream_pipeline.dream(accumulated_smiles, accumulated_labels, condition)
            combined_smiles = accumulated_smiles + dream_result['synthetic_smiles']
            combined_labels = accumulated_labels + dream_result['synthetic_labels']
            combined_weights = [1.0] * len(accumulated_smiles) + dream_result['synthetic_weights']
            ftb.update(combined_smiles, combined_labels, combined_weights)

        # Final MAE
        preds, _ = model.predict(test_smiles)
        valid_mask = ~np.isnan(preds)
        metrics.final_mae = np.mean(np.abs(preds[valid_mask] - test_labels[valid_mask]))
        metrics.total_synthetics = total_synthetics
        metrics.acceptance_rate = np.mean(acceptance_rates) if acceptance_rates else 0.0

        return metrics


class NeSyFTBStrategy(BaseStrategy):
    """FTB + Dreams + Semantic Memory strategy."""

    name = "NeSyFTB"

    def __init__(self, update_interval: int = 10, retention_threshold: float = 0.25,
                 confidence_threshold: float = 0.85, max_synthetics_ratio: float = 0.3):
        super().__init__(update_interval, retention_threshold)
        self.confidence_threshold = confidence_threshold
        self.max_synthetics_ratio = max_synthetics_ratio

    def run(self, data: Dict, seed: int, n_steps: int = 50, condition: str = 'clean') -> ExperimentMetrics:
        metrics = ExperimentMetrics()

        candidate_df = data['candidate_pool']
        test_df = data['test_set']

        seed_size = 20
        seed_smiles = candidate_df['smiles'].tolist()[:seed_size]
        seed_labels = candidate_df['logS'].tolist()[:seed_size]

        test_smiles = test_df['smiles'].tolist()
        test_labels = np.array(test_df['logS'].tolist())

        probe_smiles = test_smiles[:30]
        probe_labels = test_labels[:30].tolist()

        query_smiles = candidate_df['smiles'].tolist()[seed_size:seed_size + n_steps]
        query_labels = candidate_df['logS'].tolist()[seed_size:seed_size + n_steps]

        # Initialize
        model = MolecularWorldModel(n_estimators=50, random_state=seed)
        accumulated_smiles = list(seed_smiles)
        accumulated_labels = list(seed_labels)
        model.fit(accumulated_smiles, accumulated_labels)

        ftb = SimplifiedFTB(
            world_model=model,
            probe_smiles=probe_smiles,
            probe_labels=probe_labels,
            retention_threshold=self.retention_threshold,
            random_state=seed
        )

        analog_gen = AnalogGenerator(random_state=seed)
        sar_ext = SARExtractor()
        dream_pipeline = DreamPipeline(
            world_model=model,
            analog_generator=analog_gen,
            sar_extractor=sar_ext,
            confidence_threshold=self.confidence_threshold,
            max_synthetics_ratio=self.max_synthetics_ratio,
            random_state=seed
        )

        semantic_memory = SemanticMemory()
        semantic_memory.update_global_stats(list(seed_labels))

        consistency_history = []

        metrics.updates_performed = 1
        batch_smiles = []
        batch_labels = []
        total_synthetics = 0
        acceptance_rates = []
        episode_counter = 0

        # Record initial MAE
        preds, _ = model.predict(test_smiles)
        valid_mask = ~np.isnan(preds)
        initial_mae = np.mean(np.abs(preds[valid_mask] - test_labels[valid_mask]))
        metrics.mae_history.append(initial_mae)

        # Run steps
        for step, (smiles, label) in enumerate(zip(query_smiles, query_labels)):
            batch_smiles.append(smiles)
            batch_labels.append(label)

            if len(batch_smiles) >= self.update_interval:
                episode_counter += 1
                accumulated_smiles.extend(batch_smiles)
                accumulated_labels.extend(batch_labels)

                # Dream
                dream_result = dream_pipeline.dream(
                    accumulated_smiles, accumulated_labels, condition
                )
                total_synthetics += dream_result['n_after_cap']
                acceptance_rates.append(dream_result['acceptance_rate'])

                # Ingest SAR rules
                if dream_result['sar_rules']:
                    semantic_memory.ingest_rules(
                        dream_result['sar_rules'],
                        episode_id=f"episode_{episode_counter}"
                    )

                # Check consistency
                if len(semantic_memory) > 0:
                    consistency_checker = ConsistencyChecker(model, semantic_memory)
                    result = consistency_checker.check_rule_consistency(
                        accumulated_smiles, accumulated_labels
                    )
                    consistency_history.append(result['overall_consistency'])

                # Combined update
                combined_smiles = accumulated_smiles + dream_result['synthetic_smiles']
                combined_labels = accumulated_labels + dream_result['synthetic_labels']
                combined_weights = [1.0] * len(accumulated_smiles) + dream_result['synthetic_weights']

                semantic_memory.update_global_stats(batch_labels)

                result = ftb.update(combined_smiles, combined_labels, combined_weights)
                metrics.updates_performed += 1
                if result['was_repaired']:
                    metrics.repair_count += result['repair_attempts']

                batch_smiles = []
                batch_labels = []

                # Record MAE
                preds, _ = model.predict(test_smiles)
                valid_mask = ~np.isnan(preds)
                mae = np.mean(np.abs(preds[valid_mask] - test_labels[valid_mask]))
                metrics.mae_history.append(mae)

        # Final update
        if batch_smiles:
            accumulated_smiles.extend(batch_smiles)
            accumulated_labels.extend(batch_labels)
            dream_result = dream_pipeline.dream(accumulated_smiles, accumulated_labels, condition)
            if dream_result['sar_rules']:
                semantic_memory.ingest_rules(dream_result['sar_rules'])
            combined_smiles = accumulated_smiles + dream_result['synthetic_smiles']
            combined_labels = accumulated_labels + dream_result['synthetic_labels']
            combined_weights = [1.0] * len(accumulated_smiles) + dream_result['synthetic_weights']
            ftb.update(combined_smiles, combined_labels, combined_weights)

        # Final MAE
        preds, _ = model.predict(test_smiles)
        valid_mask = ~np.isnan(preds)
        metrics.final_mae = np.mean(np.abs(preds[valid_mask] - test_labels[valid_mask]))
        metrics.total_synthetics = total_synthetics
        metrics.acceptance_rate = np.mean(acceptance_rates) if acceptance_rates else 0.0
        metrics.n_rules = len(semantic_memory)
        metrics.final_consistency = consistency_history[-1] if consistency_history else 0.0

        return metrics


# =============================================================================
# MULTI-SEED EXPERIMENT RUNNER
# =============================================================================

def run_single_seed(
    seed: int,
    condition: str,
    strategy_name: str,
    n_steps: int = 50,
    confidence_threshold: float = 0.85
) -> Dict:
    """Run one strategy on one condition with one seed."""
    data = load_data_with_condition(condition, seed=seed)

    if strategy_name == "FTB":
        strategy = FTBStrategy()
        metrics = strategy.run(data, seed, n_steps)
    elif strategy_name == "DreamFTB":
        strategy = DreamFTBStrategy(confidence_threshold=confidence_threshold)
        metrics = strategy.run(data, seed, n_steps, condition)
    elif strategy_name == "NeSyFTB":
        strategy = NeSyFTBStrategy(confidence_threshold=confidence_threshold)
        metrics = strategy.run(data, seed, n_steps, condition)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return {
        'seed': seed,
        'condition': condition,
        'strategy': strategy_name,
        'final_mae': metrics.final_mae,
        'updates': metrics.updates_performed,
        'repairs': metrics.repair_count,
        'synthetics': metrics.total_synthetics,
        'acceptance_rate': metrics.acceptance_rate,
        'n_rules': metrics.n_rules,
        'consistency': metrics.final_consistency,
        'mae_history': metrics.mae_history
    }


def run_multi_seed_experiment(
    seeds: List[int],
    conditions: List[str],
    strategies: List[str],
    n_steps: int = 50,
    confidence_threshold: float = 0.85,
    show_progress: bool = True
) -> Dict:
    """Run all combinations of strategies x conditions x seeds."""
    results = {}
    total = len(strategies) * len(conditions) * len(seeds)

    iterator = tqdm(total=total, desc="Running experiments") if show_progress else None

    for strategy in strategies:
        for condition in conditions:
            for seed in seeds:
                key = f"{strategy}_{condition}_{seed}"
                try:
                    results[key] = run_single_seed(seed, condition, strategy, n_steps, confidence_threshold)
                except Exception as e:
                    logger.warning(f"Failed {key}: {e}")
                    results[key] = {
                        'seed': seed, 'condition': condition, 'strategy': strategy,
                        'final_mae': np.nan, 'error': str(e)
                    }

                if iterator:
                    iterator.update(1)

    if iterator:
        iterator.close()

    return results


def compute_statistical_significance(
    results: Dict,
    strategy_a: str,
    strategy_b: str,
    condition: str,
    seeds: List[int]
) -> Dict:
    """Compute paired t-test between two strategies."""
    # Extract MAEs paired by seed
    mae_a = np.array([results[f"{strategy_a}_{condition}_{seed}"]['final_mae']
                      for seed in seeds])
    mae_b = np.array([results[f"{strategy_b}_{condition}_{seed}"]['final_mae']
                      for seed in seeds])

    # Remove NaN pairs
    valid = ~(np.isnan(mae_a) | np.isnan(mae_b))
    mae_a = mae_a[valid]
    mae_b = mae_b[valid]

    if len(mae_a) < 3:
        return {
            'comparison': f"{strategy_a} vs {strategy_b}",
            'condition': condition,
            'error': 'Insufficient valid pairs',
            'n_valid': len(mae_a)
        }

    stats_result = paired_ttest_with_ci(mae_a, mae_b)

    # Interpretation
    interpretation = interpret_effect_size(stats_result['cohens_d'], stats_result['p_value'])

    # Significance markers
    p = stats_result['p_value']
    if p < 0.001:
        sig = "***"
    elif p < 0.01:
        sig = "**"
    elif p < 0.05:
        sig = "*"
    else:
        sig = "ns"

    return {
        'comparison': f"{strategy_a} vs {strategy_b}",
        'condition': condition,
        'strategy_a_mean': float(np.mean(mae_a)),
        'strategy_a_std': float(np.std(mae_a)),
        'strategy_b_mean': float(np.mean(mae_b)),
        'strategy_b_std': float(np.std(mae_b)),
        **stats_result,
        'significance_marker': sig,
        'interpretation': interpretation
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_boxplots(
    results: Dict,
    conditions: List[str],
    strategies: List[str],
    seeds: List[int],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create box plots showing MAE distribution across seeds."""
    # Prepare data for plotting
    plot_data = []
    for condition in conditions:
        for strategy in strategies:
            for seed in seeds:
                key = f"{strategy}_{condition}_{seed}"
                if key in results and 'final_mae' in results[key]:
                    mae = results[key]['final_mae']
                    if not np.isnan(mae):
                        plot_data.append({
                            'Condition': condition,
                            'Strategy': strategy,
                            'MAE': mae
                        })

    df = pd.DataFrame(plot_data)

    # Create plot
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 6))
    if len(conditions) == 1:
        axes = [axes]

    for ax, condition in zip(axes, conditions):
        condition_df = df[df['Condition'] == condition]

        # Box plot
        sns.boxplot(data=condition_df, x='Strategy', y='MAE', ax=ax,
                   palette='Set2', width=0.6)

        # Add individual points
        sns.stripplot(data=condition_df, x='Strategy', y='MAE', ax=ax,
                     color='black', alpha=0.5, size=4)

        ax.set_title(f'{condition.replace("_", " ").title()}')
        ax.set_xlabel('')
        ax.set_ylabel('Mean Absolute Error' if ax == axes[0] else '')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved boxplots to {save_path}")

    return fig


def create_violin_plots(
    results: Dict,
    conditions: List[str],
    strategies: List[str],
    seeds: List[int],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create violin plots showing full distribution."""
    plot_data = []
    for condition in conditions:
        for strategy in strategies:
            for seed in seeds:
                key = f"{strategy}_{condition}_{seed}"
                if key in results and 'final_mae' in results[key]:
                    mae = results[key]['final_mae']
                    if not np.isnan(mae):
                        plot_data.append({
                            'Condition': condition,
                            'Strategy': strategy,
                            'MAE': mae
                        })

    df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Combined x-axis
    df['Group'] = df['Condition'] + '\n' + df['Strategy']

    sns.violinplot(data=df, x='Condition', y='MAE', hue='Strategy',
                  ax=ax, palette='Set2', inner='box', cut=0)

    ax.set_xlabel('')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('MAE Distribution by Condition and Strategy')
    ax.legend(title='Strategy', loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved violin plots to {save_path}")

    return fig


def create_significance_heatmap(
    comparisons: List[Dict],
    conditions: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create heatmap of effect sizes with significance markers."""
    # Build matrix
    comparison_names = sorted(set(c['comparison'] for c in comparisons))
    data = np.zeros((len(comparison_names), len(conditions)))
    annotations = [['' for _ in conditions] for _ in comparison_names]

    for comp in comparisons:
        i = comparison_names.index(comp['comparison'])
        j = conditions.index(comp['condition'])
        if 'cohens_d' in comp:
            data[i, j] = comp['cohens_d']
            annotations[i][j] = f"{comp['cohens_d']:.2f}{comp.get('significance_marker', '')}"

    fig, ax = plt.subplots(figsize=(10, 6))

    # Heatmap
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-1.5, vmax=1.5)

    # Labels
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([c.replace('_', '\n') for c in conditions])
    ax.set_yticks(range(len(comparison_names)))
    ax.set_yticklabels(comparison_names)

    # Annotations
    for i in range(len(comparison_names)):
        for j in range(len(conditions)):
            ax.text(j, i, annotations[i][j], ha='center', va='center',
                   fontsize=10, fontweight='bold')

    plt.colorbar(im, ax=ax, label="Cohen's d (positive = first strategy better)")

    ax.set_title("Effect Sizes: Strategy Comparisons\n(*p<0.05, **p<0.01, ***p<0.001)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")

    return fig


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_statistical_validation(
    seeds: Optional[List[int]] = None,
    conditions: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
    n_steps: int = 50,
    confidence_threshold: float = 0.85,
    output_dir: str = 'results/statistical_validation'
) -> Dict:
    """Run full statistical validation experiment."""

    if seeds is None:
        seeds = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
    if conditions is None:
        conditions = ['clean', 'noisy_15pct', 'distribution_shift']
    if strategies is None:
        strategies = ['FTB', 'DreamFTB', 'NeSyFTB']

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'plots').mkdir(exist_ok=True)

    print("=" * 70)
    print("STATISTICAL VALIDATION EXPERIMENT")
    print("=" * 70)
    print(f"Seeds: {seeds}")
    print(f"Conditions: {conditions}")
    print(f"Strategies: {strategies}")
    print(f"Steps per run: {n_steps}")
    print(f"Confidence threshold: {confidence_threshold}")
    print("=" * 70)

    # Warn if few seeds
    if len(seeds) < 5:
        warnings.warn("Fewer than 5 seeds - statistical power is low")

    # Run experiments
    print("\n1. Running multi-seed experiments...")
    results = run_multi_seed_experiment(seeds, conditions, strategies, n_steps, confidence_threshold)

    # Save seed-level results
    with open(output_path / 'seed_level_results.json', 'w') as f:
        # Convert numpy types
        results_serializable = {}
        for k, v in results.items():
            results_serializable[k] = {
                kk: (float(vv) if isinstance(vv, (np.floating, float)) else
                     int(vv) if isinstance(vv, (np.integer, int)) else
                     [float(x) for x in vv] if isinstance(vv, (list, np.ndarray)) and len(vv) > 0 and isinstance(vv[0], (float, np.floating)) else
                     vv)
                for kk, vv in v.items()
            }
        json.dump(results_serializable, f, indent=2)

    # Compute statistical comparisons
    print("\n2. Computing statistical significance...")
    comparisons = []

    for condition in conditions:
        # FTB vs DreamFTB
        comp = compute_statistical_significance(results, 'FTB', 'DreamFTB', condition, seeds)
        comparisons.append(comp)

        # DreamFTB vs NeSyFTB
        comp = compute_statistical_significance(results, 'DreamFTB', 'NeSyFTB', condition, seeds)
        comparisons.append(comp)

    # Apply Bonferroni correction
    p_values = [c.get('p_value', 1.0) for c in comparisons]
    p_adjusted = bonferroni_correction(p_values)

    for i, comp in enumerate(comparisons):
        if 'p_value' in comp:
            comp['p_value_adjusted'] = p_adjusted[i]
            # Update interpretation with adjusted p
            comp['interpretation_adjusted'] = interpret_effect_size(
                comp['cohens_d'], p_adjusted[i]
            )

    # Save comparisons
    with open(output_path / 'significance_tests.json', 'w') as f:
        json.dump(comparisons, f, indent=2, default=float)

    # Create visualizations
    print("\n3. Creating visualizations...")

    create_boxplots(results, conditions, strategies, seeds,
                   save_path=str(output_path / 'plots' / 'boxplots_by_strategy.png'))
    plt.close()

    create_violin_plots(results, conditions, strategies, seeds,
                       save_path=str(output_path / 'plots' / 'violin_plots.png'))
    plt.close()

    create_significance_heatmap(comparisons, conditions,
                               save_path=str(output_path / 'plots' / 'significance_comparison.png'))
    plt.close()

    # Compute summary statistics
    print("\n4. Computing summary statistics...")
    summary = {}

    for condition in conditions:
        summary[condition] = {}
        for strategy in strategies:
            maes = [results[f"{strategy}_{condition}_{seed}"]['final_mae']
                   for seed in seeds
                   if f"{strategy}_{condition}_{seed}" in results]
            maes = [m for m in maes if not np.isnan(m)]

            if maes:
                summary[condition][strategy] = {
                    'mae_mean': float(np.mean(maes)),
                    'mae_std': float(np.std(maes)),
                    'mae_median': float(np.median(maes)),
                    'mae_min': float(np.min(maes)),
                    'mae_max': float(np.max(maes)),
                    'n_valid': len(maes)
                }

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for condition in conditions:
        print(f"\n{condition.upper()}")
        print("-" * 60)
        print(f"{'Strategy':<12} {'MAE (mean±std)':<20} {'Median':<10} {'Range':<20}")
        print("-" * 60)

        for strategy in strategies:
            if strategy in summary[condition]:
                s = summary[condition][strategy]
                mae_str = f"{s['mae_mean']:.4f} ± {s['mae_std']:.4f}"
                range_str = f"[{s['mae_min']:.4f}, {s['mae_max']:.4f}]"
                print(f"{strategy:<12} {mae_str:<20} {s['mae_median']:<10.4f} {range_str:<20}")

    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 70)

    for comp in comparisons:
        if 'error' in comp:
            print(f"\n{comp['comparison']} on {comp['condition']}: {comp['error']}")
            continue

        print(f"\n{comp['comparison']} on {comp['condition']}:")
        print(f"  Mean difference: {comp['mean_diff']:.4f}")
        print(f"  95% CI: [{comp['ci_95'][0]:.4f}, {comp['ci_95'][1]:.4f}]")
        print(f"  p-value (raw): {comp['p_value']:.4f}")
        print(f"  p-value (Bonferroni): {comp.get('p_value_adjusted', 'N/A'):.4f}")
        print(f"  Cohen's d: {comp['cohens_d']:.3f}")
        print(f"  Interpretation: {comp.get('interpretation_adjusted', comp['interpretation'])}")

    # Final output
    final_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'seeds': seeds,
            'conditions': conditions,
            'strategies': strategies,
            'n_steps': n_steps,
            'confidence_threshold': confidence_threshold
        },
        'summary': summary,
        'comparisons': comparisons,
        'seed_level_results_path': str(output_path / 'seed_level_results.json')
    }

    with open(output_path / 'validation_summary.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=float)

    print(f"\n\nResults saved to: {output_path}")

    return final_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run statistical validation")
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021],
                       help='Random seeds to use')
    parser.add_argument('--conditions', nargs='+',
                       default=['clean', 'noisy_15pct', 'distribution_shift'],
                       help='Data conditions to test')
    parser.add_argument('--n-steps', type=int, default=50,
                       help='Number of steps per experiment')
    parser.add_argument('--confidence-threshold', type=float, default=0.85,
                       help='Dream confidence threshold (default: 0.85, optimal: 0.70)')
    parser.add_argument('--output-dir', default='results/statistical_validation',
                       help='Output directory')

    args = parser.parse_args()

    results = run_statistical_validation(
        seeds=args.seeds,
        conditions=args.conditions,
        n_steps=args.n_steps,
        confidence_threshold=args.confidence_threshold,
        output_dir=args.output_dir
    )
