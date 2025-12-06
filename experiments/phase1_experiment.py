"""
Phase 1 Experiment: Testing Offline Consolidation on Molecular Property Prediction

Experimental Conditions:
1. Static Model + Random Policy (baseline)
2. Static Model + Uncertainty Sampling (baseline)
3. Online Updates + Uncertainty Sampling (no OC gate)
4. Full Stack (OC + FTB) + Uncertainty Sampling
5. Oracle Upper Bound

Measures whether OC architecture provides value for molecular world models.
"""

import copy
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from molecular_world_model import MolecularWorldModel, load_esol_data
from molecular_design_env import MolecularDesignEnv
from molecular_consolidation_pipeline import MolecularConsolidationPipeline
from molecular_oc_adapter import MolecularOCAdapter


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for Phase 1 experiment."""
    # Data settings
    seed_size: int = 10  # Initial molecules to seed world model
    query_budget: int = 50  # Total queries per episode

    # Model settings
    n_estimators: int = 50  # RF trees (reduced for speed)
    n_scaffold_clusters: int = 20
    n_mw_bins: int = 5
    n_logp_bins: int = 5

    # Consolidation settings
    consolidation_interval: int = 10  # Consolidate every N queries
    retention_threshold: float = 0.25
    cv_threshold: float = 0.20

    # Experiment settings
    n_seeds: int = 5
    checkpoint_steps: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50])

    # Output
    output_dir: str = 'results/phase1'


# ============================================================================
# POLICIES
# ============================================================================

class BasePolicy:
    """Base class for action selection policies."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def select(self, state: Dict) -> int:
        raise NotImplementedError

    def reset(self, seed: int):
        self.rng = np.random.RandomState(seed)


class RandomPolicy(BasePolicy):
    """Select random unqueried molecule each step."""

    def select(self, state: Dict) -> int:
        unqueried = state['unqueried_indices']
        return self.rng.choice(unqueried)


class UncertaintySamplingPolicy(BasePolicy):
    """Select molecule with highest uncertainty."""

    def select(self, state: Dict) -> int:
        unqueried = state['unqueried_indices']
        uncertainties = state['uncertainties']

        # Get uncertainties for unqueried molecules
        unqueried_uncerts = [(i, uncertainties[i]) for i in unqueried]

        # Filter out NaN
        valid = [(i, u) for i, u in unqueried_uncerts if not np.isnan(u)]

        if not valid:
            return self.rng.choice(unqueried)

        # Select highest uncertainty
        return max(valid, key=lambda x: x[1])[0]


class ContextBalancedPolicy(BasePolicy):
    """
    Balance exploration across contexts.

    Selects from least-queried context, breaking ties with uncertainty.
    """

    def __init__(self, seed: int = 42, exploration_weight: float = 0.5):
        super().__init__(seed)
        self.exploration_weight = exploration_weight

    def select(self, state: Dict) -> int:
        unqueried = state['unqueried_indices']
        uncertainties = state['uncertainties']
        contexts = state['contexts']
        context_counts = state['context_counts']

        # Group unqueried by context
        context_to_indices = {}
        for i in unqueried:
            ctx = contexts[i]
            ctx_key = tuple(ctx) if isinstance(ctx, (list, tuple)) else ctx
            if ctx_key not in context_to_indices:
                context_to_indices[ctx_key] = []
            context_to_indices[ctx_key].append(i)

        # Score each molecule: uncertainty + exploration bonus
        best_idx = None
        best_score = -float('inf')

        for ctx, indices in context_to_indices.items():
            count = context_counts.get(ctx, 0)
            exploration_bonus = 1.0 / np.sqrt(1 + count)

            for i in indices:
                unc = uncertainties[i]
                if np.isnan(unc):
                    continue

                score = unc + self.exploration_weight * exploration_bonus

                if score > best_score:
                    best_score = score
                    best_idx = i

        return best_idx if best_idx is not None else self.rng.choice(unqueried)


# ============================================================================
# EXPERIMENTAL CONDITIONS
# ============================================================================

class ExperimentCondition:
    """Base class for experimental conditions."""

    name: str = "base"
    description: str = ""

    def __init__(self, config: ExperimentConfig, seed: int):
        self.config = config
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def get_policy(self) -> BasePolicy:
        raise NotImplementedError

    def should_update_model(self, step: int) -> bool:
        """Whether to update model at this step."""
        return False

    def update_model(self, world_model: MolecularWorldModel, env: MolecularDesignEnv,
                     pipeline: Optional[MolecularConsolidationPipeline]) -> Dict:
        """Update model and return metrics."""
        return {}


class Condition1_StaticRandom(ExperimentCondition):
    """Static Model + Random Policy (baseline)."""

    name = "static_random"
    description = "Static model with random policy (baseline)"

    def get_policy(self) -> BasePolicy:
        return RandomPolicy(seed=self.seed)


class Condition2_StaticUncertainty(ExperimentCondition):
    """Static Model + Uncertainty Sampling (baseline)."""

    name = "static_uncertainty"
    description = "Static model with uncertainty sampling (baseline)"

    def get_policy(self) -> BasePolicy:
        return UncertaintySamplingPolicy(seed=self.seed)


class Condition3_OnlineNoOC(ExperimentCondition):
    """Online Updates + Uncertainty Sampling (no OC gate)."""

    name = "online_no_oc"
    description = "Online updates without OC (ablation)"

    def get_policy(self) -> BasePolicy:
        return UncertaintySamplingPolicy(seed=self.seed)

    def should_update_model(self, step: int) -> bool:
        # Update every step
        return True

    def update_model(self, world_model: MolecularWorldModel, env: MolecularDesignEnv,
                     pipeline: Optional[MolecularConsolidationPipeline]) -> Dict:
        """Naive online update: retrain on all queried data."""
        queried_indices = list(env.queried_indices)
        smiles = [env.all_smiles[i] for i in queried_indices]
        labels = [env.oracle[i] for i in queried_indices]

        if len(smiles) >= 5:  # Minimum data for training
            world_model.fit(smiles, labels)

        return {'update_type': 'online', 'n_samples': len(smiles)}


class Condition4_FullStack(ExperimentCondition):
    """Full Stack (OC + FTB) + Uncertainty Sampling."""

    name = "full_stack_oc"
    description = "Full OC+FTB stack with uncertainty sampling"

    def get_policy(self) -> BasePolicy:
        return ContextBalancedPolicy(seed=self.seed)

    def should_update_model(self, step: int) -> bool:
        # Update at consolidation intervals, but only after enough steps
        # Require at least 20 steps before first consolidation
        return (step + 1) >= 20 and (step + 1) % self.config.consolidation_interval == 0

    def update_model(self, world_model: MolecularWorldModel, env: MolecularDesignEnv,
                     pipeline: Optional[MolecularConsolidationPipeline]) -> Dict:
        """Update via consolidation pipeline."""
        if pipeline is None:
            return {'error': 'No pipeline provided'}

        episode = env.get_episode()
        result = pipeline.consolidate([episode])

        return {
            'update_type': 'consolidation',
            'gate_status': result.gate_status,
            'ftb_triggered': result.ftb_triggered,
            'retention_score': result.retention_score if result.ftb_triggered else None
        }


class Condition5_Oracle(ExperimentCondition):
    """Oracle Upper Bound - train on all data."""

    name = "oracle"
    description = "Oracle trained on all candidate data (upper bound)"

    def get_policy(self) -> BasePolicy:
        return RandomPolicy(seed=self.seed)  # Doesn't matter, we don't run queries


# ============================================================================
# METRICS
# ============================================================================

@dataclass
class StepMetrics:
    """Metrics at a single step."""
    step: int
    test_mae: float
    test_rmse: float
    test_r2: float
    calibration_corr: float
    n_contexts_covered: int
    n_unique_contexts: int
    pool_uncertainty: float


@dataclass
class RunResult:
    """Result from a single experimental run."""
    condition: str
    seed: int
    step_metrics: List[StepMetrics]
    final_metrics: Dict
    consolidation_metrics: List[Dict] = field(default_factory=list)
    total_time: float = 0.0


def compute_metrics(world_model: MolecularWorldModel,
                    test_smiles: List[str],
                    test_labels: List[float],
                    env: Optional[MolecularDesignEnv] = None,
                    step: int = 0) -> StepMetrics:
    """Compute all metrics at current state."""
    # Get calibration metrics from world model
    cal_metrics = world_model.get_calibration_metrics(test_smiles, test_labels)

    # Context coverage (from env if available)
    n_contexts = 0
    n_unique = 0
    pool_unc = 0.0

    if env is not None and env.episode is not None:
        n_contexts = len(env.episode.contexts_covered)
        n_unique = len(set(env._contexts_cache)) if env._contexts_cache else 0
        pool_unc = env.get_pool_uncertainty()

    return StepMetrics(
        step=step,
        test_mae=cal_metrics.get('mae', float('inf')),
        test_rmse=cal_metrics.get('rmse', float('inf')),
        test_r2=cal_metrics.get('r2', 0.0),
        calibration_corr=cal_metrics.get('uncertainty_error_correlation', 0.0),
        n_contexts_covered=n_contexts,
        n_unique_contexts=n_unique,
        pool_uncertainty=pool_unc
    )


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class Phase1Experiment:
    """Main experiment runner for Phase 1."""

    CONDITIONS = [
        Condition1_StaticRandom,
        Condition2_StaticUncertainty,
        Condition3_OnlineNoOC,
        Condition4_FullStack,
        Condition5_Oracle
    ]

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.data = load_esol_data('data/esol_processed.pkl')
        self.candidate_pool = self.data['candidate_pool']
        self.test_set = self.data['test_set']

        self.test_smiles = self.test_set['smiles'].tolist()
        self.test_labels = self.test_set['logS'].tolist()

        # Results storage
        self.results: Dict[str, List[RunResult]] = {
            cond.name: [] for cond in self.CONDITIONS
        }

    def create_world_model(self, seed: int) -> MolecularWorldModel:
        """Create a fresh world model."""
        return MolecularWorldModel(
            n_scaffold_clusters=self.config.n_scaffold_clusters,
            n_mw_bins=self.config.n_mw_bins,
            n_logp_bins=self.config.n_logp_bins,
            n_estimators=self.config.n_estimators,
            random_state=seed
        )

    def create_env(self, world_model: MolecularWorldModel, seed: int) -> MolecularDesignEnv:
        """Create a fresh environment."""
        return MolecularDesignEnv(
            candidate_pool=self.candidate_pool,
            test_set=self.test_set,
            world_model=world_model,
            query_budget=self.config.query_budget,
            reward_type='hybrid',
            random_state=seed
        )

    def create_pipeline(self, world_model: MolecularWorldModel, seed: int) -> MolecularConsolidationPipeline:
        """Create consolidation pipeline."""
        return MolecularConsolidationPipeline(
            world_model=world_model,
            test_smiles=self.test_smiles,
            test_labels=self.test_labels,
            probe_size=50,
            retention_threshold=self.config.retention_threshold,
            cv_threshold=self.config.cv_threshold,
            output_dir=str(self.output_dir / 'consolidation'),
            random_state=seed
        )

    def run_oracle(self, seed: int) -> RunResult:
        """Run oracle condition (train on all data)."""
        import time
        start_time = time.time()

        # Train on ALL candidate pool
        world_model = self.create_world_model(seed)
        all_smiles = self.candidate_pool['smiles'].tolist()
        all_labels = self.candidate_pool['logS'].tolist()
        world_model.fit(all_smiles, all_labels)

        # Evaluate
        metrics = compute_metrics(world_model, self.test_smiles, self.test_labels, step=0)

        # Create result with same metrics at all checkpoints
        step_metrics = [
            StepMetrics(
                step=s,
                test_mae=metrics.test_mae,
                test_rmse=metrics.test_rmse,
                test_r2=metrics.test_r2,
                calibration_corr=metrics.calibration_corr,
                n_contexts_covered=0,
                n_unique_contexts=0,
                pool_uncertainty=0.0
            )
            for s in self.config.checkpoint_steps
        ]

        return RunResult(
            condition='oracle',
            seed=seed,
            step_metrics=step_metrics,
            final_metrics={
                'test_mae': metrics.test_mae,
                'test_rmse': metrics.test_rmse,
                'test_r2': metrics.test_r2,
                'calibration_corr': metrics.calibration_corr
            },
            total_time=time.time() - start_time
        )

    def run_condition(self, condition_class, seed: int, verbose: bool = False) -> RunResult:
        """Run a single condition with given seed."""
        import time
        start_time = time.time()

        # Handle oracle separately
        if condition_class == Condition5_Oracle:
            return self.run_oracle(seed)

        # Initialize
        condition = condition_class(self.config, seed)
        world_model = self.create_world_model(seed)
        env = self.create_env(world_model, seed)
        policy = condition.get_policy()

        # Create pipeline only for full stack
        pipeline = None
        if condition_class == Condition4_FullStack:
            pipeline = self.create_pipeline(world_model, seed)

        # Reset environment with seed data
        state = env.reset(seed_size=self.config.seed_size)

        # Fit initial model on seed data
        seed_indices = list(env.queried_indices)
        seed_smiles = [env.all_smiles[i] for i in seed_indices]
        seed_labels = [env.oracle[i] for i in seed_indices]
        world_model.fit(seed_smiles, seed_labels)

        # Invalidate env cache after model update
        env._invalidate_cache()

        # Track metrics
        step_metrics = []
        consolidation_metrics = []

        # Run queries
        for step in range(self.config.query_budget):
            # Get current state
            state = env.get_state()

            # Select action
            action = policy.select(state)

            # Execute step
            obs, reward, done, info = env.step(action)

            # Model update if applicable
            if condition.should_update_model(step):
                update_result = condition.update_model(world_model, env, pipeline)

                if condition_class == Condition4_FullStack and update_result:
                    consolidation_metrics.append(update_result)

                # Invalidate cache after update
                env._invalidate_cache()

            # Checkpoint metrics
            current_step = step + 1
            if current_step in self.config.checkpoint_steps:
                metrics = compute_metrics(
                    world_model, self.test_smiles, self.test_labels, env, current_step
                )
                step_metrics.append(metrics)

                if verbose:
                    print(f"  Step {current_step}: MAE={metrics.test_mae:.4f}, R²={metrics.test_r2:.4f}")

            if done:
                break

        # Final metrics
        final_metrics = {
            'test_mae': step_metrics[-1].test_mae if step_metrics else float('inf'),
            'test_rmse': step_metrics[-1].test_rmse if step_metrics else float('inf'),
            'test_r2': step_metrics[-1].test_r2 if step_metrics else 0.0,
            'calibration_corr': step_metrics[-1].calibration_corr if step_metrics else 0.0,
            'n_contexts': step_metrics[-1].n_contexts_covered if step_metrics else 0
        }

        return RunResult(
            condition=condition.name,
            seed=seed,
            step_metrics=step_metrics,
            final_metrics=final_metrics,
            consolidation_metrics=consolidation_metrics,
            total_time=time.time() - start_time
        )

    def run_all(self, verbose: bool = True) -> Dict[str, List[RunResult]]:
        """Run all conditions with all seeds."""
        total_runs = len(self.CONDITIONS) * self.config.n_seeds
        current_run = 0

        for condition_class in self.CONDITIONS:
            condition_name = condition_class.name

            if verbose:
                print(f"\n{'='*70}")
                print(f"Condition: {condition_class.description}")
                print(f"{'='*70}")

            for seed in range(self.config.n_seeds):
                current_run += 1

                if verbose:
                    print(f"\n  Seed {seed+1}/{self.config.n_seeds} ({current_run}/{total_runs})")

                result = self.run_condition(condition_class, seed, verbose=verbose)
                self.results[condition_name].append(result)

                if verbose:
                    print(f"    Final MAE: {result.final_metrics['test_mae']:.4f}")
                    print(f"    Time: {result.total_time:.1f}s")

        return self.results

    def save_results(self, filepath: Optional[str] = None):
        """Save results to JSON."""
        if filepath is None:
            filepath = self.output_dir / 'results.json'

        # Convert to serializable format
        def to_dict(obj):
            if hasattr(obj, '__dict__'):
                return {k: to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [to_dict(v) for v in obj]
            elif isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        data = {
            'config': to_dict(self.config),
            'results': to_dict(self.results),
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to {filepath}")

    def analyze_results(self) -> Dict:
        """Analyze results and compute statistics."""
        analysis = {}

        for condition_name, runs in self.results.items():
            if not runs:
                continue

            # Aggregate final metrics
            maes = [r.final_metrics['test_mae'] for r in runs]
            rmses = [r.final_metrics['test_rmse'] for r in runs]
            r2s = [r.final_metrics['test_r2'] for r in runs]
            calibrations = [r.final_metrics['calibration_corr'] for r in runs]

            analysis[condition_name] = {
                'mae_mean': np.mean(maes),
                'mae_std': np.std(maes),
                'rmse_mean': np.mean(rmses),
                'rmse_std': np.std(rmses),
                'r2_mean': np.mean(r2s),
                'r2_std': np.std(r2s),
                'calibration_mean': np.mean(calibrations),
                'calibration_std': np.std(calibrations),
                'n_runs': len(runs)
            }

            # Learning curve at each checkpoint
            checkpoints = {}
            for step in self.config.checkpoint_steps:
                step_maes = []
                for run in runs:
                    for sm in run.step_metrics:
                        if sm.step == step:
                            step_maes.append(sm.test_mae)
                            break

                if step_maes:
                    checkpoints[step] = {
                        'mae_mean': np.mean(step_maes),
                        'mae_std': np.std(step_maes)
                    }

            analysis[condition_name]['checkpoints'] = checkpoints

        return analysis

    def compute_significance(self) -> Dict:
        """Compute statistical significance between conditions."""
        significance = {}

        # Compare Condition 4 (full stack) vs Condition 2 (static uncertainty)
        if 'full_stack_oc' in self.results and 'static_uncertainty' in self.results:
            full_maes = [r.final_metrics['test_mae'] for r in self.results['full_stack_oc']]
            static_maes = [r.final_metrics['test_mae'] for r in self.results['static_uncertainty']]

            if len(full_maes) == len(static_maes):
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(full_maes, static_maes)
                significance['full_vs_static'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'full_better': np.mean(full_maes) < np.mean(static_maes)
                }

        # Compare Condition 4 vs Condition 3 (online no OC)
        if 'full_stack_oc' in self.results and 'online_no_oc' in self.results:
            full_maes = [r.final_metrics['test_mae'] for r in self.results['full_stack_oc']]
            online_maes = [r.final_metrics['test_mae'] for r in self.results['online_no_oc']]

            if len(full_maes) == len(online_maes):
                t_stat, p_value = stats.ttest_rel(full_maes, online_maes)
                significance['full_vs_online'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'full_better': np.mean(full_maes) < np.mean(online_maes)
                }

        return significance

    def print_summary(self):
        """Print human-readable summary."""
        analysis = self.analyze_results()
        significance = self.compute_significance()

        print("\n" + "="*80)
        print("PHASE 1 EXPERIMENT RESULTS")
        print("="*80)

        # Results table
        print(f"\n{'Condition':<25} {'MAE':>15} {'RMSE':>15} {'R²':>15} {'Calibration':>15}")
        print("-"*80)

        for cond, stats in analysis.items():
            print(f"{cond:<25} "
                  f"{stats['mae_mean']:.4f}±{stats['mae_std']:.4f}  "
                  f"{stats['rmse_mean']:.4f}±{stats['rmse_std']:.4f}  "
                  f"{stats['r2_mean']:.4f}±{stats['r2_std']:.4f}  "
                  f"{stats['calibration_mean']:.4f}±{stats['calibration_std']:.4f}")

        # Learning curves
        print("\n" + "-"*80)
        print("LEARNING CURVES (MAE at each checkpoint)")
        print("-"*80)

        header = f"{'Condition':<25}"
        for step in self.config.checkpoint_steps:
            header += f"{step:>12}"
        print(header)
        print("-"*80)

        for cond, stats in analysis.items():
            row = f"{cond:<25}"
            for step in self.config.checkpoint_steps:
                if step in stats.get('checkpoints', {}):
                    mae = stats['checkpoints'][step]['mae_mean']
                    row += f"{mae:>12.4f}"
                else:
                    row += f"{'N/A':>12}"
            print(row)

        # Statistical significance
        print("\n" + "-"*80)
        print("STATISTICAL SIGNIFICANCE")
        print("-"*80)

        for comparison, result in significance.items():
            status = "✓ Significant" if result['significant'] else "✗ Not significant"
            direction = "lower MAE" if result['full_better'] else "higher MAE"
            print(f"\n{comparison}:")
            print(f"  t-statistic: {result['t_statistic']:.4f}")
            print(f"  p-value: {result['p_value']:.4f}")
            print(f"  {status} (p < 0.05)")
            print(f"  Full stack has {direction}")

        # Success criteria evaluation
        print("\n" + "-"*80)
        print("SUCCESS CRITERIA EVALUATION")
        print("-"*80)

        criteria = self.evaluate_success_criteria(analysis, significance)
        for criterion, result in criteria.items():
            status = "✓ MET" if result['met'] else "✗ NOT MET"
            print(f"\n{criterion}: {status}")
            print(f"  {result['description']}")

        print("\n" + "="*80)

    def evaluate_success_criteria(self, analysis: Dict, significance: Dict) -> Dict:
        """Evaluate success criteria."""
        criteria = {}

        # Criterion 1: Sample efficiency (same MAE with fewer queries)
        if 'full_stack_oc' in analysis and 'static_uncertainty' in analysis:
            full_checkpoints = analysis['full_stack_oc'].get('checkpoints', {})
            static_final = analysis['static_uncertainty']['mae_mean']

            # Check if full stack reaches static's final MAE earlier
            early_match = None
            for step in sorted(full_checkpoints.keys()):
                if full_checkpoints[step]['mae_mean'] <= static_final:
                    early_match = step
                    break

            criteria['sample_efficiency'] = {
                'met': early_match is not None and early_match < self.config.query_budget,
                'description': f"Full stack reaches baseline MAE at step {early_match}" if early_match else "Full stack does not reach baseline MAE early"
            }

        # Criterion 2: Accuracy (lower final MAE)
        if 'full_vs_static' in significance:
            criteria['accuracy'] = {
                'met': significance['full_vs_static']['significant'] and significance['full_vs_static']['full_better'],
                'description': f"p={significance['full_vs_static']['p_value']:.4f}, full stack {'lower' if significance['full_vs_static']['full_better'] else 'higher'} MAE"
            }

        # Criterion 3: Calibration (better than online)
        if 'full_stack_oc' in analysis and 'online_no_oc' in analysis:
            full_cal = analysis['full_stack_oc']['calibration_mean']
            online_cal = analysis['online_no_oc']['calibration_mean']
            criteria['calibration'] = {
                'met': full_cal > online_cal,
                'description': f"Full stack calibration: {full_cal:.4f}, Online: {online_cal:.4f}"
            }

        # Criterion 4: Stability (check consolidation metrics)
        if 'full_stack_oc' in self.results:
            retention_scores = []
            for run in self.results['full_stack_oc']:
                for cm in run.consolidation_metrics:
                    if cm.get('retention_score') is not None:
                        retention_scores.append(cm['retention_score'])

            if retention_scores:
                mean_retention = np.mean(retention_scores)
                criteria['stability'] = {
                    'met': mean_retention > 0.8,
                    'description': f"Mean retention score: {mean_retention:.4f} (threshold: 0.8)"
                }

        return criteria

    def generate_report(self, filepath: Optional[str] = None):
        """Generate markdown report."""
        if filepath is None:
            filepath = self.output_dir / 'report.md'

        analysis = self.analyze_results()
        significance = self.compute_significance()
        criteria = self.evaluate_success_criteria(analysis, significance)

        lines = [
            "# Phase 1 Experiment Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Configuration",
            "",
            f"- Seeds: {self.config.n_seeds}",
            f"- Query budget: {self.config.query_budget}",
            f"- Seed size: {self.config.seed_size}",
            f"- Consolidation interval: {self.config.consolidation_interval}",
            "",
            "## Results Summary",
            "",
            "| Condition | MAE | RMSE | R² | Calibration |",
            "|-----------|-----|------|----|-----------  |"
        ]

        for cond, stats in analysis.items():
            lines.append(f"| {cond} | {stats['mae_mean']:.4f}±{stats['mae_std']:.4f} | "
                        f"{stats['rmse_mean']:.4f}±{stats['rmse_std']:.4f} | "
                        f"{stats['r2_mean']:.4f}±{stats['r2_std']:.4f} | "
                        f"{stats['calibration_mean']:.4f}±{stats['calibration_std']:.4f} |")

        lines.extend([
            "",
            "## Statistical Significance",
            ""
        ])

        for comparison, result in significance.items():
            status = "✓ Significant" if result['significant'] else "✗ Not significant"
            lines.append(f"- **{comparison}**: p={result['p_value']:.4f} ({status})")

        lines.extend([
            "",
            "## Success Criteria",
            ""
        ])

        any_met = False
        for criterion, result in criteria.items():
            status = "✓" if result['met'] else "✗"
            lines.append(f"- {status} **{criterion}**: {result['description']}")
            if result['met']:
                any_met = True

        lines.extend([
            "",
            "## Conclusion",
            "",
            f"{'At least one success criterion was met.' if any_met else 'No success criteria were met.'}"
        ])

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

        print(f"\nReport saved to {filepath}")


def main():
    """Run Phase 1 experiment."""
    config = ExperimentConfig(
        n_seeds=5,
        query_budget=50,
        seed_size=10,
        consolidation_interval=10,
        n_estimators=50,
        output_dir='results/phase1'
    )

    experiment = Phase1Experiment(config)
    experiment.run_all(verbose=True)
    experiment.save_results()
    experiment.print_summary()
    experiment.generate_report()


if __name__ == '__main__':
    main()
