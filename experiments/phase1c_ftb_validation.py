"""
Phase 1c Experiment: FTB Validation with Replay Repair.

This experiment validates that we can update the World Model using a
Fine-Tuning Bridge (FTB) without suffering catastrophic forgetting.

Strategies Compared:
1. StaticStrategy: Train once on seed, never update.
2. OnlineStrategy: Retrain on all accumulated history every single step.
3. FTBStrategy: Retrain every N steps using SimplifiedFTB with Replay Repair.

Metrics:
- test_mae: Accuracy on held-out test set
- retention_score: From FTB logs (how well we remember old knowledge)
- repair_count: Total number of times the repair loop was triggered
- updates_performed: Total number of fit calls
"""

import copy
import json
import os
import pickle
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from molecular_world_model import MolecularWorldModel
from molecular_design_env import MolecularDesignEnv, UncertaintySamplingPolicy
from molecular_consolidation_pipeline import SimplifiedFTB


# =============================================================================
# SYNTHETIC DATA GENERATION (if real data not available)
# =============================================================================

def generate_synthetic_data(n_samples: int = 500, seed: int = 42) -> Dict:
    """
    Generate synthetic SMILES and LogS data for testing.

    Creates simple alkane-like SMILES with a pseudo-LogS based on chain length.
    This allows the experiment to run immediately without real data.
    """
    rng = np.random.RandomState(seed)

    # Generate simple alkane chains: C, CC, CCC, CCCC, etc.
    # LogS roughly decreases with chain length (hydrophobicity)
    smiles_list = []
    logs_list = []

    for i in range(n_samples):
        # Chain length from 1 to 12
        chain_length = rng.randint(1, 13)

        # Basic alkane
        smiles = 'C' * chain_length

        # Add some branching occasionally
        if chain_length > 3 and rng.random() < 0.3:
            branch_pos = rng.randint(1, chain_length - 1)
            smiles = 'C' * branch_pos + '(C)' + 'C' * (chain_length - branch_pos - 1)

        # Add some oxygen occasionally (alcohols are more soluble)
        if rng.random() < 0.2:
            smiles = smiles + 'O'
            logs_modifier = 1.5  # More soluble
        else:
            logs_modifier = 0.0

        # Pseudo LogS: decreases with chain length, plus noise
        base_logs = 2.0 - 0.5 * chain_length + logs_modifier
        noise = rng.normal(0, 0.3)
        logs = base_logs + noise

        smiles_list.append(smiles)
        logs_list.append(logs)

    # Split into train/test
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    train_size = int(0.7 * n_samples)
    test_size = int(0.15 * n_samples)

    train_idx = indices[:train_size]
    test_idx = indices[train_size:train_size + test_size]
    candidate_idx = indices[train_size + test_size:]

    import pandas as pd

    return {
        'seed_data': pd.DataFrame({
            'smiles': [smiles_list[i] for i in train_idx[:20]],
            'logS': [logs_list[i] for i in train_idx[:20]]
        }),
        'candidate_pool': pd.DataFrame({
            'smiles': [smiles_list[i] for i in candidate_idx],
            'logS': [logs_list[i] for i in candidate_idx]
        }),
        'test_set': pd.DataFrame({
            'smiles': [smiles_list[i] for i in test_idx],
            'logS': [logs_list[i] for i in test_idx]
        }),
        'all_smiles': smiles_list,
        'all_logs': logs_list
    }


def load_or_generate_data(data_path: str = 'data/esol_processed.pkl') -> Dict:
    """Load real data if available, otherwise generate synthetic data."""
    if os.path.exists(data_path):
        print(f"Loading real data from {data_path}")
        with open(data_path, 'rb') as f:
            return pickle.load(f)
    else:
        print("Real data not found. Generating synthetic data...")
        return generate_synthetic_data()


# =============================================================================
# UPDATE STRATEGIES
# =============================================================================

@dataclass
class StrategyMetrics:
    """Metrics collected during strategy execution."""
    test_mae_history: List[float] = field(default_factory=list)
    stability_ratios: List[float] = field(default_factory=list)  # Renamed from retention_scores
    repair_count: int = 0
    updates_performed: int = 0
    total_time: float = 0.0
    final_test_mae: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'test_mae_history': self.test_mae_history,
            'stability_ratios': self.stability_ratios,
            'repair_count': self.repair_count,
            'updates_performed': self.updates_performed,
            'total_time': self.total_time,
            'final_test_mae': self.final_test_mae,
            'mean_stability': float(np.mean(self.stability_ratios)) if self.stability_ratios else 1.0
        }


class UpdateStrategy(ABC):
    """Base class for model update strategies."""

    def __init__(self, name: str):
        self.name = name
        self.metrics = StrategyMetrics()

    @abstractmethod
    def initialize(self, seed_smiles: List[str], seed_labels: List[float],
                   probe_smiles: List[str], probe_labels: List[float]) -> MolecularWorldModel:
        """Initialize strategy with seed data. Returns the world model."""
        pass

    @abstractmethod
    def on_step(self, step: int, smiles: str, label: float, weight: float = 1.0):
        """Called after each environment step with new data."""
        pass

    @abstractmethod
    def get_world_model(self) -> MolecularWorldModel:
        """Return the current world model for evaluation."""
        pass

    def evaluate(self, test_smiles: List[str], test_labels: List[float]) -> float:
        """Evaluate current model on test set."""
        model = self.get_world_model()
        preds, _ = model.predict(test_smiles, return_uncertainty=True)
        valid_mask = ~np.isnan(preds)
        if np.sum(valid_mask) == 0:
            return float('inf')
        return np.mean(np.abs(preds[valid_mask] - np.array(test_labels)[valid_mask]))


class StaticStrategy(UpdateStrategy):
    """Train once on seed, never update."""

    def __init__(self):
        super().__init__("Static")
        self.world_model = None

    def initialize(self, seed_smiles: List[str], seed_labels: List[float],
                   probe_smiles: List[str], probe_labels: List[float]) -> MolecularWorldModel:
        self.world_model = MolecularWorldModel(n_estimators=50, random_state=42)
        self.world_model.fit(seed_smiles, seed_labels)
        self.metrics.updates_performed = 1
        return self.world_model

    def on_step(self, step: int, smiles: str, label: float, weight: float = 1.0):
        # Never update
        pass

    def get_world_model(self) -> MolecularWorldModel:
        return self.world_model


class OnlineStrategy(UpdateStrategy):
    """Retrain on all accumulated history every single step."""

    def __init__(self):
        super().__init__("Online")
        self.world_model = None
        self.accumulated_smiles: List[str] = []
        self.accumulated_labels: List[float] = []
        self.accumulated_weights: List[float] = []

    def initialize(self, seed_smiles: List[str], seed_labels: List[float],
                   probe_smiles: List[str], probe_labels: List[float]) -> MolecularWorldModel:
        self.world_model = MolecularWorldModel(n_estimators=50, random_state=42)
        self.accumulated_smiles = list(seed_smiles)
        self.accumulated_labels = list(seed_labels)
        self.accumulated_weights = [1.0] * len(seed_smiles)
        self.world_model.fit(self.accumulated_smiles, self.accumulated_labels,
                            sample_weight=self.accumulated_weights)
        self.metrics.updates_performed = 1
        return self.world_model

    def on_step(self, step: int, smiles: str, label: float, weight: float = 1.0):
        # Add new data
        self.accumulated_smiles.append(smiles)
        self.accumulated_labels.append(label)
        self.accumulated_weights.append(weight)

        # Retrain on everything
        self.world_model.fit(self.accumulated_smiles, self.accumulated_labels,
                            sample_weight=self.accumulated_weights)
        self.metrics.updates_performed += 1

    def get_world_model(self) -> MolecularWorldModel:
        return self.world_model


class FTBStrategy(UpdateStrategy):
    """Retrain every N steps using SimplifiedFTB with Replay Repair."""

    def __init__(self, update_interval: int = 10, retention_threshold: float = 0.25):
        super().__init__("FTB")
        self.update_interval = update_interval
        self.retention_threshold = retention_threshold
        self.world_model = None
        self.ftb = None
        self.batch_smiles: List[str] = []
        self.batch_labels: List[float] = []
        self.batch_weights: List[float] = []
        self.accumulated_smiles: List[str] = []
        self.accumulated_labels: List[float] = []

    def initialize(self, seed_smiles: List[str], seed_labels: List[float],
                   probe_smiles: List[str], probe_labels: List[float]) -> MolecularWorldModel:
        self.world_model = MolecularWorldModel(n_estimators=50, random_state=42)
        self.accumulated_smiles = list(seed_smiles)
        self.accumulated_labels = list(seed_labels)
        self.world_model.fit(self.accumulated_smiles, self.accumulated_labels)

        # Initialize FTB with probe set
        self.ftb = SimplifiedFTB(
            world_model=self.world_model,
            probe_smiles=probe_smiles,
            probe_labels=probe_labels,
            retention_threshold=self.retention_threshold,
            repair_attempts=3,
            replay_ratio=0.5,
            random_state=42
        )

        self.metrics.updates_performed = 1
        return self.world_model

    def on_step(self, step: int, smiles: str, label: float, weight: float = 1.0):
        # Accumulate in batch
        self.batch_smiles.append(smiles)
        self.batch_labels.append(label)
        self.batch_weights.append(weight)

        # Check if it's time to update
        if len(self.batch_smiles) >= self.update_interval:
            self._perform_update()

    def _perform_update(self):
        """Perform FTB update with accumulated batch."""
        if not self.batch_smiles:
            return

        # Add batch to accumulated data
        self.accumulated_smiles.extend(self.batch_smiles)
        self.accumulated_labels.extend(self.batch_labels)

        # Use FTB to update with retention guarantees
        result = self.ftb.update(
            smiles=self.accumulated_smiles,
            labels=self.accumulated_labels,
            weights=[1.0] * len(self.accumulated_smiles)  # Uniform weights for now
        )

        # Record metrics (using stability_ratio from improvement_ratio)
        self.metrics.updates_performed += 1
        self.metrics.stability_ratios.append(result['improvement_ratio'])

        if result['was_repaired']:
            self.metrics.repair_count += result['repair_attempts']

        # Clear batch
        self.batch_smiles = []
        self.batch_labels = []
        self.batch_weights = []

    def finalize(self):
        """Flush any remaining batch data."""
        if self.batch_smiles:
            self._perform_update()

    def get_world_model(self) -> MolecularWorldModel:
        return self.world_model


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(
    data: Dict,
    n_steps: int = 50,
    seed_size: int = 20,
    eval_interval: int = 5,
    random_state: int = 42
) -> Dict[str, StrategyMetrics]:
    """
    Run the Phase 1c experiment comparing update strategies.

    Args:
        data: Dict with 'candidate_pool', 'test_set', 'seed_data' DataFrames
        n_steps: Number of environment steps to run
        seed_size: Number of molecules for initial training
        eval_interval: Evaluate test MAE every N steps
        random_state: Random seed

    Returns:
        Dict mapping strategy name to StrategyMetrics
    """
    import pandas as pd

    # Extract data
    candidate_df = data['candidate_pool']
    test_df = data['test_set']

    # Use first seed_size from candidate pool as seed data
    seed_smiles = candidate_df['smiles'].tolist()[:seed_size]
    seed_labels = candidate_df['logS'].tolist()[:seed_size]

    test_smiles = test_df['smiles'].tolist()
    test_labels = test_df['logS'].tolist()

    # Use test set as probe set for retention checking
    probe_smiles = test_smiles[:min(30, len(test_smiles))]
    probe_labels = test_labels[:min(30, len(test_labels))]

    # Pre-generate the sequence of actions for reproducibility
    # This ensures all strategies see the same data in the same order
    rng = np.random.RandomState(random_state)

    # Get candidate pool beyond seed data
    query_pool_smiles = candidate_df['smiles'].tolist()[seed_size:]
    query_pool_labels = candidate_df['logS'].tolist()[seed_size:]

    # Shuffle to simulate random exploration order
    indices = list(range(len(query_pool_smiles)))
    rng.shuffle(indices)
    query_sequence = [(query_pool_smiles[i], query_pool_labels[i]) for i in indices[:n_steps]]

    print(f"\n{'='*70}")
    print("Phase 1c: FTB Validation Experiment")
    print(f"{'='*70}")
    print(f"Seed size: {seed_size}")
    print(f"Query pool: {len(query_pool_smiles)}")
    print(f"Test set: {len(test_df)}")
    print(f"Probe set: {len(probe_smiles)}")
    print(f"Steps to run: {n_steps}")
    print(f"{'='*70}\n")

    # Initialize strategies
    strategies = [
        StaticStrategy(),
        OnlineStrategy(),
        FTBStrategy(update_interval=10, retention_threshold=0.25)
    ]

    results = {}

    for strategy in strategies:
        print(f"\n--- Running {strategy.name} Strategy ---")
        start_time = time.time()

        # Initialize with seed data
        world_model = strategy.initialize(
            seed_smiles=seed_smiles,
            seed_labels=seed_labels,
            probe_smiles=probe_smiles,
            probe_labels=probe_labels
        )

        # Initial evaluation
        initial_mae = strategy.evaluate(test_smiles, test_labels)
        strategy.metrics.test_mae_history.append(initial_mae)
        print(f"  Initial test MAE: {initial_mae:.4f}")

        # Run through query sequence (same data for all strategies)
        for step, (smiles, label) in enumerate(query_sequence):
            # Update strategy with new observation
            strategy.on_step(step, smiles, label, weight=1.0)

            # Periodic evaluation
            if (step + 1) % eval_interval == 0:
                mae = strategy.evaluate(test_smiles, test_labels)
                strategy.metrics.test_mae_history.append(mae)
                print(f"  Step {step + 1}: test MAE = {mae:.4f}")

        # Finalize (flush any remaining batches for FTB)
        if hasattr(strategy, 'finalize'):
            strategy.finalize()

        # Final evaluation
        final_mae = strategy.evaluate(test_smiles, test_labels)
        strategy.metrics.final_test_mae = final_mae
        strategy.metrics.total_time = time.time() - start_time

        print(f"  Final test MAE: {final_mae:.4f}")
        print(f"  Updates performed: {strategy.metrics.updates_performed}")
        print(f"  Repair count: {strategy.metrics.repair_count}")
        print(f"  Time: {strategy.metrics.total_time:.2f}s")

        results[strategy.name] = strategy.metrics

    return results


def print_comparison_table(results: Dict[str, StrategyMetrics]):
    """Print a formatted comparison table."""
    print(f"\n{'='*70}")
    print("STRATEGY COMPARISON")
    print(f"{'='*70}")
    print(f"{'Strategy':<12} {'Final MAE':<12} {'Updates':<10} {'Repairs':<10} {'Stability':<12} {'Time(s)':<10}")
    print(f"{'-'*70}")

    for name, metrics in results.items():
        mean_stability = np.mean(metrics.stability_ratios) if metrics.stability_ratios else 1.0
        print(f"{name:<12} {metrics.final_test_mae:<12.4f} {metrics.updates_performed:<10} "
              f"{metrics.repair_count:<10} {mean_stability:<12.4f} {metrics.total_time:<10.2f}")

    print(f"{'='*70}\n")


def save_results(results: Dict[str, StrategyMetrics], output_dir: str = 'results/phase1c'):
    """Save experiment results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    report = {
        'experiment': 'phase1c_ftb_validation',
        'timestamp': datetime.now().isoformat(),
        'strategies': {name: metrics.to_dict() for name, metrics in results.items()}
    }

    # Add summary
    report['summary'] = {
        'best_final_mae': min(m.final_test_mae for m in results.values()),
        'best_strategy': min(results.items(), key=lambda x: x[1].final_test_mae)[0],
        'ftb_repair_effectiveness': results['FTB'].repair_count if 'FTB' in results else 0,
        'ftb_stability_maintained': (
            np.mean(results['FTB'].stability_ratios) >= 0.75
            if 'FTB' in results and results['FTB'].stability_ratios
            else True
        )
    }

    # Save
    output_file = output_path / 'results.json'
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Results saved to {output_file}")
    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the Phase 1c experiment."""
    # Load or generate data
    data = load_or_generate_data()

    # Run experiment
    results = run_experiment(
        data=data,
        n_steps=50,
        seed_size=20,
        eval_interval=5,
        random_state=42
    )

    # Print comparison
    print_comparison_table(results)

    # Save results
    report = save_results(results)

    # Print key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print(f"Best strategy: {report['summary']['best_strategy']}")
    print(f"Best final MAE: {report['summary']['best_final_mae']:.4f}")

    if 'FTB' in results:
        ftb = results['FTB']
        print(f"\nFTB Statistics:")
        print(f"  - Updates performed: {ftb.updates_performed}")
        print(f"  - Repair loops triggered: {ftb.repair_count}")
        print(f"  - Mean stability ratio: {np.mean(ftb.stability_ratios) if ftb.stability_ratios else 1.0:.4f}")
        print(f"  - Stability maintained: {report['summary']['ftb_stability_maintained']}")

    print("="*70)

    return results


if __name__ == '__main__':
    main()
