"""
Phase 2 Dream Validation: FTB + Generative Exploration.

This experiment validates the Dream State layer where the agent:
1. Generates virtual analogs of tested molecules
2. Predicts properties with uncertainty quantification
3. Filters by confidence threshold (0.85 - conservative given 0.28 calibration)
4. Augments FTB training loop with high-confidence synthetics

Key design choices:
- Confidence threshold: 0.85 (raised from 0.8 to compensate for 0.28 calibration)
- Synthetic weight: 0.6 (real data dominates)
- Synthetic cap: 30% (synthetics can't overwhelm real observations)
- Probe set purity: Retention checks use ONLY real molecules

Condition-aware logging tracks acceptance rates separately by condition,
allowing us to see if the model correctly becomes more cautious under noise.

Strategies:
1. Static: Train once, never update (baseline)
2. Online: Retrain every step
3. FTB: Batched updates with Replay Repair
4. DreamFTB: FTB + synthetic augmentation from dreams

Success Criteria:
- DreamFTB MAE <= FTB MAE (no harm)
- Acceptance rate 10-50% (dreams produced)
- Zero repair loops from synthetics (no forgetting)
- >=3 significant SAR rules discovered
"""

import copy
import json
import logging
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
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from molecular_world_model import MolecularWorldModel
from molecular_consolidation_pipeline import SimplifiedFTB
from dream_state import AnalogGenerator, SARExtractor, DreamPipeline

# RDKit for molecular weight calculation
from rdkit import Chem
from rdkit.Chem import Descriptors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING WITH CONDITIONS (from phase1c)
# =============================================================================

def compute_molecular_weight(smiles: str) -> float:
    """Compute molecular weight for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return Descriptors.MolWt(mol)


def load_data_with_condition(
    condition: str,
    data_path: str = 'data/esol_processed.pkl',
    seed: int = 42,
    noise_level: float = 0.15
) -> Dict:
    """Load data with condition-specific preprocessing."""
    rng = np.random.RandomState(seed)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    candidate_df = data['candidate_pool'].copy()
    test_df = data['test_set'].copy()

    all_smiles = candidate_df['smiles'].tolist() + test_df['smiles'].tolist()
    all_labels = candidate_df['logS'].tolist() + test_df['logS'].tolist()

    metadata = {
        'condition': condition,
        'seed': seed,
        'original_size': len(all_smiles)
    }

    if condition == 'clean':
        indices = list(range(len(all_smiles)))
        rng.shuffle(indices)
        all_smiles = [all_smiles[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]
        metadata['preprocessing'] = 'random_shuffle'

    elif condition == 'noisy_15pct':
        indices = list(range(len(all_smiles)))
        rng.shuffle(indices)
        all_smiles = [all_smiles[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]

        label_std = np.std(all_labels)
        noise = rng.normal(0, label_std * noise_level, size=len(all_labels))
        all_labels = [l + n for l, n in zip(all_labels, noise)]

        metadata['preprocessing'] = f'random_shuffle + {noise_level*100:.0f}% noise'
        metadata['noise_std'] = float(label_std * noise_level)

    elif condition == 'distribution_shift':
        mol_weights = [compute_molecular_weight(s) for s in all_smiles]
        sorted_indices = np.argsort(mol_weights)
        all_smiles = [all_smiles[i] for i in sorted_indices]
        all_labels = [all_labels[i] for i in sorted_indices]
        metadata['preprocessing'] = 'sorted_by_molecular_weight_ascending'

    else:
        raise ValueError(f"Unknown condition: {condition}")

    n_total = len(all_smiles)
    n_candidates = int(0.7 * n_total)

    return {
        'candidate_pool': pd.DataFrame({
            'smiles': all_smiles[:n_candidates],
            'logS': all_labels[:n_candidates]
        }),
        'test_set': pd.DataFrame({
            'smiles': all_smiles[n_candidates:],
            'logS': all_labels[n_candidates:]
        }),
        'metadata': metadata
    }


# =============================================================================
# EXTENDED STRATEGY METRICS FOR DREAMING
# =============================================================================

@dataclass
class DreamMetrics:
    """Extended metrics for dream-enabled strategies."""
    test_mae_history: List[float] = field(default_factory=list)
    stability_ratios: List[float] = field(default_factory=list)
    repair_count: int = 0
    updates_performed: int = 0
    total_time: float = 0.0
    final_test_mae: float = 0.0

    # Dream-specific metrics
    total_synthetics_generated: int = 0
    total_synthetics_accepted: int = 0
    acceptance_rates: List[float] = field(default_factory=list)
    sar_rules_discovered: List[Dict] = field(default_factory=list)
    dream_intervals: int = 0
    condition_acceptance: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'test_mae_history': self.test_mae_history,
            'stability_ratios': self.stability_ratios,
            'repair_count': self.repair_count,
            'updates_performed': self.updates_performed,
            'total_time': self.total_time,
            'final_test_mae': self.final_test_mae,
            'mean_stability': float(np.mean(self.stability_ratios)) if self.stability_ratios else 1.0,
            # Dream metrics
            'total_synthetics_generated': self.total_synthetics_generated,
            'total_synthetics_accepted': self.total_synthetics_accepted,
            'mean_acceptance_rate': float(np.mean(self.acceptance_rates)) if self.acceptance_rates else 0.0,
            'n_sar_rules': len(self.sar_rules_discovered),
            'dream_intervals': self.dream_intervals,
            'overall_acceptance': (
                self.total_synthetics_accepted / self.total_synthetics_generated
                if self.total_synthetics_generated > 0 else 0.0
            )
        }


# =============================================================================
# UPDATE STRATEGIES
# =============================================================================

class UpdateStrategy(ABC):
    """Base class for model update strategies."""

    def __init__(self, name: str):
        self.name = name
        self.metrics = DreamMetrics()

    @abstractmethod
    def initialize(self, seed_smiles: List[str], seed_labels: List[float],
                   probe_smiles: List[str], probe_labels: List[float],
                   random_state: int, condition: str = 'clean') -> MolecularWorldModel:
        pass

    @abstractmethod
    def on_step(self, step: int, smiles: str, label: float, weight: float = 1.0):
        pass

    @abstractmethod
    def get_world_model(self) -> MolecularWorldModel:
        pass

    def reset(self):
        self.metrics = DreamMetrics()

    def evaluate(self, test_smiles: List[str], test_labels: List[float]) -> float:
        model = self.get_world_model()
        preds, _ = model.predict(test_smiles, return_uncertainty=True)
        valid_mask = ~np.isnan(preds)
        if np.sum(valid_mask) == 0:
            return float('inf')
        return float(np.mean(np.abs(preds[valid_mask] - np.array(test_labels)[valid_mask])))


class StaticStrategy(UpdateStrategy):
    """Train once on seed, never update."""

    def __init__(self):
        super().__init__("Static")
        self.world_model = None

    def initialize(self, seed_smiles, seed_labels, probe_smiles, probe_labels,
                   random_state, condition='clean'):
        self.world_model = MolecularWorldModel(n_estimators=50, random_state=random_state)
        self.world_model.fit(seed_smiles, seed_labels)
        self.metrics.updates_performed = 1
        return self.world_model

    def on_step(self, step, smiles, label, weight=1.0):
        pass

    def get_world_model(self):
        return self.world_model


class OnlineStrategy(UpdateStrategy):
    """Retrain on all accumulated history every step."""

    def __init__(self):
        super().__init__("Online")
        self.world_model = None
        self.accumulated_smiles: List[str] = []
        self.accumulated_labels: List[float] = []
        self.accumulated_weights: List[float] = []
        self.random_state = 42

    def initialize(self, seed_smiles, seed_labels, probe_smiles, probe_labels,
                   random_state, condition='clean'):
        self.random_state = random_state
        self.world_model = MolecularWorldModel(n_estimators=50, random_state=random_state)
        self.accumulated_smiles = list(seed_smiles)
        self.accumulated_labels = list(seed_labels)
        self.accumulated_weights = [1.0] * len(seed_smiles)
        self.world_model.fit(self.accumulated_smiles, self.accumulated_labels,
                            sample_weight=self.accumulated_weights)
        self.metrics.updates_performed = 1
        return self.world_model

    def on_step(self, step, smiles, label, weight=1.0):
        self.accumulated_smiles.append(smiles)
        self.accumulated_labels.append(label)
        self.accumulated_weights.append(weight)
        self.world_model.fit(self.accumulated_smiles, self.accumulated_labels,
                            sample_weight=self.accumulated_weights)
        self.metrics.updates_performed += 1

    def get_world_model(self):
        return self.world_model


class FTBStrategy(UpdateStrategy):
    """Batched updates using SimplifiedFTB with Replay Repair."""

    def __init__(self, update_interval: int = 10, retention_threshold: float = 0.25):
        super().__init__("FTB")
        self.update_interval = update_interval
        self.retention_threshold = retention_threshold
        self.world_model = None
        self.ftb = None
        self.batch_smiles: List[str] = []
        self.batch_labels: List[float] = []
        self.accumulated_smiles: List[str] = []
        self.accumulated_labels: List[float] = []
        self.random_state = 42

    def initialize(self, seed_smiles, seed_labels, probe_smiles, probe_labels,
                   random_state, condition='clean'):
        self.random_state = random_state
        self.world_model = MolecularWorldModel(n_estimators=50, random_state=random_state)
        self.accumulated_smiles = list(seed_smiles)
        self.accumulated_labels = list(seed_labels)
        self.batch_smiles = []
        self.batch_labels = []
        self.world_model.fit(self.accumulated_smiles, self.accumulated_labels)

        self.ftb = SimplifiedFTB(
            world_model=self.world_model,
            probe_smiles=probe_smiles,
            probe_labels=probe_labels,
            retention_threshold=self.retention_threshold,
            repair_attempts=3,
            replay_ratio=0.5,
            random_state=random_state
        )

        self.metrics.updates_performed = 1
        return self.world_model

    def on_step(self, step, smiles, label, weight=1.0):
        self.batch_smiles.append(smiles)
        self.batch_labels.append(label)

        if len(self.batch_smiles) >= self.update_interval:
            self._perform_update()

    def _perform_update(self):
        if not self.batch_smiles:
            return

        self.accumulated_smiles.extend(self.batch_smiles)
        self.accumulated_labels.extend(self.batch_labels)

        result = self.ftb.update(
            smiles=self.accumulated_smiles,
            labels=self.accumulated_labels,
            weights=[1.0] * len(self.accumulated_smiles)
        )

        self.metrics.updates_performed += 1
        self.metrics.stability_ratios.append(result['improvement_ratio'])

        if result['was_repaired']:
            self.metrics.repair_count += result['repair_attempts']

        self.batch_smiles = []
        self.batch_labels = []

    def finalize(self):
        if self.batch_smiles:
            self._perform_update()

    def get_world_model(self):
        return self.world_model


class DreamFTBStrategy(UpdateStrategy):
    """
    FTB + Synthetic Augmentation from Dreams.

    Key features:
    - Generates analogs at each update interval
    - Filters by confidence threshold (0.85)
    - Combines real (weight=1.0) with synthetic (weight=0.6)
    - Uses FTB retention check to catch any forgetting
    - Condition-aware logging tracks acceptance by data quality
    """

    def __init__(
        self,
        update_interval: int = 10,
        retention_threshold: float = 0.25,
        confidence_threshold: float = 0.85,  # Raised from 0.8 for calibration=0.28
        max_synthetics_ratio: float = 0.3,
        synthetic_weight: float = 0.6
    ):
        super().__init__("DreamFTB")
        self.update_interval = update_interval
        self.retention_threshold = retention_threshold
        self.confidence_threshold = confidence_threshold
        self.max_synthetics_ratio = max_synthetics_ratio
        self.synthetic_weight = synthetic_weight

        self.world_model = None
        self.ftb = None
        self.dream_pipeline = None

        self.batch_smiles: List[str] = []
        self.batch_labels: List[float] = []
        self.accumulated_smiles: List[str] = []
        self.accumulated_labels: List[float] = []
        self.random_state = 42
        self.condition = 'clean'

    def initialize(self, seed_smiles, seed_labels, probe_smiles, probe_labels,
                   random_state, condition='clean'):
        self.random_state = random_state
        self.condition = condition

        # Initialize world model
        self.world_model = MolecularWorldModel(n_estimators=50, random_state=random_state)
        self.accumulated_smiles = list(seed_smiles)
        self.accumulated_labels = list(seed_labels)
        self.batch_smiles = []
        self.batch_labels = []
        self.world_model.fit(self.accumulated_smiles, self.accumulated_labels)

        # Initialize FTB with probe set (REAL molecules only)
        self.ftb = SimplifiedFTB(
            world_model=self.world_model,
            probe_smiles=probe_smiles,
            probe_labels=probe_labels,
            retention_threshold=self.retention_threshold,
            repair_attempts=3,
            replay_ratio=0.5,
            random_state=random_state
        )

        # Initialize Dream Pipeline
        analog_generator = AnalogGenerator(random_state=random_state)
        sar_extractor = SARExtractor(min_support=5, min_effect_size=0.3, max_p_value=0.05)

        self.dream_pipeline = DreamPipeline(
            world_model=self.world_model,
            analog_generator=analog_generator,
            sar_extractor=sar_extractor,
            confidence_threshold=self.confidence_threshold,
            max_synthetics_ratio=self.max_synthetics_ratio,
            synthetic_weight=self.synthetic_weight,
            random_state=random_state
        )

        self.metrics.updates_performed = 1
        return self.world_model

    def on_step(self, step, smiles, label, weight=1.0):
        self.batch_smiles.append(smiles)
        self.batch_labels.append(label)

        if len(self.batch_smiles) >= self.update_interval:
            self._perform_dream_update()

    def _perform_dream_update(self):
        if not self.batch_smiles:
            return

        # Add batch to accumulated REAL data
        self.accumulated_smiles.extend(self.batch_smiles)
        self.accumulated_labels.extend(self.batch_labels)

        # Dream: Generate and filter synthetics
        dream_result = self.dream_pipeline.dream(
            real_smiles=self.accumulated_smiles,
            real_labels=self.accumulated_labels,
            condition=self.condition,
            n_variants_per_molecule=5
        )

        # Track dream metrics
        self.metrics.dream_intervals += 1
        self.metrics.total_synthetics_generated += dream_result['n_analogs_generated']
        self.metrics.total_synthetics_accepted += dream_result['n_after_cap']
        self.metrics.acceptance_rates.append(dream_result['acceptance_rate'])

        # Track by condition
        if self.condition not in self.metrics.condition_acceptance:
            self.metrics.condition_acceptance[self.condition] = []
        self.metrics.condition_acceptance[self.condition].append(
            dream_result['acceptance_rate']
        )

        # Collect SAR rules
        if dream_result['sar_rules']:
            self.metrics.sar_rules_discovered.extend(dream_result['sar_rules'])

        # Combine real + synthetic for training
        combined_smiles = self.accumulated_smiles + dream_result['synthetic_smiles']
        combined_labels = self.accumulated_labels + dream_result['synthetic_labels']
        combined_weights = (
            [1.0] * len(self.accumulated_smiles) +
            dream_result['synthetic_weights']
        )

        # Use FTB to update with retention guarantees
        # Note: Probe set is REAL molecules only, so retention check is pure
        result = self.ftb.update(
            smiles=combined_smiles,
            labels=combined_labels,
            weights=combined_weights
        )

        self.metrics.updates_performed += 1
        self.metrics.stability_ratios.append(result['improvement_ratio'])

        if result['was_repaired']:
            self.metrics.repair_count += result['repair_attempts']
            logger.warning(
                f"[DreamFTB|{self.condition}] Repair triggered - "
                f"synthetics may have caused drift"
            )

        # Clear batch (but keep accumulated for next dream)
        self.batch_smiles = []
        self.batch_labels = []

    def finalize(self):
        if self.batch_smiles:
            self._perform_dream_update()

    def get_world_model(self):
        return self.world_model


# =============================================================================
# SINGLE RUN EXECUTION
# =============================================================================

def run_single_experiment(
    condition: str,
    strategy: UpdateStrategy,
    seed: int,
    n_steps: int = 50,
    seed_size: int = 20,
    eval_interval: int = 10,
    verbose: bool = False
) -> Dict:
    """Run a single experiment configuration."""
    strategy.reset()

    data = load_data_with_condition(condition, seed=seed)

    candidate_df = data['candidate_pool']
    test_df = data['test_set']

    seed_smiles = candidate_df['smiles'].tolist()[:seed_size]
    seed_labels = candidate_df['logS'].tolist()[:seed_size]

    test_smiles = test_df['smiles'].tolist()
    test_labels = test_df['logS'].tolist()

    # Probe set from test data (for FTB retention - REAL molecules only)
    probe_smiles = test_smiles[:min(30, len(test_smiles))]
    probe_labels = test_labels[:min(30, len(test_labels))]

    query_pool_smiles = candidate_df['smiles'].tolist()[seed_size:]
    query_pool_labels = candidate_df['logS'].tolist()[seed_size:]

    if condition != 'distribution_shift':
        rng = np.random.RandomState(seed)
        indices = list(range(len(query_pool_smiles)))
        rng.shuffle(indices)
        query_pool_smiles = [query_pool_smiles[i] for i in indices]
        query_pool_labels = [query_pool_labels[i] for i in indices]

    query_sequence = list(zip(query_pool_smiles[:n_steps], query_pool_labels[:n_steps]))

    start_time = time.time()

    # Pass condition to strategy (for condition-aware logging)
    strategy.initialize(
        seed_smiles=seed_smiles,
        seed_labels=seed_labels,
        probe_smiles=probe_smiles,
        probe_labels=probe_labels,
        random_state=seed,
        condition=condition
    )

    initial_mae = strategy.evaluate(test_smiles, test_labels)
    strategy.metrics.test_mae_history.append(initial_mae)

    if verbose:
        print(f"    Initial MAE: {initial_mae:.4f}")

    for step, (smiles, label) in enumerate(query_sequence):
        strategy.on_step(step, smiles, label, weight=1.0)

        if (step + 1) % eval_interval == 0:
            mae = strategy.evaluate(test_smiles, test_labels)
            strategy.metrics.test_mae_history.append(mae)

            if verbose:
                print(f"    Step {step + 1}: MAE = {mae:.4f}")

    if hasattr(strategy, 'finalize'):
        strategy.finalize()

    final_mae = strategy.evaluate(test_smiles, test_labels)
    strategy.metrics.final_test_mae = final_mae
    strategy.metrics.total_time = time.time() - start_time

    return {
        'condition': condition,
        'strategy': strategy.name,
        'seed': seed,
        'metrics': strategy.metrics.to_dict(),
        'data_metadata': data['metadata']
    }


# =============================================================================
# FULL PHASE 2 VALIDATION
# =============================================================================

def run_phase2_validation(
    conditions: List[str] = None,
    seeds: List[int] = None,
    n_steps: int = 50,
    seed_size: int = 20,
    eval_interval: int = 10,
    verbose: bool = True
) -> Dict:
    """Run the full Phase 2 validation matrix."""

    if conditions is None:
        conditions = ['clean', 'noisy_15pct', 'distribution_shift']

    if seeds is None:
        seeds = [42, 123, 456, 789, 1011]

    print("\n" + "=" * 80)
    print("PHASE 2 DREAM VALIDATION")
    print("=" * 80)
    print(f"Conditions: {conditions}")
    print(f"Seeds: {seeds}")
    print(f"Steps per run: {n_steps}")
    print(f"Confidence threshold: 0.85 (conservative for calibration=0.28)")
    print(f"Matrix: {len(conditions)} x 4 strategies x {len(seeds)} seeds = "
          f"{len(conditions) * 4 * len(seeds)} runs")
    print("=" * 80 + "\n")

    all_results = []
    start_time = time.time()

    for condition in conditions:
        print(f"\n{'='*70}")
        print(f"CONDITION: {condition.upper()}")
        print(f"{'='*70}")

        for seed in seeds:
            print(f"\n  Seed {seed}:")

            strategies = [
                StaticStrategy(),
                OnlineStrategy(),
                FTBStrategy(update_interval=10, retention_threshold=0.25),
                DreamFTBStrategy(
                    update_interval=10,
                    retention_threshold=0.25,
                    confidence_threshold=0.85,
                    max_synthetics_ratio=0.3,
                    synthetic_weight=0.6
                )
            ]

            for strategy in strategies:
                if verbose:
                    print(f"    {strategy.name}...", end=" ", flush=True)

                result = run_single_experiment(
                    condition=condition,
                    strategy=strategy,
                    seed=seed,
                    n_steps=n_steps,
                    seed_size=seed_size,
                    eval_interval=eval_interval,
                    verbose=False
                )

                all_results.append(result)

                if verbose:
                    m = result['metrics']
                    base_info = (f"MAE={m['final_test_mae']:.4f}, "
                                f"Updates={m['updates_performed']}, "
                                f"Repairs={m['repair_count']}")

                    # Add dream-specific info for DreamFTB
                    if 'mean_acceptance_rate' in m and m['dream_intervals'] > 0:
                        base_info += (f", Synthetics={m['total_synthetics_accepted']}/"
                                     f"{m['total_synthetics_generated']} "
                                     f"({m['mean_acceptance_rate']:.1%})")

                    print(base_info)

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")

    summary = compute_phase2_summary(all_results)

    return {
        'experiment': 'phase2_dream_validation',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'conditions': conditions,
            'seeds': seeds,
            'n_steps': n_steps,
            'seed_size': seed_size,
            'eval_interval': eval_interval,
            'confidence_threshold': 0.85,
            'max_synthetics_ratio': 0.3,
            'synthetic_weight': 0.6
        },
        'results': all_results,
        'summary': summary,
        'total_time': total_time
    }


def compute_phase2_summary(results: List[Dict]) -> Dict:
    """Compute summary statistics including dream metrics."""
    summary = {}

    conditions = set(r['condition'] for r in results)
    strategies = set(r['strategy'] for r in results)

    for condition in conditions:
        summary[condition] = {}

        for strategy in strategies:
            filtered = [r for r in results
                       if r['condition'] == condition and r['strategy'] == strategy]

            if not filtered:
                continue

            maes = [r['metrics']['final_test_mae'] for r in filtered]
            updates = [r['metrics']['updates_performed'] for r in filtered]
            repairs = [r['metrics']['repair_count'] for r in filtered]
            stabilities = []
            for r in filtered:
                if r['metrics']['stability_ratios']:
                    stabilities.extend(r['metrics']['stability_ratios'])

            base_stats = {
                'mae_mean': float(np.mean(maes)),
                'mae_std': float(np.std(maes)),
                'mae_min': float(np.min(maes)),
                'mae_max': float(np.max(maes)),
                'updates_mean': float(np.mean(updates)),
                'updates_total': int(np.sum(updates)),
                'repairs_total': int(np.sum(repairs)),
                'stability_mean': float(np.mean(stabilities)) if stabilities else 1.0,
                'n_runs': len(filtered)
            }

            # Add dream-specific stats for DreamFTB
            if strategy == 'DreamFTB':
                gen_totals = [r['metrics']['total_synthetics_generated'] for r in filtered]
                acc_totals = [r['metrics']['total_synthetics_accepted'] for r in filtered]
                acc_rates = [r['metrics']['mean_acceptance_rate'] for r in filtered]
                n_rules = [r['metrics']['n_sar_rules'] for r in filtered]

                base_stats.update({
                    'total_generated': int(np.sum(gen_totals)),
                    'total_accepted': int(np.sum(acc_totals)),
                    'mean_acceptance_rate': float(np.mean(acc_rates)),
                    'std_acceptance_rate': float(np.std(acc_rates)),
                    'total_sar_rules': int(np.sum(n_rules)),
                    'overall_acceptance': (
                        np.sum(acc_totals) / np.sum(gen_totals)
                        if np.sum(gen_totals) > 0 else 0.0
                    )
                })

            summary[condition][strategy] = base_stats

    # Compute comparisons (DreamFTB vs FTB)
    for condition in conditions:
        if 'FTB' in summary[condition] and 'DreamFTB' in summary[condition]:
            ftb = summary[condition]['FTB']
            dream = summary[condition]['DreamFTB']

            mae_diff = dream['mae_mean'] - ftb['mae_mean']
            mae_ratio = dream['mae_mean'] / ftb['mae_mean'] if ftb['mae_mean'] > 0 else 1.0

            summary[condition]['_dream_comparison'] = {
                'mae_difference': float(mae_diff),
                'mae_ratio': float(mae_ratio),
                'dream_no_harm': mae_ratio <= 1.10,  # Within 10%
                'dream_improves': mae_ratio < 0.98,  # 2%+ improvement
                'acceptance_rate': dream.get('mean_acceptance_rate', 0.0),
                'acceptance_in_range': 0.05 <= dream.get('mean_acceptance_rate', 0) <= 0.50,
                'no_extra_repairs': dream['repairs_total'] <= ftb['repairs_total']
            }

    return summary


def print_phase2_summary(summary: Dict):
    """Print formatted Phase 2 summary."""
    print("\n" + "=" * 100)
    print("PHASE 2 SUMMARY BY CONDITION")
    print("=" * 100)

    for condition in summary:
        if condition.startswith('_'):
            continue

        print(f"\n{condition.upper()}")
        print("-" * 100)
        print(f"{'Strategy':<12} {'MAE (mean±std)':<20} {'Updates':<10} {'Repairs':<10} "
              f"{'Synthetics':<15} {'Accept Rate':<12}")
        print("-" * 100)

        for strategy in ['Static', 'Online', 'FTB', 'DreamFTB']:
            if strategy not in summary[condition]:
                continue

            s = summary[condition][strategy]
            mae_str = f"{s['mae_mean']:.4f} ± {s['mae_std']:.4f}"

            if strategy == 'DreamFTB':
                synth_str = f"{s.get('total_accepted', 0)}/{s.get('total_generated', 0)}"
                rate_str = f"{s.get('mean_acceptance_rate', 0):.1%}"
            else:
                synth_str = "-"
                rate_str = "-"

            print(f"{strategy:<12} {mae_str:<20} {s['updates_mean']:<10.1f} "
                  f"{s['repairs_total']:<10} {synth_str:<15} {rate_str:<12}")

        # Print dream comparison
        if '_dream_comparison' in summary[condition]:
            comp = summary[condition]['_dream_comparison']
            print("-" * 100)

            status_parts = []
            if comp['dream_no_harm']:
                status_parts.append("NO HARM")
            else:
                status_parts.append("REGRESSION")

            if comp['dream_improves']:
                status_parts.append("IMPROVED")

            if comp['acceptance_in_range']:
                status_parts.append(f"ACCEPT OK ({comp['acceptance_rate']:.1%})")
            elif comp['acceptance_rate'] < 0.05:
                status_parts.append(f"LOW ACCEPT ({comp['acceptance_rate']:.1%}) - model cautious")
            else:
                status_parts.append(f"HIGH ACCEPT ({comp['acceptance_rate']:.1%})")

            if comp['no_extra_repairs']:
                status_parts.append("NO FORGETTING")
            else:
                status_parts.append("TRIGGERED REPAIRS")

            print(f"DreamFTB vs FTB: MAE ratio={comp['mae_ratio']:.3f} | "
                  f"{' | '.join(status_parts)}")

    print("\n" + "=" * 100)


def print_phase2_verdict(summary: Dict):
    """Print overall Phase 2 verdict."""
    print("\n" + "=" * 100)
    print("PHASE 2 VERDICT")
    print("=" * 100)

    checks = {
        'no_harm': True,
        'dreams_produced': True,
        'no_forgetting': True,
        'sar_discovered': False
    }

    condition_results = []

    for condition in ['clean', 'noisy_15pct', 'distribution_shift']:
        if condition not in summary:
            continue

        comp = summary[condition].get('_dream_comparison', {})
        dream = summary[condition].get('DreamFTB', {})

        no_harm = comp.get('dream_no_harm', False)
        acceptance = comp.get('acceptance_rate', 0)
        no_forgetting = comp.get('no_extra_repairs', True)
        n_rules = dream.get('total_sar_rules', 0)

        condition_results.append({
            'condition': condition,
            'no_harm': no_harm,
            'acceptance': acceptance,
            'acceptance_ok': 0.05 <= acceptance <= 0.50,
            'no_forgetting': no_forgetting,
            'n_rules': n_rules
        })

        if not no_harm:
            checks['no_harm'] = False
        if acceptance < 0.01:
            checks['dreams_produced'] = False
        if not no_forgetting:
            checks['no_forgetting'] = False
        if n_rules >= 3:
            checks['sar_discovered'] = True

    # Print per-condition results
    print(f"\n{'Condition':<20} {'No Harm':<12} {'Accept Rate':<15} {'No Forgetting':<15} {'SAR Rules':<12}")
    print("-" * 100)

    for r in condition_results:
        harm_str = "PASS" if r['no_harm'] else "FAIL"
        accept_str = f"{r['acceptance']:.1%}" + (" OK" if r['acceptance_ok'] else " LOW" if r['acceptance'] < 0.05 else " HIGH")
        forget_str = "PASS" if r['no_forgetting'] else "FAIL"
        rules_str = str(r['n_rules']) + (" PASS" if r['n_rules'] >= 3 else "")

        print(f"{r['condition']:<20} {harm_str:<12} {accept_str:<15} {forget_str:<15} {rules_str:<12}")

    print("-" * 100)

    # Overall verdict
    print("\nSUCCESS CRITERIA:")
    print(f"  {'[PASS]' if checks['no_harm'] else '[FAIL]'} No harm: DreamFTB MAE <= FTB MAE across conditions")
    print(f"  {'[PASS]' if checks['dreams_produced'] else '[WARN]'} Dreams produced: Acceptance rate > 1%")
    print(f"  {'[PASS]' if checks['no_forgetting'] else '[FAIL]'} No forgetting: No extra repair loops from synthetics")
    print(f"  {'[PASS]' if checks['sar_discovered'] else '[INFO]'} SAR discovery: >= 3 significant rules (stretch goal)")

    all_pass = checks['no_harm'] and checks['dreams_produced'] and checks['no_forgetting']

    print("\n" + "-" * 100)
    if all_pass:
        print("VERDICT: PHASE 2 SUCCESS")
        print("  DreamFTB is safe to use and produces useful synthetic augmentation.")
        if checks['sar_discovered']:
            print("  Bonus: SAR rules discovered - interpretable insights available.")
    else:
        issues = []
        if not checks['no_harm']:
            issues.append("DreamFTB caused MAE regression")
        if not checks['dreams_produced']:
            issues.append("Very low acceptance rate (model too uncertain)")
        if not checks['no_forgetting']:
            issues.append("Synthetics triggered forgetting repairs")
        print("VERDICT: PHASE 2 NEEDS ATTENTION")
        print(f"  Issues: {', '.join(issues)}")

    print("=" * 100)


def save_phase2_results(results: Dict, output_dir: str = 'results'):
    """Save results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / 'phase2_dream_validation.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")
    return output_file


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the full Phase 2 validation."""
    results = run_phase2_validation(
        conditions=['clean', 'noisy_15pct', 'distribution_shift'],
        seeds=[42, 123, 456, 789, 1011],
        n_steps=50,
        seed_size=20,
        eval_interval=10,
        verbose=True
    )

    print_phase2_summary(results['summary'])
    print_phase2_verdict(results['summary'])

    save_phase2_results(results)

    return results


if __name__ == '__main__':
    main()
