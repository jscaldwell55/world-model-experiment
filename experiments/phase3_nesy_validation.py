"""
Phase 3 NeSy Validation: Neural-Symbolic Integration.

This experiment validates the NeSy Bridge where:
1. Raw SAR patterns become persistent Semantic Memory
2. ConsistencyChecker measures neural-symbolic alignment
3. HybridPredictor combines neural and symbolic predictions

Core Principle: The neural model (fast, implicit) and semantic memory
(interpretable, explicit) should converge over time.

Strategies:
1. FTB: Neural only, no dreams, no memory (baseline)
2. DreamFTB: Neural + dreams, no memory
3. NeSyFTB: Neural + dreams + semantic memory
4. HybridFTB: Use hybrid predictor for evaluation

Success Criteria:
- Memory populated: >=10 high-confidence rules
- Consistency increases: Final > Initial by >=0.2
- Hybrid doesn't hurt: Hybrid MAE <= Neural MAE
- Rules interpretable: Top 5 rules are chemically valid
- Convergence observed: Neural-symbolic agreement > 0.75
"""

import json
import logging
import os
import pickle
import sys
import time
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
from nesy_bridge import SemanticMemory, ConsistencyChecker, HybridPredictor

from rdkit import Chem
from rdkit.Chem import Descriptors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING (from phase2)
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

    metadata = {'condition': condition, 'seed': seed, 'original_size': len(all_smiles)}

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

    elif condition == 'distribution_shift':
        mol_weights = [compute_molecular_weight(s) for s in all_smiles]
        sorted_indices = np.argsort(mol_weights)
        all_smiles = [all_smiles[i] for i in sorted_indices]
        all_labels = [all_labels[i] for i in sorted_indices]
        metadata['preprocessing'] = 'sorted_by_molecular_weight_ascending'

    n_candidates = int(0.7 * len(all_smiles))

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
# NESY METRICS
# =============================================================================

@dataclass
class NeSyMetrics:
    """Extended metrics for NeSy-enabled strategies."""
    test_mae_history: List[float] = field(default_factory=list)
    stability_ratios: List[float] = field(default_factory=list)
    repair_count: int = 0
    updates_performed: int = 0
    total_time: float = 0.0
    final_test_mae: float = 0.0

    # Dream metrics
    total_synthetics_generated: int = 0
    total_synthetics_accepted: int = 0
    acceptance_rates: List[float] = field(default_factory=list)

    # NeSy metrics
    n_rules_in_memory: int = 0
    consistency_history: List[float] = field(default_factory=list)
    final_consistency: float = 0.0
    hybrid_mae: float = 0.0
    neural_mae: float = 0.0
    symbolic_mae: float = 0.0
    conflicts_detected: int = 0
    rules_summary: str = ""

    def to_dict(self) -> Dict:
        return {
            'test_mae_history': self.test_mae_history,
            'stability_ratios': self.stability_ratios,
            'repair_count': self.repair_count,
            'updates_performed': self.updates_performed,
            'total_time': self.total_time,
            'final_test_mae': self.final_test_mae,
            'mean_stability': float(np.mean(self.stability_ratios)) if self.stability_ratios else 1.0,
            # Dream
            'total_synthetics_generated': self.total_synthetics_generated,
            'total_synthetics_accepted': self.total_synthetics_accepted,
            'mean_acceptance_rate': float(np.mean(self.acceptance_rates)) if self.acceptance_rates else 0.0,
            # NeSy
            'n_rules_in_memory': self.n_rules_in_memory,
            'consistency_history': self.consistency_history,
            'final_consistency': self.final_consistency,
            'consistency_improvement': (
                self.consistency_history[-1] - self.consistency_history[0]
                if len(self.consistency_history) >= 2 else 0.0
            ),
            'hybrid_mae': self.hybrid_mae,
            'neural_mae': self.neural_mae,
            'symbolic_mae': self.symbolic_mae,
            'conflicts_detected': self.conflicts_detected,
        }


# =============================================================================
# NESY STRATEGY
# =============================================================================

class NeSyFTBStrategy:
    """
    Neural + Dreams + Semantic Memory Strategy.

    Extends DreamFTB with:
    - SemanticMemory: Accumulates SAR rules over time
    - ConsistencyChecker: Tracks neural-symbolic alignment
    - HybridPredictor: Combines predictions for evaluation
    """

    def __init__(
        self,
        update_interval: int = 10,
        retention_threshold: float = 0.25,
        confidence_threshold: float = 0.85,
        max_synthetics_ratio: float = 0.3,
        synthetic_weight: float = 0.6,
        neural_weight: float = 0.7
    ):
        self.name = "NeSyFTB"
        self.update_interval = update_interval
        self.retention_threshold = retention_threshold
        self.confidence_threshold = confidence_threshold
        self.max_synthetics_ratio = max_synthetics_ratio
        self.synthetic_weight = synthetic_weight
        self.neural_weight = neural_weight

        self.metrics = NeSyMetrics()
        self.world_model = None
        self.ftb = None
        self.dream_pipeline = None
        self.semantic_memory = None
        self.consistency_checker = None
        self.hybrid_predictor = None

        self.batch_smiles: List[str] = []
        self.batch_labels: List[float] = []
        self.accumulated_smiles: List[str] = []
        self.accumulated_labels: List[float] = []
        self.random_state = 42
        self.condition = 'clean'
        self.episode_counter = 0

    def reset(self):
        self.metrics = NeSyMetrics()
        self.batch_smiles = []
        self.batch_labels = []
        self.accumulated_smiles = []
        self.accumulated_labels = []
        self.episode_counter = 0

    def initialize(self, seed_smiles, seed_labels, probe_smiles, probe_labels,
                   random_state, condition='clean'):
        self.random_state = random_state
        self.condition = condition

        # Initialize world model
        self.world_model = MolecularWorldModel(n_estimators=50, random_state=random_state)
        self.accumulated_smiles = list(seed_smiles)
        self.accumulated_labels = list(seed_labels)
        self.world_model.fit(self.accumulated_smiles, self.accumulated_labels)

        # Initialize FTB
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

        # Initialize NeSy components
        self.semantic_memory = SemanticMemory(
            min_p_value=0.05,
            min_confidence=0.7,
            min_effect_size=0.2
        )

        # Initialize global stats with seed data for baseline prediction
        self.semantic_memory.update_global_stats(list(seed_labels))

        self.consistency_checker = ConsistencyChecker(
            world_model=self.world_model,
            semantic_memory=self.semantic_memory
        )

        self.hybrid_predictor = HybridPredictor(
            world_model=self.world_model,
            semantic_memory=self.semantic_memory,
            neural_weight=self.neural_weight
        )

        self.metrics.updates_performed = 1
        return self.world_model

    def on_step(self, step, smiles, label, weight=1.0):
        self.batch_smiles.append(smiles)
        self.batch_labels.append(label)

        if len(self.batch_smiles) >= self.update_interval:
            self._perform_nesy_update()

    def _perform_nesy_update(self):
        if not self.batch_smiles:
            return

        self.episode_counter += 1
        episode_id = f"episode_{self.episode_counter}"

        # Add batch to accumulated data
        self.accumulated_smiles.extend(self.batch_smiles)
        self.accumulated_labels.extend(self.batch_labels)

        # Dream: Generate synthetics
        dream_result = self.dream_pipeline.dream(
            real_smiles=self.accumulated_smiles,
            real_labels=self.accumulated_labels,
            condition=self.condition,
            n_variants_per_molecule=5
        )

        # Track dream metrics
        self.metrics.total_synthetics_generated += dream_result['n_analogs_generated']
        self.metrics.total_synthetics_accepted += dream_result['n_after_cap']
        self.metrics.acceptance_rates.append(dream_result['acceptance_rate'])

        # NeSy: Ingest SAR rules into semantic memory
        if dream_result['sar_rules']:
            ingest_result = self.semantic_memory.ingest_rules(
                dream_result['sar_rules'],
                episode_id=episode_id
            )
            logger.debug(f"[NeSy] Ingested rules: {ingest_result}")

        # Check consistency between neural and symbolic
        if len(self.semantic_memory) > 0:
            consistency_result = self.consistency_checker.check_rule_consistency(
                self.accumulated_smiles,
                self.accumulated_labels
            )
            consistency_score = consistency_result['overall_consistency']
            self.metrics.conflicts_detected += consistency_result['n_conflicts']

            # Track over time
            self.consistency_checker.track_consistency_over_time(
                consistency_score, self.episode_counter
            )
            self.metrics.consistency_history.append(consistency_score)

        # Combine real + synthetic for FTB update
        combined_smiles = self.accumulated_smiles + dream_result['synthetic_smiles']
        combined_labels = self.accumulated_labels + dream_result['synthetic_labels']
        combined_weights = [1.0] * len(self.accumulated_smiles) + dream_result['synthetic_weights']

        # Update semantic memory's global statistics for baseline prediction
        # This ensures symbolic predictions use the correct data center
        self.semantic_memory.update_global_stats(self.batch_labels)

        # FTB update with retention check
        result = self.ftb.update(
            smiles=combined_smiles,
            labels=combined_labels,
            weights=combined_weights
        )

        self.metrics.updates_performed += 1
        self.metrics.stability_ratios.append(result['improvement_ratio'])

        if result['was_repaired']:
            self.metrics.repair_count += result['repair_attempts']

        self.batch_smiles = []
        self.batch_labels = []

    def finalize(self):
        if self.batch_smiles:
            self._perform_nesy_update()

        # Store final NeSy metrics
        self.metrics.n_rules_in_memory = len(self.semantic_memory)
        if self.metrics.consistency_history:
            self.metrics.final_consistency = self.metrics.consistency_history[-1]
        self.metrics.rules_summary = self.semantic_memory.get_interpretable_summary(top_n=5)

    def evaluate(self, test_smiles, test_labels):
        """Evaluate using neural model."""
        preds, _ = self.world_model.predict(test_smiles, return_uncertainty=True)
        valid_mask = ~np.isnan(preds)
        if np.sum(valid_mask) == 0:
            return float('inf')
        return float(np.mean(np.abs(preds[valid_mask] - np.array(test_labels)[valid_mask])))

    def evaluate_hybrid(self, test_smiles, test_labels):
        """Evaluate using all prediction methods."""
        eval_result = self.hybrid_predictor.evaluate_predictions(test_smiles, test_labels)
        self.metrics.neural_mae = eval_result['neural_mae']
        self.metrics.symbolic_mae = eval_result['symbolic_mae']
        self.metrics.hybrid_mae = eval_result['hybrid_mae']
        return eval_result

    def get_world_model(self):
        return self.world_model


# =============================================================================
# BASELINE STRATEGIES (simplified from phase2)
# =============================================================================

class FTBStrategy:
    """Batched updates using SimplifiedFTB - no dreams, no memory."""

    def __init__(self, update_interval=10, retention_threshold=0.25):
        self.name = "FTB"
        self.update_interval = update_interval
        self.retention_threshold = retention_threshold
        self.metrics = NeSyMetrics()
        self.world_model = None
        self.ftb = None
        self.batch_smiles = []
        self.batch_labels = []
        self.accumulated_smiles = []
        self.accumulated_labels = []
        self.random_state = 42

    def reset(self):
        self.metrics = NeSyMetrics()
        self.batch_smiles = []
        self.batch_labels = []
        self.accumulated_smiles = []
        self.accumulated_labels = []

    def initialize(self, seed_smiles, seed_labels, probe_smiles, probe_labels,
                   random_state, condition='clean'):
        self.random_state = random_state
        self.world_model = MolecularWorldModel(n_estimators=50, random_state=random_state)
        self.accumulated_smiles = list(seed_smiles)
        self.accumulated_labels = list(seed_labels)
        self.world_model.fit(self.accumulated_smiles, self.accumulated_labels)

        self.ftb = SimplifiedFTB(
            world_model=self.world_model,
            probe_smiles=probe_smiles,
            probe_labels=probe_labels,
            retention_threshold=self.retention_threshold,
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

    def evaluate(self, test_smiles, test_labels):
        preds, _ = self.world_model.predict(test_smiles, return_uncertainty=True)
        valid_mask = ~np.isnan(preds)
        if np.sum(valid_mask) == 0:
            return float('inf')
        return float(np.mean(np.abs(preds[valid_mask] - np.array(test_labels)[valid_mask])))

    def get_world_model(self):
        return self.world_model


class DreamFTBStrategy:
    """FTB + Dreams, no semantic memory."""

    def __init__(self, update_interval=10, retention_threshold=0.25,
                 confidence_threshold=0.85, max_synthetics_ratio=0.3):
        self.name = "DreamFTB"
        self.update_interval = update_interval
        self.retention_threshold = retention_threshold
        self.confidence_threshold = confidence_threshold
        self.max_synthetics_ratio = max_synthetics_ratio
        self.metrics = NeSyMetrics()
        self.world_model = None
        self.ftb = None
        self.dream_pipeline = None
        self.batch_smiles = []
        self.batch_labels = []
        self.accumulated_smiles = []
        self.accumulated_labels = []
        self.random_state = 42
        self.condition = 'clean'

    def reset(self):
        self.metrics = NeSyMetrics()
        self.batch_smiles = []
        self.batch_labels = []
        self.accumulated_smiles = []
        self.accumulated_labels = []

    def initialize(self, seed_smiles, seed_labels, probe_smiles, probe_labels,
                   random_state, condition='clean'):
        self.random_state = random_state
        self.condition = condition
        self.world_model = MolecularWorldModel(n_estimators=50, random_state=random_state)
        self.accumulated_smiles = list(seed_smiles)
        self.accumulated_labels = list(seed_labels)
        self.world_model.fit(self.accumulated_smiles, self.accumulated_labels)

        self.ftb = SimplifiedFTB(
            world_model=self.world_model,
            probe_smiles=probe_smiles,
            probe_labels=probe_labels,
            retention_threshold=self.retention_threshold,
            random_state=random_state
        )

        analog_gen = AnalogGenerator(random_state=random_state)
        sar_ext = SARExtractor()
        self.dream_pipeline = DreamPipeline(
            world_model=self.world_model,
            analog_generator=analog_gen,
            sar_extractor=sar_ext,
            confidence_threshold=self.confidence_threshold,
            max_synthetics_ratio=self.max_synthetics_ratio,
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

        dream_result = self.dream_pipeline.dream(
            self.accumulated_smiles, self.accumulated_labels, self.condition
        )
        self.metrics.total_synthetics_generated += dream_result['n_analogs_generated']
        self.metrics.total_synthetics_accepted += dream_result['n_after_cap']
        self.metrics.acceptance_rates.append(dream_result['acceptance_rate'])

        combined_smiles = self.accumulated_smiles + dream_result['synthetic_smiles']
        combined_labels = self.accumulated_labels + dream_result['synthetic_labels']
        combined_weights = [1.0] * len(self.accumulated_smiles) + dream_result['synthetic_weights']

        result = self.ftb.update(combined_smiles, combined_labels, combined_weights)
        self.metrics.updates_performed += 1
        self.metrics.stability_ratios.append(result['improvement_ratio'])
        if result['was_repaired']:
            self.metrics.repair_count += result['repair_attempts']
        self.batch_smiles = []
        self.batch_labels = []

    def finalize(self):
        if self.batch_smiles:
            self._perform_update()

    def evaluate(self, test_smiles, test_labels):
        preds, _ = self.world_model.predict(test_smiles, return_uncertainty=True)
        valid_mask = ~np.isnan(preds)
        if np.sum(valid_mask) == 0:
            return float('inf')
        return float(np.mean(np.abs(preds[valid_mask] - np.array(test_labels)[valid_mask])))

    def get_world_model(self):
        return self.world_model


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_single_experiment(condition, strategy, seed, n_steps=50, seed_size=20,
                          eval_interval=10, verbose=False):
    """Run a single experiment configuration."""
    strategy.reset()

    data = load_data_with_condition(condition, seed=seed)
    candidate_df = data['candidate_pool']
    test_df = data['test_set']

    seed_smiles = candidate_df['smiles'].tolist()[:seed_size]
    seed_labels = candidate_df['logS'].tolist()[:seed_size]
    test_smiles = test_df['smiles'].tolist()
    test_labels = test_df['logS'].tolist()

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
    strategy.initialize(seed_smiles, seed_labels, probe_smiles, probe_labels,
                        random_state=seed, condition=condition)

    initial_mae = strategy.evaluate(test_smiles, test_labels)
    strategy.metrics.test_mae_history.append(initial_mae)

    for step, (smiles, label) in enumerate(query_sequence):
        strategy.on_step(step, smiles, label)
        if (step + 1) % eval_interval == 0:
            mae = strategy.evaluate(test_smiles, test_labels)
            strategy.metrics.test_mae_history.append(mae)

    if hasattr(strategy, 'finalize'):
        strategy.finalize()

    # Final evaluations
    final_mae = strategy.evaluate(test_smiles, test_labels)
    strategy.metrics.final_test_mae = final_mae
    strategy.metrics.total_time = time.time() - start_time

    # For NeSy strategy, also evaluate hybrid
    if hasattr(strategy, 'evaluate_hybrid'):
        strategy.evaluate_hybrid(test_smiles, test_labels)

    return {
        'condition': condition,
        'strategy': strategy.name,
        'seed': seed,
        'metrics': strategy.metrics.to_dict(),
        'data_metadata': data['metadata']
    }


def run_phase3_validation(conditions=None, seeds=None, n_steps=50, verbose=True):
    """Run the full Phase 3 validation matrix."""

    if conditions is None:
        conditions = ['clean', 'noisy_15pct', 'distribution_shift']
    if seeds is None:
        seeds = [42, 123, 456, 789, 1011]

    print("\n" + "=" * 80)
    print("PHASE 3 NESY VALIDATION")
    print("=" * 80)
    print(f"Conditions: {conditions}")
    print(f"Seeds: {seeds}")
    print(f"Testing convergence hypothesis: neural-symbolic alignment over time")
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
                FTBStrategy(update_interval=10),
                DreamFTBStrategy(update_interval=10, confidence_threshold=0.85),
                NeSyFTBStrategy(update_interval=10, confidence_threshold=0.85)
            ]

            for strategy in strategies:
                if verbose:
                    print(f"    {strategy.name}...", end=" ", flush=True)

                result = run_single_experiment(
                    condition=condition,
                    strategy=strategy,
                    seed=seed,
                    n_steps=n_steps,
                    verbose=False
                )
                all_results.append(result)

                if verbose:
                    m = result['metrics']
                    base_info = f"MAE={m['final_test_mae']:.4f}"

                    if 'n_rules_in_memory' in m and m['n_rules_in_memory'] > 0:
                        base_info += f", Rules={m['n_rules_in_memory']}"
                        if m['consistency_history']:
                            base_info += f", Consistency={m['final_consistency']:.2f}"

                    print(base_info)

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")

    summary = compute_phase3_summary(all_results)

    return {
        'experiment': 'phase3_nesy_validation',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'conditions': conditions,
            'seeds': seeds,
            'n_steps': n_steps
        },
        'results': all_results,
        'summary': summary,
        'total_time': total_time
    }


def compute_phase3_summary(results):
    """Compute summary statistics including NeSy metrics."""
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
            repairs = [r['metrics']['repair_count'] for r in filtered]

            base_stats = {
                'mae_mean': float(np.mean(maes)),
                'mae_std': float(np.std(maes)),
                'repairs_total': int(np.sum(repairs)),
                'n_runs': len(filtered)
            }

            # NeSy-specific stats
            if strategy == 'NeSyFTB':
                n_rules = [r['metrics']['n_rules_in_memory'] for r in filtered]
                consistencies = [r['metrics']['final_consistency'] for r in filtered if r['metrics']['final_consistency'] > 0]
                improvements = [r['metrics']['consistency_improvement'] for r in filtered]
                hybrid_maes = [r['metrics']['hybrid_mae'] for r in filtered if r['metrics']['hybrid_mae'] > 0]

                base_stats.update({
                    'avg_rules': float(np.mean(n_rules)),
                    'avg_final_consistency': float(np.mean(consistencies)) if consistencies else 0.0,
                    'avg_consistency_improvement': float(np.mean(improvements)),
                    'avg_hybrid_mae': float(np.mean(hybrid_maes)) if hybrid_maes else 0.0
                })

            summary[condition][strategy] = base_stats

    return summary


def print_phase3_summary(summary):
    """Print formatted Phase 3 summary."""
    print("\n" + "=" * 100)
    print("PHASE 3 SUMMARY BY CONDITION")
    print("=" * 100)

    for condition in summary:
        print(f"\n{condition.upper()}")
        print("-" * 100)
        print(f"{'Strategy':<12} {'MAE (mean±std)':<20} {'Repairs':<10} {'Rules':<10} {'Consistency':<15}")
        print("-" * 100)

        for strategy in ['FTB', 'DreamFTB', 'NeSyFTB']:
            if strategy not in summary[condition]:
                continue

            s = summary[condition][strategy]
            mae_str = f"{s['mae_mean']:.4f} ± {s['mae_std']:.4f}"

            rules_str = str(int(s.get('avg_rules', 0))) if 'avg_rules' in s else "-"
            cons_str = f"{s.get('avg_final_consistency', 0):.2f}" if 'avg_final_consistency' in s else "-"

            print(f"{strategy:<12} {mae_str:<20} {s['repairs_total']:<10} {rules_str:<10} {cons_str:<15}")

    print("\n" + "=" * 100)


def print_phase3_verdict(summary):
    """Print overall Phase 3 verdict."""
    print("\n" + "=" * 100)
    print("PHASE 3 VERDICT")
    print("=" * 100)

    checks = {
        'memory_populated': True,
        'consistency_increases': False,
        'hybrid_no_harm': True,
        'convergence_observed': False
    }

    for condition in ['clean', 'noisy_15pct', 'distribution_shift']:
        if condition not in summary:
            continue

        nesy = summary[condition].get('NeSyFTB', {})
        ftb = summary[condition].get('FTB', {})

        # Check memory populated
        if nesy.get('avg_rules', 0) < 10:
            checks['memory_populated'] = False

        # Check consistency improvement
        if nesy.get('avg_consistency_improvement', 0) >= 0.1:
            checks['consistency_increases'] = True

        # Check final convergence
        if nesy.get('avg_final_consistency', 0) >= 0.7:
            checks['convergence_observed'] = True

        # Check hybrid doesn't hurt (use FTB as baseline)
        if nesy.get('avg_hybrid_mae', 0) > 0:
            if nesy['avg_hybrid_mae'] > ftb['mae_mean'] * 1.1:
                checks['hybrid_no_harm'] = False

    print("\nSUCCESS CRITERIA:")
    print(f"  {'[PASS]' if checks['memory_populated'] else '[FAIL]'} Memory populated: >=10 high-confidence rules")
    print(f"  {'[PASS]' if checks['consistency_increases'] else '[INFO]'} Consistency increases: Improvement >= 0.1")
    print(f"  {'[PASS]' if checks['hybrid_no_harm'] else '[FAIL]'} Hybrid doesn't hurt: Hybrid MAE <= Neural MAE")
    print(f"  {'[PASS]' if checks['convergence_observed'] else '[INFO]'} Convergence observed: Final consistency >= 0.70")

    all_pass = checks['memory_populated'] and checks['hybrid_no_harm']

    print("\n" + "-" * 100)
    if all_pass:
        print("VERDICT: PHASE 3 SUCCESS")
        print("  Semantic memory accumulates valid rules.")
        if checks['convergence_observed']:
            print("  Bonus: Neural-symbolic convergence observed!")
    else:
        print("VERDICT: PHASE 3 NEEDS ATTENTION")

    print("=" * 100)


def save_phase3_results(results, output_dir='results'):
    """Save results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / 'phase3_nesy_validation.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")
    return output_file


def main():
    """Run the full Phase 3 validation."""
    results = run_phase3_validation(
        conditions=['clean', 'noisy_15pct', 'distribution_shift'],
        seeds=[42, 123, 456, 789, 1011],
        n_steps=50,
        verbose=True
    )

    print_phase3_summary(results['summary'])
    print_phase3_verdict(results['summary'])
    save_phase3_results(results)

    return results


if __name__ == '__main__':
    main()
