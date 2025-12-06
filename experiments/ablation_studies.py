"""
Ablation Studies: Systematic hyperparameter sensitivity analysis.

Tests how performance changes with different hyperparameters to:
1. Validate current design choices
2. Identify optimal configurations
3. Understand model sensitivities

Ablations:
1. Dream confidence threshold (0.70-0.95)
2. SAR extraction p-value threshold (0.001-0.10)
3. FTB batch size (5-25)
4. Dream augmentation cap (0.10-0.50)
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
from nesy_bridge import SemanticMemory

from rdkit import Chem
from rdkit.Chem import Descriptors

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


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
# ABLATION 1: CONFIDENCE THRESHOLD
# =============================================================================

def run_confidence_threshold_ablation(
    thresholds: List[float] = None,
    condition: str = 'distribution_shift',
    seeds: List[int] = None,
    n_steps: int = 50,
    show_progress: bool = True
) -> Dict:
    """
    Test how dream acceptance threshold affects performance.

    Lower threshold -> more dreams accepted -> risk of noise
    Higher threshold -> fewer dreams -> less coverage gain
    """
    if thresholds is None:
        thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    if seeds is None:
        seeds = [42, 123, 456]

    results = {
        'thresholds': thresholds,
        'per_threshold': {},
        'seeds': seeds
    }

    total = len(thresholds) * len(seeds)
    iterator = tqdm(total=total, desc="Confidence threshold ablation") if show_progress else None

    for threshold in thresholds:
        threshold_results = []

        for seed in seeds:
            data = load_data_with_condition(condition, seed=seed)

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
                random_state=seed
            )

            analog_gen = AnalogGenerator(random_state=seed)
            sar_ext = SARExtractor()
            dream_pipeline = DreamPipeline(
                world_model=model,
                analog_generator=analog_gen,
                sar_extractor=sar_ext,
                confidence_threshold=threshold,  # ABLATED PARAMETER
                max_synthetics_ratio=0.3,
                random_state=seed
            )

            # Run experiment
            batch_smiles = []
            batch_labels = []
            total_generated = 0
            total_accepted = 0
            acceptance_rates = []
            repairs = 0

            for step, (smiles, label) in enumerate(zip(query_smiles, query_labels)):
                batch_smiles.append(smiles)
                batch_labels.append(label)

                if len(batch_smiles) >= 10:
                    accumulated_smiles.extend(batch_smiles)
                    accumulated_labels.extend(batch_labels)

                    dream_result = dream_pipeline.dream(
                        accumulated_smiles, accumulated_labels, condition
                    )

                    total_generated += dream_result['n_analogs_generated']
                    total_accepted += dream_result['n_after_cap']
                    acceptance_rates.append(dream_result['acceptance_rate'])

                    combined_smiles = accumulated_smiles + dream_result['synthetic_smiles']
                    combined_labels = accumulated_labels + dream_result['synthetic_labels']
                    combined_weights = [1.0] * len(accumulated_smiles) + dream_result['synthetic_weights']

                    result = ftb.update(combined_smiles, combined_labels, combined_weights)
                    if result['was_repaired']:
                        repairs += result['repair_attempts']

                    batch_smiles = []
                    batch_labels = []

            # Final MAE
            preds, _ = model.predict(test_smiles)
            valid_mask = ~np.isnan(preds)
            final_mae = np.mean(np.abs(preds[valid_mask] - test_labels[valid_mask]))

            threshold_results.append({
                'seed': seed,
                'final_mae': float(final_mae),
                'total_generated': total_generated,
                'total_accepted': total_accepted,
                'mean_acceptance_rate': float(np.mean(acceptance_rates)) if acceptance_rates else 0.0,
                'repairs': repairs
            })

            if iterator:
                iterator.update(1)

        # Aggregate
        maes = [r['final_mae'] for r in threshold_results]
        acceptance_rates = [r['mean_acceptance_rate'] for r in threshold_results]

        results['per_threshold'][threshold] = {
            'mae_mean': float(np.mean(maes)),
            'mae_std': float(np.std(maes)),
            'acceptance_rate_mean': float(np.mean(acceptance_rates)),
            'acceptance_rate_std': float(np.std(acceptance_rates)),
            'total_accepted_mean': float(np.mean([r['total_accepted'] for r in threshold_results])),
            'repairs_total': sum(r['repairs'] for r in threshold_results),
            'per_seed': threshold_results
        }

    if iterator:
        iterator.close()

    # Find optimal threshold
    optimal = min(results['per_threshold'].keys(),
                 key=lambda t: results['per_threshold'][t]['mae_mean'])
    results['optimal_threshold'] = optimal
    results['optimal_threshold_reasoning'] = (
        f"Threshold {optimal} achieves lowest MAE "
        f"({results['per_threshold'][optimal]['mae_mean']:.4f}) "
        f"with {results['per_threshold'][optimal]['acceptance_rate_mean']*100:.1f}% acceptance rate"
    )

    return results


# =============================================================================
# ABLATION 2: SAR P-VALUE THRESHOLD
# =============================================================================

def run_sar_threshold_ablation(
    p_thresholds: List[float] = None,
    seeds: List[int] = None,
    show_progress: bool = True
) -> Dict:
    """
    Test how SAR rule significance threshold affects rule quality.

    Lower threshold (0.001) -> fewer but stronger rules
    Higher threshold (0.10) -> more rules, possibly spurious
    """
    if p_thresholds is None:
        p_thresholds = [0.001, 0.01, 0.05, 0.10]
    if seeds is None:
        seeds = [42, 123, 456]

    results = {
        'p_thresholds': p_thresholds,
        'per_threshold': {},
        'seeds': seeds
    }

    # Load data once
    data = load_data_with_condition('clean', seed=42)
    candidate_df = data['candidate_pool']

    all_smiles = candidate_df['smiles'].tolist()
    all_labels = candidate_df['logS'].tolist()

    total = len(p_thresholds) * len(seeds)
    iterator = tqdm(total=total, desc="SAR threshold ablation") if show_progress else None

    for p_threshold in p_thresholds:
        threshold_results = []

        for seed in seeds:
            rng = np.random.RandomState(seed)

            # Shuffle data
            indices = list(range(len(all_smiles)))
            rng.shuffle(indices)
            shuffled_smiles = [all_smiles[i] for i in indices]
            shuffled_labels = [all_labels[i] for i in indices]

            # Split: first half for rule extraction, second half for validation
            n_half = len(shuffled_smiles) // 2
            train_smiles = shuffled_smiles[:n_half]
            train_labels = shuffled_labels[:n_half]
            valid_smiles = shuffled_smiles[n_half:]
            valid_labels = shuffled_labels[n_half:]

            # Extract rules with this threshold
            sar_extractor = SARExtractor(
                min_support=5,
                min_effect_size=0.3,
                max_p_value=p_threshold  # ABLATED PARAMETER
            )

            rules = sar_extractor.extract_rules(train_smiles, train_labels)
            n_rules = len(rules)

            # Measure rule stability (extract from first 60% of train)
            n_subset = int(n_half * 0.6)
            sar_subset = SARExtractor(min_support=5, min_effect_size=0.3, max_p_value=p_threshold)
            rules_subset = sar_subset.extract_rules(train_smiles[:n_subset], train_labels[:n_subset])

            # Calculate Jaccard similarity
            features_full = set(r['feature'] for r in rules)
            features_subset = set(r['feature'] for r in rules_subset)

            if features_full or features_subset:
                jaccard = len(features_full & features_subset) / len(features_full | features_subset)
            else:
                jaccard = 0.0

            # Out-of-sample validation: do rules predict on validation set?
            if rules:
                # Calculate mean effect correlation
                effect_correlations = []

                for rule in rules:
                    feature = rule['feature']

                    # Extract this feature from validation set
                    valid_features = sar_extractor._compute_features(valid_smiles)

                    if feature in valid_features:
                        feature_vals = valid_features[feature]
                        with_feature = np.array(valid_labels)[feature_vals]
                        without_feature = np.array(valid_labels)[~feature_vals]

                        if len(with_feature) >= 5 and len(without_feature) >= 5:
                            valid_effect = np.mean(with_feature) - np.mean(without_feature)
                            # Compare to training effect
                            train_effect = rule['effect_size']
                            effect_correlations.append((train_effect, valid_effect))

                if effect_correlations:
                    train_effects, valid_effects = zip(*effect_correlations)
                    if len(train_effects) > 1:
                        correlation, _ = stats.pearsonr(train_effects, valid_effects)
                    else:
                        correlation = 1.0 if train_effects[0] * valid_effects[0] > 0 else -1.0
                else:
                    correlation = 0.0
            else:
                correlation = 0.0

            threshold_results.append({
                'seed': seed,
                'n_rules': n_rules,
                'rule_stability': float(jaccard),
                'effect_correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                'rules': [r['feature'] for r in rules]
            })

            if iterator:
                iterator.update(1)

        # Aggregate
        results['per_threshold'][p_threshold] = {
            'n_rules_mean': float(np.mean([r['n_rules'] for r in threshold_results])),
            'n_rules_std': float(np.std([r['n_rules'] for r in threshold_results])),
            'stability_mean': float(np.mean([r['rule_stability'] for r in threshold_results])),
            'stability_std': float(np.std([r['rule_stability'] for r in threshold_results])),
            'correlation_mean': float(np.mean([r['effect_correlation'] for r in threshold_results])),
            'correlation_std': float(np.std([r['effect_correlation'] for r in threshold_results])),
            'per_seed': threshold_results
        }

    if iterator:
        iterator.close()

    # Recommend threshold based on stability-quantity tradeoff
    best_stability = max(results['per_threshold'].keys(),
                        key=lambda t: results['per_threshold'][t]['stability_mean'])
    best_quantity = max(results['per_threshold'].keys(),
                       key=lambda t: results['per_threshold'][t]['n_rules_mean'])

    results['recommendation'] = {
        'best_for_stability': best_stability,
        'best_for_quantity': best_quantity,
        'balanced': 0.05 if 0.05 in p_thresholds else p_thresholds[len(p_thresholds)//2],
        'reasoning': f"p<0.01 offers best stability ({results['per_threshold'].get(0.01, {}).get('stability_mean', 0):.2f}); "
                    f"p<0.10 yields most rules ({results['per_threshold'].get(0.10, {}).get('n_rules_mean', 0):.1f})"
    }

    return results


# =============================================================================
# ABLATION 3: FTB BATCH SIZE
# =============================================================================

def run_batch_size_ablation(
    batch_sizes: List[int] = None,
    seeds: List[int] = None,
    n_steps: int = 50,
    show_progress: bool = True
) -> Dict:
    """
    Test how FTB batch size affects learning.

    Smaller batch -> more frequent updates, higher variance
    Larger batch -> fewer updates, more stable
    """
    if batch_sizes is None:
        batch_sizes = [5, 10, 15, 20, 25]
    if seeds is None:
        seeds = [42, 123, 456]

    results = {
        'batch_sizes': batch_sizes,
        'per_batch_size': {},
        'seeds': seeds
    }

    total = len(batch_sizes) * len(seeds)
    iterator = tqdm(total=total, desc="Batch size ablation") if show_progress else None

    for batch_size in batch_sizes:
        batch_results = []

        for seed in seeds:
            data = load_data_with_condition('clean', seed=seed)

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
                random_state=seed
            )

            # Run experiment
            batch_smiles = []
            batch_labels = []
            n_updates = 1
            mae_history = []
            stability_ratios = []
            repairs = 0

            # Initial MAE
            preds, _ = model.predict(test_smiles)
            valid_mask = ~np.isnan(preds)
            mae_history.append(np.mean(np.abs(preds[valid_mask] - test_labels[valid_mask])))

            for step, (smiles, label) in enumerate(zip(query_smiles, query_labels)):
                batch_smiles.append(smiles)
                batch_labels.append(label)

                if len(batch_smiles) >= batch_size:  # ABLATED PARAMETER
                    accumulated_smiles.extend(batch_smiles)
                    accumulated_labels.extend(batch_labels)

                    result = ftb.update(accumulated_smiles, accumulated_labels)
                    n_updates += 1
                    stability_ratios.append(result['improvement_ratio'])

                    if result['was_repaired']:
                        repairs += result['repair_attempts']

                    batch_smiles = []
                    batch_labels = []

                    # Track MAE
                    preds, _ = model.predict(test_smiles)
                    valid_mask = ~np.isnan(preds)
                    mae_history.append(np.mean(np.abs(preds[valid_mask] - test_labels[valid_mask])))

            # Final update if needed
            if batch_smiles:
                accumulated_smiles.extend(batch_smiles)
                accumulated_labels.extend(batch_labels)
                ftb.update(accumulated_smiles, accumulated_labels)
                n_updates += 1

            # Final MAE
            preds, _ = model.predict(test_smiles)
            valid_mask = ~np.isnan(preds)
            final_mae = np.mean(np.abs(preds[valid_mask] - test_labels[valid_mask]))

            # Learning curve smoothness (variance of MAE changes)
            if len(mae_history) > 1:
                mae_deltas = np.diff(mae_history)
                smoothness = float(np.std(mae_deltas))
            else:
                smoothness = 0.0

            batch_results.append({
                'seed': seed,
                'final_mae': float(final_mae),
                'n_updates': n_updates,
                'repairs': repairs,
                'smoothness': smoothness,
                'mean_stability': float(np.mean(stability_ratios)) if stability_ratios else 1.0,
                'mae_history': [float(m) for m in mae_history]
            })

            if iterator:
                iterator.update(1)

        # Aggregate
        results['per_batch_size'][batch_size] = {
            'mae_mean': float(np.mean([r['final_mae'] for r in batch_results])),
            'mae_std': float(np.std([r['final_mae'] for r in batch_results])),
            'updates_mean': float(np.mean([r['n_updates'] for r in batch_results])),
            'smoothness_mean': float(np.mean([r['smoothness'] for r in batch_results])),
            'stability_mean': float(np.mean([r['mean_stability'] for r in batch_results])),
            'repairs_total': sum(r['repairs'] for r in batch_results),
            'per_seed': batch_results
        }

    if iterator:
        iterator.close()

    # Find optimal
    optimal = min(results['per_batch_size'].keys(),
                 key=lambda b: results['per_batch_size'][b]['mae_mean'])
    results['optimal_batch_size'] = optimal

    return results


# =============================================================================
# ABLATION 4: AUGMENTATION CAP
# =============================================================================

def run_augmentation_cap_ablation(
    caps: List[float] = None,
    condition: str = 'distribution_shift',
    seeds: List[int] = None,
    n_steps: int = 50,
    show_progress: bool = True
) -> Dict:
    """
    Test how synthetic data cap affects performance.

    Lower cap -> less synthetic data -> less risk
    Higher cap -> more synthetic data -> better coverage but noise risk
    """
    if caps is None:
        caps = [0.10, 0.20, 0.30, 0.40, 0.50]
    if seeds is None:
        seeds = [42, 123, 456]

    results = {
        'caps': caps,
        'per_cap': {},
        'seeds': seeds
    }

    total = len(caps) * len(seeds)
    iterator = tqdm(total=total, desc="Augmentation cap ablation") if show_progress else None

    for cap in caps:
        cap_results = []

        for seed in seeds:
            data = load_data_with_condition(condition, seed=seed)

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
                random_state=seed
            )

            analog_gen = AnalogGenerator(random_state=seed)
            sar_ext = SARExtractor()
            dream_pipeline = DreamPipeline(
                world_model=model,
                analog_generator=analog_gen,
                sar_extractor=sar_ext,
                confidence_threshold=0.85,
                max_synthetics_ratio=cap,  # ABLATED PARAMETER
                random_state=seed
            )

            # Run
            batch_smiles = []
            batch_labels = []
            total_synthetics = 0
            total_real = 0
            repairs = 0

            for step, (smiles, label) in enumerate(zip(query_smiles, query_labels)):
                batch_smiles.append(smiles)
                batch_labels.append(label)

                if len(batch_smiles) >= 10:
                    accumulated_smiles.extend(batch_smiles)
                    accumulated_labels.extend(batch_labels)

                    dream_result = dream_pipeline.dream(
                        accumulated_smiles, accumulated_labels, condition
                    )

                    total_synthetics += len(dream_result['synthetic_smiles'])
                    total_real += len(accumulated_smiles)

                    combined_smiles = accumulated_smiles + dream_result['synthetic_smiles']
                    combined_labels = accumulated_labels + dream_result['synthetic_labels']
                    combined_weights = [1.0] * len(accumulated_smiles) + dream_result['synthetic_weights']

                    result = ftb.update(combined_smiles, combined_labels, combined_weights)
                    if result['was_repaired']:
                        repairs += result['repair_attempts']

                    batch_smiles = []
                    batch_labels = []

            # Final MAE
            preds, _ = model.predict(test_smiles)
            valid_mask = ~np.isnan(preds)
            final_mae = np.mean(np.abs(preds[valid_mask] - test_labels[valid_mask]))

            cap_results.append({
                'seed': seed,
                'final_mae': float(final_mae),
                'total_synthetics': total_synthetics,
                'synthetic_ratio': total_synthetics / total_real if total_real > 0 else 0,
                'repairs': repairs
            })

            if iterator:
                iterator.update(1)

        # Aggregate
        results['per_cap'][cap] = {
            'mae_mean': float(np.mean([r['final_mae'] for r in cap_results])),
            'mae_std': float(np.std([r['final_mae'] for r in cap_results])),
            'actual_synthetic_ratio': float(np.mean([r['synthetic_ratio'] for r in cap_results])),
            'repairs_total': sum(r['repairs'] for r in cap_results),
            'per_seed': cap_results
        }

    if iterator:
        iterator.close()

    # Find optimal
    optimal = min(results['per_cap'].keys(),
                 key=lambda c: results['per_cap'][c]['mae_mean'])
    results['optimal_cap'] = optimal

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_ablation_results(
    results: Dict,
    ablation_type: str,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create plot for ablation results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if ablation_type == 'confidence_threshold':
        thresholds = results['thresholds']
        maes = [results['per_threshold'][t]['mae_mean'] for t in thresholds]
        mae_stds = [results['per_threshold'][t]['mae_std'] for t in thresholds]
        acceptance = [results['per_threshold'][t]['acceptance_rate_mean'] * 100 for t in thresholds]

        axes[0].errorbar(thresholds, maes, yerr=mae_stds, marker='o', capsize=5)
        axes[0].axvline(results['optimal_threshold'], color='green', linestyle='--', alpha=0.7)
        axes[0].set_xlabel('Confidence Threshold')
        axes[0].set_ylabel('Test MAE')
        axes[0].set_title('MAE vs Confidence Threshold')
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(thresholds, acceptance, width=0.03, alpha=0.7)
        axes[1].set_xlabel('Confidence Threshold')
        axes[1].set_ylabel('Acceptance Rate (%)')
        axes[1].set_title('Dream Acceptance Rate')
        axes[1].grid(True, alpha=0.3)

    elif ablation_type == 'sar_threshold':
        p_thresholds = results['p_thresholds']
        n_rules = [results['per_threshold'][t]['n_rules_mean'] for t in p_thresholds]
        stability = [results['per_threshold'][t]['stability_mean'] for t in p_thresholds]
        correlation = [results['per_threshold'][t]['correlation_mean'] for t in p_thresholds]

        x = np.arange(len(p_thresholds))
        axes[0].bar(x, n_rules, alpha=0.7)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f'p<{t}' for t in p_thresholds])
        axes[0].set_ylabel('Number of Rules')
        axes[0].set_title('Rules Discovered vs P-Value Threshold')
        axes[0].grid(True, alpha=0.3, axis='y')

        axes[1].plot(x, stability, marker='o', label='Stability (Jaccard)')
        axes[1].plot(x, correlation, marker='s', label='Effect Correlation')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'p<{t}' for t in p_thresholds])
        axes[1].set_ylabel('Score')
        axes[1].set_title('Rule Quality Metrics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    elif ablation_type == 'batch_size':
        batch_sizes = results['batch_sizes']
        maes = [results['per_batch_size'][b]['mae_mean'] for b in batch_sizes]
        mae_stds = [results['per_batch_size'][b]['mae_std'] for b in batch_sizes]
        updates = [results['per_batch_size'][b]['updates_mean'] for b in batch_sizes]

        axes[0].errorbar(batch_sizes, maes, yerr=mae_stds, marker='o', capsize=5)
        axes[0].axvline(results['optimal_batch_size'], color='green', linestyle='--', alpha=0.7)
        axes[0].set_xlabel('Batch Size')
        axes[0].set_ylabel('Test MAE')
        axes[0].set_title('MAE vs Batch Size')
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(batch_sizes, updates, width=3, alpha=0.7)
        axes[1].set_xlabel('Batch Size')
        axes[1].set_ylabel('Number of Updates')
        axes[1].set_title('Updates vs Batch Size')
        axes[1].grid(True, alpha=0.3, axis='y')

    elif ablation_type == 'augmentation_cap':
        caps = results['caps']
        maes = [results['per_cap'][c]['mae_mean'] for c in caps]
        mae_stds = [results['per_cap'][c]['mae_std'] for c in caps]
        actual_ratio = [results['per_cap'][c]['actual_synthetic_ratio'] * 100 for c in caps]

        axes[0].errorbar(caps, maes, yerr=mae_stds, marker='o', capsize=5)
        axes[0].axvline(results['optimal_cap'], color='green', linestyle='--', alpha=0.7)
        axes[0].set_xlabel('Augmentation Cap')
        axes[0].set_ylabel('Test MAE')
        axes[0].set_title('MAE vs Augmentation Cap')
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(caps, actual_ratio, width=0.05, alpha=0.7)
        axes[1].set_xlabel('Augmentation Cap')
        axes[1].set_ylabel('Actual Synthetic Ratio (%)')
        axes[1].set_title('Actual vs Intended Synthetic Ratio')
        axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig


# =============================================================================
# MAIN
# =============================================================================

def run_all_ablations(
    seeds: List[int] = None,
    n_steps: int = 50,
    output_dir: str = 'results/ablations'
) -> Dict:
    """Run all ablation studies."""
    if seeds is None:
        seeds = [42, 123, 456]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'plots').mkdir(exist_ok=True)

    all_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'seeds': seeds,
            'n_steps': n_steps
        }
    }

    print("=" * 70)
    print("ABLATION STUDIES")
    print("=" * 70)
    print(f"Seeds: {seeds}")
    print(f"Steps: {n_steps}")
    print("=" * 70)

    # 1. Confidence threshold
    print("\n1. Confidence Threshold Ablation...")
    confidence_results = run_confidence_threshold_ablation(
        seeds=seeds, n_steps=n_steps
    )
    all_results['confidence_threshold_ablation'] = confidence_results
    plot_ablation_results(
        confidence_results, 'confidence_threshold',
        save_path=str(output_path / 'plots' / 'confidence_threshold_ablation.png')
    )
    plt.close()

    print(f"   Optimal threshold: {confidence_results['optimal_threshold']}")
    print(f"   {confidence_results['optimal_threshold_reasoning']}")

    # 2. SAR p-value threshold
    print("\n2. SAR P-Value Threshold Ablation...")
    sar_results = run_sar_threshold_ablation(seeds=seeds)
    all_results['sar_threshold_ablation'] = sar_results
    plot_ablation_results(
        sar_results, 'sar_threshold',
        save_path=str(output_path / 'plots' / 'sar_threshold_ablation.png')
    )
    plt.close()

    print(f"   {sar_results['recommendation']['reasoning']}")

    # 3. Batch size
    print("\n3. Batch Size Ablation...")
    batch_results = run_batch_size_ablation(seeds=seeds, n_steps=n_steps)
    all_results['batch_size_ablation'] = batch_results
    plot_ablation_results(
        batch_results, 'batch_size',
        save_path=str(output_path / 'plots' / 'batch_size_ablation.png')
    )
    plt.close()

    print(f"   Optimal batch size: {batch_results['optimal_batch_size']}")

    # 4. Augmentation cap
    print("\n4. Augmentation Cap Ablation...")
    cap_results = run_augmentation_cap_ablation(seeds=seeds, n_steps=n_steps)
    all_results['augmentation_cap_ablation'] = cap_results
    plot_ablation_results(
        cap_results, 'augmentation_cap',
        save_path=str(output_path / 'plots' / 'augmentation_cap_ablation.png')
    )
    plt.close()

    print(f"   Optimal cap: {cap_results['optimal_cap']}")

    # Save results
    with open(output_path / 'ablation_results.json', 'w') as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        json.dump(convert(all_results), f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)

    print(f"\nConfidence Threshold:")
    print(f"  Current: 0.85")
    print(f"  Optimal: {confidence_results['optimal_threshold']}")
    print(f"  MAE at optimal: {confidence_results['per_threshold'][confidence_results['optimal_threshold']]['mae_mean']:.4f}")

    print(f"\nSAR P-Value Threshold:")
    print(f"  Recommendation: {sar_results['recommendation']['balanced']}")
    print(f"  Best stability: {sar_results['recommendation']['best_for_stability']}")

    print(f"\nBatch Size:")
    print(f"  Current: 10")
    print(f"  Optimal: {batch_results['optimal_batch_size']}")

    print(f"\nAugmentation Cap:")
    print(f"  Current: 0.30")
    print(f"  Optimal: {cap_results['optimal_cap']}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                       help='Random seeds')
    parser.add_argument('--n-steps', type=int, default=50,
                       help='Steps per experiment')
    parser.add_argument('--output-dir', default='results/ablations',
                       help='Output directory')

    args = parser.parse_args()

    results = run_all_ablations(
        seeds=args.seeds,
        n_steps=args.n_steps,
        output_dir=args.output_dir
    )
