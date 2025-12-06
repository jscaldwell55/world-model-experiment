"""
Enhanced Consistency Validation: Richer neural-symbolic metrics.

The current "100% consistency" metric is too coarse. This module implements:

1. Effect Magnitude Correlation: Do symbolic rules capture neural model behavior?
2. Out-of-Sample Rule Validation: Do rules generalize?
3. Rule Stability Across Seeds: Are rules robust?
4. Intervention Testing: Do rules predict on designed molecules?

These metrics validate that the semantic memory contains meaningful knowledge.
"""

import json
import logging
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from molecular_world_model import MolecularWorldModel
from dream_state import SARExtractor
from nesy_bridge import SemanticMemory

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_esol_data(data_path: str = 'data/esol_processed.pkl') -> Dict:
    """Load processed ESOL data."""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def compute_molecular_features(smiles: str) -> Dict[str, bool]:
    """Extract binary features from molecule for rule matching."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    features = {}

    # Atom presence
    atoms = set(atom.GetAtomicNum() for atom in mol.GetAtoms())
    features['has_fluorine'] = 9 in atoms
    features['has_chlorine'] = 17 in atoms
    features['has_bromine'] = 35 in atoms
    features['has_nitrogen'] = 7 in atoms
    features['has_sulfur'] = 16 in atoms
    features['has_oxygen'] = 8 in atoms

    # Ring features
    n_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
    n_aliphatic = rdMolDescriptors.CalcNumAliphaticRings(mol)
    ring_info = mol.GetRingInfo()

    features['has_aromatic_ring'] = n_aromatic > 0
    features['has_aliphatic_ring'] = n_aliphatic > 0
    features['has_multiple_rings'] = ring_info.NumRings() > 1

    # Property bins
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        n_rotatable = Descriptors.NumRotatableBonds(mol)
        n_hbd = Descriptors.NumHDonors(mol)
        n_hba = Descriptors.NumHAcceptors(mol)

        features['mw_over_300'] = mw > 300
        features['mw_over_400'] = mw > 400
        features['logp_over_2'] = logp > 2
        features['logp_over_3'] = logp > 3
        features['logp_negative'] = logp < 0
        features['tpsa_over_50'] = tpsa > 50
        features['tpsa_over_100'] = tpsa > 100
        features['n_rotatable_over_5'] = n_rotatable > 5
        features['n_hbd_over_2'] = n_hbd > 2
        features['n_hba_over_5'] = n_hba > 5
    except Exception:
        pass

    return features


# =============================================================================
# METRIC 1: EFFECT MAGNITUDE CORRELATION
# =============================================================================

def compute_effect_magnitude_correlation(
    model: MolecularWorldModel,
    rules: List[Dict],
    smiles_list: List[str],
    labels: List[float]
) -> Dict:
    """
    Compare neural model's implicit feature effects to symbolic rule effects.

    For each rule feature:
    1. Get neural model predictions for molecules with/without feature
    2. Compute neural model's implicit effect (pred_with - pred_without)
    3. Compare to symbolic rule's effect size

    Returns correlation across all rules.
    """
    neural_effects = []
    symbolic_effects = []
    feature_names = []

    # Get predictions for all molecules
    preds, _ = model.predict(smiles_list, return_uncertainty=True)

    # Compute features for all molecules
    all_features = [compute_molecular_features(s) for s in smiles_list]

    for rule in rules:
        feature = rule['feature']
        symbolic_effect = rule['effect_size']

        # Find molecules with/without this feature
        with_feature_idx = [i for i, f in enumerate(all_features)
                          if f.get(feature, False)]
        without_feature_idx = [i for i, f in enumerate(all_features)
                              if not f.get(feature, False)]

        if len(with_feature_idx) < 5 or len(without_feature_idx) < 5:
            continue

        # Get neural predictions for each group
        preds_with = [preds[i] for i in with_feature_idx if not np.isnan(preds[i])]
        preds_without = [preds[i] for i in without_feature_idx if not np.isnan(preds[i])]

        if len(preds_with) < 3 or len(preds_without) < 3:
            continue

        # Compute neural model's implicit effect
        neural_effect = np.mean(preds_with) - np.mean(preds_without)

        neural_effects.append(neural_effect)
        symbolic_effects.append(symbolic_effect)
        feature_names.append(feature)

    if len(neural_effects) < 2:
        return {
            'correlation': 0.0,
            'p_value': 1.0,
            'n_rules_compared': len(neural_effects),
            'error': 'Insufficient rules for correlation'
        }

    # Compute correlation
    correlation, p_value = stats.pearsonr(neural_effects, symbolic_effects)

    return {
        'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
        'p_value': float(p_value) if not np.isnan(p_value) else 1.0,
        'n_rules_compared': len(neural_effects),
        'neural_effects': [float(e) for e in neural_effects],
        'symbolic_effects': [float(e) for e in symbolic_effects],
        'features': feature_names
    }


# =============================================================================
# METRIC 2: OUT-OF-SAMPLE RULE VALIDATION
# =============================================================================

def compute_out_of_sample_validation(
    smiles_list: List[str],
    labels: List[float],
    train_fraction: float = 0.5,
    seed: int = 42
) -> Dict:
    """
    Test if rules from one data split predict on another.

    1. Split data 50/50
    2. Extract SAR rules from first half
    3. Use rules to predict second half
    4. Compare symbolic predictions to actual values

    This tests generalization of discovered rules.
    """
    rng = np.random.RandomState(seed)

    # Shuffle and split
    indices = list(range(len(smiles_list)))
    rng.shuffle(indices)

    n_train = int(len(indices) * train_fraction)
    train_idx = indices[:n_train]
    valid_idx = indices[n_train:]

    train_smiles = [smiles_list[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    valid_smiles = [smiles_list[i] for i in valid_idx]
    valid_labels = np.array([labels[i] for i in valid_idx])

    # Extract rules from training set
    sar_extractor = SARExtractor(min_support=5, min_effect_size=0.3, max_p_value=0.05)
    rules = sar_extractor.extract_rules(train_smiles, train_labels)

    if not rules:
        return {
            'r2': 0.0,
            'mae': np.nan,
            'n_rules': 0,
            'error': 'No rules discovered'
        }

    # Create semantic memory with rules
    memory = SemanticMemory()
    memory.update_global_stats(train_labels)
    memory.ingest_rules(rules, episode_id='train')

    # Make symbolic predictions on validation set
    symbolic_preds = []
    valid_labels_used = []

    for i, smiles in enumerate(valid_smiles):
        pred, _ = memory.predict_from_rules(smiles)
        if not np.isnan(pred):
            symbolic_preds.append(pred)
            valid_labels_used.append(valid_labels[i])

    if len(symbolic_preds) < 10:
        return {
            'r2': 0.0,
            'mae': np.nan,
            'n_rules': len(rules),
            'n_predictions': len(symbolic_preds),
            'error': 'Insufficient valid predictions'
        }

    symbolic_preds = np.array(symbolic_preds)
    valid_labels_used = np.array(valid_labels_used)

    # Compute R² and MAE
    ss_res = np.sum((valid_labels_used - symbolic_preds) ** 2)
    ss_tot = np.sum((valid_labels_used - np.mean(valid_labels_used)) ** 2)

    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    mae = np.mean(np.abs(valid_labels_used - symbolic_preds))

    return {
        'r2': float(r2),
        'mae': float(mae),
        'n_rules': len(rules),
        'n_predictions': len(symbolic_preds),
        'mean_actual': float(np.mean(valid_labels_used)),
        'mean_predicted': float(np.mean(symbolic_preds)),
        'std_actual': float(np.std(valid_labels_used)),
        'std_predicted': float(np.std(symbolic_preds))
    }


# =============================================================================
# METRIC 3: RULE STABILITY ACROSS SEEDS
# =============================================================================

def compute_rule_stability(
    smiles_list: List[str],
    labels: List[float],
    seeds: List[int] = None,
    show_progress: bool = True
) -> Dict:
    """
    Test if rules are consistent across random seeds.

    For each seed:
    1. Shuffle data differently
    2. Extract rules
    3. Compare rule sets across seeds

    Measures:
    - Jaccard similarity of rule feature sets
    - Correlation of effect sizes for common rules
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1011]

    rules_by_seed = {}

    iterator = tqdm(seeds, desc="Computing rule stability") if show_progress else seeds

    for seed in iterator:
        rng = np.random.RandomState(seed)

        # Shuffle data
        indices = list(range(len(smiles_list)))
        rng.shuffle(indices)
        shuffled_smiles = [smiles_list[i] for i in indices]
        shuffled_labels = [labels[i] for i in indices]

        # Extract rules
        sar_extractor = SARExtractor(min_support=5, min_effect_size=0.3, max_p_value=0.05)
        rules = sar_extractor.extract_rules(shuffled_smiles, shuffled_labels)

        rules_by_seed[seed] = {
            'features': set(r['feature'] for r in rules),
            'effects': {r['feature']: r['effect_size'] for r in rules},
            'n_rules': len(rules)
        }

    # Compute pairwise Jaccard similarity
    jaccard_scores = []
    effect_correlations = []

    seed_pairs = [(seeds[i], seeds[j]) for i in range(len(seeds))
                  for j in range(i + 1, len(seeds))]

    for seed_a, seed_b in seed_pairs:
        features_a = rules_by_seed[seed_a]['features']
        features_b = rules_by_seed[seed_b]['features']

        # Jaccard
        if features_a or features_b:
            jaccard = len(features_a & features_b) / len(features_a | features_b)
        else:
            jaccard = 0.0
        jaccard_scores.append(jaccard)

        # Effect correlation for common rules
        common = features_a & features_b
        if len(common) >= 2:
            effects_a = [rules_by_seed[seed_a]['effects'][f] for f in common]
            effects_b = [rules_by_seed[seed_b]['effects'][f] for f in common]
            corr, _ = stats.pearsonr(effects_a, effects_b)
            if not np.isnan(corr):
                effect_correlations.append(corr)

    return {
        'mean_jaccard': float(np.mean(jaccard_scores)) if jaccard_scores else 0.0,
        'std_jaccard': float(np.std(jaccard_scores)) if jaccard_scores else 0.0,
        'mean_effect_correlation': float(np.mean(effect_correlations)) if effect_correlations else 0.0,
        'std_effect_correlation': float(np.std(effect_correlations)) if effect_correlations else 0.0,
        'n_seeds': len(seeds),
        'n_pairs': len(seed_pairs),
        'rules_per_seed': {seed: rules_by_seed[seed]['n_rules'] for seed in seeds},
        'jaccard_scores': [float(j) for j in jaccard_scores],
        'effect_correlations': [float(c) for c in effect_correlations]
    }


# =============================================================================
# METRIC 4: INTERVENTION TESTING
# =============================================================================

def compute_intervention_test(
    smiles_list: List[str],
    labels: List[float],
    seed: int = 42
) -> Dict:
    """
    Create synthetic test cases that should trigger specific rules.

    1. Extract rules from full data
    2. Find molecules that SHOULD trigger strongest rules
    3. Check if symbolic predictions match neural predictions

    This is a "sanity check" that rules make sensible predictions.
    """
    # Train world model
    model = MolecularWorldModel(n_estimators=50, random_state=seed)
    model.fit(smiles_list, labels)

    # Extract rules
    sar_extractor = SARExtractor(min_support=5, min_effect_size=0.3, max_p_value=0.05)
    rules = sar_extractor.extract_rules(smiles_list, labels)

    if not rules:
        return {
            'error': 'No rules discovered',
            'n_rules': 0
        }

    # Create semantic memory
    memory = SemanticMemory()
    memory.update_global_stats(labels)
    memory.ingest_rules(rules, episode_id='full')

    # Get top rules
    top_rules = sorted(rules, key=lambda r: abs(r['effect_size']), reverse=True)[:5]

    # For each top rule, find molecules that trigger it
    intervention_results = []

    for rule in top_rules:
        feature = rule['feature']
        rule_effect = rule['effect_size']

        # Find molecules with this feature
        test_molecules = []
        for i, smiles in enumerate(smiles_list):
            mol_features = compute_molecular_features(smiles)
            if mol_features.get(feature, False):
                test_molecules.append({
                    'smiles': smiles,
                    'actual': labels[i]
                })

        if len(test_molecules) < 3:
            continue

        # Get predictions
        test_smiles = [m['smiles'] for m in test_molecules[:20]]
        actuals = [m['actual'] for m in test_molecules[:20]]

        # Neural predictions
        neural_preds, _ = model.predict(test_smiles, return_uncertainty=True)

        # Symbolic predictions
        symbolic_preds = []
        for smiles in test_smiles:
            pred, _ = memory.predict_from_rules(smiles)
            symbolic_preds.append(pred)

        symbolic_preds = np.array(symbolic_preds)
        valid = ~np.isnan(neural_preds) & ~np.isnan(symbolic_preds)

        if np.sum(valid) < 3:
            continue

        neural_valid = neural_preds[valid]
        symbolic_valid = symbolic_preds[valid]
        actual_valid = np.array(actuals)[valid]

        # Compute agreement
        neural_mae = np.mean(np.abs(neural_valid - actual_valid))
        symbolic_mae = np.mean(np.abs(symbolic_valid - actual_valid))
        neural_symbolic_mae = np.mean(np.abs(neural_valid - symbolic_valid))

        intervention_results.append({
            'feature': feature,
            'rule_effect': float(rule_effect),
            'n_molecules': int(np.sum(valid)),
            'neural_mae': float(neural_mae),
            'symbolic_mae': float(symbolic_mae),
            'neural_symbolic_mae': float(neural_symbolic_mae),
            'agreement_good': neural_symbolic_mae < max(neural_mae, symbolic_mae)
        })

    # Aggregate
    if not intervention_results:
        return {
            'error': 'No valid interventions',
            'n_rules': len(rules)
        }

    return {
        'n_interventions': len(intervention_results),
        'n_agreements': sum(r['agreement_good'] for r in intervention_results),
        'agreement_rate': sum(r['agreement_good'] for r in intervention_results) / len(intervention_results),
        'mean_neural_mae': float(np.mean([r['neural_mae'] for r in intervention_results])),
        'mean_symbolic_mae': float(np.mean([r['symbolic_mae'] for r in intervention_results])),
        'mean_neural_symbolic_gap': float(np.mean([r['neural_symbolic_mae'] for r in intervention_results])),
        'per_rule': intervention_results
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_consistency_plots(
    results: Dict,
    save_dir: Optional[str] = None
) -> List[plt.Figure]:
    """Create visualizations for consistency metrics."""
    figures = []
    save_path = Path(save_dir) if save_dir else None

    # 1. Effect correlation scatter plot
    if 'effect_correlation' in results and results['effect_correlation']['n_rules_compared'] > 0:
        fig, ax = plt.subplots(figsize=(8, 6))

        neural = results['effect_correlation']['neural_effects']
        symbolic = results['effect_correlation']['symbolic_effects']
        features = results['effect_correlation']['features']

        ax.scatter(symbolic, neural, alpha=0.7, s=100)

        # Add diagonal
        max_val = max(max(abs(x) for x in neural), max(abs(x) for x in symbolic))
        ax.plot([-max_val, max_val], [-max_val, max_val], 'k--', alpha=0.5)

        # Add labels for top points
        for i, feat in enumerate(features):
            ax.annotate(feat.replace('_', ' '), (symbolic[i], neural[i]),
                       fontsize=8, alpha=0.7)

        ax.set_xlabel('Symbolic Rule Effect')
        ax.set_ylabel('Neural Model Effect')
        ax.set_title(f"Neural-Symbolic Effect Correlation\nr={results['effect_correlation']['correlation']:.3f}")
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path / 'effect_correlation.png', dpi=150, bbox_inches='tight')

        figures.append(fig)

    # 2. Rule stability box plot
    if 'rule_stability' in results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        jaccard = results['rule_stability']['jaccard_scores']
        effect_corr = results['rule_stability']['effect_correlations']

        axes[0].boxplot([jaccard], labels=['Jaccard'])
        axes[0].axhline(0.6, color='green', linestyle='--', alpha=0.5, label='Good threshold')
        axes[0].set_ylabel('Similarity Score')
        axes[0].set_title(f"Rule Set Overlap Across Seeds\n(mean={results['rule_stability']['mean_jaccard']:.2f})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        if effect_corr:
            axes[1].boxplot([effect_corr], labels=['Effect Correlation'])
            axes[1].axhline(0.7, color='green', linestyle='--', alpha=0.5, label='Good threshold')
            axes[1].set_ylabel('Correlation')
            axes[1].set_title(f"Effect Size Consistency\n(mean={results['rule_stability']['mean_effect_correlation']:.2f})")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')
        else:
            axes[1].text(0.5, 0.5, 'Insufficient common rules',
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Effect Size Consistency')

        if save_path:
            fig.savefig(save_path / 'rule_stability.png', dpi=150, bbox_inches='tight')

        figures.append(fig)

    # 3. Intervention test results
    if 'intervention_test' in results and 'per_rule' in results['intervention_test']:
        fig, ax = plt.subplots(figsize=(10, 6))

        per_rule = results['intervention_test']['per_rule']
        features = [r['feature'].replace('_', '\n') for r in per_rule]
        neural_mae = [r['neural_mae'] for r in per_rule]
        symbolic_mae = [r['symbolic_mae'] for r in per_rule]

        x = np.arange(len(features))
        width = 0.35

        ax.bar(x - width/2, neural_mae, width, label='Neural MAE', alpha=0.8)
        ax.bar(x + width/2, symbolic_mae, width, label='Symbolic MAE', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Intervention Test: Neural vs Symbolic Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path / 'intervention_test.png', dpi=150, bbox_inches='tight')

        figures.append(fig)

    return figures


# =============================================================================
# MAIN
# =============================================================================

def run_enhanced_consistency_validation(
    data_path: str = 'data/esol_processed.pkl',
    output_dir: str = 'results/enhanced_metrics',
    seeds: List[int] = None
) -> Dict:
    """Run all enhanced consistency metrics."""
    if seeds is None:
        seeds = [42, 123, 456, 789, 1011]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'plots').mkdir(exist_ok=True)

    print("=" * 70)
    print("ENHANCED CONSISTENCY VALIDATION")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    data = load_esol_data(data_path)
    candidate_df = data['candidate_pool']
    test_df = data['test_set']

    all_smiles = candidate_df['smiles'].tolist() + test_df['smiles'].tolist()
    all_labels = candidate_df['logS'].tolist() + test_df['logS'].tolist()

    print(f"Total molecules: {len(all_smiles)}")

    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'data_path': data_path,
            'n_molecules': len(all_smiles),
            'seeds': seeds
        }
    }

    # Train model and extract rules for effect correlation
    print("\n1. Computing effect magnitude correlation...")
    model = MolecularWorldModel(n_estimators=50, random_state=42)
    model.fit(all_smiles, all_labels)

    sar_extractor = SARExtractor(min_support=5, min_effect_size=0.3, max_p_value=0.05)
    rules = sar_extractor.extract_rules(all_smiles, all_labels)

    effect_result = compute_effect_magnitude_correlation(
        model, rules, all_smiles, all_labels
    )
    results['effect_correlation'] = effect_result
    print(f"   Correlation: {effect_result['correlation']:.3f} (n={effect_result['n_rules_compared']} rules)")

    # Out-of-sample validation
    print("\n2. Computing out-of-sample rule validation...")
    oos_result = compute_out_of_sample_validation(all_smiles, all_labels)
    results['out_of_sample'] = oos_result
    print(f"   R²: {oos_result['r2']:.3f}, MAE: {oos_result['mae']:.3f}")

    # Rule stability
    print("\n3. Computing rule stability across seeds...")
    stability_result = compute_rule_stability(all_smiles, all_labels, seeds=seeds)
    results['rule_stability'] = stability_result
    print(f"   Mean Jaccard: {stability_result['mean_jaccard']:.3f}")
    print(f"   Mean effect correlation: {stability_result['mean_effect_correlation']:.3f}")

    # Intervention testing
    print("\n4. Running intervention tests...")
    intervention_result = compute_intervention_test(all_smiles, all_labels)
    results['intervention_test'] = intervention_result
    if 'agreement_rate' in intervention_result:
        print(f"   Agreement rate: {intervention_result['agreement_rate']*100:.1f}%")
        print(f"   Neural-symbolic gap: {intervention_result['mean_neural_symbolic_gap']:.3f}")

    # Create visualizations
    print("\n5. Creating visualizations...")
    figures = create_consistency_plots(results, save_dir=str(output_path / 'plots'))
    for fig in figures:
        plt.close(fig)

    # Save results
    with open(output_path / 'consistency_metrics.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    # Print summary
    print("\n" + "=" * 70)
    print("CONSISTENCY VALIDATION SUMMARY")
    print("=" * 70)

    print("\nMetric 1: Effect Magnitude Correlation")
    print(f"  Correlation: {effect_result['correlation']:.3f}")
    print(f"  P-value: {effect_result['p_value']:.4f}")
    if effect_result['correlation'] > 0.7:
        print("  [PASS] Strong correlation - rules capture neural behavior")
    elif effect_result['correlation'] > 0.4:
        print("  [PARTIAL] Moderate correlation - rules partially capture behavior")
    else:
        print("  [NEEDS ATTENTION] Weak correlation - rules may not match neural model")

    print("\nMetric 2: Out-of-Sample Validation")
    print(f"  R²: {oos_result['r2']:.3f}")
    print(f"  MAE: {oos_result['mae']:.3f}")
    if oos_result['r2'] > 0.3:
        print("  [PASS] Rules generalize to held-out data")
    elif oos_result['r2'] > 0:
        print("  [PARTIAL] Some generalization")
    else:
        print("  [NEEDS ATTENTION] Poor generalization")

    print("\nMetric 3: Rule Stability")
    print(f"  Jaccard similarity: {stability_result['mean_jaccard']:.3f}")
    print(f"  Effect correlation: {stability_result['mean_effect_correlation']:.3f}")
    if stability_result['mean_jaccard'] > 0.6:
        print("  [PASS] Rules are stable across seeds")
    elif stability_result['mean_jaccard'] > 0.4:
        print("  [PARTIAL] Moderate stability")
    else:
        print("  [NEEDS ATTENTION] Rules vary significantly across seeds")

    if 'agreement_rate' in intervention_result:
        print("\nMetric 4: Intervention Testing")
        print(f"  Agreement rate: {intervention_result['agreement_rate']*100:.1f}%")
        print(f"  Neural-symbolic gap: {intervention_result['mean_neural_symbolic_gap']:.3f}")
        if intervention_result['agreement_rate'] > 0.6:
            print("  [PASS] Neural and symbolic predictions agree")
        else:
            print("  [NEEDS ATTENTION] Predictions diverge")

    print(f"\n\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run enhanced consistency validation")
    parser.add_argument('--data-path', default='data/esol_processed.pkl',
                       help='Path to ESOL data')
    parser.add_argument('--output-dir', default='results/enhanced_metrics',
                       help='Output directory')
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[42, 123, 456, 789, 1011],
                       help='Random seeds for stability testing')

    args = parser.parse_args()

    results = run_enhanced_consistency_validation(
        data_path=args.data_path,
        output_dir=args.output_dir,
        seeds=args.seeds
    )
