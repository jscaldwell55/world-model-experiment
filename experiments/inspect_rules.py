"""
Diagnostic: Inspect Phase 3 Semantic Rules.

This script runs a training loop under distribution_shift conditions
and inspects the quality and coverage of rules stored in SemanticMemory.

Goal: Answer key questions:
1. Are the rules chemically valid (e.g., "aromatic ring increases solubility") or noise?
2. Do the rules apply to enough data (Coverage > 10%)?
3. Is the Symbolic prediction wildly off-scale compared to the Neural prediction?
"""

import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from molecular_world_model import MolecularWorldModel
from molecular_consolidation_pipeline import SimplifiedFTB
from dream_state import AnalogGenerator, SARExtractor, DreamPipeline
from nesy_bridge import SemanticMemory, ConsistencyChecker, HybridPredictor

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

def compute_molecular_weight(smiles: str) -> float:
    """Compute molecular weight from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol) if mol else 0.0


def load_distribution_shift_data(
    data_path: str = 'data/esol_processed.pkl',
    seed: int = 42
) -> Dict:
    """
    Load ESOL data sorted by molecular weight (distribution_shift condition).

    This forces the model to learn rules to extrapolate from light to heavy molecules.
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    candidate_df = data['candidate_pool'].copy()
    test_df = data['test_set'].copy()

    all_smiles = candidate_df['smiles'].tolist() + test_df['smiles'].tolist()
    all_labels = candidate_df['logS'].tolist() + test_df['logS'].tolist()

    # Sort by molecular weight (ascending)
    mol_weights = [compute_molecular_weight(s) for s in all_smiles]
    sorted_indices = np.argsort(mol_weights)
    all_smiles = [all_smiles[i] for i in sorted_indices]
    all_labels = [all_labels[i] for i in sorted_indices]
    mol_weights = [mol_weights[i] for i in sorted_indices]

    # Split: 70% candidates (light molecules), 30% test (heavy molecules)
    n_candidates = int(0.7 * len(all_smiles))

    return {
        'candidate_smiles': all_smiles[:n_candidates],
        'candidate_labels': all_labels[:n_candidates],
        'test_smiles': all_smiles[n_candidates:],
        'test_labels': all_labels[n_candidates:],
        'mol_weights': mol_weights,
        'n_candidates': n_candidates
    }


# =============================================================================
# NESYFTB INITIALIZATION
# =============================================================================

def initialize_nesy_system(
    seed_smiles: List[str],
    seed_labels: List[float],
    probe_smiles: List[str],
    probe_labels: List[float],
    random_state: int = 42
) -> Dict:
    """
    Initialize the NeSyFTB system: World Model + Dream Pipeline + Semantic Memory.
    """
    # World Model
    world_model = MolecularWorldModel(n_estimators=50, random_state=random_state)
    world_model.fit(seed_smiles, seed_labels)

    # FTB for stability-aware updates
    ftb = SimplifiedFTB(
        world_model=world_model,
        probe_smiles=probe_smiles,
        probe_labels=probe_labels,
        retention_threshold=0.25,
        repair_attempts=3,
        replay_ratio=0.5,
        random_state=random_state
    )

    # Dream Pipeline with SAR extraction
    # Lower thresholds for diagnostic mode to see what rules emerge
    analog_generator = AnalogGenerator(random_state=random_state)
    sar_extractor = SARExtractor(min_support=3, min_effect_size=0.2, max_p_value=0.1)

    # Lower confidence threshold (0.5) to allow more synthetics for rule discovery
    # This is diagnostic mode - we want to see what rules CAN be learned
    dream_pipeline = DreamPipeline(
        world_model=world_model,
        analog_generator=analog_generator,
        sar_extractor=sar_extractor,
        confidence_threshold=0.5,  # Lowered from 0.85 for diagnostic
        max_synthetics_ratio=0.5,  # Increased from 0.3
        synthetic_weight=0.6,
        random_state=random_state
    )

    # NeSy components - lower thresholds for diagnostic
    semantic_memory = SemanticMemory(
        min_p_value=0.1,       # Allow p<0.1 rules
        min_confidence=0.5,    # Lower confidence threshold
        min_effect_size=0.15   # Smaller effects allowed
    )

    consistency_checker = ConsistencyChecker(
        world_model=world_model,
        semantic_memory=semantic_memory
    )

    hybrid_predictor = HybridPredictor(
        world_model=world_model,
        semantic_memory=semantic_memory,
        neural_weight=0.7
    )

    return {
        'world_model': world_model,
        'ftb': ftb,
        'dream_pipeline': dream_pipeline,
        'semantic_memory': semantic_memory,
        'consistency_checker': consistency_checker,
        'hybrid_predictor': hybrid_predictor
    }


# =============================================================================
# TRAINING LOOP
# =============================================================================

def run_training_loop(
    system: Dict,
    candidate_smiles: List[str],
    candidate_labels: List[float],
    n_steps: int = 20,
    update_interval: int = 10,
    seed_size: int = 20
) -> Dict:
    """
    Run a training loop for n_steps under distribution_shift conditions.

    Returns training history including rules discovered at each step.
    """
    world_model = system['world_model']
    ftb = system['ftb']
    dream_pipeline = system['dream_pipeline']
    semantic_memory = system['semantic_memory']

    # Track accumulated data
    accumulated_smiles = candidate_smiles[:seed_size].copy()
    accumulated_labels = candidate_labels[:seed_size].copy()

    # Query pool (after seed)
    query_smiles = candidate_smiles[seed_size:]
    query_labels = candidate_labels[seed_size:]

    batch_smiles = []
    batch_labels = []
    history = []

    logger.info(f"Starting training loop: {n_steps} steps, update every {update_interval}")

    for step in range(min(n_steps, len(query_smiles))):
        # Simulate acquisition
        smiles = query_smiles[step]
        label = query_labels[step]
        batch_smiles.append(smiles)
        batch_labels.append(label)

        # Perform update at intervals
        if len(batch_smiles) >= update_interval:
            accumulated_smiles.extend(batch_smiles)
            accumulated_labels.extend(batch_labels)

            # Dream: Generate synthetics and extract SAR rules
            dream_result = dream_pipeline.dream(
                real_smiles=accumulated_smiles,
                real_labels=accumulated_labels,
                condition='distribution_shift',
                n_variants_per_molecule=5
            )

            # Ingest SAR rules into semantic memory
            episode_id = f"episode_{(step + 1) // update_interval}"
            if dream_result['sar_rules']:
                ingest_result = semantic_memory.ingest_rules(
                    dream_result['sar_rules'],
                    episode_id=episode_id
                )
                logger.info(f"Step {step+1}: Ingested rules - {ingest_result}")

            # FTB update with combined data
            combined_smiles = accumulated_smiles + dream_result['synthetic_smiles']
            combined_labels = accumulated_labels + dream_result['synthetic_labels']
            combined_weights = [1.0] * len(accumulated_smiles) + dream_result['synthetic_weights']

            ftb.update(
                smiles=combined_smiles,
                labels=combined_labels,
                weights=combined_weights
            )

            # Record state
            history.append({
                'step': step + 1,
                'n_rules': len(semantic_memory),
                'n_accumulated': len(accumulated_smiles),
                'n_synthetics': len(dream_result['synthetic_smiles'])
            })

            batch_smiles = []
            batch_labels = []

    logger.info(f"Training complete. Final rules in memory: {len(semantic_memory)}")

    return {
        'history': history,
        'final_n_rules': len(semantic_memory),
        'accumulated_smiles': accumulated_smiles,
        'accumulated_labels': accumulated_labels
    }


# =============================================================================
# RULE INSPECTION REPORT
# =============================================================================

def print_top_rules(semantic_memory: SemanticMemory, top_n: int = 10) -> None:
    """
    Print the top N rules sorted by confidence.

    Format: [Rule ID] Feature: {name} | Effect: {value:.3f} | Conf: {conf:.2f} | p-val: {p:.4f} | Support: {n}
    """
    rules = semantic_memory.get_all_rules()

    if not rules:
        print("\n" + "=" * 80)
        print("TOP 10 RULES")
        print("=" * 80)
        print("No rules in semantic memory.")
        return

    # Sort by confidence (descending)
    sorted_rules = sorted(rules, key=lambda r: r['confidence'], reverse=True)[:top_n]

    print("\n" + "=" * 80)
    print(f"TOP {min(top_n, len(sorted_rules))} RULES (sorted by confidence)")
    print("=" * 80)

    for i, rule in enumerate(sorted_rules, 1):
        feature = rule['feature']
        effect = rule['effect_size']
        conf = rule['confidence']
        p_val = rule['p_value']
        support = rule['n_observations']

        print(f"[Rule {i:2d}] Feature: {feature:<25} | Effect: {effect:+.3f} | "
              f"Conf: {conf:.2f} | p-val: {p_val:.4f} | Support: {support}")

    print("=" * 80)


def compute_test_set_coverage(
    semantic_memory: SemanticMemory,
    test_smiles: List[str]
) -> Dict:
    """
    Compute what percentage of test molecules trigger at least one rule.

    Returns coverage percentage and breakdown.
    """
    if len(semantic_memory) == 0:
        return {
            'coverage_pct': 0.0,
            'n_covered': 0,
            'n_total': len(test_smiles),
            'n_rules_per_molecule': [],
            'avg_rules_per_molecule': 0.0,
            'max_rules_per_molecule': 0
        }

    n_covered = 0
    n_rules_per_mol = []

    for smiles in test_smiles:
        applicable_rules = semantic_memory.get_applicable_rules(smiles)
        n_rules = len(applicable_rules)
        n_rules_per_mol.append(n_rules)
        if n_rules > 0:
            n_covered += 1

    coverage_pct = (n_covered / len(test_smiles)) * 100 if test_smiles else 0.0

    return {
        'coverage_pct': coverage_pct,
        'n_covered': n_covered,
        'n_total': len(test_smiles),
        'n_rules_per_molecule': n_rules_per_mol,
        'avg_rules_per_molecule': np.mean(n_rules_per_mol) if n_rules_per_mol else 0.0,
        'max_rules_per_molecule': max(n_rules_per_mol) if n_rules_per_mol else 0
    }


def print_coverage_report(coverage: Dict) -> None:
    """Print the test set coverage report."""
    print("\n" + "=" * 80)
    print("TEST SET COVERAGE")
    print("=" * 80)
    print(f"Global Rule Coverage: {coverage['coverage_pct']:.1f}%")
    print(f"  Molecules with rules: {coverage['n_covered']} / {coverage['n_total']}")
    print(f"  Avg rules per molecule: {coverage['avg_rules_per_molecule']:.2f}")
    print(f"  Max rules per molecule: {coverage['max_rules_per_molecule']}")

    # Check against 10% threshold
    if coverage['coverage_pct'] >= 10:
        print(f"  [PASS] Coverage >= 10%")
    else:
        print(f"  [WARN] Coverage < 10% - rules may not generalize well")
    print("=" * 80)


# =============================================================================
# CONFLICT ANALYSIS
# =============================================================================

def run_conflict_analysis(
    semantic_memory: SemanticMemory,
    hybrid_predictor: HybridPredictor,
    test_smiles: List[str],
    test_labels: List[float],
    n_samples: int = 5,
    seed: int = 42
) -> List[Dict]:
    """
    Pick random test molecules where at least one rule triggers and analyze
    the difference between Neural and Symbolic predictions.
    """
    random.seed(seed)

    # Find molecules where rules apply
    molecules_with_rules = []
    for i, smiles in enumerate(test_smiles):
        applicable = semantic_memory.get_applicable_rules(smiles)
        if applicable:
            molecules_with_rules.append((i, smiles, test_labels[i], applicable))

    if not molecules_with_rules:
        return []

    # Random sample
    n_samples = min(n_samples, len(molecules_with_rules))
    sampled = random.sample(molecules_with_rules, n_samples)

    analysis = []
    for idx, smiles, true_label, rules in sampled:
        # Get hybrid prediction (contains neural, symbolic, hybrid)
        pred = hybrid_predictor.predict(smiles)

        neural_pred = pred['neural_prediction']
        symbolic_pred = pred['symbolic_prediction']
        delta = symbolic_pred - neural_pred

        analysis.append({
            'smiles': smiles,
            'true_label': true_label,
            'neural_prediction': neural_pred,
            'symbolic_prediction': symbolic_pred,
            'delta_symbolic_minus_neural': delta,
            'rules_applied': [r['feature'] for r in rules],
            'rule_effects': [(r['feature'], r['effect_size']) for r in rules]
        })

    return analysis


def print_conflict_analysis(analysis: List[Dict], test_labels: List[float] = None) -> None:
    """Print the conflict analysis report."""
    print("\n" + "=" * 80)
    print("CONFLICT ANALYSIS (5 Random Test Molecules with Rules)")
    print("=" * 80)

    if not analysis:
        print("No molecules with applicable rules found.")
        return

    # Compute error metrics
    neural_errors = []
    symbolic_errors = []

    for i, item in enumerate(analysis, 1):
        print(f"\n--- Molecule {i} ---")
        print(f"SMILES: {item['smiles']}")
        print(f"True Label (logS): {item['true_label']:.3f}")
        print(f"Neural Prediction:   {item['neural_prediction']:.3f}")
        print(f"Symbolic Prediction: {item['symbolic_prediction']:.3f}")
        print(f"Delta (Symbolic - Neural): {item['delta_symbolic_minus_neural']:+.3f}")
        print(f"Rules Applied: {', '.join(item['rules_applied'])}")
        print(f"Rule Effects:")
        for feature, effect in item['rule_effects']:
            print(f"    {feature}: {effect:+.3f}")

        # Compute errors vs true label
        neural_err = abs(item['neural_prediction'] - item['true_label'])
        symbolic_err = abs(item['symbolic_prediction'] - item['true_label'])
        neural_errors.append(neural_err)
        symbolic_errors.append(symbolic_err)

        print(f"  Neural Error:   {neural_err:.3f}")
        print(f"  Symbolic Error: {symbolic_err:.3f}")

        # Check if delta is "wildly off-scale" (> 2.0)
        if abs(item['delta_symbolic_minus_neural']) > 2.0:
            print(f"  [WARN] Large discrepancy between Neural and Symbolic!")
        elif abs(item['delta_symbolic_minus_neural']) > 1.0:
            print(f"  [INFO] Moderate discrepancy")
        else:
            print(f"  [OK] Predictions reasonably aligned")

    # Summary statistics
    if neural_errors:
        print("\n--- Error Summary ---")
        print(f"Mean Neural Error:   {np.mean(neural_errors):.3f}")
        print(f"Mean Symbolic Error: {np.mean(symbolic_errors):.3f}")
        if np.mean(symbolic_errors) > np.mean(neural_errors) * 1.5:
            print("[ISSUE] Symbolic predictions significantly worse than Neural!")
            print("        Consider: Symbolic uses baseline=0.0, but test labels may be offset.")

    print("\n" + "=" * 80)


# =============================================================================
# CHEMICAL VALIDITY ANALYSIS
# =============================================================================

def analyze_rule_validity(semantic_memory: SemanticMemory) -> Dict:
    """
    Analyze whether the discovered rules are chemically plausible.

    Known chemistry facts (for solubility/logS):
    - Increased logP (lipophilicity) -> DECREASES solubility
    - Polar groups (O, N, S) -> INCREASE solubility
    - Aromatic rings -> DECREASE solubility (generally)
    - Larger MW -> DECREASES solubility
    - TPSA (polar surface area) -> INCREASES solubility
    - H-bond donors/acceptors -> INCREASE solubility
    """
    rules = semantic_memory.get_all_rules()

    if not rules:
        return {'valid': [], 'questionable': [], 'summary': 'No rules to analyze'}

    # Expected directions based on chemistry
    expected_directions = {
        'logp_over_2': 'decreases',
        'logp_over_3': 'decreases',
        'logp_negative': 'increases',
        'has_oxygen': 'increases',
        'has_nitrogen': 'increases',
        'has_sulfur': 'uncertain',  # Context dependent
        'has_aromatic_ring': 'decreases',
        'has_multiple_rings': 'decreases',
        'mw_over_300': 'decreases',
        'mw_over_400': 'decreases',
        'tpsa_over_50': 'increases',
        'tpsa_over_100': 'increases',
        'n_hbd_over_2': 'increases',
        'n_hba_over_5': 'increases',
        'has_fluorine': 'uncertain',  # Context dependent
        'has_chlorine': 'decreases',
        'has_bromine': 'decreases',
        'n_rotatable_over_5': 'uncertain',
        'has_aliphatic_ring': 'uncertain'
    }

    valid = []
    questionable = []

    for rule in rules:
        feature = rule['feature']
        learned_direction = rule['direction']
        expected = expected_directions.get(feature, 'uncertain')

        if expected == 'uncertain':
            valid.append({
                'feature': feature,
                'learned': learned_direction,
                'expected': 'uncertain',
                'status': 'neutral'
            })
        elif learned_direction == expected:
            valid.append({
                'feature': feature,
                'learned': learned_direction,
                'expected': expected,
                'status': 'valid'
            })
        else:
            questionable.append({
                'feature': feature,
                'learned': learned_direction,
                'expected': expected,
                'status': 'questionable'
            })

    return {
        'valid': valid,
        'questionable': questionable,
        'n_valid': len([r for r in valid if r['status'] == 'valid']),
        'n_neutral': len([r for r in valid if r['status'] == 'neutral']),
        'n_questionable': len(questionable),
        'total': len(rules)
    }


def print_validity_analysis(validity: Dict) -> None:
    """Print the chemical validity analysis."""
    print("\n" + "=" * 80)
    print("CHEMICAL VALIDITY ANALYSIS")
    print("=" * 80)

    if validity.get('total', 0) == 0:
        print("No rules to analyze.")
        return

    print(f"Total Rules: {validity['total']}")
    print(f"  Valid (matches chemistry): {validity['n_valid']}")
    print(f"  Neutral (uncertain expected): {validity['n_neutral']}")
    print(f"  Questionable (contradicts chemistry): {validity['n_questionable']}")

    if validity['questionable']:
        print("\nQuestionable Rules (may be noise):")
        for r in validity['questionable']:
            print(f"  - {r['feature']}: learned '{r['learned']}', expected '{r['expected']}'")

    if validity['valid']:
        print("\nValid Rules (chemically plausible):")
        for r in validity['valid']:
            if r['status'] == 'valid':
                print(f"  + {r['feature']}: {r['learned']} (correct)")

    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the full diagnostic inspection."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC: INSPECT PHASE 3 SEMANTIC RULES")
    print("=" * 80)
    print("Purpose: Analyze rule quality and coverage under distribution_shift")
    print("=" * 80)

    # 1. Load data
    logger.info("Loading distribution_shift data (sorted by molecular weight)...")
    data = load_distribution_shift_data()

    print(f"\nData loaded:")
    print(f"  Candidates (light molecules): {len(data['candidate_smiles'])}")
    print(f"  Test set (heavy molecules): {len(data['test_smiles'])}")
    print(f"  Weight range: {min(data['mol_weights']):.1f} - {max(data['mol_weights']):.1f}")

    # 2. Initialize NeSy system
    logger.info("Initializing NeSyFTB system...")
    seed_size = 20
    seed_smiles = data['candidate_smiles'][:seed_size]
    seed_labels = data['candidate_labels'][:seed_size]
    probe_smiles = data['test_smiles'][:30]
    probe_labels = data['test_labels'][:30]

    system = initialize_nesy_system(
        seed_smiles=seed_smiles,
        seed_labels=seed_labels,
        probe_smiles=probe_smiles,
        probe_labels=probe_labels,
        random_state=42
    )

    print(f"\nSystem initialized with {seed_size} seed molecules.")

    # 3. Run training loop (50 steps to allow more rule discovery)
    # 20 steps was in the original task but we increase for better diagnostics
    n_training_steps = 50
    logger.info(f"Running {n_training_steps}-step training loop...")
    training_result = run_training_loop(
        system=system,
        candidate_smiles=data['candidate_smiles'],
        candidate_labels=data['candidate_labels'],
        n_steps=n_training_steps,
        update_interval=10,
        seed_size=seed_size
    )

    print(f"\nTraining complete:")
    print(f"  Final rules in memory: {training_result['final_n_rules']}")
    print(f"  Accumulated training data: {len(training_result['accumulated_smiles'])}")

    # 4. Print Top 10 Rules
    print_top_rules(system['semantic_memory'], top_n=10)

    # 5. Compute and print test set coverage
    coverage = compute_test_set_coverage(
        semantic_memory=system['semantic_memory'],
        test_smiles=data['test_smiles']
    )
    print_coverage_report(coverage)

    # 6. Run and print conflict analysis
    conflict_analysis = run_conflict_analysis(
        semantic_memory=system['semantic_memory'],
        hybrid_predictor=system['hybrid_predictor'],
        test_smiles=data['test_smiles'],
        test_labels=data['test_labels'],
        n_samples=5,
        seed=42
    )
    print_conflict_analysis(conflict_analysis)

    # 7. Analyze chemical validity
    validity = analyze_rule_validity(system['semantic_memory'])
    print_validity_analysis(validity)

    # 8. Baseline Analysis
    print("\n" + "=" * 80)
    print("BASELINE ANALYSIS")
    print("=" * 80)
    mean_test_label = np.mean(data['test_labels'])
    mean_train_label = np.mean(training_result['accumulated_labels'])
    print(f"Mean training label: {mean_train_label:.3f}")
    print(f"Mean test label:     {mean_test_label:.3f}")
    print(f"Symbolic baseline:   0.000 (hardcoded)")
    print(f"\nNote: Symbolic predictions use baseline=0.0 but true labels are offset.")
    print(f"      This explains why Symbolic MAE may be higher than Neural MAE.")
    print("=" * 80)

    # 9. Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"1. Rules discovered: {training_result['final_n_rules']}")
    print(f"2. Test coverage: {coverage['coverage_pct']:.1f}%")
    print(f"3. Valid rules: {validity.get('n_valid', 0)} / {validity.get('total', 0)}")
    print(f"4. Questionable rules: {validity.get('n_questionable', 0)}")

    # Verdict
    print("\n" + "-" * 40)
    issues = []
    if training_result['final_n_rules'] < 3:
        issues.append("Too few rules discovered (<3)")
    if coverage['coverage_pct'] < 10:
        issues.append("Rule coverage below 10%")
    if validity.get('n_questionable', 0) > validity.get('n_valid', 0):
        issues.append("More questionable than valid rules")

    if not issues:
        print("VERDICT: Rules appear chemically valid with adequate coverage")
    else:
        print("VERDICT: Issues detected:")
        for issue in issues:
            print(f"  - {issue}")

    print("=" * 80)

    return {
        'training_result': training_result,
        'coverage': coverage,
        'conflict_analysis': conflict_analysis,
        'validity': validity,
        'semantic_memory': system['semantic_memory']
    }


if __name__ == '__main__':
    results = main()
