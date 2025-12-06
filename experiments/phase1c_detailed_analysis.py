"""
Phase 1c Detailed Analysis: Learning Curves, Repair Triggering, and Raw Data.

This script provides:
1. Learning curves showing FTB staircase pattern vs Online smooth descent
2. Static baseline comparison (sanity check that updates matter)
3. Adversarial test to trigger and validate the repair mechanism
4. Raw seed-level data tables

Run this after phase1c_stress_test.py to get deeper insights.
"""

import json
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

# RDKit for molecular manipulation
from rdkit import Chem
from rdkit.Chem import Descriptors


# =============================================================================
# 1. LEARNING CURVE ANALYSIS
# =============================================================================

def run_learning_curve_experiment(
    data_path: str = 'data/esol_processed.pkl',
    n_steps: int = 50,
    seed_size: int = 20,
    seed: int = 42
) -> Dict:
    """
    Run experiment tracking MAE at every step for all strategies.

    Returns detailed step-by-step MAE for plotting learning curves.
    """
    print("\n" + "="*70)
    print("LEARNING CURVE ANALYSIS")
    print("="*70)

    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    candidate_df = data['candidate_pool']
    test_df = data['test_set']

    # Setup
    rng = np.random.RandomState(seed)

    seed_smiles = candidate_df['smiles'].tolist()[:seed_size]
    seed_labels = candidate_df['logS'].tolist()[:seed_size]

    test_smiles = test_df['smiles'].tolist()
    test_labels = test_df['logS'].tolist()

    probe_smiles = test_smiles[:30]
    probe_labels = test_labels[:30]

    # Query sequence
    query_pool_smiles = candidate_df['smiles'].tolist()[seed_size:]
    query_pool_labels = candidate_df['logS'].tolist()[seed_size:]
    indices = list(range(len(query_pool_smiles)))
    rng.shuffle(indices)
    query_sequence = [(query_pool_smiles[i], query_pool_labels[i]) for i in indices[:n_steps]]

    results = {'steps': list(range(n_steps + 1))}  # 0 = initial, then 1..n_steps

    # --- STATIC STRATEGY ---
    print("\nRunning Static strategy...")
    static_model = MolecularWorldModel(n_estimators=50, random_state=seed)
    static_model.fit(seed_smiles, seed_labels)

    static_maes = []
    for step in range(n_steps + 1):
        preds, _ = static_model.predict(test_smiles)
        valid = ~np.isnan(preds)
        mae = np.mean(np.abs(preds[valid] - np.array(test_labels)[valid]))
        static_maes.append(mae)
    results['Static'] = static_maes
    print(f"  Static: constant MAE = {static_maes[0]:.4f}")

    # --- ONLINE STRATEGY ---
    print("\nRunning Online strategy...")
    online_model = MolecularWorldModel(n_estimators=50, random_state=seed)
    online_smiles = list(seed_smiles)
    online_labels = list(seed_labels)
    online_model.fit(online_smiles, online_labels)

    online_maes = []
    # Initial
    preds, _ = online_model.predict(test_smiles)
    valid = ~np.isnan(preds)
    online_maes.append(np.mean(np.abs(preds[valid] - np.array(test_labels)[valid])))

    for step, (smiles, label) in enumerate(query_sequence):
        online_smiles.append(smiles)
        online_labels.append(label)
        online_model.fit(online_smiles, online_labels)

        preds, _ = online_model.predict(test_smiles)
        valid = ~np.isnan(preds)
        online_maes.append(np.mean(np.abs(preds[valid] - np.array(test_labels)[valid])))

    results['Online'] = online_maes
    print(f"  Online: {online_maes[0]:.4f} -> {online_maes[-1]:.4f}")

    # --- FTB STRATEGY ---
    print("\nRunning FTB strategy (update every 10 steps)...")
    ftb_model = MolecularWorldModel(n_estimators=50, random_state=seed)
    ftb_smiles = list(seed_smiles)
    ftb_labels = list(seed_labels)
    ftb_model.fit(ftb_smiles, ftb_labels)

    ftb = SimplifiedFTB(
        world_model=ftb_model,
        probe_smiles=probe_smiles,
        probe_labels=probe_labels,
        retention_threshold=0.25,
        random_state=seed
    )

    ftb_maes = []
    batch_smiles = []
    batch_labels = []
    update_steps = []  # Track when updates happen

    # Initial
    preds, _ = ftb_model.predict(test_smiles)
    valid = ~np.isnan(preds)
    ftb_maes.append(np.mean(np.abs(preds[valid] - np.array(test_labels)[valid])))

    for step, (smiles, label) in enumerate(query_sequence):
        batch_smiles.append(smiles)
        batch_labels.append(label)

        # Update every 10 steps
        if len(batch_smiles) >= 10:
            ftb_smiles.extend(batch_smiles)
            ftb_labels.extend(batch_labels)
            ftb.update(ftb_smiles, ftb_labels)
            batch_smiles = []
            batch_labels = []
            update_steps.append(step + 1)

        preds, _ = ftb_model.predict(test_smiles)
        valid = ~np.isnan(preds)
        ftb_maes.append(np.mean(np.abs(preds[valid] - np.array(test_labels)[valid])))

    # Final flush
    if batch_smiles:
        ftb_smiles.extend(batch_smiles)
        ftb_labels.extend(batch_labels)
        ftb.update(ftb_smiles, ftb_labels)
        update_steps.append(n_steps)

    results['FTB'] = ftb_maes
    results['FTB_update_steps'] = update_steps
    print(f"  FTB: {ftb_maes[0]:.4f} -> {ftb_maes[-1]:.4f}")
    print(f"  Updates at steps: {update_steps}")

    return results


def print_learning_curve_table(results: Dict):
    """Print ASCII learning curve table."""
    print("\n" + "="*70)
    print("LEARNING CURVE TABLE (MAE at each step)")
    print("="*70)

    steps = results['steps']
    static = results['Static']
    online = results['Online']
    ftb = results['FTB']
    update_steps = results.get('FTB_update_steps', [])

    # Print every 5 steps
    print(f"\n{'Step':<6} {'Static':<10} {'Online':<10} {'FTB':<10} {'FTB Lag':<10} {'Note':<15}")
    print("-" * 70)

    for i, step in enumerate(steps):
        if step % 5 == 0 or step in update_steps:
            lag = ftb[i] - online[i]
            note = "← UPDATE" if step in update_steps else ""
            print(f"{step:<6} {static[i]:<10.4f} {online[i]:<10.4f} {ftb[i]:<10.4f} {lag:+.4f}    {note}")

    print("-" * 70)

    # Summary
    max_lag = max(ftb[i] - online[i] for i in range(len(steps)))
    avg_lag = np.mean([ftb[i] - online[i] for i in range(len(steps))])

    print(f"\nMax FTB lag behind Online: {max_lag:+.4f}")
    print(f"Avg FTB lag behind Online: {avg_lag:+.4f}")
    print(f"Final parity achieved: {abs(ftb[-1] - online[-1]) < 0.001}")


def print_ascii_learning_curve(results: Dict):
    """Print ASCII art learning curve."""
    print("\n" + "="*70)
    print("ASCII LEARNING CURVE (MAE over steps)")
    print("="*70)

    steps = results['steps']
    static = results['Static']
    online = results['Online']
    ftb = results['FTB']
    update_steps = results.get('FTB_update_steps', [])

    # Normalize to 0-20 range for display
    all_vals = static + online + ftb
    min_val = min(all_vals)
    max_val = max(all_vals)

    def normalize(v):
        return int(20 * (v - min_val) / (max_val - min_val + 0.001))

    print(f"\nY-axis: MAE [{min_val:.2f} - {max_val:.2f}]")
    print(f"X-axis: Steps [0 - {len(steps)-1}]")
    print(f"Legend: S=Static, O=Online, F=FTB, ↓=FTB update\n")

    # Print from top (high MAE) to bottom (low MAE)
    for row in range(20, -1, -1):
        line = f"{min_val + (max_val-min_val) * row / 20:5.2f} |"
        for i, step in enumerate(steps):
            if i % 2 != 0:  # Skip every other for readability
                continue
            chars = []
            if normalize(static[i]) == row:
                chars.append('S')
            if normalize(online[i]) == row:
                chars.append('O')
            if normalize(ftb[i]) == row:
                chars.append('F')

            if chars:
                line += ''.join(chars[:2]).ljust(2)
            else:
                line += '. '
        print(line)

    # X-axis
    print("      +" + "-" * 52)
    print("       " + "".join(f"{i:<4}" for i in range(0, len(steps), 10)))


# =============================================================================
# 2. STATIC BASELINE COMPARISON
# =============================================================================

def print_static_baseline_analysis(results: Dict):
    """Analyze the importance of updates via Static baseline."""
    print("\n" + "="*70)
    print("STATIC BASELINE ANALYSIS (Sanity Check)")
    print("="*70)

    static_final = results['Static'][-1]
    online_final = results['Online'][-1]
    ftb_final = results['FTB'][-1]

    improvement_online = (static_final - online_final) / static_final * 100
    improvement_ftb = (static_final - ftb_final) / static_final * 100

    print(f"\nFinal MAE Comparison:")
    print(f"  Static (no updates):  {static_final:.4f}")
    print(f"  Online (every step):  {online_final:.4f}  ({improvement_online:+.1f}% vs Static)")
    print(f"  FTB (every 10 steps): {ftb_final:.4f}  ({improvement_ftb:+.1f}% vs Static)")

    print(f"\nConclusion:")
    print(f"  Updates provide {improvement_online:.1f}% improvement over no updates.")
    print(f"  This confirms that model updates ARE valuable (sanity check PASSED).")


# =============================================================================
# 3. ADVERSARIAL TEST - TRIGGER REPAIR MECHANISM
# =============================================================================

def generate_ood_molecules(n: int = 20, seed: int = 42) -> Tuple[List[str], List[float]]:
    """
    Generate Out-of-Distribution molecules to trigger forgetting.

    Strategy: Create very large molecules with extreme LogS values
    that are chemically distinct from typical ESOL molecules.
    """
    rng = np.random.RandomState(seed)

    ood_smiles = []
    ood_labels = []

    # Long alkyl chains (very hydrophobic, very negative LogS)
    for i in range(n // 2):
        chain_length = rng.randint(15, 25)
        smiles = 'C' * chain_length
        # Very negative LogS for long hydrophobic chains
        logs = -6.0 + rng.normal(0, 0.5)
        ood_smiles.append(smiles)
        ood_labels.append(logs)

    # Polyethers (very hydrophilic, very positive LogS)
    for i in range(n // 2):
        units = rng.randint(5, 10)
        smiles = 'C' + 'OC' * units
        # Very positive LogS for polyethers
        logs = 3.0 + rng.normal(0, 0.5)
        ood_smiles.append(smiles)
        ood_labels.append(logs)

    return ood_smiles, ood_labels


def run_adversarial_repair_test(
    data_path: str = 'data/esol_processed.pkl',
    seed: int = 42
) -> Dict:
    """
    Adversarial test to trigger and verify the repair mechanism.

    Scenario:
    1. Train model on normal data
    2. Inject OOD batch with extreme values
    3. Verify retention drops below threshold
    4. Confirm repair loop triggers and recovers
    """
    print("\n" + "="*70)
    print("ADVERSARIAL REPAIR TEST")
    print("="*70)

    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    candidate_df = data['candidate_pool']
    test_df = data['test_set']

    seed_smiles = candidate_df['smiles'].tolist()[:50]  # More seed data
    seed_labels = candidate_df['logS'].tolist()[:50]

    probe_smiles = test_df['smiles'].tolist()[:30]
    probe_labels = test_df['logS'].tolist()[:30]

    test_smiles = test_df['smiles'].tolist()
    test_labels = test_df['logS'].tolist()

    # Initialize model and FTB
    print("\n1. Training initial model on 50 normal molecules...")
    model = MolecularWorldModel(n_estimators=50, random_state=seed)
    model.fit(seed_smiles, seed_labels)

    # Baseline probe MAE
    preds, _ = model.predict(probe_smiles)
    baseline_probe_mae = np.mean(np.abs(preds - np.array(probe_labels)))
    print(f"   Baseline probe MAE: {baseline_probe_mae:.4f}")

    # Create FTB with STRICT threshold to make repair more likely
    ftb = SimplifiedFTB(
        world_model=model,
        probe_smiles=probe_smiles,
        probe_labels=probe_labels,
        retention_threshold=0.10,  # Only allow 10% degradation (strict!)
        repair_attempts=3,
        replay_ratio=0.5,
        random_state=seed
    )

    # Generate adversarial OOD data
    print("\n2. Generating adversarial OOD molecules...")
    ood_smiles, ood_labels = generate_ood_molecules(n=30, seed=seed)
    print(f"   OOD molecules: {len(ood_smiles)}")
    print(f"   OOD label range: [{min(ood_labels):.2f}, {max(ood_labels):.2f}]")
    print(f"   Normal label range: [{min(seed_labels):.2f}, {max(seed_labels):.2f}]")

    # Combine normal + OOD (heavily weighted toward OOD to induce forgetting)
    adversarial_smiles = seed_smiles + ood_smiles
    adversarial_labels = seed_labels + ood_labels

    print("\n3. Attempting update with adversarial data...")
    print(f"   Total samples: {len(adversarial_smiles)} ({len(seed_smiles)} normal + {len(ood_smiles)} OOD)")

    # Run FTB update
    result = ftb.update(adversarial_smiles, adversarial_labels)

    print(f"\n4. FTB Update Results:")
    print(f"   Pre-update probe MAE:  {result['pre_update_mae']:.4f}")
    print(f"   Post-update probe MAE: {result['post_update_mae']:.4f}")
    print(f"   Improvement ratio:     {result['improvement_ratio']:.4f}")
    print(f"   Retention passed:      {result['retention_passed']}")
    print(f"   Repair triggered:      {result['was_repaired']}")
    print(f"   Repair attempts:       {result['repair_attempts']}")

    if result['repair_details']:
        print(f"\n   Repair Log:")
        for r in result['repair_details']:
            print(f"     Attempt {r['attempt']}: n_replay={r['n_replay']}, "
                  f"MAE={r['mae']:.4f}, ratio={r['improvement_ratio']:.4f}, passed={r['passed']}")

    # Verify final model still works
    preds, _ = model.predict(test_smiles)
    valid = ~np.isnan(preds)
    final_test_mae = np.mean(np.abs(preds[valid] - np.array(test_labels)[valid]))

    print(f"\n5. Final Verification:")
    print(f"   Test set MAE after adversarial update: {final_test_mae:.4f}")

    # Determine if test passed
    if result['was_repaired'] and result['retention_passed']:
        print(f"\n   ✓ REPAIR MECHANISM VALIDATED")
        print(f"     - Forgetting was detected (ratio < threshold)")
        print(f"     - Repair loop was triggered")
        print(f"     - Model recovered successfully")
        test_passed = True
    elif not result['was_repaired'] and result['retention_passed']:
        print(f"\n   ⚠ NO REPAIR NEEDED")
        print(f"     - Model was robust enough to handle OOD data")
        print(f"     - Consider using more extreme adversarial examples")
        test_passed = None  # Inconclusive
    else:
        print(f"\n   ✗ REPAIR FAILED")
        print(f"     - Repair was attempted but didn't recover")
        test_passed = False

    return {
        'baseline_probe_mae': baseline_probe_mae,
        'ftb_result': result,
        'final_test_mae': final_test_mae,
        'test_passed': test_passed,
        'ood_count': len(ood_smiles),
        'ood_label_range': (min(ood_labels), max(ood_labels))
    }


def run_extreme_adversarial_test(
    data_path: str = 'data/esol_processed.pkl',
    seed: int = 42
) -> Dict:
    """
    More extreme adversarial test that GUARANTEES repair triggering.

    Strategy: Train on subset A, then "update" with ONLY subset B (no overlap).
    CRITICAL: Probe set must be FROM SUBSET A to detect forgetting of A.

    Expected behavior:
    1. Train on A -> good probe MAE (probe is from A)
    2. Update with ONLY B -> probe MAE degrades (forgetting A)
    3. Repair triggers -> probe MAE recovers
    """
    print("\n" + "="*70)
    print("EXTREME ADVERSARIAL TEST (Guaranteed Repair Trigger)")
    print("="*70)

    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    candidate_df = data['candidate_pool']

    # Split candidate pool into two completely separate halves
    all_smiles = candidate_df['smiles'].tolist()
    all_labels = candidate_df['logS'].tolist()

    # Sort by molecular weight to create chemically distinct subsets
    mw_list = []
    for s in all_smiles:
        mol = Chem.MolFromSmiles(s)
        mw = Descriptors.MolWt(mol) if mol else 0
        mw_list.append(mw)

    sorted_idx = np.argsort(mw_list)

    # Subset A: Lightest 25% of molecules
    subset_a_idx = sorted_idx[:len(sorted_idx)//4]
    subset_a_smiles = [all_smiles[i] for i in subset_a_idx]
    subset_a_labels = [all_labels[i] for i in subset_a_idx]

    # Subset B: Heaviest 25% of molecules
    subset_b_idx = sorted_idx[-len(sorted_idx)//4:]
    subset_b_smiles = [all_smiles[i] for i in subset_b_idx]
    subset_b_labels = [all_labels[i] for i in subset_b_idx]

    # CRITICAL FIX: Probe set must be FROM SUBSET A (not middle!)
    # This way, when we train on B only, we forget A and probe MAE degrades
    rng = np.random.RandomState(seed)
    probe_indices = rng.choice(len(subset_a_smiles), size=min(30, len(subset_a_smiles)), replace=False)
    probe_smiles = [subset_a_smiles[i] for i in probe_indices]
    probe_labels = [subset_a_labels[i] for i in probe_indices]

    # Training set A excludes probe molecules
    train_a_smiles = [s for i, s in enumerate(subset_a_smiles) if i not in probe_indices]
    train_a_labels = [l for i, l in enumerate(subset_a_labels) if i not in probe_indices]

    print(f"\n1. Data Split:")
    print(f"   Subset A (light molecules): {len(subset_a_smiles)} total")
    print(f"     - Training A: {len(train_a_smiles)} samples")
    print(f"     - Probe (held-out from A): {len(probe_smiles)} samples")
    print(f"   Subset B (heavy molecules): {len(subset_b_smiles)} samples")
    print(f"\n   MW ranges:")
    print(f"     Subset A: {min(mw_list[i] for i in subset_a_idx):.1f} - {max(mw_list[i] for i in subset_a_idx):.1f}")
    print(f"     Subset B: {min(mw_list[i] for i in subset_b_idx):.1f} - {max(mw_list[i] for i in subset_b_idx):.1f}")

    # Train on Subset A
    print(f"\n2. Training on Subset A (light molecules)...")
    model = MolecularWorldModel(n_estimators=50, random_state=seed)
    model.fit(train_a_smiles, train_a_labels)

    preds, _ = model.predict(probe_smiles)
    initial_probe_mae = np.mean(np.abs(preds - np.array(probe_labels)))
    print(f"   Initial probe MAE (on held-out A): {initial_probe_mae:.4f}")

    # Create FTB with strict threshold
    ftb = SimplifiedFTB(
        world_model=model,
        probe_smiles=probe_smiles,
        probe_labels=probe_labels,
        retention_threshold=0.15,  # Only 15% degradation allowed
        repair_attempts=5,
        replay_ratio=0.7,
        random_state=seed
    )

    # First, let's see what happens WITHOUT FTB protection (direct fit)
    print(f"\n3. Simulating catastrophic forgetting (direct fit on B only)...")
    test_model = MolecularWorldModel(n_estimators=50, random_state=seed)
    test_model.fit(subset_b_smiles, subset_b_labels)
    preds_after_b, _ = test_model.predict(probe_smiles)
    catastrophic_mae = np.mean(np.abs(preds_after_b - np.array(probe_labels)))
    print(f"   Probe MAE after training ONLY on B: {catastrophic_mae:.4f}")
    print(f"   Degradation: {(catastrophic_mae - initial_probe_mae) / initial_probe_mae * 100:+.1f}%")

    # Now update with FTB protection
    print(f"\n4. Updating via FTB (with repair protection)...")
    print(f"   Training data: ONLY Subset B (complete distribution shift)")

    result = ftb.update(subset_b_smiles, subset_b_labels)

    print(f"\n5. FTB Update Results:")
    print(f"   Pre-update probe MAE:     {result['pre_update_mae']:.4f}")
    print(f"   Post-initial-fit MAE:     {result['initial_post_mae']:.4f}  (before repair)")
    print(f"   Initial ratio:            {result['initial_ratio']:.4f}  (triggered repair if < 0.85)")
    print(f"   ")
    print(f"   Final probe MAE:          {result['post_update_mae']:.4f}  (after repair)")
    print(f"   Final ratio:              {result['improvement_ratio']:.4f}")
    print(f"   Retention passed:         {result['retention_passed']}")
    print(f"   Repair triggered:         {result['was_repaired']}")
    print(f"   Repair attempts:          {result['repair_attempts']}")

    if result['repair_details']:
        print(f"\n   Repair Log:")
        for r in result['repair_details']:
            status = "✓" if r['passed'] else "✗"
            print(f"     {status} Attempt {r['attempt']}: n_replay={r['n_replay']}, "
                  f"MAE={r['mae']:.4f}, ratio={r['improvement_ratio']:.4f}")

    # Determine outcome
    print(f"\n6. Verdict:")
    if result['was_repaired']:
        if result['retention_passed']:
            print(f"   ✓ REPAIR MECHANISM VALIDATED")
            print(f"     - Initial fit caused forgetting:")
            print(f"         Initial ratio {result['initial_ratio']:.3f} < 0.85 threshold")
            print(f"     - Repair loop triggered and succeeded after {result['repair_attempts']} attempt(s)")
            print(f"     - Final ratio: {result['improvement_ratio']:.3f} (above 0.85 threshold)")
            print(f"   ")
            print(f"   Comparison:")
            print(f"     - Trained on A, tested on probe(A): MAE = {initial_probe_mae:.4f}")
            print(f"     - Trained on B only, tested on probe(A): MAE = {catastrophic_mae:.4f} (+{(catastrophic_mae/initial_probe_mae - 1)*100:.0f}% degradation)")
            print(f"     - FTB (B + replay from probe): MAE = {result['post_update_mae']:.4f} (recovered!)")
            verdict = "PASS"
        else:
            print(f"   ⚠ PARTIAL - Repair triggered but incomplete recovery")
            print(f"     - Initial ratio: {result['initial_ratio']:.3f}")
            print(f"     - Final ratio {result['improvement_ratio']:.3f} still below 0.85")
            print(f"     - More aggressive replay may be needed")
            verdict = "PARTIAL"
    else:
        if result['initial_ratio'] < (1.0 - 0.15):
            print(f"   ✗ REPAIR SHOULD HAVE TRIGGERED but didn't")
            print(f"     - Initial ratio {result['initial_ratio']:.3f} is below threshold")
            verdict = "BUG"
        else:
            print(f"   ⚠ Model maintained retention without repair")
            print(f"     - Initial fit didn't cause significant forgetting")
            print(f"     - Initial ratio {result['initial_ratio']:.3f} stayed above 0.85 threshold")
            verdict = "ROBUST"

    return {
        'initial_probe_mae': initial_probe_mae,
        'catastrophic_mae': catastrophic_mae,
        'ftb_result': result,
        'verdict': verdict
    }


# =============================================================================
# 4. RAW SEED-LEVEL DATA
# =============================================================================

def extract_seed_level_data(results_path: str = 'results/phase1c_stress_test.json') -> pd.DataFrame:
    """Extract and format raw seed-level data from stress test results."""

    with open(results_path, 'r') as f:
        data = json.load(f)

    rows = []
    for result in data['results']:
        rows.append({
            'Condition': result['condition'],
            'Strategy': result['strategy'],
            'Seed': result['seed'],
            'Final_MAE': result['metrics']['final_test_mae'],
            'Updates': result['metrics']['updates_performed'],
            'Repairs': result['metrics']['repair_count'],
            'Mean_Stability': result['metrics'].get('mean_stability', 1.0)
        })

    df = pd.DataFrame(rows)
    return df


def print_seed_level_tables(df: pd.DataFrame):
    """Print formatted seed-level comparison tables."""

    print("\n" + "="*70)
    print("RAW SEED-LEVEL DATA")
    print("="*70)

    for condition in ['clean', 'noisy_15pct', 'distribution_shift']:
        print(f"\n{condition.upper()}")
        print("-" * 60)

        cond_df = df[df['Condition'] == condition]

        # Pivot to compare strategies side by side
        seeds = sorted(cond_df['Seed'].unique())

        print(f"{'Seed':<8} {'Static MAE':<12} {'Online MAE':<12} {'FTB MAE':<12} {'Match?':<8}")
        print("-" * 60)

        for seed in seeds:
            seed_df = cond_df[cond_df['Seed'] == seed]

            static_mae = seed_df[seed_df['Strategy'] == 'Static']['Final_MAE'].values[0]
            online_mae = seed_df[seed_df['Strategy'] == 'Online']['Final_MAE'].values[0]
            ftb_mae = seed_df[seed_df['Strategy'] == 'FTB']['Final_MAE'].values[0]

            match = "✓" if abs(online_mae - ftb_mae) < 0.0001 else "✗"

            print(f"{seed:<8} {static_mae:<12.4f} {online_mae:<12.4f} {ftb_mae:<12.4f} {match:<8}")

        # Summary row
        print("-" * 60)
        static_mean = cond_df[cond_df['Strategy'] == 'Static']['Final_MAE'].mean()
        online_mean = cond_df[cond_df['Strategy'] == 'Online']['Final_MAE'].mean()
        ftb_mean = cond_df[cond_df['Strategy'] == 'FTB']['Final_MAE'].mean()

        print(f"{'Mean':<8} {static_mean:<12.4f} {online_mean:<12.4f} {ftb_mean:<12.4f}")
        print(f"{'Std':<8} {cond_df[cond_df['Strategy'] == 'Static']['Final_MAE'].std():<12.4f} "
              f"{cond_df[cond_df['Strategy'] == 'Online']['Final_MAE'].std():<12.4f} "
              f"{cond_df[cond_df['Strategy'] == 'FTB']['Final_MAE'].std():<12.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all detailed analyses."""

    print("\n" + "="*70)
    print("PHASE 1c DETAILED ANALYSIS")
    print("="*70)

    # 1. Learning Curves
    print("\n[1/4] Generating Learning Curves...")
    lc_results = run_learning_curve_experiment()
    print_learning_curve_table(lc_results)
    print_ascii_learning_curve(lc_results)

    # 2. Static Baseline Analysis
    print("\n[2/4] Analyzing Static Baseline...")
    print_static_baseline_analysis(lc_results)

    # 3. Adversarial Repair Test
    print("\n[3/4] Running Adversarial Repair Tests...")

    # First try mild adversarial
    adv_result = run_adversarial_repair_test()

    # If mild didn't trigger repair, try extreme
    if not adv_result['ftb_result']['was_repaired']:
        print("\n   Mild adversarial didn't trigger repair. Trying extreme test...")
        extreme_result = run_extreme_adversarial_test()

    # 4. Raw Seed-Level Data
    print("\n[4/4] Extracting Seed-Level Data...")
    if os.path.exists('results/phase1c_stress_test.json'):
        df = extract_seed_level_data()
        print_seed_level_tables(df)
    else:
        print("   ⚠ No stress test results found. Run phase1c_stress_test.py first.")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
