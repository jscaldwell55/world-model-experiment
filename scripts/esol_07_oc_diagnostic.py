"""
Comprehensive OC+FTB Diagnostic for ESOL

This script runs critical analyses to determine if offline consolidation
with synthetic data will work for ESOL solubility prediction.

Priority diagnostics:
1. Oracle gap deep dive - why is oracle 50% better?
2. Pseudo-label validity - are synthetic predictions trustworthy?
3. Bias region analysis - train vs test distribution mismatch
4. Feature importance - what drives predictions?
5. Sanity checks - data leakage, overfitting, split quality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs
from scipy.stats import pearsonr, spearmanr
from agents.molecular_world_model import MolecularWorldModel
from utils.context_spec_molecular import extract_molecular_context


def get_morgan_fp(smiles):
    """Get Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)


def get_descriptors(smiles):
    """Get molecular descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol)
    }


def rdkit_esol_prediction(smiles):
    """
    RDKit's built-in ESOL estimator.
    This serves as a proxy 'ground truth' for synthetic molecules.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # ESOL = 0.16 - 0.63*cLogP - 0.0062*MW + 0.066*RB - 0.74*AP
    logP = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = Descriptors.NumRotatableBonds(mol)
    ap = Descriptors.NumAromaticRings(mol)  # Aromatic rings

    esol = 0.16 - 0.63*logP - 0.0062*mw + 0.066*rb - 0.74*ap
    return esol


print("=" * 80)
print("OFFLINE CONSOLIDATION DIAGNOSTIC FOR ESOL")
print("=" * 80)

# Load data
train_df = pd.read_csv('memory/esol_train.csv')
test_df = pd.read_csv('memory/esol_test.csv')
baseline_preds = pd.read_csv('memory/esol_baseline_predictions.csv')

# Load model
model = MolecularWorldModel.load('memory/esol_baseline_model.pkl')

# Load existing analyses
baseline_results = json.load(open('memory/esol_baseline_results.json'))
oracle_analysis = json.load(open('memory/esol_oracle_analysis.json'))
bias_analysis = json.load(open('memory/esol_bias_analysis.json'))

print(f"\nBaseline MAE: {baseline_results['mae']:.3f}")
print(f"Oracle MAE:   0.183 (from oracle experiment)")
print(f"Gap:          {baseline_results['mae'] - 0.183:.3f} ({(baseline_results['mae'] - 0.183)/baseline_results['mae']:.1%})")

# ============================================================================
# SECTION 2: ORACLE GAP DEEP DIVE (CRITICAL)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: ORACLE GAP ANALYSIS (WHY IS ORACLE 50% BETTER?)")
print("=" * 80)

# Compute tanimoto similarities
print("\nComputing molecular similarities...")
train_fps = [get_morgan_fp(s) for s in train_df['smiles']]
train_fps = [fp for fp in train_fps if fp is not None]

test_similarities = []
test_errors = baseline_preds['error'].values

for test_smiles in test_df['smiles']:
    test_fp = get_morgan_fp(test_smiles)
    if test_fp is None:
        test_similarities.append(0.0)
        continue

    max_sim = max([DataStructs.TanimotoSimilarity(test_fp, train_fp)
                   for train_fp in train_fps])
    test_similarities.append(max_sim)

test_similarities = np.array(test_similarities)

# Bin by similarity and analyze errors
sim_bins = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 1.0)]
print("\nError vs similarity to training set:")
print(f"{'Similarity Range':<20} {'Count':>8} {'Mean Error':>12} {'Implication'}")
print("-" * 80)

for sim_min, sim_max in sim_bins:
    mask = (test_similarities >= sim_min) & (test_similarities < sim_max)
    count = mask.sum()
    if count > 0:
        mean_err = test_errors[mask].mean()

        if sim_min < 0.4:
            implication = "← EXTRAPOLATION (coverage gap)"
        elif sim_min < 0.5:
            implication = "← Moderate interpolation"
        else:
            implication = "← Strong interpolation"

        print(f"{sim_min:.1f} - {sim_max:.1f}  {count:8d} {mean_err:12.3f}  {implication}")

# Key finding
extrapolation_fraction = (test_similarities < 0.4).mean()
print(f"\n✅ Finding: {extrapolation_fraction:.1%} of test molecules are extrapolative (sim < 0.4)")

if extrapolation_fraction > 0.5:
    print("   → Oracle's advantage is primarily from COVERAGE")
    print("   → Synthetics should FILL GAPS in chemical space")
    coverage_vs_novelty = "COVERAGE"
else:
    print("   → Oracle's advantage is from SCAFFOLD NOVELTY")
    print("   → Synthetics should mimic test distribution patterns")
    coverage_vs_novelty = "NOVELTY"

# ============================================================================
# SECTION 3: PSEUDO-LABEL VALIDITY CHECK (CRITICAL)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: PSEUDO-LABEL VALIDITY (ARE SYNTHETIC PREDICTIONS TRUSTWORTHY?)")
print("=" * 80)

print("\nValidating world model predictions against RDKit ESOL...")

# Sample 30 training molecules with good coverage
np.random.seed(42)
sample_indices = np.random.choice(len(train_df), min(30, len(train_df)), replace=False)
sample_molecules = train_df.iloc[sample_indices]

# For each, get world model prediction and RDKit ESOL prediction
world_preds = []
rdkit_preds = []
ground_truth = []

for _, row in sample_molecules.iterrows():
    smiles = row['smiles']
    true_sol = row['solubility']

    # World model prediction
    world_pred, _, _ = model.predict_property(smiles)

    # RDKit ESOL prediction
    rdkit_pred = rdkit_esol_prediction(smiles)

    if rdkit_pred is not None:
        world_preds.append(world_pred)
        rdkit_preds.append(rdkit_pred)
        ground_truth.append(true_sol)

world_preds = np.array(world_preds)
rdkit_preds = np.array(rdkit_preds)
ground_truth = np.array(ground_truth)

# Correlations
world_vs_truth, _ = pearsonr(world_preds, ground_truth)
rdkit_vs_truth, _ = pearsonr(rdkit_preds, ground_truth)
world_vs_rdkit, _ = pearsonr(world_preds, rdkit_preds)

print(f"\nCorrelation analysis (n={len(world_preds)} molecules):")
print(f"  World model vs Ground truth:  r = {world_vs_truth:.3f}")
print(f"  RDKit ESOL vs Ground truth:   r = {rdkit_vs_truth:.3f}")
print(f"  World model vs RDKit ESOL:    r = {world_vs_rdkit:.3f}")

# Key finding
if world_vs_rdkit > 0.7:
    print(f"\n✅ Finding: Strong agreement (r={world_vs_rdkit:.3f})")
    print("   → Pseudo-labels will be RELIABLE")
    print("   → OC+FTB should work well")
    pseudo_label_quality = "HIGH"
elif world_vs_rdkit > 0.5:
    print(f"\n⚠️  Finding: Moderate agreement (r={world_vs_rdkit:.3f})")
    print("   → Pseudo-labels are NOISY but useful")
    print("   → Need high-quality filtering")
    pseudo_label_quality = "MEDIUM"
else:
    print(f"\n🚨 Finding: Weak agreement (r={world_vs_rdkit:.3f})")
    print("   → Pseudo-labels are UNRELIABLE")
    print("   → OC+FTB will likely fail")
    pseudo_label_quality = "LOW"

# ============================================================================
# SECTION 4: BIAS REGION CHARACTERIZATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: BIAS REGION ANALYSIS (TRAIN VS TEST DISTRIBUTION)")
print("=" * 80)

# Extract descriptors for all molecules
print("\nComputing molecular descriptors...")
train_descs = [get_descriptors(s) for s in train_df['smiles']]
test_descs = [get_descriptors(s) for s in test_df['smiles']]

# Get bias thresholds from analysis
biases = bias_analysis['biases']

print(f"\nFound {len(biases)} bias regions:")
for feature, info in biases.items():
    threshold = info['threshold']

    # Count train/test in bias region
    train_in_region = sum(1 for d in train_descs if d and d[feature] > threshold)
    test_in_region = sum(1 for d in test_descs if d and d[feature] > threshold)

    train_frac = train_in_region / len(train_df)
    test_frac = test_in_region / len(test_df)

    print(f"\n{feature} > {threshold:.2f}:")
    print(f"  Training:  {train_in_region:4d} / {len(train_df)} ({train_frac:.1%})")
    print(f"  Test:      {test_in_region:4d} / {len(test_df)} ({test_frac:.1%})")
    print(f"  Error increase: +{info['error_increase']:.3f} ({info['error_increase']/info['low_error']:.0%})")

    if test_frac > train_frac * 1.5:
        print(f"  → TEST OVERREPRESENTED - synthetics should target this region")
    elif train_frac > test_frac * 1.5:
        print(f"  → TRAIN OVERREPRESENTED - not a priority for synthetics")
    else:
        print(f"  → BALANCED - moderate priority")

# ============================================================================
# SECTION 5: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: FEATURE IMPORTANCE VS ERROR CORRELATION")
print("=" * 80)

# Get test descriptors and errors
test_desc_list = []
for idx, row in test_df.iterrows():
    desc = get_descriptors(row['smiles'])
    if desc:
        desc['error'] = baseline_preds.iloc[idx]['error']
        test_desc_list.append(desc)

test_desc_df = pd.DataFrame(test_desc_list)

print("\nFeature correlation with prediction error:")
print(f"{'Feature':<20} {'Correlation':>12} {'P-value':>10} {'Implication'}")
print("-" * 80)

feature_correlations = {}
for feature in ['MW', 'LogP', 'TPSA', 'NumRotatableBonds', 'NumAromaticRings']:
    if feature in test_desc_df.columns:
        corr, pval = spearmanr(test_desc_df[feature], test_desc_df['error'])
        feature_correlations[feature] = corr

        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""

        if abs(corr) > 0.3 and pval < 0.05:
            implication = "← Strong predictor of error"
        elif abs(corr) > 0.15 and pval < 0.05:
            implication = "← Weak predictor"
        else:
            implication = ""

        print(f"{feature:<20} {corr:12.3f} {pval:10.4f}{sig:>3} {implication}")

# ============================================================================
# SECTION 6: SANITY CHECKS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: SANITY CHECKS (DATA QUALITY)")
print("=" * 80)

# Check 1: Train/test overlap
print("\nCheck 1: Train/test SMILES overlap")
train_smiles_set = set(train_df['smiles'])
test_smiles_set = set(test_df['smiles'])
overlap = train_smiles_set & test_smiles_set

if len(overlap) == 0:
    print(f"  ✅ No overlap - clean split")
else:
    print(f"  🚨 OVERLAP FOUND: {len(overlap)} molecules in both sets!")
    print(f"     This is DATA LEAKAGE - results are invalid")

# Check 2: Train/test distribution comparison
print("\nCheck 2: Train vs test distribution")
train_descs_df = pd.DataFrame([d for d in train_descs if d])
test_descs_df = pd.DataFrame([d for d in test_descs if d])

from scipy.stats import ks_2samp
distribution_mismatches = []

for feature in ['MW', 'LogP', 'NumRotatableBonds']:
    statistic, pval = ks_2samp(train_descs_df[feature], test_descs_df[feature])

    if pval < 0.01:
        distribution_mismatches.append(feature)
        print(f"  ⚠️  {feature}: distributions differ (KS p={pval:.4f})")
    else:
        print(f"  ✅ {feature}: similar distributions (KS p={pval:.4f})")

if len(distribution_mismatches) > 2:
    print(f"\n  → Many distribution mismatches - test set may be out-of-distribution")
    print(f"  → This explains oracle's large advantage")

# Check 3: Overfitting check
print("\nCheck 3: Overfitting assessment")
train_preds = []
for smiles in train_df['smiles'].sample(min(200, len(train_df)), random_state=42):
    pred, _, _ = model.predict_property(smiles)
    train_preds.append(pred)

# This is a simplification - ideally we'd use cross-validation
print(f"  Note: Baseline model uses online learning, so train predictions")
print(f"  are not directly comparable. Using test performance as proxy.")
print(f"  Test R² = {baseline_results['r2']:.3f}")

if baseline_results['r2'] > 0.7:
    print(f"  ✅ Good generalization (R² > 0.7)")
else:
    print(f"  ⚠️  Potential underfitting (R² < 0.7)")

# ============================================================================
# SECTION 7: FINAL RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("FINAL RECOMMENDATIONS FOR OC+FTB")
print("=" * 80)

recommendations = []
go_decision = "PROCEED"

print("\n1. ORACLE GAP SOURCE:")
print(f"   Finding: {coverage_vs_novelty}")
if coverage_vs_novelty == "COVERAGE":
    print(f"   Recommendation: Generate synthetics to fill chemical space gaps")
    print(f"   Strategy: Use analog generation with diversity maximization")
    recommendations.append("diversity_maximization")
else:
    print(f"   Recommendation: Generate synthetics matching test distribution")
    print(f"   Strategy: Use bias-informed sampling")
    recommendations.append("bias_informed_sampling")

print("\n2. PSEUDO-LABEL QUALITY:")
print(f"   Finding: {pseudo_label_quality}")
if pseudo_label_quality == "HIGH":
    print(f"   Recommendation: PROCEED with standard OC+FTB")
elif pseudo_label_quality == "MEDIUM":
    print(f"   Recommendation: Use aggressive filtering (top 50% by uncertainty)")
    recommendations.append("aggressive_filtering")
else:
    print(f"   Recommendation: ABORT - pseudo-labels too noisy")
    go_decision = "ABORT"

print("\n3. BIAS REGIONS:")
bias_targets = []
for feature, info in biases.items():
    if info['error_increase'] > 0.08:  # Significant error increase
        bias_targets.append(feature)

if len(bias_targets) > 0:
    print(f"   Finding: {len(bias_targets)} high-impact bias regions")
    print(f"   Targets: {', '.join(bias_targets)}")
    print(f"   Recommendation: Oversample synthetics in these regions")
    recommendations.append(f"oversample_{','.join(bias_targets)}")
else:
    print(f"   Finding: No critical bias regions")
    print(f"   Recommendation: Use uniform sampling")

print("\n4. EXPECTED PERFORMANCE:")
if go_decision == "PROCEED":
    # Conservative estimate
    if pseudo_label_quality == "HIGH":
        synthetic_effectiveness = 0.5
    else:
        synthetic_effectiveness = 0.3

    synthetic_count = 600  # After filtering
    effective_molecules = synthetic_count * synthetic_effectiveness
    data_efficiency = oracle_analysis['data_efficiency']

    improvement_pct = (effective_molecules / len(train_df)) * 100 * data_efficiency
    expected_mae = baseline_results['mae'] * (1 - improvement_pct / 100)
    gap_captured = (baseline_results['mae'] - expected_mae) / (baseline_results['mae'] - 0.183)

    print(f"   Assumptions:")
    print(f"   - Generate 1200 synthetics, filter to {synthetic_count}")
    print(f"   - Synthetic effectiveness: {synthetic_effectiveness:.0%}")
    print(f"   - Data efficiency: {data_efficiency:.1f}x")
    print(f"\n   Projections:")
    print(f"   - Expected MAE: {expected_mae:.3f}")
    print(f"   - Gap captured: {gap_captured:.1%}")

    if gap_captured > 0.5:
        print(f"\n   ✅ EXCEEDS 50% target - STRONG GO")
    elif gap_captured > 0.3:
        print(f"\n   ✅ Meets 30% threshold - GO")
    else:
        print(f"\n   ⚠️  Below 30% target - MODIFY strategy")
        go_decision = "MODIFY"

print(f"\n{'='*80}")
print(f"FINAL DECISION: {go_decision}")
print(f"{'='*80}")

if go_decision == "PROCEED":
    print("\n✅ Proceed with OC+FTB")
    print(f"   Strategy: {', '.join(recommendations)}")
elif go_decision == "MODIFY":
    print("\n⚠️  Modify strategy before proceeding")
    print(f"   Issues: Pseudo-label quality or expected performance")
else:
    print("\n🚨 ABORT - OC+FTB unlikely to work")
    print(f"   Reason: Pseudo-labels are too unreliable")

# Save diagnostic results
diagnostic_results = {
    'oracle_gap_source': coverage_vs_novelty,
    'pseudo_label_quality': pseudo_label_quality,
    'pseudo_label_correlation': float(world_vs_rdkit),
    'extrapolation_fraction': float(extrapolation_fraction),
    'bias_targets': bias_targets,
    'distribution_mismatches': distribution_mismatches,
    'recommendations': recommendations,
    'decision': go_decision,
    'feature_correlations': {k: float(v) for k, v in feature_correlations.items()}
}

with open('memory/esol_oc_diagnostic.json', 'w') as f:
    json.dump(diagnostic_results, f, indent=2)

print(f"\n✅ Saved diagnostic results to memory/esol_oc_diagnostic.json")
print("\n" + "=" * 80)
