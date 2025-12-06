"""
ESOL Offline Consolidation: Full Pipeline with Robust Analog Generation

Complete pipeline using RWMol-based analog generation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from utils.robust_analog_generator import generate_diverse_analogs
from agents.molecular_world_model import MolecularWorldModel


print("=" * 80)
print("ESOL OFFLINE CONSOLIDATION: ROBUST PIPELINE")
print("=" * 80)

# ============================================================================
# STEP 1: Load data and model
# ============================================================================
print("\nSTEP 1: Loading data and model...")
train_df = pd.read_csv('memory/esol_train.csv')
test_df = pd.read_csv('memory/esol_test.csv')

print(f"  Train: {len(train_df)} molecules")
print(f"  Test: {len(test_df)} molecules")

# Load baseline model for pseudo-labeling
model = MolecularWorldModel.load('memory/esol_baseline_model.pkl')
print(f"  ✅ Loaded baseline model")

# ============================================================================
# STEP 2: Generate diverse synthetic candidates
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Generating diverse synthetic candidates (robust method)")
print("=" * 80)

candidates_df, selected_df = generate_diverse_analogs(
    training_df=train_df,
    n_candidates=1200,
    n_select=600,
    bias_mw_threshold=337.46,
    bias_logp_threshold=4.20,
    min_diversity=0.3
)

if len(candidates_df) == 0:
    print("\n🚨 ERROR: No candidates generated!")
    sys.exit(1)

if len(selected_df) == 0:
    print("\n🚨 ERROR: No analogs selected!")
    sys.exit(1)

print(f"\n✅ Generated {len(candidates_df)} candidates")
print(f"✅ Selected {len(selected_df)} diverse synthetics")

# ============================================================================
# STEP 3: Add pseudo-labels from world model
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Adding pseudo-labels from world model")
print("=" * 80)

print("Predicting solubility for selected synthetics...")
solubility_preds = []
uncertainties = []

for idx, row in selected_df.iterrows():
    smiles = row['smiles']
    pred, unc, _ = model.predict_property(smiles)
    solubility_preds.append(pred)
    uncertainties.append(unc)

selected_df['solubility'] = solubility_preds
selected_df['uncertainty'] = uncertainties

print(f"  ✅ Added pseudo-labels")
print(f"  Mean solubility: {np.mean(solubility_preds):.3f}")
print(f"  Std solubility: {np.std(solubility_preds):.3f}")
print(f"  Mean uncertainty: {np.mean(uncertainties):.3f}")

# Also add to candidates for random synthetics condition
print("\nPredicting solubility for all candidates...")
candidates_solubility = []
for idx, row in candidates_df.iterrows():
    smiles = row['smiles']
    pred, _, _ = model.predict_property(smiles)
    candidates_solubility.append(pred)

    if (idx + 1) % 200 == 0:
        print(f"  Processed {idx + 1}/{len(candidates_df)}...")

candidates_df['solubility'] = candidates_solubility

# Save updated dataframes
selected_df.to_csv('memory/esol_synthetics_filtered.csv', index=False)
candidates_df.to_csv('memory/esol_synthetics_candidates.csv', index=False)

print(f"\n✅ Saved synthetics with pseudo-labels")

# ============================================================================
# STEP 4: Validate synthetics
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Validating synthetic quality")
print("=" * 80)

# Import validation function
from scripts.esol_08_validate_synthetics import validate_synthetics

bias_regions = {'MW': 337.46, 'LogP': 4.20}

validation_results = validate_synthetics(
    synthetics_df=selected_df,
    train_df=train_df,
    test_df=test_df,
    model=model,
    bias_regions=bias_regions,
    output_prefix='memory/esol_synthetic_validation'
)

verdict = validation_results.get('verdict', 'UNKNOWN')
print(f"\nValidation verdict: {verdict}")

if verdict == 'FAIL':
    print("\n🚨 Validation FAILED. Stopping pipeline.")
    print("   Synthetics do not meet quality criteria.")
    print("   Review validation report and regenerate if needed.")
    sys.exit(1)
elif verdict == 'PARTIAL':
    print("\n⚠️  Validation PARTIAL. Proceeding with caution.")
else:
    print("\n✅ Validation PASSED. Proceeding to ablation study.")

# ============================================================================
# STEP 5: Run ablation study
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Running ablation study")
print("=" * 80)

from scripts.esol_09_ablation_study import run_ablation_study

results_df = run_ablation_study(
    train_df=train_df,
    test_df=test_df,
    synthetics_all_df=candidates_df,
    synthetics_diverse_df=selected_df,
    output_prefix='memory/esol_ablation'
)

# ============================================================================
# STEP 6: Final summary
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

baseline_mae = 0.365
oracle_mae = 0.183
oracle_gap = baseline_mae - oracle_mae

# Get best synthetic condition (exclude Oracle and Baseline)
synthetic_conditions = results_df[~results_df['condition'].isin(['Baseline', 'Oracle'])]

if len(synthetic_conditions) > 0:
    best_synthetic = synthetic_conditions.nsmallest(1, 'mae').iloc[0]

    print(f"\nBest synthetic condition: {best_synthetic['condition']}")
    print(f"  MAE: {best_synthetic['mae']:.3f}")
    print(f"  vs Baseline: {baseline_mae:.3f} → {best_synthetic['mae']:.3f} ({best_synthetic['mae_improvement_pct']:.1f}% improvement)")
    print(f"  vs Oracle: {oracle_mae:.3f} ({best_synthetic['gap_captured_pct']:.1f}% of gap captured)")

    # Success evaluation
    print("\n" + "=" * 80)
    print("SUCCESS EVALUATION")
    print("=" * 80)

    if best_synthetic['gap_captured_pct'] >= 35:
        print(f"\n🎉 STRONG SUCCESS - Captured {best_synthetic['gap_captured_pct']:.0f}% of oracle gap (target: 35%)")
        success_level = "STRONG SUCCESS"
    elif best_synthetic['gap_captured_pct'] >= 25:
        print(f"\n✅ SUCCESS - Captured {best_synthetic['gap_captured_pct']:.0f}% of oracle gap (target: 25%)")
        success_level = "SUCCESS"
    elif best_synthetic['gap_captured_pct'] >= 15:
        print(f"\n⚠️  PARTIAL - Captured {best_synthetic['gap_captured_pct']:.0f}% of oracle gap (below 25% target)")
        success_level = "PARTIAL"
    else:
        print(f"\n🚨 INSUFFICIENT - Only captured {best_synthetic['gap_captured_pct']:.0f}% of oracle gap")
        success_level = "INSUFFICIENT"

    if best_synthetic['mae_improvement_pct'] >= 18:
        print(f"🎉 STRONG IMPROVEMENT - {best_synthetic['mae_improvement_pct']:.1f}% MAE reduction (target: 18%)")
    elif best_synthetic['mae_improvement_pct'] >= 12:
        print(f"✅ GOOD IMPROVEMENT - {best_synthetic['mae_improvement_pct']:.1f}% MAE reduction (target: 12%)")
    else:
        print(f"⚠️  Below improvement target")

    print(f"\nKey findings:")
    print(f"  1. Synthetic generation: {len(selected_df)} diverse molecules")
    print(f"     - Mean diversity: {selected_df['diversity'].mean():.3f}")
    print(f"     - MW bias coverage: {selected_df['in_mw_bias'].mean()*100:.1f}%")
    print(f"     - LogP bias coverage: {selected_df['in_logp_bias'].mean()*100:.1f}%")
    print(f"  2. Validation: {verdict}")
    print(f"  3. Best approach: {best_synthetic['condition']}")
    print(f"  4. Performance gain: {best_synthetic['mae_improvement_pct']:.1f}%")
    print(f"  5. Gap captured: {best_synthetic['gap_captured_pct']:.1f}%")
    print(f"  6. Final result: {success_level}")

    print(f"\nFiles generated:")
    print(f"  memory/esol_synthetics_candidates.csv  ({len(candidates_df)} molecules)")
    print(f"  memory/esol_synthetics_filtered.csv    ({len(selected_df)} molecules)")
    print(f"  memory/esol_synthetic_validation.json")
    print(f"  memory/esol_synthetic_validation.png")
    print(f"  memory/esol_ablation_results.csv")
    print(f"  memory/esol_ablation_results.json")
    print(f"  memory/esol_ablation_results.png")

    # Save final summary
    summary = {
        'best_condition': best_synthetic['condition'],
        'baseline_mae': baseline_mae,
        'best_mae': float(best_synthetic['mae']),
        'oracle_mae': oracle_mae,
        'improvement_pct': float(best_synthetic['mae_improvement_pct']),
        'gap_captured_pct': float(best_synthetic['gap_captured_pct']),
        'n_synthetics': len(selected_df),
        'mean_diversity': float(selected_df['diversity'].mean()),
        'validation_verdict': verdict,
        'success_level': success_level
    }

    import json
    with open('memory/esol_final_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Saved final summary to memory/esol_final_summary.json")

else:
    print("\n⚠️  No synthetic conditions evaluated")

print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)
