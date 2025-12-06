"""
ESOL Ablation Study: Evaluate Synthetic Data Impact

Conditions:
1. Baseline: No augmentation (902 train only)
2. Random synthetics: 600 random analogs
3. Diversity-only: 600 analogs, diversity maximized
4. Diversity + bias: 600 analogs, diversity + bias targeting
5. Oracle: +113 test molecules (upper bound)

Success criteria:
- Min improvement: 12% MAE reduction (0.365 → 0.321)
- Target improvement: 18% MAE reduction (0.365 → 0.299)
- Min gap captured: 25% of oracle gap
- Target gap captured: 35% of oracle gap
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from agents.molecular_world_model import MolecularWorldModel
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import random


def get_morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)


def compute_max_similarity(smiles, train_fps):
    """Compute max Tanimoto to training set."""
    fp = get_morgan_fp(smiles)
    if fp is None or len(train_fps) == 0:
        return 0.0
    return max([DataStructs.TanimotoSimilarity(fp, tfp) for tfp in train_fps])


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    condition_name: str,
    augment_df: pd.DataFrame = None,
    sample_weights: dict = None
) -> dict:
    """
    Train model and evaluate on test set.

    Args:
        train_df: Original training data
        test_df: Test data
        condition_name: Name of experimental condition
        augment_df: Optional synthetic data to add
        sample_weights: Optional dict with weight multipliers per condition

    Returns:
        Dict with evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"Training: {condition_name}")
    print(f"{'='*80}")

    # Combine datasets
    if augment_df is not None:
        combined_df = pd.concat([train_df, augment_df], ignore_index=True)
        print(f"Training size: {len(train_df)} original + {len(augment_df)} augmented = {len(combined_df)}")
    else:
        combined_df = train_df
        print(f"Training size: {len(combined_df)}")

    # Initialize model
    model = MolecularWorldModel(n_estimators=100, max_depth=10, min_samples_context=50)

    # Train
    print("Training...")
    for idx, row in combined_df.iterrows():
        smiles = row['smiles']
        solubility = row['solubility']

        # Update model
        model.update_belief(smiles, solubility)

        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(combined_df)}")

    stats = model.get_statistics()
    print(f"✅ Training complete:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Trained models: {len(stats['trained_models'])}/2")

    # Evaluate on test set
    print("\nEvaluating on test set...")

    predictions = []
    uncertainties = []
    actuals = []
    contexts = []

    for idx, row in test_df.iterrows():
        smiles = row['smiles']
        true_solubility = row['solubility']

        pred, unc, ctx = model.predict_property(smiles)

        predictions.append(pred)
        uncertainties.append(unc)
        actuals.append(true_solubility)
        contexts.append(ctx)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Compute metrics
    errors = np.abs(predictions - actuals)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print(f"\nOverall performance:")
    print(f"  MAE:  {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R²:   {r2:.3f}")

    # Compute metrics on extrapolation subset
    # (molecules with low similarity to training)
    train_fps = [get_morgan_fp(s) for s in train_df['smiles']]
    train_fps = [fp for fp in train_fps if fp is not None]

    test_sims = []
    for smiles in test_df['smiles']:
        sim = compute_max_similarity(smiles, train_fps)
        test_sims.append(sim)

    test_sims = np.array(test_sims)
    extrapolation_mask = test_sims < 0.4

    if extrapolation_mask.sum() > 0:
        extrap_mae = errors[extrapolation_mask].mean()
        print(f"\nExtrapolation subset (Tanimoto < 0.4, n={extrapolation_mask.sum()}):")
        print(f"  MAE: {extrap_mae:.3f}")
    else:
        extrap_mae = None

    # Compute metrics on bias region subset
    # (high MW or high LogP)
    from rdkit.Chem import Descriptors

    bias_mask = []
    for smiles in test_df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            in_bias = (mw > 337.46) or (logp > 4.20)
            bias_mask.append(in_bias)
        else:
            bias_mask.append(False)

    bias_mask = np.array(bias_mask)

    if bias_mask.sum() > 0:
        bias_mae = errors[bias_mask].mean()
        print(f"\nBias region subset (MW>337 OR LogP>4.2, n={bias_mask.sum()}):")
        print(f"  MAE: {bias_mae:.3f}")
    else:
        bias_mae = None

    results = {
        'condition': condition_name,
        'n_train': len(combined_df),
        'n_augmented': len(augment_df) if augment_df is not None else 0,
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'extrapolation_mae': float(extrap_mae) if extrap_mae is not None else None,
        'bias_mae': float(bias_mae) if bias_mae is not None else None,
        'n_extrapolation': int(extrapolation_mask.sum()),
        'n_bias': int(bias_mask.sum())
    }

    return results


def run_ablation_study(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    synthetics_all_df: pd.DataFrame = None,
    synthetics_diverse_df: pd.DataFrame = None,
    output_prefix: str = 'memory/esol_ablation'
):
    """
    Run full ablation study across conditions.
    """
    print("=" * 80)
    print("ESOL ABLATION STUDY")
    print("=" * 80)

    all_results = []
    baseline_mae = 0.365  # From baseline experiment
    oracle_mae = 0.183  # From oracle experiment
    oracle_gap = baseline_mae - oracle_mae

    # ========================================================================
    # Condition 1: Baseline (no augmentation)
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONDITION 1: BASELINE (No Augmentation)")
    print("=" * 80)

    baseline_results = train_and_evaluate(
        train_df=train_df,
        test_df=test_df,
        condition_name="Baseline"
    )
    all_results.append(baseline_results)

    # ========================================================================
    # Condition 2: Random Synthetics
    # ========================================================================
    if synthetics_all_df is not None and len(synthetics_all_df) > 0:
        print("\n" + "=" * 80)
        print("CONDITION 2: RANDOM SYNTHETICS")
        print("=" * 80)

        # Sample 600 random synthetics from all candidates
        n_random = min(600, len(synthetics_all_df))
        random_synthetics = synthetics_all_df.sample(n_random, random_state=42)

        random_results = train_and_evaluate(
            train_df=train_df,
            test_df=test_df,
            condition_name="Random Synthetics",
            augment_df=random_synthetics[['smiles', 'solubility']]
        )
        all_results.append(random_results)

    # ========================================================================
    # Condition 3: Diversity-Only Synthetics
    # ========================================================================
    if synthetics_diverse_df is not None and len(synthetics_diverse_df) > 0:
        print("\n" + "=" * 80)
        print("CONDITION 3: DIVERSITY-ONLY SYNTHETICS")
        print("=" * 80)

        # Use top 600 by diversity score (no bias weighting)
        diversity_synthetics = synthetics_all_df.nlargest(600, 'diversity_score')

        diversity_results = train_and_evaluate(
            train_df=train_df,
            test_df=test_df,
            condition_name="Diversity-Only",
            augment_df=diversity_synthetics[['smiles', 'solubility']]
        )
        all_results.append(diversity_results)

    # ========================================================================
    # Condition 4: Diversity + Bias Targeting
    # ========================================================================
    if synthetics_diverse_df is not None and len(synthetics_diverse_df) > 0:
        print("\n" + "=" * 80)
        print("CONDITION 4: DIVERSITY + BIAS TARGETING")
        print("=" * 80)

        # This is the filtered set (already selected with bias targeting)
        bias_results = train_and_evaluate(
            train_df=train_df,
            test_df=test_df,
            condition_name="Diversity + Bias",
            augment_df=synthetics_diverse_df[['smiles', 'solubility']]
        )
        all_results.append(bias_results)

    # ========================================================================
    # Condition 5: Oracle (test molecules as training)
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONDITION 5: ORACLE (Upper Bound)")
    print("=" * 80)

    oracle_results = train_and_evaluate(
        train_df=pd.concat([train_df, test_df], ignore_index=True),
        test_df=test_df,
        condition_name="Oracle"
    )
    all_results.append(oracle_results)

    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 80)

    results_df = pd.DataFrame(all_results)

    # Compute improvements and gap captured
    results_df['mae_improvement'] = baseline_mae - results_df['mae']
    results_df['mae_improvement_pct'] = (results_df['mae_improvement'] / baseline_mae) * 100
    results_df['gap_captured'] = (baseline_mae - results_df['mae']) / oracle_gap
    results_df['gap_captured_pct'] = results_df['gap_captured'] * 100

    print(f"\n{results_df[['condition', 'mae', 'mae_improvement_pct', 'gap_captured_pct']].to_string(index=False)}")

    # Success evaluation
    print("\n" + "=" * 80)
    print("SUCCESS EVALUATION")
    print("=" * 80)

    success_thresholds = {
        'min_improvement_pct': 12,
        'target_improvement_pct': 18,
        'min_gap_captured_pct': 25,
        'target_gap_captured_pct': 35
    }

    best_condition = results_df[results_df['condition'] != 'Oracle'].nsmallest(1, 'mae').iloc[0]

    print(f"\nBest condition: {best_condition['condition']}")
    print(f"  MAE: {best_condition['mae']:.3f}")
    print(f"  Improvement: {best_condition['mae_improvement_pct']:.1f}%")
    print(f"  Gap captured: {best_condition['gap_captured_pct']:.1f}%")

    if best_condition['mae_improvement_pct'] >= success_thresholds['target_improvement_pct']:
        print(f"\n✅ EXCEEDS target improvement ({success_thresholds['target_improvement_pct']}%)")
    elif best_condition['mae_improvement_pct'] >= success_thresholds['min_improvement_pct']:
        print(f"\n✅ MEETS minimum improvement ({success_thresholds['min_improvement_pct']}%)")
    else:
        print(f"\n🚨 BELOW minimum threshold")

    if best_condition['gap_captured_pct'] >= success_thresholds['target_gap_captured_pct']:
        print(f"✅ EXCEEDS target gap capture ({success_thresholds['target_gap_captured_pct']}%)")
    elif best_condition['gap_captured_pct'] >= success_thresholds['min_gap_captured_pct']:
        print(f"✅ MEETS minimum gap capture ({success_thresholds['min_gap_captured_pct']}%)")
    else:
        print(f"🚨 BELOW minimum threshold")

    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("Generating Ablation Plots")
    print("=" * 80)

    fig = plt.figure(figsize=(16, 10))

    # Plot 1: MAE comparison
    ax1 = plt.subplot(2, 3, 1)
    conditions = results_df['condition']
    maes = results_df['mae']
    colors = ['blue' if c == 'Baseline' else 'red' if c == 'Oracle' else 'green' for c in conditions]

    ax1.bar(range(len(conditions)), maes, color=colors, alpha=0.7)
    ax1.axhline(baseline_mae, color='blue', linestyle='--', alpha=0.5, label='Baseline')
    ax1.axhline(oracle_mae, color='red', linestyle='--', alpha=0.5, label='Oracle')
    ax1.set_xticks(range(len(conditions)))
    ax1.set_xticklabels(conditions, rotation=45, ha='right')
    ax1.set_ylabel('MAE')
    ax1.set_title('Test MAE by Condition')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')

    # Plot 2: Improvement percentage
    ax2 = plt.subplot(2, 3, 2)
    improvements = results_df[results_df['condition'] != 'Oracle']['mae_improvement_pct']
    conditions_no_oracle = results_df[results_df['condition'] != 'Oracle']['condition']

    ax2.bar(range(len(improvements)), improvements, color='green', alpha=0.7)
    ax2.axhline(success_thresholds['min_improvement_pct'], color='orange',
                linestyle='--', label='Min (12%)')
    ax2.axhline(success_thresholds['target_improvement_pct'], color='green',
                linestyle='--', label='Target (18%)')
    ax2.set_xticks(range(len(conditions_no_oracle)))
    ax2.set_xticklabels(conditions_no_oracle, rotation=45, ha='right')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('MAE Improvement vs Baseline')
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')

    # Plot 3: Gap captured
    ax3 = plt.subplot(2, 3, 3)
    gaps = results_df[results_df['condition'] != 'Oracle']['gap_captured_pct']

    ax3.bar(range(len(gaps)), gaps, color='purple', alpha=0.7)
    ax3.axhline(success_thresholds['min_gap_captured_pct'], color='orange',
                linestyle='--', label='Min (25%)')
    ax3.axhline(success_thresholds['target_gap_captured_pct'], color='green',
                linestyle='--', label='Target (35%)')
    ax3.set_xticks(range(len(conditions_no_oracle)))
    ax3.set_xticklabels(conditions_no_oracle, rotation=45, ha='right')
    ax3.set_ylabel('Gap Captured (%)')
    ax3.set_title('Oracle Gap Captured')
    ax3.legend()
    ax3.grid(alpha=0.3, axis='y')

    # Plot 4: Extrapolation performance
    ax4 = plt.subplot(2, 3, 4)
    extrap_maes = [r['extrapolation_mae'] for r in all_results if r['extrapolation_mae'] is not None]
    extrap_conditions = [r['condition'] for r in all_results if r['extrapolation_mae'] is not None]

    ax4.bar(range(len(extrap_maes)), extrap_maes, color='teal', alpha=0.7)
    ax4.set_xticks(range(len(extrap_conditions)))
    ax4.set_xticklabels(extrap_conditions, rotation=45, ha='right')
    ax4.set_ylabel('MAE')
    ax4.set_title('Extrapolation Subset Performance (sim < 0.4)')
    ax4.grid(alpha=0.3, axis='y')

    # Plot 5: Bias region performance
    ax5 = plt.subplot(2, 3, 5)
    bias_maes = [r['bias_mae'] for r in all_results if r['bias_mae'] is not None]
    bias_conditions = [r['condition'] for r in all_results if r['bias_mae'] is not None]

    ax5.bar(range(len(bias_maes)), bias_maes, color='coral', alpha=0.7)
    ax5.set_xticks(range(len(bias_conditions)))
    ax5.set_xticklabels(bias_conditions, rotation=45, ha='right')
    ax5.set_ylabel('MAE')
    ax5.set_title('Bias Region Performance (MW>337 OR LogP>4.2)')
    ax5.grid(alpha=0.3, axis='y')

    # Plot 6: Summary table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = f"""ABLATION STUDY SUMMARY

Best Condition: {best_condition['condition']}
  MAE: {best_condition['mae']:.3f}
  Improvement: {best_condition['mae_improvement_pct']:.1f}%
  Gap Captured: {best_condition['gap_captured_pct']:.1f}%

Baseline: {baseline_mae:.3f}
Oracle: {oracle_mae:.3f}
Gap: {oracle_gap:.3f}

Success Criteria:
  Min Improvement: {success_thresholds['min_improvement_pct']}%
  Target Improvement: {success_thresholds['target_improvement_pct']}%
  Min Gap: {success_thresholds['min_gap_captured_pct']}%
  Target Gap: {success_thresholds['target_gap_captured_pct']}%
"""

    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_results.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved plots to {output_prefix}_results.png")

    # Save results
    results_df.to_csv(f'{output_prefix}_results.csv', index=False)
    print(f"✅ Saved results to {output_prefix}_results.csv")

    with open(f'{output_prefix}_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ Saved results to {output_prefix}_results.json")

    return results_df


if __name__ == '__main__':
    print("Loading data...")
    train_df = pd.read_csv('memory/esol_train.csv')
    test_df = pd.read_csv('memory/esol_test.csv')

    # Load synthetics if available
    synthetics_all_df = None
    synthetics_diverse_df = None

    if os.path.exists('memory/esol_synthetics_candidates.csv'):
        synthetics_all_df = pd.read_csv('memory/esol_synthetics_candidates.csv')
        print(f"Loaded {len(synthetics_all_df)} synthetic candidates")

    if os.path.exists('memory/esol_synthetics_filtered.csv'):
        synthetics_diverse_df = pd.read_csv('memory/esol_synthetics_filtered.csv')
        print(f"Loaded {len(synthetics_diverse_df)} filtered synthetics")

    # Run ablation study
    results_df = run_ablation_study(
        train_df=train_df,
        test_df=test_df,
        synthetics_all_df=synthetics_all_df,
        synthetics_diverse_df=synthetics_diverse_df,
        output_prefix='memory/esol_ablation'
    )

    print("\n" + "=" * 80)
    print("Ablation study complete!")
    print("=" * 80)
