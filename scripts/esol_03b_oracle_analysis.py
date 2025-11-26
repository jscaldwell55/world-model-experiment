"""
Analyze why oracle performs so much better than baseline.

This diagnostic script tests hypotheses about oracle's advantage:
1. Test molecules are similar to train (interpolation vs extrapolation)
2. Errors correlate with distance to training set
3. Context-specific sample size impacts performance
4. Data efficiency analysis

Insights will inform synthetic data generation strategy.
"""
import pandas as pd
import numpy as np
from agents.molecular_world_model import MolecularWorldModel
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_morgan_fingerprint(smiles):
    """Get Morgan fingerprint for similarity calculations."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)


def main():
    print("=" * 60)
    print("Oracle Performance Analysis")
    print("=" * 60)

    # Load data
    train_df = pd.read_csv('memory/esol_train.csv')
    test_df = pd.read_csv('memory/esol_test.csv')
    baseline_preds = pd.read_csv('memory/esol_baseline_predictions.csv')

    # Load results
    baseline_results = json.load(open('memory/esol_baseline_results.json'))
    oracle_results = json.load(open('memory/esol_oracle_results.json'))

    baseline_mae = baseline_results['mae']
    oracle_mae = oracle_results['oracle_mae']

    print(f"\nBaseline MAE: {baseline_mae:.3f}")
    print(f"Oracle MAE:   {oracle_mae:.3f}")
    print(f"Gap:          {baseline_mae - oracle_mae:.3f} ({(baseline_mae-oracle_mae)/baseline_mae:.1%})")

    # Hypothesis 1: Test molecules are similar to train
    print(f"\n{'='*60}")
    print("HYPOTHESIS 1: Test molecules are similar to train")
    print("=" * 60)

    train_fps = []
    for smiles in train_df['smiles']:
        fp = get_morgan_fingerprint(smiles)
        if fp is not None:
            train_fps.append(fp)

    similarities_to_train = []
    for test_smiles in test_df['smiles']:
        test_fp = get_morgan_fingerprint(test_smiles)
        if test_fp is None:
            continue

        max_sim = max([DataStructs.TanimotoSimilarity(test_fp, train_fp)
                       for train_fp in train_fps])
        similarities_to_train.append(max_sim)

    print(f"Test molecule similarity to nearest train molecule:")
    print(f"  Mean: {np.mean(similarities_to_train):.3f}")
    print(f"  Median: {np.median(similarities_to_train):.3f}")
    print(f"  Q1-Q3: {np.percentile(similarities_to_train, 25):.3f} - {np.percentile(similarities_to_train, 75):.3f}")
    print(f"  Min-Max: {np.min(similarities_to_train):.3f} - {np.max(similarities_to_train):.3f}")

    # If mean similarity > 0.6, test is very similar to train
    if np.mean(similarities_to_train) > 0.6:
        print(f"\n✅ Test molecules are SIMILAR to train (mean Tanimoto = {np.mean(similarities_to_train):.3f})")
        print(f"   Oracle benefits from seeing near-duplicates!")
    else:
        print(f"\n⚠️  Test molecules are DIVERSE from train (mean Tanimoto = {np.mean(similarities_to_train):.3f})")
        print(f"   Oracle's advantage comes from other factors")

    # Hypothesis 2: Error correlates with distance to train
    print(f"\n{'='*60}")
    print("HYPOTHESIS 2: Errors are higher for molecules far from train")
    print("=" * 60)

    baseline_errors = baseline_preds['error'].values

    from scipy.stats import pearsonr
    corr, p_value = pearsonr(similarities_to_train, -baseline_errors)  # Negative because higher sim → lower error

    print(f"Correlation(similarity_to_train, -error): r={corr:.3f}, p={p_value:.4f}")

    if abs(corr) > 0.3 and p_value < 0.05:
        print(f"✅ Strong correlation! Molecules far from train have higher errors")
        print(f"   Implication: Synthetics should cover underrepresented chemical space")
    else:
        print(f"⚠️  Weak correlation. Error doesn't depend strongly on similarity")
        print(f"   Implication: Other factors (molecular complexity, etc.) matter more")

    # Hypothesis 3: Sample size per context matters
    print(f"\n{'='*60}")
    print("HYPOTHESIS 3: Context-specific sample size impacts errors")
    print("=" * 60)

    # Get context counts
    from utils.context_spec_molecular import extract_molecular_context

    train_contexts = {}
    for smiles in train_df['smiles']:
        ctx = extract_molecular_context({'smiles': smiles})
        train_contexts[ctx] = train_contexts.get(ctx, 0) + 1

    test_contexts = {}
    test_errors_by_context = {}
    for idx, smiles in enumerate(test_df['smiles']):
        ctx = extract_molecular_context({'smiles': smiles})
        test_contexts[ctx] = test_contexts.get(ctx, 0) + 1

        if ctx not in test_errors_by_context:
            test_errors_by_context[ctx] = []
        test_errors_by_context[ctx].append(baseline_errors[idx])

    print(f"\nContext-specific analysis:")
    for ctx in sorted(train_contexts.keys()):
        train_n = train_contexts.get(ctx, 0)
        test_n = test_contexts.get(ctx, 0)

        if ctx in test_errors_by_context:
            ctx_errors = test_errors_by_context[ctx]
            mean_error = np.mean(ctx_errors)

            print(f"  {ctx}:")
            print(f"    Train samples: {train_n}")
            print(f"    Test samples:  {test_n}")
            print(f"    Mean error:    {mean_error:.3f}")

    # Check if context with fewer samples has higher errors
    context_list = list(train_contexts.keys())
    if len(context_list) == 2:
        ctx1, ctx2 = context_list
        n1, n2 = train_contexts[ctx1], train_contexts[ctx2]

        if ctx1 in test_errors_by_context and ctx2 in test_errors_by_context:
            err1 = np.mean(test_errors_by_context[ctx1])
            err2 = np.mean(test_errors_by_context[ctx2])

            smaller_ctx = ctx1 if n1 < n2 else ctx2
            larger_ctx = ctx2 if n1 < n2 else ctx1
            smaller_err = err1 if n1 < n2 else err2
            larger_err = err2 if n1 < n2 else err1

            if smaller_err > larger_err * 1.1:
                print(f"\n✅ Context with fewer samples ({smaller_ctx}: {min(n1,n2)}) has higher error ({smaller_err:.3f})")
                print(f"   Implication: Focus synthetics on underrepresented context")
            else:
                print(f"\n⚠️  Both contexts have similar errors despite sample size difference")
                print(f"   Implication: Context balance may not be critical")

    # Hypothesis 4: Oracle advantage breakdown
    print(f"\n{'='*60}")
    print("HYPOTHESIS 4: Where does oracle's advantage come from?")
    print("=" * 60)

    # If oracle sees test data, it's essentially adding 113 molecules
    # to a 902-molecule dataset → 12.5% more data
    data_increase_pct = len(test_df) / len(train_df) * 100
    improvement_pct = (baseline_mae - oracle_mae) / baseline_mae * 100

    efficiency = improvement_pct / data_increase_pct

    print(f"Data increase: {data_increase_pct:.1f}% (added {len(test_df)} molecules)")
    print(f"Performance improvement: {improvement_pct:.1f}%")
    print(f"Efficiency: {efficiency:.1f}x")
    print(f"  (Every 1% more data → {efficiency:.1f}% better performance)")

    if efficiency > 3.0:
        print(f"\n🎯 VERY HIGH efficiency! Oracle data is exceptionally valuable")
        print(f"   Implication: Synthetics that mimic test distribution will be very effective")
    elif efficiency > 1.5:
        print(f"\n✅ Good efficiency. Oracle data is valuable")
        print(f"   Implication: Quality synthetics should show measurable improvement")
    else:
        print(f"\n⚠️  Low efficiency. Oracle data isn't much better than random samples")
        print(f"   Implication: Synthetic generation strategy needs to be very selective")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 60)

    print(f"\n1. Oracle's {improvement_pct:.0f}% improvement comes from:")
    print(f"   - {data_increase_pct:.1f}% more training data")
    print(f"   - {efficiency:.1f}x data efficiency")

    print(f"\n2. For OC+FTB to succeed, synthetics should:")
    if np.mean(similarities_to_train) < 0.5:
        print(f"   ✅ Cover underrepresented chemical space (test is diverse)")
    else:
        print(f"   ⚠️  Mimic train distribution (test is similar to train)")

    if abs(corr) > 0.3:
        print(f"   ✅ Fill gaps in chemical space (errors correlate with distance)")

    print(f"\n3. Expected OC+FTB performance:")
    synthetic_efficiency = 0.4  # Assume synthetics are 40% as effective as real data

    # If we generate 1200 synthetics, filter to 600, that's like 240 real molecules
    synthetic_count_expected = 600
    effective_real_molecules = synthetic_count_expected * synthetic_efficiency
    synthetic_data_increase = effective_real_molecules / len(train_df)
    expected_improvement = synthetic_data_increase * efficiency
    expected_mae = baseline_mae * (1 - expected_improvement / 100)

    print(f"   Assuming {int(synthetic_count_expected)} synthetics @ {synthetic_efficiency:.0%} effectiveness:")
    print(f"   - Effective real molecules: ~{int(effective_real_molecules)}")
    print(f"   - Expected improvement: ~{expected_improvement:.0f}%")
    print(f"   - Expected MAE: ~{expected_mae:.3f}")

    gap_captured = (baseline_mae - expected_mae) / (baseline_mae - oracle_mae)
    print(f"   - Gap captured: ~{gap_captured:.0%}")

    if gap_captured > 0.5:
        print(f"\n🎉 This would EXCEED the 50% gap target!")
    elif gap_captured > 0.3:
        print(f"\n✅ This would be a strong result (>30% of gap)")
    else:
        print(f"\n⚠️  This would be below target (<30% of gap)")

    # Save analysis results
    analysis_results = {
        'mean_similarity_to_train': float(np.mean(similarities_to_train)),
        'median_similarity_to_train': float(np.median(similarities_to_train)),
        'similarity_error_correlation': float(corr),
        'similarity_error_pvalue': float(p_value),
        'data_increase_pct': float(data_increase_pct),
        'improvement_pct': float(improvement_pct),
        'data_efficiency': float(efficiency),
        'context_performance': {str(k): {'train_n': train_contexts.get(k, 0),
                                         'test_n': test_contexts.get(k, 0),
                                         'mean_error': float(np.mean(test_errors_by_context[k]))}
                               for k in test_errors_by_context.keys()},
        'expected_oc_ftb_mae': float(expected_mae),
        'expected_gap_captured': float(gap_captured)
    }

    with open('memory/esol_oracle_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"\n✅ Saved analysis to memory/esol_oracle_analysis.json")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
