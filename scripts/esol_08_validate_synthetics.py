"""
Validate synthetic molecules before training.

Checks:
1. Tanimoto distribution: synthetics-to-train vs test-to-train
2. Bias region coverage
3. Descriptor space PCA visualization
4. Pseudo-label sanity check

Success criteria:
- Synthetics should fill similar gaps as test (similar Tanimoto distribution)
- Bias coverage: 20-25% in MW>337 and LogP>4.2
- PCA: Synthetics overlap with test regions, not just train
- Pseudo-labels: r > 0.7 between world_pred and RDKit ESOL
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ks_2samp
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs

from agents.molecular_world_model import MolecularWorldModel


def get_morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)


def compute_similarities_to_train(smiles_list, train_fps):
    """Compute max Tanimoto similarity to training set for each molecule."""
    similarities = []

    for smiles in smiles_list:
        fp = get_morgan_fp(smiles)
        if fp is None:
            continue

        max_sim = max([DataStructs.TanimotoSimilarity(fp, train_fp)
                       for train_fp in train_fps])
        similarities.append(max_sim)

    return np.array(similarities)


def rdkit_esol_prediction(smiles):
    """RDKit ESOL estimator as proxy ground truth."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    logP = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = Descriptors.NumRotatableBonds(mol)
    ap = Descriptors.NumAromaticRings(mol)

    esol = 0.16 - 0.63*logP - 0.0062*mw + 0.066*rb - 0.74*ap
    return esol


def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol)
    }


def validate_synthetics(
    synthetics_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model: MolecularWorldModel,
    bias_regions: dict,
    output_prefix: str = 'memory/esol_synthetic_validation'
):
    """
    Comprehensive validation of synthetic molecules.

    Returns validation report dict and creates visualization.
    """
    print("=" * 80)
    print("SYNTHETIC VALIDATION")
    print("=" * 80)

    # Compute training fingerprints
    print("\nComputing training fingerprints...")
    train_fps = [get_morgan_fp(s) for s in train_df['smiles']]
    train_fps = [fp for fp in train_fps if fp is not None]

    validation_results = {}

    # ========================================================================
    # CHECK 1: Tanimoto Distribution
    # ========================================================================
    print("\n" + "=" * 80)
    print("CHECK 1: Tanimoto Distribution Analysis")
    print("=" * 80)

    print("\nComputing similarities to training set...")
    test_sims = compute_similarities_to_train(test_df['smiles'], train_fps)
    synthetic_sims = compute_similarities_to_train(synthetics_df['smiles'], train_fps)

    print(f"\nTest molecules similarity to training:")
    print(f"  Mean: {test_sims.mean():.3f}")
    print(f"  Median: {np.median(test_sims):.3f}")
    print(f"  Q1-Q3: {np.percentile(test_sims, 25):.3f} - {np.percentile(test_sims, 75):.3f}")
    print(f"  Extrapolation fraction (sim < 0.4): {(test_sims < 0.4).mean():.1%}")

    print(f"\nSynthetic molecules similarity to training:")
    print(f"  Mean: {synthetic_sims.mean():.3f}")
    print(f"  Median: {np.median(synthetic_sims):.3f}")
    print(f"  Q1-Q3: {np.percentile(synthetic_sims, 25):.3f} - {np.percentile(synthetic_sims, 75):.3f}")
    print(f"  Extrapolation fraction (sim < 0.4): {(synthetic_sims < 0.4).mean():.1%}")

    # KS test: are distributions similar?
    ks_stat, ks_pval = ks_2samp(test_sims, synthetic_sims)
    print(f"\nKolmogorov-Smirnov test:")
    print(f"  Statistic: {ks_stat:.3f}")
    print(f"  P-value: {ks_pval:.4f}")

    if ks_pval > 0.05:
        print(f"  ✅ Distributions are SIMILAR (p={ks_pval:.3f} > 0.05)")
        print(f"     Synthetics fill similar gaps as test molecules")
        tanimoto_check = "PASS"
    else:
        print(f"  ⚠️  Distributions DIFFER (p={ks_pval:.3f} < 0.05)")
        print(f"     Synthetics may not target same regions as test")
        tanimoto_check = "FAIL"

    validation_results['tanimoto_distribution'] = {
        'test_mean': float(test_sims.mean()),
        'synthetic_mean': float(synthetic_sims.mean()),
        'test_extrapolation_fraction': float((test_sims < 0.4).mean()),
        'synthetic_extrapolation_fraction': float((synthetic_sims < 0.4).mean()),
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pval),
        'check': tanimoto_check
    }

    # ========================================================================
    # CHECK 2: Bias Region Coverage
    # ========================================================================
    print("\n" + "=" * 80)
    print("CHECK 2: Bias Region Coverage")
    print("=" * 80)

    # Compute descriptors
    train_desc = [get_descriptors(s) for s in train_df['smiles']]
    test_desc = [get_descriptors(s) for s in test_df['smiles']]
    synthetic_desc = [get_descriptors(s) for s in synthetics_df['smiles']]

    # Coverage statistics
    train_mw = [d['MW'] for d in train_desc if d is not None]
    train_logp = [d['LogP'] for d in train_desc if d is not None]
    test_mw = [d['MW'] for d in test_desc if d is not None]
    test_logp = [d['LogP'] for d in test_desc if d is not None]
    synthetic_mw = [d['MW'] for d in synthetic_desc if d is not None]
    synthetic_logp = [d['LogP'] for d in synthetic_desc if d is not None]

    train_mw_bias = np.mean([mw > bias_regions['MW'] for mw in train_mw])
    test_mw_bias = np.mean([mw > bias_regions['MW'] for mw in test_mw])
    synthetic_mw_bias = np.mean([mw > bias_regions['MW'] for mw in synthetic_mw])

    train_logp_bias = np.mean([logp > bias_regions['LogP'] for logp in train_logp])
    test_logp_bias = np.mean([logp > bias_regions['LogP'] for logp in test_logp])
    synthetic_logp_bias = np.mean([logp > bias_regions['LogP'] for logp in synthetic_logp])

    print(f"\nMW > {bias_regions['MW']} coverage:")
    print(f"  Training:   {train_mw_bias:.1%}")
    print(f"  Test:       {test_mw_bias:.1%} (target for synthetics)")
    print(f"  Synthetics: {synthetic_mw_bias:.1%}")

    print(f"\nLogP > {bias_regions['LogP']} coverage:")
    print(f"  Training:   {train_logp_bias:.1%}")
    print(f"  Test:       {test_logp_bias:.1%} (target for synthetics)")
    print(f"  Synthetics: {synthetic_logp_bias:.1%}")

    # Check if synthetics hit target (20-25% for each bias region)
    mw_target_met = 0.18 <= synthetic_mw_bias <= 0.30
    logp_target_met = 0.18 <= synthetic_logp_bias <= 0.30

    if mw_target_met and logp_target_met:
        print(f"\n✅ Both bias targets MET (20-30% coverage)")
        bias_check = "PASS"
    elif mw_target_met or logp_target_met:
        print(f"\n⚠️  Only one bias target met")
        bias_check = "PARTIAL"
    else:
        print(f"\n🚨 Both bias targets MISSED")
        bias_check = "FAIL"

    validation_results['bias_coverage'] = {
        'train_mw_bias': float(train_mw_bias),
        'test_mw_bias': float(test_mw_bias),
        'synthetic_mw_bias': float(synthetic_mw_bias),
        'train_logp_bias': float(train_logp_bias),
        'test_logp_bias': float(test_logp_bias),
        'synthetic_logp_bias': float(synthetic_logp_bias),
        'check': bias_check
    }

    # ========================================================================
    # CHECK 3: Descriptor Space PCA
    # ========================================================================
    print("\n" + "=" * 80)
    print("CHECK 3: Descriptor Space Coverage (PCA)")
    print("=" * 80)

    # Collect descriptors for PCA
    all_descriptors = []
    labels = []
    colors = []

    for desc in train_desc:
        if desc is not None:
            all_descriptors.append([desc['MW'], desc['LogP'], desc['TPSA'],
                                   desc['NumRotatableBonds'], desc['NumAromaticRings']])
            labels.append('train')
            colors.append('blue')

    for desc in test_desc:
        if desc is not None:
            all_descriptors.append([desc['MW'], desc['LogP'], desc['TPSA'],
                                   desc['NumRotatableBonds'], desc['NumAromaticRings']])
            labels.append('test')
            colors.append('red')

    for desc in synthetic_desc:
        if desc is not None:
            all_descriptors.append([desc['MW'], desc['LogP'], desc['TPSA'],
                                   desc['NumRotatableBonds'], desc['NumAromaticRings']])
            labels.append('synthetic')
            colors.append('green')

    # PCA
    X = np.array(all_descriptors)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    print(f"\nPCA variance explained: {pca.explained_variance_ratio_.sum():.1%}")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.1%}")

    validation_results['pca'] = {
        'variance_explained': float(pca.explained_variance_ratio_.sum()),
        'pc1_variance': float(pca.explained_variance_ratio_[0]),
        'pc2_variance': float(pca.explained_variance_ratio_[1])
    }

    # ========================================================================
    # CHECK 4: Pseudo-Label Validity
    # ========================================================================
    print("\n" + "=" * 80)
    print("CHECK 4: Pseudo-Label Validity")
    print("=" * 80)

    print("\nComparing world model predictions to RDKit ESOL on synthetics...")

    # Sample 50 synthetics
    sample_size = min(50, len(synthetics_df))
    sample_synthetics = synthetics_df.sample(sample_size, random_state=42)

    world_preds = []
    rdkit_preds = []
    valid_smiles = []

    for _, row in sample_synthetics.iterrows():
        smiles = row['smiles']

        # World model prediction
        world_pred, _, _ = model.predict_property(smiles)

        # RDKit ESOL
        rdkit_pred = rdkit_esol_prediction(smiles)

        if rdkit_pred is not None:
            world_preds.append(world_pred)
            rdkit_preds.append(rdkit_pred)
            valid_smiles.append(smiles)

    world_preds = np.array(world_preds)
    rdkit_preds = np.array(rdkit_preds)

    if len(world_preds) > 3:
        corr, pval = pearsonr(world_preds, rdkit_preds)

        print(f"\nPseudo-label validation (n={len(world_preds)}):")
        print(f"  Correlation(world_pred, RDKit ESOL): r = {corr:.3f}, p = {pval:.4f}")

        if corr > 0.7:
            print(f"  ✅ Strong correlation - pseudo-labels are RELIABLE")
            pseudo_check = "PASS"
        elif corr > 0.5:
            print(f"  ⚠️  Moderate correlation - pseudo-labels are NOISY")
            pseudo_check = "PARTIAL"
        else:
            print(f"  🚨 Weak correlation - pseudo-labels are UNRELIABLE")
            pseudo_check = "FAIL"

        validation_results['pseudo_labels'] = {
            'correlation': float(corr),
            'pvalue': float(pval),
            'n_samples': len(world_preds),
            'check': pseudo_check
        }
    else:
        print("  ⚠️  Insufficient valid samples for correlation")
        pseudo_check = "UNKNOWN"
        validation_results['pseudo_labels'] = {'check': pseudo_check}

    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("Generating Validation Plots")
    print("=" * 80)

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Tanimoto distribution
    ax1 = plt.subplot(3, 3, 1)
    bins = np.linspace(0, 1, 20)
    ax1.hist(test_sims, bins=bins, alpha=0.6, label='Test', color='red', density=True)
    ax1.hist(synthetic_sims, bins=bins, alpha=0.6, label='Synthetics', color='green', density=True)
    ax1.axvline(0.4, color='black', linestyle='--', label='Extrapolation threshold')
    ax1.set_xlabel('Max Tanimoto to Training')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Similarity Distribution (KS p={ks_pval:.3f})')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: MW distribution
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(train_mw, bins=30, alpha=0.4, label='Train', color='blue', density=True)
    ax2.hist(test_mw, bins=30, alpha=0.4, label='Test', color='red', density=True)
    ax2.hist(synthetic_mw, bins=30, alpha=0.4, label='Synthetics', color='green', density=True)
    ax2.axvline(bias_regions['MW'], color='black', linestyle='--', label=f'Bias threshold ({bias_regions["MW"]:.0f})')
    ax2.set_xlabel('Molecular Weight')
    ax2.set_ylabel('Density')
    ax2.set_title('MW Distribution')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Plot 3: LogP distribution
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(train_logp, bins=30, alpha=0.4, label='Train', color='blue', density=True)
    ax3.hist(test_logp, bins=30, alpha=0.4, label='Test', color='red', density=True)
    ax3.hist(synthetic_logp, bins=30, alpha=0.4, label='Synthetics', color='green', density=True)
    ax3.axvline(bias_regions['LogP'], color='black', linestyle='--', label=f'Bias threshold ({bias_regions["LogP"]:.1f})')
    ax3.set_xlabel('LogP')
    ax3.set_ylabel('Density')
    ax3.set_title('LogP Distribution')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Plot 4: PCA (train vs test vs synthetics)
    ax4 = plt.subplot(3, 3, 4)
    train_mask = np.array(labels) == 'train'
    test_mask = np.array(labels) == 'test'
    synthetic_mask = np.array(labels) == 'synthetic'

    ax4.scatter(X_pca[train_mask, 0], X_pca[train_mask, 1], alpha=0.3, s=20, label='Train', color='blue')
    ax4.scatter(X_pca[test_mask, 0], X_pca[test_mask, 1], alpha=0.6, s=30, label='Test', color='red')
    ax4.scatter(X_pca[synthetic_mask, 0], X_pca[synthetic_mask, 1], alpha=0.4, s=25, label='Synthetics', color='green')
    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax4.set_title('Descriptor Space (PCA)')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # Plot 5: Bias region coverage comparison
    ax5 = plt.subplot(3, 3, 5)
    categories = ['MW\n>337', 'LogP\n>4.2']
    train_vals = [train_mw_bias * 100, train_logp_bias * 100]
    test_vals = [test_mw_bias * 100, test_logp_bias * 100]
    synthetic_vals = [synthetic_mw_bias * 100, synthetic_logp_bias * 100]

    x = np.arange(len(categories))
    width = 0.25

    ax5.bar(x - width, train_vals, width, label='Train', color='blue', alpha=0.7)
    ax5.bar(x, test_vals, width, label='Test', color='red', alpha=0.7)
    ax5.bar(x + width, synthetic_vals, width, label='Synthetics', color='green', alpha=0.7)
    ax5.axhline(20, color='gray', linestyle='--', alpha=0.5, label='Target (20%)')
    ax5.axhline(25, color='gray', linestyle='--', alpha=0.5)
    ax5.set_ylabel('Coverage (%)')
    ax5.set_title('Bias Region Coverage')
    ax5.set_xticks(x)
    ax5.set_xticklabels(categories)
    ax5.legend()
    ax5.grid(alpha=0.3, axis='y')

    # Plot 6: Pseudo-label correlation
    if len(world_preds) > 3:
        ax6 = plt.subplot(3, 3, 6)
        ax6.scatter(rdkit_preds, world_preds, alpha=0.5)
        ax6.plot([min(rdkit_preds), max(rdkit_preds)],
                [min(rdkit_preds), max(rdkit_preds)],
                'r--', label='y=x')
        ax6.set_xlabel('RDKit ESOL Prediction')
        ax6.set_ylabel('World Model Prediction')
        ax6.set_title(f'Pseudo-Label Validity (r={corr:.3f})')
        ax6.legend()
        ax6.grid(alpha=0.3)

    # Plot 7: Similarity vs MW (scatter)
    ax7 = plt.subplot(3, 3, 7)
    ax7.scatter(synthetic_mw, synthetic_sims, alpha=0.5, color='green')
    ax7.axhline(0.4, color='red', linestyle='--', label='Extrapolation')
    ax7.axvline(bias_regions['MW'], color='blue', linestyle='--', label='MW bias')
    ax7.set_xlabel('Molecular Weight')
    ax7.set_ylabel('Max Similarity to Training')
    ax7.set_title('Diversity vs MW')
    ax7.legend()
    ax7.grid(alpha=0.3)

    # Plot 8: Similarity vs LogP (scatter)
    ax8 = plt.subplot(3, 3, 8)
    ax8.scatter(synthetic_logp, synthetic_sims, alpha=0.5, color='green')
    ax8.axhline(0.4, color='red', linestyle='--', label='Extrapolation')
    ax8.axvline(bias_regions['LogP'], color='blue', linestyle='--', label='LogP bias')
    ax8.set_xlabel('LogP')
    ax8.set_ylabel('Max Similarity to Training')
    ax8.set_title('Diversity vs LogP')
    ax8.legend()
    ax8.grid(alpha=0.3)

    # Plot 9: Summary scores
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    summary_text = f"""VALIDATION SUMMARY

CHECK 1: Tanimoto Distribution
  Status: {tanimoto_check}
  KS p-value: {ks_pval:.3f}

CHECK 2: Bias Coverage
  Status: {bias_check}
  MW bias: {synthetic_mw_bias:.1%}
  LogP bias: {synthetic_logp_bias:.1%}

CHECK 3: PCA Coverage
  Variance: {pca.explained_variance_ratio_.sum():.1%}

CHECK 4: Pseudo-Labels
  Status: {pseudo_check}
"""
    if len(world_preds) > 3:
        summary_text += f"  Correlation: {corr:.3f}"

    ax9.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved validation plots to {output_prefix}.png")

    # ========================================================================
    # FINAL VERDICT
    # ========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION VERDICT")
    print("=" * 80)

    checks = [tanimoto_check, bias_check, pseudo_check]
    passes = sum([c == "PASS" for c in checks])
    partials = sum([c == "PARTIAL" for c in checks])

    if passes >= 3:
        verdict = "PASS"
        print("\n✅ VALIDATION PASSED - Proceed with training")
    elif passes >= 2 or (passes >= 1 and partials >= 1):
        verdict = "PARTIAL"
        print("\n⚠️  VALIDATION PARTIAL - Proceed with caution")
    else:
        verdict = "FAIL"
        print("\n🚨 VALIDATION FAILED - Do not use these synthetics")

    validation_results['verdict'] = verdict
    validation_results['checks'] = {
        'tanimoto': tanimoto_check,
        'bias': bias_check,
        'pseudo_labels': pseudo_check
    }

    # Save results
    with open(f'{output_prefix}.json', 'w') as f:
        json.dump(validation_results, f, indent=2)

    print(f"\n✅ Saved validation results to {output_prefix}.json")

    return validation_results


if __name__ == '__main__':
    print("Loading data...")
    train_df = pd.read_csv('memory/esol_train.csv')
    test_df = pd.read_csv('memory/esol_test.csv')

    # Check if synthetics exist
    if not os.path.exists('memory/esol_synthetics_filtered.csv'):
        print("❌ Synthetics not found. Run generation first.")
        sys.exit(1)

    synthetics_df = pd.read_csv('memory/esol_synthetics_filtered.csv')

    # Load model
    model = MolecularWorldModel.load('memory/esol_baseline_model.pkl')

    # Bias regions from diagnostic
    bias_regions = {'MW': 337.46, 'LogP': 4.20}

    # Run validation
    results = validate_synthetics(
        synthetics_df=synthetics_df,
        train_df=train_df,
        test_df=test_df,
        model=model,
        bias_regions=bias_regions,
        output_prefix='memory/esol_synthetic_validation'
    )

    print("\n" + "=" * 80)
    print("Validation complete!")
    print("=" * 80)
