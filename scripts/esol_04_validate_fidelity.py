"""
Validate fidelity scoring on small test set.
Compare circular vs non-circular approaches.

This generates molecular analogs and scores them with:
- v1 (circular): ensemble agreement + parent similarity
- v2 (non-circular): ensemble + kNN + descriptors

Goal: Show that v2 catches failure modes that v1 misses.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fidelity_scoring import MolecularFidelityScorer
from utils.analog_generator import generate_molecular_analogs_v0
from agents.molecular_world_model import MolecularWorldModel
from sklearn.ensemble import RandomForestRegressor


def validate_fidelity_scoring():
    """
    Generate small set of synthetics, score with v1 and v2,
    check if v2 catches different failure modes.
    """

    print("=" * 60)
    print("Validating Fidelity Scoring")
    print("=" * 60)

    # Load data
    train_df = pd.read_csv('memory/esol_train.csv')
    test_df = pd.read_csv('memory/esol_test.csv')

    with open('memory/esol_descriptor_stats.json', 'r') as f:
        descriptor_stats = json.load(f)

    print(f"\nLoaded {len(train_df)} train, {len(test_df)} test molecules")

    # Load trained world model
    world_model = MolecularWorldModel.load('memory/esol_baseline_model.pkl')

    # Train simple external model (global RandomForest)
    from rdkit.Chem import AllChem
    print("Training external RandomForest model...")

    train_fps = []
    for smiles in train_df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            train_fps.append(list(fp))

    external_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    external_model.fit(train_fps, train_df['solubility'].values)
    print("✅ External model trained")

    # Initialize fidelity scorer
    scorer = MolecularFidelityScorer(
        training_smiles=train_df['smiles'].tolist(),
        descriptor_stats=descriptor_stats
    )

    # Generate analogs from 50 random training molecules
    print("\nGenerating analogs from 50 training molecules...")
    np.random.seed(42)
    sample_molecules = train_df.sample(min(50, len(train_df)))['smiles'].tolist()

    results = []
    successful_parents = 0

    for i, parent_smiles in enumerate(sample_molecules):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/50 parents...")

        # Generate 2 analogs
        analogs = generate_molecular_analogs_v0(
            parent_smiles,
            n_analogs=2,
            tanimoto_range=(0.6, 0.85)
        )

        if len(analogs) > 0:
            successful_parents += 1

        for analog_smiles, transform, reason in analogs:
            # Get predictions from both models
            world_pred, world_unc, world_ctx = world_model.predict_property(analog_smiles)

            mol = Chem.MolFromSmiles(analog_smiles)
            if mol is None:
                continue

            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            external_pred = external_model.predict([list(fp)])[0]

            # Score with v2 (non-circular)
            fidelity, components = scorer.score_synthetic(
                analog_smiles,
                parent_smiles,
                world_pred,
                external_pred
            )

            # V1 score (circular - just ensemble agreement + similarity to parent)
            from rdkit import DataStructs
            parent_mol = Chem.MolFromSmiles(parent_smiles)
            parent_fp = AllChem.GetMorganFingerprintAsBitVect(parent_mol, 2, nBits=2048)
            analog_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            similarity = DataStructs.TanimotoSimilarity(parent_fp, analog_fp)

            agreement = np.exp(-abs(world_pred - external_pred))
            fidelity_v1 = 0.7 * agreement + 0.3 * similarity

            results.append({
                'parent': parent_smiles,
                'analog': analog_smiles,
                'transform': transform,
                'world_pred': world_pred,
                'external_pred': external_pred,
                'disagreement': abs(world_pred - external_pred),
                'parent_similarity': similarity,
                'fidelity_v1': fidelity_v1,
                'fidelity_v2': fidelity,
                'ensemble_agreement': components['ensemble_agreement'],
                'knn_plausibility': components['knn_plausibility'],
                'descriptor_consistency': components['descriptor_consistency']
            })

    results_df = pd.DataFrame(results)

    print(f"\n✅ Generated {len(results_df)} analogs from {successful_parents} parents")
    print(f"Mean fidelity v1 (circular): {results_df['fidelity_v1'].mean():.3f}")
    print(f"Mean fidelity v2 (non-circular): {results_df['fidelity_v2'].mean():.3f}")

    # Analyze differences
    results_df['fidelity_diff'] = results_df['fidelity_v2'] - results_df['fidelity_v1']

    print("\n" + "=" * 60)
    print("FAILURE MODE ANALYSIS")
    print("=" * 60)

    print("\n=== Molecules where v2 scores MUCH LOWER (v2 catches issues) ===")
    low_v2 = results_df.nsmallest(min(5, len(results_df)), 'fidelity_diff')
    for idx, row in low_v2.iterrows():
        print(f"\nAnalog: {row['analog']}")
        print(f"  Fidelity: v1={row['fidelity_v1']:.3f}, v2={row['fidelity_v2']:.3f} (diff={row['fidelity_diff']:.3f})")
        print(f"  Why v2 is lower:")
        print(f"    kNN plausibility: {row['knn_plausibility']:.3f} (is molecule chemically unusual?)")
        print(f"    Descriptor consistency: {row['descriptor_consistency']:.3f} (weird properties?)")
        print(f"  v1 components:")
        print(f"    Parent similarity: {row['parent_similarity']:.3f}")
        print(f"    Ensemble agreement: {row['ensemble_agreement']:.3f}")

    print("\n=== Molecules where v2 scores MUCH HIGHER (v1 too harsh) ===")
    high_v2 = results_df.nlargest(min(5, len(results_df)), 'fidelity_diff')
    for idx, row in high_v2.iterrows():
        print(f"\nAnalog: {row['analog']}")
        print(f"  Fidelity: v1={row['fidelity_v1']:.3f}, v2={row['fidelity_v2']:.3f} (diff={row['fidelity_diff']:.3f})")
        print(f"  Why v2 is higher:")
        print(f"    kNN plausibility: {row['knn_plausibility']:.3f}")
        print(f"    Descriptor consistency: {row['descriptor_consistency']:.3f}")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Scatter: v1 vs v2
    axes[0, 0].scatter(results_df['fidelity_v1'], results_df['fidelity_v2'], alpha=0.5)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', label='y=x')
    axes[0, 0].set_xlabel('Fidelity v1 (circular)')
    axes[0, 0].set_ylabel('Fidelity v2 (non-circular)')
    axes[0, 0].legend()
    axes[0, 0].set_title('v1 vs v2 Fidelity Scores')
    axes[0, 0].grid(True, alpha=0.3)

    # Distribution of components
    components_data = results_df[['ensemble_agreement', 'knn_plausibility', 'descriptor_consistency']]
    components_data.boxplot(ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of Fidelity Components')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].grid(True, alpha=0.3)

    # kNN vs ensemble
    axes[1, 0].scatter(results_df['ensemble_agreement'],
                      results_df['knn_plausibility'], alpha=0.5)
    axes[1, 0].set_xlabel('Ensemble Agreement')
    axes[1, 0].set_ylabel('kNN Plausibility')
    axes[1, 0].set_title('Independence Check: Ensemble vs kNN')
    axes[1, 0].grid(True, alpha=0.3)

    # Histogram of fidelity difference
    axes[1, 1].hist(results_df['fidelity_diff'], bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(0, color='r', linestyle='--', linewidth=2, label='No difference')
    axes[1, 1].set_xlabel('Fidelity v2 - v1')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend()
    axes[1, 1].set_title('Difference in Fidelity Scores')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('memory/fidelity_validation.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved visualization to memory/fidelity_validation.png")

    # Save results
    results_df.to_csv('memory/fidelity_validation_results.csv', index=False)
    print("✅ Saved results to memory/fidelity_validation_results.csv")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Correlation(v1, v2): r={results_df['fidelity_v1'].corr(results_df['fidelity_v2']):.3f}")
    print(f"Correlation(ensemble, kNN): r={results_df['ensemble_agreement'].corr(results_df['knn_plausibility']):.3f}")
    print(f"  → {'Low correlation = components are independent ✅' if abs(results_df['ensemble_agreement'].corr(results_df['knn_plausibility'])) < 0.3 else 'High correlation = some redundancy ⚠️'}")

    print(f"\n% analogs with fidelity_v2 > 0.5: {(results_df['fidelity_v2'] > 0.5).mean() * 100:.1f}%")
    print(f"% analogs with fidelity_v2 > 0.6: {(results_df['fidelity_v2'] > 0.6).mean() * 100:.1f}%")
    print(f"% analogs with fidelity_v2 > 0.7: {(results_df['fidelity_v2'] > 0.7).mean() * 100:.1f}%")

    # Filtering recommendation
    if (results_df['fidelity_v2'] > 0.6).mean() > 0.5:
        print(f"\n✅ Recommendation: Use threshold 0.6 (keeps {(results_df['fidelity_v2'] > 0.6).mean() * 100:.0f}% of analogs)")
    elif (results_df['fidelity_v2'] > 0.5).mean() > 0.5:
        print(f"\n⚠️  Recommendation: Use threshold 0.5 (keeps {(results_df['fidelity_v2'] > 0.5).mean() * 100:.0f}% of analogs)")
    else:
        print(f"\n⚠️  Warning: Most analogs have low fidelity. Consider improving analog generation")

    print("\n" + "=" * 60)
    print("✅ Fidelity validation complete!")
    print("=" * 60)


if __name__ == "__main__":
    validate_fidelity_scoring()
