"""
Non-circular fidelity scoring for molecular synthetics.

Combines three independent components:
1. Ensemble agreement: Do world model and external model agree?
2. kNN plausibility: Is molecule similar to training data?
3. Descriptor consistency: Are molecular properties reasonable?

This breaks circularity by using kNN and descriptor checks that don't depend
on the world model's predictions of the synthetic's label.
"""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from typing import List, Dict, Tuple
import pandas as pd
import sys
import os


class MolecularFidelityScorer:
    """
    Non-circular fidelity scoring for synthetic molecules.

    Fidelity measures trustworthiness of synthetic data without creating
    circular dependencies on world model predictions.
    """

    def __init__(self, training_smiles: List[str], descriptor_stats: Dict):
        """
        Initialize fidelity scorer with training data.

        Args:
            training_smiles: List of SMILES from training set
            descriptor_stats: Dict with mean/std for descriptors
                            (from esol_descriptor_stats.json)
        """
        self.training_smiles = training_smiles
        self.descriptor_stats = descriptor_stats

        # Precompute training fingerprints for kNN
        print("Precomputing training fingerprints for kNN...")
        self.training_fps = []
        for smiles in training_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                self.training_fps.append(fp)
        print(f"✅ Precomputed {len(self.training_fps)} training fingerprints")

    def score_synthetic(self,
                       synthetic_smiles: str,
                       parent_smiles: str,
                       world_model_pred: float,
                       external_model_pred: float) -> Tuple[float, Dict]:
        """
        Score synthetic molecule with non-circular fidelity.

        Args:
            synthetic_smiles: SMILES of synthetic molecule
            parent_smiles: SMILES of parent molecule (not used currently)
            world_model_pred: World model's prediction for synthetic
            external_model_pred: External model's prediction for synthetic

        Returns:
            fidelity: Overall fidelity score [0, 1], higher = more trustworthy
            components: Dict with breakdown of score components
        """

        # Component 1: Ensemble agreement (do models agree?)
        ensemble_score = self._ensemble_agreement(world_model_pred, external_model_pred)

        # Component 2: kNN chemical plausibility (is molecule similar to training?)
        knn_score = self._knn_plausibility(synthetic_smiles)

        # Component 3: Descriptor consistency (are properties reasonable?)
        descriptor_score = self._descriptor_consistency(synthetic_smiles)

        # Combined fidelity with weighted average
        # Weights: ensemble=40%, kNN=40%, descriptors=20%
        fidelity = (0.4 * ensemble_score +
                   0.4 * knn_score +
                   0.2 * descriptor_score)

        components = {
            'ensemble_agreement': ensemble_score,
            'knn_plausibility': knn_score,
            'descriptor_consistency': descriptor_score,
            'fidelity': fidelity
        }

        return fidelity, components

    def _ensemble_agreement(self, pred1: float, pred2: float) -> float:
        """
        Score based on agreement between two models.

        High agreement → high fidelity (models converge on same answer)
        Low agreement → low fidelity (models disagree, uncertainty)

        Args:
            pred1: First model's prediction
            pred2: Second model's prediction

        Returns:
            Agreement score [0, 1], where 1 = perfect agreement
        """
        disagreement = abs(pred1 - pred2)
        # Convert to [0, 1] score using exponential decay
        # disagreement=0 → 1.0, disagreement=1 → 0.37, disagreement=2 → 0.14
        agreement = np.exp(-disagreement)
        return agreement

    def _knn_plausibility(self, synthetic_smiles: str, k: int = 5) -> float:
        """
        Score based on similarity to k nearest training molecules.

        This breaks circularity - asks "is this chemically plausible?"
        independent of model predictions.

        High similarity → high fidelity (molecule is in known chemical space)
        Low similarity → low fidelity (molecule is unusual/implausible)

        Args:
            synthetic_smiles: SMILES of synthetic molecule
            k: Number of nearest neighbors to consider

        Returns:
            Plausibility score [0, 1], where 1 = very similar to training
        """
        mol = Chem.MolFromSmiles(synthetic_smiles)
        if mol is None:
            return 0.0

        synthetic_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

        # Compute Tanimoto similarity to all training molecules
        similarities = []
        for train_fp in self.training_fps:
            sim = DataStructs.TanimotoSimilarity(synthetic_fp, train_fp)
            similarities.append(sim)

        if len(similarities) == 0:
            return 0.0

        # Average similarity to k nearest neighbors
        k_nearest = sorted(similarities, reverse=True)[:k]
        avg_similarity = np.mean(k_nearest)
        max_similarity = k_nearest[0]

        # Penalize if even closest neighbor is far
        # If max_sim < 0.4, molecule is very different from training
        if max_similarity < 0.4:
            plausibility = avg_similarity * 0.5  # Harsh penalty
        else:
            plausibility = avg_similarity

        return plausibility

    def _descriptor_consistency(self, synthetic_smiles: str) -> float:
        """
        Check if descriptors are reasonable compared to training distribution.

        Catches molecules with weird properties (e.g., MW=1000, LogP=15).
        Uses z-score to measure how far descriptors are from training mean.

        Args:
            synthetic_smiles: SMILES of synthetic molecule

        Returns:
            Consistency score [0, 1], where 1 = very consistent with training
        """
        mol = Chem.MolFromSmiles(synthetic_smiles)
        if mol is None:
            return 0.0

        # Compute key descriptors
        # Note: Use keys that match esol_descriptor_stats.json
        descriptors = {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol)
        }

        # Check each descriptor against training distribution
        z_scores = []
        for desc_name, desc_value in descriptors.items():
            if desc_name in self.descriptor_stats:
                mean = self.descriptor_stats[desc_name]['mean']
                std = self.descriptor_stats[desc_name]['std']

                if std > 0:
                    z_score = abs(desc_value - mean) / std
                    z_scores.append(z_score)

        if len(z_scores) == 0:
            return 0.5  # Neutral if can't compute

        # Average z-score across descriptors
        avg_z = np.mean(z_scores)

        # Convert to [0, 1] score where 1 = very consistent
        # z=0 → score=1.0, z=2 → score=0.37, z=3 → score=0.22, z=5 → score=0.08
        consistency = np.exp(-avg_z / 2)

        return consistency


def test_fidelity_scorer():
    """Test the fidelity scorer on example molecules."""

    print("=" * 60)
    print("Fidelity Scorer Unit Test")
    print("=" * 60)

    # Load training data
    train_df = pd.read_csv('memory/esol_train.csv')

    import json
    with open('memory/esol_descriptor_stats.json', 'r') as f:
        descriptor_stats = json.load(f)

    print(f"\nLoaded {len(train_df)} training molecules")
    print(f"Descriptor stats: {list(descriptor_stats.keys())}")

    # Initialize scorer
    scorer = MolecularFidelityScorer(
        training_smiles=train_df['smiles'].tolist(),
        descriptor_stats=descriptor_stats
    )

    # Test cases
    test_cases = [
        {
            'name': 'High fidelity (close to training, models agree)',
            'smiles': 'c1ccccc1',  # Benzene (common aromatic)
            'world_pred': -2.5,
            'external_pred': -2.6
        },
        {
            'name': 'Moderate fidelity (uncommon scaffold, slight disagreement)',
            'smiles': 'C1=CC=C2C(=C1)C=CN=C2C(=O)O',  # Quinoline derivative
            'world_pred': -1.0,
            'external_pred': -1.3
        },
        {
            'name': 'Low fidelity (high disagreement)',
            'smiles': 'CCO',  # Ethanol
            'world_pred': -1.0,
            'external_pred': -2.5
        },
        {
            'name': 'Very low fidelity (unusual molecule, far from training)',
            'smiles': 'CC(C)(C)C(C)(C)C(C)(C)C(C)(C)C',  # Highly branched aliphatic
            'world_pred': -5.0,
            'external_pred': -6.0
        }
    ]

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)

    for test in test_cases:
        fidelity, components = scorer.score_synthetic(
            test['smiles'],
            parent_smiles=None,
            world_model_pred=test['world_pred'],
            external_model_pred=test['external_pred']
        )

        print(f"\n{test['name']}")
        print(f"  SMILES: {test['smiles']}")
        print(f"  Predictions: WM={test['world_pred']:.1f}, Ext={test['external_pred']:.1f}")
        print(f"  Fidelity: {fidelity:.3f}")
        print(f"  Components:")
        for k, v in components.items():
            if k != 'fidelity':
                print(f"    {k:25s}: {v:.3f}")

    print("\n" + "=" * 60)
    print("✅ Fidelity scorer test complete!")
    print("=" * 60)

    print("\nInterpretation guide:")
    print("  Fidelity > 0.7: High confidence - use synthetic")
    print("  Fidelity 0.5-0.7: Moderate - use with caution")
    print("  Fidelity < 0.5: Low confidence - filter out")


if __name__ == "__main__":
    test_fidelity_scorer()
