"""
SARExtractor: Extract interpretable Structure-Activity Relationship rules.

Analyzes molecular features and their correlation with property values
to discover interpretable rules like "Fluorine increases solubility".

Features detected:
- Atom presence: has_fluorine, has_chlorine, has_nitrogen, has_sulfur
- Ring features: has_aromatic_ring, has_aliphatic_ring
- Property bins: mw_over_300, logp_over_2, tpsa_over_50
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

logger = logging.getLogger(__name__)


class SARExtractor:
    """
    Extract Structure-Activity Relationship rules from molecular data.

    Uses statistical testing to identify features significantly
    correlated with property values.
    """

    # Feature definitions
    ATOM_FEATURES = {
        'has_fluorine': 9,    # Atomic number for F
        'has_chlorine': 17,   # Cl
        'has_bromine': 35,    # Br
        'has_nitrogen': 7,    # N
        'has_sulfur': 16,     # S
        'has_oxygen': 8,      # O
    }

    def __init__(
        self,
        min_support: int = 5,
        min_effect_size: float = 0.3,
        max_p_value: float = 0.05
    ):
        """
        Initialize SAR extractor.

        Args:
            min_support: Minimum samples with/without feature to consider
            min_effect_size: Minimum absolute effect size (in property units)
            max_p_value: Maximum p-value for significance
        """
        self.min_support = min_support
        self.min_effect_size = min_effect_size
        self.max_p_value = max_p_value
        self._rules: List[Dict] = []

    def extract_rules(
        self,
        smiles_list: List[str],
        labels: List[float]
    ) -> List[Dict]:
        """
        Extract SAR rules from molecular data.

        Args:
            smiles_list: List of SMILES strings
            labels: Corresponding property values

        Returns:
            List of rule dicts sorted by effect size
        """
        labels = np.array(labels)

        # Compute all features
        feature_matrix = self._compute_features(smiles_list)

        # Test each feature
        rules = []

        for feature_name, feature_values in feature_matrix.items():
            rule = self._test_feature(feature_name, feature_values, labels)
            if rule is not None:
                rules.append(rule)

        # Sort by absolute effect size
        rules.sort(key=lambda x: abs(x['effect_size']), reverse=True)

        self._rules = rules
        return rules

    def get_top_rules(self, n: int = 10) -> List[Dict]:
        """Get top N rules by effect size."""
        return self._rules[:n]

    def _compute_features(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """
        Compute binary features for all molecules.

        Returns dict mapping feature_name -> boolean array.
        """
        n = len(smiles_list)
        features = {
            # Atom presence
            'has_fluorine': np.zeros(n, dtype=bool),
            'has_chlorine': np.zeros(n, dtype=bool),
            'has_bromine': np.zeros(n, dtype=bool),
            'has_nitrogen': np.zeros(n, dtype=bool),
            'has_sulfur': np.zeros(n, dtype=bool),
            'has_oxygen': np.zeros(n, dtype=bool),
            # Ring features
            'has_aromatic_ring': np.zeros(n, dtype=bool),
            'has_aliphatic_ring': np.zeros(n, dtype=bool),
            'has_multiple_rings': np.zeros(n, dtype=bool),
            # Property bins
            'mw_over_300': np.zeros(n, dtype=bool),
            'mw_over_400': np.zeros(n, dtype=bool),
            'logp_over_2': np.zeros(n, dtype=bool),
            'logp_over_3': np.zeros(n, dtype=bool),
            'logp_negative': np.zeros(n, dtype=bool),
            'tpsa_over_50': np.zeros(n, dtype=bool),
            'tpsa_over_100': np.zeros(n, dtype=bool),
            'n_rotatable_over_5': np.zeros(n, dtype=bool),
            'n_hbd_over_2': np.zeros(n, dtype=bool),
            'n_hba_over_5': np.zeros(n, dtype=bool),
        }

        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # Atom presence
            atoms = set(atom.GetAtomicNum() for atom in mol.GetAtoms())
            features['has_fluorine'][i] = 9 in atoms
            features['has_chlorine'][i] = 17 in atoms
            features['has_bromine'][i] = 35 in atoms
            features['has_nitrogen'][i] = 7 in atoms
            features['has_sulfur'][i] = 16 in atoms
            features['has_oxygen'][i] = 8 in atoms

            # Ring features
            ring_info = mol.GetRingInfo()
            n_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
            n_aliphatic = rdMolDescriptors.CalcNumAliphaticRings(mol)
            n_total = ring_info.NumRings()

            features['has_aromatic_ring'][i] = n_aromatic > 0
            features['has_aliphatic_ring'][i] = n_aliphatic > 0
            features['has_multiple_rings'][i] = n_total > 1

            # Property bins
            try:
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                n_rotatable = Descriptors.NumRotatableBonds(mol)
                n_hbd = Descriptors.NumHDonors(mol)
                n_hba = Descriptors.NumHAcceptors(mol)

                features['mw_over_300'][i] = mw > 300
                features['mw_over_400'][i] = mw > 400
                features['logp_over_2'][i] = logp > 2
                features['logp_over_3'][i] = logp > 3
                features['logp_negative'][i] = logp < 0
                features['tpsa_over_50'][i] = tpsa > 50
                features['tpsa_over_100'][i] = tpsa > 100
                features['n_rotatable_over_5'][i] = n_rotatable > 5
                features['n_hbd_over_2'][i] = n_hbd > 2
                features['n_hba_over_5'][i] = n_hba > 5
            except Exception:
                pass

        return features

    def _test_feature(
        self,
        feature_name: str,
        feature_values: np.ndarray,
        labels: np.ndarray
    ) -> Optional[Dict]:
        """
        Test if a feature has significant effect on property.

        Returns rule dict if significant, None otherwise.
        """
        # Split by feature presence
        with_feature = labels[feature_values]
        without_feature = labels[~feature_values]

        # Check minimum support
        n_with = len(with_feature)
        n_without = len(without_feature)

        if n_with < self.min_support or n_without < self.min_support:
            return None

        # Compute effect size (difference in means)
        mean_with = np.mean(with_feature)
        mean_without = np.mean(without_feature)
        effect_size = mean_with - mean_without

        # Check minimum effect size
        if abs(effect_size) < self.min_effect_size:
            return None

        # Statistical test (Welch's t-test for unequal variances)
        try:
            t_stat, p_value = stats.ttest_ind(
                with_feature, without_feature,
                equal_var=False
            )
        except Exception:
            return None

        # Check significance
        if p_value > self.max_p_value:
            return None

        # Determine direction
        direction = 'increases' if effect_size > 0 else 'decreases'

        # Generate human-readable rule text
        feature_readable = feature_name.replace('_', ' ').replace('has ', '')
        rule_text = (
            f"{feature_readable.capitalize()} {direction} "
            f"solubility by {abs(effect_size):.2f} logS units"
        )

        return {
            'feature': feature_name,
            'effect_size': float(effect_size),
            'direction': direction,
            'p_value': float(p_value),
            't_statistic': float(t_stat),
            'n_with_feature': n_with,
            'n_without_feature': n_without,
            'mean_with_feature': float(mean_with),
            'mean_without_feature': float(mean_without),
            'rule_text': rule_text
        }

    def get_rule_summary(self) -> str:
        """Get human-readable summary of discovered rules."""
        if not self._rules:
            return "No significant SAR rules discovered."

        lines = ["Discovered SAR Rules:", "-" * 50]

        for i, rule in enumerate(self._rules[:10], 1):
            lines.append(
                f"{i}. {rule['rule_text']} "
                f"(p={rule['p_value']:.3f}, n={rule['n_with_feature']})"
            )

        return "\n".join(lines)
