"""
SemanticMemory: Persistent storage of validated SAR rules with Bayesian updating.

This module provides a queryable symbolic knowledge base that runs alongside
the neural world model. Rules are accumulated over time, never deleted, and
updated via Bayesian combination when new evidence arrives.

Key Features:
- Bayesian updating of effect sizes and confidence
- Provenance tracking for all rules
- Molecule-to-rules matching via RDKit feature extraction
- Symbolic-only prediction capability
- JSON serialization for persistence
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
from scipy import stats

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

logger = logging.getLogger(__name__)


class SemanticMemory:
    """
    Persistent storage of validated SAR rules with Bayesian updating.

    Rules are stored with full provenance and updated via weighted
    averaging when new evidence for the same feature arrives.
    """

    def __init__(
        self,
        min_p_value: float = 0.05,
        min_confidence: float = 0.7,
        min_effect_size: float = 0.2
    ):
        """
        Initialize semantic memory.

        Args:
            min_p_value: Maximum p-value for rule acceptance
            min_confidence: Minimum confidence for rule acceptance
            min_effect_size: Minimum absolute effect size
        """
        self.min_p_value = min_p_value
        self.min_confidence = min_confidence
        self.min_effect_size = min_effect_size

        # Rule storage: feature -> rule dict
        self._rules: Dict[str, Dict] = {}

        # History for analysis
        self._ingestion_history: List[Dict] = []

        # Global statistics for baseline prediction
        # The memory must know the "average" value of the world
        self.global_mean: float = 0.0
        self.global_sum: float = 0.0
        self.n_observed: int = 0

    def update_global_stats(self, labels: List[float]) -> None:
        """
        Update the global mean using online averaging.

        This allows the memory to know the "average" value of the world,
        which is used as the baseline for symbolic predictions.

        Args:
            labels: List of observed property values
        """
        if not labels:
            return

        # Online update: accumulate sum and count
        for label in labels:
            if not np.isnan(label):
                self.global_sum += label
                self.n_observed += 1

        # Update mean
        if self.n_observed > 0:
            self.global_mean = self.global_sum / self.n_observed

        logger.debug(f"[SemanticMemory] Updated global stats: mean={self.global_mean:.3f}, n={self.n_observed}")

    def ingest_rules(
        self,
        rules: List[Dict],
        episode_id: Optional[str] = None
    ) -> Dict:
        """
        Ingest rules from SAR extractor.

        Args:
            rules: List of rule dicts from SARExtractor
            episode_id: Optional episode identifier for provenance

        Returns:
            Dict with counts: {'added': n, 'updated': n, 'rejected': n}
        """
        if episode_id is None:
            episode_id = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        added = 0
        updated = 0
        rejected = 0

        for rule in rules:
            feature = rule.get('feature')
            if not feature:
                rejected += 1
                continue

            # Filter by quality thresholds
            p_value = rule.get('p_value', 1.0)
            effect_size = abs(rule.get('effect_size', 0.0))

            if p_value > self.min_p_value:
                rejected += 1
                continue

            if effect_size < self.min_effect_size:
                rejected += 1
                continue

            # Calculate confidence from p-value
            # confidence = 1 - p_value (simple approximation)
            confidence = 1.0 - p_value

            if confidence < self.min_confidence:
                rejected += 1
                continue

            # Check if rule already exists
            if feature in self._rules:
                self._update_rule(feature, rule, episode_id)
                updated += 1
            else:
                self._add_rule(feature, rule, episode_id)
                added += 1

        # Log ingestion
        result = {'added': added, 'updated': updated, 'rejected': rejected}
        self._ingestion_history.append({
            'episode_id': episode_id,
            'timestamp': datetime.now().isoformat(),
            'result': result
        })

        logger.info(f"[SemanticMemory] Ingested rules: +{added} new, ~{updated} updated, -{rejected} rejected")

        return result

    def _add_rule(self, feature: str, rule: Dict, episode_id: str) -> None:
        """Add a new rule to memory."""
        now = datetime.now().isoformat()

        effect_size = rule.get('effect_size', 0.0)
        p_value = rule.get('p_value', 0.05)
        n_with = rule.get('n_with_feature', 0)
        n_without = rule.get('n_without_feature', 0)
        n_observations = n_with + n_without

        # Calculate confidence interval (approximate)
        # Using standard error of difference in means
        std_with = rule.get('std_with_feature', 0.5)
        std_without = rule.get('std_without_feature', 0.5)

        if n_with > 1 and n_without > 1:
            se = math.sqrt((std_with**2 / n_with) + (std_without**2 / n_without))
            ci_half = 1.96 * se  # 95% CI
            effect_ci = (effect_size - ci_half, effect_size + ci_half)
        else:
            effect_ci = (effect_size - 0.5, effect_size + 0.5)

        self._rules[feature] = {
            'feature': feature,
            'effect_size': effect_size,
            'effect_ci': effect_ci,
            'direction': 'increases' if effect_size > 0 else 'decreases',
            'confidence': 1.0 - p_value,
            'p_value': p_value,
            'n_observations': n_observations,
            'n_with_feature': n_with,
            'n_without_feature': n_without,
            'first_discovered': now,
            'last_updated': now,
            'update_count': 1,
            'provenance': [episode_id],
            'rule_text': rule.get('rule_text', f'{feature} affects property')
        }

    def _update_rule(self, feature: str, new_rule: Dict, episode_id: str) -> None:
        """Update existing rule via Bayesian combination."""
        old = self._rules[feature]

        # Extract values
        old_effect = old['effect_size']
        old_n = old['n_observations']
        old_p = old['p_value']

        new_effect = new_rule.get('effect_size', 0.0)
        new_n = new_rule.get('n_with_feature', 0) + new_rule.get('n_without_feature', 0)
        new_p = new_rule.get('p_value', 0.05)

        # Bayesian update: weighted average by sample size
        combined_n = old_n + new_n
        if combined_n > 0:
            updated_effect = (old_effect * old_n + new_effect * new_n) / combined_n
        else:
            updated_effect = new_effect

        # Combined p-value (Fisher's method approximation)
        # More conservative: use geometric mean
        updated_p = math.sqrt(old_p * new_p)
        updated_confidence = 1.0 - updated_p

        # Update confidence interval
        se_estimate = abs(updated_effect) * 0.2  # Rough estimate
        effect_ci = (updated_effect - 1.96 * se_estimate,
                     updated_effect + 1.96 * se_estimate)

        # Update rule
        old['effect_size'] = updated_effect
        old['effect_ci'] = effect_ci
        old['direction'] = 'increases' if updated_effect > 0 else 'decreases'
        old['confidence'] = min(updated_confidence, 0.999)  # Cap at 0.999
        old['p_value'] = max(updated_p, 0.001)  # Floor at 0.001
        old['n_observations'] = combined_n
        old['n_with_feature'] = old.get('n_with_feature', 0) + new_rule.get('n_with_feature', 0)
        old['n_without_feature'] = old.get('n_without_feature', 0) + new_rule.get('n_without_feature', 0)
        old['last_updated'] = datetime.now().isoformat()
        old['update_count'] += 1

        # Append provenance (avoid duplicates)
        if episode_id not in old['provenance']:
            old['provenance'].append(episode_id)

    def query_rule(self, feature: str) -> Optional[Dict]:
        """Return full rule dict if exists, None otherwise."""
        return self._rules.get(feature)

    def query_effect(self, feature: str) -> Optional[float]:
        """Return just the effect size (convenience method)."""
        rule = self._rules.get(feature)
        return rule['effect_size'] if rule else None

    def get_applicable_rules(self, smiles: str) -> List[Dict]:
        """
        Get all rules that apply to a molecule.

        Args:
            smiles: SMILES string

        Returns:
            List of applicable rule dicts
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        # Extract features from molecule
        mol_features = self._extract_features(mol)

        # Find matching rules
        applicable = []
        for feature, rule in self._rules.items():
            if feature in mol_features and mol_features[feature]:
                applicable.append(rule)

        return applicable

    def _extract_features(self, mol: Chem.Mol) -> Dict[str, bool]:
        """Extract feature presence from molecule."""
        features = {}

        # Atom presence
        atoms = set(atom.GetAtomicNum() for atom in mol.GetAtoms())
        features['has_fluorine'] = 9 in atoms
        features['has_chlorine'] = 17 in atoms
        features['has_bromine'] = 35 in atoms
        features['has_nitrogen'] = 7 in atoms
        features['has_sulfur'] = 16 in atoms
        features['has_oxygen'] = 8 in atoms

        # Ring features
        n_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
        n_aliphatic = rdMolDescriptors.CalcNumAliphaticRings(mol)
        ring_info = mol.GetRingInfo()

        features['has_aromatic_ring'] = n_aromatic > 0
        features['has_aliphatic_ring'] = n_aliphatic > 0
        features['has_multiple_rings'] = ring_info.NumRings() > 1

        # Property bins
        try:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            n_rotatable = Descriptors.NumRotatableBonds(mol)
            n_hbd = Descriptors.NumHDonors(mol)
            n_hba = Descriptors.NumHAcceptors(mol)

            features['mw_over_300'] = mw > 300
            features['mw_over_400'] = mw > 400
            features['logp_over_2'] = logp > 2
            features['logp_over_3'] = logp > 3
            features['logp_negative'] = logp < 0
            features['tpsa_over_50'] = tpsa > 50
            features['tpsa_over_100'] = tpsa > 100
            features['n_rotatable_over_5'] = n_rotatable > 5
            features['n_hbd_over_2'] = n_hbd > 2
            features['n_hba_over_5'] = n_hba > 5
        except Exception:
            pass

        return features

    def predict_from_rules(
        self,
        smiles: str,
        baseline: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Make symbolic-only prediction from applicable rules.

        Args:
            smiles: SMILES string
            baseline: Baseline prediction to add effects to.
                      If None, uses self.global_mean (learned from data).

        Returns:
            (predicted_value, uncertainty_from_rules)

        Note:
            Effect sizes represent mean differences (with_feature - without_feature).
            Since features are often correlated (e.g., logP>3 and MW>300 both indicate
            hydrophobic molecules), we use WEIGHTED AVERAGE of effects instead of
            summing them, to avoid double-counting correlated effects.
        """
        # Use global mean as default baseline if not specified
        if baseline is None:
            baseline = self.global_mean

        applicable = self.get_applicable_rules(smiles)

        if not applicable:
            return baseline, 1.0  # High uncertainty if no rules apply

        # Use confidence-weighted AVERAGE of effects (not sum!)
        # This avoids double-counting correlated features
        effects = np.array([rule['effect_size'] for rule in applicable])
        confidences = np.array([rule['confidence'] for rule in applicable])

        # Weighted average effect
        weighted_effect = np.average(effects, weights=confidences)
        prediction = baseline + weighted_effect

        # Uncertainty: inverse of average confidence, reduced by having more rules
        avg_confidence = np.mean(confidences)
        # More rules = slightly lower uncertainty (but diminishing returns)
        n_rules_factor = 1.0 / (1.0 + 0.1 * len(applicable))
        uncertainty = (1.0 - avg_confidence) * n_rules_factor

        return prediction, uncertainty

    def get_interpretable_summary(self, top_n: int = 10) -> str:
        """Return formatted string of strongest rules."""
        if not self._rules:
            return "No rules in semantic memory."

        # Sort by absolute effect size
        sorted_rules = sorted(
            self._rules.values(),
            key=lambda r: abs(r['effect_size']),
            reverse=True
        )[:top_n]

        lines = ["Semantic Memory - Top Rules:", "=" * 50]

        for rule in sorted_rules:
            conf = rule['confidence']
            feature = rule['feature'].replace('_', ' ')
            direction = '+' if rule['effect_size'] > 0 else ''
            effect = rule['effect_size']
            n = rule['n_observations']
            p = rule['p_value']

            p_str = f"p<0.001" if p < 0.001 else f"p={p:.3f}"
            lines.append(f"[{conf:.2f}] {feature} → {direction}{effect:.2f} logS (n={n}, {p_str})")

        return "\n".join(lines)

    def export_rules(self, filepath: str) -> None:
        """Save all rules to JSON with full provenance."""
        data = {
            'metadata': {
                'exported_at': datetime.now().isoformat(),
                'n_rules': len(self._rules),
                'min_p_value': self.min_p_value,
                'min_confidence': self.min_confidence
            },
            'rules': list(self._rules.values()),
            'ingestion_history': self._ingestion_history
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Exported {len(self._rules)} rules to {filepath}")

    def import_rules(self, filepath: str) -> None:
        """Load rules from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Restore rules
        for rule in data.get('rules', []):
            feature = rule.get('feature')
            if feature:
                self._rules[feature] = rule

        # Restore history
        self._ingestion_history = data.get('ingestion_history', [])

        logger.info(f"Imported {len(self._rules)} rules from {filepath}")

    def get_statistics(self) -> Dict:
        """Return summary statistics about the memory."""
        if not self._rules:
            return {
                'n_rules': 0,
                'avg_confidence': 0.0,
                'avg_effect_size': 0.0,
                'total_observations': 0,
                'features': [],
                'global_mean': self.global_mean,
                'n_observed': self.n_observed
            }

        rules = list(self._rules.values())

        return {
            'n_rules': len(rules),
            'avg_confidence': float(np.mean([r['confidence'] for r in rules])),
            'avg_effect_size': float(np.mean([abs(r['effect_size']) for r in rules])),
            'total_observations': sum(r['n_observations'] for r in rules),
            'features': list(self._rules.keys()),
            'avg_update_count': float(np.mean([r['update_count'] for r in rules])),
            'total_provenance_links': sum(len(r['provenance']) for r in rules),
            'global_mean': self.global_mean,
            'n_observed': self.n_observed
        }

    def get_all_rules(self) -> List[Dict]:
        """Return all rules as a list."""
        return list(self._rules.values())

    def __len__(self) -> int:
        return len(self._rules)

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return f"SemanticMemory(n_rules={stats['n_rules']}, avg_confidence={stats['avg_confidence']:.2f})"
