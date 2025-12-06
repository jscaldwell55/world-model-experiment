"""
ConsistencyChecker: Measure alignment between neural model and symbolic rules.

This module tests the convergence hypothesis: as learning progresses,
the neural model's predictions should increasingly align with the
symbolic rules in semantic memory.

Key Metrics:
- Rule consistency: Does neural agree with rule direction?
- Conflict detection: Cases where neural contradicts symbolic
- Convergence tracking: Consistency trend over time
"""

import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

logger = logging.getLogger(__name__)


class ConsistencyChecker:
    """
    Measure alignment between neural model predictions and symbolic rules.

    Tests whether the neural model learns patterns consistent with
    the interpretable rules in semantic memory.
    """

    def __init__(self, world_model, semantic_memory):
        """
        Initialize consistency checker.

        Args:
            world_model: MolecularWorldModel for neural predictions
            semantic_memory: SemanticMemory with symbolic rules
        """
        self.world_model = world_model
        self.semantic_memory = semantic_memory

        # History for convergence tracking
        self._consistency_history: List[Dict] = []

    def check_rule_consistency(
        self,
        smiles_list: List[str],
        labels: List[float]
    ) -> Dict:
        """
        Check consistency between neural predictions and symbolic rules.

        For each rule, tests whether the neural model's predictions
        agree with the rule's direction (e.g., "fluorine increases solubility").

        Args:
            smiles_list: List of SMILES strings
            labels: True property values (for reference)

        Returns:
            Dict with consistency metrics and conflict details
        """
        if not self.semantic_memory.get_all_rules():
            return {
                'overall_consistency': 0.0,
                'per_rule_consistency': {},
                'conflicts': [],
                'n_testable_pairs': 0,
                'message': 'No rules in semantic memory'
            }

        # Get neural predictions for all molecules
        predictions, uncertainties = self.world_model.predict(
            smiles_list, return_uncertainty=True
        )

        # Track consistency per rule
        rule_agreements = defaultdict(list)
        conflicts = []

        # For each rule, find molecule pairs that differ only in that feature
        rules = self.semantic_memory.get_all_rules()

        for rule in rules:
            feature = rule['feature']
            expected_direction = 1 if rule['direction'] == 'increases' else -1

            # Group molecules by feature presence
            with_feature = []
            without_feature = []

            for i, smiles in enumerate(smiles_list):
                if np.isnan(predictions[i]):
                    continue

                applicable = self.semantic_memory.get_applicable_rules(smiles)
                has_feature = any(r['feature'] == feature for r in applicable)

                if has_feature:
                    with_feature.append((i, smiles, predictions[i]))
                else:
                    without_feature.append((i, smiles, predictions[i]))

            # Compare predictions between groups
            if len(with_feature) >= 3 and len(without_feature) >= 3:
                mean_with = np.mean([x[2] for x in with_feature])
                mean_without = np.mean([x[2] for x in without_feature])

                neural_direction = 1 if mean_with > mean_without else -1
                is_consistent = neural_direction == expected_direction

                rule_agreements[feature].append(is_consistent)

                if not is_consistent:
                    conflicts.append({
                        'rule': feature,
                        'rule_direction': rule['direction'],
                        'neural_mean_with': float(mean_with),
                        'neural_mean_without': float(mean_without),
                        'neural_direction': 'increases' if neural_direction > 0 else 'decreases',
                        'n_with': len(with_feature),
                        'n_without': len(without_feature)
                    })

        # Calculate overall consistency
        all_agreements = []
        per_rule = {}

        for feature, agreements in rule_agreements.items():
            if agreements:
                consistency = np.mean(agreements)
                per_rule[feature] = float(consistency)
                all_agreements.extend(agreements)

        overall = float(np.mean(all_agreements)) if all_agreements else 0.0

        return {
            'overall_consistency': overall,
            'per_rule_consistency': per_rule,
            'conflicts': conflicts,
            'n_testable_pairs': len(all_agreements),
            'n_rules_tested': len(rule_agreements),
            'n_conflicts': len(conflicts)
        }

    def check_pairwise_consistency(
        self,
        smiles_list: List[str],
        max_pairs: int = 500
    ) -> Dict:
        """
        Alternative consistency check using molecule pairs.

        For each rule, finds pairs of similar molecules that differ
        in that feature and checks if neural predictions align.

        More granular but computationally heavier.
        """
        rules = self.semantic_memory.get_all_rules()
        if not rules:
            return {'overall_consistency': 0.0, 'message': 'No rules'}

        # Get predictions
        predictions, _ = self.world_model.predict(smiles_list, return_uncertainty=True)

        # Build feature presence map
        feature_map = {}  # smiles -> set of features
        for smiles in smiles_list:
            applicable = self.semantic_memory.get_applicable_rules(smiles)
            feature_map[smiles] = set(r['feature'] for r in applicable)

        # Check consistency per rule
        rule_scores = {}
        total_consistent = 0
        total_pairs = 0

        for rule in rules:
            feature = rule['feature']
            expected_sign = 1 if rule['direction'] == 'increases' else -1

            consistent = 0
            pairs_checked = 0

            # Find pairs differing in this feature
            for i, smi_i in enumerate(smiles_list):
                if np.isnan(predictions[i]):
                    continue

                has_i = feature in feature_map.get(smi_i, set())

                for j, smi_j in enumerate(smiles_list[i+1:], i+1):
                    if np.isnan(predictions[j]):
                        continue

                    has_j = feature in feature_map.get(smi_j, set())

                    # Only check pairs where exactly one has the feature
                    if has_i == has_j:
                        continue

                    pairs_checked += 1
                    if pairs_checked > max_pairs // len(rules):
                        break

                    # Check if neural prediction matches rule direction
                    if has_i:
                        neural_diff = predictions[i] - predictions[j]
                    else:
                        neural_diff = predictions[j] - predictions[i]

                    neural_sign = 1 if neural_diff > 0 else -1

                    if neural_sign == expected_sign:
                        consistent += 1

                if pairs_checked > max_pairs // len(rules):
                    break

            if pairs_checked > 0:
                rule_scores[feature] = consistent / pairs_checked
                total_consistent += consistent
                total_pairs += pairs_checked

        overall = total_consistent / total_pairs if total_pairs > 0 else 0.0

        return {
            'overall_consistency': float(overall),
            'per_rule_consistency': rule_scores,
            'total_pairs_checked': total_pairs
        }

    def track_consistency_over_time(
        self,
        consistency_score: float,
        episode: int,
        additional_metrics: Optional[Dict] = None
    ) -> None:
        """
        Record consistency score for convergence analysis.

        Args:
            consistency_score: Current consistency value
            episode: Episode/step number
            additional_metrics: Optional extra metrics to track
        """
        record = {
            'episode': episode,
            'consistency': consistency_score,
            'n_rules': len(self.semantic_memory),
            'timestamp': np.datetime64('now')
        }

        if additional_metrics:
            record.update(additional_metrics)

        self._consistency_history.append(record)

    def get_convergence_report(self) -> str:
        """
        Generate report on consistency trend over episodes.

        Tests the hypothesis that consistency should increase
        as learning progresses.
        """
        if len(self._consistency_history) < 2:
            return "Insufficient data for convergence analysis (need >= 2 checkpoints)"

        lines = ["Neural-Symbolic Convergence Report", "=" * 50]

        # Extract history
        episodes = [h['episode'] for h in self._consistency_history]
        consistencies = [h['consistency'] for h in self._consistency_history]
        n_rules = [h['n_rules'] for h in self._consistency_history]

        # Basic stats
        lines.append(f"\nCheckpoints: {len(self._consistency_history)}")
        lines.append(f"Episode range: {min(episodes)} - {max(episodes)}")
        lines.append(f"Initial consistency: {consistencies[0]:.3f}")
        lines.append(f"Final consistency: {consistencies[-1]:.3f}")
        lines.append(f"Change: {consistencies[-1] - consistencies[0]:+.3f}")

        # Trend analysis
        if len(consistencies) >= 3:
            # Simple linear regression
            x = np.array(episodes)
            y = np.array(consistencies)
            slope = np.polyfit(x, y, 1)[0]

            lines.append(f"\nTrend: {'INCREASING' if slope > 0.001 else 'DECREASING' if slope < -0.001 else 'STABLE'}")
            lines.append(f"Slope: {slope:.4f} per episode")

            # Check convergence hypothesis
            if consistencies[-1] > consistencies[0] + 0.1:
                lines.append("\n✓ CONVERGENCE HYPOTHESIS SUPPORTED")
                lines.append("  Neural and symbolic predictions are aligning over time.")
            elif consistencies[-1] < consistencies[0] - 0.1:
                lines.append("\n✗ DIVERGENCE DETECTED")
                lines.append("  Neural model may be learning patterns that contradict rules.")
            else:
                lines.append("\n~ CONSISTENCY STABLE")
                lines.append("  No significant convergence or divergence observed.")

        # Rules growth
        lines.append(f"\nRules in memory: {n_rules[0]} → {n_rules[-1]}")

        return "\n".join(lines)

    def get_history(self) -> List[Dict]:
        """Return full consistency history."""
        return self._consistency_history.copy()

    def get_latest_consistency(self) -> float:
        """Return most recent consistency score."""
        if not self._consistency_history:
            return 0.0
        return self._consistency_history[-1]['consistency']

    def reset_history(self) -> None:
        """Clear consistency history for fresh tracking."""
        self._consistency_history = []
