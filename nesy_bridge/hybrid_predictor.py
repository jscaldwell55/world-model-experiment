"""
HybridPredictor: Combine neural and symbolic predictions.

This module provides the foundation for Phase 3+ dual-path prediction:
- Neural path: Fast, implicit predictions from world model
- Symbolic path: Interpretable predictions from SAR rules
- Hybrid: Weighted combination with conflict detection

Future Direction:
- This becomes the "slow path" that validates LLM "fast path"
- Disagreement between LLM and Hybrid triggers uncertainty flag
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

logger = logging.getLogger(__name__)


class HybridPredictor:
    """
    Combine neural and symbolic predictions.

    Provides confidence-weighted combination of neural model
    predictions and symbolic rule-based predictions.
    """

    # Agreement thresholds
    CLOSE_THRESHOLD = 0.5  # Predictions within 0.5 are "close"
    CONFLICT_THRESHOLD = 1.0  # Predictions >1.0 apart are "conflicts"

    def __init__(
        self,
        world_model,
        semantic_memory,
        neural_weight: float = 0.7,
        baseline_prediction: float = 0.0
    ):
        """
        Initialize hybrid predictor.

        Args:
            world_model: MolecularWorldModel for neural predictions
            semantic_memory: SemanticMemory with symbolic rules
            neural_weight: Weight for neural prediction (0-1)
                          Symbolic weight = 1 - neural_weight
            baseline_prediction: Default prediction when no rules apply
        """
        self.world_model = world_model
        self.semantic_memory = semantic_memory
        self.neural_weight = neural_weight
        self.symbolic_weight = 1.0 - neural_weight
        self.baseline_prediction = baseline_prediction

    def predict(self, smiles: str) -> Dict:
        """
        Make hybrid prediction combining neural and symbolic.

        Args:
            smiles: SMILES string

        Returns:
            Dict with neural, symbolic, and hybrid predictions plus metadata
        """
        # Get neural prediction
        neural_pred, neural_unc = self.world_model.predict(
            smiles, return_uncertainty=True
        )

        if np.isnan(neural_pred):
            return self._invalid_prediction(smiles)

        # Get applicable symbolic rules
        applicable_rules = self.semantic_memory.get_applicable_rules(smiles)
        rule_names = [r['feature'] for r in applicable_rules]

        # Calculate symbolic prediction
        # Note: predict_from_rules uses semantic_memory.global_mean by default
        # which is the learned average from observed data
        if applicable_rules:
            symbolic_pred, symbolic_unc = self.semantic_memory.predict_from_rules(smiles)
        else:
            # No rules apply - fall back to neural only
            symbolic_pred = neural_pred  # Use neural as fallback
            symbolic_unc = 1.0  # High uncertainty

        # Calculate agreement
        diff = abs(neural_pred - symbolic_pred)
        if diff < self.CLOSE_THRESHOLD:
            agreement = True
            confidence = 'high'
        elif diff < self.CONFLICT_THRESHOLD:
            agreement = True
            confidence = 'medium'
        else:
            agreement = False
            confidence = 'low'

        # Hybrid prediction: weighted combination
        # Adjust weights based on confidence
        if applicable_rules:
            # More rules = more weight to symbolic
            rule_confidence = np.mean([r['confidence'] for r in applicable_rules])
            adjusted_neural_weight = self.neural_weight * (1 + neural_unc) / 2
            adjusted_symbolic_weight = self.symbolic_weight * rule_confidence

            # Normalize
            total_weight = adjusted_neural_weight + adjusted_symbolic_weight
            w_neural = adjusted_neural_weight / total_weight
            w_symbolic = adjusted_symbolic_weight / total_weight
        else:
            # No rules - use neural only
            w_neural = 1.0
            w_symbolic = 0.0

        hybrid_pred = w_neural * neural_pred + w_symbolic * symbolic_pred

        # Hybrid uncertainty
        hybrid_unc = w_neural * neural_unc + w_symbolic * symbolic_unc

        return {
            'smiles': smiles,
            'neural_prediction': float(neural_pred),
            'neural_uncertainty': float(neural_unc),
            'symbolic_prediction': float(symbolic_pred),
            'symbolic_uncertainty': float(symbolic_unc),
            'symbolic_rules_applied': rule_names,
            'n_rules_applied': len(applicable_rules),
            'hybrid_prediction': float(hybrid_pred),
            'hybrid_uncertainty': float(hybrid_unc),
            'neural_weight_used': float(w_neural),
            'symbolic_weight_used': float(w_symbolic),
            'agreement': agreement,
            'prediction_difference': float(diff),
            'confidence': confidence
        }

    def _invalid_prediction(self, smiles: str) -> Dict:
        """Return structure for invalid molecule."""
        return {
            'smiles': smiles,
            'neural_prediction': float('nan'),
            'neural_uncertainty': float('nan'),
            'symbolic_prediction': float('nan'),
            'symbolic_uncertainty': float('nan'),
            'symbolic_rules_applied': [],
            'n_rules_applied': 0,
            'hybrid_prediction': float('nan'),
            'hybrid_uncertainty': float('nan'),
            'neural_weight_used': 0.0,
            'symbolic_weight_used': 0.0,
            'agreement': False,
            'prediction_difference': float('nan'),
            'confidence': 'invalid'
        }

    def batch_predict(self, smiles_list: List[str]) -> List[Dict]:
        """
        Make hybrid predictions for multiple molecules.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            List of prediction dicts
        """
        return [self.predict(smiles) for smiles in smiles_list]

    def get_prediction_stats(self, smiles_list: List[str]) -> Dict:
        """
        Get statistics on hybrid predictions for a batch.

        Useful for understanding how often neural and symbolic agree.
        """
        predictions = self.batch_predict(smiles_list)

        # Filter valid predictions
        valid = [p for p in predictions if p['confidence'] != 'invalid']

        if not valid:
            return {
                'n_total': len(predictions),
                'n_valid': 0,
                'agreement_rate': 0.0,
                'avg_n_rules': 0.0
            }

        # Calculate stats
        agreements = [p['agreement'] for p in valid]
        n_rules = [p['n_rules_applied'] for p in valid]
        confidence_counts = {
            'high': sum(1 for p in valid if p['confidence'] == 'high'),
            'medium': sum(1 for p in valid if p['confidence'] == 'medium'),
            'low': sum(1 for p in valid if p['confidence'] == 'low')
        }

        neural_preds = [p['neural_prediction'] for p in valid]
        symbolic_preds = [p['symbolic_prediction'] for p in valid]
        hybrid_preds = [p['hybrid_prediction'] for p in valid]

        return {
            'n_total': len(predictions),
            'n_valid': len(valid),
            'agreement_rate': float(np.mean(agreements)),
            'avg_n_rules': float(np.mean(n_rules)),
            'confidence_distribution': confidence_counts,
            'neural_pred_mean': float(np.mean(neural_preds)),
            'neural_pred_std': float(np.std(neural_preds)),
            'symbolic_pred_mean': float(np.mean(symbolic_preds)),
            'symbolic_pred_std': float(np.std(symbolic_preds)),
            'hybrid_pred_mean': float(np.mean(hybrid_preds)),
            'hybrid_pred_std': float(np.std(hybrid_preds)),
            'avg_prediction_diff': float(np.mean([p['prediction_difference'] for p in valid]))
        }

    def evaluate_predictions(
        self,
        smiles_list: List[str],
        true_labels: List[float]
    ) -> Dict:
        """
        Evaluate neural, symbolic, and hybrid predictions against truth.

        Args:
            smiles_list: List of SMILES
            true_labels: True property values

        Returns:
            Dict with MAE for each prediction type
        """
        predictions = self.batch_predict(smiles_list)
        true_labels = np.array(true_labels)

        # Collect predictions
        neural_preds = []
        symbolic_preds = []
        hybrid_preds = []
        valid_labels = []

        for i, p in enumerate(predictions):
            if p['confidence'] != 'invalid' and not np.isnan(p['neural_prediction']):
                neural_preds.append(p['neural_prediction'])
                symbolic_preds.append(p['symbolic_prediction'])
                hybrid_preds.append(p['hybrid_prediction'])
                valid_labels.append(true_labels[i])

        if not neural_preds:
            return {
                'neural_mae': float('nan'),
                'symbolic_mae': float('nan'),
                'hybrid_mae': float('nan'),
                'n_valid': 0
            }

        neural_preds = np.array(neural_preds)
        symbolic_preds = np.array(symbolic_preds)
        hybrid_preds = np.array(hybrid_preds)
        valid_labels = np.array(valid_labels)

        return {
            'neural_mae': float(np.mean(np.abs(neural_preds - valid_labels))),
            'symbolic_mae': float(np.mean(np.abs(symbolic_preds - valid_labels))),
            'hybrid_mae': float(np.mean(np.abs(hybrid_preds - valid_labels))),
            'n_valid': len(valid_labels),
            'neural_better_pct': float(np.mean(
                np.abs(neural_preds - valid_labels) < np.abs(symbolic_preds - valid_labels)
            )),
            'hybrid_vs_neural_improvement': float(
                np.mean(np.abs(neural_preds - valid_labels)) -
                np.mean(np.abs(hybrid_preds - valid_labels))
            )
        }

    def get_conflict_analysis(
        self,
        smiles_list: List[str],
        true_labels: Optional[List[float]] = None
    ) -> Dict:
        """
        Analyze cases where neural and symbolic disagree.

        Helps understand why the systems might diverge.
        """
        predictions = self.batch_predict(smiles_list)

        conflicts = []
        for i, p in enumerate(predictions):
            if p['confidence'] == 'low':
                conflict_info = {
                    'smiles': p['smiles'],
                    'neural': p['neural_prediction'],
                    'symbolic': p['symbolic_prediction'],
                    'difference': p['prediction_difference'],
                    'rules_applied': p['symbolic_rules_applied']
                }

                if true_labels is not None:
                    # Add which one was right
                    neural_error = abs(p['neural_prediction'] - true_labels[i])
                    symbolic_error = abs(p['symbolic_prediction'] - true_labels[i])
                    conflict_info['true_label'] = true_labels[i]
                    conflict_info['neural_error'] = float(neural_error)
                    conflict_info['symbolic_error'] = float(symbolic_error)
                    conflict_info['neural_was_better'] = neural_error < symbolic_error

                conflicts.append(conflict_info)

        # Summarize
        if conflicts and true_labels is not None:
            neural_better_in_conflicts = sum(
                1 for c in conflicts if c.get('neural_was_better', False)
            )
            pct_neural_better = neural_better_in_conflicts / len(conflicts)
        else:
            pct_neural_better = None

        return {
            'n_conflicts': len(conflicts),
            'conflict_rate': len(conflicts) / len(predictions) if predictions else 0,
            'conflicts': conflicts,
            'neural_better_in_conflicts_pct': pct_neural_better
        }

    def set_neural_weight(self, weight: float) -> None:
        """Update the neural weight (symbolic = 1 - neural)."""
        self.neural_weight = max(0.0, min(1.0, weight))
        self.symbolic_weight = 1.0 - self.neural_weight
        logger.info(f"Updated weights: neural={self.neural_weight:.2f}, symbolic={self.symbolic_weight:.2f}")
