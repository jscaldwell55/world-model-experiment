"""
DreamPipeline: Orchestrate generative exploration with confidence filtering.

This pipeline:
1. Generates virtual analogs for input molecules
2. Predicts properties with uncertainty quantification
3. Filters by confidence threshold (0.85 for cautious approach)
4. Caps synthetics to prevent overwhelming real data
5. Extracts SAR rules from combined data

Key design choices:
- High confidence threshold (0.85) compensates for calibration at 0.28
- 30% synthetic cap ensures real data dominates
- 0.6 synthetic weight provides additional safety margin
- Condition-aware logging tracks behavior across clean/noisy data
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .analog_generator import AnalogGenerator
from .sar_extractor import SARExtractor

logger = logging.getLogger(__name__)


class DreamPipeline:
    """
    Orchestrate generative exploration with confidence-based filtering.

    The pipeline generates virtual molecular analogs, predicts their
    properties, and filters to retain only high-confidence predictions
    for training augmentation.
    """

    def __init__(
        self,
        world_model,
        analog_generator: AnalogGenerator,
        sar_extractor: Optional[SARExtractor] = None,
        confidence_threshold: float = 0.85,  # Raised from 0.8 for calibration=0.28
        max_synthetics_ratio: float = 0.3,
        synthetic_weight: float = 0.6,
        random_state: int = 42
    ):
        """
        Initialize dream pipeline.

        Args:
            world_model: MolecularWorldModel for predictions
            analog_generator: AnalogGenerator for creating variants
            sar_extractor: Optional SARExtractor for rule discovery
            confidence_threshold: Minimum confidence to accept synthetic
                                  (0.85 = conservative given calibration)
            max_synthetics_ratio: Maximum synthetics as fraction of real data
            synthetic_weight: Weight for synthetic samples (0.6 < 1.0 for real)
            random_state: Random seed
        """
        self.world_model = world_model
        self.analog_generator = analog_generator
        self.sar_extractor = sar_extractor
        self.confidence_threshold = confidence_threshold
        self.max_synthetics_ratio = max_synthetics_ratio
        self.synthetic_weight = synthetic_weight
        self.rng = np.random.RandomState(random_state)

        # Tracking for condition-aware analysis
        self._dream_history: List[Dict] = []

    def dream(
        self,
        real_smiles: List[str],
        real_labels: List[float],
        condition: Optional[str] = None,
        n_variants_per_molecule: int = 5
    ) -> Dict:
        """
        Generate and filter synthetic training data.

        Args:
            real_smiles: Real observed SMILES
            real_labels: Real observed property values
            condition: Optional condition label for logging ('clean', 'noisy_15pct', etc.)
            n_variants_per_molecule: Number of analogs to generate per molecule

        Returns:
            Dict with:
            - synthetic_smiles: Filtered synthetic SMILES
            - synthetic_labels: Predicted labels for synthetics
            - synthetic_weights: Weights for training (all 0.6)
            - synthetic_details: Full provenance for each synthetic
            - statistics: Generation/filtering statistics
            - sar_rules: Discovered SAR rules (if extractor provided)
        """
        real_smiles_set: Set[str] = set(real_smiles)

        # 1. Generate analogs for ALL input molecules
        all_analogs = []
        for smiles in real_smiles:
            analogs = self.analog_generator.generate_analogs(
                smiles, n_variants=n_variants_per_molecule
            )
            all_analogs.extend(analogs)

        n_generated = len(all_analogs)

        if n_generated == 0:
            return self._empty_result(condition)

        # 2. Predict properties with uncertainty
        analog_smiles = [a['smiles'] for a in all_analogs]
        predictions, uncertainties = self.world_model.predict(
            analog_smiles, return_uncertainty=True
        )

        # 3. Calculate confidence scores
        # confidence = 1 - min(uncertainty, 1.0)
        # Higher uncertainty -> lower confidence
        confidences = 1.0 - np.minimum(uncertainties, 1.0)

        # 4. Filter by confidence threshold
        passed_confidence = []
        for i, analog in enumerate(all_analogs):
            if np.isnan(predictions[i]):
                continue  # Skip invalid predictions

            if confidences[i] >= self.confidence_threshold:
                passed_confidence.append({
                    **analog,
                    'predicted_label': float(predictions[i]),
                    'uncertainty': float(uncertainties[i]),
                    'confidence': float(confidences[i])
                })

        n_passed = len(passed_confidence)

        # 5. Deduplicate: remove analogs already in real data
        deduplicated = [
            a for a in passed_confidence
            if a['smiles'] not in real_smiles_set
        ]
        n_after_dedup = len(deduplicated)

        # 6. Cap to max_synthetics_ratio
        max_synthetics = int(len(real_smiles) * self.max_synthetics_ratio)

        if len(deduplicated) > max_synthetics:
            # Sort by confidence and take top
            deduplicated.sort(key=lambda x: x['confidence'], reverse=True)
            deduplicated = deduplicated[:max_synthetics]

        n_final = len(deduplicated)

        # 7. Extract SAR rules if extractor provided
        sar_rules = []
        if self.sar_extractor is not None and n_final > 0:
            combined_smiles = real_smiles + [d['smiles'] for d in deduplicated]
            combined_labels = list(real_labels) + [d['predicted_label'] for d in deduplicated]
            sar_rules = self.sar_extractor.extract_rules(combined_smiles, combined_labels)

        # 8. Prepare output
        synthetic_smiles = [d['smiles'] for d in deduplicated]
        synthetic_labels = [d['predicted_label'] for d in deduplicated]
        synthetic_weights = [self.synthetic_weight] * n_final

        # Calculate statistics
        acceptance_rate = n_passed / n_generated if n_generated > 0 else 0.0
        final_rate = n_final / n_generated if n_generated > 0 else 0.0

        # Confidence distribution for passed samples
        if passed_confidence:
            confidence_values = [p['confidence'] for p in passed_confidence]
            confidence_stats = {
                'mean': float(np.mean(confidence_values)),
                'std': float(np.std(confidence_values)),
                'min': float(np.min(confidence_values)),
                'max': float(np.max(confidence_values))
            }
        else:
            confidence_stats = {'mean': 0, 'std': 0, 'min': 0, 'max': 0}

        result = {
            # Synthetic data
            'synthetic_smiles': synthetic_smiles,
            'synthetic_labels': synthetic_labels,
            'synthetic_weights': synthetic_weights,
            'synthetic_details': deduplicated,

            # Statistics
            'n_real': len(real_smiles),
            'n_analogs_generated': n_generated,
            'n_passed_confidence': n_passed,
            'n_after_dedup': n_after_dedup,
            'n_after_cap': n_final,
            'acceptance_rate': acceptance_rate,
            'final_rate': final_rate,
            'confidence_threshold': self.confidence_threshold,
            'confidence_stats': confidence_stats,

            # Condition tracking
            'condition': condition,

            # SAR output
            'sar_rules': sar_rules,
            'n_rules': len(sar_rules)
        }

        # Log condition-aware statistics
        self._log_dream_result(result)
        self._dream_history.append(result)

        return result

    def _empty_result(self, condition: Optional[str]) -> Dict:
        """Return empty result when no analogs could be generated."""
        return {
            'synthetic_smiles': [],
            'synthetic_labels': [],
            'synthetic_weights': [],
            'synthetic_details': [],
            'n_real': 0,
            'n_analogs_generated': 0,
            'n_passed_confidence': 0,
            'n_after_dedup': 0,
            'n_after_cap': 0,
            'acceptance_rate': 0.0,
            'final_rate': 0.0,
            'confidence_threshold': self.confidence_threshold,
            'confidence_stats': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
            'condition': condition,
            'sar_rules': [],
            'n_rules': 0
        }

    def _log_dream_result(self, result: Dict) -> None:
        """Log dream results with condition-aware context."""
        condition = result.get('condition', 'unknown')
        n_gen = result['n_analogs_generated']
        n_passed = result['n_passed_confidence']
        n_final = result['n_after_cap']
        rate = result['acceptance_rate']

        log_msg = (
            f"[Dream|{condition}] Generated: {n_gen}, "
            f"Passed confidence ({self.confidence_threshold:.0%}): {n_passed} ({rate:.1%}), "
            f"Final: {n_final}"
        )

        # Warn if acceptance rate is very low (< 5%) for noisy conditions
        if rate < 0.05 and n_gen > 0:
            log_msg += " [LOW ACCEPTANCE - model correctly cautious]"
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        # Log SAR rules if any discovered
        if result['n_rules'] > 0:
            logger.info(f"  Discovered {result['n_rules']} SAR rules")

    def get_condition_summary(self) -> Dict[str, Dict]:
        """
        Get summary statistics grouped by condition.

        Useful for analyzing whether dreams help/hurt differently
        by data quality (clean vs noisy).
        """
        summary = {}

        for result in self._dream_history:
            condition = result.get('condition', 'unknown')

            if condition not in summary:
                summary[condition] = {
                    'n_dreams': 0,
                    'total_generated': 0,
                    'total_passed': 0,
                    'total_final': 0,
                    'acceptance_rates': [],
                    'confidence_means': []
                }

            s = summary[condition]
            s['n_dreams'] += 1
            s['total_generated'] += result['n_analogs_generated']
            s['total_passed'] += result['n_passed_confidence']
            s['total_final'] += result['n_after_cap']
            s['acceptance_rates'].append(result['acceptance_rate'])
            if result['confidence_stats']['mean'] > 0:
                s['confidence_means'].append(result['confidence_stats']['mean'])

        # Compute aggregates
        for condition, s in summary.items():
            if s['n_dreams'] > 0:
                s['mean_acceptance_rate'] = float(np.mean(s['acceptance_rates']))
                s['std_acceptance_rate'] = float(np.std(s['acceptance_rates']))
                s['overall_acceptance'] = (
                    s['total_passed'] / s['total_generated']
                    if s['total_generated'] > 0 else 0.0
                )
                if s['confidence_means']:
                    s['mean_confidence'] = float(np.mean(s['confidence_means']))
                else:
                    s['mean_confidence'] = 0.0

        return summary

    def reset_history(self) -> None:
        """Clear dream history for fresh tracking."""
        self._dream_history = []
