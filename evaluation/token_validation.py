"""Validation variants for token prediction robustness.

This module tests the robustness and validity of token predictions through
various ablations and manipulations of the input.
"""

import re
from typing import List, Tuple, Dict
import numpy as np

from token_prediction.predictor import NextSentencePredictor
from textualization.base import TextualizationLayer


class ValidationVariants:
    """Test robustness and validity of token predictions.

    Implements validation tests:
    - Paraphrase robustness: Do synonyms yield similar NLL?
    - Stopword removal: Do content words matter more?
    - Candidate ranking: Does true observation rank highest?
    - Action conditioning: Does removing action increase NLL?
    """

    def __init__(
        self,
        predictor: NextSentencePredictor,
        textualizer: TextualizationLayer
    ):
        """Initialize validation suite.

        Args:
            predictor: Token-level predictor
            textualizer: Textualization layer
        """
        self.predictor = predictor
        self.textualizer = textualizer

    # === PARAPHRASE ROBUSTNESS ===

    def test_paraphrase_robustness(
        self,
        obs: Dict,
        paraphrase_rules: List[Tuple[str, str]]
    ) -> Dict[str, any]:
        """Test if template paraphrases yield similar NLL.

        Note: Full implementation requires completion API with echo=True
        to score exact strings. This is a placeholder implementation.

        Args:
            obs: Observation dict
            paraphrase_rules: List of (original, replacement) pairs

        Returns:
            Dict with paraphrase statistics
        """
        original = self.textualizer.textualize_observation(obs)

        # Generate paraphrases
        paraphrases = [original]
        for old, new in paraphrase_rules:
            para = original.replace(old, new)
            if para != original:
                paraphrases.append(para)

        # NOTE: To properly score exact text, we'd need completion API
        # with echo=True. OpenAI removed this in newer APIs.
        # Current workaround: return structure only
        return {
            'original_text': original,
            'num_paraphrases': len(paraphrases) - 1,
            'paraphrases': paraphrases[1:],
            'note': 'Full NLL scoring requires completion API with echo=True'
        }

    # === STOPWORD REMOVAL ===

    def test_stopword_removal(
        self,
        context: str,
        obs_text: str
    ) -> Dict[str, float]:
        """Remove stopwords and recompute NLL.

        Tests whether content words contribute more to prediction
        than function words.

        Args:
            context: Full context leading to observation
            obs_text: True observation text (not currently used)

        Returns:
            Dict with full and reduced NLLs
        """
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were',
            'it', 'this', 'that', 'to', 'of', 'in', 'on', 'at'
        }

        def remove_stopwords(text: str) -> str:
            words = text.split()
            return ' '.join(w for w in words if w.lower() not in stopwords)

        try:
            # Full context prediction
            full_pred = self.predictor.predict_next_observation(context)

            # Reduced context prediction
            reduced_context = remove_stopwords(context)
            reduced_pred = self.predictor.predict_next_observation(reduced_context)

            return {
                'full_nll': full_pred.sequence_nll,
                'reduced_nll': reduced_pred.sequence_nll,
                'delta_nll': reduced_pred.sequence_nll - full_pred.sequence_nll,
                'stopword_impact': (reduced_pred.sequence_nll - full_pred.sequence_nll) / full_pred.sequence_nll if full_pred.sequence_nll > 0 else 0,
                'num_stopwords_removed': len(context.split()) - len(reduced_context.split())
            }
        except Exception as e:
            return {
                'error': str(e),
                'full_nll': None,
                'reduced_nll': None
            }

    # === CANDIDATE RANKING ===

    def test_candidate_ranking(
        self,
        context: str,
        true_obs: str,
        decoys: List[str]
    ) -> Dict[str, any]:
        """Rank true observation among decoys.

        Args:
            context: Context leading to observation
            true_obs: True observation text
            decoys: List of plausible alternative observations

        Returns:
            Dict with rank, log-odds gap, and full rankings
        """
        candidates = [true_obs] + decoys

        try:
            # Rank candidates
            ranked = self.predictor.rank_candidates(context, candidates)

            # Find rank of true observation (0 = best)
            true_rank = next(
                (i for i, (c, _) in enumerate(ranked) if c == true_obs),
                len(ranked)  # If not found, worst possible rank
            )

            # Compute log-odds gap between true and best decoy
            true_nll = next((nll for c, nll in ranked if c == true_obs), float('inf'))
            best_decoy_nll = min((nll for c, nll in ranked if c != true_obs), default=float('inf'))
            log_odds_gap = best_decoy_nll - true_nll  # Positive = true is better

            return {
                'true_rank': true_rank,
                'log_odds_gap': log_odds_gap,
                'true_nll': true_nll,
                'best_decoy_nll': best_decoy_nll,
                'num_candidates': len(candidates),
                'all_rankings': ranked,
                'true_is_best': (true_rank == 0)
            }
        except Exception as e:
            return {
                'error': str(e),
                'true_rank': None
            }

    # === ACTION CONDITIONING ABLATION ===

    def test_action_conditioning(
        self,
        full_context: str,
        last_action: str
    ) -> Dict[str, float]:
        """Compare NLL with/without last action.

        Expect NLL to increase when action is removed (less information).

        Args:
            full_context: Full context including last action
            last_action: The last action text to remove

        Returns:
            Dict with full and no-action NLLs, and their difference
        """
        try:
            # Predict with full context
            full_pred = self.predictor.predict_next_observation(full_context)

            # Remove last action from context
            context_no_action = full_context.replace(last_action, "").strip()
            # Clean up multiple newlines
            context_no_action = re.sub(r'\n\n+', '\n', context_no_action)

            # Predict without action
            no_action_pred = self.predictor.predict_next_observation(context_no_action)

            # Expect no_action_nll > full_nll (action provides information)
            delta = no_action_pred.sequence_nll - full_pred.sequence_nll

            return {
                'full_nll': full_pred.sequence_nll,
                'no_action_nll': no_action_pred.sequence_nll,
                'delta_nll': delta,
                'action_effect': 'informative' if delta > 0 else 'not_informative',
                'relative_increase': delta / full_pred.sequence_nll if full_pred.sequence_nll > 0 else 0,
                'removed_action': last_action
            }
        except Exception as e:
            return {
                'error': str(e),
                'full_nll': None,
                'no_action_nll': None
            }

    # === NUMERICAL PERTURBATION ===

    def test_numerical_perturbation(
        self,
        context: str,
        true_value: float,
        perturbations: List[float] = [-5.0, -1.0, +1.0, +5.0]
    ) -> Dict[str, any]:
        """Test NLL sensitivity to numerical value changes.

        For observations like "Thermometer reads 23.5°C", test whether
        changing 23.5 to 24.5 affects NLL.

        Args:
            context: Context with numerical value
            true_value: The true numerical value
            perturbations: Deltas to apply to true value

        Returns:
            Dict with NLL for each perturbation
        """
        results = []

        # Extract pattern for numerical value (simplified)
        value_pattern = re.compile(r'\b' + re.escape(str(true_value)) + r'\b')

        for delta in perturbations:
            perturbed_value = true_value + delta
            perturbed_context = value_pattern.sub(str(perturbed_value), context, count=1)

            try:
                pred = self.predictor.predict_next_observation(perturbed_context)
                results.append({
                    'delta': delta,
                    'value': perturbed_value,
                    'nll': pred.sequence_nll
                })
            except Exception as e:
                results.append({
                    'delta': delta,
                    'value': perturbed_value,
                    'error': str(e)
                })

        return {
            'true_value': true_value,
            'perturbations': results,
            'note': 'Lower NLL for values closer to true indicates numerical sensitivity'
        }


# === VALIDATION SUITE RUNNER ===

def run_validation_suite(
    predictor: NextSentencePredictor,
    textualizer: TextualizationLayer,
    context: str,
    observation: Dict,
    true_obs_text: str
) -> Dict[str, any]:
    """Run complete validation suite on a single step.

    Args:
        predictor: Token-level predictor
        textualizer: Textualization layer
        context: Full context up to this step
        observation: Observation dictionary
        true_obs_text: Textualized observation

    Returns:
        Dictionary with all validation results
    """
    validator = ValidationVariants(predictor, textualizer)

    results = {}

    # Paraphrase robustness
    paraphrase_rules = [
        ("Thermometer reads", "Temperature measured"),
        ("Time elapsed", "Time passed"),
        ("seconds", "sec")
    ]
    results['paraphrase'] = validator.test_paraphrase_robustness(
        observation, paraphrase_rules
    )

    # Stopword removal
    results['stopwords'] = validator.test_stopword_removal(context, true_obs_text)

    # Candidate ranking (generate plausible decoys)
    decoys = [
        true_obs_text.replace("23.5", "25.0"),  # Wrong temperature
        true_obs_text.replace("23.5", "22.0"),  # Another wrong temperature
        "No reading available."  # Implausible
    ]
    results['ranking'] = validator.test_candidate_ranking(context, true_obs_text, decoys)

    # Action conditioning (extract last action)
    lines = context.strip().split('\n')
    last_action = next((line for line in reversed(lines) if 'Action taken:' in line), "")
    if last_action:
        results['action_ablation'] = validator.test_action_conditioning(context, last_action)

    return results
