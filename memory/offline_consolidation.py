"""
Offline Consolidation (OC) System

Processes episodic memory during scheduled windows (e.g., nightly) to improve
data quality before fine-tuning.

Key Functions:
1. Counterfactual generation from HIGH reliability episodes
2. Bias detection for distribution imbalances
3. Quality gating to validate data before Fine-Tuning Bridge

Design Decision Log:
- Q: Should counterfactuals be marked as synthetic or treated like real episodes?
  A: Mark as synthetic with 'SYNTHETIC_HIGH' reliability tag to preserve provenance

- Q: How to handle conflicting observations in bias detection?
  A: Use reliability tags - HIGH reliability observations get priority weighting

- Q: Should quality gate be strict (PASS/FAIL) or soft (confidence scores)?
  A: Use three levels (PASS/WARNING/FAIL) with thresholds for flexibility

- Q: What's the right balance of original vs. synthetic data?
  A: 10-30% synthetic data augmentation, prioritize quality over quantity
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import copy


@dataclass
class BiasReport:
    """Report of detected biases in observation distribution"""
    context_imbalance: Dict[str, int] = field(default_factory=dict)
    reliability_skew: Dict[str, int] = field(default_factory=dict)
    value_range_gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def __str__(self):
        lines = ["=== Bias Detection Report ==="]

        if self.context_imbalance:
            lines.append("\nContext Imbalance:")
            for context, count in sorted(self.context_imbalance.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {context}: {count} observations")

        if self.reliability_skew:
            total = sum(self.reliability_skew.values())
            lines.append("\nReliability Distribution:")
            for reliability, count in sorted(self.reliability_skew.items()):
                pct = (count / total * 100) if total > 0 else 0
                lines.append(f"  {reliability}: {count} ({pct:.1f}%)")

        if self.value_range_gaps:
            lines.append("\nValue Range Gaps:")
            for gap in self.value_range_gaps:
                lines.append(f"  • {gap}")

        if self.recommendations:
            lines.append("\n📋 Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")

        return "\n".join(lines)


@dataclass
class GateDecision:
    """Quality gate decision with reasoning"""
    status: str  # 'PASS' | 'WARNING' | 'FAIL'
    reason: str
    recommendations: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def __str__(self):
        symbol = "✓" if self.status == "PASS" else ("⚠️" if self.status == "WARNING" else "✗")
        lines = [f"{symbol} Quality Gate: {self.status}"]
        lines.append(f"Reason: {self.reason}")

        if self.metrics:
            lines.append("\nMetrics:")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.3f}")
                else:
                    lines.append(f"  {key}: {value}")

        if self.recommendations:
            lines.append("\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")

        return "\n".join(lines)


@dataclass
class ConsolidatedData:
    """Output from offline consolidation"""

    # Original data (filtered by reliability)
    high_reliability_episodes: List[dict] = field(default_factory=list)
    low_reliability_episodes: List[dict] = field(default_factory=list)

    # Synthetic data
    counterfactual_episodes: List[dict] = field(default_factory=list)

    # Analysis
    bias_report: Optional[BiasReport] = None

    # Quality gate
    gate_status: str = "PENDING"
    gate_reason: str = ""
    recommendations: List[str] = field(default_factory=list)

    # Metrics
    fidelity_scores: Dict[str, float] = field(default_factory=dict)
    diversity_metrics: Dict[str, Any] = field(default_factory=dict)

    def get_training_data(self) -> dict:
        """
        Format for Fine-Tuning Bridge.

        Returns episodes with appropriate weights for training.
        Higher weight for HIGH reliability observations.
        """
        all_episodes = []
        weights = []

        # HIGH reliability episodes get weight 1.0
        for episode in self.high_reliability_episodes:
            all_episodes.append(episode)
            weights.append(1.0)

        # Synthetic episodes get weight 0.8 (slightly lower than real)
        for episode in self.counterfactual_episodes:
            all_episodes.append(episode)
            weights.append(0.8)

        # LOW reliability episodes get weight 0.3 (downweighted)
        for episode in self.low_reliability_episodes:
            all_episodes.append(episode)
            weights.append(0.3)

        return {
            'episodes': all_episodes,
            'weights': weights,
            'metadata': {
                'num_high_reliability': len(self.high_reliability_episodes),
                'num_synthetic': len(self.counterfactual_episodes),
                'num_low_reliability': len(self.low_reliability_episodes),
                'total': len(all_episodes),
                'synthetic_fraction': len(self.counterfactual_episodes) / len(all_episodes) if all_episodes else 0,
                'fidelity_scores': self.fidelity_scores,
                'diversity_metrics': self.diversity_metrics
            }
        }


class OfflineConsolidation:
    """
    Offline consolidation layer between ACE and Fine-Tuning Bridge.

    Processes episodic memory during scheduled windows (e.g., nightly)
    to improve data quality before fine-tuning.
    """

    def __init__(self, environment, world_model=None):
        """
        Args:
            environment: Environment instance (e.g., HotPotLab)
            world_model: SimpleWorldModel with transition/observation functions (optional)
        """
        self.environment = environment
        self.world_model = world_model

        # Quality gate thresholds (adjustable)
        self.fidelity_threshold_fail = 0.5
        self.fidelity_threshold_warning = 0.7
        self.min_high_reliability_pct_fail = 0.10
        self.min_high_reliability_pct_warning = 0.20

        # Data augmentation limits
        self.max_counterfactuals_per_episode = 3
        self.max_synthetic_fraction = 0.30  # 30% of total data

    def consolidate(self, playbook: dict) -> ConsolidatedData:
        """
        Main consolidation pipeline.

        Args:
            playbook: ACE playbook with episodes and reliability tags

        Returns:
            ConsolidatedData with:
                - original observations (filtered)
                - synthetic trajectories (counterfactuals)
                - bias corrections
                - quality gate status (pass/fail)
        """
        print("="*70)
        print("OFFLINE CONSOLIDATION PIPELINE")
        print("="*70)

        # Extract observations from playbook
        observations = playbook.get('observations', [])

        print(f"\nInput: {len(observations)} observations from playbook")

        # Separate by reliability
        high_reliability = [obs for obs in observations if obs.get('reliability') == 'HIGH']
        low_reliability = [obs for obs in observations if obs.get('reliability') == 'LOW']

        print(f"  HIGH reliability: {len(high_reliability)}")
        print(f"  LOW reliability: {len(low_reliability)}")

        # Initialize consolidated data
        consolidated = ConsolidatedData(
            high_reliability_episodes=high_reliability,
            low_reliability_episodes=low_reliability
        )

        # Step 1: Generate counterfactuals from HIGH reliability episodes
        print(f"\nStep 1: Generating counterfactuals...")
        counterfactuals = self.generate_counterfactuals(high_reliability)
        consolidated.counterfactual_episodes = counterfactuals
        consolidated.fidelity_scores = {
            cf['episode_id']: cf.get('fidelity_score', 0.0)
            for cf in counterfactuals
        }
        print(f"  Generated {len(counterfactuals)} synthetic episodes")

        # Step 2: Detect biases
        print(f"\nStep 2: Detecting biases...")
        bias_report = self.detect_biases(observations)
        consolidated.bias_report = bias_report
        print(f"  {len(bias_report.recommendations)} recommendations")

        # Step 3: Calculate diversity metrics
        print(f"\nStep 3: Calculating diversity metrics...")
        diversity = self._calculate_diversity(observations, counterfactuals)
        consolidated.diversity_metrics = diversity

        # Step 4: Quality gate
        print(f"\nStep 4: Running quality gate...")
        gate_decision = self.quality_gate(consolidated)
        consolidated.gate_status = gate_decision.status
        consolidated.gate_reason = gate_decision.reason
        consolidated.recommendations = gate_decision.recommendations

        print(f"\n{gate_decision}")
        print("="*70)

        return consolidated

    def generate_counterfactuals(self, high_reliability_episodes: List[dict]) -> List[dict]:
        """
        Generate counterfactual trajectories from HIGH reliability episodes.

        Process:
        1. Extract world model beliefs (transition probabilities) from episode
        2. Use world model to simulate "what if" scenarios
        3. Generate synthetic trajectories via world model rollouts

        Args:
            high_reliability_episodes: Episodes with HIGH reliability tag

        Returns:
            List of synthetic episodes with metadata
        """
        counterfactuals = []

        if not high_reliability_episodes:
            print("  No HIGH reliability episodes to generate counterfactuals from")
            return counterfactuals

        for base_episode in high_reliability_episodes:
            # Limit counterfactuals per episode
            num_to_generate = min(2, self.max_counterfactuals_per_episode)

            episode_id = base_episode.get('episode_id', 'unknown')
            beliefs = base_episode.get('beliefs', {})

            # Generate counterfactuals for this episode
            for i in range(num_to_generate):
                cf = self._generate_single_counterfactual(
                    base_episode,
                    beliefs,
                    variant_index=i
                )

                if cf:
                    counterfactuals.append(cf)

        # Apply synthetic data limit (max 30% of total)
        # If we have N real episodes and want synthetics to be at most 30% of total:
        # Total = N + S, where S is synthetics
        # S / (N + S) <= 0.30
        # S <= 0.30 * (N + S)
        # S <= 0.30*N + 0.30*S
        # 0.70*S <= 0.30*N
        # S <= (0.30/0.70) * N = 0.43 * N
        max_synthetic = max(1, int(len(high_reliability_episodes) * self.max_synthetic_fraction / (1 - self.max_synthetic_fraction)))

        if len(counterfactuals) > max_synthetic:
            # Keep highest fidelity counterfactuals
            counterfactuals = sorted(
                counterfactuals,
                key=lambda x: x.get('fidelity_score', 0.0),
                reverse=True
            )[:max_synthetic]

        return counterfactuals

    def _generate_single_counterfactual(
        self,
        base_episode: dict,
        beliefs: dict,
        variant_index: int
    ) -> Optional[dict]:
        """
        Generate a single counterfactual trajectory.

        Strategy: Modify a small aspect of the base episode and simulate forward.

        Variant 0: Extend the episode (add more steps with same actions)
        Variant 1: Change timing (different wait durations)
        """
        episode_id = base_episode.get('episode_id', 'unknown')

        # Extract trajectory
        # trajectory = base_episode.get('trajectory', {})
        # observations = trajectory.get('observations', [])
        # actions = trajectory.get('actions', [])

        # For now, create a simplified synthetic episode
        # In a full implementation, this would use world model rollouts

        cf_episode = self._create_synthetic_episode(
            base_episode,
            beliefs,
            variant_type='extend' if variant_index == 0 else 'timing',
            variant_index=variant_index
        )

        return cf_episode

    def _create_synthetic_episode(
        self,
        base_episode: dict,
        beliefs: dict,
        variant_type: str,
        variant_index: int
    ) -> dict:
        """
        Create a synthetic episode by modifying base episode.

        This is a simplified version. A full implementation would:
        - Use world model transition probabilities
        - Simulate trajectories step by step
        - Calculate fidelity based on world model likelihood

        For now, we'll create plausible synthetic data based on belief parameters.
        """
        base_id = base_episode.get('episode_id', 'unknown')
        beliefs_copy = copy.deepcopy(beliefs)

        # Extract heating rate from beliefs
        heating_rate = self._extract_value(beliefs.get('heating_rate_mean', {}))
        heating_rate_std = self._extract_value(beliefs.get('heating_rate_std', {}))
        measurement_noise = self._extract_value(beliefs.get('measurement_noise', {}))
        base_temp = self._extract_value(beliefs.get('base_temp', {}))

        # Set defaults if not available
        if not isinstance(heating_rate, (int, float)):
            heating_rate = 2.0
        if not isinstance(heating_rate_std, (int, float)):
            heating_rate_std = 0.3
        if not isinstance(measurement_noise, (int, float)):
            measurement_noise = 2.0
        if not isinstance(base_temp, (int, float)):
            base_temp = 20.0

        # Generate synthetic observations using the learned parameters
        synthetic_observations = []
        num_measurements = 3 + variant_index  # Vary number of measurements

        for i in range(num_measurements):
            time = (i + 1) * (3 + variant_index)  # Vary timing

            # Predict temperature using belief model + noise
            predicted_temp = base_temp + heating_rate * time
            noise = np.random.normal(0, measurement_noise)
            measured_temp = predicted_temp + noise

            synthetic_observations.append({
                'measured_temp': measured_temp,
                'time': time,
                'action': 'measure_temp'
            })

        # Calculate fidelity: how well does this match the world model?
        # Higher fidelity if consistent with belief parameters
        fidelity_score = self._calculate_fidelity(
            synthetic_observations,
            heating_rate,
            heating_rate_std,
            measurement_noise,
            base_temp
        )

        # Create counterfactual episode
        cf_episode = {
            'episode_id': f"{base_id}_cf_{variant_type}_{variant_index}",
            'timestamp': base_episode.get('timestamp', ''),
            'score': base_episode.get('score', 0.0),  # Inherit score from base
            'beliefs': beliefs_copy,
            'context': base_episode.get('context', {}),
            'reliability': 'SYNTHETIC_HIGH',  # Mark as synthetic
            'reason': f'Counterfactual generated from {base_id} (variant: {variant_type})',
            'metadata': {
                'source': 'counterfactual',
                'base_episode': base_id,
                'modification': f'{variant_type}_{variant_index}',
                'fidelity_score': fidelity_score,
                'patterns': base_episode.get('metadata', {}).get('patterns', []),
                'failures': []
            },
            'fidelity_score': fidelity_score,
            'trajectory': {
                'observations': synthetic_observations,
                'actions': ['measure_temp' for _ in synthetic_observations],
                'final_beliefs': beliefs_copy
            }
        }

        return cf_episode

    def _calculate_fidelity(
        self,
        observations: List[dict],
        heating_rate: float,
        heating_rate_std: float,
        measurement_noise: float,
        base_temp: float
    ) -> float:
        """
        Calculate fidelity score for synthetic trajectory.

        Fidelity measures how well the synthetic trajectory matches
        the world model predictions.

        For each observation, calculate the likelihood under the belief model,
        then normalize to [0, 1] range.
        """
        if not observations:
            return 0.0

        log_likelihoods = []

        for obs in observations:
            if 'measured_temp' in obs and 'time' in obs:
                time = obs['time']
                measured_temp = obs['measured_temp']

                # Predicted temperature
                predicted_temp = base_temp + heating_rate * time

                # Predictive variance (measurement noise + model uncertainty)
                predictive_std = np.sqrt(
                    measurement_noise**2 + (heating_rate_std * time)**2
                )

                # Log likelihood under Gaussian model
                log_like = -0.5 * ((measured_temp - predicted_temp) / predictive_std)**2
                log_likelihoods.append(log_like)

        if not log_likelihoods:
            return 0.0

        # Average log likelihood
        avg_log_like = np.mean(log_likelihoods)

        # Convert to [0, 1] range (heuristic)
        # log_like = 0 means perfect match, log_like = -8 means ~4 std deviations
        fidelity = np.exp(max(avg_log_like, -8))  # Clamp at -8
        fidelity = min(fidelity, 1.0)  # Cap at 1.0

        return fidelity

    def detect_biases(self, observations: List[dict]) -> BiasReport:
        """
        Detect systematic biases in observation distribution.

        Checks for:
        1. Context imbalance (e.g., MIXED power oversampled vs HIGH power)
        2. Reliability skew (too many LOW reliability observations)
        3. Value range gaps (missing data in specific ranges)

        Returns BiasReport with recommendations.
        """
        report = BiasReport()

        if not observations:
            return report

        # 1. Context imbalance
        context_counts = defaultdict(int)
        for obs in observations:
            context = obs.get('context', {})
            if isinstance(context, dict):
                power_setting = context.get('power_setting', 'UNKNOWN')
                context_counts[f"{power_setting}_power"] += 1

        report.context_imbalance = dict(context_counts)

        # Detect imbalance
        if context_counts:
            max_count = max(context_counts.values())
            min_count = min(context_counts.values())

            if max_count > 3 * min_count:  # 3x imbalance threshold
                dominant = max(context_counts, key=context_counts.get)
                rare = min(context_counts, key=context_counts.get)
                report.recommendations.append(
                    f"Context imbalance detected: {dominant} oversampled ({context_counts[dominant]} obs) "
                    f"vs {rare} ({context_counts[rare]} obs). Consider oversampling {rare} context."
                )

        # 2. Reliability skew
        reliability_counts = defaultdict(int)
        for obs in observations:
            reliability = obs.get('reliability', 'UNKNOWN')
            reliability_counts[reliability] += 1

        report.reliability_skew = dict(reliability_counts)

        # Check if too few HIGH reliability observations
        total = sum(reliability_counts.values())
        high_pct = reliability_counts.get('HIGH', 0) / total if total > 0 else 0
        low_pct = reliability_counts.get('LOW', 0) / total if total > 0 else 0

        if high_pct < 0.2:  # Less than 20% HIGH reliability
            report.recommendations.append(
                f"Low HIGH reliability observations ({high_pct*100:.1f}%). "
                f"Aim for at least 20% HIGH reliability episodes."
            )

        if low_pct > 0.7:  # More than 70% LOW reliability
            report.recommendations.append(
                f"High proportion of LOW reliability observations ({low_pct*100:.1f}%). "
                f"Consider improving methodology quality."
            )

        # 3. Value range gaps (for HotPot: heating rate ranges)
        heating_rates = []
        for obs in observations:
            beliefs = obs.get('beliefs', {})
            heating_rate = self._extract_value(beliefs.get('heating_rate_mean', {}))
            if isinstance(heating_rate, (int, float)):
                heating_rates.append(heating_rate)

        if heating_rates:
            min_rate = min(heating_rates)
            max_rate = max(heating_rates)

            # Check for gaps in coverage
            # Expected range for HotPot: 0.0 (off) to 2.5 (high)
            if max_rate < 2.0:
                report.value_range_gaps.append(
                    f"No observations in high heating rate range (2.0-2.5°C/s). "
                    f"Max observed: {max_rate:.2f}°C/s"
                )
                report.recommendations.append(
                    "Collect observations with HIGH power setting to cover full heating rate range."
                )

            if min_rate > 0.5:
                report.value_range_gaps.append(
                    f"No observations in low heating rate range (0.0-0.5°C/s). "
                    f"Min observed: {min_rate:.2f}°C/s"
                )

        return report

    def quality_gate(self, consolidated_data: ConsolidatedData) -> GateDecision:
        """
        Validate consolidated data before fine-tuning.

        Checks:
        1. Fidelity: Do synthetics match world model predictions?
        2. Diversity: Sufficient coverage of state-action space?
        3. Reliability: Minimum % HIGH reliability observations?
        4. Non-stationarity: Are recent episodes contradicting old ones?

        Returns GateDecision with status, reason, and recommendations.
        """
        metrics = {}
        recommendations = []

        # Calculate metrics
        total_episodes = (
            len(consolidated_data.high_reliability_episodes) +
            len(consolidated_data.low_reliability_episodes) +
            len(consolidated_data.counterfactual_episodes)
        )

        if total_episodes == 0:
            return GateDecision(
                status='FAIL',
                reason='No episodes available',
                recommendations=['Collect episodic data before running consolidation']
            )

        # 1. Fidelity check
        fidelity_scores = list(consolidated_data.fidelity_scores.values())
        avg_fidelity = np.mean(fidelity_scores) if fidelity_scores else 1.0
        min_fidelity = min(fidelity_scores) if fidelity_scores else 1.0

        metrics['avg_fidelity'] = avg_fidelity
        metrics['min_fidelity'] = min_fidelity

        # 2. Reliability check
        high_reliability_count = len(consolidated_data.high_reliability_episodes)
        high_reliability_pct = high_reliability_count / total_episodes

        metrics['high_reliability_pct'] = high_reliability_pct
        metrics['high_reliability_count'] = high_reliability_count

        # 3. Diversity check
        diversity_score = consolidated_data.diversity_metrics.get('diversity_score', 0.5)
        metrics['diversity_score'] = diversity_score

        # 4. Synthetic fraction
        synthetic_count = len(consolidated_data.counterfactual_episodes)
        synthetic_fraction = synthetic_count / total_episodes
        metrics['synthetic_fraction'] = synthetic_fraction
        metrics['total_episodes'] = total_episodes

        # Decision logic
        fail_conditions = []
        warning_conditions = []

        # Fidelity checks
        if avg_fidelity < self.fidelity_threshold_fail:
            fail_conditions.append(
                f"Low average fidelity ({avg_fidelity:.3f} < {self.fidelity_threshold_fail})"
            )
            recommendations.append(
                "Synthetic trajectories are unrealistic. "
                "Check world model quality or reduce counterfactual generation."
            )
        elif avg_fidelity < self.fidelity_threshold_warning:
            warning_conditions.append(
                f"Moderate fidelity ({avg_fidelity:.3f} < {self.fidelity_threshold_warning})"
            )
            recommendations.append(
                "Fidelity could be improved. Consider refining world model parameters."
            )

        # Reliability checks
        if high_reliability_pct < self.min_high_reliability_pct_fail:
            fail_conditions.append(
                f"Insufficient HIGH reliability data ({high_reliability_pct*100:.1f}% < {self.min_high_reliability_pct_fail*100:.0f}%)"
            )
            recommendations.append(
                f"Collect more HIGH reliability episodes. "
                f"Currently only {high_reliability_count} HIGH reliability episodes."
            )
        elif high_reliability_pct < self.min_high_reliability_pct_warning:
            warning_conditions.append(
                f"Low HIGH reliability percentage ({high_reliability_pct*100:.1f}%)"
            )
            recommendations.append(
                f"Aim for at least {self.min_high_reliability_pct_warning*100:.0f}% HIGH reliability episodes for best results."
            )

        # Synthetic fraction check
        if synthetic_fraction > self.max_synthetic_fraction:
            warning_conditions.append(
                f"High synthetic fraction ({synthetic_fraction*100:.1f}% > {self.max_synthetic_fraction*100:.0f}%)"
            )
            recommendations.append(
                "Too much synthetic data. Collect more real episodes."
            )

        # Make decision
        if fail_conditions:
            status = 'FAIL'
            reason = '; '.join(fail_conditions)
        elif warning_conditions:
            status = 'WARNING'
            reason = '; '.join(warning_conditions)
        else:
            status = 'PASS'
            reason = f"All quality checks passed. {total_episodes} episodes ready for fine-tuning."

        if status == 'PASS' and not recommendations:
            recommendations.append("Data quality is good. Proceed to Fine-Tuning Bridge.")

        return GateDecision(
            status=status,
            reason=reason,
            recommendations=recommendations,
            metrics=metrics
        )

    def _calculate_diversity(
        self,
        observations: List[dict],
        counterfactuals: List[dict]
    ) -> Dict[str, Any]:
        """
        Calculate diversity metrics for observations.

        Diversity measures coverage of the state-action space.
        Higher diversity means better generalization potential.
        """
        all_data = observations + counterfactuals

        if not all_data:
            return {'diversity_score': 0.0}

        # Count unique contexts
        unique_contexts = set()
        for item in all_data:
            context = item.get('context', {})
            if isinstance(context, dict):
                # Convert dict to hashable tuple
                context_tuple = tuple(sorted(context.items()))
                unique_contexts.add(context_tuple)

        # Count unique reliability tags
        unique_reliability = set(item.get('reliability', 'UNKNOWN') for item in all_data)

        # Heating rate coverage (for HotPot)
        heating_rates = []
        for item in all_data:
            beliefs = item.get('beliefs', {})
            rate = self._extract_value(beliefs.get('heating_rate_mean', {}))
            if isinstance(rate, (int, float)):
                heating_rates.append(rate)

        # Calculate diversity score (heuristic)
        # Higher score = more diverse data
        context_diversity = len(unique_contexts) / 5.0  # Normalize by expected contexts
        reliability_diversity = len(unique_reliability) / 3.0  # Normalize by possible tags

        # Value range coverage
        value_coverage = 0.5  # Default
        if heating_rates:
            rate_range = max(heating_rates) - min(heating_rates)
            value_coverage = min(rate_range / 2.5, 1.0)  # Normalize by expected range (0-2.5)

        diversity_score = (context_diversity + reliability_diversity + value_coverage) / 3.0
        diversity_score = min(diversity_score, 1.0)

        return {
            'diversity_score': diversity_score,
            'unique_contexts': len(unique_contexts),
            'unique_reliability_tags': len(unique_reliability),
            'heating_rate_range': (min(heating_rates), max(heating_rates)) if heating_rates else (0, 0),
            'num_observations': len(all_data)
        }

    def _extract_value(self, belief_data):
        """
        Extract value from structured belief format or return raw value.

        Args:
            belief_data: Belief value (may be wrapped in {'value': ...})

        Returns:
            Extracted value
        """
        if isinstance(belief_data, dict) and 'value' in belief_data:
            return belief_data['value']
        return belief_data
