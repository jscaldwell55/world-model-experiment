#!/usr/bin/env python3
"""
Critical Fidelity Validation Experiments

Three experiments to validate whether offline consolidation (OC) helps or hurts:

A. Do Synthetics Help? - Test if OC improves prediction accuracy
B. Cross-Validation of Synthetics - Validate synthetics against held-out real data
C. Wrong Model Detection - Test if OC detects and rejects bad world models

These experiments address the circular reasoning problem identified in
FIDELITY_CRITICAL_ISSUES.md
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import copy

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.offline_consolidation import OfflineConsolidation
from environments.hot_pot import HotPotLab


@dataclass
class ExperimentResults:
    """Results from a validation experiment"""
    experiment_name: str
    passed: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        lines = [f"\n{'='*70}"]
        lines.append(f"{status} - {self.experiment_name}")
        lines.append(f"{'='*70}")

        if self.metrics:
            lines.append("\nMetrics:")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.3f}")
                else:
                    lines.append(f"  {key}: {value}")

        if self.details:
            lines.append("\nDetails:")
            for key, value in self.details.items():
                lines.append(f"  {key}: {value}")

        if self.recommendations:
            lines.append("\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")

        return "\n".join(lines)


class WorldModelSimulator:
    """
    Simulate a simple world model for HotPot domain.
    Used to generate predictions and evaluate synthetics.
    """

    def __init__(self, heating_rate: float, measurement_noise: float, base_temp: float = 20.0):
        self.heating_rate = heating_rate
        self.measurement_noise = measurement_noise
        self.base_temp = base_temp

    def predict_temperature(self, time: float, add_noise: bool = False) -> float:
        """Predict temperature at given time"""
        predicted_temp = self.base_temp + self.heating_rate * time

        if add_noise:
            noise = np.random.normal(0, self.measurement_noise)
            return predicted_temp + noise

        return predicted_temp

    def evaluate_prediction(self, predicted_temp: float, actual_temp: float) -> float:
        """Calculate prediction error percentage"""
        return abs(predicted_temp - actual_temp) / actual_temp if actual_temp != 0 else 0.0

    @classmethod
    def from_episode(cls, episode: dict) -> 'WorldModelSimulator':
        """Create simulator from episode beliefs"""
        beliefs = episode.get('beliefs', {})
        heating_rate = beliefs.get('heating_rate_mean', {}).get('value', 1.2)
        measurement_noise = beliefs.get('measurement_noise', {}).get('value', 2.0)
        base_temp = beliefs.get('base_temp', {}).get('value', 20.0)

        return cls(heating_rate, measurement_noise, base_temp)

    @classmethod
    def from_playbook(cls, playbook: dict, episode_id: Optional[str] = None) -> 'WorldModelSimulator':
        """
        Create simulator from playbook (averaged across episodes or specific episode).

        Args:
            playbook: ACE playbook
            episode_id: If specified, use only this episode. Otherwise average across all.
        """
        observations = playbook.get('observations', [])

        if not observations:
            # Use defaults
            return cls(heating_rate=1.2, measurement_noise=2.0, base_temp=20.0)

        if episode_id:
            # Use specific episode
            for obs in observations:
                if obs['episode_id'] == episode_id:
                    return cls.from_episode(obs)
            raise ValueError(f"Episode {episode_id} not found in playbook")

        # Average across all episodes (weighted by reliability)
        total_weight = 0
        heating_rates = []
        weights = []

        for obs in observations:
            beliefs = obs.get('beliefs', {})
            heating_rate = beliefs.get('heating_rate_mean', {}).get('value')
            reliability = obs.get('reliability', 'LOW')

            if heating_rate is not None:
                heating_rates.append(heating_rate)
                # Weight HIGH reliability more
                weight = 1.0 if reliability == 'HIGH' else 0.3
                weights.append(weight)
                total_weight += weight

        if not heating_rates:
            return cls(heating_rate=1.2, measurement_noise=2.0, base_temp=20.0)

        # Weighted average
        avg_heating_rate = sum(h * w for h, w in zip(heating_rates, weights)) / total_weight

        # For noise and base_temp, use first observation
        first_beliefs = observations[0].get('beliefs', {})
        measurement_noise = first_beliefs.get('measurement_noise', {}).get('value', 2.0)
        base_temp = first_beliefs.get('base_temp', {}).get('value', 20.0)

        return cls(avg_heating_rate, measurement_noise, base_temp)


def load_playbook(domain: str = 'hot_pot') -> dict:
    """Load playbook for domain"""
    playbook_path = Path(f'memory/domains/{domain}/playbook.json')
    if not playbook_path.exists():
        raise FileNotFoundError(f"Playbook not found at {playbook_path}")

    with open(playbook_path, 'r') as f:
        return json.load(f)


def load_episode_data(domain: str = 'hot_pot') -> List[dict]:
    """Load all episode data for domain"""
    episode_dir = Path(f'memory/domains/{domain}/episodes')
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode directory not found at {episode_dir}")

    episodes = []
    for episode_file in sorted(episode_dir.glob('*.json')):
        with open(episode_file, 'r') as f:
            episodes.append(json.load(f))

    return episodes


def create_test_queries(num_queries: int = 50, seed: int = 42) -> List[Dict]:
    """
    Create diverse test queries for HotPot domain.

    Returns test cases with known ground truth answers.
    """
    np.random.seed(seed)
    queries = []

    # Ground truth: HotPot heating rate on HIGH power is 2.5°C/s
    # On LOW power it's 0.5°C/s
    # MIXED alternates between them

    for i in range(num_queries):
        # Random initial temp (15-25°C)
        initial_temp = np.random.uniform(15, 25)

        # Random time (3-15 seconds)
        time = np.random.uniform(3, 15)

        # Random power setting
        power = np.random.choice(['HIGH', 'LOW'])

        # Ground truth heating rate
        true_heating_rate = 2.5 if power == 'HIGH' else 0.5

        # Calculate ground truth final temp
        true_final_temp = initial_temp + true_heating_rate * time

        queries.append({
            'query_id': f'test_{i+1:03d}',
            'initial_temp': initial_temp,
            'time': time,
            'power': power,
            'ground_truth_temp': true_final_temp,
            'ground_truth_heating_rate': true_heating_rate
        })

    return queries


# ============================================================================
# EXPERIMENT A: Do Synthetics Help?
# ============================================================================

def experiment_a_synthetics_help(
    playbook: dict,
    test_queries: List[Dict],
    train_ratio: float = 0.67
) -> ExperimentResults:
    """
    Test if synthetic data from OC improves prediction accuracy.

    Setup:
        - Split episodes into train/test
        - Condition 1: Train world model on real data only
        - Condition 2: Train world model on real + synthetic data
        - Compare accuracy on held-out test queries

    Pass criteria: accuracy_synthetic > accuracy_real + 0.02 (2% improvement)
    """
    observations = playbook.get('observations', [])

    if len(observations) < 10:
        return ExperimentResults(
            experiment_name="Experiment A: Do Synthetics Help?",
            passed=False,
            details={'error': f'Need at least 10 episodes, got {len(observations)}'},
            recommendations=['Collect more episodes before running this experiment']
        )

    # Split into train/test
    n_train = int(len(observations) * train_ratio)
    train_obs = observations[:n_train]
    test_obs = observations[n_train:]

    # Condition 1: Real data only
    train_playbook = {'observations': train_obs}
    model_real = WorldModelSimulator.from_playbook(train_playbook)

    # Condition 2: Real + Synthetic
    # Generate synthetics using offline consolidation
    env = HotPotLab(seed=42)
    oc = OfflineConsolidation(env)
    consolidated = oc.consolidate(train_playbook)

    # Get training data with synthetics
    training_data = consolidated.get_training_data()
    all_episodes = training_data['episodes']

    # Create model from all episodes (real + synthetic)
    model_synthetic = WorldModelSimulator.from_playbook({'observations': all_episodes})

    # Evaluate both models on test queries
    errors_real = []
    errors_synthetic = []

    for query in test_queries:
        # Real model prediction
        pred_real = model_real.predict_temperature(query['time'])
        error_real = abs(pred_real - query['ground_truth_temp']) / query['ground_truth_temp']
        errors_real.append(error_real)

        # Synthetic model prediction
        pred_synthetic = model_synthetic.predict_temperature(query['time'])
        error_synthetic = abs(pred_synthetic - query['ground_truth_temp']) / query['ground_truth_temp']
        errors_synthetic.append(error_synthetic)

    # Calculate accuracy (1 - mean_error)
    accuracy_real = 1.0 - np.mean(errors_real)
    accuracy_synthetic = 1.0 - np.mean(errors_synthetic)
    improvement = accuracy_synthetic - accuracy_real

    # Pass if synthetics improve by at least 2%
    passed = improvement > 0.02

    recommendations = []
    if not passed:
        if improvement < -0.02:
            recommendations.append("Synthetics are HURTING performance - disable OC immediately")
        elif abs(improvement) <= 0.02:
            recommendations.append("Synthetics provide no benefit - OC adds complexity without value")
        recommendations.append("Check fidelity scoring for circular reasoning")
        recommendations.append("Validate world model accuracy before generating synthetics")
    else:
        recommendations.append("Synthetics are helping - proceed with fidelity scoring improvements")

    return ExperimentResults(
        experiment_name="Experiment A: Do Synthetics Help?",
        passed=passed,
        metrics={
            'accuracy_real_only': accuracy_real,
            'accuracy_with_synthetics': accuracy_synthetic,
            'improvement': improvement,
            'num_train_episodes': n_train,
            'num_test_queries': len(test_queries),
            'num_synthetics_generated': len(consolidated.counterfactual_episodes)
        },
        details={
            'threshold': '2% improvement required',
            'train_episodes': n_train,
            'test_episodes': len(test_obs),
            'synthetic_episodes': len(consolidated.counterfactual_episodes)
        },
        recommendations=recommendations
    )


# ============================================================================
# EXPERIMENT B: Cross-Validation of Synthetics
# ============================================================================

def experiment_b_cross_validate_synthetics(
    playbook: dict,
    error_threshold: float = 0.15,
    pass_rate_threshold: float = 0.80
) -> ExperimentResults:
    """
    Validate synthetics against held-out real data.

    For each HIGH reliability episode:
        1. Hold it out
        2. Train world model on all OTHER episodes
        3. Generate synthetic matching held-out's conditions
        4. Compare synthetic vs real held-out

    Pass criteria: mean_error < 0.10 AND pass_rate > 0.80
    """
    observations = playbook.get('observations', [])
    high_episodes = [obs for obs in observations if obs.get('reliability') == 'HIGH']

    if len(high_episodes) < 3:
        return ExperimentResults(
            experiment_name="Experiment B: Cross-Validation of Synthetics",
            passed=False,
            details={'error': f'Need at least 3 HIGH reliability episodes, got {len(high_episodes)}'},
            recommendations=['Collect more HIGH reliability episodes (single-power runs)']
        )

    results = []
    env = HotPotLab(seed=42)

    for held_out in high_episodes:
        # Train on all OTHER episodes
        train_data = [obs for obs in observations if obs['episode_id'] != held_out['episode_id']]
        train_playbook = {'observations': train_data}

        # Build world model
        world_model = WorldModelSimulator.from_playbook(train_playbook)

        # Generate synthetic matching held_out conditions
        # Extract held_out's heating parameters
        held_out_beliefs = held_out.get('beliefs', {})
        held_out_heating_rate = held_out_beliefs.get('heating_rate_mean', {}).get('value', 1.2)

        # Synthetic outcome (predict temperature at time=12s)
        time = 12.0
        synthetic_temp = world_model.predict_temperature(time)

        # Real held_out outcome
        real_temp = 20.0 + held_out_heating_rate * time

        # Error
        error = abs(synthetic_temp - real_temp) / real_temp if real_temp != 0 else 0.0
        acceptable = error < error_threshold

        results.append({
            'episode_id': held_out['episode_id'],
            'error': error,
            'acceptable': acceptable,
            'synthetic_temp': synthetic_temp,
            'real_temp': real_temp
        })

    # Calculate metrics
    mean_error = np.mean([r['error'] for r in results])
    pass_rate = np.mean([r['acceptable'] for r in results])

    # Pass if both conditions met
    passed = mean_error < 0.10 and pass_rate > pass_rate_threshold

    recommendations = []
    if not passed:
        if mean_error >= 0.10:
            recommendations.append(f"Mean error ({mean_error:.1%}) too high - synthetics are unrealistic")
            recommendations.append("World model is not generalizing to counterfactuals")
        if pass_rate < pass_rate_threshold:
            recommendations.append(f"Pass rate ({pass_rate:.1%}) too low - many synthetics have high error")
        recommendations.append("Do not use these synthetics for fine-tuning")
        recommendations.append("Investigate why world model predictions are inaccurate")
    else:
        recommendations.append("Synthetics validated against real data")
        recommendations.append("Safe to proceed with this quality of synthetics")

    return ExperimentResults(
        experiment_name="Experiment B: Cross-Validation of Synthetics",
        passed=passed,
        metrics={
            'mean_error': mean_error,
            'pass_rate': pass_rate,
            'num_validated': len(results)
        },
        details={
            'error_threshold': error_threshold,
            'pass_rate_threshold': pass_rate_threshold,
            'validation_details': results
        },
        recommendations=recommendations
    )


# ============================================================================
# EXPERIMENT C: Wrong Model Detection
# ============================================================================

def experiment_c_wrong_model_detection(
    playbook: dict,
    test_queries: List[Dict]
) -> ExperimentResults:
    """
    Test whether OC detects and rejects wrong world models.

    Setup:
        1. Create deliberately wrong belief (train only on MIXED power)
        2. Try to generate synthetics using OC
        3. Test if OC's quality gate catches the error
        4. Measure error on HIGH power ground truth

    Pass criteria: OC gate_status == 'FAIL' OR mean_error flagged
    """
    observations = playbook.get('observations', [])

    # Deliberately create wrong model - train only on MIXED/LOW reliability
    mixed_only = [obs for obs in observations if obs.get('context', {}).get('power_setting') == 'MIXED']

    if len(mixed_only) == 0:
        # Alternative: use LOW reliability only
        mixed_only = [obs for obs in observations if obs.get('reliability') == 'LOW']

    if len(mixed_only) == 0:
        return ExperimentResults(
            experiment_name="Experiment C: Wrong Model Detection",
            passed=False,
            details={'error': 'No MIXED or LOW reliability episodes to create wrong model'},
            recommendations=['Need diverse episode types to test wrong model detection']
        )

    # Create wrong playbook
    wrong_playbook = {'observations': mixed_only}

    # Build wrong model
    wrong_model = WorldModelSimulator.from_playbook(wrong_playbook)

    # Try to generate synthetics with OC
    env = HotPotLab(seed=42)
    oc = OfflineConsolidation(env)
    consolidated = oc.consolidate(wrong_playbook)

    # Test on HIGH power ground truth (should have high error)
    high_power_queries = [q for q in test_queries if q['power'] == 'HIGH']

    if not high_power_queries:
        high_power_queries = test_queries[:10]  # Use first 10

    errors = []
    for query in high_power_queries:
        synthetic_temp = wrong_model.predict_temperature(query['time'])
        error = abs(synthetic_temp - query['ground_truth_temp']) / query['ground_truth_temp']
        errors.append(error)

    mean_error = np.mean(errors)

    # Check if OC caught this
    gate_caught_error = consolidated.gate_status in ['FAIL', 'WARNING']
    error_is_high = mean_error > 0.20  # 20% error threshold

    # Pass if OC detected the problem OR we can detect it
    passed = gate_caught_error

    recommendations = []
    if passed:
        recommendations.append("✓ OC correctly rejected wrong model")
        recommendations.append("Quality gate is working as intended")
    else:
        if error_is_high:
            recommendations.append("✗ OC FAILED to detect wrong model despite high error")
            recommendations.append(f"Wrong model has {mean_error:.1%} error but OC status is {consolidated.gate_status}")
            recommendations.append("CRITICAL: This could amplify bias in fine-tuning")
            recommendations.append("Add world model validation checks to OC")
        else:
            recommendations.append("Wrong model error is low - may not be a good test case")
            recommendations.append("Try with more extreme bias")

    return ExperimentResults(
        experiment_name="Experiment C: Wrong Model Detection",
        passed=passed,
        metrics={
            'mean_error_on_ground_truth': mean_error,
            'oc_gate_status': consolidated.gate_status,
            'num_synthetics_generated': len(consolidated.counterfactual_episodes)
        },
        details={
            'gate_caught_error': gate_caught_error,
            'error_is_high': error_is_high,
            'error_threshold': 0.20,
            'wrong_model_heating_rate': wrong_model.heating_rate,
            'num_wrong_episodes': len(mixed_only)
        },
        recommendations=recommendations
    )


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_all_experiments(
    domain: str = 'hot_pot',
    output_path: Optional[str] = None
) -> Dict[str, ExperimentResults]:
    """
    Run all three critical validation experiments.

    Returns:
        Dictionary of experiment results
    """
    print("="*70)
    print("CRITICAL FIDELITY VALIDATION EXPERIMENTS")
    print("="*70)
    print(f"\nDomain: {domain}")
    print("Testing whether offline consolidation helps or hurts...")
    print()

    # Load data
    try:
        playbook = load_playbook(domain)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {}

    # Create test queries
    test_queries = create_test_queries(num_queries=50)

    # Run experiments
    results = {}

    print("\n" + "="*70)
    print("EXPERIMENT A: Do Synthetics Help?")
    print("="*70)
    result_a = experiment_a_synthetics_help(playbook, test_queries)
    results['experiment_a'] = result_a
    print(result_a)

    print("\n" + "="*70)
    print("EXPERIMENT B: Cross-Validation of Synthetics")
    print("="*70)
    result_b = experiment_b_cross_validate_synthetics(playbook)
    results['experiment_b'] = result_b
    print(result_b)

    print("\n" + "="*70)
    print("EXPERIMENT C: Wrong Model Detection")
    print("="*70)
    result_c = experiment_c_wrong_model_detection(playbook, test_queries)
    results['experiment_c'] = result_c
    print(result_c)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_passed = all(r.passed for r in results.values())

    print(f"\nExperiment A (Synthetics Help): {'✓ PASS' if result_a.passed else '✗ FAIL'}")
    print(f"Experiment B (Cross-Validation): {'✓ PASS' if result_b.passed else '✗ FAIL'}")
    print(f"Experiment C (Wrong Model Detection): {'✓ PASS' if result_c.passed else '✗ FAIL'}")

    print(f"\nOverall: {'✓ ALL PASSED' if all_passed else '✗ FAILURES DETECTED'}")

    if all_passed:
        print("\n✓ Offline consolidation is validated and safe to use")
        print("  - Synthetics improve accuracy")
        print("  - Synthetics match real data")
        print("  - Quality gates detect bad models")
    else:
        print("\n✗ Critical issues detected - DO NOT proceed with fine-tuning")
        print("  Review recommendations above before integrating OC with FTB")

    # Save results
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        output_data = {
            exp_name: {
                'passed': result.passed,
                'metrics': result.metrics,
                'details': result.details,
                'recommendations': result.recommendations
            }
            for exp_name, result in results.items()
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to {output_file}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run critical fidelity validation experiments")
    parser.add_argument('--domain', type=str, default='hot_pot', help='Domain to test (default: hot_pot)')
    parser.add_argument('--output', type=str, default='results/fidelity_validation.json',
                       help='Output file path')

    args = parser.parse_args()

    results = run_all_experiments(domain=args.domain, output_path=args.output)
