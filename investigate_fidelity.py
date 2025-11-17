#!/usr/bin/env python3
"""
Investigation: Synthetic Fidelity and World Model Correctness

Three critical questions:
1. What's the actual calculation for fidelity? Is it just likelihood under the world model?
2. Have you validated that high-fidelity synthetics actually help downstream?
3. What if the world model is wrong? High fidelity to a wrong model = systematic bias
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path.cwd()))

from memory.offline_consolidation import OfflineConsolidation
from environments.hot_pot import HotPotLab


def question_1_fidelity_calculation():
    """
    Q1: What's the actual calculation for fidelity?

    Answer: YES, it's just likelihood under the belief model.
    But there's a critical issue...
    """
    print("="*70)
    print("QUESTION 1: What's the Fidelity Calculation?")
    print("="*70)

    print("\n📊 FIDELITY FORMULA:")
    print("-" * 70)
    print("""
For each synthetic observation (measured_temp, time):

    1. Predicted temperature:
       predicted_temp = base_temp + heating_rate * time

    2. Predictive uncertainty:
       predictive_std = sqrt(measurement_noise² + (heating_rate_std * time)²)

    3. Log likelihood (Gaussian):
       log_like = -0.5 * ((measured_temp - predicted_temp) / predictive_std)²

    4. Average over all observations:
       avg_log_like = mean(log_likelihoods)

    5. Convert to [0, 1] range:
       fidelity = exp(max(avg_log_like, -8))
       fidelity = min(fidelity, 1.0)
    """)

    print("\n⚠️  CRITICAL ISSUE:")
    print("-" * 70)
    print("""
The synthetic observations are GENERATED using the SAME model they're scored against!

    Generation (line 395-397):
        predicted_temp = base_temp + heating_rate * time
        noise = np.random.normal(0, measurement_noise)
        measured_temp = predicted_temp + noise

    Fidelity Scoring (line 470-478):
        predicted_temp = base_temp + heating_rate * time  # SAME FORMULA!
        predictive_std = sqrt(measurement_noise² + ...)
        log_like = -0.5 * ((measured_temp - predicted_temp) / predictive_std)²

    This is CIRCULAR!
    Synthetics will ALWAYS have high fidelity because they're drawn from
    the same distribution they're evaluated against.

    Fidelity scores of 0.888, 0.918 are EXPECTED, not validating quality!
    """)


def question_2_downstream_validation():
    """
    Q2: Have you validated that high-fidelity synthetics actually help downstream?

    Answer: NO - this is a critical gap that needs immediate testing.
    """
    print("\n" + "="*70)
    print("QUESTION 2: Do High-Fidelity Synthetics Help Downstream?")
    print("="*70)

    print("\n❌ STATUS: NOT VALIDATED")
    print("-" * 70)
    print("""
We have NOT tested whether synthetic data improves:
    1. Prediction accuracy on held-out test questions
    2. Generalization to new scenarios
    3. Robustness to distribution shifts

This is a CRITICAL GAP before claiming OC provides value.
    """)

    print("\n🧪 PROPOSED VALIDATION EXPERIMENTS:")
    print("-" * 70)
    print("""
Experiment A: Prediction Accuracy
    - Baseline: Train on 5 real episodes (current hot_pot data)
    - OC: Train on 5 real + 1 synthetic episode
    - Metric: Accuracy on 10 held-out test questions
    - Expected: If synthetics help, OC > Baseline

Experiment B: Data Efficiency
    - Baseline: Train on N real episodes
    - OC: Train on N/2 real + N/2 synthetic episodes
    - Metric: Accuracy on test set
    - Expected: OC ≈ Baseline (same total data, half collection cost)

Experiment C: Distribution Robustness
    - Baseline: Train on MIXED power episodes (biased)
    - OC: Train on MIXED + synthetic HIGH power episodes
    - Metric: Accuracy on HIGH power test questions
    - Expected: OC > Baseline on undersampled contexts

Experiment D: Wrong Model Detection
    - Setup: Inject systematic bias in learned beliefs
    - Baseline: Train on biased real data
    - OC: Train on biased real + high-fidelity synthetics (to wrong model)
    - Metric: Accuracy on test set
    - Expected: OC < Baseline (amplified bias)
    """)

    print("\n⚠️  RECOMMENDATION:")
    print("-" * 70)
    print("""
DO NOT proceed to fine-tuning bridge until Experiments A-D are run.

If synthetics DON'T help (or hurt), then OC is adding complexity without value.
    """)


def question_3_wrong_world_model():
    """
    Q3: What if the world model is wrong?

    This is the MOST CRITICAL question.
    High fidelity to a wrong model = systematic bias amplification.
    """
    print("\n" + "="*70)
    print("QUESTION 3: What if the World Model is Wrong?")
    print("="*70)

    # Load real playbook to check learned beliefs vs ground truth
    playbook_path = Path('memory/domains/hot_pot/playbook.json')

    if not playbook_path.exists():
        print("\n⚠️  No playbook found - cannot check world model accuracy")
        return

    with open(playbook_path, 'r') as f:
        playbook = json.load(f)

    print("\n🎯 GROUND TRUTH (from HotPotLab environment):")
    print("-" * 70)
    print(f"  heating_rate (HIGH power): 2.5°C/s")
    print(f"  heating_rate (LOW power):  1.0°C/s")
    print(f"  heating_rate (OFF):        0.0°C/s")
    print(f"  measurement_noise:         2.0°C")
    print(f"  base_temp:                 20.0°C")

    print("\n📊 LEARNED BELIEFS (from ACE playbook):")
    print("-" * 70)

    observations = playbook.get('observations', [])

    if not observations:
        print("  No observations found")
        return

    # Analyze learned beliefs vs ground truth
    errors = []

    for obs in observations:
        episode_id = obs.get('episode_id', 'unknown')
        beliefs = obs.get('beliefs', {})
        context = obs.get('context', {})
        reliability = obs.get('reliability', 'UNKNOWN')

        # Extract learned heating rate
        heating_rate_mean = beliefs.get('heating_rate_mean', {})
        if isinstance(heating_rate_mean, dict):
            learned_rate = heating_rate_mean.get('value', None)
        else:
            learned_rate = heating_rate_mean

        if learned_rate is None:
            continue

        # Determine expected ground truth based on context
        power_setting = context.get('power_setting', 'UNKNOWN')

        if power_setting == 'HIGH':
            expected_rate = 2.5
        elif power_setting == 'LOW':
            expected_rate = 1.0
        elif power_setting == 'MIXED':
            expected_rate = None  # Can't determine - averaged across settings
        else:
            expected_rate = None

        print(f"\n  Episode: {episode_id}")
        print(f"    Power setting: {power_setting}")
        print(f"    Reliability: {reliability}")
        print(f"    Learned heating_rate: {learned_rate:.2f}°C/s")

        if expected_rate is not None:
            error = abs(learned_rate - expected_rate)
            error_pct = (error / expected_rate) * 100
            errors.append({
                'episode': episode_id,
                'learned': learned_rate,
                'expected': expected_rate,
                'error': error,
                'error_pct': error_pct,
                'reliability': reliability
            })

            print(f"    Expected (ground truth): {expected_rate:.2f}°C/s")
            print(f"    Error: {error:.2f}°C/s ({error_pct:.1f}%)")

            if error_pct > 20:
                print(f"    ⚠️  WARNING: >20% error - world model may be inaccurate")
        else:
            print(f"    Expected: Unknown (MIXED power setting)")
            print(f"    ⚠️  WARNING: Averaged across contexts - unreliable belief")

    # Summary of errors
    if errors:
        print("\n" + "-" * 70)
        print("WORLD MODEL ACCURACY SUMMARY:")
        print("-" * 70)

        high_rel_errors = [e for e in errors if e['reliability'] == 'HIGH']
        low_rel_errors = [e for e in errors if e['reliability'] == 'LOW']

        if high_rel_errors:
            avg_error = np.mean([e['error_pct'] for e in high_rel_errors])
            print(f"\nHIGH reliability episodes:")
            print(f"  Average error: {avg_error:.1f}%")

            if avg_error > 20:
                print(f"  ⚠️  CRITICAL: Even HIGH reliability has >20% error!")
                print(f"     Generating synthetics will AMPLIFY this bias!")

        if low_rel_errors:
            avg_error = np.mean([e['error_pct'] for e in low_rel_errors])
            print(f"\nLOW reliability episodes:")
            print(f"  Average error: {avg_error:.1f}%")
            print(f"  (Expected - these have methodology issues)")

    print("\n" + "="*70)
    print("SYSTEMATIC BIAS RISK ANALYSIS")
    print("="*70)
    print("""
Scenario: World model learns heating_rate = 1.2°C/s from MIXED power episodes
    Ground truth (HIGH power): 2.5°C/s
    Error: 52% underestimate

If we generate high-fidelity synthetics:
    1. Synthetics will use heating_rate = 1.2°C/s
    2. They'll have high fidelity (0.9+) to the wrong model
    3. Fine-tuning on these synthetics reinforces the wrong belief
    4. Agent learns to predict 1.2°C/s instead of 2.5°C/s
    5. Test accuracy DECREASES instead of improves

This is SYSTEMATIC BIAS AMPLIFICATION.

Current OC system has NO safeguards against this!
    """)

    print("\n🛡️  PROPOSED SAFEGUARDS:")
    print("-" * 70)
    print("""
1. World Model Validation:
   - Compare learned beliefs to ground truth (when available)
   - Flag beliefs with >20% error
   - Don't generate synthetics from inaccurate beliefs

2. Cross-Validation:
   - Hold out episodes for validation
   - Generate synthetics from train set beliefs
   - Test on held-out real episodes
   - If synthetics hurt validation accuracy, REJECT them

3. Diversity Requirements:
   - Don't generate synthetics unless we have HIGH reliability
     observations across MULTIPLE contexts
   - Prevents averaging bias (MIXED power issue)

4. Fidelity Calibration:
   - Current fidelity is circular (always high)
   - Need: fidelity = P(synthetic | real_world) not P(synthetic | belief)
   - Use held-out real data to calibrate

5. Conservative Generation:
   - Only generate from HIGH reliability + LOW error beliefs
   - Limit synthetic fraction (10% not 30%)
   - Weight synthetics lower (0.5 not 0.8)
    """)


def demonstrate_bias_amplification():
    """
    Concrete example of how wrong world model + high fidelity = disaster
    """
    print("\n" + "="*70)
    print("DEMONSTRATION: Bias Amplification in Action")
    print("="*70)

    print("\n📊 Setup:")
    print("-" * 70)
    print("""
Real data (from MIXED power episodes):
    - 4 episodes with power toggles
    - Learned heating_rate = 1.2°C/s (averaged across HIGH/LOW)
    - Ground truth (HIGH): 2.5°C/s, (LOW): 1.0°C/s
    - World model is WRONG (52% error on HIGH power)
    """)

    # Simulate generating synthetics from wrong model
    print("\n🧪 Generating synthetics from wrong model:")
    print("-" * 70)

    wrong_heating_rate = 1.2  # Learned from MIXED episodes
    true_heating_rate_high = 2.5  # Ground truth for HIGH power
    measurement_noise = 2.0
    base_temp = 20.0

    # Generate synthetic observations
    np.random.seed(42)
    synthetic_temps = []

    for time in [3, 6, 9, 12]:
        predicted_temp = base_temp + wrong_heating_rate * time
        noise = np.random.normal(0, measurement_noise)
        measured_temp = predicted_temp + noise
        synthetic_temps.append((time, measured_temp))

        # What it SHOULD be (ground truth)
        true_temp = base_temp + true_heating_rate_high * time

        print(f"  Time {time}s:")
        print(f"    Synthetic: {measured_temp:.1f}°C (from wrong model)")
        print(f"    Ground truth: {true_temp:.1f}°C")
        print(f"    Bias: {measured_temp - true_temp:.1f}°C ({((measured_temp - true_temp)/true_temp)*100:.1f}%)")

    # Calculate fidelity (will be high because it's circular)
    log_likes = []
    for time, measured_temp in synthetic_temps:
        predicted_temp = base_temp + wrong_heating_rate * time
        predictive_std = measurement_noise
        log_like = -0.5 * ((measured_temp - predicted_temp) / predictive_std)**2
        log_likes.append(log_like)

    avg_log_like = np.mean(log_likes)
    fidelity = np.exp(max(avg_log_like, -8))

    print(f"\n📈 Fidelity Score: {fidelity:.3f}")
    print(f"   ⚠️  HIGH fidelity despite systematic bias!")

    print("\n" + "-" * 70)
    print("Impact on downstream learning:")
    print("-" * 70)
    print(f"""
Training data composition:
    - 4 real episodes (heating_rate ≈ 1.2°C/s, LOW reliability)
    - 1 synthetic episode (heating_rate = 1.2°C/s, HIGH fidelity)

Weighted training:
    - Real: 4 episodes × 0.3 weight = 1.2 effective episodes
    - Synthetic: 1 episode × 0.8 weight = 0.8 effective episodes

Total effective data: 2.0 episodes, ALL biased toward 1.2°C/s

Expected learned rate after fine-tuning:
    (1.2 * 1.2 + 1.2 * 0.8) / 2.0 = 1.2°C/s

Ground truth: 2.5°C/s
Error: 52% underestimate

WITHOUT synthetics:
    - Just 4 real episodes with 0.3 weight
    - Model might learn to be uncertain
    - Might default to prior

WITH synthetics:
    - High-fidelity synthetic REINFORCES the wrong belief
    - Model becomes MORE CONFIDENT in 1.2°C/s
    - Test accuracy DECREASES

This is WORSE than doing nothing!
    """)


def main():
    """Run all investigations"""
    print("\n" + "="*70)
    print(" " * 10 + "FIDELITY AND WORLD MODEL INVESTIGATION")
    print("="*70)

    # Question 1
    question_1_fidelity_calculation()

    # Question 2
    question_2_downstream_validation()

    # Question 3
    question_3_wrong_world_model()

    # Demonstration
    demonstrate_bias_amplification()

    print("\n" + "="*70)
    print("SUMMARY: CRITICAL ISSUES IDENTIFIED")
    print("="*70)
    print("""
1. Fidelity is CIRCULAR ✗
   - Synthetics scored against same model that generated them
   - High fidelity is EXPECTED, not validating quality
   - Need: fidelity = P(synthetic | real_world)

2. No Downstream Validation ✗
   - Haven't tested if synthetics actually help
   - Critical experiments needed before FTB integration
   - Risk: Adding complexity without value

3. Systematic Bias Risk ✗
   - Wrong world model + high fidelity = amplified bias
   - Current data has LOW reliability (MIXED power averaging)
   - Generating synthetics will REINFORCE wrong beliefs
   - Need: safeguards against bias amplification

RECOMMENDATION:
   PAUSE OC → FTB integration until these issues are addressed.

Priority fixes:
   1. Run Experiments A-D to validate synthetic value
   2. Add world model validation (beliefs vs ground truth)
   3. Implement cross-validation for synthetic quality
   4. Add conservative generation limits (HIGH reliability only)
   5. Recalibrate fidelity scoring against real data
    """)

    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
