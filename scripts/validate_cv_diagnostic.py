#!/usr/bin/env python3
"""
Cross-Validation Implementation Diagnostic

Validates that CV is actually testing on held-out data and not leaking information.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.ace_playbook import ACEPlaybook
from utils.offline_consolidation import WorldModelSimulator
from utils.context_spec import HOT_POT_CONTEXT


def validate_cv_implementation():
    """Ensure CV is actually testing on held-out data"""

    print("\n" + "="*70)
    print("CROSS-VALIDATION IMPLEMENTATION DIAGNOSTIC")
    print("="*70)

    # Load playbook
    playbook_obj = ACEPlaybook('hot_pot')
    playbook = playbook_obj.playbook

    observations = playbook.get('observations', [])
    high_rel_obs = [obs for obs in observations if obs.get('reliability') == 'HIGH']

    print(f"\nPlaybook Statistics:")
    print(f"  Total observations: {len(observations)}")
    print(f"  HIGH reliability:   {len(high_rel_obs)}")

    if len(high_rel_obs) < 3:
        print("⚠️  WARNING: Too few HIGH observations for meaningful CV")
        return

    print(f"\n=== Running Manual CV Fold (Fold 1/{len(high_rel_obs)}) ===")

    # Manually run one CV fold
    held_out = high_rel_obs[0]
    train = high_rel_obs[1:]

    print(f"\nHeld-out observation:")
    print(f"  Episode ID: {held_out.get('episode_id', 'N/A')}")
    print(f"  Context: {held_out.get('context', {})}")
    print(f"  Beliefs: {list(held_out.get('beliefs', {}).keys())}")

    print(f"\nTraining set:")
    print(f"  {len(train)} observations")

    # Build model from training data only
    train_playbook = {
        'domain': 'hot_pot',
        'observations': train
    }

    try:
        model = WorldModelSimulator.from_playbook(train_playbook, HOT_POT_CONTEXT)

        print(f"\nTrained model contexts: {list(model.context_models.keys())}")
        for ctx, params in model.context_models.items():
            print(f"  {ctx}: heating_rate = {params.get('heating_rate', 0.0):.3f}")

    except Exception as e:
        print(f"✗ Failed to build model: {e}")
        return

    # Predict held-out observation
    print(f"\n=== Predicting Held-Out Observation ===")

    try:
        # Extract actual heating rate from held-out observation
        beliefs = held_out.get('beliefs', {})
        actual_rate = WorldModelSimulator._extract_value(beliefs.get('heating_rate_mean'))

        if actual_rate is None:
            print("⚠️  No heating_rate_mean in held-out observation")
            return

        # Get predicted rate from model
        context_key = HOT_POT_CONTEXT.extract_context(held_out)

        if context_key not in model.context_models:
            print(f"⚠️  Context '{context_key}' not in training data (this is OK for rare contexts)")
            return

        predicted_rate = model.context_models[context_key].get('heating_rate', 0.0)

        # Calculate error
        error = abs(predicted_rate - actual_rate) / (abs(actual_rate) + 1e-6)

        print(f"\nContext: {context_key}")
        print(f"  Predicted heating_rate: {predicted_rate:.3f}")
        print(f"  Actual heating_rate:    {actual_rate:.3f}")
        print(f"  Relative error:         {error:.1%}")

        # Interpretation
        print(f"\n=== Diagnostic Results ===")

        if error < 0.01:
            print("✓ EXCELLENT: Model is highly accurate on held-out data (< 1% error)")
            print("  This suggests either:")
            print("  1. The world model is very accurate (good!)")
            print("  2. There's very little variation in the data (check context diversity)")
        elif error < 0.10:
            print("✓ GOOD: Acceptable CV error (< 10%)")
            print("  Model generalizes reasonably well to unseen data")
        elif error < 0.20:
            print("⚠️  ACCEPTABLE: Moderate CV error (10-20%)")
            print("  Model has some generalization capability but could be improved")
        else:
            print("✗ WARNING: High CV error (> 20%)")
            print("  Model may not generalize well - consider:")
            print("  1. Collecting more data")
            print("  2. Checking for context contamination")
            print("  3. Reviewing model assumptions")

        # Additional checks
        print(f"\n=== CV Implementation Validation ===")

        # Check 1: Held-out obs not in training set
        held_out_id = held_out.get('episode_id', '')
        train_ids = [o.get('episode_id', '') for o in train]

        if held_out_id in train_ids:
            print("✗ CRITICAL: Held-out observation found in training set!")
            print("  This indicates data leakage - CV is invalid")
        else:
            print("✓ Held-out observation NOT in training set (correct)")

        # Check 2: Training set size
        print(f"✓ Training set has {len(train)} observations")

        # Check 3: Context distribution
        train_contexts = [HOT_POT_CONTEXT.extract_context(o) for o in train]
        unique_contexts = set(train_contexts)
        print(f"✓ Training set covers {len(unique_contexts)} contexts: {unique_contexts}")

    except Exception as e:
        print(f"✗ Failed to predict: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70 + "\n")


def run_full_cv_comparison():
    """Compare manual CV implementation with automated one"""

    print("\n" + "="*70)
    print("COMPARING MANUAL vs AUTOMATED CV")
    print("="*70)

    from utils.offline_consolidation import OfflineConsolidation

    # Run automated CV
    playbook_obj = ACEPlaybook('hot_pot')
    playbook = playbook_obj.playbook

    oc = OfflineConsolidation()
    cv_result = oc.cross_validate(playbook, HOT_POT_CONTEXT, threshold=0.15)

    print(f"\nAutomated CV Results:")
    print(f"  Mean error:        {cv_result['mean_error']:.1%}")
    print(f"  Per-context error: {cv_result['per_context_error']}")
    print(f"  Passed:            {cv_result['passed']}")
    print(f"  Message:           {cv_result['message']}")

    # Now run manual CV on all folds
    observations = playbook.get('observations', [])
    high_rel_obs = [obs for obs in observations if obs.get('reliability') == 'HIGH']

    manual_errors = []

    print(f"\nManual CV on {len(high_rel_obs)} folds:")

    for i in range(min(len(high_rel_obs), 5)):  # First 5 folds
        held_out = high_rel_obs[i]
        train = high_rel_obs[:i] + high_rel_obs[i+1:]

        train_playbook = {'domain': 'hot_pot', 'observations': train}

        try:
            model = WorldModelSimulator.from_playbook(train_playbook, HOT_POT_CONTEXT)

            beliefs = held_out.get('beliefs', {})
            actual_rate = WorldModelSimulator._extract_value(beliefs.get('heating_rate_mean'))

            context_key = HOT_POT_CONTEXT.extract_context(held_out)

            if context_key in model.context_models and actual_rate is not None:
                predicted_rate = model.context_models[context_key].get('heating_rate', 0.0)
                error = abs(predicted_rate - actual_rate) / (abs(actual_rate) + 1e-6)
                manual_errors.append(error)

                print(f"  Fold {i+1}: error = {error:.1%} (context: {context_key})")

        except Exception as e:
            print(f"  Fold {i+1}: failed - {e}")

    if manual_errors:
        manual_mean = np.mean(manual_errors)
        print(f"\nManual CV mean error: {manual_mean:.1%}")
        print(f"Automated CV mean error: {cv_result['mean_error']:.1%}")

        diff = abs(manual_mean - cv_result['mean_error'])

        if diff < 0.01:
            print(f"✓ Manual and automated CV match (diff = {diff:.3%})")
        else:
            print(f"⚠️  Manual and automated CV differ by {diff:.1%}")

    print(f"\n" + "="*70 + "\n")


if __name__ == '__main__':
    validate_cv_implementation()
    run_full_cv_comparison()
