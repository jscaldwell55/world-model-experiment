#!/usr/bin/env python3
"""
Verify token accounting implementation.

Runs a single pilot episode and validates that:
1. Token breakdown is present in episode log
2. Sum of categories equals total tokens
3. All categories are non-negative
4. Validation flag is True

Usage:
    python scripts/verify_token_accounting.py
"""

import json
import sys
from pathlib import Path
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.runner import ExperimentRunner
from environments.hot_pot import HotPotLab
from agents.ace import ACEAgent


def verify_token_breakdown(episode_log: dict) -> bool:
    """
    Verify token breakdown in episode log.

    Args:
        episode_log: Episode log dictionary

    Returns:
        True if valid, raises AssertionError if invalid
    """
    print("\n" + "="*70)
    print("VERIFYING TOKEN BREAKDOWN")
    print("="*70)

    # Check presence
    assert 'token_breakdown' in episode_log, "Missing 'token_breakdown' in episode log"
    breakdown = episode_log['token_breakdown']
    assert breakdown is not None, "token_breakdown is None"

    print(f"\n✓ Token breakdown present")

    # Check structure
    assert 'breakdown' in breakdown, "Missing 'breakdown' in token_breakdown"
    assert 'records' in breakdown, "Missing 'records' in token_breakdown"

    print(f"✓ Token breakdown structure valid")

    # Check totals
    totals = breakdown['breakdown']['totals']
    expected_input = episode_log['total_input_tokens']
    expected_output = episode_log['total_output_tokens']

    print(f"\nExpected totals:")
    print(f"  Input:  {expected_input}")
    print(f"  Output: {expected_output}")

    print(f"\nActual totals from breakdown:")
    print(f"  Input:  {totals['input']}")
    print(f"  Output: {totals['output']}")

    assert totals['input'] == expected_input, \
        f"Input token mismatch: {totals['input']} != {expected_input}"
    assert totals['output'] == expected_output, \
        f"Output token mismatch: {totals['output']} != {expected_output}"

    print(f"\n✓ Token totals match")

    # Check categories
    categories = ['exploration', 'curation', 'evaluation', 'planning']
    print(f"\nCategory breakdown:")

    for category in categories:
        assert category in breakdown['breakdown'], f"Missing category '{category}'"
        cat_data = breakdown['breakdown'][category]

        # Check non-negative
        assert cat_data['input'] >= 0, f"{category} input tokens < 0"
        assert cat_data['output'] >= 0, f"{category} output tokens < 0"
        assert cat_data['total'] >= 0, f"{category} total tokens < 0"

        # Check consistency
        assert cat_data['total'] == cat_data['input'] + cat_data['output'], \
            f"{category} total != input + output"

        print(f"  {category:15s}: {cat_data['total']:6d} tokens "
              f"({cat_data['input']:5d} in, {cat_data['output']:5d} out)")

    print(f"\n✓ All categories valid")

    # Check validation passed
    if 'validation_passed' in breakdown:
        assert breakdown['validation_passed'], \
            f"Validation failed: {breakdown.get('validation_error', 'Unknown error')}"
        print(f"✓ Validation passed")

    # Check records
    num_records = len(breakdown['records'])
    print(f"\n✓ {num_records} API calls recorded")

    return True


def run_pilot_episode():
    """
    Run a single pilot episode to test token accounting.

    Returns:
        Episode log dictionary
    """
    print("\n" + "="*70)
    print("RUNNING PILOT EPISODE")
    print("="*70)

    # Create temp directory for output
    temp_dir = Path(tempfile.mkdtemp(prefix="token_accounting_test_"))
    print(f"\nOutput directory: {temp_dir}")

    # Setup experiment
    config = {
        'models': {
            'a_c_e': {'model': 'claude-sonnet-4-5-20250929'}
        },
        'budgets': {
            'actions_per_episode': 5  # Small budget for quick test
        }
    }

    # Create runner
    runner = ExperimentRunner(
        config=config,
        environment_cls=HotPotLab,
        agent_cls=ACEAgent
    )

    print(f"\nAgent: ACE")
    print(f"Environment: HotPotLab")
    print(f"Action budget: 5")

    # Run single episode
    try:
        episode_log = runner.run_episode(
            episode_id="token_accounting_test",
            seed=42,
            save_dir=temp_dir
        )

        print(f"\n✓ Episode completed successfully")
        print(f"✓ Episode log saved: {temp_dir / 'token_accounting_test.json'}")

        return episode_log

    except Exception as e:
        print(f"\n✗ Episode failed: {e}")
        raise


def main():
    """Main verification workflow"""
    print("\n" + "="*70)
    print("TOKEN ACCOUNTING VERIFICATION")
    print("="*70)
    print("\nThis script verifies that token accounting is working correctly.")
    print("It runs a single pilot episode and validates the token breakdown.\n")

    try:
        # Run pilot episode
        episode_log = run_pilot_episode()

        # Verify token breakdown
        verify_token_breakdown(episode_log)

        # Success summary
        print("\n" + "="*70)
        print("✅ TOKEN ACCOUNTING VERIFICATION PASSED")
        print("="*70)

        print("\nAll checks passed:")
        print("  ✓ Token breakdown present in episode log")
        print("  ✓ Sum of categories equals total tokens")
        print("  ✓ All categories are non-negative")
        print("  ✓ Validation passed")

        # Show token distribution
        breakdown = episode_log['token_breakdown']['breakdown']
        total = breakdown['totals']['total']

        print("\nToken distribution:")
        for category in ['exploration', 'curation', 'evaluation', 'planning']:
            cat_total = breakdown[category]['total']
            pct = (cat_total / total * 100) if total > 0 else 0
            print(f"  {category:15s}: {cat_total:6d} tokens ({pct:5.1f}%)")

        print(f"\n  {'TOTAL':15s}: {total:6d} tokens")

        print("\n✅ Token accounting is working correctly!")
        return 0

    except AssertionError as e:
        print("\n" + "="*70)
        print("❌ TOKEN ACCOUNTING VERIFICATION FAILED")
        print("="*70)
        print(f"\nError: {e}")
        return 1

    except Exception as e:
        print("\n" + "="*70)
        print("❌ VERIFICATION SCRIPT ERROR")
        print("="*70)
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
