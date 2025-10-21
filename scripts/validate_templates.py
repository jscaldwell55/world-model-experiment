#!/usr/bin/env python3
"""Validate textualization templates for determinism and leakage.

This script tests that textualization layers:
1. Produce deterministic output (same obs → same text)
2. Never leak ground truth (forbidden keys don't appear)
3. Maintain numerical precision (consistent formatting)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from textualization import (
    HotPotTextualization,
    SwitchLightTextualization,
    ChemTileTextualization
)


def test_hot_pot():
    """Test HotPot textualization."""
    print("Testing HotPot Textualization...")

    textualizer = HotPotTextualization()

    # Test determinism
    obs = {'measured_temp': 23.456, 'time': 10.7, 'action': 'measure_temp'}
    text1 = textualizer.textualize_observation(obs)
    text2 = textualizer.textualize_observation(obs)

    assert text1 == text2, "Determinism failed"
    print(f"  ✓ Determinism: '{text1}'")

    # Test no leakage
    assert textualizer.validate_no_leakage(text1), "Leakage detected"
    print("  ✓ No leakage")

    # Test numerical precision (1 decimal for temp, 0 for time)
    assert '23.5' in text1, f"Expected '23.5' in text, got: {text1}"
    assert '11 seconds' in text1, f"Expected '11 seconds' in text, got: {text1}"
    print("  ✓ Numerical precision")

    # Test initial description
    initial = textualizer.get_initial_description()
    assert textualizer.validate_no_leakage(initial), "Leakage in initial description"
    print(f"  ✓ Initial description: '{initial[:50]}...'")

    # Test all action types
    test_cases = [
        ({'measured_temp': 25.0, 'time': 5.0, 'action': 'measure_temp'},
         'Thermometer reads 25.0°C'),
        ({'time': 10.0, 'action': 'wait(5)', 'duration': 5.0},
         'Time elapsed: 10 seconds'),
        ({'sensation': 'warm', 'time': 15.0, 'action': 'touch_pot'},
         'You touched the pot. Warm'),
        ({'stove_light': 'bright', 'time': 20.0, 'action': 'toggle_stove'},
         'Stove indicator light is now bright'),
    ]

    for obs, expected_substring in test_cases:
        text = textualizer.textualize_observation(obs)
        assert expected_substring in text, f"Expected '{expected_substring}' in '{text}'"

    print("  ✓ All action types")
    print()


def test_switch_light():
    """Test SwitchLight textualization."""
    print("Testing SwitchLight Textualization...")

    textualizer = SwitchLightTextualization()

    # Test determinism
    obs = {'action': 'flip_switch', 'switch_position': 'on', 'light_on': True, 'time': 1.0}
    text1 = textualizer.textualize_observation(obs)
    text2 = textualizer.textualize_observation(obs)

    assert text1 == text2, "Determinism failed"
    print(f"  ✓ Determinism: '{text1}'")

    # Test no leakage
    assert textualizer.validate_no_leakage(text1), "Leakage detected"
    print("  ✓ No leakage")

    # Test initial description
    initial = textualizer.get_initial_description()
    assert textualizer.validate_no_leakage(initial), "Leakage in initial description"
    print(f"  ✓ Initial description: '{initial[:50]}...'")

    # Test all action types
    test_cases = [
        ({'action': 'flip_switch', 'switch_position': 'on', 'light_on': True, 'time': 1.0},
         'Switch toggled'),
        ({'action': 'observe_light', 'switch_position': 'off', 'light_on': False, 'time': 2.0},
         'Observed without touching'),
        ({'action': 'jiggle_relay', 'message': 'Relay clicked.', 'time': 3.0},
         'Relay clicked'),
    ]

    for obs, expected_substring in test_cases:
        text = textualizer.textualize_observation(obs)
        assert expected_substring in text, f"Expected '{expected_substring}' in '{text}'"

    print("  ✓ All action types")
    print()


def test_chem_tile():
    """Test ChemTile textualization."""
    print("Testing ChemTile Textualization...")

    textualizer = ChemTileTextualization()

    # Test determinism
    obs = {
        'reaction': 'A+B',
        'outcome': 'C',
        'message': 'Success!',
        'available_compounds': ['C', 'B'],
        'temperature': 'medium'
    }
    text1 = textualizer.textualize_observation(obs)
    text2 = textualizer.textualize_observation(obs)

    assert text1 == text2, "Determinism failed"
    print(f"  ✓ Determinism: '{text1}'")

    # Test no leakage
    assert textualizer.validate_no_leakage(text1), "Leakage detected"
    print("  ✓ No leakage")

    # Test initial description
    initial = textualizer.get_initial_description()
    assert textualizer.validate_no_leakage(initial), "Leakage in initial description"
    print(f"  ✓ Initial description: '{initial[:50]}...'")

    # Test all action types
    test_cases = [
        ({
            'reaction': 'A+B',
            'outcome': 'C',
            'message': 'Success!',
            'available_compounds': ['C', 'B'],
            'temperature': 'medium'
        }, 'Mixed A with B'),
        ({
            'action': 'heat',
            'temperature': 'high',
            'message': 'Temperature increased to high.'
        }, 'Temperature increased'),
        ({
            'action': 'inspect(A)',
            'compound': 'A',
            'info': 'Base reagent.'
        }, 'Inspected compound A'),
    ]

    for obs, expected_substring in test_cases:
        text = textualizer.textualize_observation(obs)
        assert expected_substring in text, f"Expected '{expected_substring}' in '{text}'"

    print("  ✓ All action types")
    print()


def main():
    print("=" * 70)
    print("TEMPLATE VALIDATION")
    print("=" * 70)
    print()

    try:
        test_hot_pot()
        test_switch_light()
        test_chem_tile()

        print("=" * 70)
        print("ALL VALIDATIONS PASSED ✓")
        print("=" * 70)

        return 0

    except AssertionError as e:
        print(f"\n✗ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
