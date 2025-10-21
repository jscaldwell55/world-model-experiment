"""Validation utilities for textualization layers."""

from typing import Dict
from textualization.hot_pot_text import HotPotTextualization
from textualization.switch_light_text import SwitchLightTextualization
from textualization.chem_tile_text import ChemTileTextualization


def test_determinism_all_envs() -> Dict[str, bool]:
    """Test all textualization layers for determinism.

    Returns:
        Dictionary mapping environment name to pass/fail boolean
    """
    results = {}

    # Test HotPot
    hot_pot = HotPotTextualization()
    test_obs_hot_pot = [
        {'measured_temp': 23.5, 'time': 10.0, 'action': 'measure_temp'},
        {'time': 15.0, 'action': 'wait(5)', 'duration': 5.0},
        {'sensation': 'cool', 'time': 20.0, 'action': 'touch_pot'},
        {'label': 'Boiling!', 'stove_light': 'on', 'time': 0.0}
    ]
    results['hot_pot'] = all(
        hot_pot.validate_determinism(obs, num_trials=10)
        for obs in test_obs_hot_pot
    )

    # Test SwitchLight
    switch_light = SwitchLightTextualization()
    test_obs_switch = [
        {'action': 'flip_switch', 'switch_position': 'on', 'light_on': True, 'time': 1.0},
        {'action': 'observe_light', 'switch_position': 'off', 'light_on': False, 'time': 2.0},
        {'action': 'jiggle_relay', 'message': 'Relay clicked.', 'time': 3.0},
        {'switch_position': 'off', 'time': 0.0, 'message': 'Laboratory initialized.'}
    ]
    results['switch_light'] = all(
        switch_light.validate_determinism(obs, num_trials=10)
        for obs in test_obs_switch
    )

    # Test ChemTile
    chem_tile = ChemTileTextualization()
    test_obs_chem = [
        {'reaction': 'A+B', 'outcome': 'C', 'message': 'Success!', 'available_compounds': ['C', 'B'], 'temperature': 'medium'},
        {'action': 'heat', 'temperature': 'high', 'message': 'Temperature increased to high.'},
        {'action': 'inspect(A)', 'compound': 'A', 'info': 'Base reagent.'},
        {'available_compounds': ['A', 'B', 'B'], 'temperature': 'medium', 'message': 'Chemistry lab initialized.', 'time': 0.0}
    ]
    results['chem_tile'] = all(
        chem_tile.validate_determinism(obs, num_trials=10)
        for obs in test_obs_chem
    )

    return results


def test_no_leakage_all_envs() -> Dict[str, bool]:
    """Test all textualization layers for ground truth leakage.

    Returns:
        Dictionary mapping environment name to pass/fail boolean
    """
    results = {}

    # Test HotPot
    hot_pot = HotPotTextualization()
    test_obs_hot_pot = [
        {'measured_temp': 23.5, 'time': 10.0, 'action': 'measure_temp'},
        {'time': 15.0, 'action': 'wait(5)', 'duration': 5.0},
        {'sensation': 'cool', 'time': 20.0, 'action': 'touch_pot'},
        {'label': 'Boiling!', 'stove_light': 'on', 'time': 0.0}
    ]
    hot_pot_texts = [hot_pot.textualize_observation(obs) for obs in test_obs_hot_pot]
    results['hot_pot'] = all(
        hot_pot.validate_no_leakage(text) for text in hot_pot_texts
    )

    # Test SwitchLight
    switch_light = SwitchLightTextualization()
    test_obs_switch = [
        {'action': 'flip_switch', 'switch_position': 'on', 'light_on': True, 'time': 1.0},
        {'action': 'observe_light', 'switch_position': 'off', 'light_on': False, 'time': 2.0},
        {'action': 'jiggle_relay', 'message': 'Relay clicked.', 'time': 3.0},
        {'switch_position': 'off', 'time': 0.0, 'message': 'Laboratory initialized.'}
    ]
    switch_texts = [switch_light.textualize_observation(obs) for obs in test_obs_switch]
    results['switch_light'] = all(
        switch_light.validate_no_leakage(text) for text in switch_texts
    )

    # Test ChemTile
    chem_tile = ChemTileTextualization()
    test_obs_chem = [
        {'reaction': 'A+B', 'outcome': 'C', 'message': 'Success!', 'available_compounds': ['C', 'B'], 'temperature': 'medium'},
        {'action': 'heat', 'temperature': 'high', 'message': 'Temperature increased to high.'},
        {'action': 'inspect(A)', 'compound': 'A', 'info': 'Base reagent.'},
        {'available_compounds': ['A', 'B', 'B'], 'temperature': 'medium', 'message': 'Chemistry lab initialized.', 'time': 0.0}
    ]
    chem_texts = [chem_tile.textualize_observation(obs) for obs in test_obs_chem]
    results['chem_tile'] = all(
        chem_tile.validate_no_leakage(text) for text in chem_texts
    )

    return results


def test_numerical_precision() -> Dict[str, bool]:
    """Test that numerical formatting is consistent.

    Returns:
        Dictionary mapping environment name to pass/fail boolean
    """
    results = {}

    # Test HotPot - temperature should be 1 decimal place, time should be 0
    hot_pot = HotPotTextualization()
    obs = {'measured_temp': 23.567, 'time': 10.999, 'action': 'measure_temp'}
    text = hot_pot.textualize_observation(obs)
    # Check that temperature has 1 decimal place (23.567 rounds to 23.6) and time has 0
    results['hot_pot'] = '23.6' in text and '11 seconds' in text

    # Test SwitchLight - no numerical precision requirements
    results['switch_light'] = True

    # Test ChemTile - no numerical precision requirements for now
    results['chem_tile'] = True

    return results


def run_all_validations() -> Dict[str, Dict[str, bool]]:
    """Run all validation tests.

    Returns:
        Dictionary mapping test name to results dictionary
    """
    return {
        'determinism': test_determinism_all_envs(),
        'no_leakage': test_no_leakage_all_envs(),
        'numerical_precision': test_numerical_precision()
    }


if __name__ == '__main__':
    """Run validation tests when executed directly."""
    results = run_all_validations()

    print("Validation Results:")
    print("=" * 50)

    for test_name, test_results in results.items():
        print(f"\n{test_name.upper()}:")
        for env_name, passed in test_results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {env_name}: {status}")

    # Check if all tests passed
    all_passed = all(
        all(test_results.values())
        for test_results in results.values()
    )

    print("\n" + "=" * 50)
    if all_passed:
        print("All validation tests PASSED ✓")
    else:
        print("Some validation tests FAILED ✗")

    exit(0 if all_passed else 1)
