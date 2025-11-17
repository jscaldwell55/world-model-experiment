#!/usr/bin/env python3
"""Verification script for rollback configuration"""

import sys
import importlib.util

# Load the agent module
spec = importlib.util.spec_from_file_location("simple_world_model", "agents/simple_world_model.py")
agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_module)

# Load belief state module
spec = importlib.util.spec_from_file_location("belief_state", "models/belief_state.py")
belief_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(belief_module)

def verify_rollback():
    """Verify all critical rollback values"""
    issues = []

    # Check 1: Default prior_strength in __init__
    import inspect
    sig = inspect.signature(agent_module.SimpleWorldModel.__init__)
    prior_strength_default = sig.parameters['prior_strength'].default

    if prior_strength_default != 0.1:
        issues.append(f"❌ prior_strength default is {prior_strength_default}, should be 0.1")
    else:
        print(f"✓ prior_strength = {prior_strength_default}")

    # Check 2: HotPotBelief defaults
    hotpot_defaults = belief_module.HotPotBelief.__fields__
    heating_mean = hotpot_defaults['heating_rate_mean'].default
    heating_std = hotpot_defaults['heating_rate_std'].default

    if heating_mean != 1.5:
        issues.append(f"❌ heating_rate_mean is {heating_mean}, should be 1.5")
    else:
        print(f"✓ heating_rate_mean = {heating_mean}")

    if heating_std != 0.3:
        issues.append(f"❌ heating_rate_std is {heating_std}, should be 0.3")
    else:
        print(f"✓ heating_rate_std = {heating_std}")

    # Check 3: No removed variables in __init__
    with open("agents/simple_world_model.py", 'r') as f:
        content = f.read()

    forbidden_vars = [
        'current_stove_power',
        'temperature_history',
        'burn_threshold_learned',
        'surprisal_history',
        'belief_history',
        'beliefs_converged'
    ]

    for var in forbidden_vars:
        if f"self.{var}" in content:
            issues.append(f"❌ Found forbidden variable: {var}")

    if not any(f"self.{var}" in content for var in forbidden_vars):
        print(f"✓ No forbidden variables (stove_power, temperature_history, etc.)")

    # Check 4: No np.polyfit
    if 'np.polyfit' in content or 'polyfit' in content:
        issues.append(f"❌ Found np.polyfit in code")
    else:
        print(f"✓ No np.polyfit regression code")

    # Check 5: No adaptive budget methods
    if '_get_adaptive_budget' in content:
        issues.append(f"❌ Found _get_adaptive_budget method")
    else:
        print(f"✓ No adaptive budget code")

    # Check 6: No belief convergence checking
    if '_check_belief_convergence' in content:
        issues.append(f"❌ Found _check_belief_convergence method")
    else:
        print(f"✓ No belief convergence checking")

    # Check 7: No boundary exploration hints
    if 'exploration_hint' in content or 'EXPLORATION HINT' in content:
        issues.append(f"❌ Found exploration hints")
    else:
        print(f"✓ No boundary exploration hints")

    # Summary
    print("\n" + "="*60)
    if issues:
        print("ROLLBACK VERIFICATION FAILED")
        print("="*60)
        for issue in issues:
            print(issue)
        return False
    else:
        print("✅ ROLLBACK VERIFICATION PASSED")
        print("="*60)
        print("\nAll critical values restored to original configuration:")
        print(f"  - prior_strength = 0.1")
        print(f"  - heating_rate_mean = 1.5")
        print(f"  - heating_rate_std = 0.3")
        print(f"  - No regression code")
        print(f"  - No power tracking")
        print(f"  - No adaptive features")
        return True

if __name__ == "__main__":
    success = verify_rollback()
    sys.exit(0 if success else 1)
