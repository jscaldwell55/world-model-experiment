#!/usr/bin/env python3
"""
Analyze actual hot pot observations to understand why learned beliefs are wrong
"""

import json
from pathlib import Path

results_dir = Path("results/memory_validation_9ep/raw")

print("=" * 80)
print("HOT POT OBSERVATION ANALYSIS")
print("=" * 80)

for file in sorted(results_dir.glob("hot_pot*.json")):
    with open(file, 'r') as f:
        data = json.load(f)

    episode_num = int(file.stem.split('_ep')[-1])

    print(f"\nEPISODE {episode_num}")
    print("-" * 80)

    steps = data.get('steps', [])

    temp_measurements = []
    stove_states = []
    times = []

    for i, step in enumerate(steps):
        action = step.get('action', '')
        observation = step.get('observation', {})

        # Extract time
        time_val = observation.get('time', observation.get('time_elapsed', None))
        if time_val is not None:
            times.append(time_val)

        # Check for temperature measurements
        if 'measure_temp' in action.lower() or 'temperature' in action.lower():
            # Try to extract temperature from observation
            obs_str = str(observation)

            # Look for temperature in observation
            if 'measured_temp' in observation:
                temp = observation['measured_temp']
                temp_measurements.append({
                    'step': i,
                    'time': time_val,
                    'temp': temp,
                    'action': action
                })
                print(f"  Step {i:2d} (t={time_val:4.1f}s): {action:30s} → Temp: {temp}°C")

        # Track stove state changes
        if 'toggle' in action.lower() or 'stove' in action.lower():
            stove_states.append({
                'step': i,
                'time': time_val,
                'action': action
            })
            print(f"  Step {i:2d} (t={time_val:4.1f}s): {action:30s}")

    # Analyze heating rate if we have enough data
    if len(temp_measurements) >= 2:
        print(f"\n  Temperature Measurements:")
        for tm in temp_measurements:
            print(f"    t={tm['time']:4.1f}s: {tm['temp']}°C")

        # Calculate observed heating rate
        if len(temp_measurements) >= 2:
            first_temp = temp_measurements[0]['temp']
            last_temp = temp_measurements[-1]['temp']
            first_time = temp_measurements[0]['time']
            last_time = temp_measurements[-1]['time']

            time_diff = last_time - first_time
            temp_diff = last_temp - first_temp

            if time_diff > 0:
                observed_rate = temp_diff / time_diff
                print(f"\n  Observed Heating Rate:")
                print(f"    ΔTemp: {temp_diff:.1f}°C over {time_diff:.1f}s")
                print(f"    Rate: {observed_rate:.2f}°C/s")
            else:
                print(f"\n  ⚠️ Not enough time elapsed between measurements")
    else:
        print(f"\n  ⚠️ Only {len(temp_measurements)} temperature measurement(s) - need 2+ to estimate heating rate")

    # Show test results
    test_results = data.get('test_results', [])
    correct = sum(1 for q in test_results if q.get('correct', False))
    total = len(test_results)

    print(f"\n  Test Performance: {correct}/{total} correct ({correct/total*100:.0f}%)")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

print("\nPossible reasons for inaccurate learned beliefs:")
print("  1. Too few temperature measurements (4-5 per episode)")
print("  2. Measurements taken at inappropriate times")
print("  3. Agent not doing controlled experiments (heating → measure → heat → measure)")
print("  4. High variance in observations leading to incorrect parameter estimates")
print("\n" + "=" * 80)
