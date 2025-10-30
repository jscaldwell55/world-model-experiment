# evaluation/trajectory_extraction.py
"""
Extract trajectory data from episode steps for exploration-dependent evaluation.

The episode steps already contain all measurements and observations.
We just need to extract and structure them for evaluation.
"""

from typing import List, Dict, Any


def extract_temperature_trajectory(steps: List[Dict]) -> List[tuple]:
    """
    Extract temperature measurements from episode steps.

    Args:
        steps: List of step dictionaries from episode

    Returns:
        List of (time, temperature) tuples
    """
    trajectory = []

    for step in steps:
        obs = step.get('observation', {})
        if 'measured_temp' in obs:
            time = obs.get('time', step.get('step_num', 0))
            temp = obs['measured_temp']
            trajectory.append((time, temp))

    return trajectory


def extract_switch_observations(steps: List[Dict]) -> List[Dict]:
    """
    Extract switch flip observations from episode steps.

    Args:
        steps: List of step dictionaries from episode

    Returns:
        List of observation dicts with switch_position and light_on
    """
    observations = []

    for step in steps:
        obs = step.get('observation', {})
        if 'switch_position' in obs and 'light_on' in obs:
            observations.append({
                'time': obs.get('time', step.get('step_num', 0)),
                'switch_position': obs['switch_position'],
                'light_on': obs['light_on'],
                'action': obs.get('action', 'unknown')
            })

    return observations


def build_hotpot_ground_truth(steps: List[Dict], final_ground_truth: Dict) -> Dict:
    """
    Build comprehensive ground truth for HotPot including trajectory.

    Args:
        steps: Episode steps with observations
        final_ground_truth: Final state from environment.get_ground_truth()

    Returns:
        Enhanced ground truth dictionary
    """
    trajectory = extract_temperature_trajectory(steps)

    # Extract measurements
    measurements = [temp for _, temp in trajectory]
    times = [time for time, _ in trajectory]

    # Count measurements
    num_measurements = len(measurements)

    # Initial and final temps
    initial_temp = measurements[0] if measurements else final_ground_truth.get('actual_temp', 20.0)
    final_temp = measurements[-1] if measurements else final_ground_truth.get('actual_temp', 20.0)

    # Calculate heating rate from measurements (if multiple measurements)
    heating_rate = final_ground_truth.get('heating_rate', 0.0)
    if len(measurements) >= 2:
        # Calculate from actual measurements
        temp_diff = measurements[-1] - measurements[0]
        time_diff = times[-1] - times[0]
        if time_diff > 0:
            observed_rate = temp_diff / time_diff
            # Use observed rate if available
            heating_rate = observed_rate

    # Find first measurement time
    first_measurement_time = times[0] if times else 0

    # Build temperature_trajectory for specific time queries
    temperature_trajectory = trajectory

    # Find when stove was toggled (if any)
    temp_after_toggle = None
    for i, step in enumerate(steps):
        obs = step.get('observation', {})
        if 'toggle_stove' in obs.get('action', ''):
            # Find next temperature measurement after toggle
            for j in range(i + 1, len(steps)):
                next_obs = steps[j].get('observation', {})
                if 'measured_temp' in next_obs:
                    temp_after_toggle = next_obs['measured_temp']
                    break
            break

    return {
        # Original ground truth
        **final_ground_truth,

        # Trajectory data
        'temperature_trajectory': temperature_trajectory,
        'num_measurements': num_measurements,
        'initial_temp': initial_temp,
        'final_temp': final_temp,
        'heating_rate': heating_rate,
        'first_measurement_time': first_measurement_time,

        # Action-specific data
        'temp_after_toggle': temp_after_toggle if temp_after_toggle is not None else initial_temp,

        # For temporal queries (extract specific times)
        'temp_at_t10': _get_temp_at_time(trajectory, 10),
        'temp_at_t20': _get_temp_at_time(trajectory, 20),
        'temp_at_t30': _get_temp_at_time(trajectory, 30),
    }


def build_switchlight_ground_truth(steps: List[Dict], final_ground_truth: Dict) -> Dict:
    """
    Build comprehensive ground truth for SwitchLight including observations.

    Args:
        steps: Episode steps with observations
        final_ground_truth: Final state from environment.get_ground_truth()

    Returns:
        Enhanced ground truth dictionary
    """
    observations = extract_switch_observations(steps)

    # Count flips
    num_flips = sum(1 for obs in observations if 'flip' in obs.get('action', '').lower())

    # Count jiggle_relay actions
    jiggled_relay = any('jiggle' in step.get('observation', {}).get('action', '').lower() for step in steps)

    # Build tested configurations
    tested_configurations = {}
    for obs in observations:
        switch_pos = obs['switch_position']
        light_on = obs['light_on']
        tested_configurations[switch_pos] = light_on

    # Count how many resulted in light ON
    num_on_states = sum(1 for light_on in tested_configurations.values() if light_on)

    # First observation
    first_switch_pos = observations[0]['switch_position'] if observations else None
    first_light_on = observations[0]['light_on'] if observations else None

    # Check if state ever changed
    light_states = [obs['light_on'] for obs in observations]
    observed_state_change = len(set(light_states)) > 1 if light_states else False

    # Infer fault probability from consistency
    # If all observations match expected wiring, fault_prob is low
    # If inconsistent, fault_prob is higher
    wire_layout = final_ground_truth.get('wire_layout', 'layout_A')
    faulty_relay = final_ground_truth.get('faulty_relay', False)

    # Simple heuristic: if behavior is consistent, infer low fault probability
    if observed_state_change and len(observations) > 1:
        inferred_fault_probability = 0.05  # Looks normal
    else:
        inferred_fault_probability = 0.1  # Default prior

    if faulty_relay:
        inferred_fault_probability = 0.9  # Should be detected as faulty

    # Predict next state based on wiring
    current_switch = final_ground_truth.get('switch_position', 'off')
    current_light = final_ground_truth.get('light_should_be_on', False)

    if wire_layout == 'layout_A':
        # Normal: switch toggles light
        predicted_next_state = "light will toggle" if current_switch else "light will toggle"
    else:
        # Inverted
        predicted_next_state = "light will toggle (inverted)"

    return {
        # Original ground truth
        **final_ground_truth,

        # Observation data
        'tested_configurations': tested_configurations,
        'num_flips': num_flips,
        'num_on_states': num_on_states,
        'jiggled_relay': jiggled_relay,

        # First observation
        'first_switch_pos': first_switch_pos,
        'first_light_on': first_light_on,

        # Pattern recognition
        'observed_state_change': observed_state_change,
        'inferred_fault_probability': inferred_fault_probability,
        'predicted_next_state': predicted_next_state,
    }


def _get_temp_at_time(trajectory: List[tuple], target_time: float) -> float:
    """
    Get temperature at specific time from trajectory.

    Args:
        trajectory: List of (time, temp) tuples
        target_time: Target time in seconds

    Returns:
        Temperature at that time, or None if not measured
    """
    # Find closest measurement within 1 second
    for time, temp in trajectory:
        if abs(time - target_time) <= 1.0:
            return temp

    return None


def enhance_ground_truth_with_trajectory(
    steps: List[Dict],
    environment_ground_truth: Dict,
    environment_name: str
) -> Dict:
    """
    Enhance ground truth with trajectory data from steps.

    Args:
        steps: Episode steps
        environment_ground_truth: Ground truth from environment
        environment_name: Name of environment

    Returns:
        Enhanced ground truth dictionary
    """
    if environment_name == "HotPotLab":
        return build_hotpot_ground_truth(steps, environment_ground_truth)
    elif environment_name == "SwitchLight":
        return build_switchlight_ground_truth(steps, environment_ground_truth)
    else:
        # Unknown environment, return original
        return environment_ground_truth
