"""Neutral textualization layer for HotPot environment.

This variant removes all emotive/misleading language and uses neutral phrasing.
Designed to test whether anti-coupling is an artifact of misleading labels.
"""

from typing import Dict
from textualization.base import TextualizationLayer


class HotPotTextualizationNeutral(TextualizationLayer):
    """Neutral variant: Remove emotive labels, use technical phrasing.

    Templates:
    - Initial: "Monitoring system initialized. Heating element status: {active/inactive}."
    - measure_temp: "Temperature sensor output: {temp} degrees Celsius. Measurement timestamp: {time} seconds."
    - wait: "System idle. Elapsed time: {time} seconds."
    - touch_pot: "Tactile sensor activated. Thermal sensation: {sensation}. Timestamp: {time} seconds."
    - toggle_stove: "Heating element status: {active/inactive}. Timestamp: {time} seconds."

    Key changes from original:
    - Removes "Boiling!" label (misleading emotional cue)
    - Uses "heating element" instead of "stove indicator light"
    - Uses "temperature sensor output" instead of "thermometer reads"
    - Uses "timestamp" instead of "time elapsed"
    - Removes personal pronouns ("You are in...")
    """

    def __init__(self):
        super().__init__()
        self.set_forbidden_keys([
            'actual_temp', 'stove_power', 'ground_truth',
            'hidden_state', 'heating_rate', 'true_temp'
        ])

    def textualize_observation(self, obs: Dict) -> str:
        """Convert observation to neutral canonical string.

        Args:
            obs: Observation from HotPot environment

        Returns:
            Neutral canonical natural language description

        Raises:
            ValueError: If observation format is invalid
        """
        action = obs.get('action', '')

        if action == 'measure_temp' or 'measured_temp' in obs:
            # measure_temp observation (neutral phrasing)
            measured_temp = obs.get('measured_temp')
            time = obs.get('time', 0.0)

            if measured_temp is None:
                raise ValueError("measure_temp observation missing 'measured_temp' key")

            return f"Temperature sensor output: {measured_temp:.1f} degrees Celsius. Measurement timestamp: {time:.0f} seconds."

        elif action.startswith('wait') or 'duration' in obs:
            # wait observation (neutral phrasing)
            time = obs.get('time', 0.0)
            return f"System idle. Elapsed time: {time:.0f} seconds."

        elif action == 'touch_pot' or 'sensation' in obs:
            # touch_pot observation (neutral phrasing)
            sensation = obs.get('sensation', 'unknown')
            time = obs.get('time', 0.0)
            return f"Tactile sensor activated. Thermal sensation: {sensation}. Timestamp: {time:.0f} seconds."

        elif action == 'toggle_stove' or ('stove_light' in obs and 'label' not in obs):
            # toggle_stove observation (neutral phrasing)
            stove_light = obs.get('stove_light', 'off')
            time = obs.get('time', 0.0)

            # Convert "on"/"off" or "bright"/"dim" to "active"/"inactive"
            status = "active" if stove_light in ['on', 'bright'] else "inactive"

            return f"Heating element status: {status}. Timestamp: {time:.0f} seconds."

        elif 'label' in obs and 'stove_light' in obs:
            # Initial observation (neutral phrasing, NO misleading label)
            stove_light = obs.get('stove_light', 'off')

            # Convert "on"/"off" or "bright"/"dim" to "active"/"inactive"
            status = "active" if stove_light in ['on', 'bright'] else "inactive"

            # CRITICAL: Remove the misleading "Boiling!" label entirely
            return f"Monitoring system initialized. Heating element status: {status}. Temperature measurement device ready."

        elif 'message' in obs:
            # Generic message observation
            return obs['message']

        else:
            # Unknown observation type
            raise ValueError(f"Unknown observation format: {obs}")

    def textualize_action(self, action: str) -> str:
        """Convert action to neutral canonical sentence.

        Args:
            action: Action string (e.g., "measure_temp", "wait(5)")

        Returns:
            Neutral canonical action description
        """
        action = action.strip()

        if action == "measure_temp":
            return "Action taken: measure_temp()"
        elif action.startswith("wait"):
            # Extract duration if present
            try:
                duration_str = action.replace("wait", "").strip("()")
                duration = float(duration_str)
                return f"Action taken: wait({duration:.0f})"
            except (ValueError, AttributeError):
                return "Action taken: wait(1)"
        elif action == "touch_pot":
            return "Action taken: touch_pot()"
        elif action == "toggle_stove":
            return "Action taken: toggle_stove()"
        else:
            return f"Action taken: {action}"

    def get_initial_description(self) -> str:
        """Get initial environment description (neutral phrasing).

        Returns:
            Neutral generic initial description (no specific state)
        """
        return "Laboratory monitoring system. Available operations: temperature measurement, system idle wait, tactile sensor activation, heating element toggle."
