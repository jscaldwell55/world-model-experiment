"""Textualization layer for HotPot environment."""

from typing import Dict
from textualization.base import TextualizationLayer


class HotPotTextualization(TextualizationLayer):
    """Convert HotPot observations to canonical text.

    Templates:
    - Initial: "You are in a laboratory with a pot on a stove..."
    - measure_temp: "Thermometer reads {temp}°C. Time elapsed: {time} seconds."
    - wait: "Time elapsed: {time} seconds."
    - touch_pot: "You touched the pot. {sensation}. Time elapsed: {time} seconds."
    - toggle_stove: "Stove indicator light is now {light}. Time elapsed: {time} seconds."

    Forbidden keys: actual_temp, stove_power, ground_truth, hidden_state, heating_rate, true_temp
    """

    def __init__(self):
        super().__init__()
        self.set_forbidden_keys([
            'actual_temp', 'stove_power', 'ground_truth',
            'hidden_state', 'heating_rate', 'true_temp'
        ])

    def textualize_observation(self, obs: Dict) -> str:
        """Convert observation to canonical string.

        Args:
            obs: Observation from HotPot environment

        Returns:
            Canonical natural language description

        Raises:
            ValueError: If observation format is invalid
        """
        # Check for action type to determine template
        action = obs.get('action', '')

        if action == 'measure_temp' or 'measured_temp' in obs:
            # measure_temp observation
            measured_temp = obs.get('measured_temp')
            time = obs.get('time', 0.0)

            if measured_temp is None:
                raise ValueError("measure_temp observation missing 'measured_temp' key")

            return f"Thermometer reads {measured_temp:.1f}°C. Time elapsed: {time:.0f} seconds."

        elif action.startswith('wait') or 'duration' in obs:
            # wait observation
            time = obs.get('time', 0.0)
            return f"Time elapsed: {time:.0f} seconds."

        elif action == 'touch_pot' or 'sensation' in obs:
            # touch_pot observation
            sensation = obs.get('sensation', 'unknown')
            time = obs.get('time', 0.0)
            return f"You touched the pot. {sensation.capitalize()}. Time elapsed: {time:.0f} seconds."

        elif action == 'toggle_stove' or ('stove_light' in obs and 'label' not in obs):
            # toggle_stove observation (not initial state)
            stove_light = obs.get('stove_light', 'off')
            time = obs.get('time', 0.0)
            return f"Stove indicator light is now {stove_light}. Time elapsed: {time:.0f} seconds."

        elif 'label' in obs and 'stove_light' in obs:
            # Initial observation
            label = obs.get('label', 'Unknown')
            stove_light = obs.get('stove_light', 'off')
            return f"You are in a laboratory with a pot on a stove. The pot has a label that says '{label}'. The stove indicator light is {stove_light}."

        elif 'message' in obs:
            # Generic message observation
            return obs['message']

        else:
            # Unknown observation type
            raise ValueError(f"Unknown observation format: {obs}")

    def textualize_action(self, action: str) -> str:
        """Convert action to canonical sentence.

        Args:
            action: Action string (e.g., "measure_temp", "wait(5)")

        Returns:
            Canonical action description
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
        """Get initial environment description.

        Returns:
            Generic initial description (no specific state)
        """
        return "You are in a laboratory with a pot on a stove. You can measure temperature, wait, touch the pot, or toggle the stove."
