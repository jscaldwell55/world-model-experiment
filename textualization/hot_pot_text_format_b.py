"""Format B textualization layer for HotPot environment.

This variant uses abbreviated formatting with different units/precision.
Designed to test whether anti-coupling is sensitive to formatting changes.
"""

from typing import Dict
from textualization.base import TextualizationLayer


class HotPotTextualizationFormatB(TextualizationLayer):
    """Format B variant: Abbreviated formatting, no decimals.

    Templates:
    - Initial: "Lab: pot on stove. Label: '{label}'. Stove: {on/off}."
    - measure_temp: "Temp: {temp} C. Time: {time} sec."
    - wait: "Time: {time} sec."
    - touch_pot: "Touch: {sensation}. Time: {time} sec."
    - toggle_stove: "Stove now: {on/off}. Time: {time} sec."

    Key changes from original:
    - Abbreviated words (Temp, Time, sec, C instead of °C)
    - No decimal precision (21 C instead of 21.0°C)
    - Shorter sentence structure
    - Keeps the "Boiling!" label (to test format vs content)
    """

    def __init__(self):
        super().__init__()
        self.set_forbidden_keys([
            'actual_temp', 'stove_power', 'ground_truth',
            'hidden_state', 'heating_rate', 'true_temp'
        ])

    def textualize_observation(self, obs: Dict) -> str:
        """Convert observation to abbreviated canonical string.

        Args:
            obs: Observation from HotPot environment

        Returns:
            Abbreviated canonical natural language description

        Raises:
            ValueError: If observation format is invalid
        """
        action = obs.get('action', '')

        if action == 'measure_temp' or 'measured_temp' in obs:
            # measure_temp observation (abbreviated format)
            measured_temp = obs.get('measured_temp')
            time = obs.get('time', 0.0)

            if measured_temp is None:
                raise ValueError("measure_temp observation missing 'measured_temp' key")

            # No decimals, abbreviated units
            return f"Temp: {int(round(measured_temp))} C. Time: {int(time)} sec."

        elif action.startswith('wait') or 'duration' in obs:
            # wait observation (abbreviated format)
            time = obs.get('time', 0.0)
            return f"Time: {int(time)} sec."

        elif action == 'touch_pot' or 'sensation' in obs:
            # touch_pot observation (abbreviated format)
            sensation = obs.get('sensation', 'unknown')
            time = obs.get('time', 0.0)
            return f"Touch: {sensation}. Time: {int(time)} sec."

        elif action == 'toggle_stove' or ('stove_light' in obs and 'label' not in obs):
            # toggle_stove observation (abbreviated format)
            stove_light = obs.get('stove_light', 'off')
            time = obs.get('time', 0.0)
            return f"Stove now: {stove_light}. Time: {int(time)} sec."

        elif 'label' in obs and 'stove_light' in obs:
            # Initial observation (abbreviated format, KEEPS label)
            label = obs.get('label', 'Unknown')
            stove_light = obs.get('stove_light', 'off')

            # Abbreviated but keeps the misleading label
            return f"Lab: pot on stove. Label: '{label}'. Stove: {stove_light}."

        elif 'message' in obs:
            # Generic message observation
            return obs['message']

        else:
            # Unknown observation type
            raise ValueError(f"Unknown observation format: {obs}")

    def textualize_action(self, action: str) -> str:
        """Convert action to abbreviated canonical sentence.

        Args:
            action: Action string (e.g., "measure_temp", "wait(5)")

        Returns:
            Abbreviated canonical action description
        """
        action = action.strip()

        if action == "measure_temp":
            return "Action taken: measure_temp()"
        elif action.startswith("wait"):
            # Extract duration if present
            try:
                duration_str = action.replace("wait", "").strip("()")
                duration = float(duration_str)
                return f"Action taken: wait({int(duration)})"
            except (ValueError, AttributeError):
                return "Action taken: wait(1)"
        elif action == "touch_pot":
            return "Action taken: touch_pot()"
        elif action == "toggle_stove":
            return "Action taken: toggle_stove()"
        else:
            return f"Action taken: {action}"

    def get_initial_description(self) -> str:
        """Get initial environment description (abbreviated format).

        Returns:
            Abbreviated generic initial description (no specific state)
        """
        return "Lab setup: pot, stove. Actions: measure temp, wait, touch pot, toggle stove."
