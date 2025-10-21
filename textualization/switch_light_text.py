"""Textualization layer for SwitchLight environment."""

from typing import Dict
from textualization.base import TextualizationLayer


class SwitchLightTextualization(TextualizationLayer):
    """Convert SwitchLight observations to canonical text.

    Templates:
    - Initial: "You are in a room with a light switch and a light bulb..."
    - flip_switch: "Switch toggled. Switch is now {position}. Bulb is {state}."
    - observe_light: "Observed without touching. Switch is {position}. Bulb is {state}."
    - jiggle_relay: Message from observation
    - inspect_wires: Hint from observation

    Forbidden keys: broken, connection, ground_truth, wire_layout, faulty_relay, true_layout, actual_wiring
    """

    def __init__(self):
        super().__init__()
        self.set_forbidden_keys([
            'broken', 'connection', 'ground_truth',
            'wire_layout', 'faulty_relay', 'true_layout', 'actual_wiring'
        ])

    def textualize_observation(self, obs: Dict) -> str:
        """Convert observation to canonical string.

        Args:
            obs: Observation from SwitchLight environment

        Returns:
            Canonical natural language description

        Raises:
            ValueError: If observation format is invalid
        """
        action = obs.get('action', '')

        if action == 'flip_switch' or (action == '' and 'light_on' in obs and 'switch_position' in obs and 'message' not in obs):
            # flip_switch observation
            switch_position = obs.get('switch_position', 'unknown')
            light_on = obs.get('light_on', False)
            bulb_state = "on" if light_on else "off"

            return f"Switch toggled. Switch is now {switch_position}. Bulb is {bulb_state}."

        elif action == 'observe_light':
            # observe_light observation
            switch_position = obs.get('switch_position', 'unknown')
            light_on = obs.get('light_on', False)
            bulb_state = "on" if light_on else "off"

            return f"Observed without touching. Switch is {switch_position}. Bulb is {bulb_state}."

        elif action == 'jiggle_relay' or (action == '' and 'message' in obs and 'hint' not in obs and 'switch_position' not in obs):
            # jiggle_relay observation
            message = obs.get('message', 'No change observed.')
            return message

        elif action == 'inspect_wires' or 'hint' in obs:
            # inspect_wires observation
            hint = obs.get('hint', 'Wiring is complex.')
            inspection_count = obs.get('inspection_count', 1)
            return f"Inspection #{inspection_count}: {hint}"

        elif 'switch_position' in obs and 'message' in obs and action == '':
            # Initial observation
            switch_position = obs.get('switch_position', 'off')
            message = obs.get('message', '')
            return f"You are in a room with a light switch and a light bulb. The switch is currently {switch_position}. {message}"

        elif 'message' in obs:
            # Generic message observation
            return obs['message']

        else:
            # Unknown observation type
            raise ValueError(f"Unknown observation format: {obs}")

    def textualize_action(self, action: str) -> str:
        """Convert action to canonical sentence.

        Args:
            action: Action string (e.g., "flip_switch", "observe_light")

        Returns:
            Canonical action description
        """
        action = action.strip()

        if action == "flip_switch":
            return "Action taken: flip_switch()"
        elif action == "observe_light":
            return "Action taken: observe_light()"
        elif action == "jiggle_relay":
            return "Action taken: jiggle_relay()"
        elif action == "inspect_wires":
            return "Action taken: inspect_wires()"
        else:
            return f"Action taken: {action}"

    def get_initial_description(self) -> str:
        """Get initial environment description.

        Returns:
            Generic initial description (no specific state)
        """
        return "You are in a room with a light switch and a light bulb. You can flip the switch, observe the light, jiggle the relay, or inspect the wires."
