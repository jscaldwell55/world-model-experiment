# environments/switch_light.py
from typing import Literal, Optional
from dataclasses import dataclass
import numpy as np
from environments.base import Environment

ActionType = Literal["flip_switch", "jiggle_relay", "inspect_wires", "observe_light"]

@dataclass
class SwitchLightState:
    """Internal state representation"""
    wire_layout: Literal["layout_A", "layout_B"]
    faulty_relay: bool
    switch_position: Literal["on", "off"]

    def copy(self):
        return SwitchLightState(
            wire_layout=self.wire_layout,
            faulty_relay=self.faulty_relay,
            switch_position=self.switch_position
        )


class SwitchLight(Environment):
    """
    Test intervention vs observation (do-calculus).

    Two possible wiring layouts:
    - Layout A: switch ON -> light ON (normal)
    - Layout B: switch OFF -> light ON (inverted)

    Relay can be faulty (10% chance), breaking the causal chain.
    Spurious correlations appear unless agent intervenes.
    """

    RELAY_FAILURE_RATE = 0.1
    JIGGLE_FIX_PROB = 0.6  # Probability that jiggling fixes faulty relay

    def __init__(self, seed: int):
        super().__init__(seed)
        self.rng = np.random.RandomState(seed)
        self.state: Optional[SwitchLightState] = None
        self.time_elapsed = 0.0
        self.inspection_count = 0  # Track how many inspections done

    def reset(self, seed: int) -> dict:
        """
        Reset environment with random wiring and relay state.
        Agent must discover the wiring through intervention.
        """
        self.rng = np.random.RandomState(seed)
        self.time_elapsed = 0.0
        self.inspection_count = 0

        # Randomly determine hidden state
        wire_layout = self.rng.choice(["layout_A", "layout_B"])
        faulty_relay = self.rng.random() < self.RELAY_FAILURE_RATE
        switch_position = self.rng.choice(["on", "off"])

        self.state = SwitchLightState(
            wire_layout=wire_layout,
            faulty_relay=faulty_relay,
            switch_position=switch_position
        )

        # Initial observation - just see the switch position
        obs = {
            'switch_position': switch_position,
            'time': 0.0,
            'message': 'Laboratory initialized. Switch and light system ready for testing.'
        }

        self._validate_observation(obs)
        return obs

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        """
        Execute action and return (observation, reward, done, info).

        Actions:
        - flip_switch(): Toggle switch position
        - jiggle_relay(): Attempt to fix faulty relay
        - inspect_wires(): Get partial information about wiring (costly)
        - observe_light(): Check if light is on
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")

        reward = 0.0
        done = False
        info = {}

        action = action.strip()
        self.time_elapsed += 1.0

        if action == "flip_switch":
            obs = self._flip_switch()

        elif action == "jiggle_relay":
            obs = self._jiggle_relay()

        elif action == "inspect_wires":
            obs = self._inspect_wires()
            reward = -1.0  # Costly action

        elif action == "observe_light":
            obs = self._observe_light()

        else:
            obs = {'time': self.time_elapsed, 'message': 'Unknown action'}

        self._validate_observation(obs)
        return obs, reward, done, info

    def get_ground_truth(self) -> dict:
        """Return hidden state for EVALUATION ONLY."""
        if self.state is None:
            return {}

        return {
            'wire_layout': self.state.wire_layout,
            'faulty_relay': self.state.faulty_relay,
            'switch_position': self.state.switch_position,
            'time': self.time_elapsed,
            'light_should_be_on': self._compute_light_state()
        }

    def counterfactual_query(
        self,
        action_sequence: list[str],
        seed: int
    ) -> dict:
        """
        Simulate action_sequence WITHOUT side effects.
        Used to answer counterfactual questions like:
        "What if I had flipped the switch instead?"
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before counterfactual_query()")

        # Save current state
        saved_state = self.state.copy()
        saved_rng_state = self.rng.get_state()
        saved_time = self.time_elapsed
        saved_inspection_count = self.inspection_count

        # Create temporary RNG for counterfactual
        self.rng = np.random.RandomState(seed)

        # Simulate actions
        final_obs = None
        for action in action_sequence:
            final_obs, _, _, _ = self.step(action)

        # Restore original state
        self.state = saved_state
        self.rng.set_state(saved_rng_state)
        self.time_elapsed = saved_time
        self.inspection_count = saved_inspection_count

        return final_obs if final_obs is not None else {}

    def get_time_elapsed(self) -> float:
        """Return simulation time for belief likelihood calculations"""
        return self.time_elapsed

    # Private helper methods

    def _flip_switch(self) -> dict:
        """Toggle switch and observe result"""
        # Toggle switch
        self.state.switch_position = "off" if self.state.switch_position == "on" else "on"

        # Observe light
        light_on = self._compute_light_state()

        return {
            'action': 'flip_switch',
            'switch_position': self.state.switch_position,
            'light_on': light_on,
            'time': self.time_elapsed
        }

    def _jiggle_relay(self) -> dict:
        """Attempt to fix faulty relay by jiggling it"""
        was_faulty = self.state.faulty_relay

        if was_faulty and self.rng.random() < self.JIGGLE_FIX_PROB:
            self.state.faulty_relay = False
            message = "Relay clicked into place. Might have fixed something."
        else:
            message = "Jiggled the relay. No obvious change."

        return {
            'action': 'jiggle_relay',
            'message': message,
            'time': self.time_elapsed
        }

    def _inspect_wires(self) -> dict:
        """Get partial information about wiring (costly action)"""
        self.inspection_count += 1

        # On first inspection, give vague hint
        if self.inspection_count == 1:
            hint = "Wiring looks complex. Multiple configurations possible."
        # On second inspection, give stronger hint
        elif self.inspection_count == 2:
            if self.state.wire_layout == "layout_A":
                hint = "Standard wiring pattern observed."
            else:
                hint = "Unusual wire routing detected."
        # On third+ inspection, nearly reveal answer
        else:
            if self.state.wire_layout == "layout_A":
                hint = "Wiring follows conventional switch-to-light path."
            else:
                hint = "Wiring appears inverted from standard configuration."

        return {
            'action': 'inspect_wires',
            'hint': hint,
            'inspection_count': self.inspection_count,
            'time': self.time_elapsed
        }

    def _observe_light(self) -> dict:
        """Check if light is currently on"""
        light_on = self._compute_light_state()

        return {
            'action': 'observe_light',
            'light_on': light_on,
            'switch_position': self.state.switch_position,
            'time': self.time_elapsed
        }

    def _compute_light_state(self) -> bool:
        """
        Compute whether light should be on given current state.

        Logic:
        - Layout A: light = switch (normal)
        - Layout B: light = NOT switch (inverted)
        - If relay faulty: light is always off
        """
        if self.state.faulty_relay:
            return False

        switch_on = (self.state.switch_position == "on")

        if self.state.wire_layout == "layout_A":
            return switch_on
        else:  # layout_B is inverted
            return not switch_on

    def _validate_observation(self, obs: dict):
        """Guard rail: ensure observation never leaks ground truth."""
        forbidden_keys = ['ground_truth', 'hidden_state', 'wire_layout',
                         'faulty_relay', 'true_layout', 'actual_wiring']

        for key in forbidden_keys:
            assert key not in obs, f"Ground truth leaked: {key} in observation"
