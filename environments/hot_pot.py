# environments/hot_pot.py
from typing import Literal, Optional
from dataclasses import dataclass
import numpy as np
from environments.base import Environment

ActionType = Literal["measure_temp", "wait", "touch_pot", "toggle_stove"]

@dataclass
class HotPotState:
    """Internal state representation"""
    stove_power: Literal["off", "low", "high"]
    pot_temp: float
    time_elapsed: float
    base_temp: float = 20.0

    def copy(self):
        return HotPotState(
            stove_power=self.stove_power,
            pot_temp=self.pot_temp,
            time_elapsed=self.time_elapsed,
            base_temp=self.base_temp
        )


class HotPotLab(Environment):
    """
    Test causal understanding vs misleading linguistic priors.

    The environment presents misleading labels but accurate measurements.
    Agents must learn to trust observations over initial descriptions.
    """

    # Physical constants
    HEATING_RATES = {
        "off": 0.0,
        "low": 1.0,  # C per second
        "high": 2.5
    }
    MEASUREMENT_NOISE_STD = 2.0
    BASE_TEMP = 20.0
    BURN_THRESHOLD = 60.0
    BURN_PENALTY = -10.0
    WAIT_PENALTY = -0.1  # Small penalty per second to encourage efficiency

    def __init__(self, seed: int):
        super().__init__(seed)
        self.rng = np.random.RandomState(seed)
        self.state: Optional[HotPotState] = None

    def reset(self, seed: int) -> dict:
        """
        Reset environment with misleading initial observation.
        The label says "Boiling!" but pot is actually cold.
        """
        self.rng = np.random.RandomState(seed)

        # Hidden state: pot is actually cold, stove is off
        self.state = HotPotState(
            stove_power="off",
            pot_temp=self.BASE_TEMP,
            time_elapsed=0.0,
            base_temp=self.BASE_TEMP
        )

        # Misleading observation - tests if agent trusts labels over measurements
        obs = {
            'label': 'Boiling!',
            'stove_light': 'on',  # Misleading visual cue
            'time': 0.0
        }

        # Guard rail: ensure no ground truth leaked
        self._validate_observation(obs)

        return obs

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        """
        Execute action and return (observation, reward, done, info).

        Actions:
        - measure_temp(): Get noisy temperature measurement
        - wait(X): Wait X seconds (updates temperature)
        - touch_pot(): Touch pot (burns if hot)
        - toggle_stove(): Cycle through off -> low -> high -> off
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")

        reward = 0.0
        done = False
        info = {}

        # Parse action
        action = action.strip()

        if action == "measure_temp":
            obs = self._measure_temp()

        elif action.startswith("wait"):
            # Parse wait duration
            try:
                duration = float(action.replace("wait", "").strip("()"))
            except ValueError:
                duration = 1.0  # Default to 1 second

            obs = self._wait(duration)
            reward = self.WAIT_PENALTY * duration

        elif action == "touch_pot":
            obs, touch_reward = self._touch_pot()
            reward = touch_reward

        elif action == "toggle_stove":
            obs = self._toggle_stove()

        else:
            # Unknown action - return current state
            obs = {'time': self.state.time_elapsed, 'message': 'Unknown action'}

        # Guard rail: ensure no ground truth leaked
        self._validate_observation(obs)

        return obs, reward, done, info

    def get_ground_truth(self) -> dict:
        """
        Return hidden state for EVALUATION ONLY.
        Must never be accessible to agents.
        """
        if self.state is None:
            return {}

        return {
            'stove_power': self.state.stove_power,
            'actual_temp': self.state.pot_temp,
            'time': self.state.time_elapsed,
            'heating_rate': self.HEATING_RATES[self.state.stove_power]
        }

    def counterfactual_query(
        self,
        action_sequence: list[str],
        seed: int
    ) -> dict:
        """
        Simulate action_sequence WITHOUT side effects.

        Guarantees:
        1. Deterministic given seed
        2. Side-effect free (doesn't modify self.state)
        3. Returns final observation after sequence
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before counterfactual_query()")

        # Save current state
        saved_state = self.state.copy()
        saved_rng_state = self.rng.get_state()

        # Create temporary RNG for counterfactual
        self.rng = np.random.RandomState(seed)

        # Simulate actions
        final_obs = None
        for action in action_sequence:
            final_obs, _, _, _ = self.step(action)

        # Restore original state
        self.state = saved_state
        self.rng.set_state(saved_rng_state)

        return final_obs if final_obs is not None else {}

    def get_time_elapsed(self) -> float:
        """Return simulation time for belief likelihood calculations"""
        if self.state is None:
            return 0.0
        return self.state.time_elapsed

    # Private helper methods

    def _measure_temp(self) -> dict:
        """Return noisy temperature measurement"""
        true_temp = self.state.pot_temp
        measured_temp = true_temp + self.rng.normal(0, self.MEASUREMENT_NOISE_STD)

        return {
            'measured_temp': float(measured_temp),
            'time': self.state.time_elapsed,
            'action': 'measure_temp'
        }

    def _wait(self, duration: float) -> dict:
        """Wait for duration seconds, temperature evolves"""
        # Update temperature based on heating rate
        heating_rate = self.HEATING_RATES[self.state.stove_power]
        self.state.pot_temp += heating_rate * duration
        self.state.time_elapsed += duration

        return {
            'time': self.state.time_elapsed,
            'action': f'wait({duration})',
            'duration': duration
        }

    def _touch_pot(self) -> tuple[dict, float]:
        """Touch pot - burns if temperature > threshold"""
        true_temp = self.state.pot_temp

        if true_temp > self.BURN_THRESHOLD:
            sensation = 'burning'
            reward = self.BURN_PENALTY
        else:
            sensation = 'cool' if true_temp < 40 else 'warm'
            reward = 0.0

        return {
            'sensation': sensation,
            'time': self.state.time_elapsed,
            'action': 'touch_pot'
        }, reward

    def _toggle_stove(self) -> dict:
        """Cycle through stove power levels: off -> low -> high -> off"""
        transitions = {
            "off": "low",
            "low": "high",
            "high": "off"
        }

        old_power = self.state.stove_power
        self.state.stove_power = transitions[old_power]

        # Return observable indicator (light pattern)
        light_pattern = {
            "off": "off",
            "low": "dim",
            "high": "bright"
        }

        return {
            'stove_light': light_pattern[self.state.stove_power],
            'time': self.state.time_elapsed,
            'action': 'toggle_stove'
        }

    def _save_state(self) -> HotPotState:
        """Save current state for counterfactual restoration"""
        return self.state.copy()

    def _restore_state(self, state: HotPotState):
        """Restore state from saved copy"""
        self.state = state

    def _validate_observation(self, obs: dict):
        """
        Guard rail: ensure observation never leaks ground truth.
        Raises AssertionError if ground truth is present.
        """
        forbidden_keys = ['ground_truth', 'hidden_state', 'actual_temp',
                         'stove_power', 'heating_rate', 'true_temp']

        for key in forbidden_keys:
            assert key not in obs, f"Ground truth leaked: {key} in observation"
