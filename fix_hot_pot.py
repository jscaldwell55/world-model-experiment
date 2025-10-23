# fix_hot_pot.py

fix_content = '''# environments/hot_pot.py (FIXED)
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
    WAIT_PENALTY = -0.1
    INSTANT_ACTION_TIME = 1.0  # NEW: Time for instantaneous actions

    def __init__(self, seed: int):
        super().__init__(seed)
        self.rng = np.random.RandomState(seed)
        self.state: Optional[HotPotState] = None

    def reset(self, seed: int) -> dict:
        self.rng = np.random.RandomState(seed)
        self.state = HotPotState(
            stove_power="off",
            pot_temp=self.BASE_TEMP,
            time_elapsed=0.0,
            base_temp=self.BASE_TEMP
        )
        obs = {
            'label': 'Boiling!',
            'stove_light': 'on',
            'time': 0.0
        }
        self._validate_observation(obs)
        return obs

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")

        reward = 0.0
        done = False
        info = {}
        action = action.strip()

        if action == "measure_temp":
            obs = self._measure_temp()
            self._advance_time(self.INSTANT_ACTION_TIME)  # FIX

        elif action.startswith("wait"):
            try:
                duration = float(action.replace("wait", "").strip("()"))
            except ValueError:
                duration = 1.0
            obs = self._wait(duration)
            reward = self.WAIT_PENALTY * duration

        elif action == "touch_pot":
            obs, touch_reward = self._touch_pot()
            reward = touch_reward
            self._advance_time(self.INSTANT_ACTION_TIME)  # FIX

        elif action == "toggle_stove":
            obs = self._toggle_stove()
            self._advance_time(self.INSTANT_ACTION_TIME)  # FIX

        else:
            obs = {'time': self.state.time_elapsed, 'message': 'Unknown action'}

        self._validate_observation(obs)
        return obs, reward, done, info

    def get_ground_truth(self) -> dict:
        if self.state is None:
            return {}
        return {
            'stove_power': self.state.stove_power,
            'actual_temp': self.state.pot_temp,
            'time': self.state.time_elapsed,
            'heating_rate': self.HEATING_RATES[self.state.stove_power]
        }

    def counterfactual_query(self, action_sequence: list[str], seed: int) -> dict:
        if self.state is None:
            raise RuntimeError("Must call reset() before counterfactual_query()")
        saved_state = self.state.copy()
        saved_rng_state = self.rng.get_state()
        self.rng = np.random.RandomState(seed)
        final_obs = None
        for action in action_sequence:
            final_obs, _, _, _ = self.step(action)
        self.state = saved_state
        self.rng.set_state(saved_rng_state)
        return final_obs if final_obs is not None else {}

    def get_time_elapsed(self) -> float:
        if self.state is None:
            return 0.0
        return self.state.time_elapsed

    def _advance_time(self, duration: float):
        """NEW METHOD: Advance time and update temperature"""
        heating_rate = self.HEATING_RATES[self.state.stove_power]
        self.state.pot_temp += heating_rate * duration
        self.state.time_elapsed += duration

    def _measure_temp(self) -> dict:
        true_temp = self.state.pot_temp
        measured_temp = true_temp + self.rng.normal(0, self.MEASUREMENT_NOISE_STD)
        return {
            'measured_temp': float(measured_temp),
            'time': self.state.time_elapsed,
            'action': 'measure_temp'
        }

    def _wait(self, duration: float) -> dict:
        self._advance_time(duration)  # CHANGED: Use new method
        return {
            'time': self.state.time_elapsed,
            'action': f'wait({duration})',
            'duration': duration
        }

    def _touch_pot(self) -> tuple[dict, float]:
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
        transitions = {"off": "low", "low": "high", "high": "off"}
        old_power = self.state.stove_power
        self.state.stove_power = transitions[old_power]
        light_pattern = {"off": "off", "low": "dim", "high": "bright"}
        return {
            'stove_light': light_pattern[self.state.stove_power],
            'time': self.state.time_elapsed,
            'action': 'toggle_stove'
        }

    def _save_state(self) -> HotPotState:
        return self.state.copy()

    def _restore_state(self, state: HotPotState):
        self.state = state

    def _validate_observation(self, obs: dict):
        forbidden_keys = ['ground_truth', 'hidden_state', 'actual_temp',
                         'stove_power', 'heating_rate', 'true_temp']
        for key in forbidden_keys:
            assert key not in obs, f"Ground truth leaked: {key} in observation"
'''

# Backup original
import shutil
shutil.copy('environments/hot_pot.py', 'environments/hot_pot.py.backup')
print("✅ Backed up original to hot_pot.py.backup")

# Write fixed version
with open('environments/hot_pot.py', 'w') as f:
    f.write(fix_content)

print("✅ Applied fix to environments/hot_pot.py")
print("\nChanges:")
print("  1. Added INSTANT_ACTION_TIME = 1.0")
print("  2. Added _advance_time() method")
print("  3. measure_temp(), touch_pot(), toggle_stove() now advance time")
print("  4. All actions now properly update time_elapsed")