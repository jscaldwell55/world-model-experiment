# environments/chem_tile.py
from typing import Literal, Optional
from dataclasses import dataclass
import numpy as np
from environments.base import Environment

ActionType = Literal["mix", "heat", "cool", "inspect"]
Temperature = Literal["low", "medium", "high"]
Compound = Literal["A", "B", "C", "D", "nothing", "explode"]

@dataclass
class ChemTileState:
    """Internal state representation"""
    temperature: Temperature
    available_compounds: list[str]
    last_reaction: Optional[str] = None
    explosion_count: int = 0

    def copy(self):
        return ChemTileState(
            temperature=self.temperature,
            available_compounds=self.available_compounds.copy(),
            last_reaction=self.last_reaction,
            explosion_count=self.explosion_count
        )


class ChemTile(Environment):
    """
    Test compositional reasoning with safety constraints.

    Reaction table (temperature-dependent):
    - A + B -> C (80% success at medium temp)
    - C + B -> D (70% success at low temp)
    - Higher temperature increases explosion risk
    - Lower temperature reduces reaction success

    Goal: Produce compound D without explosions.
    """

    # Base reaction probabilities at medium temperature
    BASE_REACTIONS = {
        'A+B': {'C': 0.80, 'explode': 0.10, 'nothing': 0.10},
        'C+B': {'D': 0.70, 'explode': 0.20, 'nothing': 0.10},
        'A+C': {'nothing': 0.90, 'explode': 0.10},
        'A+D': {'nothing': 0.95, 'explode': 0.05},
    }

    # Temperature modifiers
    TEMP_MODIFIERS = {
        'low': {'success': 0.7, 'explode': 0.5, 'nothing': 1.3},
        'medium': {'success': 1.0, 'explode': 1.0, 'nothing': 1.0},
        'high': {'success': 1.2, 'explode': 2.0, 'nothing': 0.5}
    }

    EXPLOSION_PENALTY = -5.0
    SUCCESS_REWARD = 10.0  # For creating D

    def __init__(self, seed: int):
        super().__init__(seed)
        self.rng = np.random.RandomState(seed)
        self.state: Optional[ChemTileState] = None
        self.time_elapsed = 0.0

    def reset(self, seed: int) -> dict:
        """
        Reset environment with initial compounds A and B.
        Goal: produce compound D safely.
        """
        self.rng = np.random.RandomState(seed)
        self.time_elapsed = 0.0

        # Initial state: compounds A and B available, medium temperature
        self.state = ChemTileState(
            temperature="medium",
            available_compounds=["A", "B", "B"],  # Two B's available
            last_reaction=None,
            explosion_count=0
        )

        obs = {
            'available_compounds': self.state.available_compounds.copy(),
            'temperature': self.state.temperature,
            'message': 'Chemistry lab initialized. Handle compounds carefully.',
            'time': 0.0
        }

        self._validate_observation(obs)
        return obs

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        """
        Execute action and return (observation, reward, done, info).

        Actions:
        - mix(compound_a, compound_b): Attempt reaction
        - heat(): Increase temperature
        - cool(): Decrease temperature
        - inspect(compound): Get information about a compound
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")

        reward = 0.0
        done = False
        info = {}

        action = action.strip()
        self.time_elapsed += 1.0

        # DEBUG: Log action execution
        import os
        if os.environ.get('DEBUG_CHEMTILE'):
            print(f"[ChemTile.step] Action: {repr(action)}")
            print(f"[ChemTile.step] Available BEFORE: {self.state.available_compounds}")

        if action.startswith("mix"):
            obs, mix_reward = self._mix_compounds(action)
            reward = mix_reward

            # Check if produced D (goal)
            if 'D' in self.state.available_compounds:
                done = True

        elif action == "heat":
            obs = self._heat()

        elif action == "cool":
            obs = self._cool()

        elif action.startswith("inspect"):
            obs = self._inspect(action)

        else:
            obs = {'time': self.time_elapsed, 'message': 'Unknown action'}

        # DEBUG: Log results
        if os.environ.get('DEBUG_CHEMTILE'):
            print(f"[ChemTile.step] Available AFTER: {self.state.available_compounds}")
            print(f"[ChemTile.step] Observation: {obs}")

        self._validate_observation(obs)
        return obs, reward, done, info

    def get_ground_truth(self) -> dict:
        """Return hidden state for EVALUATION ONLY."""
        if self.state is None:
            return {}

        return {
            'temperature': self.state.temperature,
            'available_compounds': self.state.available_compounds.copy(),
            'last_reaction': self.state.last_reaction,
            'explosion_count': self.state.explosion_count,
            'time': self.time_elapsed
        }

    def counterfactual_query(
        self,
        action_sequence: list[str],
        seed: int
    ) -> dict:
        """
        Simulate action_sequence WITHOUT side effects.
        Used to answer: "What if I mixed A+B at high temperature?"
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before counterfactual_query()")

        # Save current state
        saved_state = self.state.copy()
        saved_rng_state = self.rng.get_state()
        saved_time = self.time_elapsed

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

        return final_obs if final_obs is not None else {}

    def get_time_elapsed(self) -> float:
        """Return simulation time for belief likelihood calculations"""
        return self.time_elapsed

    # Private helper methods

    def _mix_compounds(self, action: str) -> tuple[dict, float]:
        """
        Mix two compounds according to reaction table.
        Returns observation and reward.
        """
        # Parse compounds from action string like "mix(A, B)" or "mix('A', 'B')"
        try:
            compounds_str = action.replace("mix", "").strip("()")
            parts = [p.strip().strip("'\"") for p in compounds_str.split(",")]
            compound_a, compound_b = parts[0], parts[1]
        except (ValueError, IndexError):
            return {
                'message': 'Invalid mix command format. Use: mix(A, B)',
                'time': self.time_elapsed
            }, 0.0

        # Check if compounds are available
        if compound_a not in self.state.available_compounds:
            return {
                'message': f'Compound {compound_a} not available.',
                'available_compounds': self.state.available_compounds.copy(),
                'time': self.time_elapsed
            }, 0.0

        if compound_b not in self.state.available_compounds:
            return {
                'message': f'Compound {compound_b} not available.',
                'available_compounds': self.state.available_compounds.copy(),
                'time': self.time_elapsed
            }, 0.0

        # Perform reaction
        reaction_key = f"{compound_a}+{compound_b}"
        outcome, reward = self._simulate_reaction(reaction_key)

        # Update state based on outcome
        if outcome == "explode":
            self.state.explosion_count += 1
            # Remove one of each compound involved
            self.state.available_compounds.remove(compound_a)
            if compound_b in self.state.available_compounds:
                self.state.available_compounds.remove(compound_b)

            obs = {
                'reaction': reaction_key,
                'outcome': 'explode',
                'message': 'EXPLOSION! Compounds destroyed.',
                'available_compounds': self.state.available_compounds.copy(),
                'temperature': self.state.temperature,
                'time': self.time_elapsed
            }

        elif outcome == "nothing":
            # Compounds consumed but no product
            self.state.available_compounds.remove(compound_a)
            if compound_b in self.state.available_compounds:
                self.state.available_compounds.remove(compound_b)

            obs = {
                'reaction': reaction_key,
                'outcome': 'nothing',
                'message': 'Reaction fizzled. No product formed.',
                'available_compounds': self.state.available_compounds.copy(),
                'temperature': self.state.temperature,
                'time': self.time_elapsed
            }

        else:  # Successful reaction
            # Remove reactants
            self.state.available_compounds.remove(compound_a)
            if compound_b in self.state.available_compounds:
                self.state.available_compounds.remove(compound_b)

            # Add product
            self.state.available_compounds.append(outcome)

            obs = {
                'reaction': reaction_key,
                'outcome': outcome,
                'message': f'Success! Produced compound {outcome}.',
                'available_compounds': self.state.available_compounds.copy(),
                'temperature': self.state.temperature,
                'time': self.time_elapsed
            }

        self.state.last_reaction = reaction_key
        return obs, reward

    def _simulate_reaction(self, reaction_key: str) -> tuple[str, float]:
        """
        Simulate reaction outcome based on probabilities and temperature.
        Returns (outcome, reward).
        """
        # Get base probabilities
        if reaction_key not in self.BASE_REACTIONS:
            # Unknown reaction - mostly nothing with small explosion risk
            outcomes = ['nothing', 'explode']
            probs = [0.90, 0.10]
        else:
            base_probs = self.BASE_REACTIONS[reaction_key]
            outcomes = list(base_probs.keys())
            probs = list(base_probs.values())

        # Apply temperature modifiers
        temp_mod = self.TEMP_MODIFIERS[self.state.temperature]
        modified_probs = []

        for outcome, prob in zip(outcomes, probs):
            if outcome == 'explode':
                modified_probs.append(prob * temp_mod['explode'])
            elif outcome == 'nothing':
                modified_probs.append(prob * temp_mod['nothing'])
            else:  # Successful product
                modified_probs.append(prob * temp_mod['success'])

        # Normalize probabilities
        total = sum(modified_probs)
        normalized_probs = [p / total for p in modified_probs]

        # Sample outcome
        outcome = self.rng.choice(outcomes, p=normalized_probs)

        # Determine reward
        if outcome == 'explode':
            reward = self.EXPLOSION_PENALTY
        elif outcome == 'D':
            reward = self.SUCCESS_REWARD
        else:
            reward = 0.0

        return outcome, reward

    def _heat(self) -> dict:
        """Increase temperature"""
        temp_order = ['low', 'medium', 'high']
        current_idx = temp_order.index(self.state.temperature)

        if current_idx < len(temp_order) - 1:
            self.state.temperature = temp_order[current_idx + 1]
            message = f"Temperature increased to {self.state.temperature}."
        else:
            message = "Already at maximum temperature."

        return {
            'action': 'heat',
            'temperature': self.state.temperature,
            'message': message,
            'time': self.time_elapsed
        }

    def _cool(self) -> dict:
        """Decrease temperature"""
        temp_order = ['low', 'medium', 'high']
        current_idx = temp_order.index(self.state.temperature)

        if current_idx > 0:
            self.state.temperature = temp_order[current_idx - 1]
            message = f"Temperature decreased to {self.state.temperature}."
        else:
            message = "Already at minimum temperature."

        return {
            'action': 'cool',
            'temperature': self.state.temperature,
            'message': message,
            'time': self.time_elapsed
        }

    def _inspect(self, action: str) -> dict:
        """Get information about a specific compound"""
        try:
            compound = action.replace("inspect", "").strip("()").strip("'\"")
        except:
            compound = None

        if not compound or compound not in ['A', 'B', 'C', 'D']:
            return {
                'message': 'Invalid inspect command. Use: inspect(A), inspect(B), etc.',
                'time': self.time_elapsed
            }

        # Provide information about compound properties
        compound_info = {
            'A': 'Base reagent. Stable but reactive with B.',
            'B': 'Catalyst compound. Required for multiple reactions.',
            'C': 'Intermediate product. Can be further reacted.',
            'D': 'Target compound. Goal of synthesis.'
        }

        return {
            'action': f'inspect({compound})',
            'compound': compound,
            'info': compound_info[compound],
            'time': self.time_elapsed
        }

    def _validate_observation(self, obs: dict):
        """Guard rail: ensure observation never leaks ground truth."""
        # Note: available_compounds IS observable (agent can see what they have)
        # Temperature IS observable (there's a thermometer)
        # But explosion_count and reaction probabilities are not

        forbidden_keys = ['ground_truth', 'hidden_state', 'explosion_count',
                         'reaction_probs', 'true_probabilities']

        for key in forbidden_keys:
            assert key not in obs, f"Ground truth leaked: {key} in observation"
