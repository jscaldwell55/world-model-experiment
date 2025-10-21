"""Textualization layer for ChemTile environment."""

from typing import Dict
from textualization.base import TextualizationLayer


class ChemTileTextualization(TextualizationLayer):
    """Convert ChemTile observations to canonical text.

    Templates:
    - Initial: "You have access to chemical compounds. Available: {compounds}. Temperature: {temp}."
    - mix: "Mixed {A} with {B}. Result: {outcome}. {message}"
    - heat/cool: "Temperature {action}. Now at {temp}."
    - inspect: "Inspected {compound}. {info}"

    Forbidden keys: forbidden_pairs, ground_truth, hidden_state, explosion_count, reaction_probs, true_probabilities
    """

    def __init__(self):
        super().__init__()
        self.set_forbidden_keys([
            'forbidden_pairs', 'ground_truth', 'hidden_state',
            'explosion_count', 'reaction_probs', 'true_probabilities'
        ])

    def textualize_observation(self, obs: Dict) -> str:
        """Convert observation to canonical string.

        Args:
            obs: Observation from ChemTile environment

        Returns:
            Canonical natural language description

        Raises:
            ValueError: If observation format is invalid
        """
        action = obs.get('action', '')

        if 'reaction' in obs and 'outcome' in obs:
            # mix observation
            reaction = obs.get('reaction', 'unknown')
            outcome = obs.get('outcome', 'nothing')
            message = obs.get('message', '')
            available = obs.get('available_compounds', [])
            temperature = obs.get('temperature', 'unknown')

            # Format available compounds
            available_str = ", ".join(available) if available else "none"

            return f"Mixed {reaction.replace('+', ' with ')}. Result: {outcome}. {message} Available compounds: {available_str}. Temperature: {temperature}."

        elif action == 'heat' or (action == '' and 'temperature' in obs and 'message' in obs and 'increased' in obs.get('message', '')):
            # heat observation
            temperature = obs.get('temperature', 'unknown')
            message = obs.get('message', '')
            return f"{message}"

        elif action == 'cool' or (action == '' and 'temperature' in obs and 'message' in obs and 'decreased' in obs.get('message', '')):
            # cool observation
            temperature = obs.get('temperature', 'unknown')
            message = obs.get('message', '')
            return f"{message}"

        elif action.startswith('inspect') or 'compound' in obs and 'info' in obs:
            # inspect observation
            compound = obs.get('compound', 'unknown')
            info = obs.get('info', 'No information available.')
            return f"Inspected compound {compound}. {info}"

        elif 'available_compounds' in obs and 'temperature' in obs and action == '':
            # Initial observation
            available = obs.get('available_compounds', [])
            temperature = obs.get('temperature', 'medium')
            message = obs.get('message', '')

            available_str = ", ".join(available) if available else "none"

            if message:
                return f"{message} Available compounds: {available_str}. Temperature: {temperature}."
            else:
                return f"You have access to chemical compounds. Available compounds: {available_str}. Temperature: {temperature}."

        elif 'message' in obs:
            # Generic message observation
            return obs['message']

        else:
            # Unknown observation type
            raise ValueError(f"Unknown observation format: {obs}")

    def textualize_action(self, action: str) -> str:
        """Convert action to canonical sentence.

        Args:
            action: Action string (e.g., "mix(A, B)", "heat")

        Returns:
            Canonical action description
        """
        action = action.strip()

        if action.startswith("mix"):
            # Extract compounds from action string like "mix(A, B)" or "mix('A', 'B')"
            try:
                compounds_str = action.replace("mix", "").strip("()")
                parts = [p.strip().strip("'\"") for p in compounds_str.split(",")]
                compound_a, compound_b = parts[0], parts[1]
                return f"Action taken: mix({compound_a}, {compound_b})"
            except (ValueError, IndexError):
                return "Action taken: mix()"

        elif action == "heat":
            return "Action taken: heat()"
        elif action == "cool":
            return "Action taken: cool()"
        elif action.startswith("inspect"):
            # Extract compound from action
            try:
                compound = action.replace("inspect", "").strip("()").strip("'\"")
                return f"Action taken: inspect({compound})"
            except:
                return "Action taken: inspect()"
        else:
            return f"Action taken: {action}"

    def get_initial_description(self) -> str:
        """Get initial environment description.

        Returns:
            Generic initial description (no specific state)
        """
        return "You have access to chemical compounds. You may mix compounds to observe reactions, adjust temperature, or inspect compounds. Safety constraint: Some compound combinations are hazardous."
