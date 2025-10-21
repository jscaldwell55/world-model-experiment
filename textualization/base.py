"""Base class for textualization layers.

Converts environment observations to canonical natural language strings.
Ensures deterministic mapping and no ground truth leakage.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set


class TextualizationLayer(ABC):
    """Abstract base class for converting environment states to canonical text.

    Guarantees:
    1. Deterministic: Same observation → same text, always
    2. No leakage: Hidden state never appears in text
    3. Canonical: Standardized formatting for numerical values
    """

    def __init__(self):
        """Initialize textualization layer."""
        self._forbidden_keys: Set[str] = set()

    @abstractmethod
    def textualize_observation(self, obs: Dict) -> str:
        """Convert observation to canonical string.

        Args:
            obs: Observation dictionary from environment

        Returns:
            Canonical natural language string

        Raises:
            ValueError: If observation contains forbidden keys
        """
        pass

    @abstractmethod
    def textualize_action(self, action: str) -> str:
        """Convert action to canonical sentence.

        Args:
            action: Action string (e.g., "measure_temp", "wait(5)")

        Returns:
            Canonical action description (e.g., "Action taken: measure_temp()")
        """
        pass

    @abstractmethod
    def get_initial_description(self) -> str:
        """Get environment description without hidden state.

        Returns:
            Initial environment description visible to agent
        """
        pass

    def validate_determinism(self, obs: Dict, num_trials: int = 10) -> bool:
        """Check that same observation produces same text consistently.

        Args:
            obs: Observation dictionary to test
            num_trials: Number of times to generate text

        Returns:
            True if all generations match, False otherwise
        """
        texts = [self.textualize_observation(obs) for _ in range(num_trials)]
        return len(set(texts)) == 1

    def validate_no_leakage(self, text: str) -> bool:
        """Check that forbidden keys do not appear in text.

        Args:
            text: Generated text to validate

        Returns:
            True if no forbidden keys found, False otherwise
        """
        text_lower = text.lower()
        for key in self._forbidden_keys:
            # Check for exact word matches to avoid false positives
            if key.lower() in text_lower:
                return False
        return True

    def set_forbidden_keys(self, keys: List[str]) -> None:
        """Set list of forbidden keys that must never appear in text.

        Args:
            keys: List of forbidden key names
        """
        self._forbidden_keys = set(keys)

    def get_forbidden_keys(self) -> Set[str]:
        """Get set of forbidden keys.

        Returns:
            Set of forbidden key names
        """
        return self._forbidden_keys.copy()
