"""Negative control textualizations to test for spurious coupling.

These textualizers are designed to break the semantic connection between
text and observations while preserving surface statistical properties
(vocabulary distribution, sentence length, etc.).

If coupling remains high with these controls, it suggests spurious correlation
rather than genuine world model learning.
"""

import random
import re
from typing import Dict, List
from textualization.base import TextualizationLayer


class ShuffledTextualization(TextualizationLayer):
    """Shuffles word order while preserving words.

    This control breaks semantics while maintaining:
    - Vocabulary distribution
    - Word frequencies
    - Sentence length

    If LLM couples to shuffled text, coupling is likely spurious.

    Example:
        Original: "Thermometer reads 25.0°C. Time elapsed: 10 seconds."
        Shuffled: "elapsed reads Time seconds 10 25.0°C. Thermometer ."
    """

    def __init__(self, base_textualizer: TextualizationLayer, seed: int = 42):
        """Initialize shuffled textualization wrapper.

        Args:
            base_textualizer: Underlying textualizer to wrap
            seed: Random seed for reproducible shuffling
        """
        super().__init__()
        self.base = base_textualizer
        self.rng = random.Random(seed)
        self._forbidden_keys = base_textualizer.get_forbidden_keys()

    def textualize_observation(self, obs: Dict) -> str:
        """Generate observation text, then shuffle words.

        Args:
            obs: Observation dictionary

        Returns:
            Shuffled text with broken semantics
        """
        # Get normal text from base textualizer
        normal_text = self.base.textualize_observation(obs)

        # Shuffle words while preserving punctuation attachment
        return self._shuffle_text(normal_text)

    def textualize_action(self, action: str) -> str:
        """Generate action text, then shuffle words.

        Args:
            action: Action string

        Returns:
            Shuffled action description
        """
        normal_text = self.base.textualize_action(action)
        return self._shuffle_text(normal_text)

    def get_initial_description(self) -> str:
        """Get shuffled initial description.

        Returns:
            Shuffled environment description
        """
        normal_text = self.base.get_initial_description()
        return self._shuffle_text(normal_text)

    def _shuffle_text(self, text: str) -> str:
        """Shuffle word order while preserving punctuation.

        Strategy:
        1. Tokenize into words and punctuation
        2. Shuffle only the words
        3. Rejoin preserving original spacing

        Args:
            text: Original text

        Returns:
            Shuffled text
        """
        # Split into tokens (words and punctuation)
        # Pattern: word characters, numbers, or single punctuation
        tokens = re.findall(r'\w+[\.\d]*|[^\w\s]', text)

        if not tokens:
            return text

        # Separate words from punctuation
        words = []
        non_words = []
        word_positions = []

        for i, token in enumerate(tokens):
            if re.match(r'\w', token):  # Is a word or number
                words.append(token)
                word_positions.append(i)
            else:
                non_words.append((i, token))

        # Shuffle only the words
        shuffled_words = words.copy()
        self.rng.shuffle(shuffled_words)

        # Reconstruct with shuffled words
        result_tokens = tokens.copy()
        for pos, word in zip(word_positions, shuffled_words):
            result_tokens[pos] = word

        # Rejoin with spacing
        result = ' '.join(result_tokens)

        # Clean up spacing around punctuation
        result = re.sub(r'\s+([.,;:!?])', r'\1', result)
        result = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', result)

        return result

    def validate_shuffling(self, text: str, shuffled: str) -> Dict[str, bool]:
        """Validate that shuffling preserves vocabulary but changes semantics.

        Args:
            text: Original text
            shuffled: Shuffled text

        Returns:
            Dict with validation results:
                same_words: True if word sets are identical
                different_order: True if order changed
                same_length: True if token count preserved
        """
        original_words = sorted(re.findall(r'\w+', text.lower()))
        shuffled_words = sorted(re.findall(r'\w+', shuffled.lower()))

        return {
            'same_words': original_words == shuffled_words,
            'different_order': text != shuffled,
            'same_length': len(original_words) == len(shuffled_words)
        }


class RandomSubstitutionTextualization(TextualizationLayer):
    """Replaces observations with random valid observations from same environment.

    This control tests whether coupling depends on actual observation content
    or just statistical properties of the text distribution.

    Strategy:
        - Maintain a cache of valid observations per environment
        - When asked to textualize, return a random cached observation instead
        - Cache is seeded, so results are reproducible

    Example:
        True obs: "Thermometer reads 25.0°C"
        Random:   "Thermometer reads 87.3°C" (from different episode)
    """

    def __init__(
        self,
        base_textualizer: TextualizationLayer,
        seed: int = 42,
        cache_size: int = 100
    ):
        """Initialize random substitution wrapper.

        Args:
            base_textualizer: Underlying textualizer
            seed: Random seed for reproducible substitution
            cache_size: Maximum observations to cache
        """
        super().__init__()
        self.base = base_textualizer
        self.rng = random.Random(seed)
        self.cache: List[str] = []
        self.cache_size = cache_size
        self._forbidden_keys = base_textualizer.get_forbidden_keys()

    def textualize_observation(self, obs: Dict) -> str:
        """Generate normal text, add to cache, return random cached text.

        Args:
            obs: Observation dictionary

        Returns:
            Random previously-seen observation text
        """
        # Generate true observation text
        true_text = self.base.textualize_observation(obs)

        # Add to cache (maintain max size)
        if true_text not in self.cache:
            self.cache.append(true_text)
            if len(self.cache) > self.cache_size:
                self.cache.pop(0)  # Remove oldest

        # Return random cached observation
        if len(self.cache) > 1:
            # Don't return the text we just added (too easy)
            other_texts = [t for t in self.cache if t != true_text]
            if other_texts:
                return self.rng.choice(other_texts)

        # Fallback: return true text if cache too small
        return true_text

    def textualize_action(self, action: str) -> str:
        """Actions are NOT randomized (keeps temporal structure).

        We only randomize observations, not actions, to preserve
        the action sequence structure.

        Args:
            action: Action string

        Returns:
            Normal action description (NOT randomized)
        """
        return self.base.textualize_action(action)

    def get_initial_description(self) -> str:
        """Get normal initial description.

        Returns:
            Normal environment description (NOT randomized)
        """
        return self.base.get_initial_description()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cached observations.

        Returns:
            Dict with cache size and uniqueness stats
        """
        return {
            'cache_size': len(self.cache),
            'unique_observations': len(set(self.cache))
        }


def create_negative_control_textualizer(
    base_textualizer: TextualizationLayer,
    control_type: str,
    seed: int = 42
) -> TextualizationLayer:
    """Factory function for negative control textualizers.

    Args:
        base_textualizer: Normal textualizer to wrap
        control_type: 'shuffled' or 'random'
        seed: Random seed for reproducibility

    Returns:
        Negative control textualizer

    Raises:
        ValueError: If control_type is invalid
    """
    if control_type == 'shuffled':
        return ShuffledTextualization(base_textualizer, seed=seed)
    elif control_type == 'random':
        return RandomSubstitutionTextualization(base_textualizer, seed=seed)
    else:
        raise ValueError(
            f"Invalid control_type: {control_type}. "
            f"Must be 'shuffled' or 'random'."
        )
