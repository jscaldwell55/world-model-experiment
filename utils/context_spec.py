"""
Context Specification System

Defines how to extract context keys from observations/episodes.

Context = the conditioning variables that affect world model predictions.
Example: In HotPot, power_setting (HIGH/LOW/OFF) determines heating rate.

Why needed: Prevents averaging beliefs across different contexts,
which destroys conditional information.
"""

from typing import Callable, Hashable, List, Dict, Any
from collections import Counter
import warnings


class ContextSpec:
    """
    Defines how to extract context keys from observations/episodes.

    Context = the conditioning variables that affect world model predictions.
    Example: In HotPot, power_setting (HIGH/LOW/OFF) determines heating rate.

    Why needed: Prevents averaging beliefs across different contexts,
    which destroys conditional information.
    """

    def __init__(self, name: str, key_fn: Callable[[dict], Hashable]):
        """
        Args:
            name: Domain name (e.g., 'hot_pot')
            key_fn: Function that extracts context key from observation/episode
                    Returns a hashable value (str, tuple, etc.)
        """
        self.name = name
        self.key_fn = key_fn

    def extract_context(self, observation: dict) -> Hashable:
        """
        Extract context key from observation.

        Args:
            observation: Observation dictionary or episode with context info

        Returns:
            Hashable context key
        """
        try:
            return self.key_fn(observation)
        except (KeyError, TypeError, AttributeError) as e:
            # If extraction fails, return default context
            warnings.warn(f"Failed to extract context from observation: {e}. "
                         f"Using 'UNKNOWN' context.")
            return "UNKNOWN"

    def validate_coverage(self, episodes: List[dict]) -> Dict[Hashable, int]:
        """
        Check if context distribution is reasonable.
        Warns if one context dominates (>80% of data).

        Args:
            episodes: List of episode dictionaries

        Returns:
            Dictionary mapping context_key to count
        """
        if not episodes:
            warnings.warn("No episodes provided for context coverage validation")
            return {}

        # Extract contexts from all episodes
        contexts = []
        for episode in episodes:
            try:
                # Handle both episode-level and observation-level contexts
                if 'context' in episode:
                    ctx = self.extract_context(episode)
                elif 'observations' in episode:
                    # Episode with nested observations
                    for obs in episode['observations']:
                        ctx = self.extract_context(obs)
                        contexts.append(ctx)
                    continue
                else:
                    # Try extracting directly
                    ctx = self.extract_context(episode)
                contexts.append(ctx)
            except Exception as e:
                warnings.warn(f"Error extracting context from episode: {e}")
                continue

        if not contexts:
            warnings.warn("No contexts could be extracted from episodes")
            return {}

        # Count contexts
        context_counts = Counter(contexts)
        total = len(contexts)

        # Check for domination (one context >80%)
        for context, count in context_counts.items():
            proportion = count / total
            if proportion > 0.8:
                warnings.warn(
                    f"Context '{context}' dominates with {proportion:.1%} of data. "
                    f"This may indicate insufficient context diversity."
                )

        # Check for very rare contexts (<5%)
        rare_contexts = [ctx for ctx, count in context_counts.items()
                        if count / total < 0.05]
        if rare_contexts:
            warnings.warn(
                f"Rare contexts detected (< 5% of data): {rare_contexts}. "
                f"Consider collecting more data for these contexts."
            )

        return dict(context_counts)


# Define specs for existing domains

def _hot_pot_context_extractor(obs: dict) -> Hashable:
    """Extract power_setting from HotPot observation"""
    # Handle both direct context and nested context
    if 'context' in obs and isinstance(obs['context'], dict):
        return obs['context'].get('power_setting', 'UNKNOWN')
    elif 'power_setting' in obs:
        return obs['power_setting']
    else:
        # Fallback - try to infer from observation structure
        return 'UNKNOWN'


def _switch_light_context_extractor(obs: dict) -> Hashable:
    """Extract switch configuration from SwitchLight observation"""
    if 'context' in obs and isinstance(obs['context'], dict):
        switch_id = obs['context'].get('switch_id', 'unknown')
        effectiveness = obs['context'].get('effectiveness', 'normal')
        return (switch_id, effectiveness)
    elif 'switch_id' in obs:
        switch_id = obs['switch_id']
        effectiveness = obs.get('effectiveness', 'normal')
        return (switch_id, effectiveness)
    else:
        return ('UNKNOWN', 'normal')


def _chem_tile_context_extractor(obs: dict) -> Hashable:
    """Extract tile configuration from ChemTile observation"""
    if 'context' in obs and isinstance(obs['context'], dict):
        return obs['context'].get('tile_type', 'default')
    elif 'tile_type' in obs:
        return obs['tile_type']
    else:
        return 'default'


# Pre-defined context specs for toy domains
HOT_POT_CONTEXT = ContextSpec(
    name="hot_pot",
    key_fn=_hot_pot_context_extractor
)

SWITCH_LIGHT_CONTEXT = ContextSpec(
    name="switch_light",
    key_fn=_switch_light_context_extractor
)

CHEM_TILE_CONTEXT = ContextSpec(
    name="chem_tile",
    key_fn=_chem_tile_context_extractor
)


# Domain lookup
DOMAIN_CONTEXT_SPECS = {
    'hot_pot': HOT_POT_CONTEXT,
    'switch_light': SWITCH_LIGHT_CONTEXT,
    'chem_tile': CHEM_TILE_CONTEXT,
}


def get_context_spec(domain: str) -> ContextSpec:
    """
    Get context spec for a domain.

    Args:
        domain: Domain name

    Returns:
        ContextSpec instance

    Raises:
        ValueError: If domain not recognized
    """
    if domain not in DOMAIN_CONTEXT_SPECS:
        raise ValueError(
            f"Unknown domain '{domain}'. "
            f"Available domains: {list(DOMAIN_CONTEXT_SPECS.keys())}"
        )
    return DOMAIN_CONTEXT_SPECS[domain]
