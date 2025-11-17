import json
import os
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
from datetime import datetime

class DomainSpecificMemory:
    """
    Manages domain-specific persistent memory for the Simple World Model.
    Each domain maintains isolated memory that persists across episodes.
    """

    def __init__(self, base_path: str = "memory/domains"):
        self.base_path = Path(base_path)
        self.ensure_structure()

    def ensure_structure(self):
        """Create directory structure for all domains"""
        domains = ['chem_tile', 'hot_pot', 'switch_light']
        for domain in domains:
            (self.base_path / domain / 'consolidated').mkdir(parents=True, exist_ok=True)
            (self.base_path / domain / 'episodes').mkdir(parents=True, exist_ok=True)

    def save_episode(self, domain: str, episode_id: str, beliefs: Dict, score: float):
        """
        Save beliefs from a completed episode.

        Args:
            domain: Environment name (chem_tile, hot_pot, switch_light)
            episode_id: Unique episode identifier
            beliefs: Final belief state from episode
            score: Episode performance score
        """
        episode_data = {
            'episode_id': episode_id,
            'timestamp': datetime.now().isoformat(),
            'score': score,
            'beliefs': beliefs
        }

        # Save individual episode
        episode_path = self.base_path / domain / 'episodes' / f'{episode_id}.json'
        with open(episode_path, 'w') as f:
            json.dump(episode_data, f, indent=2)

        # Update consolidated beliefs ONLY if episode performance was good
        # Quality threshold: 75% - prevents learning from poor episodes
        QUALITY_THRESHOLD = 75.0

        if score >= QUALITY_THRESHOLD:
            print(f"✅ Episode score {score:.1f}% >= {QUALITY_THRESHOLD}% - updating consolidated beliefs")
            self._update_consolidated_beliefs(domain, beliefs, score)
        else:
            print(f"⚠️ Episode score {score:.1f}% < {QUALITY_THRESHOLD}% - skipping consolidation to avoid reinforcing errors")

    def load_prior(self, domain: str) -> Optional[Dict]:
        """
        Load consolidated prior beliefs for a domain.
        Returns None if no prior exists (first episode).

        Args:
            domain: Environment name

        Returns:
            Dictionary of prior beliefs or None
        """
        consolidated_path = self.base_path / domain / 'consolidated' / 'beliefs.json'

        if consolidated_path.exists():
            with open(consolidated_path, 'r') as f:
                return json.load(f)

        return None

    def _update_consolidated_beliefs(self, domain: str, new_beliefs: Dict, score: float):
        """
        Update consolidated beliefs using weighted average based on performance.
        Each belief now tracks: value, confidence, and episode_count.
        """
        consolidated_path = self.base_path / domain / 'consolidated' / 'beliefs.json'

        # Load existing consolidated beliefs
        if consolidated_path.exists():
            with open(consolidated_path, 'r') as f:
                consolidated = json.load(f)
        else:
            consolidated = {}

        # Weight new beliefs by score (high score = more influence)
        weight = score / 100.0  # Normalize score to 0-1

        # Merge beliefs with new structure
        for key, value in new_beliefs.items():
            # Handle both old format (raw values) and new format (structured)
            if isinstance(value, dict) and 'value' in value:
                new_value = value['value']
                new_confidence = value.get('confidence', 0.5)
            else:
                new_value = value
                new_confidence = 0.5

            if key in consolidated:
                old = consolidated[key]

                # Handle old format in consolidated beliefs
                if isinstance(old, dict) and 'value' in old:
                    old_value = old['value']
                    old_confidence = old.get('confidence', 0.5)
                    old_count = old.get('episode_count', 1)
                else:
                    old_value = old
                    old_confidence = 0.5
                    old_count = 1

                # Weighted average for value
                if isinstance(new_value, (int, float)) and isinstance(old_value, (int, float)):
                    merged_value = (1 - weight) * old_value + weight * new_value
                elif isinstance(new_value, dict) and isinstance(old_value, dict):
                    # Recursively merge nested dicts
                    merged_value = self._merge_beliefs(old_value, new_value, weight)
                else:
                    # Replace if incompatible types
                    merged_value = new_value

                # Increase confidence with more episodes (cap at 0.95)
                merged_confidence = min(0.95, old_confidence * 0.9 + new_confidence * 0.1 + 0.05)

                consolidated[key] = {
                    'value': merged_value,
                    'confidence': merged_confidence,
                    'episode_count': old_count + 1,
                    'last_updated': datetime.now().isoformat()
                }
            else:
                # New belief
                consolidated[key] = {
                    'value': new_value,
                    'confidence': new_confidence,
                    'episode_count': 1,
                    'last_updated': datetime.now().isoformat()
                }

        # Save updated consolidated beliefs
        with open(consolidated_path, 'w') as f:
            json.dump(consolidated, f, indent=2)

    def _merge_beliefs(self, old_beliefs: Dict, new_beliefs: Dict, weight: float) -> Dict:
        """
        Merge two belief dictionaries using weighted average.
        Domain-specific logic for different belief types.
        """
        merged = old_beliefs.copy()

        for key, value in new_beliefs.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, (int, float)):
                # Numerical beliefs: weighted average
                merged[key] = (1 - weight) * merged[key] + weight * value
            elif isinstance(value, dict):
                # Nested beliefs: recursive merge
                merged[key] = self._merge_beliefs(
                    merged.get(key, {}),
                    value,
                    weight
                )
            else:
                # Other types: replace if new score is high
                if weight > 0.7:
                    merged[key] = value

        return merged

    def get_prior_strength(self, domain: str) -> float:
        """
        Get recommended prior strength based on confidence in consolidated beliefs.
        More episodes = stronger prior (but caps at 0.3 to maintain adaptability).
        """
        consolidated_path = self.base_path / domain / 'consolidated' / 'beliefs.json'

        if not consolidated_path.exists():
            return 0.1  # Default weak prior

        with open(consolidated_path, 'r') as f:
            consolidated = json.load(f)

        # New format: each belief has its own confidence
        # Calculate average confidence across all beliefs
        confidences = []
        for key, value in consolidated.items():
            if isinstance(value, dict) and 'confidence' in value:
                confidences.append(value['confidence'])

        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            # Scale prior strength: average confidence * 0.5, capped at 0.3
            return min(0.3, avg_confidence * 0.5)
        else:
            # Fallback for old format
            return min(0.3, consolidated.get('confidence', 0.1))
