"""
Enhanced Domain-Specific Memory System
Implements:
1. Quality-weighted consolidation (only episodes >= 75%)
2. Structured belief states with provenance tracking
3. Statistical outlier detection
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
import numpy as np
from datetime import datetime
from collections import defaultdict


class DomainSpecificMemory:
    """
    Manages domain-specific persistent memory with quality control.

    Key features:
    - Only consolidates beliefs from high-performing episodes
    - Detects and rejects statistical outliers
    - Tracks belief provenance (which episodes contributed)
    - Maintains excluded observations for debugging
    """

    # Configuration constants
    QUALITY_THRESHOLD = 75.0  # Minimum score to consolidate beliefs
    OUTLIER_THRESHOLD = 2.5   # Standard deviations for outlier detection
    MAX_PRIOR_STRENGTH = 0.25 # Maximum prior strength (reduced from 0.3)
    MIN_HISTORY_FOR_OUTLIER_DETECTION = 2  # Need at least 2 values to detect outliers

    def __init__(self, base_path: str = "memory/domains"):
        self.base_path = Path(base_path)
        self.ensure_structure()

    def ensure_structure(self):
        """Create directory structure for all domains"""
        domains = ['chem_tile', 'hot_pot', 'switch_light']
        for domain in domains:
            (self.base_path / domain / 'consolidated').mkdir(parents=True, exist_ok=True)
            (self.base_path / domain / 'episodes').mkdir(parents=True, exist_ok=True)
            (self.base_path / domain / 'metadata').mkdir(parents=True, exist_ok=True)

    def save_episode(self, domain: str, episode_id: str, beliefs: Dict, score: float):
        """
        Save beliefs from a completed episode with quality filtering.

        Args:
            domain: Environment name (chem_tile, hot_pot, switch_light)
            episode_id: Unique episode identifier
            beliefs: Final belief state from episode
            score: Episode performance score (0-100)
        """
        episode_data = {
            'episode_id': episode_id,
            'timestamp': datetime.now().isoformat(),
            'score': score,
            'beliefs': beliefs
        }

        # Always save individual episode (for analysis)
        episode_path = self.base_path / domain / 'episodes' / f'{episode_id}.json'
        with open(episode_path, 'w') as f:
            json.dump(episode_data, f, indent=2)

        # FIX #1: QUALITY-WEIGHTED CONSOLIDATION
        # Only update consolidated beliefs if episode performance was good
        if score >= self.QUALITY_THRESHOLD:
            print(f"✅ Episode score {score:.1f}% >= {self.QUALITY_THRESHOLD}% - updating consolidated beliefs")
            self._update_consolidated_beliefs(domain, beliefs, score, episode_id)
        else:
            print(f"⚠️ Episode score {score:.1f}% < {self.QUALITY_THRESHOLD}% - skipping consolidation")
            self._log_excluded_episode(domain, episode_id, score, reason="low_score")

    def load_prior(self, domain: str) -> Optional[Dict]:
        """
        Load consolidated prior beliefs for a domain.

        Returns:
            Dictionary of prior beliefs with full metadata, or None if no prior exists
        """
        consolidated_path = self.base_path / domain / 'consolidated' / 'beliefs.json'

        if consolidated_path.exists():
            with open(consolidated_path, 'r') as f:
                return json.load(f)

        return None

    def _update_consolidated_beliefs(self, domain: str, new_beliefs: Dict, score: float, episode_id: str):
        """
        Update consolidated beliefs with quality control and outlier detection.

        FIX #2: STRUCTURED BELIEF STATES
        Each belief now contains:
        - value: The actual belief value
        - confidence: Confidence score (0-1)
        - source_episodes: List of episode IDs that contributed
        - observation_count: Total observations
        - excluded_observations: List of (value, reason) tuples for rejected data
        - last_updated: ISO timestamp

        FIX #3: OUTLIER DETECTION
        Rejects observations that are statistical outliers (>2.5 std devs from mean)
        """
        consolidated_path = self.base_path / domain / 'consolidated' / 'beliefs.json'

        # Load existing consolidated beliefs
        if consolidated_path.exists():
            with open(consolidated_path, 'r') as f:
                consolidated = json.load(f)
        else:
            consolidated = {}

        # Weight new beliefs by score (high score = more influence)
        weight = score / 100.0

        # Process each belief with outlier detection
        for key, value in new_beliefs.items():
            # Extract value and metadata from new belief
            if isinstance(value, dict) and 'value' in value:
                new_value = value['value']
                new_confidence = value.get('confidence', 0.5)
                new_obs_count = value.get('observation_count', 1)
            else:
                new_value = value
                new_confidence = 0.5
                new_obs_count = 1

            # Check for outlier (only for numeric values with history)
            is_outlier = False
            outlier_reason = None

            if key in consolidated and isinstance(new_value, (int, float)):
                is_outlier, outlier_reason = self._is_outlier(
                    key, new_value, consolidated[key], domain
                )

            if is_outlier:
                # Reject outlier - don't update consolidated belief
                print(f"  🚫 Rejecting outlier for '{key}': {new_value:.3f} ({outlier_reason})")

                # Track rejected observation
                if key in consolidated and isinstance(consolidated[key], dict):
                    if 'excluded_observations' not in consolidated[key]:
                        consolidated[key]['excluded_observations'] = []

                    consolidated[key]['excluded_observations'].append({
                        'value': float(new_value),
                        'reason': outlier_reason,
                        'episode_id': episode_id,
                        'timestamp': datetime.now().isoformat()
                    })

                continue  # Skip this belief, don't consolidate

            # Not an outlier - proceed with consolidation
            if key in consolidated:
                old = consolidated[key]

                # Handle structured format
                if isinstance(old, dict) and 'value' in old:
                    old_value = old['value']
                    old_confidence = old.get('confidence', 0.5)
                    old_count = old.get('episode_count', 1)
                    old_obs_count = old.get('observation_count', 1)
                    source_episodes = old.get('source_episodes', [])
                    excluded_obs = old.get('excluded_observations', [])
                else:
                    # Migrate old format
                    old_value = old
                    old_confidence = 0.5
                    old_count = 1
                    old_obs_count = 1
                    source_episodes = []
                    excluded_obs = []

                # Weighted average for value (score-weighted)
                if isinstance(new_value, (int, float)) and isinstance(old_value, (int, float)):
                    merged_value = (1 - weight) * old_value + weight * new_value
                elif isinstance(new_value, dict) and isinstance(old_value, dict):
                    # Recursively merge nested dicts
                    merged_value = self._merge_beliefs(old_value, new_value, weight)
                else:
                    # Replace if incompatible types
                    merged_value = new_value

                # Confidence increases with more episodes but caps lower (0.85 instead of 0.95)
                # This prevents over-confidence that blocks adaptation
                merged_confidence = min(0.85, old_confidence * 0.9 + new_confidence * 0.1 + 0.05)

                # Build updated belief with full metadata
                consolidated[key] = {
                    'value': merged_value,
                    'confidence': merged_confidence,
                    'episode_count': old_count + 1,
                    'observation_count': old_obs_count + new_obs_count,
                    'source_episodes': source_episodes + [episode_id],
                    'excluded_observations': excluded_obs,
                    'last_updated': datetime.now().isoformat()
                }

                # Print update message (handle both numeric and non-numeric values)
                if isinstance(merged_value, (int, float)) and isinstance(old_value, (int, float)):
                    print(f"  ✓ Updated '{key}': {old_value:.3f} → {merged_value:.3f} (confidence: {merged_confidence:.3f})")
                else:
                    print(f"  ✓ Updated '{key}' (confidence: {merged_confidence:.3f})")

            else:
                # New belief - initialize with metadata
                consolidated[key] = {
                    'value': new_value,
                    'confidence': new_confidence,
                    'episode_count': 1,
                    'observation_count': new_obs_count,
                    'source_episodes': [episode_id],
                    'excluded_observations': [],
                    'last_updated': datetime.now().isoformat()
                }

                print(f"  ✓ New belief '{key}': {new_value}")

        # Save updated consolidated beliefs
        with open(consolidated_path, 'w') as f:
            json.dump(consolidated, f, indent=2)

    def _is_outlier(self, belief_key: str, new_value: float, historical_belief: Dict, domain: str) -> Tuple[bool, Optional[str]]:
        """
        FIX #3: OUTLIER DETECTION

        Detect if a new observation is a statistical outlier.

        Args:
            belief_key: Name of the belief being checked
            new_value: New observed value
            historical_belief: Historical belief data with metadata
            domain: Domain name (for loading episode history)

        Returns:
            (is_outlier, reason): Tuple of boolean and explanation string
        """
        # Need structured format with history
        if not isinstance(historical_belief, dict) or 'value' not in historical_belief:
            return False, None

        # Need multiple data points for statistical detection
        episode_count = historical_belief.get('episode_count', 1)
        if episode_count < self.MIN_HISTORY_FOR_OUTLIER_DETECTION:
            return False, None  # Not enough history

        # Collect historical values from source episodes
        source_episodes = historical_belief.get('source_episodes', [])
        if not source_episodes:
            return False, None

        historical_values = []

        # Load values from source episodes
        for ep_id in source_episodes:
            ep_file = self.base_path / domain / 'episodes' / f'{ep_id}.json'
            if ep_file.exists():
                try:
                    with open(ep_file, 'r') as f:
                        ep_data = json.load(f)

                    # Extract this belief's value from the episode
                    ep_beliefs = ep_data.get('beliefs', {})
                    if belief_key in ep_beliefs:
                        ep_value = ep_beliefs[belief_key]

                        # Extract numeric value
                        if isinstance(ep_value, dict) and 'value' in ep_value:
                            ep_value = ep_value['value']

                        if isinstance(ep_value, (int, float)):
                            historical_values.append(float(ep_value))

                except Exception as e:
                    print(f"  ⚠️ Error loading episode {ep_id}: {e}")
                    continue

        # Need at least 2 values for outlier detection
        if len(historical_values) < self.MIN_HISTORY_FOR_OUTLIER_DETECTION:
            return False, None

        # Calculate statistics
        mean = np.mean(historical_values)
        std = np.std(historical_values)

        # Avoid division by zero
        if std < 1e-6:
            # Very low variance - any significantly different value is an outlier
            if abs(new_value - mean) > 0.1 * abs(mean):
                return True, f"differs from consistent history (mean={mean:.3f}, std≈0)"
            return False, None

        # Calculate z-score
        z_score = abs(new_value - mean) / std

        # Check if outlier
        if z_score > self.OUTLIER_THRESHOLD:
            return True, f"z-score={z_score:.2f} (mean={mean:.3f}, std={std:.3f})"

        return False, None

    def _merge_beliefs(self, old_beliefs: Dict, new_beliefs: Dict, weight: float) -> Dict:
        """
        Recursively merge nested belief dictionaries.
        Used for complex beliefs like reaction probabilities.
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

    def _log_excluded_episode(self, domain: str, episode_id: str, score: float, reason: str):
        """
        Log episodes that were excluded from consolidation.
        Useful for debugging why learning isn't improving.
        """
        exclusion_log_path = self.base_path / domain / 'metadata' / 'excluded_episodes.json'

        # Load existing log
        if exclusion_log_path.exists():
            with open(exclusion_log_path, 'r') as f:
                exclusions = json.load(f)
        else:
            exclusions = []

        # Add new exclusion
        exclusions.append({
            'episode_id': episode_id,
            'score': score,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })

        # Save updated log
        with open(exclusion_log_path, 'w') as f:
            json.dump(exclusions, f, indent=2)

    def get_prior_strength(self, domain: str) -> float:
        """
        Calculate prior strength based on consolidated belief confidence.

        FIX: Reduced MAX_PRIOR_STRENGTH from 0.3 to 0.25 to allow more adaptation.

        Returns:
            Prior strength value between 0.1 and 0.25
        """
        consolidated_path = self.base_path / domain / 'consolidated' / 'beliefs.json'

        if not consolidated_path.exists():
            return 0.1  # Default weak prior

        with open(consolidated_path, 'r') as f:
            consolidated = json.load(f)

        # Calculate average confidence across all beliefs
        confidences = []
        episode_counts = []

        for key, value in consolidated.items():
            if isinstance(value, dict) and 'confidence' in value:
                confidences.append(value['confidence'])
                episode_counts.append(value.get('episode_count', 1))

        if not confidences:
            return 0.1  # Fallback

        avg_confidence = sum(confidences) / len(confidences)
        max_episodes = max(episode_counts) if episode_counts else 1

        # Scale prior strength based on confidence AND episode count
        # More episodes + higher confidence = stronger prior (but capped)
        base_strength = avg_confidence * 0.4  # Reduced from 0.5
        episode_bonus = min(0.1, max_episodes * 0.02)  # Up to +0.1 for many episodes

        total_strength = min(self.MAX_PRIOR_STRENGTH, base_strength + episode_bonus)

        print(f"  Prior strength: {total_strength:.3f} (confidence={avg_confidence:.3f}, episodes={max_episodes})")

        return total_strength

    def get_belief_summary(self, domain: str) -> Dict[str, Any]:
        """
        Get a summary of consolidated beliefs for debugging.

        Returns:
            Dictionary with belief statistics and quality metrics
        """
        consolidated_path = self.base_path / domain / 'consolidated' / 'beliefs.json'

        if not consolidated_path.exists():
            return {'status': 'no_beliefs'}

        with open(consolidated_path, 'r') as f:
            consolidated = json.load(f)

        summary = {
            'total_beliefs': len(consolidated),
            'beliefs': {}
        }

        for key, value in consolidated.items():
            if isinstance(value, dict) and 'value' in value:
                summary['beliefs'][key] = {
                    'value': value['value'],
                    'confidence': value.get('confidence', 0),
                    'episode_count': value.get('episode_count', 0),
                    'observation_count': value.get('observation_count', 0),
                    'num_excluded': len(value.get('excluded_observations', []))
                }

        return summary
