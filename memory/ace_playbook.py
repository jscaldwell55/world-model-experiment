"""
ACE (Agentic Context Engineering) Playbook System

Replaces consolidation-based memory with context-aware learning.
Stores observations with methodology tracking to prevent belief traps.

Key Innovation:
- Separates episode score (answer quality) from reliability (methodology quality)
- Stores full context (actions, settings) instead of consolidated values
- Generates warnings instead of rejecting outliers
"""

import json
import os
import re
import fcntl
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict
import numpy as np


class ACEPlaybook:
    """
    ACE-based memory system that stores context-aware observations
    instead of consolidating beliefs.
    """

    def __init__(self, domain_name: str, base_path: str = "memory/domains"):
        """
        Initialize playbook for a domain (hot_pot, chem_tile, switch_light).

        Storage structure:
        memory/domains/{domain_name}/
        ├── playbook.json              # Structured playbook with context-aware bullets
        ├── episodes/*.json            # Raw episode data (kept for reflection)
        └── metadata/stats.json        # Playbook statistics

        Args:
            domain_name: Domain identifier (hot_pot, chem_tile, switch_light)
            base_path: Base directory for memory storage
        """
        self.domain_name = domain_name
        self.base_path = Path(base_path)

        # Create directory structure
        self.domain_path = self.base_path / domain_name
        (self.domain_path / 'episodes').mkdir(parents=True, exist_ok=True)
        (self.domain_path / 'metadata').mkdir(parents=True, exist_ok=True)

        # Playbook structure
        self.playbook = {
            'observations': [],           # Context-aware observations
            'strategies_and_rules': [],   # Validated patterns
            'troubleshooting': [],        # Known failure modes
            'context_patterns': [],       # Context-dependent observations
        }

        # Track current context for this episode
        self.current_context = ""

        # Load existing playbook
        self.load_playbook()

    def _acquire_playbook_lock(self, lock_file_path: Path, exclusive: bool = True, timeout: float = 30.0):
        """
        Acquire a file lock with timeout to prevent race conditions in parallel execution.

        Args:
            lock_file_path: Path to the lock file
            exclusive: If True, acquire exclusive lock (for writing). If False, shared lock (for reading)
            timeout: Maximum time to wait for lock in seconds

        Returns:
            File handle for the lock file (must be kept open to maintain lock)

        Raises:
            TimeoutError: If lock cannot be acquired within timeout
        """
        lock_mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        lock_file = open(lock_file_path, 'a')

        start_time = time.time()
        while True:
            try:
                # Try to acquire lock with non-blocking mode
                fcntl.flock(lock_file.fileno(), lock_mode | fcntl.LOCK_NB)
                return lock_file
            except BlockingIOError:
                # Lock is held by another process
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    lock_file.close()
                    raise TimeoutError(
                        f"Could not acquire {'exclusive' if exclusive else 'shared'} lock "
                        f"on {lock_file_path} after {timeout} seconds"
                    )
                # Wait a bit before retrying
                time.sleep(0.1)

    def _release_playbook_lock(self, lock_file):
        """
        Release a file lock.

        Args:
            lock_file: File handle returned by _acquire_playbook_lock
        """
        if lock_file and not lock_file.closed:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass  # Ignore errors during unlock
            finally:
                lock_file.close()

    def get_context(self, task_info: dict) -> str:
        """
        Build natural language context for SimpleWorldModel initialization.

        This is the core of ACE - provide rich context instead of consolidated values.

        Args:
            task_info: Dict with context like {'environment': 'HotPotLab'}

        Returns:
            Natural language context string with warnings about methodology issues
        """
        if not self.playbook['observations']:
            return self._generate_default_context(task_info)

        # Build context from stored observations
        context_parts = []

        # Header
        env_name = task_info.get('environment', self.domain_name.upper())
        context_parts.append(f"=== {env_name} KNOWLEDGE BASE ===\n")

        # Group observations by reliability and context
        high_reliability = []
        medium_reliability = []
        low_reliability = []

        for obs in self.playbook['observations']:
            reliability = obs.get('reliability', 'MEDIUM')
            if reliability == 'HIGH':
                high_reliability.append(obs)
            elif reliability == 'MEDIUM':
                medium_reliability.append(obs)
            else:
                low_reliability.append(obs)

        # Present high-reliability observations first
        if high_reliability:
            context_parts.append("\n✓ HIGH-RELIABILITY OBSERVATIONS:")
            context_parts.append(self._format_observations(high_reliability))

        # Medium-reliability observations
        if medium_reliability:
            context_parts.append("\n○ MEDIUM-RELIABILITY OBSERVATIONS:")
            context_parts.append(self._format_observations(medium_reliability))

        # Low-reliability observations with warnings
        if low_reliability:
            context_parts.append("\n⚠️  LOW-RELIABILITY OBSERVATIONS (USE WITH CAUTION):")
            context_parts.append(self._format_observations(low_reliability))

        # Add strategies if available
        if self.playbook['strategies_and_rules']:
            context_parts.append("\n📋 LEARNED STRATEGIES:")
            for strategy in self.playbook['strategies_and_rules'][-3:]:  # Last 3
                context_parts.append(f"  • {strategy['content']}")

        # Add troubleshooting tips
        if self.playbook['troubleshooting']:
            context_parts.append("\n⚠️  KNOWN ISSUES:")
            for issue in self.playbook['troubleshooting'][-3:]:  # Last 3
                context_parts.append(f"  • {issue['content']}")

        # Add recommendation
        context_parts.append("\n💡 RECOMMENDATION:")
        context_parts.append("  Initialize with WEAK priors (prior_strength=0.1)")
        context_parts.append("  Trust current observations over past averages")
        context_parts.append("  Pay attention to context (settings, actions taken)")

        context = "\n".join(context_parts)
        self.current_context = context  # Store for later reference
        return context

    def _format_observations(self, observations: List[Dict]) -> str:
        """
        Format a list of observations into readable text.

        Args:
            observations: List of observation dictionaries

        Returns:
            Formatted string
        """
        if not observations:
            return "  (No observations)"

        lines = []
        for obs in observations:
            # Extract key info
            episode_id = obs.get('episode_id', 'unknown')
            score = obs.get('score', 0) * 100
            beliefs = obs.get('beliefs', {})
            obs_context = obs.get('context', {})
            reason = obs.get('reason', '')

            # Format based on domain (hot_pot_test → hot_pot, hot_pot_controlled → hot_pot)
            domain = self.domain_name.replace('_test', '').replace('_controlled', '')

            if domain == 'hot_pot':
                heating_rate = self._extract_value(beliefs.get('heating_rate_mean', 0))
                if heating_rate == 0:
                    heating_rate = self._extract_value(beliefs.get('heating_rate_mean'))

                power = obs_context.get('power_setting', 'UNKNOWN')

                if isinstance(heating_rate, (int, float)):
                    lines.append(f"  • Episode {episode_id} (score: {score:.0f}%): heating_rate ~{heating_rate:.2f}°C/s [power: {power}]")
                else:
                    lines.append(f"  • Episode {episode_id} (score: {score:.0f}%): [power: {power}]")

                if reason:
                    lines.append(f"    → {reason}")

            elif domain == 'switch_light':
                wiring_probs = self._extract_value(beliefs.get('wiring_probs', {}))
                lines.append(f"  • Episode {episode_id} (score: {score:.0f}%): {wiring_probs}")
                if reason:
                    lines.append(f"    → {reason}")

            elif domain == 'chem_tile':
                reaction_probs = self._extract_value(beliefs.get('reaction_probs', {}))
                lines.append(f"  • Episode {episode_id} (score: {score:.0f}%)")
                if reason:
                    lines.append(f"    → {reason}")
            else:
                # Generic format
                lines.append(f"  • Episode {episode_id} (score: {score:.0f}%)")
                if reason:
                    lines.append(f"    → {reason}")

        return "\n".join(lines)

    def _generate_default_context(self, task_info: dict) -> str:
        """
        Generate default context when no prior observations exist.

        Args:
            task_info: Task information

        Returns:
            Default context string
        """
        env_name = task_info.get('environment', self.domain_name.upper())
        return f"""=== {env_name} KNOWLEDGE BASE ===

No prior observations available.

💡 RECOMMENDATION:
  Use broad, uninformative priors
  Explore systematically to gather reliable data
  Keep prior_strength=0.1 for maximum adaptability
"""

    def reflect(self, trajectory: dict, outcome: dict) -> dict:
        """
        REFLECTOR ROLE: Analyze trajectory and extract insights.

        Key innovation: Detects methodology issues by analyzing action sequences.

        Args:
            trajectory: {
                'observations': [...],
                'actions': [...],
                'final_beliefs': {...},
                'context_used': "..."
            }
            outcome: {
                'score': 0.89,
                'test_results': [...]
            }

        Returns:
            insights: {
                'patterns_found': [...],
                'failures_identified': [...],
                'context_dependencies': [...],
                'methodology_quality': 'HIGH'|'MEDIUM'|'LOW',
                'reliability_reason': "..."
            }
        """
        insights = {
            'patterns_found': [],
            'failures_identified': [],
            'context_dependencies': [],
            'methodology_quality': 'MEDIUM',
            'reliability_reason': ''
        }

        # Analyze action sequence for methodology issues
        actions = trajectory.get('actions', [])
        observations = trajectory.get('observations', [])

        # Domain-specific methodology analysis
        # Handle various domain name variations (hot_pot, hot_pot_test, hot_pot_controlled, etc.)
        domain_base = self.domain_name.replace('_test', '').replace('_controlled', '')

        if domain_base == 'hot_pot':
            methodology_analysis = self._analyze_hotpot_methodology(actions, observations)
            insights['methodology_quality'] = methodology_analysis['quality']
            insights['reliability_reason'] = methodology_analysis['reason']
            insights['context_dependencies'] = methodology_analysis['context']

        elif domain_base == 'switch_light':
            methodology_analysis = self._analyze_switchlight_methodology(actions, observations)
            insights['methodology_quality'] = methodology_analysis['quality']
            insights['reliability_reason'] = methodology_analysis['reason']

        elif domain_base == 'chem_tile':
            methodology_analysis = self._analyze_chemtile_methodology(actions, observations)
            insights['methodology_quality'] = methodology_analysis['quality']
            insights['reliability_reason'] = methodology_analysis['reason']

        # Extract patterns from successful outcomes
        if outcome['score'] > 0.7:
            insights['patterns_found'].append(f"Achieved {outcome['score']*100:.0f}% accuracy")

        # Identify failures
        if outcome['score'] < 0.5:
            insights['failures_identified'].append(f"Low score: {outcome['score']*100:.0f}%")

        return insights

    def _analyze_hotpot_methodology(self, actions: List, observations: List) -> Dict:
        """
        Analyze HotPot methodology for reliability.

        Key issue: Power setting toggles create mixed-context data

        Args:
            actions: List of action strings
            observations: List of observation dicts

        Returns:
            {
                'quality': 'HIGH'|'MEDIUM'|'LOW',
                'reason': explanation string,
                'context': {'power_setting': 'HIGH'|'LOW'|'MIXED'}
            }
        """
        # Count power toggles - check both string actions and None values
        toggle_count = 0
        for a in actions:
            if a is None:
                continue
            action_str = str(a).lower()
            if 'toggle' in action_str or 'set_power' in action_str or 'power' in action_str:
                toggle_count += 1

        # Detect power setting changes from observations
        power_settings = []
        for obs in observations:
            if isinstance(obs, dict) and 'power_setting' in obs:
                power_settings.append(obs['power_setting'])

        # Check if power settings are consistent
        unique_power_settings = set(power_settings) if power_settings else set()

        # Determine methodology quality
        if toggle_count == 0 and len(unique_power_settings) <= 1:
            quality = 'HIGH'
            reason = 'Consistent power setting - reliable measurement'
            power_context = power_settings[0] if power_settings else 'UNKNOWN'
        elif toggle_count == 1 or len(unique_power_settings) == 2:
            quality = 'LOW'
            reason = f'Power toggle detected - mixed contexts (averaged data)'
            power_context = 'MIXED'
        elif toggle_count >= 2:
            quality = 'LOW'
            reason = f'Multiple power toggles ({toggle_count}) - averaged across contexts'
            power_context = 'MIXED'
        else:
            # Default case
            quality = 'MEDIUM'
            reason = 'Unable to determine power consistency'
            power_context = 'UNKNOWN'

        return {
            'quality': quality,
            'reason': reason,
            'context': {'power_setting': power_context}
        }

    def _analyze_switchlight_methodology(self, actions: List, observations: List) -> Dict:
        """
        Analyze SwitchLight methodology for reliability.

        Args:
            actions: List of action strings
            observations: List of observation dicts

        Returns:
            Methodology analysis
        """
        # Count switch flips
        flip_count = sum(1 for a in actions if 'flip' in str(a).lower())

        if flip_count >= 3:
            quality = 'HIGH'
            reason = f'Systematic exploration ({flip_count} flips) - good coverage'
        elif flip_count >= 1:
            quality = 'MEDIUM'
            reason = f'Some exploration ({flip_count} flips)'
        else:
            quality = 'LOW'
            reason = 'Minimal exploration - limited data'

        return {'quality': quality, 'reason': reason}

    def _analyze_chemtile_methodology(self, actions: List, observations: List) -> Dict:
        """
        Analyze ChemTile methodology for reliability.

        Args:
            actions: List of action strings
            observations: List of observation dicts

        Returns:
            Methodology analysis
        """
        # Count reactions performed
        reaction_count = sum(1 for a in actions if 'react' in str(a).lower())

        if reaction_count >= 3:
            quality = 'HIGH'
            reason = f'Multiple reactions ({reaction_count}) - good data'
        elif reaction_count >= 1:
            quality = 'MEDIUM'
            reason = f'Some reactions ({reaction_count})'
        else:
            quality = 'LOW'
            reason = 'Few reactions - limited exploration'

        return {'quality': quality, 'reason': reason}

    def curate(self, insights: dict, trajectory: dict, outcome: dict) -> list:
        """
        CURATOR ROLE: Generate delta updates from insights.

        Creates new observation entries tagged with methodology quality.

        Args:
            insights: Output from reflect()
            trajectory: Original trajectory
            outcome: Episode outcome with score

        Returns:
            deltas: [
                {
                    'section': 'observations',
                    'operation': 'ADD',
                    'content': {...observation data...},
                    'metadata': {...}
                }
            ]
        """
        deltas = []

        # Add observation entry
        observation_entry = {
            'episode_id': trajectory.get('episode_id', f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            'timestamp': datetime.now().isoformat(),
            'score': outcome['score'],
            'beliefs': trajectory.get('final_beliefs', {}),
            'context': insights.get('context_dependencies', {}),
            'reliability': insights['methodology_quality'],
            'reason': insights['reliability_reason'],
            'metadata': {
                'patterns': insights.get('patterns_found', []),
                'failures': insights.get('failures_identified', [])
            }
        }

        deltas.append({
            'section': 'observations',
            'operation': 'ADD',
            'content': observation_entry
        })

        # Add strategies for high-scoring, high-reliability episodes
        if outcome['score'] > 0.8 and insights['methodology_quality'] == 'HIGH':
            for pattern in insights.get('patterns_found', []):
                deltas.append({
                    'section': 'strategies_and_rules',
                    'operation': 'ADD',
                    'content': {
                        'content': pattern,
                        'confidence': outcome['score'],
                        'source_episode': observation_entry['episode_id']
                    }
                })

        # Add troubleshooting for failures
        if insights.get('failures_identified'):
            for failure in insights['failures_identified']:
                deltas.append({
                    'section': 'troubleshooting',
                    'operation': 'ADD',
                    'content': {
                        'content': failure,
                        'source_episode': observation_entry['episode_id']
                    }
                })

        return deltas

    def merge_deltas(self, deltas: list):
        """
        Deterministic (non-LLM) merge of delta updates into playbook.

        Args:
            deltas: List of delta operations from curate()
        """
        for delta in deltas:
            section = delta['section']
            operation = delta['operation']
            content = delta['content']

            if operation == 'ADD':
                if section not in self.playbook:
                    self.playbook[section] = []

                self.playbook[section].append(content)

            # Keep only recent entries (last 10 per section)
            if section in self.playbook and len(self.playbook[section]) > 10:
                # Keep high-reliability observations preferentially
                if section == 'observations':
                    sorted_obs = sorted(
                        self.playbook[section],
                        key=lambda x: (
                            1 if x.get('reliability') == 'HIGH' else (0 if x.get('reliability') == 'MEDIUM' else -1),
                            x.get('score', 0)
                        ),
                        reverse=True
                    )
                    self.playbook[section] = sorted_obs[:10]
                else:
                    self.playbook[section] = self.playbook[section][-10:]

    def save_playbook(self):
        """
        Save playbook to disk with file locking to prevent race conditions.

        Uses exclusive locking and read-modify-write pattern:
        1. Acquire exclusive lock
        2. Read current playbook from disk
        3. Merge our changes with current state
        4. Write atomically
        5. Release lock

        This prevents the race condition where:
        - Process A loads playbook (5 observations)
        - Process B loads playbook (5 observations)
        - Process A adds 1, saves (6 observations)
        - Process B adds 1, saves (6 observations) ← loses A's observation
        """
        playbook_path = self.domain_path / 'playbook.json'
        lock_path = self.domain_path / 'playbook.lock'

        # Acquire exclusive lock
        lock_file = None
        try:
            lock_file = self._acquire_playbook_lock(lock_path, exclusive=True, timeout=30.0)

            # Read current state from disk (may have been updated by another process)
            current_playbook = {
                'observations': [],
                'strategies_and_rules': [],
                'troubleshooting': [],
                'context_patterns': [],
            }

            if playbook_path.exists():
                with open(playbook_path, 'r') as f:
                    current_playbook = json.load(f)

            # Merge our playbook with current state
            # For each section, combine and deduplicate by episode_id or content
            merged_playbook = {}

            for section in ['observations', 'strategies_and_rules', 'troubleshooting', 'context_patterns']:
                current_items = current_playbook.get(section, [])
                our_items = self.playbook.get(section, [])

                # Combine and deduplicate
                if section == 'observations':
                    # Deduplicate by episode_id
                    seen_ids = set()
                    merged = []

                    for item in current_items + our_items:
                        episode_id = item.get('episode_id')
                        if episode_id and episode_id not in seen_ids:
                            seen_ids.add(episode_id)
                            merged.append(item)

                    # Apply the limit of 10 observations (keep high-reliability preferentially)
                    if len(merged) > 10:
                        sorted_obs = sorted(
                            merged,
                            key=lambda x: (
                                1 if x.get('reliability') == 'HIGH' else (0 if x.get('reliability') == 'MEDIUM' else -1),
                                x.get('score', 0)
                            ),
                            reverse=True
                        )
                        merged = sorted_obs[:10]

                    merged_playbook[section] = merged

                else:
                    # For other sections, keep last 10 items
                    merged = current_items + our_items
                    if len(merged) > 10:
                        merged = merged[-10:]
                    merged_playbook[section] = merged

            # Write merged playbook atomically: write to temp file, then rename
            temp_path = playbook_path.with_suffix('.json.tmp')
            with open(temp_path, 'w') as f:
                json.dump(merged_playbook, f, indent=2)

            # Atomic rename (overwrites existing file)
            temp_path.rename(playbook_path)

            # Update our in-memory copy to reflect what was saved
            self.playbook = merged_playbook

        finally:
            # Always release lock
            self._release_playbook_lock(lock_file)

    def load_playbook(self):
        """
        Load playbook from disk with file locking to prevent race conditions.

        Uses shared locking to allow concurrent reads but prevent reading during writes.
        """
        playbook_path = self.domain_path / 'playbook.json'

        if not playbook_path.exists():
            # No playbook exists yet - nothing to load
            return

        lock_path = self.domain_path / 'playbook.lock'
        lock_file = None
        try:
            # Acquire shared lock (allows multiple readers)
            lock_file = self._acquire_playbook_lock(lock_path, exclusive=False, timeout=30.0)

            # Read playbook
            with open(playbook_path, 'r') as f:
                self.playbook = json.load(f)

        finally:
            # Always release lock
            self._release_playbook_lock(lock_file)

    def save_episode(self, episode_id: str, trajectory: dict, outcome: dict):
        """
        Save raw episode data for future reflection.

        Args:
            episode_id: Episode identifier
            trajectory: Full trajectory data
            outcome: Episode outcome
        """
        episode_data = {
            'episode_id': episode_id,
            'timestamp': datetime.now().isoformat(),
            'trajectory': trajectory,
            'outcome': outcome
        }

        episode_path = self.domain_path / 'episodes' / f'{episode_id}.json'
        with open(episode_path, 'w') as f:
            json.dump(episode_data, f, indent=2)

    def _extract_value(self, belief_data):
        """
        Extract value from structured belief format or return raw value.

        Args:
            belief_data: Belief value (may be wrapped in {'value': ...})

        Returns:
            Extracted value
        """
        if isinstance(belief_data, dict) and 'value' in belief_data:
            return belief_data['value']
        return belief_data
