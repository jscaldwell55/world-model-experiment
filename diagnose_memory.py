#!/usr/bin/env python3
"""
Memory Persistence Diagnostic & Fix Script
For debugging the flat learning curves in the 30-episode experiment
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

class MemoryDiagnostics:
    """Diagnostic tools for memory persistence debugging"""

    def __init__(self, memory_base_path: str = "memory/domains"):
        self.memory_base = Path(memory_base_path)
        self.domains = ["hot_pot", "switch_light", "chem_tile"]
        self.diagnostic_results = {}

    def run_full_diagnostic(self) -> Dict[str, bool]:
        """Run all diagnostic tests"""
        print("=" * 60)
        print("MEMORY PERSISTENCE DIAGNOSTIC SUITE")
        print("=" * 60)

        tests = {
            "Memory directories exist": self.check_directory_structure(),
            "Episode files exist": self.check_episode_files(),
            "Consolidated files exist": self.check_consolidated_files(),
            "Files are growing": self.check_file_growth(),
            "Beliefs accumulating": self.check_belief_accumulation(),
            "Domain isolation maintained": self.check_domain_isolation(),
            "Prior loading works": self.test_prior_loading()
        }

        print("\n" + "=" * 60)
        print("DIAGNOSTIC RESULTS")
        print("=" * 60)

        all_passed = True
        for test_name, result in tests.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {test_name}")
            if not result:
                all_passed = False

        return all_passed

    def check_directory_structure(self) -> bool:
        """Check if memory directory structure exists"""
        print("\nChecking directory structure...")

        expected_dirs = []
        for domain in self.domains:
            expected_dirs.append(self.memory_base / domain)
            expected_dirs.append(self.memory_base / domain / "episodes")
            expected_dirs.append(self.memory_base / domain / "consolidated")

        missing = []
        for dir_path in expected_dirs:
            if not dir_path.exists():
                missing.append(str(dir_path))
                print(f"  ❌ Missing: {dir_path}")

        if missing:
            print(f"  Found {len(expected_dirs) - len(missing)}/{len(expected_dirs)} directories")
            return False
        else:
            print(f"  ✅ All {len(expected_dirs)} directories exist")
            return True

    def check_episode_files(self) -> bool:
        """Check if episode files are being created"""
        print("\nChecking episode files...")

        total_episodes = 0
        for domain in self.domains:
            episode_dir = self.memory_base / domain / "episodes"
            if episode_dir.exists():
                episodes = list(episode_dir.glob("*.json"))
                total_episodes += len(episodes)
                print(f"  {domain}: {len(episodes)} episode files")

        if total_episodes == 0:
            print("  ❌ No episode files found!")
            return False
        elif total_episodes < 30:
            print(f"  ⚠️ Only {total_episodes} episode files (expected 30)")
            return False
        else:
            print(f"  ✅ Found {total_episodes} episode files")
            return True

    def check_consolidated_files(self) -> bool:
        """Check if consolidated belief files exist"""
        print("\nChecking consolidated files...")

        found = 0
        for domain in self.domains:
            consolidated_file = self.memory_base / domain / "consolidated" / "beliefs.json"
            if consolidated_file.exists():
                found += 1
                size = consolidated_file.stat().st_size
                print(f"  ✅ {domain}: beliefs.json ({size} bytes)")
            else:
                print(f"  ❌ {domain}: beliefs.json missing")

        return found == len(self.domains)

    def check_file_growth(self) -> bool:
        """Check if consolidated files are growing over time"""
        print("\nChecking file growth over episodes...")

        growth_detected = False
        for domain in self.domains:
            consolidated_file = self.memory_base / domain / "consolidated" / "beliefs.json"
            if consolidated_file.exists():
                try:
                    with open(consolidated_file, 'r') as f:
                        beliefs = json.load(f)

                    # Check for episode counters or accumulated data
                    has_counters = any(
                        isinstance(v, dict) and 'episode_count' in v
                        for v in beliefs.values()
                    )

                    if has_counters:
                        max_count = max(
                            v.get('episode_count', 1)
                            for v in beliefs.values()
                            if isinstance(v, dict)
                        )
                        print(f"  {domain}: Max episode count = {max_count}")
                        if max_count > 1:
                            growth_detected = True
                    else:
                        print(f"  ⚠️ {domain}: No episode counters found")
                except Exception as e:
                    print(f"  ❌ {domain}: Error reading file: {e}")

        return growth_detected

    def check_belief_accumulation(self) -> bool:
        """Check if beliefs are accumulating confidence over episodes"""
        print("\nChecking belief accumulation...")

        accumulation_found = False
        for domain in self.domains:
            consolidated_file = self.memory_base / domain / "consolidated" / "beliefs.json"
            if consolidated_file.exists():
                try:
                    with open(consolidated_file, 'r') as f:
                        beliefs = json.load(f)

                    confidences = []
                    for key, value in beliefs.items():
                        if isinstance(value, dict) and 'confidence' in value:
                            confidences.append(value['confidence'])

                    if confidences:
                        avg_conf = np.mean(confidences)
                        max_conf = max(confidences)
                        print(f"  {domain}: Avg confidence = {avg_conf:.3f}, Max = {max_conf:.3f}")

                        # High confidence suggests accumulation
                        if max_conf > 0.7:
                            accumulation_found = True
                    else:
                        print(f"  ⚠️ {domain}: No confidence scores found")

                except Exception as e:
                    print(f"  ❌ {domain}: Error: {e}")

        return accumulation_found

    def check_domain_isolation(self) -> bool:
        """Verify domains are isolated (no cross-contamination)"""
        print("\nChecking domain isolation...")

        domain_specific_keys = {
            'hot_pot': ['heating_rate', 'temperature', 'burn_threshold'],
            'switch_light': ['wiring', 'circuit', 'connection'],
            'chem_tile': ['reaction', 'chemical', 'mix']
        }

        isolation_maintained = True
        for domain in self.domains:
            consolidated_file = self.memory_base / domain / "consolidated" / "beliefs.json"
            if consolidated_file.exists():
                try:
                    with open(consolidated_file, 'r') as f:
                        beliefs = json.load(f)

                    # Check for cross-contamination
                    other_domains = [d for d in self.domains if d != domain]
                    contamination = []

                    for other_domain in other_domains:
                        for key in domain_specific_keys.get(other_domain, []):
                            if any(key in belief_key for belief_key in beliefs.keys()):
                                contamination.append(f"{other_domain}:{key}")

                    if contamination:
                        print(f"  ❌ {domain}: Cross-contamination detected: {contamination}")
                        isolation_maintained = False
                    else:
                        print(f"  ✅ {domain}: No cross-contamination")

                except Exception as e:
                    print(f"  ❌ {domain}: Error: {e}")
                    isolation_maintained = False

        return isolation_maintained

    def test_prior_loading(self) -> bool:
        """Test if prior loading mechanism works"""
        print("\nTesting prior loading mechanism...")

        # Simulate loading priors
        test_passed = True
        for domain in self.domains:
            consolidated_file = self.memory_base / domain / "consolidated" / "beliefs.json"
            if consolidated_file.exists():
                try:
                    with open(consolidated_file, 'r') as f:
                        beliefs = json.load(f)

                    if beliefs:
                        print(f"  ✅ {domain}: {len(beliefs)} beliefs available for loading")
                    else:
                        print(f"  ⚠️ {domain}: Empty beliefs file")
                        test_passed = False
                except Exception as e:
                    print(f"  ❌ {domain}: Cannot load beliefs: {e}")
                    test_passed = False
            else:
                print(f"  ❌ {domain}: No consolidated beliefs to load")
                test_passed = False

        return test_passed


class MemoryFixer:
    """Fixes for common memory persistence issues"""

    def __init__(self, memory_base_path: str = "memory/domains"):
        self.memory_base = Path(memory_base_path)
        self.domains = ["hot_pot", "switch_light", "chem_tile"]

    def create_directory_structure(self):
        """Create the required directory structure"""
        print("\nCreating memory directory structure...")

        for domain in self.domains:
            domain_path = self.memory_base / domain
            episodes_path = domain_path / "episodes"
            consolidated_path = domain_path / "consolidated"

            for path in [domain_path, episodes_path, consolidated_path]:
                path.mkdir(parents=True, exist_ok=True)
                print(f"  ✅ Created: {path}")

    def initialize_consolidated_files(self):
        """Initialize empty consolidated belief files"""
        print("\nInitializing consolidated belief files...")

        for domain in self.domains:
            consolidated_file = self.memory_base / domain / "consolidated" / "beliefs.json"

            if not consolidated_file.exists():
                # Create with domain-specific initial beliefs
                initial_beliefs = self.get_domain_initial_beliefs(domain)

                with open(consolidated_file, 'w') as f:
                    json.dump(initial_beliefs, f, indent=2)

                print(f"  ✅ Initialized: {consolidated_file}")
            else:
                print(f"  ℹ️ Already exists: {consolidated_file}")

    def get_domain_initial_beliefs(self, domain: str) -> Dict:
        """Get domain-specific initial beliefs"""

        if domain == "hot_pot":
            return {
                "heating_rate_mean": {
                    "value": 2.5,
                    "confidence": 0.3,
                    "episode_count": 0
                },
                "measurement_noise": {
                    "value": 1.0,
                    "confidence": 0.3,
                    "episode_count": 0
                }
            }
        elif domain == "switch_light":
            return {
                "connection_probability": {
                    "value": 0.5,
                    "confidence": 0.3,
                    "episode_count": 0
                },
                "wire_reliability": {
                    "value": 0.8,
                    "confidence": 0.3,
                    "episode_count": 0
                }
            }
        elif domain == "chem_tile":
            return {
                "reaction_success_rate": {
                    "value": 0.7,
                    "confidence": 0.3,
                    "episode_count": 0
                },
                "mixing_effectiveness": {
                    "value": 0.8,
                    "confidence": 0.3,
                    "episode_count": 0
                }
            }
        else:
            return {}

    def add_proper_memory_methods(self) -> str:
        """Generate code to add to SimpleWorldModel for proper memory handling"""

        code = '''
# Add these methods to SimpleWorldModel class:

def save_episode_beliefs(self, environment_name: str, episode_num: int):
    """Save beliefs after episode completion"""

    episode_file = f"memory/domains/{environment_name}/episodes/episode_{episode_num:03d}.json"

    # Prepare beliefs for saving
    beliefs_to_save = {
        'episode': episode_num,
        'timestamp': datetime.now().isoformat(),
        'belief_state': self.belief_state,
        'confidence_scores': self.confidence_scores,
        'total_reward': self.total_reward
    }

    # Save episode beliefs
    os.makedirs(os.path.dirname(episode_file), exist_ok=True)
    with open(episode_file, 'w') as f:
        json.dump(beliefs_to_save, f, indent=2)

    print(f"✅ Saved episode {episode_num} beliefs to {episode_file}")

    # Also update consolidated beliefs
    self.update_consolidated_beliefs(environment_name)

def update_consolidated_beliefs(self, environment_name: str):
    """Update consolidated beliefs with latest episode data"""

    consolidated_file = f"memory/domains/{environment_name}/consolidated/beliefs.json"

    # Load existing consolidated beliefs
    if os.path.exists(consolidated_file):
        with open(consolidated_file, 'r') as f:
            consolidated = json.load(f)
    else:
        consolidated = {}

    # Aggregate with current beliefs
    for key, value in self.belief_state.items():
        if key in consolidated:
            old = consolidated[key]

            # Weighted average based on episode count
            if isinstance(old, dict) and 'value' in old:
                old_weight = old.get('episode_count', 1)
                new_weight = 1

                consolidated[key] = {
                    'value': (old['value'] * old_weight + value * new_weight) / (old_weight + new_weight),
                    'confidence': min(0.95, old.get('confidence', 0.5) * 1.1),
                    'episode_count': old_weight + 1
                }
            else:
                consolidated[key] = {
                    'value': value,
                    'confidence': 0.5,
                    'episode_count': 1
                }
        else:
            consolidated[key] = {
                'value': value if not isinstance(value, dict) else value.get('value', value),
                'confidence': 0.5,
                'episode_count': 1
            }

    # Save updated consolidated beliefs
    with open(consolidated_file, 'w') as f:
        json.dump(consolidated, f, indent=2)

    print(f"✅ Updated consolidated beliefs for {environment_name}")

def load_prior_beliefs(self, environment_name: str) -> bool:
    """Load prior beliefs from consolidated memory"""

    consolidated_file = f"memory/domains/{environment_name}/consolidated/beliefs.json"

    if os.path.exists(consolidated_file):
        print(f"📚 Loading prior beliefs from {consolidated_file}")

        with open(consolidated_file, 'r') as f:
            prior_beliefs = json.load(f)

        if prior_beliefs:
            # Load beliefs into current state
            for key, value in prior_beliefs.items():
                if isinstance(value, dict) and 'value' in value:
                    self.belief_state[key] = value['value']

                    # Set confidence based on episode count
                    if value.get('episode_count', 0) > 0:
                        self.confidence_scores[key] = value.get('confidence', 0.5)
                else:
                    self.belief_state[key] = value

            # Adjust prior strength based on accumulated knowledge
            max_episodes = max(
                v.get('episode_count', 0)
                for v in prior_beliefs.values()
                if isinstance(v, dict)
            )

            if max_episodes > 0:
                # Scale prior strength with experience (0.1 to 0.3)
                self.prior_strength = min(0.3, 0.1 + (max_episodes * 0.02))
                print(f"  ✅ Loaded {len(prior_beliefs)} beliefs")
                print(f"  ✅ Prior strength adjusted to {self.prior_strength:.2f} (based on {max_episodes} episodes)")
                return True
        else:
            print(f"  ⚠️ Empty beliefs file")
    else:
        print(f"  ℹ️ No prior beliefs found for {environment_name}")

    return False
'''
        return code


def main():
    """Main diagnostic and fix routine"""

    print("\n" + "=" * 60)
    print("MEMORY PERSISTENCE DIAGNOSTIC AND FIX TOOL")
    print("=" * 60)
    print("\nThis tool will diagnose and fix memory persistence issues")
    print("that are preventing cross-episode learning.\n")

    # Run diagnostics
    diagnostics = MemoryDiagnostics()
    all_tests_passed = diagnostics.run_full_diagnostic()

    if not all_tests_passed:
        print("\n" + "=" * 60)
        print("APPLYING FIXES")
        print("=" * 60)

        fixer = MemoryFixer()

        # Apply fixes
        fixer.create_directory_structure()
        fixer.initialize_consolidated_files()

        print("\n" + "=" * 60)
        print("CODE FIXES NEEDED")
        print("=" * 60)
        print("\nAdd the following methods to your SimpleWorldModel class:")
        print("-" * 60)
        print(fixer.add_proper_memory_methods())

        print("\n" + "=" * 60)
        print("INTEGRATION INSTRUCTIONS")
        print("=" * 60)
        print("""
1. Add the memory methods to SimpleWorldModel class
2. Call load_prior_beliefs() at episode start:

   def start_episode(self, environment_name):
       self.load_prior_beliefs(environment_name)
       # ... rest of initialization

3. Call save_episode_beliefs() at episode end:

   def end_episode(self, environment_name, episode_num):
       self.save_episode_beliefs(environment_name, episode_num)
       # ... rest of cleanup

4. Run a 5-episode test to verify learning curves improve

5. If successful, run full 30-episode validation
""")
    else:
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nMemory persistence appears to be working correctly.")
        print("If learning curves are still flat, check:")
        print("  1. Prior strength scaling")
        print("  2. Belief aggregation weights")
        print("  3. Confidence thresholds")


if __name__ == "__main__":
    main()
