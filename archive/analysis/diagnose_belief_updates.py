#!/usr/bin/env python3
"""
Diagnostic script to check belief state updates and identify update failures.

This script analyzes episode JSON files to verify that belief states update correctly
in response to observations, and identifies cases where updates fail or behave incorrectly.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

class BeliefUpdateDiagnostic:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.update_failures = []
        self.update_stats = {
            "total_updates": 0,
            "no_change": 0,
            "wrong_direction": 0,
            "uncertainty_increased": 0,
            "successful": 0
        }

    def analyze_episode(self, episode_file: Path) -> Dict:
        """Analyze belief updates in a single episode."""
        with open(episode_file) as f:
            data = json.load(f)

        episode_id = data["episode_id"]
        agent_type = data["agent_type"]

        # Only Actor and Model-Based agents have belief states
        if agent_type not in ["actor", "model_based"]:
            return {
                "episode_id": episode_id,
                "agent_type": agent_type,
                "has_beliefs": False
            }

        environment = data["environment"]
        steps = data["steps"]

        episode_issues = []
        prev_belief = None
        prev_obs = None

        for i, step in enumerate(steps):
            if "belief_state" not in step:
                continue

            belief = step["belief_state"]
            obs = step["observation"]
            action = step.get("action", "unknown")

            # Check if belief updated since last step
            if prev_belief is not None and prev_obs is not None:
                self.update_stats["total_updates"] += 1

                # Check if belief changed at all
                if belief == prev_belief:
                    self.update_stats["no_change"] += 1
                    issue = {
                        "step": i,
                        "type": "no_change",
                        "action": action,
                        "prev_obs": prev_obs,
                        "belief_before": prev_belief,
                        "belief_after": belief
                    }
                    episode_issues.append(issue)
                    self.update_failures.append({
                        "episode": episode_id,
                        **issue
                    })

                # For HotPot, check heating rate updates
                elif environment == "HotPotLab":
                    # Check if uncertainty decreased (learning)
                    prev_std = prev_belief.get("heating_rate_std", 0)
                    curr_std = belief.get("heating_rate_std", 0)

                    if curr_std >= prev_std:
                        self.update_stats["uncertainty_increased"] += 1
                        issue = {
                            "step": i,
                            "type": "uncertainty_increased",
                            "action": action,
                            "prev_std": prev_std,
                            "curr_std": curr_std,
                            "obs_time": obs.get("time", None)
                        }
                        episode_issues.append(issue)

                    # Check if belief updated in sensible direction
                    if "measured_temp" in prev_obs:
                        prev_time = prev_obs.get("time", 0)
                        curr_time = obs.get("time", 0)
                        time_delta = curr_time - prev_time

                        if time_delta == 0:
                            # Can't compute heating rate with zero time delta!
                            issue = {
                                "step": i,
                                "type": "zero_time_delta",
                                "action": action,
                                "prev_time": prev_time,
                                "curr_time": curr_time,
                                "message": "Cannot update belief: time_delta = 0"
                            }
                            episode_issues.append(issue)
                            self.update_failures.append({
                                "episode": episode_id,
                                **issue
                            })

                if belief != prev_belief and "zero_time_delta" not in [iss.get("type") for iss in episode_issues]:
                    self.update_stats["successful"] += 1

            prev_belief = belief.copy() if isinstance(belief, dict) else belief
            prev_obs = obs.copy() if isinstance(obs, dict) else obs

        return {
            "episode_id": episode_id,
            "environment": environment,
            "agent_type": agent_type,
            "has_beliefs": True,
            "num_steps": len(steps),
            "num_issues": len(episode_issues),
            "issues": episode_issues
        }

    def run_diagnostics(self) -> List[Dict]:
        """Run diagnostics on all episode files."""
        episode_files = sorted(self.results_dir.glob("*.json"))

        print(f"Found {len(episode_files)} episode files")
        print("=" * 80)

        results = []
        for episode_file in episode_files:
            result = self.analyze_episode(episode_file)
            if result["has_beliefs"]:
                results.append(result)

        return results

    def print_detailed_report(self, results: List[Dict]):
        """Print detailed report of belief update issues."""
        print("\n" + "=" * 80)
        print("BELIEF UPDATE DIAGNOSTIC REPORT")
        print("=" * 80)

        # Show episodes with issues
        episodes_with_issues = [r for r in results if r["num_issues"] > 0]

        if episodes_with_issues:
            print(f"\nFound {len(episodes_with_issues)} episodes with belief update issues:")

            for result in episodes_with_issues[:5]:  # Show first 5
                print(f"\nEpisode: {result['episode_id']}")
                print(f"  Environment: {result['environment']}")
                print(f"  Agent: {result['agent_type']}")
                print(f"  Issues: {result['num_issues']}")

                for issue in result['issues'][:3]:  # Show first 3 issues
                    print(f"\n  Step {issue['step']}: {issue['type']}")
                    if issue['type'] == 'zero_time_delta':
                        print(f"    Time: {issue['prev_time']} -> {issue['curr_time']} (delta: 0.0)")
                        print(f"    ❌ {issue['message']}")
                    elif issue['type'] == 'no_change':
                        print(f"    Belief state did not update despite new observation")
                    elif issue['type'] == 'uncertainty_increased':
                        print(f"    Uncertainty: {issue['prev_std']:.3f} -> {issue['curr_std']:.3f} (increased!)")

        # Print summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"\nTotal episodes with beliefs: {len(results)}")
        print(f"Episodes with issues: {len(episodes_with_issues)}")
        print(f"\nBelief updates analyzed: {self.update_stats['total_updates']}")
        print(f"  Successful updates: {self.update_stats['successful']} "
              f"({100*self.update_stats['successful']/max(1,self.update_stats['total_updates']):.1f}%)")
        print(f"  No change (belief didn't update): {self.update_stats['no_change']} "
              f"({100*self.update_stats['no_change']/max(1,self.update_stats['total_updates']):.1f}%)")
        print(f"  Uncertainty increased (should decrease): {self.update_stats['uncertainty_increased']} "
              f"({100*self.update_stats['uncertainty_increased']/max(1,self.update_stats['total_updates']):.1f}%)")

        # Count zero time delta issues
        zero_time_delta_count = sum(1 for f in self.update_failures if f.get('type') == 'zero_time_delta')

        print("\n" + "=" * 80)
        print("ROOT CAUSE ANALYSIS")
        print("=" * 80)

        if zero_time_delta_count > 0:
            print(f"\n❌ CRITICAL: {zero_time_delta_count} belief updates failed due to time_delta = 0")
            print("\nWhen consecutive observations have the same timestamp:")
            print("  1. Cannot compute heating rate (requires time_delta > 0)")
            print("  2. Bayesian update math breaks (division by zero)")
            print("  3. Belief state cannot learn from observations")
            print("  4. Surprisal cannot decrease (agent appears not to learn)")
            print("\nThis is a DIRECT consequence of the time advancement bug.")
            print("Instant actions (measure_temp, toggle_stove) don't advance time,")
            print("so observations contain stale timestamps.")

        if self.update_stats["no_change"] > 0:
            print(f"\n⚠️  {self.update_stats['no_change']} updates resulted in no belief change")
            print("This could indicate:")
            print("  - Observations don't provide new information")
            print("  - Update logic is broken")
            print("  - Edge cases not handled properly")

        if self.update_stats["successful"] > 0:
            success_rate = 100 * self.update_stats["successful"] / self.update_stats["total_updates"]
            if success_rate < 50:
                print(f"\n❌ Only {success_rate:.1f}% of belief updates were successful")
                print("This indicates widespread failure in the belief update mechanism.")
            else:
                print(f"\n✓ {success_rate:.1f}% of belief updates were successful")

    def save_failures_json(self, output_file: str = "belief_update_failures.json"):
        """Save update failures to JSON for further analysis."""
        output = {
            "total_failures": len(self.update_failures),
            "statistics": self.update_stats,
            "failures": self.update_failures
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nDetailed failures saved to: {output_file}")


def main():
    results_dir = "results/pilot_h1h5/raw"

    if len(sys.argv) > 1:
        results_dir = sys.argv[1]

    print(f"Running belief update diagnostics on: {results_dir}")

    diagnostic = BeliefUpdateDiagnostic(results_dir)
    results = diagnostic.run_diagnostics()
    diagnostic.print_detailed_report(results)
    diagnostic.save_failures_json()


if __name__ == "__main__":
    main()
