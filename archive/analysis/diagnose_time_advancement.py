#!/usr/bin/env python3
"""
Diagnostic script to systematically check time advancement across all episodes.

This script analyzes episode JSON files to identify cases where time fails to advance
on instant actions (measure_temp, toggle_stove, flip_switch, measure_light, etc.).
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

class TimeAdvancementDiagnostic:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.violations = []
        self.action_stats = defaultdict(lambda: {"total": 0, "violations": 0, "deltas": []})

    def analyze_episode(self, episode_file: Path) -> Dict:
        """Analyze a single episode for time advancement issues."""
        with open(episode_file) as f:
            data = json.load(f)

        episode_id = data["episode_id"]
        environment = data["environment"]
        agent_type = data["agent_type"]
        steps = data["steps"]

        episode_violations = []
        prev_time = None

        for i, step in enumerate(steps):
            step_num = step["step_num"]
            action = step["action"]
            obs = step["observation"]
            current_time = obs.get("time", obs.get("time_elapsed", None))

            if current_time is None:
                continue

            # Calculate time delta
            if prev_time is not None:
                time_delta = current_time - prev_time
            else:
                time_delta = 0.0  # First step

            # Categorize action type
            if action is None:
                action_name = "None"
            else:
                action_name = action.split("(")[0] if "(" in action else action

            # Check for violations (time delta = 0 on non-first steps)
            is_violation = (i > 0 and time_delta == 0.0)

            # Record statistics
            self.action_stats[action_name]["total"] += 1
            self.action_stats[action_name]["deltas"].append(time_delta)

            if is_violation:
                self.action_stats[action_name]["violations"] += 1
                violation = {
                    "episode": episode_id,
                    "step_num": step_num,
                    "action": action,
                    "prev_time": prev_time,
                    "current_time": current_time,
                    "time_delta": time_delta
                }
                episode_violations.append(violation)
                self.violations.append(violation)

            prev_time = current_time

        return {
            "episode_id": episode_id,
            "environment": environment,
            "agent_type": agent_type,
            "num_steps": len(steps),
            "num_violations": len(episode_violations),
            "violations": episode_violations
        }

    def run_diagnostics(self) -> Dict:
        """Run diagnostics on all episode files."""
        episode_files = sorted(self.results_dir.glob("*.json"))

        print(f"Found {len(episode_files)} episode files in {self.results_dir}")
        print("=" * 80)

        results = []
        for episode_file in episode_files:
            result = self.analyze_episode(episode_file)
            results.append(result)

        return results

    def print_detailed_report(self, results: List[Dict]):
        """Print detailed report of findings."""
        print("\n" + "=" * 80)
        print("DETAILED TIME ADVANCEMENT REPORT")
        print("=" * 80)

        # Print per-episode details
        episodes_with_bugs = 0
        for result in results:
            if result["num_violations"] > 0:
                episodes_with_bugs += 1
                print(f"\nEpisode: {result['episode_id']}")
                print(f"  Environment: {result['environment']}")
                print(f"  Agent: {result['agent_type']}")
                print(f"  Violations: {result['num_violations']}/{result['num_steps']} steps")

                for v in result["violations"][:5]:  # Show first 5 violations
                    print(f"    Step {v['step_num']}: {v['action']} -> "
                          f"time: {v['prev_time']:.1f} -> {v['current_time']:.1f} "
                          f"(delta: {v['time_delta']:.1f}) ❌")

                if len(result["violations"]) > 5:
                    print(f"    ... and {len(result['violations']) - 5} more violations")

        # Print summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"\nTotal episodes analyzed: {len(results)}")
        print(f"Episodes with time bugs: {episodes_with_bugs} ({100*episodes_with_bugs/len(results):.1f}%)")
        print(f"Total time advancement violations: {len(self.violations)}")

        # Print action-specific statistics
        print("\n" + "-" * 80)
        print("Actions that failed to advance time:")
        print("-" * 80)
        print(f"{'Action':<20} {'Total':<10} {'Violations':<12} {'Rate':<10} {'Avg Delta':<12}")
        print("-" * 80)

        for action, stats in sorted(self.action_stats.items(),
                                    key=lambda x: x[1]["violations"],
                                    reverse=True):
            total = stats["total"]
            violations = stats["violations"]
            rate = violations / total if total > 0 else 0
            avg_delta = sum(stats["deltas"]) / len(stats["deltas"]) if stats["deltas"] else 0

            print(f"{action:<20} {total:<10} {violations:<12} {rate*100:>6.1f}%    {avg_delta:>6.2f}s")

        # Identify instant actions that should advance time
        print("\n" + "=" * 80)
        print("EXPECTED vs ACTUAL BEHAVIOR")
        print("=" * 80)

        instant_actions = ["measure_temp", "toggle_stove", "touch_pot",
                          "flip_switch", "measure_light"]

        for action in instant_actions:
            if action in self.action_stats:
                stats = self.action_stats[action]
                total = stats["total"]
                violations = stats["violations"]
                print(f"\n{action}():")
                print(f"  Expected: Should advance time by ~1.0s on each call")
                if violations > 0:
                    print(f"  Actual: ❌ Time did NOT advance on {violations}/{total} calls ({100*violations/total:.1f}%)")
                else:
                    print(f"  Actual: ✓ Time advanced correctly on all {total} calls")

        print("\n" + "=" * 80)
        print("DIAGNOSIS")
        print("=" * 80)

        if len(self.violations) > 0:
            print("\n❌ TIME ADVANCEMENT BUG CONFIRMED")
            print("\nRoot cause: Instant actions (measure_temp, toggle_stove, etc.) are NOT")
            print("advancing the environment time counter. Only wait() actions advance time.")
            print("\nThis breaks belief updates because:")
            print("  1. Multiple observations occur at the same timestamp")
            print("  2. Belief update math requires time_delta > 0 to compute heating rates")
            print("  3. Division by zero or undefined behavior occurs")
            print("\nConclusion: The time advancement fix was NOT successfully applied.")
        else:
            print("\n✓ Time advancement appears to be working correctly")
            print("  All instant actions advanced time as expected")

    def save_violations_json(self, output_file: str = "time_violations.json"):
        """Save violations to JSON file for further analysis."""
        output = {
            "total_violations": len(self.violations),
            "violations": self.violations,
            "action_stats": {
                action: {
                    "total": stats["total"],
                    "violations": stats["violations"],
                    "violation_rate": stats["violations"] / stats["total"] if stats["total"] > 0 else 0
                }
                for action, stats in self.action_stats.items()
            }
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nDetailed violations saved to: {output_file}")


def main():
    # Default to pilot_h1h5 results
    results_dir = "results/pilot_h1h5/raw"

    if len(sys.argv) > 1:
        results_dir = sys.argv[1]

    print(f"Running time advancement diagnostics on: {results_dir}")

    diagnostic = TimeAdvancementDiagnostic(results_dir)
    results = diagnostic.run_diagnostics()
    diagnostic.print_detailed_report(results)
    diagnostic.save_violations_json()


if __name__ == "__main__":
    main()
