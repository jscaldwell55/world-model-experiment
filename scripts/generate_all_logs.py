#!/usr/bin/env python3
"""Generate detailed logs for all episodes in an experiment"""
import argparse
import json
import sys
from pathlib import Path

def generate_episode_log(episode_file):
    """Generate a detailed log for a single episode"""
    with open(episode_file) as f:
        ep = json.load(f)

    lines = []
    lines.append("=" * 80)
    lines.append(f"EPISODE: {ep.get('episode_id', 'unknown')}")
    lines.append("=" * 80)
    lines.append(f"Environment: {ep.get('environment', 'unknown')}")
    lines.append(f"Agent Type: {ep.get('agent_type', 'unknown')}")
    lines.append(f"Seed: {ep.get('seed', 0)}")
    lines.append(f"Total Steps: {len(ep.get('steps', []))}")

    # Metadata
    if 'metadata' in ep:
        lines.append("\n" + "-" * 80)
        lines.append("METADATA")
        lines.append("-" * 80)
        for key, value in ep['metadata'].items():
            lines.append(f"{key}: {value}")

    # Ground truth (for environments that have it)
    if 'ground_truth' in ep and ep['ground_truth']:
        lines.append("\n" + "-" * 80)
        lines.append("GROUND TRUTH")
        lines.append("-" * 80)
        for key, value in ep['ground_truth'].items():
            lines.append(f"{key}: {value}")

    # Steps
    lines.append("\n" + "=" * 80)
    lines.append(f"STEP-BY-STEP TRAJECTORY ({len(ep.get('steps', []))} steps)")
    lines.append("=" * 80)

    for step in ep.get('steps', []):
        lines.append("\n" + "-" * 80)
        lines.append(f"Step {step.get('step_num', '?')}")
        lines.append("-" * 80)

        # Action
        lines.append(f"Action: {step.get('action', 'None')}")

        # Observation
        obs = step.get('observation', {})
        if obs:
            lines.append("\nObservation:")
            for key, value in obs.items():
                if isinstance(value, (int, float, str, bool)):
                    lines.append(f"  {key}: {value}")
                elif value is None:
                    lines.append(f"  {key}: None")
                else:
                    lines.append(f"  {key}: {str(value)[:100]}")

        # Belief state
        belief = step.get('belief_state', {})
        if belief:
            lines.append("\nBelief State:")
            for key, value in belief.items():
                if isinstance(value, (int, float, str, bool)):
                    lines.append(f"  {key}: {value}")
                elif isinstance(value, dict):
                    lines.append(f"  {key}:")
                    for k, v in value.items():
                        lines.append(f"    {k}: {v}")
                else:
                    lines.append(f"  {key}: {str(value)[:100]}")

        # Surprisal
        lines.append(f"\nSurprisal: {step.get('surprisal', 0.0):.4f}")

    # Test results
    if 'test_results' in ep and ep['test_results']:
        lines.append("\n" + "=" * 80)
        lines.append("TEST RESULTS")
        lines.append("=" * 80)

        correct = sum(1 for r in ep['test_results'] if r.get('correct'))
        total = len(ep['test_results'])
        accuracy = 100 * correct / total if total > 0 else 0

        lines.append(f"\nOverall Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        lines.append("\nDetailed Results:")

        for i, result in enumerate(ep['test_results'], 1):
            status = "✓ CORRECT" if result.get('correct') else "✗ INCORRECT"
            lines.append(f"\n{i}. {status}")
            lines.append(f"   Query: {result.get('query', 'N/A')}")
            lines.append(f"   Agent Answer: {result.get('agent_answer', 'N/A')}")
            if 'correct_answer' in result:
                lines.append(f"   Correct Answer: {result['correct_answer']}")
            if 'log_likelihood' in result:
                lines.append(f"   Log Likelihood: {result['log_likelihood']:.4f}")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description='Generate logs for all episodes')
    parser.add_argument('results_dir', help='Results directory containing episode JSON files')
    parser.add_argument('--output-dir', default='logs', help='Output directory for log files')
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all episode JSON files
    episode_files = sorted(results_path.glob('*_ep*.json'))

    if not episode_files:
        print(f"No episode files found in {results_dir}")
        return 1

    print(f"Found {len(episode_files)} episodes")
    print(f"Generating logs to {output_path}/")

    for episode_file in episode_files:
        log_content = generate_episode_log(episode_file)

        # Create output filename
        output_file = output_path / f"{episode_file.stem}.log"

        with open(output_file, 'w') as f:
            f.write(log_content)

        print(f"  ✓ {episode_file.name} -> {output_file.name}")

    print(f"\n✓ Generated {len(episode_files)} log files in {output_path}/")
    return 0

if __name__ == '__main__':
    sys.exit(main())
