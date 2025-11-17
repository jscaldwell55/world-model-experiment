#!/usr/bin/env python3
"""Analyze counterfactual reasoning failures."""

import json
from pathlib import Path
from collections import defaultdict

def analyze_counterfactual_failures(results_dir):
    """Analyze where and why counterfactual reasoning fails."""

    counterfactual_results = []

    # Load all results
    for file in Path(results_dir).glob("*.json"):
        with open(file) as f:
            data = json.load(f)

        agent_type = data.get("agent_type", "unknown")
        environment = data.get("environment", "unknown")

        # Extract counterfactual test results
        for test in data.get("test_results", []):
            if test.get("query_type") == "counterfactual":
                counterfactual_results.append({
                    "agent": agent_type,
                    "env": environment,
                    "question": test.get("query"),
                    "agent_answer": test.get("agent_answer"),
                    "correct": test.get("correct"),
                    "score": test.get("score"),
                    "confidence": test.get("confidence"),
                    "difficulty": test.get("difficulty")
                })

    print("=" * 80)
    print("COUNTERFACTUAL REASONING ANALYSIS")
    print("=" * 80)
    print(f"\nTotal counterfactual questions: {len(counterfactual_results)}")

    # Overall stats
    correct_count = sum(1 for r in counterfactual_results if r["correct"])
    avg_score = sum(r["score"] for r in counterfactual_results) / len(counterfactual_results) if counterfactual_results else 0

    print(f"Correct: {correct_count}/{len(counterfactual_results)} ({100*correct_count/len(counterfactual_results):.1f}%)")
    print(f"Average score: {avg_score:.3f}")

    # By agent
    print("\n" + "=" * 80)
    print("PERFORMANCE BY AGENT")
    print("=" * 80)

    by_agent = defaultdict(list)
    for r in counterfactual_results:
        by_agent[r["agent"]].append(r)

    for agent in sorted(by_agent.keys()):
        results = by_agent[agent]
        scores = [r["score"] for r in results]
        correct = sum(1 for r in results if r["correct"])
        print(f"\n{agent.upper()}:")
        print(f"  Score: {sum(scores)/len(scores):.3f}")
        print(f"  Correct: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")

    # By environment
    print("\n" + "=" * 80)
    print("PERFORMANCE BY ENVIRONMENT")
    print("=" * 80)

    by_env = defaultdict(list)
    for r in counterfactual_results:
        by_env[r["env"]].append(r)

    for env in sorted(by_env.keys()):
        results = by_env[env]
        scores = [r["score"] for r in results]
        correct = sum(1 for r in results if r["correct"])
        print(f"\n{env}:")
        print(f"  Score: {sum(scores)/len(scores):.3f}")
        print(f"  Correct: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")

    # Sample failures
    print("\n" + "=" * 80)
    print("SAMPLE FAILURES")
    print("=" * 80)

    failures = [r for r in counterfactual_results if not r["correct"]]

    for i, failure in enumerate(failures[:10], 1):
        print(f"\n--- Failure {i} ---")
        print(f"Agent: {failure['agent']}")
        print(f"Environment: {failure['env']}")
        print(f"Question: {failure['question']}")
        print(f"Agent Answer: {failure['agent_answer']}")
        print(f"Score: {failure['score']:.2f}")
        print(f"Confidence: {failure['confidence']:.2f}")

    # Sample successes for comparison
    print("\n" + "=" * 80)
    print("SAMPLE SUCCESSES")
    print("=" * 80)

    successes = [r for r in counterfactual_results if r["correct"]]

    for i, success in enumerate(successes[:5], 1):
        print(f"\n--- Success {i} ---")
        print(f"Agent: {success['agent']}")
        print(f"Environment: {success['env']}")
        print(f"Question: {success['question']}")
        print(f"Agent Answer: {success['agent_answer']}")
        print(f"Score: {success['score']:.2f}")
        print(f"Confidence: {success['confidence']:.2f}")

if __name__ == "__main__":
    analyze_counterfactual_failures("results/actor_baseline_nov15/raw")
