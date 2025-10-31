#!/usr/bin/env python3
"""
Analyze ACE pilot experiment results.

Compares ACE agent to baselines (Observer, Actor, Model-Based) on:
- Accuracy (overall, interventional, counterfactual)
- Token efficiency (tokens per episode, tokens per % accuracy)
- Playbook quality (growth, utilization, convergence)
"""

import json
import sys
from pathlib import Path
import numpy as np
from typing import Dict, List


def load_logs(results_dir: Path) -> Dict[str, List[dict]]:
    """Load all episode logs grouped by agent type."""
    raw_dir = results_dir / 'raw'

    if not raw_dir.exists():
        print(f"Error: {raw_dir} does not exist")
        sys.exit(1)

    # Load all logs
    logs = []
    for log_file in raw_dir.glob('*.json'):
        try:
            with open(log_file) as f:
                logs.append(json.load(f))
        except Exception as e:
            print(f"Warning: Failed to load {log_file}: {e}")

    if not logs:
        print(f"Error: No episode logs found in {raw_dir}")
        sys.exit(1)

    # Group by agent
    by_agent = {}
    for log in logs:
        agent_type = log.get('agent_type', 'unknown')
        if agent_type not in by_agent:
            by_agent[agent_type] = []
        by_agent[agent_type].append(log)

    return by_agent


def compute_agent_stats(agent_logs: List[dict]) -> dict:
    """Compute statistics for an agent type."""
    if not agent_logs:
        return {}

    # Accuracy metrics
    overall_acc = []
    interventional_acc = []
    counterfactual_acc = []

    for log in agent_logs:
        test_results = log.get('test_results', [])
        if test_results:
            # Overall
            overall_acc.append(np.mean([r.get('correct', 0) for r in test_results]))

            # By query type
            interventional = [r for r in test_results if r.get('query_type') == 'interventional']
            if interventional:
                interventional_acc.append(np.mean([r.get('correct', 0) for r in interventional]))

            counterfactual = [r for r in test_results if r.get('query_type') == 'counterfactual']
            if counterfactual:
                counterfactual_acc.append(np.mean([r.get('correct', 0) for r in counterfactual]))

    # Token usage
    tokens_per_ep = [
        log.get('total_input_tokens', 0) + log.get('total_output_tokens', 0)
        for log in agent_logs
    ]

    # Compute efficiency
    mean_acc = np.mean(overall_acc) if overall_acc else 0
    mean_tokens = np.mean(tokens_per_ep) if tokens_per_ep else 0
    tokens_per_pct = mean_tokens / mean_acc if mean_acc > 0 else 0

    stats = {
        'n_episodes': len(agent_logs),
        'accuracy': {
            'overall': {'mean': np.mean(overall_acc), 'std': np.std(overall_acc)} if overall_acc else None,
            'interventional': {'mean': np.mean(interventional_acc), 'std': np.std(interventional_acc)} if interventional_acc else None,
            'counterfactual': {'mean': np.mean(counterfactual_acc), 'std': np.std(counterfactual_acc)} if counterfactual_acc else None,
        },
        'tokens': {
            'mean': np.mean(tokens_per_ep),
            'std': np.std(tokens_per_ep),
            'min': np.min(tokens_per_ep),
            'max': np.max(tokens_per_ep),
        },
        'efficiency': {
            'tokens_per_pct_accuracy': tokens_per_pct
        }
    }

    return stats


def compute_ace_stats(agent_logs: List[dict]) -> dict:
    """Compute ACE-specific statistics."""
    if not agent_logs:
        return {}

    playbook_sizes = []
    delta_items = []

    for log in agent_logs:
        if 'playbook' in log:
            playbook_sizes.append(log['playbook'].get('total_bullets', 0))
            delta_items.append(log['playbook'].get('delta_items_added', 0))

    if not playbook_sizes:
        return {'error': 'No playbook data found'}

    return {
        'playbook_size': {
            'mean': np.mean(playbook_sizes),
            'std': np.std(playbook_sizes),
            'min': np.min(playbook_sizes),
            'max': np.max(playbook_sizes),
        },
        'delta_per_episode': {
            'mean': np.mean(delta_items),
            'std': np.std(delta_items),
        },
        'growth_rate': (playbook_sizes[-1] - playbook_sizes[0]) / len(playbook_sizes) if len(playbook_sizes) > 1 else 0
    }


def print_comparison_table(by_agent: Dict[str, List[dict]]):
    """Print comparison table."""
    print("\n" + "=" * 90)
    print("AGENT COMPARISON")
    print("=" * 90)

    # Compute stats for each agent
    stats = {}
    for agent_type, logs in by_agent.items():
        stats[agent_type] = compute_agent_stats(logs)

    # Print header
    print(f"\n{'Agent':<15} {'Accuracy':<12} {'Tokens/Ep':<15} {'Tokens/%':<12} {'Episodes':<10}")
    print("-" * 90)

    # Print each agent
    for agent_type in ['observer', 'actor', 'model_based', 'a_c_e']:
        if agent_type not in stats:
            continue

        s = stats[agent_type]
        acc = s['accuracy']['overall']
        tokens = s['tokens']
        efficiency = s['efficiency']

        if acc:
            print(f"{agent_type:<15} "
                  f"{acc['mean']:.1%} ± {acc['std']:.1%}   "
                  f"{tokens['mean']:>6,.0f} ± {tokens['std']:<4,.0f}  "
                  f"{efficiency['tokens_per_pct_accuracy']:>6.0f}      "
                  f"{s['n_episodes']:<10}")

    print()


def print_detailed_results(by_agent: Dict[str, List[dict]]):
    """Print detailed results for each agent."""
    print("\n" + "=" * 90)
    print("DETAILED RESULTS")
    print("=" * 90)

    for agent_type in ['observer', 'actor', 'model_based', 'a_c_e']:
        if agent_type not in by_agent:
            continue

        logs = by_agent[agent_type]
        stats = compute_agent_stats(logs)

        print(f"\n{agent_type.upper()}")
        print("-" * 50)
        print(f"Episodes: {stats['n_episodes']}")

        # Accuracy breakdown
        if stats['accuracy']['overall']:
            print(f"\nAccuracy:")
            print(f"  Overall:        {stats['accuracy']['overall']['mean']:.1%} ± {stats['accuracy']['overall']['std']:.1%}")
            if stats['accuracy']['interventional']:
                print(f"  Interventional: {stats['accuracy']['interventional']['mean']:.1%} ± {stats['accuracy']['interventional']['std']:.1%}")
            if stats['accuracy']['counterfactual']:
                print(f"  Counterfactual: {stats['accuracy']['counterfactual']['mean']:.1%} ± {stats['accuracy']['counterfactual']['std']:.1%}")

        # Token usage
        print(f"\nToken Usage:")
        print(f"  Mean: {stats['tokens']['mean']:,.0f} ± {stats['tokens']['std']:,.0f}")
        print(f"  Range: {stats['tokens']['min']:,.0f} - {stats['tokens']['max']:,.0f}")

        # Efficiency
        print(f"\nEfficiency:")
        print(f"  Tokens per % accuracy: {stats['efficiency']['tokens_per_pct_accuracy']:.0f}")

        # ACE-specific
        if agent_type == 'a_c_e':
            ace_stats = compute_ace_stats(logs)
            if 'error' not in ace_stats:
                print(f"\nPlaybook:")
                print(f"  Size: {ace_stats['playbook_size']['mean']:.1f} ± {ace_stats['playbook_size']['std']:.1f} bullets")
                print(f"  Range: {ace_stats['playbook_size']['min']:.0f} - {ace_stats['playbook_size']['max']:.0f}")
                print(f"  Growth rate: {ace_stats['growth_rate']:.2f} bullets/episode")
                print(f"  Delta/episode: {ace_stats['delta_per_episode']['mean']:.1f} ± {ace_stats['delta_per_episode']['std']:.1f}")


def print_decision_guidance(by_agent: Dict[str, List[dict]]):
    """Print guidance on next steps."""
    print("\n" + "=" * 90)
    print("DECISION GUIDANCE")
    print("=" * 90)

    if 'a_c_e' not in by_agent:
        print("\n❌ No ACE results found. Cannot make recommendation.")
        return

    ace_stats = compute_agent_stats(by_agent['a_c_e'])
    ace_acc = ace_stats['accuracy']['overall']['mean'] if ace_stats['accuracy']['overall'] else 0
    ace_tokens = ace_stats['tokens']['mean']

    actor_stats = compute_agent_stats(by_agent.get('actor', []))
    actor_acc = actor_stats['accuracy']['overall']['mean'] if actor_stats.get('accuracy', {}).get('overall') else 0.75
    actor_tokens = actor_stats['tokens']['mean'] if actor_stats.get('tokens') else 22000

    print(f"\nACE Performance: {ace_acc:.1%} @ {ace_tokens:,.0f} tokens/episode")
    print(f"Actor Performance: {actor_acc:.1%} @ {actor_tokens:,.0f} tokens/episode")

    # Calculate efficiency gain
    if actor_tokens > 0 and ace_tokens > 0:
        token_efficiency = actor_tokens / ace_tokens
        accuracy_drop = actor_acc - ace_acc
        print(f"\nEfficiency: {token_efficiency:.1f}× fewer tokens")
        print(f"Accuracy drop: {accuracy_drop:.1%}")

    # Decision
    print("\n" + "-" * 50)
    if ace_acc >= 0.70:
        print("✅ EXCELLENT RESULT - Proceed to full experiment")
        print("   ACE achieves ≥70% accuracy. Strong evidence for H-ACE.")
        print("   Recommended: Run full experiment (600 episodes)")
    elif ace_acc >= 0.65:
        print("✅ GOOD RESULT - Investigate and proceed")
        print("   ACE achieves 65-70% accuracy. Moderate evidence for H-ACE.")
        print("   Recommended: Analyze playbook quality, then run full experiment")
    elif ace_acc >= 0.60:
        print("⚠️  MIXED RESULT - Analyze before proceeding")
        print("   ACE achieves 60-65% accuracy. Weak evidence for H-ACE.")
        print("   Recommended: Deep analysis of playbook items and failure modes")
        print("   Consider: Still publishable as 'limits of context engineering'")
    else:
        print("❌ WEAK RESULT - Debug before full experiment")
        print("   ACE achieves <60% accuracy.")
        print("   Recommended: Check implementation, prompts, playbook quality")
        print("   Debug steps:")
        print("   1. Verify Reflector is extracting insights")
        print("   2. Verify Curator is creating delta items")
        print("   3. Check playbook items are actionable")
        print("   4. Compare to Observer (should beat Observer)")


def main():
    """Main analysis function."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_ace_pilot.py <results_dir>")
        print("Example: python analyze_ace_pilot.py results/ace_pilot")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    print("=" * 90)
    print("ACE PILOT EXPERIMENT ANALYSIS")
    print("=" * 90)
    print(f"\nResults directory: {results_dir}")

    # Load logs
    by_agent = load_logs(results_dir)

    print(f"\nLoaded {sum(len(logs) for logs in by_agent.values())} episodes:")
    for agent_type, logs in by_agent.items():
        print(f"  - {agent_type}: {len(logs)} episodes")

    # Print results
    print_comparison_table(by_agent)
    print_detailed_results(by_agent)
    print_decision_guidance(by_agent)

    print("\n" + "=" * 90)
    print("ANALYSIS COMPLETE")
    print("=" * 90)
    print()


if __name__ == '__main__':
    main()
