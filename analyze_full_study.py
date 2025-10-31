#!/usr/bin/env python3
"""Analyze full study results"""

import json
import os
from pathlib import Path
from collections import defaultdict
import statistics

def load_results(results_dir):
    """Load all successful episode results"""
    raw_dir = Path(results_dir) / "raw"
    results = []

    for file in sorted(raw_dir.glob("*.json")):
        try:
            with open(file) as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return results

def analyze_by_agent(results):
    """Analyze results by agent type"""
    agent_stats = defaultdict(lambda: {
        'episodes': 0,
        'total_score': 0,
        'input_tokens': 0,
        'output_tokens': 0,
        'scores': [],
        'num_questions': 0
    })

    for r in results:
        agent = r.get('agent_type', 'unknown')
        stats = agent_stats[agent]

        stats['episodes'] += 1

        # Token usage
        stats['input_tokens'] += r.get('total_input_tokens', 0)
        stats['output_tokens'] += r.get('total_output_tokens', 0)

        # Test results
        test_results = r.get('test_results', [])
        episode_score = sum(t.get('score', 0) for t in test_results)
        stats['total_score'] += episode_score
        stats['scores'].append(episode_score)
        stats['num_questions'] += len(test_results)

    return agent_stats

def analyze_by_env(results):
    """Analyze results by environment"""
    env_stats = defaultdict(lambda: {
        'episodes': 0,
        'total_score': 0,
        'scores': []
    })

    for r in results:
        env = r.get('environment', 'unknown')
        stats = env_stats[env]

        stats['episodes'] += 1

        # Test results
        test_results = r.get('test_results', [])
        episode_score = sum(t.get('score', 0) for t in test_results)
        stats['total_score'] += episode_score
        stats['scores'].append(episode_score)

    return env_stats

def analyze_by_agent_env(results):
    """Analyze results by agent x environment"""
    combo_stats = defaultdict(lambda: {
        'episodes': 0,
        'total_score': 0,
        'scores': []
    })

    for r in results:
        agent = r.get('agent_type', 'unknown')
        env = r.get('environment', 'unknown')
        key = f"{agent}_{env}"
        stats = combo_stats[key]

        stats['episodes'] += 1

        # Test results
        test_results = r.get('test_results', [])
        episode_score = sum(t.get('score', 0) for t in test_results)
        stats['total_score'] += episode_score
        stats['scores'].append(episode_score)

    return combo_stats

def main():
    results_dir = "results/full_study_v2"

    print("=" * 80)
    print("FULL STUDY RESULTS ANALYSIS")
    print("=" * 80)
    print()

    # Load results
    results = load_results(results_dir)
    print(f"Loaded {len(results)} successful episodes\n")

    # Overall stats
    total_score = sum(sum(t.get('score', 0) for t in r.get('test_results', [])) for r in results)
    total_questions = sum(len(r.get('test_results', [])) for r in results)
    max_score = total_questions * 1.0  # Each question is worth max 1.0
    total_input = sum(r.get('total_input_tokens', 0) for r in results)
    total_output = sum(r.get('total_output_tokens', 0) for r in results)
    total_cost = sum(r.get('cost', {}).get('total_cost_usd', 0) for r in results)
    total_duration = sum(r.get('duration_seconds', 0) for r in results)

    print("OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total episodes: {len(results)}")
    print(f"Total questions answered: {total_questions}")
    print(f"Total score: {total_score:.2f} / {max_score:.0f} ({100*total_score/max_score:.1f}%)")
    print(f"Average score per episode: {total_score/len(results):.2f} / {total_questions/len(results):.1f}")
    print(f"Average score per question: {total_score/total_questions:.3f}")
    print(f"Total input tokens: {total_input:,}")
    print(f"Total output tokens: {total_output:,}")
    print(f"Total tokens: {total_input + total_output:,}")
    print(f"Total cost: ${total_cost:,.2f}")
    print(f"Average cost per episode: ${total_cost/len(results):.2f}")
    print(f"Total duration: {total_duration/3600:.1f} hours")
    print(f"Average duration per episode: {total_duration/len(results):.1f} seconds")
    print()

    # Agent analysis
    print("PERFORMANCE BY AGENT TYPE")
    print("-" * 80)
    agent_stats = analyze_by_agent(results)

    for agent in sorted(agent_stats.keys()):
        stats = agent_stats[agent]
        avg_questions = stats['num_questions'] / stats['episodes']
        max_score_agent = stats['num_questions'] * 1.0
        avg_score = stats['total_score'] / stats['episodes']
        total_tokens = stats['input_tokens'] + stats['output_tokens']
        avg_tokens = total_tokens / stats['episodes']

        print(f"\n{agent}:")
        print(f"  Episodes: {stats['episodes']}")
        print(f"  Total score: {stats['total_score']:.2f} / {max_score_agent:.0f} ({100*stats['total_score']/max_score_agent:.1f}%)")
        print(f"  Avg score/episode: {avg_score:.2f} / {avg_questions:.1f} ({100*avg_score/avg_questions:.1f}%)")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Avg tokens/episode: {avg_tokens:,.0f}")

        if len(stats['scores']) > 1:
            print(f"  Score std: {statistics.stdev(stats['scores']):.3f}")

    print("\n")

    # Environment analysis
    print("PERFORMANCE BY ENVIRONMENT")
    print("-" * 80)
    env_stats = analyze_by_env(results)

    for env in sorted(env_stats.keys()):
        stats = env_stats[env]
        avg_score = stats['total_score'] / stats['episodes']

        print(f"\n{env}:")
        print(f"  Episodes: {stats['episodes']}")
        print(f"  Avg score/episode: {avg_score:.2f}")

        if len(stats['scores']) > 1:
            print(f"  Score std: {statistics.stdev(stats['scores']):.3f}")

    print("\n")

    # Combo analysis
    print("PERFORMANCE BY AGENT x ENVIRONMENT")
    print("-" * 80)
    combo_stats = analyze_by_agent_env(results)

    # Organize by agent
    agents = set()
    envs = set()
    for key in combo_stats.keys():
        parts = key.split('_', 1)
        if len(parts) >= 2:
            # Handle multi-part agent names like a_c_e
            agent = parts[0]
            env = parts[1]
            # Try to reconstruct from known agents
            if 'a_c_e' in key:
                agent = 'a_c_e'
                env = key.replace('a_c_e_', '')
            agents.add(agent)
            envs.add(env)

    for agent in sorted(agents):
        print(f"\n{agent}:")
        for env in sorted(envs):
            key = f"{agent}_{env}"
            if key in combo_stats:
                stats = combo_stats[key]
                avg_score = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
                print(f"  {env}: {stats['episodes']} eps, {avg_score:.2f} avg score")

    print("\n")
    print("=" * 80)

if __name__ == "__main__":
    main()
