#!/usr/bin/env python3
"""
Comprehensive analysis of pilot_h1h5_fixed results
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import statistics

def load_all_results(results_dir):
    """Load all JSON result files"""
    files = glob.glob(f"{results_dir}/raw/*.json")
    results = []
    for f in files:
        with open(f, 'r') as file:
            results.append(json.load(file))
    return results

def analyze_test_performance(results):
    """Analyze test question performance by agent and environment"""
    performance = defaultdict(lambda: defaultdict(list))

    for result in results:
        agent = result['agent_type']
        env = result['environment']

        # Collect scores for each test result
        for test in result.get('test_results', []):
            performance[agent][env].append({
                'score': test['score'],
                'confidence': test['confidence'],
                'difficulty': test['difficulty'],
                'query_type': test['query_type'],
                'correct': test['correct']
            })

    return performance

def analyze_token_usage(results):
    """Analyze token usage and efficiency"""
    token_stats = defaultdict(lambda: defaultdict(list))

    for result in results:
        agent = result['agent_type']
        env = result['environment']

        token_stats[agent][env].append({
            'input_tokens': result.get('total_input_tokens', 0),
            'output_tokens': result.get('total_output_tokens', 0),
            'api_calls': result.get('total_api_calls', 0),
            'duration': result.get('duration_seconds', 0)
        })

    return token_stats

def analyze_surprisal(results):
    """Analyze surprisal patterns"""
    surprisal_data = defaultdict(lambda: defaultdict(list))

    for result in results:
        agent = result['agent_type']
        env = result['environment']

        for step in result.get('steps', []):
            if step.get('surprisal') is not None:
                surprisal_data[agent][env].append(step['surprisal'])

    return surprisal_data

def analyze_belief_updates(results):
    """Analyze belief state updates for switch_light environment"""
    belief_data = defaultdict(list)

    for result in results:
        if result['environment'] != 'SwitchLight':
            continue

        agent = result['agent_type']

        # Track belief evolution
        belief_trajectory = []
        for step in result.get('steps', []):
            belief = step.get('belief_state', {})
            if 'wiring_probs' in belief:
                belief_trajectory.append({
                    'step': step['step_num'],
                    'layout_A': belief['wiring_probs'].get('layout_A', 0),
                    'layout_B': belief['wiring_probs'].get('layout_B', 0),
                    'failure_prob': belief.get('failure_prob', 0)
                })

        ground_truth = result.get('ground_truth', {})
        actual_layout = ground_truth.get('wire_layout', 'unknown')
        faulty_relay = ground_truth.get('faulty_relay', False)

        belief_data[agent].append({
            'episode': result['episode_id'],
            'trajectory': belief_trajectory,
            'ground_truth_layout': actual_layout,
            'faulty_relay': faulty_relay
        })

    return belief_data

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def main():
    results_dir = "results/pilot_h1h5_fixed"
    results = load_all_results(results_dir)

    print_section(f"PILOT H1-H5 RESULTS ANALYSIS ({len(results)} episodes)")

    # Basic statistics
    agents = set(r['agent_type'] for r in results)
    envs = set(r['environment'] for r in results)

    print(f"\nAgent Types: {sorted(agents)}")
    print(f"Environments: {sorted(envs)}")
    print(f"Episodes per condition: {len(results) // (len(agents) * len(envs))}")

    # Test Performance Analysis
    print_section("TEST PERFORMANCE BY AGENT TYPE")
    performance = analyze_test_performance(results)

    for agent in sorted(agents):
        print(f"\n{agent.upper()} Agent:")
        for env in sorted(envs):
            if env in performance[agent]:
                scores = [t['score'] for t in performance[agent][env]]
                confidences = [t['confidence'] for t in performance[agent][env]]

                avg_score = statistics.mean(scores) if scores else 0
                avg_confidence = statistics.mean(confidences) if confidences else 0

                print(f"  {env:20s}: Avg Score = {avg_score:.3f}, Avg Confidence = {avg_confidence:.3f}, N = {len(scores)}")

                # Breakdown by difficulty
                by_difficulty = defaultdict(list)
                for t in performance[agent][env]:
                    by_difficulty[t['difficulty']].append(t['score'])

                for diff in ['easy', 'medium', 'hard']:
                    if diff in by_difficulty:
                        diff_scores = by_difficulty[diff]
                        print(f"    {diff:8s}: {statistics.mean(diff_scores):.3f} (n={len(diff_scores)})")

    # Overall agent comparison
    print_section("OVERALL AGENT COMPARISON (All Environments)")

    agent_totals = {}
    for agent in sorted(agents):
        all_scores = []
        for env in envs:
            if env in performance[agent]:
                all_scores.extend([t['score'] for t in performance[agent][env]])

        agent_totals[agent] = {
            'mean': statistics.mean(all_scores) if all_scores else 0,
            'stdev': statistics.stdev(all_scores) if len(all_scores) > 1 else 0,
            'n': len(all_scores)
        }

    print("\nAgent Rankings by Mean Score:")
    ranked = sorted(agent_totals.items(), key=lambda x: x[1]['mean'], reverse=True)
    for i, (agent, stats) in enumerate(ranked, 1):
        print(f"{i}. {agent:15s}: {stats['mean']:.3f} ± {stats['stdev']:.3f} (n={stats['n']})")

    # Token Usage Analysis
    print_section("TOKEN USAGE & EFFICIENCY")
    token_stats = analyze_token_usage(results)

    for agent in sorted(agents):
        print(f"\n{agent.upper()} Agent:")
        for env in sorted(envs):
            if env in token_stats[agent]:
                stats = token_stats[agent][env]
                avg_input = statistics.mean([s['input_tokens'] for s in stats])
                avg_output = statistics.mean([s['output_tokens'] for s in stats])
                avg_calls = statistics.mean([s['api_calls'] for s in stats])
                avg_duration = statistics.mean([s['duration'] for s in stats])

                print(f"  {env:20s}:")
                print(f"    Input tokens:  {avg_input:8.1f}")
                print(f"    Output tokens: {avg_output:8.1f}")
                print(f"    Total tokens:  {avg_input + avg_output:8.1f}")
                print(f"    API calls:     {avg_calls:8.1f}")
                print(f"    Duration (s):  {avg_duration:8.1f}")

    # Surprisal Analysis
    print_section("SURPRISAL ANALYSIS")
    surprisal_data = analyze_surprisal(results)

    for agent in sorted(agents):
        print(f"\n{agent.upper()} Agent:")
        for env in sorted(envs):
            if env in surprisal_data[agent] and surprisal_data[agent][env]:
                values = [s for s in surprisal_data[agent][env] if s > 0]  # Exclude 0s
                if values:
                    avg = statistics.mean(values)
                    max_val = max(values)
                    print(f"  {env:20s}: Mean = {avg:.3f}, Max = {max_val:.3f}, Count = {len(values)}")

    # Belief State Analysis (SwitchLight only)
    print_section("BELIEF STATE ANALYSIS (SwitchLight Environment)")
    belief_data = analyze_belief_updates(results)

    for agent in sorted(agents):
        if agent not in belief_data:
            continue

        print(f"\n{agent.upper()} Agent:")
        for episode_data in belief_data[agent]:
            ep_id = episode_data['episode']
            trajectory = episode_data['trajectory']
            truth = episode_data['ground_truth_layout']
            faulty = episode_data['faulty_relay']

            if trajectory:
                final = trajectory[-1]
                print(f"  {ep_id}:")
                print(f"    Ground truth: {truth}, Faulty relay: {faulty}")
                print(f"    Final beliefs: A={final['layout_A']:.2f}, B={final['layout_B']:.2f}, Failure={final['failure_prob']:.2f}")

                # Check if agent converged to correct layout
                if truth == 'layout_A':
                    correct_prob = final['layout_A']
                else:
                    correct_prob = final['layout_B']
                print(f"    Confidence in correct layout: {correct_prob:.2f}")

    # Query Type Analysis
    print_section("PERFORMANCE BY QUERY TYPE")

    query_perf = defaultdict(lambda: defaultdict(list))
    for result in results:
        agent = result['agent_type']
        for test in result.get('test_results', []):
            query_perf[agent][test['query_type']].append(test['score'])

    for agent in sorted(agents):
        print(f"\n{agent.upper()} Agent:")
        for qtype in sorted(query_perf[agent].keys()):
            scores = query_perf[agent][qtype]
            print(f"  {qtype:20s}: {statistics.mean(scores):.3f} (n={len(scores)})")

    # Detailed Error Analysis
    print_section("COMMON ERRORS BY AGENT")

    for agent in sorted(agents):
        print(f"\n{agent.upper()} Agent:")
        errors = []
        for result in results:
            if result['agent_type'] != agent:
                continue
            for test in result.get('test_results', []):
                if not test['correct']:
                    errors.append({
                        'query': test['query'][:80],
                        'score': test['score'],
                        'difficulty': test['difficulty'],
                        'type': test['query_type']
                    })

        print(f"  Total errors: {len(errors)}")
        # Show a few examples
        for i, err in enumerate(errors[:3], 1):
            print(f"  {i}. [{err['difficulty']}/{err['type']}] {err['query']}... (score={err['score']:.2f})")

if __name__ == "__main__":
    main()
