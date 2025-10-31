#!/usr/bin/env python3
"""
Generate summary tables for pilot results
"""
import json
import glob
from collections import defaultdict
import statistics

def load_all_results(results_dir):
    files = glob.glob(f"{results_dir}/raw/*.json")
    results = []
    for f in files:
        with open(f, 'r') as file:
            results.append(json.load(file))
    return results

def print_markdown_table(headers, rows, title=None):
    """Print a markdown formatted table"""
    if title:
        print(f"\n### {title}\n")

    # Calculate column widths
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]

    # Print header
    print("| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + " |")
    print("| " + " | ".join("-" * w for w in col_widths) + " |")

    # Print rows
    for row in rows:
        print("| " + " | ".join(f"{str(cell):<{w}}" for cell, w in zip(row, col_widths)) + " |")

def main():
    results_dir = "results/pilot_h1h5_fixed"
    results = load_all_results(results_dir)

    print("# PILOT H1-H5 SUMMARY TABLES")
    print("=" * 80)

    # Table 1: Overall Performance
    perf = defaultdict(list)
    conf = defaultdict(list)
    for r in results:
        agent = r['agent_type']
        for t in r.get('test_results', []):
            perf[agent].append(t['score'])
            conf[agent].append(t['confidence'])

    rows = []
    for agent in sorted(perf.keys()):
        rows.append([
            agent,
            f"{statistics.mean(perf[agent]):.3f}",
            f"{statistics.stdev(perf[agent]):.3f}",
            f"{statistics.mean(conf[agent]):.3f}",
            f"{statistics.mean(conf[agent]) - statistics.mean(perf[agent]):.3f}",
            len(perf[agent])
        ])

    print_markdown_table(
        ["Agent", "Accuracy", "Std Dev", "Confidence", "Cal Gap", "N"],
        rows,
        "Overall Performance Summary"
    )

    # Table 2: Environment Breakdown
    env_perf = defaultdict(lambda: defaultdict(list))
    for r in results:
        agent = r['agent_type']
        env = r['environment']
        for t in r.get('test_results', []):
            env_perf[agent][env].append(t['score'])

    rows = []
    for agent in sorted(env_perf.keys()):
        for env in ['HotPotLab', 'SwitchLight']:
            if env in env_perf[agent]:
                scores = env_perf[agent][env]
                rows.append([
                    agent,
                    env,
                    f"{statistics.mean(scores):.3f}",
                    f"{statistics.stdev(scores):.3f}",
                    len(scores)
                ])

    print_markdown_table(
        ["Agent", "Environment", "Accuracy", "Std Dev", "N"],
        rows,
        "Performance by Environment"
    )

    # Table 3: Query Type Breakdown
    query_perf = defaultdict(lambda: defaultdict(list))
    for r in results:
        agent = r['agent_type']
        for t in r.get('test_results', []):
            query_perf[agent][t['query_type']].append(t['score'])

    rows = []
    for agent in sorted(query_perf.keys()):
        for qtype in sorted(query_perf[agent].keys()):
            scores = query_perf[agent][qtype]
            rows.append([
                agent,
                qtype,
                f"{statistics.mean(scores):.3f}",
                len(scores)
            ])

    print_markdown_table(
        ["Agent", "Query Type", "Accuracy", "N"],
        rows,
        "Performance by Query Type"
    )

    # Table 4: Difficulty Breakdown
    diff_perf = defaultdict(lambda: defaultdict(list))
    for r in results:
        agent = r['agent_type']
        for t in r.get('test_results', []):
            diff_perf[agent][t['difficulty']].append(t['score'])

    rows = []
    for agent in sorted(diff_perf.keys()):
        for diff in ['easy', 'medium', 'hard']:
            if diff in diff_perf[agent]:
                scores = diff_perf[agent][diff]
                rows.append([
                    agent,
                    diff,
                    f"{statistics.mean(scores):.3f}",
                    len(scores)
                ])

    print_markdown_table(
        ["Agent", "Difficulty", "Accuracy", "N"],
        rows,
        "Performance by Difficulty"
    )

    # Table 5: Resource Usage
    token_stats = defaultdict(lambda: defaultdict(lambda: {'input': [], 'output': [], 'calls': [], 'time': []}))
    for r in results:
        agent = r['agent_type']
        env = r['environment']
        token_stats[agent][env]['input'].append(r.get('total_input_tokens', 0))
        token_stats[agent][env]['output'].append(r.get('total_output_tokens', 0))
        token_stats[agent][env]['calls'].append(r.get('total_api_calls', 0))
        token_stats[agent][env]['time'].append(r.get('duration_seconds', 0))

    rows = []
    for agent in sorted(token_stats.keys()):
        for env in ['HotPotLab', 'SwitchLight']:
            if env in token_stats[agent]:
                stats = token_stats[agent][env]
                avg_input = statistics.mean(stats['input'])
                avg_output = statistics.mean(stats['output'])
                avg_calls = statistics.mean(stats['calls'])
                avg_time = statistics.mean(stats['time'])
                rows.append([
                    agent,
                    env,
                    f"{int(avg_input):,}",
                    f"{int(avg_output):,}",
                    f"{int(avg_input + avg_output):,}",
                    f"{avg_calls:.0f}",
                    f"{avg_time:.0f}s"
                ])

    print_markdown_table(
        ["Agent", "Environment", "Input Tok", "Output Tok", "Total Tok", "API Calls", "Time"],
        rows,
        "Resource Usage"
    )

    # Table 6: Cost-Effectiveness
    print("\n### Cost-Effectiveness Analysis\n")
    print("*(Tokens per percentage point of accuracy)*\n")

    rows = []
    for agent in sorted(perf.keys()):
        total_tokens = 0
        total_accuracy = 0
        n_episodes = 0

        for r in results:
            if r['agent_type'] == agent:
                total_tokens += r.get('total_input_tokens', 0) + r.get('total_output_tokens', 0)
                scores = [t['score'] for t in r.get('test_results', [])]
                total_accuracy += statistics.mean(scores) if scores else 0
                n_episodes += 1

        avg_tokens = total_tokens / n_episodes
        avg_accuracy = total_accuracy / n_episodes
        tokens_per_pct = avg_tokens / (avg_accuracy * 100) if avg_accuracy > 0 else 0

        rows.append([
            agent,
            f"{avg_tokens:.0f}",
            f"{avg_accuracy:.3f}",
            f"{tokens_per_pct:.0f}"
        ])

    print_markdown_table(
        ["Agent", "Avg Tokens/Ep", "Avg Accuracy", "Tokens/% Acc"],
        rows
    )

    # Table 7: SwitchLight Belief Convergence
    belief_data = defaultdict(list)
    for r in results:
        if r['environment'] != 'SwitchLight':
            continue
        agent = r['agent_type']
        steps = r.get('steps', [])
        if not steps:
            continue

        # Get final belief
        final_step = steps[-1]
        belief = final_step.get('belief_state', {})
        if 'wiring_probs' not in belief:
            continue

        gt = r.get('ground_truth', {})
        true_layout = gt.get('wire_layout', 'unknown')
        faulty = gt.get('faulty_relay', False)

        if true_layout == 'layout_A':
            correct_prob = belief['wiring_probs'].get('layout_A', 0)
        else:
            correct_prob = belief['wiring_probs'].get('layout_B', 0)

        belief_data[agent].append({
            'ep_id': r['episode_id'],
            'correct_prob': correct_prob,
            'faulty': faulty,
            'true_layout': true_layout
        })

    print("\n### SwitchLight Belief Convergence (Active Agents Only)\n")

    for agent in sorted(belief_data.keys()):
        print(f"\n**{agent.upper()} Agent:**\n")
        rows = []
        for ep in belief_data[agent]:
            converged = "✓" if ep['correct_prob'] >= 0.8 else "✗"
            rows.append([
                ep['ep_id'].split('_')[-1],  # Just episode number
                ep['true_layout'],
                "Yes" if ep['faulty'] else "No",
                f"{ep['correct_prob']:.2f}",
                converged
            ])

        print_markdown_table(
            ["Episode", "True Layout", "Faulty", "P(Correct)", "Converged?"],
            rows
        )

        # Summary
        converged_count = sum(1 for ep in belief_data[agent] if ep['correct_prob'] >= 0.8)
        print(f"\n*Convergence rate: {converged_count}/{len(belief_data[agent])} ({converged_count/len(belief_data[agent])*100:.0f}%)*")

if __name__ == "__main__":
    main()
