#!/usr/bin/env python3
"""
Visualization script for pilot_h1h5_fixed results
"""
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import statistics

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_all_results(results_dir):
    """Load all JSON result files"""
    files = glob.glob(f"{results_dir}/raw/*.json")
    results = []
    for f in files:
        with open(f, 'r') as file:
            results.append(json.load(file))
    return results

def plot_overall_performance(results, output_path):
    """Plot overall performance comparison"""
    performance = defaultdict(list)

    for result in results:
        agent = result['agent_type']
        for test in result.get('test_results', []):
            performance[agent].append(test['score'])

    agents = sorted(performance.keys())
    means = [statistics.mean(performance[a]) for a in agents]
    stds = [statistics.stdev(performance[a]) for a in agents]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(agents))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

    ax.set_xlabel('Agent Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Score', fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance by Agent Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.03, f'{m:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_performance_by_environment(results, output_path):
    """Plot performance by environment"""
    performance = defaultdict(lambda: defaultdict(list))

    for result in results:
        agent = result['agent_type']
        env = result['environment']
        for test in result.get('test_results', []):
            performance[agent][env].append(test['score'])

    agents = sorted(performance.keys())
    envs = ['HotPotLab', 'SwitchLight']

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(agents))
    width = 0.35

    means_env1 = [statistics.mean(performance[a][envs[0]]) if envs[0] in performance[a] else 0 for a in agents]
    means_env2 = [statistics.mean(performance[a][envs[1]]) if envs[1] in performance[a] else 0 for a in agents]

    bars1 = ax.bar(x - width/2, means_env1, width, label=envs[0], alpha=0.8, color='#1f77b4')
    bars2 = ax.bar(x + width/2, means_env2, width, label=envs[1], alpha=0.8, color='#ff7f0e')

    ax.set_xlabel('Agent Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance by Environment and Agent Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, m in enumerate(means_env1):
        ax.text(i - width/2, m + 0.02, f'{m:.2f}', ha='center', fontsize=9)
    for i, m in enumerate(means_env2):
        ax.text(i + width/2, m + 0.02, f'{m:.2f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_performance_by_query_type(results, output_path):
    """Plot performance by query type"""
    query_perf = defaultdict(lambda: defaultdict(list))

    for result in results:
        agent = result['agent_type']
        for test in result.get('test_results', []):
            query_perf[agent][test['query_type']].append(test['score'])

    agents = sorted(query_perf.keys())
    query_types = sorted(set(qt for agent_data in query_perf.values() for qt in agent_data.keys()))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(query_types))
    width = 0.2

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, agent in enumerate(agents):
        means = [statistics.mean(query_perf[agent][qt]) if qt in query_perf[agent] else 0 for qt in query_types]
        offset = (i - len(agents)/2 + 0.5) * width
        ax.bar(x + offset, means, width, label=agent, alpha=0.8, color=colors[i])

    ax.set_xlabel('Query Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance by Query Type and Agent', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(query_types)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_token_usage(results, output_path):
    """Plot token usage comparison"""
    token_stats = defaultdict(lambda: defaultdict(lambda: {'input': [], 'output': []}))

    for result in results:
        agent = result['agent_type']
        env = result['environment']
        token_stats[agent][env]['input'].append(result.get('total_input_tokens', 0))
        token_stats[agent][env]['output'].append(result.get('total_output_tokens', 0))

    agents = sorted(token_stats.keys())
    envs = ['HotPotLab', 'SwitchLight']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for env_idx, env in enumerate(envs):
        ax = axes[env_idx]
        x = np.arange(len(agents))

        input_means = [statistics.mean(token_stats[a][env]['input']) if env in token_stats[a] else 0 for a in agents]
        output_means = [statistics.mean(token_stats[a][env]['output']) if env in token_stats[a] else 0 for a in agents]

        ax.bar(x, input_means, label='Input Tokens', alpha=0.8, color='#1f77b4')
        ax.bar(x, output_means, bottom=input_means, label='Output Tokens', alpha=0.8, color='#ff7f0e')

        ax.set_xlabel('Agent Type', fontsize=11, fontweight='bold')
        ax.set_ylabel('Average Tokens per Episode', fontsize=11, fontweight='bold')
        ax.set_title(f'{env} Token Usage', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=15, ha='right')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Add total labels
        totals = [i + o for i, o in zip(input_means, output_means)]
        for i, t in enumerate(totals):
            ax.text(i, t + 500, f'{int(t):,}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_difficulty_analysis(results, output_path):
    """Plot performance by difficulty level"""
    difficulty_perf = defaultdict(lambda: defaultdict(list))

    for result in results:
        agent = result['agent_type']
        for test in result.get('test_results', []):
            difficulty_perf[agent][test['difficulty']].append(test['score'])

    agents = sorted(difficulty_perf.keys())
    difficulties = ['easy', 'medium', 'hard']

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(difficulties))
    width = 0.2

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, agent in enumerate(agents):
        means = [statistics.mean(difficulty_perf[agent][d]) if d in difficulty_perf[agent] else 0 for d in difficulties]
        offset = (i - len(agents)/2 + 0.5) * width
        ax.bar(x + offset, means, width, label=agent, alpha=0.8, color=colors[i])

    ax.set_xlabel('Difficulty Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance by Difficulty Level and Agent', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in difficulties])
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    # Add horizontal line at 0.5 for reference
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_confidence_calibration(results, output_path):
    """Plot confidence vs accuracy"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    agents = sorted(set(r['agent_type'] for r in results))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, agent in enumerate(agents):
        ax = axes[idx // 2, idx % 2]

        confidences = []
        scores = []

        for result in results:
            if result['agent_type'] != agent:
                continue
            for test in result.get('test_results', []):
                confidences.append(test['confidence'])
                scores.append(test['score'])

        # Scatter plot
        ax.scatter(confidences, scores, alpha=0.3, s=20, color=colors[idx])

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=2, label='Perfect Calibration')

        # Bin confidences and compute mean accuracy
        bins = np.linspace(0, 1, 11)
        bin_means_conf = []
        bin_means_acc = []
        for i in range(len(bins) - 1):
            mask = (np.array(confidences) >= bins[i]) & (np.array(confidences) < bins[i+1])
            if mask.sum() > 0:
                bin_means_conf.append((bins[i] + bins[i+1]) / 2)
                bin_means_acc.append(np.mean(np.array(scores)[mask]))

        ax.plot(bin_means_conf, bin_means_acc, 'b-o', linewidth=2, markersize=8,
                label='Actual Calibration', alpha=0.8)

        ax.set_xlabel('Confidence', fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{agent.upper()} Agent Calibration', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_belief_trajectories(results, output_path):
    """Plot belief state evolution for SwitchLight"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    agents = ['actor', 'model_based']

    for agent_idx, agent in enumerate(agents):
        # Get all episodes for this agent
        agent_results = [r for r in results if r['agent_type'] == agent and r['environment'] == 'SwitchLight']

        for ep_idx, result in enumerate(agent_results[:2]):  # Plot first 2 episodes
            ax = axes[agent_idx, ep_idx]

            steps = result.get('steps', [])
            step_nums = []
            layout_A_probs = []
            failure_probs = []

            for step in steps:
                belief = step.get('belief_state', {})
                if 'wiring_probs' in belief:
                    step_nums.append(step['step_num'])
                    layout_A_probs.append(belief['wiring_probs'].get('layout_A', 0))
                    failure_probs.append(belief.get('failure_prob', 0))

            if step_nums:
                ax.plot(step_nums, layout_A_probs, 'b-o', label='P(layout_A)', linewidth=2, markersize=6)
                ax.plot(step_nums, failure_probs, 'r-s', label='P(failure)', linewidth=2, markersize=6)

                # Mark ground truth
                gt = result.get('ground_truth', {})
                true_layout = gt.get('wire_layout', 'unknown')
                faulty = gt.get('faulty_relay', False)

                if true_layout == 'layout_A':
                    ax.axhline(y=1.0, color='b', linestyle='--', alpha=0.3, linewidth=1.5)
                    ax.text(max(step_nums) * 0.8, 0.95, 'Truth: layout_A', fontsize=9, color='b')
                else:
                    ax.axhline(y=0.0, color='b', linestyle='--', alpha=0.3, linewidth=1.5)
                    ax.text(max(step_nums) * 0.8, 0.05, 'Truth: layout_B', fontsize=9, color='b')

                if faulty:
                    ax.text(max(step_nums) * 0.8, 0.85, 'Faulty: YES', fontsize=9, color='r')

            ax.set_xlabel('Step Number', fontsize=10, fontweight='bold')
            ax.set_ylabel('Probability', fontsize=10, fontweight='bold')
            ax.set_title(f'{agent.upper()}: {result["episode_id"]}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    results_dir = "results/pilot_h1h5_fixed"
    output_dir = "results/pilot_h1h5_fixed/figures"

    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("Loading results...")
    results = load_all_results(results_dir)
    print(f"Loaded {len(results)} episodes")

    print("\nGenerating visualizations...")

    plot_overall_performance(results, f"{output_dir}/1_overall_performance.png")
    plot_performance_by_environment(results, f"{output_dir}/2_performance_by_environment.png")
    plot_performance_by_query_type(results, f"{output_dir}/3_performance_by_query_type.png")
    plot_difficulty_analysis(results, f"{output_dir}/4_performance_by_difficulty.png")
    plot_token_usage(results, f"{output_dir}/5_token_usage.png")
    plot_confidence_calibration(results, f"{output_dir}/6_confidence_calibration.png")
    plot_belief_trajectories(results, f"{output_dir}/7_belief_trajectories.png")

    print(f"\nAll visualizations saved to {output_dir}/")

if __name__ == "__main__":
    main()
