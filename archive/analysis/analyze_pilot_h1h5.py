#!/usr/bin/env python3
"""
Analysis script for H1-H5 pilot experiment results.

Computes core metrics and provides go/no-go recommendation.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

# Anthropic Claude Sonnet 4.5 pricing
COST_INPUT = 3.0 / 1_000_000  # $3 per 1M input tokens
COST_OUTPUT = 15.0 / 1_000_000  # $15 per 1M output tokens


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all episode results from directory."""
    episodes = []

    raw_dir = results_dir / 'raw'
    if not raw_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {raw_dir}")

    for episode_file in raw_dir.glob('*.json'):
        try:
            with open(episode_file) as f:
                ep = json.load(f)
                episodes.append(ep)
        except Exception as e:
            print(f"Warning: Failed to load {episode_file}: {e}")

    if not episodes:
        raise ValueError(f"No episode results found in {raw_dir}")

    print(f"Loaded {len(episodes)} episodes")

    # Extract data
    data = []
    for ep in episodes:
        agent = ep.get('agent_type', 'unknown')
        env = ep.get('environment', 'unknown')

        # Interventional accuracy
        test_results = ep.get('test_results', [])
        interventional_queries = [q for q in test_results
                                  if q.get('query_type') == 'interventional']
        if interventional_queries:
            accuracy = np.mean([q.get('correct', False) for q in interventional_queries])
        else:
            accuracy = np.nan

        # Final surprisal
        steps = ep.get('steps', [])
        if steps:
            final_surprisal = steps[-1].get('surprisal', np.nan)
        else:
            final_surprisal = np.nan

        # Token usage
        total_input = ep.get('total_input_tokens', 0)
        total_output = ep.get('total_output_tokens', 0)

        data.append({
            'agent': agent,
            'environment': env,
            'episode_id': ep.get('episode_id', 'unknown'),
            'interventional_accuracy': accuracy,
            'final_surprisal': final_surprisal,
            'episode_length': len(steps),
            'input_tokens': total_input,
            'output_tokens': total_output,
            'steps': steps
        })

    return pd.DataFrame(data)


def test_h1(df: pd.DataFrame) -> dict:
    """H1: Actor > Observer in interventional accuracy."""
    observer_acc = df[df['agent'] == 'observer']['interventional_accuracy'].dropna()
    actor_acc = df[df['agent'] == 'actor']['interventional_accuracy'].dropna()

    if len(observer_acc) == 0 or len(actor_acc) == 0:
        return {'passed': False, 'reason': 'Insufficient data'}

    diff = actor_acc.mean() - observer_acc.mean()
    pooled_std = np.sqrt((observer_acc.std()**2 + actor_acc.std()**2) / 2)
    cohens_d = diff / pooled_std if pooled_std > 0 else 0

    t_stat, p_value = ttest_ind(actor_acc, observer_acc)

    # Relaxed threshold for pilot (10% instead of 15%)
    passed = diff > 0.10

    return {
        'passed': passed,
        'observer_mean': observer_acc.mean(),
        'actor_mean': actor_acc.mean(),
        'difference': diff,
        'cohens_d': cohens_d,
        't_stat': t_stat,
        'p_value': p_value
    }


def test_h2(df: pd.DataFrame) -> dict:
    """H2: Actor shows learning (negative surprisal slope)."""
    actor_episodes = df[df['agent'] == 'actor']

    slopes = []
    for _, ep in actor_episodes.iterrows():
        steps = ep['steps']
        if not steps:
            continue

        step_nums = [s['step_num'] for s in steps if 'surprisal' in s]
        surprisals = [s['surprisal'] for s in steps if 'surprisal' in s]

        if len(step_nums) > 2:
            slope = np.polyfit(step_nums, surprisals, 1)[0]
            slopes.append(slope)

    if not slopes:
        return {'passed': False, 'reason': 'No surprisal data'}

    avg_slope = np.mean(slopes)
    passed = avg_slope < -0.05  # Relaxed threshold

    return {
        'passed': passed,
        'avg_slope': avg_slope,
        'slope_std': np.std(slopes),
        'num_episodes': len(slopes)
    }


def test_h3(df: pd.DataFrame) -> dict:
    """H3: Model-Based > Actor."""
    actor_acc = df[df['agent'] == 'actor']['interventional_accuracy'].dropna()
    model_acc = df[df['agent'] == 'model_based']['interventional_accuracy'].dropna()

    if len(actor_acc) == 0 or len(model_acc) == 0:
        return {'passed': False, 'reason': 'Insufficient data'}

    diff = model_acc.mean() - actor_acc.mean()
    pooled_std = np.sqrt((actor_acc.std()**2 + model_acc.std()**2) / 2)
    cohens_d = diff / pooled_std if pooled_std > 0 else 0

    t_stat, p_value = ttest_ind(model_acc, actor_acc)

    passed = diff > 0.05  # Relaxed threshold for pilot

    return {
        'passed': passed,
        'actor_mean': actor_acc.mean(),
        'model_based_mean': model_acc.mean(),
        'difference': diff,
        'cohens_d': cohens_d,
        't_stat': t_stat,
        'p_value': p_value
    }


def generate_visualizations(df: pd.DataFrame, output_dir: Path):
    """Generate comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Interventional Accuracy by Agent
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    agents = ['observer', 'actor', 'model_based']
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']

    agent_data = []
    for agent in agents:
        acc = df[df['agent'] == agent]['interventional_accuracy'].dropna()
        if len(acc) > 0:
            agent_data.append(acc.values)
        else:
            agent_data.append([])

    positions = range(len(agents))
    bp = ax1.boxplot(agent_data, positions=positions, labels=agents,
                     patch_artist=True, widths=0.6)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_ylabel('Interventional Accuracy', fontsize=12)
    ax1.set_title('Agent Comparison: Interventional Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Surprisal Trajectories (Actor only)
    ax2 = axes[1]
    actor_episodes = df[df['agent'] == 'actor']

    all_trajectories = []
    for _, ep in actor_episodes.iterrows():
        steps = ep['steps']
        step_nums = [s['step_num'] for s in steps if 'surprisal' in s]
        surprisals = [s['surprisal'] for s in steps if 'surprisal' in s]

        if len(step_nums) > 0:
            ax2.plot(step_nums, surprisals, alpha=0.3, color='gray', linewidth=1)
            all_trajectories.append((step_nums, surprisals))

    # Average trajectory
    if all_trajectories:
        max_steps = max(max(nums) for nums, _ in all_trajectories)
        avg_surprisal = []
        for step in range(max_steps + 1):
            step_values = [surp[nums.index(step)] for nums, surp in all_trajectories
                          if step in nums]
            if step_values:
                avg_surprisal.append(np.mean(step_values))
            else:
                avg_surprisal.append(np.nan)

        ax2.plot(range(len(avg_surprisal)), avg_surprisal,
                color='red', linewidth=3, label='Average', marker='o')

    ax2.set_xlabel('Step Number', fontsize=12)
    ax2.set_ylabel('Belief Surprisal', fontsize=12)
    ax2.set_title('Actor Agent: Learning Trajectories', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'pilot_h1h5_results.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'pilot_h1h5_results.png'}")
    plt.close()


def main():
    """Run analysis and provide go/no-go recommendation."""
    results_dir = Path('results/pilot_h1h5')

    print("=" * 70)
    print("H1-H5 PILOT ANALYSIS")
    print("=" * 70)
    print()

    # Load data
    df = load_results(results_dir)

    # Summary statistics
    print("EPISODE SUMMARY")
    print("-" * 70)
    print(f"Total episodes: {len(df)}")
    print(f"By agent:")
    for agent in df['agent'].unique():
        count = len(df[df['agent'] == agent])
        print(f"  {agent}: {count}")
    print(f"By environment:")
    for env in df['environment'].unique():
        count = len(df[df['environment'] == env])
        print(f"  {env}: {count}")
    print()

    # Test hypotheses
    print("HYPOTHESIS TESTS")
    print("-" * 70)

    h1_results = test_h1(df)
    print("\nH1: Actor > Observer (Interventional Accuracy)")
    print(f"  Observer: {h1_results.get('observer_mean', np.nan):.3f}")
    print(f"  Actor: {h1_results.get('actor_mean', np.nan):.3f}")
    print(f"  Difference: {h1_results.get('difference', np.nan)*100:.1f}%")
    print(f"  Cohen's d: {h1_results.get('cohens_d', np.nan):.3f}")
    print(f"  t-test: t={h1_results.get('t_stat', np.nan):.3f}, p={h1_results.get('p_value', np.nan):.3f}")
    print(f"  Result: {'✅ PASS' if h1_results['passed'] else '❌ FAIL'}")

    h2_results = test_h2(df)
    print("\nH2: Actor Shows Learning (Negative Surprisal Slope)")
    print(f"  Average slope: {h2_results.get('avg_slope', np.nan):.4f}")
    print(f"  Std: {h2_results.get('slope_std', np.nan):.4f}")
    print(f"  Episodes: {h2_results.get('num_episodes', 0)}")
    print(f"  Result: {'✅ PASS' if h2_results['passed'] else '❌ FAIL'}")

    h3_results = test_h3(df)
    print("\nH3: Model-Based > Actor")
    print(f"  Actor: {h3_results.get('actor_mean', np.nan):.3f}")
    print(f"  Model-Based: {h3_results.get('model_based_mean', np.nan):.3f}")
    print(f"  Difference: {h3_results.get('difference', np.nan)*100:.1f}%")
    print(f"  Cohen's d: {h3_results.get('cohens_d', np.nan):.3f}")
    print(f"  t-test: t={h3_results.get('t_stat', np.nan):.3f}, p={h3_results.get('p_value', np.nan):.3f}")
    print(f"  Result: {'✅ PASS' if h3_results['passed'] else '❌ FAIL'}")
    print()

    # Cost analysis
    total_input = df['input_tokens'].sum()
    total_output = df['output_tokens'].sum()
    total_cost = (total_input * COST_INPUT) + (total_output * COST_OUTPUT)
    cost_per_episode = total_cost / len(df)
    projected_600 = cost_per_episode * 600

    print("COST ANALYSIS")
    print("-" * 70)
    print(f"Total input tokens: {total_input:,}")
    print(f"Total output tokens: {total_output:,}")
    print(f"Pilot cost: ${total_cost:.2f} for {len(df)} episodes")
    print(f"Cost per episode: ${cost_per_episode:.2f}")
    print(f"Projected 600 episodes: ${projected_600:.2f}")
    print()

    # Generate visualizations
    generate_visualizations(df, results_dir)

    # Go/No-Go recommendation
    passed = sum([h1_results['passed'], h2_results['passed'], h3_results['passed']])

    print("=" * 70)
    print("GO/NO-GO RECOMMENDATION")
    print("=" * 70)
    print(f"\nHypotheses passed: {passed}/3")
    print(f"Projected cost: ${projected_600:.2f}")
    print()

    if passed >= 2 and projected_600 <= 200:
        print("✅ RECOMMENDATION: GO")
        print("   • Infrastructure working properly")
        print(f"   • {passed}/3 hypotheses show expected patterns")
        print(f"   • Cost is reasonable (${projected_600:.2f})")
        print("   • Proceed with full 600-episode experiment")
    elif passed >= 2:
        print("⚠️  RECOMMENDATION: GO (with cost concerns)")
        print(f"   • {passed}/3 hypotheses show expected patterns")
        print(f"   • But cost is high (${projected_600:.2f})")
        print("   • Consider reducing scale to 400 episodes")
    elif passed == 1:
        print("⚠️  RECOMMENDATION: INVESTIGATE")
        print(f"   • Only {passed}/3 hypotheses passed")
        print("   • Debug failing hypotheses before scaling")
        print("   • May need to adjust agent implementations")
    else:
        print("❌ RECOMMENDATION: NO-GO")
        print("   • Infrastructure may have issues")
        print("   • Debug before investing in full experiment")
    print("=" * 70)


if __name__ == '__main__':
    main()
