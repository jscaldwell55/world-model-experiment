#!/usr/bin/env python3
"""
Analyze ACE baseline experiment results
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np

def load_episode(filepath):
    """Load a single episode JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_results():
    """Analyze all episodes in the ACE baseline experiment"""
    results_dir = Path("results/ace_baseline/raw")
    episode_files = sorted(results_dir.glob("*.json"))

    # Storage for aggregate statistics
    episodes_by_env = defaultdict(list)
    all_episodes = []

    # Per-environment stats
    env_stats = {}

    for filepath in episode_files:
        episode = load_episode(filepath)
        env_name = episode['environment']
        episodes_by_env[env_name].append(episode)
        all_episodes.append(episode)

    # Compute overall statistics
    total_cost = sum(ep['cost']['total_cost_usd'] for ep in all_episodes)
    total_input_tokens = sum(ep['total_input_tokens'] for ep in all_episodes)
    total_output_tokens = sum(ep['total_output_tokens'] for ep in all_episodes)
    total_duration = sum(ep['duration_seconds'] for ep in all_episodes)

    # Compute test performance
    all_scores = []
    scores_by_type = defaultdict(list)
    scores_by_difficulty = defaultdict(list)

    for episode in all_episodes:
        for test in episode['test_results']:
            all_scores.append(test['score'])
            scores_by_type[test['query_type']].append(test['score'])
            scores_by_difficulty[test['difficulty']].append(test['score'])

    # Print summary
    print("=" * 70)
    print("ACE BASELINE EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"\nTotal Episodes: {len(all_episodes)}")
    print(f"  - HotPot: {len(episodes_by_env.get('HotPotLab', []))}")
    print(f"  - SwitchLight: {len(episodes_by_env.get('SwitchLight', []))}")
    print(f"  - ChemTile: {len(episodes_by_env.get('ChemTile', []))}")

    print(f"\n{'COST ANALYSIS':-^70}")
    print(f"Total Cost: ${total_cost:.4f}")
    print(f"Cost per Episode: ${total_cost/len(all_episodes):.4f}")

    print(f"\n{'TOKEN USAGE':-^70}")
    print(f"Total Input Tokens: {total_input_tokens:,}")
    print(f"Total Output Tokens: {total_output_tokens:,}")
    print(f"Total Tokens: {total_input_tokens + total_output_tokens:,}")
    print(f"Avg Input per Episode: {total_input_tokens/len(all_episodes):,.0f}")
    print(f"Avg Output per Episode: {total_output_tokens/len(all_episodes):,.0f}")

    print(f"\n{'PERFORMANCE METRICS':-^70}")
    print(f"Overall Score: {np.mean(all_scores):.3f} ± {np.std(all_scores):.3f}")
    print(f"Median Score: {np.median(all_scores):.3f}")

    print(f"\nBy Query Type:")
    for qtype in sorted(scores_by_type.keys()):
        scores = scores_by_type[qtype]
        print(f"  {qtype:20s}: {np.mean(scores):.3f} ± {np.std(scores):.3f} (n={len(scores)})")

    print(f"\nBy Difficulty:")
    for diff in ['easy', 'medium', 'hard']:
        if diff in scores_by_difficulty:
            scores = scores_by_difficulty[diff]
            print(f"  {diff:20s}: {np.mean(scores):.3f} ± {np.std(scores):.3f} (n={len(scores)})")

    print(f"\n{'TIME ANALYSIS':-^70}")
    print(f"Total Duration: {total_duration/60:.1f} minutes")
    print(f"Avg per Episode: {total_duration/len(all_episodes):.1f} seconds")

    # Token breakdown analysis
    print(f"\n{'TOKEN BREAKDOWN BY CATEGORY':-^70}")
    total_breakdown = defaultdict(lambda: {'input': 0, 'output': 0})

    for episode in all_episodes:
        breakdown = episode['token_breakdown']['breakdown']
        for category, tokens in breakdown.items():
            if category != 'totals':
                total_breakdown[category]['input'] += tokens['input']
                total_breakdown[category]['output'] += tokens['output']

    for category in sorted(total_breakdown.keys()):
        inp = total_breakdown[category]['input']
        out = total_breakdown[category]['output']
        total = inp + out
        print(f"{category:15s}: {total:7,} tokens ({inp:6,} in, {out:6,} out)")

    # Environment-specific analysis
    print(f"\n{'PERFORMANCE BY ENVIRONMENT':-^70}")
    for env_name in sorted(episodes_by_env.keys()):
        env_episodes = episodes_by_env[env_name]
        env_scores = []
        for ep in env_episodes:
            env_scores.extend([t['score'] for t in ep['test_results']])

        env_cost = sum(ep['cost']['total_cost_usd'] for ep in env_episodes)
        env_tokens = sum(ep['total_input_tokens'] + ep['total_output_tokens'] for ep in env_episodes)

        print(f"\n{env_name}:")
        print(f"  Score: {np.mean(env_scores):.3f} ± {np.std(env_scores):.3f}")
        print(f"  Cost: ${env_cost:.4f} (${env_cost/len(env_episodes):.4f}/ep)")
        print(f"  Tokens: {env_tokens:,} ({env_tokens//len(env_episodes):,}/ep)")

    # Playbook analysis
    print(f"\n{'PLAYBOOK STATISTICS':-^70}")
    playbook_sizes = [ep['playbook']['total_bullets'] for ep in all_episodes]
    print(f"Avg Playbook Size: {np.mean(playbook_sizes):.1f} bullets")
    print(f"Range: {min(playbook_sizes)} - {max(playbook_sizes)} bullets")

    # Sample playbook content from one episode
    print(f"\n{'SAMPLE PLAYBOOK CONTENT (from HotPot ep001)':-^70}")
    hotpot_ep = [ep for ep in all_episodes if ep['episode_id'] == 'hot_pot_a_c_e_ep001'][0]
    playbook = hotpot_ep['playbook']['final_playbook']

    print("\nStrategies (showing first 3):")
    for i, item in enumerate(playbook['strategies_and_hard_rules'][:3]):
        print(f"  {i+1}. {item['content']}")

    print("\nCode Snippets (showing first 2):")
    for i, item in enumerate(playbook['useful_code_snippets'][:2]):
        print(f"  {i+1}. {item['content']}")

    return {
        'total_episodes': len(all_episodes),
        'total_cost': total_cost,
        'mean_score': np.mean(all_scores),
        'total_tokens': total_input_tokens + total_output_tokens,
        'scores_by_type': {k: np.mean(v) for k, v in scores_by_type.items()},
        'scores_by_difficulty': {k: np.mean(v) for k, v in scores_by_difficulty.items()}
    }

if __name__ == "__main__":
    stats = analyze_results()
