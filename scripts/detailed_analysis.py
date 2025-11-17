#!/usr/bin/env python3
"""Detailed analysis of ACE baseline results"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    args = parser.parse_args()

    results_dir = Path(args.results)

    # Load all episodes
    episodes = []
    for episode_file in sorted(results_dir.glob('*.json')):
        if 'config' in episode_file.name:
            continue
        with open(episode_file) as f:
            episodes.append(json.load(f))

    print(f"Loaded {len(episodes)} episodes\n")

    # =========================================================================
    # QUESTION TYPE BREAKDOWN
    # =========================================================================
    print("="*70)
    print("QUESTION TYPE BREAKDOWN")
    print("="*70)

    # Collect all test results by environment and type
    breakdown_data = []
    for ep in episodes:
        env = ep.get('environment', 'unknown')
        for result in ep.get('test_results', []):
            breakdown_data.append({
                'environment': env,
                'query_type': result.get('query_type', 'unknown'),
                'score': result.get('score', result.get('correct', 0.0)),
                'correct': result.get('correct', False)
            })

    df_breakdown = pd.DataFrame(breakdown_data)

    # Summary by environment and query type
    for env in sorted(df_breakdown['environment'].unique()):
        env_data = df_breakdown[df_breakdown['environment'] == env]
        print(f"\n{env}:")

        total_questions = len(env_data)
        print(f"  Total questions: {total_questions}")

        for qtype in ['planning', 'interventional', 'counterfactual']:
            type_data = env_data[env_data['query_type'] == qtype]
            count = len(type_data)
            percentage = (count / total_questions * 100) if total_questions > 0 else 0
            avg_score = type_data['score'].mean() if len(type_data) > 0 else 0

            print(f"  {qtype.capitalize():15} questions: {count:2d} ({percentage:5.1f}%)  Avg score: {avg_score:.3f}")

    # Overall breakdown
    print(f"\nOverall:")
    print(f"  Total questions: {len(df_breakdown)}")
    for qtype in ['planning', 'interventional', 'counterfactual']:
        type_data = df_breakdown[df_breakdown['query_type'] == qtype]
        count = len(type_data)
        percentage = (count / len(df_breakdown) * 100) if len(df_breakdown) > 0 else 0
        avg_score = type_data['score'].mean() if len(type_data) > 0 else 0
        print(f"  {qtype.capitalize():15} questions: {count:2d} ({percentage:5.1f}%)  Avg score: {avg_score:.3f}")

    # =========================================================================
    # EPISODE-LEVEL DATA
    # =========================================================================
    print("\n" + "="*70)
    print("EPISODE-LEVEL DATA")
    print("="*70)

    episode_data = []
    for ep in episodes:
        ep_id = ep['episode_id']
        env = ep.get('environment', 'unknown')
        seed = ep.get('seed', 0)

        # Compute question type accuracies
        test_results = ep.get('test_results', [])

        planning_results = [r for r in test_results if r.get('query_type') == 'planning']
        interventional_results = [r for r in test_results if r.get('query_type') == 'interventional']
        counterfactual_results = [r for r in test_results if r.get('query_type') == 'counterfactual']

        planning_acc = np.mean([r.get('score', r.get('correct', 0.0)) for r in planning_results]) if planning_results else 0.0
        interventional_acc = np.mean([r.get('score', r.get('correct', 0.0)) for r in interventional_results]) if interventional_results else 0.0
        counterfactual_acc = np.mean([r.get('score', r.get('correct', 0.0)) for r in counterfactual_results]) if counterfactual_results else 0.0
        overall_acc = np.mean([r.get('score', r.get('correct', 0.0)) for r in test_results]) if test_results else 0.0

        # Cost info
        cost_info = ep.get('cost', {})
        total_cost = cost_info.get('total_cost_usd', 0.0)

        # Token info
        input_tokens = ep.get('total_input_tokens', 0)
        output_tokens = ep.get('total_output_tokens', 0)
        total_tokens = input_tokens + output_tokens

        episode_data.append({
            'episode_id': ep_id,
            'environment': env,
            'seed': seed,
            'overall_acc': overall_acc,
            'planning_acc': planning_acc,
            'interventional_acc': interventional_acc,
            'counterfactual_acc': counterfactual_acc,
            'n_planning': len(planning_results),
            'n_interventional': len(interventional_results),
            'n_counterfactual': len(counterfactual_results),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'cost_usd': total_cost
        })

    df_episodes = pd.DataFrame(episode_data)

    # Display episode table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print("\n" + df_episodes[['episode_id', 'environment', 'seed', 'overall_acc',
                              'planning_acc', 'interventional_acc', 'counterfactual_acc',
                              'total_tokens', 'cost_usd']].to_string(index=False))

    # =========================================================================
    # COST ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("COST ANALYSIS")
    print("="*70)

    # Total cost
    total_cost = df_episodes['cost_usd'].sum()
    total_tokens = df_episodes['total_tokens'].sum()

    print(f"\nTotal cost (15 episodes): ${total_cost:.4f}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average cost per episode: ${total_cost/len(episodes):.4f}")
    print(f"Average tokens per episode: {total_tokens/len(episodes):,.0f}")

    # Cost by environment
    print(f"\nCost by environment:")
    env_costs = df_episodes.groupby('environment').agg({
        'cost_usd': ['sum', 'mean', 'count'],
        'total_tokens': ['sum', 'mean']
    }).round(4)

    print(env_costs)

    # Show per-episode cost by environment
    print(f"\nPer-episode breakdown by environment:")
    for env in sorted(df_episodes['environment'].unique()):
        env_data = df_episodes[df_episodes['environment'] == env]
        print(f"\n{env}:")
        print(f"  Episodes: {len(env_data)}")
        print(f"  Total cost: ${env_data['cost_usd'].sum():.4f}")
        print(f"  Mean cost per episode: ${env_data['cost_usd'].mean():.4f} ± ${env_data['cost_usd'].std():.4f}")
        print(f"  Total tokens: {env_data['total_tokens'].sum():,}")
        print(f"  Mean tokens per episode: {env_data['total_tokens'].mean():,.0f} ± {env_data['total_tokens'].std():.0f}")

    # =========================================================================
    # SAVE DETAILED RESULTS
    # =========================================================================
    output_dir = results_dir.parent.parent / 'aggregated' / results_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    df_episodes.to_csv(output_dir / 'episode_details.csv', index=False)
    df_breakdown.to_csv(output_dir / 'question_type_breakdown.csv', index=False)

    print(f"\n\nSaved detailed results to: {output_dir}")

if __name__ == '__main__':
    main()
