#!/usr/bin/env python3
"""Analyze experiment results"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import compute_all_metrics
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    
    # Load all episodes
    episodes = []
    for episode_file in results_dir.glob('*.json'):
        if 'config' in episode_file.name:
            continue
        with open(episode_file) as f:
            episodes.append(json.load(f))
    
    print(f"Loaded {len(episodes)} episodes")
    
    # Compute metrics
    results = []
    for ep in episodes:
        ep_metrics = compute_all_metrics(ep)
        results.append({
            'episode_id': ep['episode_id'],
            'environment': ep.get('environment', 'unknown'),
            'agent': ep.get('agent_type', 'unknown'),
            'seed': ep.get('seed', 0),
            'overall_accuracy': ep_metrics['overall_accuracy'],
            'interventional_accuracy': ep_metrics['interventional_accuracy'],
            'counterfactual_accuracy': ep_metrics['counterfactual_accuracy'],
            'brier_score': ep_metrics['calibration']['brier_score'],
            'mean_surprisal': ep_metrics['surprisal_trajectory']['mean_surprisal'],
            'surprisal_slope': ep_metrics['surprisal_trajectory']['slope'],
        })
    
    df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    summary = df.groupby(['environment', 'agent']).agg({
        'overall_accuracy': ['mean', 'std'],
        'interventional_accuracy': ['mean', 'std'],
    }).round(3)
    
    print(summary)
    
    # Save
    output_dir = results_dir.parent.parent / 'aggregated' / results_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / 'detailed_metrics.csv', index=False)
    summary.to_csv(output_dir / 'summary.csv')
    
    print(f"\nSaved to: {output_dir}")

if __name__ == '__main__':
    main()
