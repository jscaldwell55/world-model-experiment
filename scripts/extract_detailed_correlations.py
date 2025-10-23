#!/usr/bin/env python3
"""
Extract detailed per-step correlation data and compare variance to Pilot 2.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

def load_episode_data(filepath: Path) -> Dict:
    """Load episode JSON data."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_detailed_steps(episode_data: Dict, env_name: str) -> pd.DataFrame:
    """
    Extract detailed per-step data for inspection.

    Returns DataFrame with columns:
    - step: Step number
    - token_nll: Per-token NLL
    - belief_surprisal: Belief surprisal
    - sequence_nll: Full sequence NLL
    - observation: True observation text (truncated)
    """
    entries = episode_data['entries']

    rows = []
    for entry in entries:
        rows.append({
            'step': entry['step'],
            'token_nll': entry['per_token_nll'],
            'belief_surprisal': entry['belief_surprisal'],
            'sequence_nll': entry['sequence_nll'],
            'observation': entry['true_observation'][:60] + ('...' if len(entry['true_observation']) > 60 else ''),
            'predicted': entry['predicted_text'][:40] + ('...' if len(entry['predicted_text']) > 40 else ''),
        })

    df = pd.DataFrame(rows)
    df['env'] = env_name
    return df

def analyze_coupling_events(df: pd.DataFrame, env_name: str):
    """
    Analyze specific high-surprisal moments to see if they couple with token NLL.
    """
    print(f"\n{'='*80}")
    print(f"COUPLING ANALYSIS: {env_name}")
    print(f"{'='*80}\n")

    # Overall statistics
    print(f"Overall Statistics:")
    print(f"  Token NLL:        μ={df['token_nll'].mean():.4f}, σ={df['token_nll'].std():.4f}")
    print(f"  Belief Surprisal: μ={df['belief_surprisal'].mean():.4f}, σ={df['belief_surprisal'].std():.4f}")
    print()

    # Find high surprisal moments (> 2.0 or > 75th percentile)
    high_surprisal_threshold = max(2.0, df['belief_surprisal'].quantile(0.75))
    high_surprisal = df[df['belief_surprisal'] > high_surprisal_threshold]

    if len(high_surprisal) > 0:
        print(f"High Surprisal Events (> {high_surprisal_threshold:.2f}):")
        print(f"{'Step':>4} | {'Token NLL':>10} | {'Surprisal':>10} | Observation")
        print(f"{'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*50}")
        for _, row in high_surprisal.iterrows():
            print(f"{row['step']:>4} | {row['token_nll']:>10.4f} | {row['belief_surprisal']:>10.4f} | {row['observation']}")
        print()

        # Check if high surprisal correlates with high token NLL
        avg_nll_high_surprisal = high_surprisal['token_nll'].mean()
        avg_nll_low_surprisal = df[df['belief_surprisal'] <= high_surprisal_threshold]['token_nll'].mean()

        print(f"Coupling Check:")
        print(f"  Avg Token NLL at HIGH surprisal: {avg_nll_high_surprisal:.4f}")
        print(f"  Avg Token NLL at LOW surprisal:  {avg_nll_low_surprisal:.4f}")
        if avg_nll_high_surprisal > avg_nll_low_surprisal:
            print(f"  → POSITIVE COUPLING: High surprisal → High NLL (Δ={avg_nll_high_surprisal - avg_nll_low_surprisal:.4f})")
        elif avg_nll_high_surprisal < avg_nll_low_surprisal:
            print(f"  → NEGATIVE COUPLING: High surprisal → Low NLL (Δ={avg_nll_high_surprisal - avg_nll_low_surprisal:.4f})")
        else:
            print(f"  → NO COUPLING: Similar NLL across surprisal levels")
    else:
        print(f"No high surprisal events detected (threshold: {high_surprisal_threshold:.2f})")

    print()

def search_pilot2_data(results_dir: Path) -> List[Path]:
    """Search for any Pilot 2 data files."""
    pilot2_candidates = []

    # Search patterns
    patterns = [
        '*pilot*2*',
        '*pilot_2*',
        '*pilot2*',
        '*baseline*',
        '*20251020*',  # Based on aggregated date seen earlier
    ]

    for pattern in patterns:
        pilot2_candidates.extend(results_dir.rglob(pattern))

    # Filter to JSON files only
    json_files = [f for f in pilot2_candidates if f.suffix == '.json']

    return json_files

def analyze_pilot2_variance(pilot2_files: List[Path]) -> pd.DataFrame:
    """Extract token NLL variance from Pilot 2 files if available."""
    results = []

    for filepath in pilot2_files:
        try:
            data = load_episode_data(filepath)
            if 'entries' in data:
                token_nlls = [entry.get('per_token_nll', entry.get('token_nll', None))
                             for entry in data['entries']]
                token_nlls = [x for x in token_nlls if x is not None]

                if token_nlls:
                    env_name = filepath.stem.split('_')[0]  # Extract env from filename
                    results.append({
                        'file': filepath.name,
                        'env': env_name,
                        'n_steps': len(token_nlls),
                        'token_nll_mean': np.mean(token_nlls),
                        'token_nll_std': np.std(token_nlls),
                        'token_nll_min': np.min(token_nlls),
                        'token_nll_max': np.max(token_nlls),
                    })
        except Exception as e:
            continue

    if results:
        return pd.DataFrame(results)
    else:
        return None

def main():
    """Main analysis pipeline."""
    # Paths
    data_dir = Path("/Users/jaycaldwell/world-model-experiment/results/raw/pilot_token_20251022_182749")
    results_dir = Path("/Users/jaycaldwell/world-model-experiment/results")

    # Episode files
    episodes = {
        'HotPotLab': data_dir / 'HotPotLab_ActorAgent_ep042_token.json',
        'SwitchLight': data_dir / 'SwitchLight_ActorAgent_ep042_token.json',
        'ChemTile': data_dir / 'ChemTile_ActorAgent_ep042_token.json',
    }

    print("\n" + "="*80)
    print("DETAILED CORRELATION ANALYSIS")
    print("="*80)

    # Part 1: Per-step data tables
    print("\n" + "="*80)
    print("PART 1: PER-STEP DATA INSPECTION")
    print("="*80 + "\n")

    all_steps = []
    for env_name, filepath in episodes.items():
        episode_data = load_episode_data(filepath)
        df = extract_detailed_steps(episode_data, env_name)
        all_steps.append(df)

        print(f"\n{env_name} - Step-by-Step Data:")
        print(f"{'-'*80}")
        print(df.to_string(index=False))
        print()

        # Coupling analysis
        analyze_coupling_events(df, env_name)

    # Part 2: Search for Pilot 2 data
    print("\n" + "="*80)
    print("PART 2: PILOT 2 VARIANCE COMPARISON")
    print("="*80 + "\n")

    print("Searching for Pilot 2 data files...")
    pilot2_files = search_pilot2_data(results_dir)

    if pilot2_files:
        print(f"Found {len(pilot2_files)} potential Pilot 2 files:")
        for f in pilot2_files[:10]:  # Show first 10
            print(f"  - {f.relative_to(results_dir)}")
        if len(pilot2_files) > 10:
            print(f"  ... and {len(pilot2_files) - 10} more")
        print()

        pilot2_variance = analyze_pilot2_variance(pilot2_files)

        if pilot2_variance is not None:
            print("\nPilot 2 Token NLL Variance:")
            print(pilot2_variance.to_string(index=False))
            print()

            # Summary comparison
            print("\nVariance Comparison:")
            print(f"{'Environment':12} | {'Mini Pilot σ':>12} | {'Pilot 2 σ (range)':>20}")
            print(f"{'-'*12}-+-{'-'*12}-+-{'-'*20}")

            mini_pilot_stats = {
                'HotPotLab': 0.041,
                'SwitchLight': 0.090,
                'ChemTile': 0.102,
            }

            for env in ['HotPotLab', 'SwitchLight', 'ChemTile']:
                pilot2_env_data = pilot2_variance[pilot2_variance['env'].str.contains(env, case=False, na=False)]
                if len(pilot2_env_data) > 0:
                    pilot2_std_range = f"{pilot2_env_data['token_nll_std'].min():.3f}-{pilot2_env_data['token_nll_std'].max():.3f}"
                else:
                    pilot2_std_range = "N/A"

                print(f"{env:12} | {mini_pilot_stats[env]:>12.3f} | {pilot2_std_range:>20}")
        else:
            print("No Pilot 2 data with token NLL metrics found.")
    else:
        print("No Pilot 2 data files found in results directory.")
        print("\nPlease provide the location of Pilot 2 data for comparison.")

    # Part 3: Combined summary table
    print("\n" + "="*80)
    print("PART 3: COMBINED VARIANCE SUMMARY")
    print("="*80 + "\n")

    combined_df = pd.concat(all_steps, ignore_index=True)

    variance_summary = combined_df.groupby('env').agg({
        'token_nll': ['mean', 'std', 'min', 'max'],
        'belief_surprisal': ['mean', 'std', 'min', 'max'],
    }).round(4)

    print("Mini Pilot Variance Summary:")
    print(variance_summary)
    print()

    # Variance collapse diagnosis
    print("Variance Collapse Diagnosis:")
    print(f"{'Environment':12} | {'Token NLL σ':>12} | Status")
    print(f"{'-'*12}-+-{'-'*12}-+-{'-'*30}")

    for env in ['HotPotLab', 'SwitchLight', 'ChemTile']:
        env_data = combined_df[combined_df['env'] == env]
        std = env_data['token_nll'].std()
        if std < 0.1:
            status = "🚩 VARIANCE COLLAPSE"
        elif std < 0.3:
            status = "⚠️  Low variance"
        else:
            status = "✅ Healthy variance"
        print(f"{env:12} | {std:>12.3f} | {status}")

    print()

    # Save detailed data
    output_file = results_dir / 'mini_pilot_analysis' / 'detailed_step_data.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"\nDetailed step data saved to: {output_file}")

if __name__ == '__main__':
    main()
