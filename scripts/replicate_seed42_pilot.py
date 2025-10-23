#!/usr/bin/env python3
"""
Replicate seed 42 Mini Pilot to check consistency.

This script re-runs the Mini Pilot experiment with:
- SAME seed (42)
- SAME environment (HotPotLab)
- SAME agent (ActorAgent)
- SAME textualization (HotPotTextualization - original)
- 5 episodes total (one per seed: 42, 43, 44, 45, 46)

Purpose: Determine if seed 42's anti-coupling (r=-0.336) replicates,
or if seeds 100-104 were outliers due to high variance.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.hot_pot import HotPotLab
from agents.base import create_llm
from agents.actor import ActorAgent
from models.belief_state import HotPotBelief
from textualization.hot_pot_text import HotPotTextualization
from token_prediction.openai_predictor import OpenAINextSentencePredictor
from experiments.token_runner import run_episode_with_tokens


def extract_metrics(episode_data):
    """Extract token NLL and belief surprisal from episode data.

    Returns:
        token_nll: array of per-token NLL values
        belief_surprisal: array of belief surprisal values
    """
    entries = episode_data['entries']

    token_nll = []
    belief_surprisal = []

    for entry in entries:
        token_nll.append(entry['per_token_nll'])
        belief_surprisal.append(entry['belief_surprisal'])

    return np.array(token_nll), np.array(belief_surprisal)


def compute_correlation(token_nll, belief_surprisal):
    """Compute Pearson correlation between token NLL and belief surprisal."""
    if len(token_nll) == 0 or len(belief_surprisal) == 0:
        return 0.0

    pearson_r, pearson_p = stats.pearsonr(token_nll, belief_surprisal)
    return pearson_r


def run_seed42_replication(output_dir: str):
    """Run 5 episodes with seeds 42-46 using ORIGINAL textualization.

    Args:
        output_dir: Directory to save episode logs

    Returns:
        List of episode data dictionaries
    """
    print("=" * 70)
    print("SEED 42 REPLICATION EXPERIMENT")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print()

    # Check for API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    # Setup
    seeds = [42, 43, 44, 45, 46]
    llm = create_llm('claude-sonnet-4-5-20250929')
    predictor = OpenAINextSentencePredictor(model='gpt-4o-mini')
    textualizer = HotPotTextualization()  # ORIGINAL textualization

    print(f"Environment: HotPotLab")
    print(f"Agent: ActorAgent")
    print(f"Textualization: HotPotTextualization (original)")
    print(f"Seeds: {seeds}")
    print(f"Token predictor: {predictor.get_model_name()}")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run episodes
    episode_data_list = []

    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Running episode with seed={seed}")
        print("-" * 70)

        try:
            # Create fresh environment and agent for each episode
            env = HotPotLab(seed=seed)
            agent = ActorAgent(llm, action_budget=10, environment_name='HotPotLab')
            agent.set_belief_state(HotPotBelief())

            # Run episode
            test_results, token_logger = run_episode_with_tokens(
                env=env,
                agent=agent,
                textualizer=textualizer,
                predictor=predictor,
                seed=seed,
                max_actions=10,
                save_dir=output_dir
            )

            # Load the saved episode data
            episode_id = f"HotPotLab_ActorAgent_ep{seed:03d}"
            log_path = os.path.join(output_dir, f"{episode_id}_token.json")

            with open(log_path, 'r') as f:
                episode_data = json.load(f)

            episode_data_list.append(episode_data)

            # Print summary
            num_steps = len(token_logger.entries)
            if num_steps > 0:
                avg_nll = sum(e.sequence_nll for e in token_logger.entries) / num_steps
                print(f"  ✓ Completed: {num_steps} steps, avg NLL = {avg_nll:.2f}")
            else:
                print(f"  ✓ Completed: {num_steps} steps")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 70)
    print("EPISODES COMPLETE")
    print(f"Total episodes: {len(episode_data_list)}/{len(seeds)}")
    print("=" * 70)

    return episode_data_list


def analyze_results(episode_data_list):
    """Analyze correlation results and compare to Mini Pilot.

    Args:
        episode_data_list: List of episode data dictionaries
    """
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    print()

    # Mini Pilot baseline (seed 42 only)
    r_mini_pilot_seed42 = -0.336

    # Compute correlation for each episode
    correlations = []

    for episode_data in episode_data_list:
        episode_id = episode_data['episode_id']
        token_nll, belief_surprisal = extract_metrics(episode_data)

        if len(token_nll) > 0:
            r = compute_correlation(token_nll, belief_surprisal)
            correlations.append(r)
            print(f"{episode_id}: r = {r:.3f} (n={len(token_nll)} steps)")
        else:
            print(f"{episode_id}: No data points")

    if len(correlations) == 0:
        print("\nERROR: No valid correlations computed")
        return

    # Compute average correlation across all episodes
    r_mean = np.mean(correlations)
    r_std = np.std(correlations)

    print()
    print("-" * 70)
    print(f"Mean correlation (across {len(correlations)} episodes): r = {r_mean:.3f} ± {r_std:.3f}")
    print(f"Mini Pilot (seed 42 only): r = {r_mini_pilot_seed42:.3f}")
    print("-" * 70)
    print()

    # Compare to Mini Pilot
    difference = abs(r_mean - r_mini_pilot_seed42)

    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    if difference < 0.15:
        print("✅ REPLICATES - Seed 42 consistently shows anti-coupling")
        print(f"   Difference from Mini Pilot: Δr = {difference:.3f} (< 0.15 threshold)")
        print()
        print("→ Seeds 100-104 were likely outliers (high variance)")
        print("→ The anti-coupling effect appears stable across seeds")
        print()
        print("RECOMMENDATION: Scale up with many seeds (n=200)")
        print("  - Run full experiment with diverse seed set")
        print("  - The negative correlation pattern is reproducible")
        print("  - H0 hypothesis is supported by replication")
    else:
        print("❌ DOES NOT REPLICATE - Mini Pilot result was a fluke")
        print(f"   Difference from Mini Pilot: Δr = {difference:.3f} (≥ 0.15 threshold)")
        print()
        print("→ True effect is likely near zero or inconsistent")
        print("→ Original seed 42 result may have been spurious")
        print()
        print("RECOMMENDATION: Pivot to H1-H5")
        print("  - The anti-coupling pattern is not stable")
        print("  - Consider alternative hypotheses or experimental designs")
        print("  - H0 may not be the right research direction")

    print()
    print("=" * 70)

    # Additional statistics
    print()
    print("VARIANCE ANALYSIS:")
    print(f"  Standard deviation across episodes: σ = {r_std:.3f}")

    if r_std > 0.3:
        print("  ⚠️  HIGH VARIANCE - Results highly seed-dependent")
        print("     → Need larger n to detect stable effect")
    elif r_std > 0.15:
        print("  ⚠️  MODERATE VARIANCE - Some seed sensitivity")
        print("     → Recommend n ≥ 50 for stable estimates")
    else:
        print("  ✓  LOW VARIANCE - Stable effect across seeds")
        print("     → n = 20-30 may be sufficient")

    print()


def main():
    """Main execution."""
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/raw/seed42_replication_{timestamp}"

    # Run episodes
    episode_data_list = run_seed42_replication(output_dir)

    # Analyze results
    if episode_data_list:
        analyze_results(episode_data_list)
    else:
        print("\nERROR: No episodes completed successfully")
        sys.exit(1)

    print(f"\nResults saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
