#!/usr/bin/env python3
"""Run model-based agent baseline experiment.

This script runs token prediction experiments with the ModelBasedAgent,
which uses an explicit learned dynamics model for planning. This serves
as a baseline to compare against Actor and Observer agents.

Expected results:
- Model-based agents should have the strongest coupling (highest r)
- Ranking: model_based > actor > observer

If model_based doesn't outperform actor, suggests:
1. Insufficient training data for model learning
2. Model learning not effective
3. Planning not providing advantage
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.hot_pot import HotPotLab
from environments.switch_light import SwitchLight
from environments.chem_tile import ChemTile
from agents.base import create_llm
from agents.model_based import ModelBasedAgent
from models.belief_state import HotPotBelief, SwitchLightBelief, ChemTileBelief
from token_prediction.openai_predictor import OpenAINextSentencePredictor
from experiments.token_runner import run_episode_with_tokens, create_textualizer


def create_model_based_agent(env, llm):
    """Create ModelBasedAgent instance for environment.

    Args:
        env: Environment instance
        llm: LLM interface

    Returns:
        ModelBasedAgent with initialized belief state
    """
    env_name = env.__class__.__name__

    # Get environment name for tool registry
    if 'HotPot' in env_name:
        env_key = 'HotPotLab'
        belief_class = HotPotBelief
    elif 'SwitchLight' in env_name:
        env_key = 'SwitchLight'
        belief_class = SwitchLightBelief
    elif 'ChemTile' in env_name or 'Chem' in env_name:
        env_key = 'ChemTile'
        belief_class = ChemTileBelief
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    # Create agent
    agent = ModelBasedAgent(llm, action_budget=10, environment_name=env_key)
    agent.set_belief_state(belief_class())

    return agent


def run_model_based_baseline(output_dir: str, num_episodes_per_env: int = 10):
    """Run model-based agent baseline experiments.

    Args:
        output_dir: Directory to save results
        num_episodes_per_env: Episodes per environment (default: 10)
    """

    print("=" * 70)
    print("MODEL-BASED AGENT BASELINE EXPERIMENT")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Episodes per environment: {num_episodes_per_env}")
    print()

    # Check API keys
    if not os.environ.get('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Setup
    env_classes = {
        'HotPot': HotPotLab,
        'SwitchLight': SwitchLight,
        'ChemTile': ChemTile,
    }

    base_seed = 42

    # Generate seeds
    seeds = list(range(base_seed, base_seed + num_episodes_per_env))

    # Total episodes
    total_episodes = len(env_classes) * len(seeds)
    episode_count = 0

    print(f"Configuration:")
    print(f"  Environments: {list(env_classes.keys())}")
    print(f"  Agent: ModelBased (with belief surprisal + dynamics model)")
    print(f"  Total episodes: {total_episodes}")
    print()

    # Create predictor
    predictor = OpenAINextSentencePredictor(model='gpt-4o-mini')
    print(f"  Token predictor: gpt-4o-mini")
    print()

    # Create LLM for agents
    llm = create_llm('claude-sonnet-4-5-20250929')

    # Track results
    all_results = []

    # Run episodes
    for env_name, EnvClass in env_classes.items():
        print(f"\n{'='*70}")
        print(f"ENVIRONMENT: {env_name}")
        print(f"{'='*70}\n")

        for seed_idx, seed in enumerate(seeds):
            episode_count += 1

            print(f"[{episode_count}/{total_episodes}] Episode {seed_idx+1}/{num_episodes_per_env} (seed={seed})... ", end="", flush=True)

            try:
                # Create instances
                env = EnvClass(seed=seed)
                agent = create_model_based_agent(env, llm)
                textualizer = create_textualizer(env)

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

                # Summary
                num_steps = len(token_logger.entries)
                avg_nll = sum(e.sequence_nll for e in token_logger.entries) / num_steps if num_steps > 0 else 0

                # Count non-zero surprisals
                non_zero_surprisals = sum(1 for e in token_logger.entries
                                         if e.belief_surprisal is not None
                                         and e.belief_surprisal not in [0.0, -0.0])

                all_results.append({
                    'environment': env_name,
                    'agent': 'ModelBased',
                    'seed': seed,
                    'num_steps': num_steps,
                    'avg_nll': avg_nll,
                    'non_zero_surprisals': non_zero_surprisals,
                    'surprisal_rate': non_zero_surprisals / num_steps if num_steps > 0 else 0,
                    'status': 'success'
                })

                print(f"✓ ({num_steps} steps, NLL={avg_nll:.2f}, surprisals={non_zero_surprisals}/{num_steps})")

            except Exception as e:
                print(f"✗ FAILED: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    'environment': env_name,
                    'agent': 'ModelBased',
                    'seed': seed,
                    'status': 'failed',
                    'error': str(e)
                })

        print()

    # Save summary
    import pandas as pd
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(output_dir, 'model_based_baseline_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    # Statistics
    success_df = summary_df[summary_df['status'] == 'success']

    print()
    print("=" * 70)
    print("MODEL-BASED BASELINE EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Episodes completed: {episode_count}/{total_episodes}")
    print(f"Success rate: {(summary_df['status'] == 'success').mean():.1%}")

    if len(success_df) > 0:
        print()
        print("Surprisal Statistics:")
        for env in success_df['environment'].unique():
            env_data = success_df[success_df['environment'] == env]
            avg_rate = env_data['surprisal_rate'].mean()
            print(f"  {env:15} - Avg surprisal rate: {avg_rate:.1%}")

    print()
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_path}")
    print()
    print("Next steps:")
    print(f"  1. Analyze coupling: python scripts/analyze_full_token_results.py {output_dir}")
    print(f"  2. Compare agents: evaluation/token_analysis.py compare_agent_coupling()")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run model-based baseline experiment")
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=10,
        help='Number of episodes per environment (default: 10)'
    )
    args = parser.parse_args()

    # Create output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/model_based_baseline_{timestamp}"
    else:
        output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)

    # Run experiment
    run_model_based_baseline(output_dir, args.num_episodes)


if __name__ == '__main__':
    main()
