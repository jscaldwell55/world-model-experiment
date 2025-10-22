#!/usr/bin/env python3
"""Run negative control experiment with shuffled/random text.

This script runs the token prediction experiment with negative control
textualizations to validate that coupling is due to semantics, not artifacts.

Expected results:
- Normal coupling: r > 0.5
- Shuffled coupling: r < 0.2
- Random coupling: r < 0.2

If controls show high coupling, it suggests spurious correlation.
"""

import os
import sys
import yaml
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
from agents.observer import ObserverAgent
from agents.actor import ActorAgent
from models.belief_state import HotPotBelief, SwitchLightBelief, ChemTileBelief
from token_prediction.openai_predictor import OpenAINextSentencePredictor
from experiments.token_runner import run_episode_with_tokens, create_textualizer


def create_agent_for_env(agent_name: str, env, llm):
    """Create agent instance for environment."""
    env_name = env.__class__.__name__

    if agent_name == 'observer':
        agent = ObserverAgent(llm, action_budget=10)
    elif agent_name == 'actor':
        agent = ActorAgent(llm, action_budget=10, environment_name=env_name.lower().replace('lab', ''))

        # Initialize belief state
        if 'HotPot' in env_name:
            agent.set_belief_state(HotPotBelief())
        elif 'SwitchLight' in env_name:
            agent.set_belief_state(SwitchLightBelief())
        elif 'ChemTile' in env_name or 'Chem' in env_name:
            agent.set_belief_state(ChemTileBelief())
    else:
        raise ValueError(f"Unknown agent type: {agent_name}")

    return agent


def run_negative_control_experiment(output_dir: str, num_episodes_per_env: int = 10):
    """Run negative control experiments with shuffled and random text.

    Args:
        output_dir: Directory to save results
        num_episodes_per_env: Episodes per environment (default: 10)
    """

    print("=" * 70)
    print("NEGATIVE CONTROL EXPERIMENT")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Episodes per environment: {num_episodes_per_env}")
    print()

    # Check API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    # Setup
    env_classes = {
        'hot_pot': HotPotLab,
        'switch_light': SwitchLight,
        'chem_tile': ChemTile,
    }

    control_modes = ['shuffled', 'random']
    agents_to_run = ['actor']  # Actor has belief_surprisal
    base_seed = 42

    # Generate seeds
    seeds = list(range(base_seed, base_seed + num_episodes_per_env))

    # Total episodes
    total_episodes = len(env_classes) * len(control_modes) * len(agents_to_run) * len(seeds)
    episode_count = 0

    print(f"Configuration:")
    print(f"  Environments: {list(env_classes.keys())}")
    print(f"  Control modes: {control_modes}")
    print(f"  Agent: actor (with belief surprisal)")
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
    for control_mode in control_modes:
        print(f"\n{'='*70}")
        print(f"CONTROL MODE: {control_mode.upper()}")
        print(f"{'='*70}\n")

        for env_name, EnvClass in env_classes.items():
            for agent_name in agents_to_run:
                print(f"Environment: {env_name} | Agent: {agent_name} | Control: {control_mode}")
                print("-" * 70)

                for seed_idx, seed in enumerate(seeds):
                    episode_count += 1

                    print(f"[{episode_count}/{total_episodes}] Episode {seed_idx+1}/{num_episodes_per_env} (seed={seed})... ", end="", flush=True)

                    try:
                        # Create instances
                        env = EnvClass(seed=seed)
                        agent = create_agent_for_env(agent_name, env, llm)
                        textualizer = create_textualizer(env)

                        # Run episode with control mode
                        test_results, token_logger = run_episode_with_tokens(
                            env=env,
                            agent=agent,
                            textualizer=textualizer,
                            predictor=predictor,
                            seed=seed,
                            max_actions=10,
                            save_dir=output_dir,
                            control_mode=control_mode  # KEY: Enable negative control
                        )

                        # Summary
                        num_steps = len(token_logger.entries)
                        avg_nll = sum(e.sequence_nll for e in token_logger.entries) / num_steps if num_steps > 0 else 0

                        all_results.append({
                            'environment': env_name,
                            'agent': agent_name,
                            'control_mode': control_mode,
                            'seed': seed,
                            'num_steps': num_steps,
                            'avg_nll': avg_nll,
                            'status': 'success'
                        })

                        print(f"✓ ({num_steps} steps, NLL={avg_nll:.2f})")

                    except Exception as e:
                        print(f"✗ FAILED: {e}")
                        all_results.append({
                            'environment': env_name,
                            'agent': agent_name,
                            'control_mode': control_mode,
                            'seed': seed,
                            'status': 'failed',
                            'error': str(e)
                        })

                print()

    # Save summary
    import pandas as pd
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(output_dir, 'negative_control_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    print()
    print("=" * 70)
    print("NEGATIVE CONTROL EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Episodes completed: {episode_count}/{total_episodes}")
    print(f"Success rate: {(summary_df['status'] == 'success').mean():.1%}")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_path}")
    print()
    print("Next steps:")
    print(f"  1. Analyze coupling: python scripts/analyze_full_token_results.py {output_dir}")
    print(f"  2. Compare to normal: evaluation/token_analysis.py compare_control_coupling()")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run negative control experiment")
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
        output_dir = f"results/negative_control_{timestamp}"
    else:
        output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)

    # Run experiment
    run_negative_control_experiment(output_dir, args.num_episodes)


if __name__ == '__main__':
    main()
