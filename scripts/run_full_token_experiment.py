#!/usr/bin/env python3
"""
Run full token prediction experiment.
Extends pilot to full scale: 50 episodes × 3 envs × 2 agents = 300 episodes
(Reduced from 600 to focus on Observer and Actor for token prediction)
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


def load_config(config_path: str = "config_token.yaml") -> dict:
    """Load token experiment configuration."""
    config_path = Path(config_path)
    if not config_path.exists():
        return {
            'token_prediction': {
                'predictors': {
                    'observer': {'model': 'gpt-4o-mini'},
                    'actor': {'model': 'gpt-4o-mini'}
                }
            },
            'num_episodes_per_env': 50,
            'environments': ['hot_pot', 'switch_light', 'chem_tile'],
            'agents': ['observer', 'actor'],
            'base_seed': 42
        }

    with open(config_path) as f:
        return yaml.safe_load(f)


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


def run_full_experiment(config: dict, output_dir: str):
    """Run full-scale token prediction experiment."""

    print("=" * 70)
    print("FULL TOKEN PREDICTION EXPERIMENT")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print()

    # Check API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    # Setup environments
    env_classes = {
        'hot_pot': HotPotLab,
        'switch_light': SwitchLight,
        'chem_tile': ChemTile,
    }

    # Get configuration
    num_episodes = config.get('num_episodes_per_env', 50)
    envs_to_run = config.get('environments', ['hot_pot', 'switch_light', 'chem_tile'])
    agents_to_run = config.get('agents', ['observer', 'actor'])
    base_seed = config.get('base_seed', 42)

    # Generate seeds
    seeds = list(range(base_seed, base_seed + num_episodes))

    # Total episodes
    total_episodes = len(envs_to_run) * len(agents_to_run) * len(seeds)
    episode_count = 0

    print(f"Configuration:")
    print(f"  Environments: {envs_to_run}")
    print(f"  Agents: {agents_to_run}")
    print(f"  Episodes per combination: {num_episodes}")
    print(f"  Total episodes: {total_episodes}")
    print()

    # Create predictors per agent type
    predictors = {}
    for agent_name in agents_to_run:
        predictor_config = config['token_prediction']['predictors'].get(
            agent_name,
            config['token_prediction']['predictors']['observer']
        )
        predictors[agent_name] = OpenAINextSentencePredictor(
            model=predictor_config['model']
        )
        print(f"  {agent_name} predictor: {predictor_config['model']}")
    print()

    # Create LLM for agents (using Anthropic as per config.yaml)
    llm = create_llm('claude-sonnet-4-5-20250929')

    # Track results
    all_results = []

    # Run episodes
    for env_name in envs_to_run:
        EnvClass = env_classes[env_name]

        for agent_name in agents_to_run:
            predictor = predictors[agent_name]

            print(f"\n{'='*70}")
            print(f"Environment: {env_name} | Agent: {agent_name}")
            print(f"{'='*70}")

            for seed_idx, seed in enumerate(seeds):
                episode_count += 1

                print(f"[{episode_count}/{total_episodes}] Episode {seed_idx+1}/{num_episodes} (seed={seed})... ", end="", flush=True)

                try:
                    # Create instances
                    env = EnvClass(seed=seed)
                    agent = create_agent_for_env(agent_name, env, llm)
                    textualizer = create_textualizer(env)

                    # Run episode with token prediction
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

                    all_results.append({
                        'environment': env_name,
                        'agent': agent_name,
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
                        'seed': seed,
                        'status': 'failed',
                        'error': str(e)
                    })

    # Save summary
    import pandas as pd
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(output_dir, 'experiment_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Episodes completed: {episode_count}/{total_episodes}")
    print(f"Success rate: {(summary_df['status'] == 'success').mean():.1%}")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_path}")
    print()
    print("Next steps:")
    print(f"  1. Analyze results: python scripts/analyze_full_token_results.py {output_dir}")
    print(f"  2. Generate figures: python scripts/generate_token_figures.py {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run full token prediction experiment")
    parser.add_argument(
        '--config',
        type=str,
        default='config_token.yaml',
        help='Path to token configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for token logs'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=None,
        help='Number of episodes per environment-agent combination'
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override num episodes if specified
    if args.num_episodes:
        config['num_episodes_per_env'] = args.num_episodes

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/raw/token_experiment_{timestamp}"
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Save config used
    with open(os.path.join(output_dir, 'config_used.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Run experiment
    run_full_experiment(config, output_dir)


if __name__ == '__main__':
    main()
