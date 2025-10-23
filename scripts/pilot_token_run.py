#!/usr/bin/env python3
"""
Pilot token prediction experiment.

Runs: 5 episodes × 3 environments × 2 agents = 30 episodes total

This is a minimal pilot to test the token prediction infrastructure.
For full agent reasoning, use run_experiment.py with token prediction enabled.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.hot_pot import HotPotLab
from environments.switch_light import SwitchLight
from environments.chem_tile import ChemTile
from agents.base import OpenAILLM
from agents.observer import ObserverAgent
from agents.actor import ActorAgent
from models.belief_state import HotPotBelief, SwitchLightBelief, ChemTileBelief
from token_prediction.openai_predictor import OpenAINextSentencePredictor
from experiments.token_runner import run_episode_with_tokens, create_textualizer


def load_config(config_path: str = "config_token.yaml") -> dict:
    """Load token experiment configuration."""
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Warning: Config file {config_path} not found, using defaults")
        return {
            'token_prediction': {
                'enabled': True,
                'predictors': {
                    'observer': {'model': 'gpt-4o-mini'},
                    'actor': {'model': 'gpt-4o-mini'}
                }
            },
            'pilot': {
                'num_episodes_per_env': 5,
                'environments': ['hot_pot', 'switch_light', 'chem_tile'],
                'agents': ['observer', 'actor'],
                'seeds': [42, 43, 44, 45, 46],
                'output_dir': 'results/token_prediction_pilot'
            }
        }

    with open(config_path) as f:
        return yaml.safe_load(f)


def create_agent_for_env(agent_name: str, env, llm):
    """Create agent instance for environment.

    Args:
        agent_name: 'observer' or 'actor'
        env: Environment instance
        llm: LLM interface

    Returns:
        Agent instance
    """
    env_name = env.__class__.__name__

    if agent_name == 'observer':
        agent = ObserverAgent(llm, action_budget=10)

    elif agent_name == 'actor':
        # Use the actual environment class name for tool lookup
        agent = ActorAgent(llm, action_budget=10, environment_name=env_name)

        # Initialize belief state for Actor agents
        if 'HotPot' in env_name:
            agent.set_belief_state(HotPotBelief())
        elif 'SwitchLight' in env_name:
            agent.set_belief_state(SwitchLightBelief())
        elif 'ChemTile' in env_name or 'Chem' in env_name:
            agent.set_belief_state(ChemTileBelief())

    else:
        raise ValueError(f"Unknown agent type: {agent_name}")

    return agent


def run_pilot(config: dict, output_dir: str):
    """Run pilot experiment.

    Args:
        config: Configuration dictionary
        output_dir: Output directory for results

    Returns:
        List of token loggers from all episodes
    """

    print("=" * 70)
    print("TOKEN PREDICTION PILOT EXPERIMENT")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print()

    # Check for API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    # Setup environments
    env_classes = {
        'hot_pot': HotPotLab,
        'switch_light': SwitchLight,
        'chem_tile': ChemTile,
    }

    # Get pilot configuration
    pilot_config = config.get('pilot', {})
    envs_to_run = pilot_config.get('environments', ['hot_pot', 'switch_light', 'chem_tile'])
    agents_to_run = pilot_config.get('agents', ['observer', 'actor'])
    seeds = pilot_config.get('seeds', [42, 43, 44, 45, 46])

    # Create predictor (use first available predictor from config)
    predictors = config['token_prediction']['predictors']
    predictor_name = next(iter(predictors.keys()))
    predictor_config = predictors[predictor_name]
    print(f"Using '{predictor_name}' predictor for token prediction")
    predictor = OpenAINextSentencePredictor(
        model=predictor_config['model']
    )

    print(f"Token predictor: {predictor.get_model_name()}")
    print()

    # Create LLM for agents (using Anthropic as per config.yaml)
    from agents.base import create_llm
    llm = create_llm('claude-sonnet-4-5-20250929')

    # Run episodes
    total_episodes = len(envs_to_run) * len(agents_to_run) * len(seeds)
    episode_count = 0

    all_token_loggers = []

    for env_name in envs_to_run:
        EnvClass = env_classes[env_name]

        for agent_name in agents_to_run:

            for seed in seeds:
                episode_count += 1
                print(f"\n[{episode_count}/{total_episodes}] Running {env_name} × {agent_name} × seed={seed}")
                print("-" * 70)

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

                    all_token_loggers.append(token_logger)

                    # Print summary
                    num_steps = len(token_logger.entries)
                    if num_steps > 0:
                        avg_nll = sum(e.sequence_nll for e in token_logger.entries) / num_steps
                        print(f"  ✓ Completed: {num_steps} steps, avg NLL = {avg_nll:.2f}")
                    else:
                        print(f"  ✓ Completed: 0 steps (Observer agent)")

                except Exception as e:
                    print(f"  ✗ Failed: {e}")
                    import traceback
                    traceback.print_exc()

    print()
    print("=" * 70)
    print("PILOT COMPLETE")
    print(f"Episodes completed: {episode_count}/{total_episodes}")
    print(f"Token logs saved to: {output_dir}")
    print("=" * 70)

    return all_token_loggers


def main():
    parser = argparse.ArgumentParser(description="Run token prediction pilot experiment")
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
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/raw/pilot_token_{timestamp}"
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Run pilot
    token_loggers = run_pilot(config, output_dir)

    print("\nTo analyze results, run:")
    print(f"  python scripts/analyze_token_pilot.py {output_dir}")


if __name__ == '__main__':
    main()
