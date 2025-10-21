#!/usr/bin/env python3
"""
Main experiment execution.

Usage:
    python scripts/run_experiment.py --test-mode
    python scripts/run_experiment.py --config config.yaml
"""

import argparse
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.runner import ExperimentRunner
from experiments.config import load_config
from environments.hot_pot import HotPotLab
from environments.switch_light import SwitchLight
from environments.chem_tile import ChemTile
from agents.observer import ObserverAgent
from agents.actor import ActorAgent
from agents.text_reader import TextReaderAgent
from agents.model_based import ModelBasedAgent


def main():
    parser = argparse.ArgumentParser(description="Run world model experiments")
    parser.add_argument('--config', default='config.yaml', help='Config file (default: config.yaml)')
    parser.add_argument('--num-episodes', type=int, default=None, help='Episodes per condition (overrides config)')
    parser.add_argument('--output-dir', default='results/raw', help='Output directory')
    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config)

    # Add runtime settings
    config['use_mock_llm'] = False  # Never use mock LLM in experiments

    # Override num_episodes if specified on command line
    if args.num_episodes is not None:
        for env_name in config['environments']:
            if env_name in config['environments']:
                config['environments'][env_name]['num_episodes'] = args.num_episodes
                # Generate seeds if not specified
                if 'seeds' not in config['environments'][env_name]:
                    base_seed = {'hot_pot': 42, 'switch_light': 100, 'chem_tile': 200, 'transfer': 300}.get(env_name, 0)
                    config['environments'][env_name]['seeds'] = list(range(base_seed, base_seed + args.num_episodes))

    # Convert model strings to dicts (runner expects {'model': 'name'} format)
    for agent_type in config['models']:
        if isinstance(config['models'][agent_type], str):
            config['models'][agent_type] = {'model': config['models'][agent_type]}

    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output_dir) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(results_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Environment mapping
    env_classes = {
        'hot_pot': HotPotLab,
        'switch_light': SwitchLight,
        'chem_tile': ChemTile,
    }

    # Agent mapping
    agent_classes = {
        'observer': ObserverAgent,
        'actor': ActorAgent,
        'text_reader': TextReaderAgent,
        'model_based': ModelBasedAgent,
    }

    # Run experiments
    total_episodes = 0
    failed_episodes = 0

    print(f"\n{'='*70}")
    print(f"EXPERIMENT RUN: {timestamp}")
    print(f"Episodes per condition: {args.num_episodes}")
    print(f"Results: {results_dir}")
    print(f"\nMODEL CONFIGURATION:")
    print(f"  Observer:     {config['models']['observer']['model']}")
    print(f"  Actor:        {config['models']['actor']['model']}")
    print(f"  Text Reader:  {config['models']['text_reader']['model']}")
    print(f"  Model Based:  {config['models']['model_based']['model']}")
    print(f"{'='*70}\n")

    for env_name, env_config in config['environments'].items():
        print(f"\n{'='*70}")
        print(f"Environment: {env_name}")
        print(f"{'='*70}")

        env_class = env_classes[env_name]

        for agent_name in ['observer', 'actor']:  # Start with two agents for testing
            print(f"\nAgent: {agent_name}")

            agent_class = agent_classes[agent_name]

            runner = ExperimentRunner(
                config=config,
                environment_cls=env_class,
                agent_cls=agent_class
            )

            # Run episodes
            for i, seed in enumerate(env_config['seeds'][:env_config['num_episodes']]):
                episode_id = f"{env_name}_{agent_name}_ep{i:03d}"

                print(f"  Episode {i+1}/{env_config['num_episodes']} (seed={seed})...", end=' ')

                try:
                    episode_log = runner.run_episode(
                        episode_id=episode_id,
                        seed=seed,
                        save_dir=results_dir
                    )
                    print(f"✓ ({len(episode_log['steps'])} steps)")
                    total_episodes += 1

                except Exception as e:
                    print(f"✗ ERROR: {e}")
                    failed_episodes += 1
                    import traceback
                    traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"COMPLETED")
    print(f"{'='*70}")
    print(f"Total episodes: {total_episodes}")
    print(f"Failed episodes: {failed_episodes}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*70}\n")

    # Print next steps
    print("Next steps:")
    print(f"  1. Analyze results:")
    print(f"     python scripts/analyze_results.py --results {results_dir}")
    print(f"\n  2. Inspect episode:")
    print(f"     python scripts/inspect_episode.py {results_dir}/<episode_file>.json")


if __name__ == '__main__':
    main()
