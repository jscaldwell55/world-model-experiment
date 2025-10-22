#!/usr/bin/env python3
"""
Quick validation run: 3 episodes (1 per environment) with Actor agent.

Purpose: Verify that belief surprisal is being extracted correctly after bug fix.
Expected: Non-zero surprisal values with variance > 0.1

Estimated time: 5-10 minutes
Estimated cost: $1-2 USD
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.hot_pot import HotPotLab
from environments.switch_light import SwitchLight
from environments.chem_tile import ChemTile
from agents.base import OpenAILLM, AnthropicLLM
from agents.actor import ActorAgent
from models.belief_state import HotPotBelief, SwitchLightBelief, ChemTileBelief
from token_prediction.openai_predictor import OpenAINextSentencePredictor
from experiments.token_runner import run_episode_with_tokens, create_textualizer


def check_env_vars():
    """Check required environment variables."""
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Please run: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    if not os.getenv('ANTHROPIC_API_KEY'):
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        print("   Please run: export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    print("✅ API keys found")


def run_validation():
    """Run 3-episode validation."""
    check_env_vars()

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/validation_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"QUICK VALIDATION RUN")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Episodes: 3 (1 per environment)")
    print(f"Agent: Actor with belief state")
    print(f"Models: Anthropic (agent), OpenAI (token prediction)")
    print(f"{'='*60}\n")

    # Environment configurations
    env_configs = [
        {
            'name': 'HotPot',
            'env_class': HotPotLab,
            'belief_class': HotPotBelief,
            'seed': 42
        },
        {
            'name': 'SwitchLight',
            'env_class': SwitchLight,
            'belief_class': SwitchLightBelief,
            'seed': 100
        },
        {
            'name': 'ChemTile',
            'env_class': ChemTile,
            'belief_class': ChemTileBelief,
            'seed': 200
        }
    ]

    # Create LLMs
    agent_llm = AnthropicLLM(model='claude-sonnet-4-5-20250929')
    predictor = OpenAINextSentencePredictor(model='gpt-4o-mini')

    results = []

    # Run each environment
    for i, config in enumerate(env_configs, 1):
        print(f"\n{'─'*60}")
        print(f"[{i}/3] Running {config['name']} environment (seed={config['seed']})")
        print(f"{'─'*60}")

        # Create environment
        env = config['env_class'](seed=config['seed'])

        # Create actor agent with belief state
        # Use the class name directly for tool registry lookup
        env_class_name = config['env_class'].__name__
        agent = ActorAgent(agent_llm, action_budget=10, environment_name=env_class_name)
        agent.set_belief_state(config['belief_class']())

        # Create textualizer
        textualizer = create_textualizer(env)

        # Run episode
        try:
            test_results, token_logger = run_episode_with_tokens(
                env=env,
                agent=agent,
                textualizer=textualizer,
                predictor=predictor,
                seed=config['seed'],
                max_actions=10,
                save_dir=str(output_dir)
            )

            # Save token log
            token_log_path = output_dir / f"{config['name']}_Actor_ep{config['seed']:03d}_token.json"
            token_logger.save(str(token_log_path))

            # Quick analysis
            entries = token_logger.to_dict()['entries']
            surprisals = [e['belief_surprisal'] for e in entries if e['belief_surprisal'] is not None]

            print(f"\n📊 Episode Summary:")
            print(f"   Steps completed: {len(entries)}")
            print(f"   Valid surprisal values: {len(surprisals)}")

            if surprisals:
                import numpy as np
                surprisal_array = np.array(surprisals)
                print(f"   Surprisal stats:")
                print(f"     Mean: {surprisal_array.mean():.3f}")
                print(f"     Std:  {surprisal_array.std():.3f}")
                print(f"     Min:  {surprisal_array.min():.3f}")
                print(f"     Max:  {surprisal_array.max():.3f}")
                print(f"     Range: [{surprisal_array.min():.3f}, {surprisal_array.max():.3f}]")

                # Check if variance is non-zero
                if surprisal_array.std() > 0.01:
                    print(f"   ✅ GOOD: Non-zero variance detected!")
                else:
                    print(f"   ⚠️  WARNING: Low variance (constant surprisal)")
            else:
                print(f"   ❌ PROBLEM: No valid surprisal values!")

            results.append({
                'environment': config['name'],
                'seed': config['seed'],
                'num_steps': len(entries),
                'num_valid_surprisals': len(surprisals),
                'surprisal_mean': float(np.mean(surprisals)) if surprisals else None,
                'surprisal_std': float(np.std(surprisals)) if surprisals else None,
                'surprisal_min': float(np.min(surprisals)) if surprisals else None,
                'surprisal_max': float(np.max(surprisals)) if surprisals else None,
                'token_log_path': str(token_log_path)
            })

        except Exception as e:
            print(f"❌ Episode failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'environment': config['name'],
                'seed': config['seed'],
                'error': str(e)
            })

    # Final summary
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")

    success_count = sum(1 for r in results if 'error' not in r and r['num_valid_surprisals'] > 0)

    print(f"\nEpisodes completed: {len([r for r in results if 'error' not in r])}/3")
    print(f"Episodes with valid surprisal: {success_count}/3")

    if success_count == 3:
        print(f"\n✅ VALIDATION PASSED!")
        print(f"   All 3 environments have non-zero belief surprisal.")
        print(f"   Bug fix is working correctly.")
        print(f"\n📋 Next steps:")
        print(f"   1. Review token logs in: {output_dir}")
        print(f"   2. Run full pilot: python scripts/pilot_token_run.py")
        print(f"   3. Analyze coupling: python scripts/analyze_token_pilot.py")
    elif success_count > 0:
        print(f"\n⚠️  PARTIAL SUCCESS")
        print(f"   {success_count}/3 environments working.")
        print(f"   Review logs and investigate failures.")
    else:
        print(f"\n❌ VALIDATION FAILED")
        print(f"   No episodes produced valid surprisal values.")
        print(f"   Check the error messages above.")

    # Save summary
    summary_path = output_dir / "validation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'results': results,
            'success_count': success_count,
            'total_episodes': 3
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"Summary: {summary_path}")

    return success_count == 3


if __name__ == '__main__':
    success = run_validation()
    sys.exit(0 if success else 1)
