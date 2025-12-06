#!/usr/bin/env python3
"""
Run Phase 1 Experiment

This script runs the full Phase 1 experiment comparing:
1. Static Model + Random Policy (baseline)
2. Static Model + Uncertainty Sampling (baseline)
3. Online Updates + Uncertainty Sampling (no OC gate)
4. Full Stack (OC + FTB) + Uncertainty Sampling
5. Oracle Upper Bound

Usage:
    python experiments/run_phase1.py [--seeds N] [--budget B] [--quick]
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.phase1_experiment import Phase1Experiment, ExperimentConfig


def run_quick_test():
    """Run a quick test with minimal settings."""
    print("Running quick test (2 seeds, 20 queries)...")

    config = ExperimentConfig(
        n_seeds=2,
        query_budget=20,
        seed_size=10,
        consolidation_interval=5,
        n_estimators=30,
        checkpoint_steps=[5, 10, 15, 20],
        output_dir='results/phase1_quick'
    )

    experiment = Phase1Experiment(config)
    experiment.run_all(verbose=True)
    experiment.save_results()
    experiment.print_summary()
    experiment.generate_report()

    return experiment


def run_full_experiment(n_seeds: int = 5, query_budget: int = 50):
    """Run the full experiment."""
    print(f"Running full experiment ({n_seeds} seeds, {query_budget} queries)...")

    config = ExperimentConfig(
        n_seeds=n_seeds,
        query_budget=query_budget,
        seed_size=10,
        consolidation_interval=10,
        n_estimators=50,
        checkpoint_steps=[10, 20, 30, 40, 50],
        output_dir='results/phase1'
    )

    experiment = Phase1Experiment(config)
    experiment.run_all(verbose=True)
    experiment.save_results()
    experiment.print_summary()
    experiment.generate_report()

    # Generate plots
    try:
        from experiments.visualize_results import generate_all_plots
        generate_all_plots(f'{config.output_dir}/results.json')
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    return experiment


def main():
    parser = argparse.ArgumentParser(description='Run Phase 1 Experiment')
    parser.add_argument('--seeds', type=int, default=5,
                       help='Number of random seeds (default: 5)')
    parser.add_argument('--budget', type=int, default=50,
                       help='Query budget per episode (default: 50)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test instead of full experiment')

    args = parser.parse_args()

    if args.quick:
        run_quick_test()
    else:
        run_full_experiment(args.seeds, args.budget)


if __name__ == '__main__':
    main()
