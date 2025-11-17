#!/usr/bin/env python3
"""
Demo: Offline Consolidation System

Shows how to use the OC layer with ACE playbooks to prepare data for fine-tuning.
"""

import json
from pathlib import Path
from memory.offline_consolidation import OfflineConsolidation
from memory.ace_playbook import ACEPlaybook
from environments.hot_pot import HotPotLab


def demo_basic_usage():
    """Demonstrate basic OC usage with a playbook"""
    print("="*70)
    print("DEMO: Basic Offline Consolidation Usage")
    print("="*70)

    # Load ACE playbook for hot_pot domain
    playbook_path = Path('memory/domains/hot_pot/playbook.json')

    if not playbook_path.exists():
        print("\n⚠️  No playbook found. Run 30-episode validation first.")
        return

    print(f"\n1. Loading ACE playbook from {playbook_path}")
    with open(playbook_path, 'r') as f:
        playbook = json.load(f)

    print(f"   Loaded {len(playbook.get('observations', []))} observations")

    # Create environment instance
    print(f"\n2. Creating environment instance")
    env = HotPotLab(seed=42)

    # Initialize OC system
    print(f"\n3. Initializing Offline Consolidation")
    oc = OfflineConsolidation(env)

    # Run consolidation pipeline
    print(f"\n4. Running consolidation pipeline...")
    print(f"   {'-'*66}")
    consolidated = oc.consolidate(playbook)
    print(f"   {'-'*66}")

    # Examine results
    print(f"\n5. Results:")
    print(f"   HIGH reliability episodes: {len(consolidated.high_reliability_episodes)}")
    print(f"   LOW reliability episodes: {len(consolidated.low_reliability_episodes)}")
    print(f"   Synthetic episodes: {len(consolidated.counterfactual_episodes)}")

    # Check quality gate
    print(f"\n6. Quality Gate Decision: {consolidated.gate_status}")
    print(f"   Reason: {consolidated.gate_reason}")

    if consolidated.recommendations:
        print(f"\n   Recommendations:")
        for rec in consolidated.recommendations:
            print(f"     • {rec}")

    # Get training data
    print(f"\n7. Preparing for Fine-Tuning Bridge...")
    training_data = consolidated.get_training_data()

    print(f"   Total episodes: {len(training_data['episodes'])}")
    print(f"   Episode weights: {training_data['weights']}")
    print(f"   Metadata:")
    for key, value in training_data['metadata'].items():
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                print(f"     {key}: {value:.3f}")
            else:
                print(f"     {key}: {value}")

    # Show bias report
    if consolidated.bias_report:
        print(f"\n8. Bias Report:")
        print(f"   {'-'*66}")
        bias_str = str(consolidated.bias_report)
        for line in bias_str.split('\n'):
            print(f"   {line}")
        print(f"   {'-'*66}")

    print(f"\n{'='*70}")
    print("Demo complete!")
    print(f"{'='*70}\n")


def demo_all_domains():
    """Process all domains and compare results"""
    print("="*70)
    print("DEMO: Multi-Domain Consolidation")
    print("="*70)

    domains = ['hot_pot', 'chem_tile', 'switch_light']
    results = {}

    for domain in domains:
        print(f"\n{'-'*70}")
        print(f"Processing: {domain.upper()}")
        print(f"{'-'*70}")

        playbook_path = Path(f'memory/domains/{domain}/playbook.json')

        if not playbook_path.exists():
            print(f"⚠️  No playbook found")
            continue

        with open(playbook_path, 'r') as f:
            playbook = json.load(f)

        env = HotPotLab(seed=42)
        oc = OfflineConsolidation(env)

        # Run quietly
        consolidated = oc.consolidate(playbook)

        # Store results
        results[domain] = {
            'gate_status': consolidated.gate_status,
            'high_reliability': len(consolidated.high_reliability_episodes),
            'synthetic': len(consolidated.counterfactual_episodes),
            'low_reliability': len(consolidated.low_reliability_episodes),
            'total': len(consolidated.high_reliability_episodes) +
                     len(consolidated.counterfactual_episodes) +
                     len(consolidated.low_reliability_episodes)
        }

    # Compare results
    print(f"\n{'='*70}")
    print("COMPARISON ACROSS DOMAINS")
    print(f"{'='*70}")

    print(f"\n{'Domain':<15} {'Gate':<10} {'HIGH':<6} {'Synth':<6} {'LOW':<6} {'Total':<6}")
    print(f"{'-'*70}")

    for domain, result in results.items():
        gate_symbol = {
            'PASS': '✓',
            'WARNING': '⚠️',
            'FAIL': '✗'
        }.get(result['gate_status'], '?')

        print(
            f"{domain:<15} "
            f"{gate_symbol} {result['gate_status']:<8} "
            f"{result['high_reliability']:<6} "
            f"{result['synthetic']:<6} "
            f"{result['low_reliability']:<6} "
            f"{result['total']:<6}"
        )

    print(f"\n{'='*70}\n")


def demo_custom_thresholds():
    """Show how to customize quality gate thresholds"""
    print("="*70)
    print("DEMO: Custom Quality Gate Thresholds")
    print("="*70)

    playbook_path = Path('memory/domains/hot_pot/playbook.json')

    if not playbook_path.exists():
        print("\n⚠️  No playbook found")
        return

    with open(playbook_path, 'r') as f:
        playbook = json.load(f)

    env = HotPotLab(seed=42)

    # Default thresholds
    print(f"\n1. Default Thresholds:")
    oc_default = OfflineConsolidation(env)
    print(f"   Fidelity threshold (fail): {oc_default.fidelity_threshold_fail}")
    print(f"   Fidelity threshold (warning): {oc_default.fidelity_threshold_warning}")
    print(f"   Min HIGH reliability (fail): {oc_default.min_high_reliability_pct_fail*100:.0f}%")
    print(f"   Min HIGH reliability (warning): {oc_default.min_high_reliability_pct_warning*100:.0f}%")

    consolidated_default = oc_default.consolidate(playbook)
    print(f"\n   Result: {consolidated_default.gate_status}")

    # Stricter thresholds
    print(f"\n2. Stricter Thresholds:")
    oc_strict = OfflineConsolidation(env)
    oc_strict.fidelity_threshold_fail = 0.8
    oc_strict.fidelity_threshold_warning = 0.9
    oc_strict.min_high_reliability_pct_fail = 0.30
    oc_strict.min_high_reliability_pct_warning = 0.40

    print(f"   Fidelity threshold (fail): {oc_strict.fidelity_threshold_fail}")
    print(f"   Fidelity threshold (warning): {oc_strict.fidelity_threshold_warning}")
    print(f"   Min HIGH reliability (fail): {oc_strict.min_high_reliability_pct_fail*100:.0f}%")
    print(f"   Min HIGH reliability (warning): {oc_strict.min_high_reliability_pct_warning*100:.0f}%")

    consolidated_strict = oc_strict.consolidate(playbook)
    print(f"\n   Result: {consolidated_strict.gate_status}")

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    import sys

    demos = {
        '1': ('Basic Usage', demo_basic_usage),
        '2': ('All Domains', demo_all_domains),
        '3': ('Custom Thresholds', demo_custom_thresholds),
    }

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("Available demos:")
        for key, (name, _) in demos.items():
            print(f"  {key}. {name}")
        print(f"\nUsage: python {sys.argv[0]} [demo_number]")
        print(f"Running all demos...\n")
        choice = 'all'

    if choice == 'all':
        for _, demo_func in demos.values():
            demo_func()
            print()
    elif choice in demos:
        _, demo_func = demos[choice]
        demo_func()
    else:
        print(f"Invalid choice: {choice}")
        sys.exit(1)
