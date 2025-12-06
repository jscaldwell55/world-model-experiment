#!/usr/bin/env python3
"""
Generate Training Data from Playbook Observations

CLI script to convert playbook observations into instruction/response training pairs
suitable for LoRA fine-tuning.

Usage:
    python scripts/generate_training_data.py --min-reliability HIGH --output data/training_pairs.json
    python scripts/generate_training_data.py --min-reliability MEDIUM --stats-only
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.training_data import TrainingDataGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate training pairs from playbook observations"
    )
    parser.add_argument(
        '--min-reliability',
        choices=['HIGH', 'MEDIUM', 'LOW'],
        default='HIGH',
        help='Minimum reliability level for filtering (default: HIGH)'
    )
    parser.add_argument(
        '--playbook-base',
        default='memory/domains',
        help='Base path for playbook files (default: memory/domains)'
    )
    parser.add_argument(
        '--output',
        default='data/training_pairs.json',
        help='Output file path (default: data/training_pairs.json)'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'jsonl'],
        default='json',
        help='Output format (default: json)'
    )
    parser.add_argument(
        '--no-dedup',
        action='store_true',
        help='Disable deduplication of similar pairs'
    )
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only print statistics, do not save file'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=3,
        help='Number of sample pairs to display per domain (default: 3)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TRAINING DATA GENERATION")
    print("=" * 70)
    print(f"Playbook base: {args.playbook_base}")
    print(f"Min reliability: {args.min_reliability}")
    print(f"Deduplication: {'disabled' if args.no_dedup else 'enabled'}")
    print(f"Output: {args.output}")
    print("=" * 70)

    # Initialize generator
    generator = TrainingDataGenerator(args.playbook_base)

    # Generate pairs
    print("\nGenerating training pairs...")
    pairs, stats = generator.generate_all_pairs(
        min_reliability=args.min_reliability,
        deduplicate=not args.no_dedup
    )

    # Print statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"\nTotal observations: {stats['total_observations']}")
    print(f"Filtered observations ({args.min_reliability}+): {stats['filtered_observations']}")
    print(f"Training pairs generated: {stats['pairs_generated']}")
    print(f"Training pairs after dedup: {stats['pairs_after_dedup']}")

    print("\nBy domain:")
    for domain, domain_stats in stats['by_domain'].items():
        print(f"  {domain}:")
        print(f"    Observations: {domain_stats['observations']}")
        print(f"    Filtered: {domain_stats['filtered']}")
        print(f"    Pairs: {domain_stats['pairs']}")

    # Print samples
    if args.samples > 0:
        print("\n" + "=" * 70)
        print("SAMPLE TRAINING PAIRS")
        print("=" * 70)

        for domain in ['hot_pot', 'chem_tile', 'switch_light']:
            domain_pairs = [p for p in pairs if p.domain == domain]
            if domain_pairs:
                print(f"\n--- {domain.upper()} ---")
                for i, pair in enumerate(domain_pairs[:args.samples]):
                    print(f"\n[{i+1}] Instruction: {pair.instruction[:100]}...")
                    print(f"    Response: {pair.response[:100]}...")
                    print(f"    Reliability: {pair.reliability}, Score: {pair.score:.2f}")

    # Check target
    target_min = 200
    target_max = 1000
    if len(pairs) < target_min:
        print(f"\n⚠️  WARNING: Only {len(pairs)} pairs generated, target is {target_min}+")
    elif len(pairs) >= target_max:
        print(f"\n✅ Generated {len(pairs)} pairs (above target of {target_min}-{target_max})")
    else:
        print(f"\n✅ Generated {len(pairs)} pairs (within target range {target_min}-{target_max})")

    # Save if not stats-only
    if not args.stats_only:
        print(f"\nSaving to {args.output}...")
        generator.save_pairs(pairs, args.output, args.format)
        print(f"✅ Saved {len(pairs)} training pairs to {args.output}")
    else:
        print("\n(Stats only - not saving file)")

    # Print summary for graduation assessment
    print("\n" + "=" * 70)
    print("GRADUATION READINESS")
    print("=" * 70)

    high_pairs = [p for p in pairs if p.reliability in ['HIGH', 'SYNTHETIC_HIGH']]
    print(f"HIGH reliability pairs: {len(high_pairs)}")
    print(f"Total pairs: {len(pairs)}")

    if len(high_pairs) >= 200:
        print("✅ Ready for graduation POC (200+ HIGH reliability pairs)")
    else:
        needed = 200 - len(high_pairs)
        print(f"❌ Need {needed} more HIGH reliability pairs for graduation")

    return 0


if __name__ == '__main__':
    sys.exit(main())
