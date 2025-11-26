#!/usr/bin/env python3
"""
Reliability Distribution Analysis

Investigates why OFF context episodes are marked as LOW reliability
and how this affects cross-validation.

Key finding: 19 OFF context episodes exist but are LOW reliability,
so they're excluded from CV training → model can't learn OFF behavior.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.ace_playbook import ACEPlaybook


def analyze_reliability_distribution():
    """Analyze reliability distribution and its impact on CV"""

    print("=" * 80)
    print("RELIABILITY DISTRIBUTION ANALYSIS")
    print("=" * 80)

    domain = "hot_pot"
    playbook = ACEPlaybook(domain)
    observations = playbook.playbook.get('observations', [])

    print(f"\n[1] Overall Distribution")
    print("-" * 80)
    print(f"Total observations: {len(observations)}")

    # Count by reliability
    reliability_counts = defaultdict(int)
    for obs in observations:
        rel = obs.get('reliability', 'UNKNOWN')
        reliability_counts[rel] += 1

    for rel in ['HIGH', 'MEDIUM', 'LOW', 'SYNTHETIC_HIGH', 'SYNTHETIC_MEDIUM', 'SYNTHETIC_LOW']:
        count = reliability_counts[rel]
        if count > 0:
            pct = 100 * count / len(observations)
            print(f"  {rel:20s}: {count:3d} ({pct:5.1f}%)")

    # Context x Reliability matrix
    print(f"\n[2] Context × Reliability Matrix")
    print("-" * 80)

    matrix = defaultdict(lambda: defaultdict(int))

    for obs in observations:
        context = obs.get('context', {})
        if isinstance(context, dict):
            power_setting = context.get('power_setting', 'UNKNOWN')
        else:
            power_setting = str(context)

        reliability = obs.get('reliability', 'UNKNOWN')
        is_synthetic = obs.get('is_synthetic', False)

        matrix[power_setting][reliability] += 1

    # Print matrix
    contexts = sorted(matrix.keys())
    reliabilities = ['HIGH', 'MEDIUM', 'LOW', 'SYNTHETIC_HIGH', 'SYNTHETIC_MEDIUM', 'SYNTHETIC_LOW']

    # Header
    print(f"{'Context':<10s}", end='')
    for rel in reliabilities:
        print(f"{rel:>12s}", end='')
    print()
    print("-" * 80)

    # Rows
    for context in contexts:
        print(f"{context:<10s}", end='')
        for rel in reliabilities:
            count = matrix[context][rel]
            if count > 0:
                print(f"{count:>12d}", end='')
            else:
                print(f"{'—':>12s}", end='')
        print()

    # Analysis
    print(f"\n[3] Cross-Validation Impact")
    print("-" * 80)

    print(f"\nCV uses ONLY HIGH reliability episodes:")

    for context in contexts:
        high_count = matrix[context]['HIGH']
        total_count = sum(matrix[context].values())

        if high_count >= 3:
            status = "✅ ADEQUATE"
        elif high_count >= 1:
            status = "⚠️  MARGINAL"
        else:
            status = "❌ MISSING"

        print(f"  {context:<10s}: {high_count:2d} HIGH / {total_count:2d} total - {status}")

    # Identify the problem
    print(f"\n[4] ROOT CAUSE ANALYSIS")
    print("=" * 80)

    # Check OFF context specifically
    off_high = matrix['OFF']['HIGH']
    off_low = matrix['OFF']['LOW']
    off_total = sum(matrix['OFF'].values())

    print(f"\nOFF Context Breakdown:")
    print(f"  Total episodes: {off_total}")
    print(f"  HIGH reliability: {off_high}")
    print(f"  LOW reliability: {off_low}")

    if off_low > off_high and off_high == 0:
        print(f"\n🔍 ISSUE IDENTIFIED:")
        print(f"   - OFF context has {off_low} episodes but ALL are LOW reliability")
        print(f"   - CV excludes LOW reliability → OFF context invisible to CV")
        print(f"   - Model cannot learn heating_rate=0 for OFF setting")
        print(f"   - This is why CV might show high errors for OFF predictions")

        print(f"\n💡 WHY ARE THEY LOW RELIABILITY?")
        print(f"   Possible reasons:")
        print(f"   1. Episodes had methodology issues (random actions, no systematicapproach)")
        print(f"   2. Low scores (failed to complete task)")
        print(f"   3. Marked as unreliable by curator")

        # Sample one LOW reliability OFF episode
        off_low_episodes = [
            obs for obs in observations
            if obs.get('context', {}).get('power_setting') == 'OFF'
            and obs.get('reliability') == 'LOW'
        ]

        if off_low_episodes:
            sample = off_low_episodes[0]
            print(f"\n   Sample LOW reliability OFF episode:")
            print(f"   - Episode ID: {sample.get('episode_id', 'N/A')}")
            print(f"   - Score: {sample.get('score', 'N/A')}")
            print(f"   - Reliability reason: {sample.get('reliability_reason', 'N/A')}")

    # Recommendations
    print(f"\n[5] RECOMMENDATIONS")
    print("=" * 80)

    missing_contexts = [ctx for ctx in contexts if matrix[ctx]['HIGH'] == 0]
    marginal_contexts = [ctx for ctx in contexts if 0 < matrix[ctx]['HIGH'] < 3]

    if missing_contexts:
        print(f"\n❌ CRITICAL: {len(missing_contexts)} contexts have 0 HIGH reliability episodes")
        for ctx in missing_contexts:
            total = sum(matrix[ctx].values())
            print(f"   - {ctx}: 0 HIGH / {total} total")
        print(f"\n   Action: Review LOW reliability episodes and consider upgrading")
        print(f"           OR collect new HIGH reliability episodes for these contexts")

    if marginal_contexts:
        print(f"\n⚠️  WARNING: {len(marginal_contexts)} contexts have < 3 HIGH reliability episodes")
        for ctx in marginal_contexts:
            high = matrix[ctx]['HIGH']
            total = sum(matrix[ctx].values())
            print(f"   - {ctx}: {high} HIGH / {total} total")
        print(f"\n   Action: Collect {3 - high} more HIGH reliability episodes per context")

    # Overall recommendation
    total_high = sum(matrix[ctx]['HIGH'] for ctx in contexts)
    if total_high < 20:
        print(f"\n📊 Overall: Only {total_high} HIGH reliability episodes")
        print(f"   - Target: ≥20 for stable CV estimates")
        print(f"   - Need {20 - total_high} more HIGH reliability episodes")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    analyze_reliability_distribution()
