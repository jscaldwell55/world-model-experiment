#!/usr/bin/env python3
"""
Analyze Synthetic Episode Distribution

Checks:
- How many synthetic vs real episodes
- Context distribution in synthetics
- Fidelity score distribution
- Provenance tracking
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_synthetics(domain='hot_pot'):
    """Analyze synthetic episodes in a domain"""

    print(f"\n{'='*70}")
    print(f"SYNTHETIC EPISODE ANALYSIS: {domain.upper()}")
    print(f"{'='*70}")

    episodes_dir = Path(f"memory/domains/{domain}/episodes")

    if not episodes_dir.exists():
        print(f"✗ Episodes directory not found: {episodes_dir}")
        return

    # Load all episodes
    real_episodes = []
    synthetic_episodes = []
    all_episodes = []

    for episode_file in episodes_dir.glob("*.json"):
        try:
            with open(episode_file, 'r') as f:
                episode = json.load(f)
                all_episodes.append(episode)

                if episode.get('is_synthetic', False):
                    synthetic_episodes.append(episode)
                else:
                    real_episodes.append(episode)

        except Exception as e:
            print(f"⚠️  Failed to load {episode_file.name}: {e}")

    # Basic counts
    print(f"\n📊 EPISODE COUNTS:")
    print(f"  Total episodes:     {len(all_episodes)}")
    print(f"  Real episodes:      {len(real_episodes)}")
    print(f"  Synthetic episodes: {len(synthetic_episodes)}")

    if len(synthetic_episodes) == 0:
        print("\n⚠️  No synthetic episodes found")
        return

    synthetic_pct = 100 * len(synthetic_episodes) / len(all_episodes) if all_episodes else 0
    print(f"  Synthetic ratio:    {synthetic_pct:.1f}%")

    # Context distribution in synthetics
    print(f"\n🌍 SYNTHETIC CONTEXT DISTRIBUTION:")
    context_counts = defaultdict(int)

    for syn in synthetic_episodes:
        context = syn.get('context', {})
        if isinstance(context, dict):
            # Extract key context variable (domain-specific)
            if 'power_setting' in context:
                context_key = context['power_setting']
            elif 'switch_id' in context:
                context_key = f"{context['switch_id']}/{context.get('effectiveness', 'normal')}"
            elif 'tile_type' in context:
                context_key = context['tile_type']
            else:
                context_key = str(context)
        else:
            context_key = str(context)

        context_counts[context_key] += 1

    for context, count in sorted(context_counts.items()):
        pct = 100 * count / len(synthetic_episodes)
        print(f"  {context:20s}: {count:3d} ({pct:5.1f}%)")

    # Fidelity scores
    print(f"\n🎯 FIDELITY SCORES:")
    fidelity_scores = [
        syn.get('fidelity_score', 0.0)
        for syn in synthetic_episodes
        if 'fidelity_score' in syn
    ]

    if fidelity_scores:
        print(f"  Mean:     {np.mean(fidelity_scores):.3f}")
        print(f"  Median:   {np.median(fidelity_scores):.3f}")
        print(f"  Min:      {np.min(fidelity_scores):.3f}")
        print(f"  Max:      {np.max(fidelity_scores):.3f}")
        print(f"  Std Dev:  {np.std(fidelity_scores):.3f}")

        # Distribution
        high_fidelity = sum(1 for f in fidelity_scores if f >= 0.9)
        medium_fidelity = sum(1 for f in fidelity_scores if 0.7 <= f < 0.9)
        low_fidelity = sum(1 for f in fidelity_scores if f < 0.7)

        print(f"\n  Distribution:")
        print(f"    High (≥0.9):     {high_fidelity} ({100*high_fidelity/len(fidelity_scores):.1f}%)")
        print(f"    Medium (0.7-0.9): {medium_fidelity} ({100*medium_fidelity/len(fidelity_scores):.1f}%)")
        print(f"    Low (<0.7):       {low_fidelity} ({100*low_fidelity/len(fidelity_scores):.1f}%)")
    else:
        print(f"  ⚠️  No fidelity scores found")

    # FTB version distribution
    print(f"\n📦 FTB VERSION DISTRIBUTION:")
    version_counts = defaultdict(int)
    for syn in synthetic_episodes:
        version = syn.get('ftb_version', 'unknown')
        version_counts[version] += 1

    for version, count in sorted(version_counts.items()):
        pct = 100 * count / len(synthetic_episodes)
        print(f"  {version:10s}: {count:3d} ({pct:5.1f}%)")

    # Reliability distribution
    print(f"\n⭐ RELIABILITY DISTRIBUTION:")
    reliability_counts = defaultdict(int)
    for syn in synthetic_episodes:
        reliability = syn.get('reliability', 'UNKNOWN')
        reliability_counts[reliability] += 1

    for reliability, count in sorted(reliability_counts.items()):
        pct = 100 * count / len(synthetic_episodes)
        print(f"  {reliability:20s}: {count:3d} ({pct:5.1f}%)")

    # Provenance tracking (v1 feature)
    print(f"\n🔗 PROVENANCE TRACKING:")
    with_parents = sum(1 for syn in synthetic_episodes if syn.get('parent_episode_ids'))
    with_generation_method = sum(1 for syn in synthetic_episodes if syn.get('generation_method'))

    print(f"  With parent IDs:       {with_parents}/{len(synthetic_episodes)} ({100*with_parents/len(synthetic_episodes):.1f}%)")
    print(f"  With generation method: {with_generation_method}/{len(synthetic_episodes)} ({100*with_generation_method/len(synthetic_episodes):.1f}%)")

    # Sample synthetic episode
    print(f"\n📄 SAMPLE SYNTHETIC EPISODE:")
    if synthetic_episodes:
        sample = synthetic_episodes[0]
        print(f"  Episode ID:       {sample.get('episode_id', 'N/A')}")
        print(f"  Context:          {sample.get('context', {})}")
        print(f"  Fidelity:         {sample.get('fidelity_score', 'N/A')}")
        print(f"  Reliability:      {sample.get('reliability', 'N/A')}")
        print(f"  FTB Version:      {sample.get('ftb_version', 'N/A')}")
        print(f"  Generated at:     {sample.get('generated_at', 'N/A')}")
        print(f"  Observations:     {len(sample.get('observations', []))}")
        print(f"  Parent IDs:       {sample.get('parent_episode_ids', [])}")

    # Check for issues
    print(f"\n⚠️  VALIDATION CHECKS:")

    issues = []

    # Check 1: All synthetics should have is_synthetic=True
    missing_flag = sum(1 for syn in synthetic_episodes if not syn.get('is_synthetic'))
    if missing_flag > 0:
        issues.append(f"{missing_flag} synthetics missing is_synthetic flag")

    # Check 2: All synthetics should have fidelity scores
    missing_fidelity = sum(1 for syn in synthetic_episodes if 'fidelity_score' not in syn)
    if missing_fidelity > 0:
        issues.append(f"{missing_fidelity} synthetics missing fidelity_score")

    # Check 3: Fidelity should be reasonable (0-1 range)
    invalid_fidelity = sum(1 for f in fidelity_scores if not (0.0 <= f <= 1.0))
    if invalid_fidelity > 0:
        issues.append(f"{invalid_fidelity} synthetics with invalid fidelity scores")

    # Check 4: Context distribution should be balanced (no single context >80%)
    if context_counts:
        max_context_pct = max(context_counts.values()) / len(synthetic_episodes)
        if max_context_pct > 0.8:
            issues.append(f"Context imbalance: one context has {max_context_pct:.1%} of synthetics")

    if issues:
        for issue in issues:
            print(f"  ✗ {issue}")
    else:
        print(f"  ✓ All validation checks passed")

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    import sys

    # Get domain from command line or use default
    domain = sys.argv[1] if len(sys.argv) > 1 else 'hot_pot'

    analyze_synthetics(domain)
