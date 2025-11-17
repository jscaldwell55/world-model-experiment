#!/usr/bin/env python3
"""
Analyze 9-episode validation experiment results
Focus: Did memory persistence fix enable learning curves?
"""

import json
import os
from pathlib import Path
from collections import defaultdict

results_dir = Path("results/memory_validation_9ep/raw")
memory_dir = Path("memory/domains")

print("=" * 80)
print("9-EPISODE MEMORY VALIDATION ANALYSIS")
print("=" * 80)

# Parse results by environment and episode
episodes_by_env = defaultdict(list)

for file in sorted(results_dir.glob("*.json")):
    with open(file, 'r') as f:
        data = json.load(f)

    env = data['environment']
    ep_num = int(file.stem.split('_ep')[-1])

    # Extract test results
    test_results = data.get('test_results', [])
    total_questions = len(test_results)
    correct = sum(1 for q in test_results if q.get('correct', False))
    scores = [q.get('score', 0) for q in test_results]
    score = sum(scores) / len(scores) if scores else 0

    # Normalize environment name
    env_map = {
        'HotPotLab': 'hot_pot',
        'SwitchLight': 'switch_light',
        'ChemTile': 'chem_tile'
    }
    env = env_map.get(env, env.lower().replace(' ', '_'))

    # Extract cost
    cost_data = data.get('cost', {})
    if isinstance(cost_data, dict):
        cost = cost_data.get('total_cost_usd', 0)
    else:
        cost = cost_data

    episodes_by_env[env].append({
        'episode': ep_num,
        'score': score,
        'correct': correct,
        'total': total_questions,
        'accuracy': (correct / max(total_questions, 1)) * 100 if total_questions > 0 else 0,
        'actions': len(data.get('steps', [])),
        'cost': cost
    })

# Analyze learning curves per environment
print("\n📊 LEARNING CURVES BY ENVIRONMENT")
print("=" * 80)

overall_improvement = {}

for env in ['hot_pot', 'switch_light', 'chem_tile']:
    if env not in episodes_by_env:
        continue

    episodes = sorted(episodes_by_env[env], key=lambda x: x['episode'])

    print(f"\n{env.upper().replace('_', ' ')}")
    print("-" * 80)
    print(f"{'Episode':<10} {'Score':<10} {'Accuracy':<12} {'Actions':<10} {'Cost':<10}")
    print("-" * 80)

    for ep in episodes:
        print(f"{ep['episode']:<10} {ep['score']:<10.1f} {ep['accuracy']:<12.1f}% {ep['actions']:<10} ${ep['cost']:.2f}")

    # Calculate improvement
    if len(episodes) >= 2:
        ep1_score = episodes[0]['score']
        ep3_score = episodes[-1]['score']
        improvement = ep3_score - ep1_score
        pct_improvement = (improvement / max(ep1_score, 1)) * 100

        print("-" * 80)
        print(f"Episode 1 → {len(episodes)}: {ep1_score:.1f} → {ep3_score:.1f} ({improvement:+.1f}, {pct_improvement:+.1f}%)")

        overall_improvement[env] = {
            'ep1': ep1_score,
            'ep_final': ep3_score,
            'change': improvement,
            'pct_change': pct_improvement
        }

# Overall summary
print("\n\n📈 LEARNING TREND SUMMARY")
print("=" * 80)

total_environments = len(overall_improvement)
improving = sum(1 for v in overall_improvement.values() if v['change'] > 0)
declining = sum(1 for v in overall_improvement.values() if v['change'] < 0)
flat = total_environments - improving - declining

print(f"Improving:  {improving}/{total_environments} environments")
print(f"Declining:  {declining}/{total_environments} environments")
print(f"Flat:       {flat}/{total_environments} environments")

# Check memory persistence
print("\n\n💾 MEMORY PERSISTENCE CHECK")
print("=" * 80)

for env in ['hot_pot', 'switch_light', 'chem_tile']:
    consolidated_file = memory_dir / env / "consolidated" / "beliefs.json"

    if consolidated_file.exists():
        with open(consolidated_file, 'r') as f:
            beliefs = json.load(f)

        # Extract confidence and episode_count
        confidences = []
        episode_counts = []

        for key, value in beliefs.items():
            if isinstance(value, dict) and 'confidence' in value:
                confidences.append(value['confidence'])
                episode_counts.append(value.get('episode_count', 0))

        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            max_count = max(episode_counts) if episode_counts else 0

            print(f"\n{env}:")
            print(f"  Beliefs tracked: {len(confidences)}")
            print(f"  Avg confidence: {avg_conf:.3f}")
            print(f"  Max episode count: {max_count}")
            print(f"  Status: {'✅ Working' if max_count >= 3 else '⚠️ Not accumulating'}")
        else:
            print(f"\n{env}: ⚠️ No structured beliefs found")
    else:
        print(f"\n{env}: ❌ No consolidated beliefs file")

# Verdict
print("\n\n🎯 VERDICT")
print("=" * 80)

if improving >= 2:
    print("✅ SUCCESS: Memory persistence is working!")
    print(f"   {improving}/{total_environments} environments show improvement")
    print("   Learning curves are upward trending")
elif improving >= 1:
    print("⚠️ PARTIAL: Some improvement detected")
    print(f"   {improving}/{total_environments} environments show improvement")
    print("   May need more episodes to confirm trend")
else:
    print("❌ ISSUE: No clear improvement detected")
    print("   Learning curves still flat - further debugging needed")

print("\n" + "=" * 80)
