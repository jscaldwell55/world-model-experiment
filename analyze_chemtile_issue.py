#!/usr/bin/env python3
"""
Investigate why ChemTile is using only 3.3 actions on average
This is suspiciously low and suggests:
1. Agent solving too quickly without exploration
2. Missing learning opportunities
3. Plateauing at 90% because it never explores edge cases
"""

import json
from pathlib import Path

results_dir = Path("results/memory_validation_9ep/raw")

print("=" * 80)
print("CHEMTILE LOW ACTION COUNT INVESTIGATION")
print("=" * 80)

for file in sorted(results_dir.glob("chem_tile*.json")):
    with open(file, 'r') as f:
        data = json.load(f)

    episode_num = int(file.stem.split('_ep')[-1])

    print(f"\nEPISODE {episode_num}")
    print("-" * 80)

    steps = data.get('steps', [])
    test_results = data.get('test_results', [])

    print(f"Total actions: {len(steps)}")
    print(f"Test results: {len(test_results)} questions")

    # Show all actions taken
    print("\nActions taken:")
    for i, step in enumerate(steps):
        action = step.get('action', '')
        observation = step.get('observation', {})
        print(f"  {i+1}. {action}")
        if observation:
            obs_str = str(observation)[:100]
            print(f"     → {obs_str}...")

    # Analyze test performance
    correct = sum(1 for q in test_results if q.get('correct', False))
    total = len(test_results)
    accuracy = (correct / total * 100) if total > 0 else 0

    print(f"\nTest Performance: {correct}/{total} correct ({accuracy:.0f}%)")

    # Check if agent is terminating early
    action_budget = 10
    if len(steps) < action_budget:
        print(f"\n⚠️ WARNING: Agent used only {len(steps)}/{action_budget} actions!")
        print(f"   This suggests:")
        print(f"   - Agent is terminating early (solving too fast?)")
        print(f"   - Not exploring enough to learn edge cases")
        print(f"   - May miss important observations about reaction dynamics")

    # Analyze question difficulty
    print("\nQuestion breakdown:")
    difficulties = {}
    for q in test_results:
        diff = q.get('difficulty', 'unknown')
        correct = q.get('correct', False)

        if diff not in difficulties:
            difficulties[diff] = {'correct': 0, 'total': 0}

        difficulties[diff]['total'] += 1
        if correct:
            difficulties[diff]['correct'] += 1

    for diff, stats in difficulties.items():
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {diff}: {stats['correct']}/{stats['total']} ({acc:.0f}%)")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

print("\nChemTile low action count patterns:")
print("  - Episodes 1-3 all use 2-4 actions (avg 3.3)")
print("  - Action budget is 10, but agent stops early")
print("  - Performance plateaus at 90% across all episodes")
print()
print("Root causes:")
print("  1. Agent reaches solution quickly and stops exploring")
print("  2. No incentive to continue after answering test questions")
print("  3. Missing observations about:")
print("     - Different reaction combinations")
print("     - Temperature effects")
print("     - Explosion probabilities")
print("     - Edge cases")
print()
print("Impact on learning:")
print("  - Limited observations = low confidence")
print("  - Observation minimum penalty will apply (< 8 obs)")
print("  - Agent won't improve beyond 90% without more data")
print()
print("Recommended fix:")
print("  Add exploration bonus for first N episodes:")
print("    - Episode 1-3: Require minimum 8 actions")
print("    - Encourages exploration even after finding solution")
print("    - Gathers more observations for better world model")

print("\n" + "=" * 80)
