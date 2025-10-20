#!/usr/bin/env python3
"""Episode inspection tool"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('episode_file', help='Episode JSON file')
    args = parser.parse_args()
    
    with open(args.episode_file) as f:
        ep = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"Episode: {ep.get('episode_id', 'unknown')}")
    print(f"{'='*70}")
    print(f"Environment: {ep.get('environment', 'unknown')}")
    print(f"Agent: {ep.get('agent_type', 'unknown')}")
    print(f"Seed: {ep.get('seed', 0)}")
    
    print(f"\n{'='*70}")
    print(f"STEPS ({len(ep.get('steps', []))} total)")
    print(f"{'='*70}")
    for i, step in enumerate(ep.get('steps', [])[:10]):  # Show first 10
        action = step.get('action', 'None')
        surprisal = step.get('surprisal', 0.0)
        print(f"{i}: {action} (surprisal={surprisal:.2f})")
    
    print(f"\n{'='*70}")
    print(f"TEST RESULTS")
    print(f"{'='*70}")
    correct = sum(1 for r in ep.get('test_results', []) if r.get('correct'))
    total = len(ep.get('test_results', []))
    print(f"Accuracy: {correct}/{total} ({100*correct/total if total > 0 else 0:.1f}%)")
    
    for r in ep.get('test_results', [])[:5]:  # Show first 5
        status = "✓" if r.get('correct') else "✗"
        print(f"{status} {r['query'][:60]}...")
        print(f"   Answer: {r['agent_answer'][:60]}...")

if __name__ == '__main__':
    main()
