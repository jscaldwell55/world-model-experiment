#!/usr/bin/env python3
"""
Diagnostic analysis of Hybrid agent decision-making.

Investigates:
1. Selection mechanism correctness (is highest score actually selected?)
2. ACTOR score correlation with actual success
3. Candidate diversity
4. Environment-specific failures (especially HotPotLab)
"""

import argparse
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.metrics import compute_all_metrics


def extract_decision_data(episode):
    """Extract all hybrid decisions from episode steps."""
    decisions = []

    for step in episode.get('steps', []):
        belief_state = step.get('belief_state', {})
        hybrid_decision = belief_state.get('hybrid_decision')

        if hybrid_decision and hybrid_decision.get('strategy') == 'hybrid':
            decisions.append({
                'step_num': step['step_num'],
                'action': step.get('action'),
                'selected_idx': hybrid_decision['selected_idx'],
                'selected_score': hybrid_decision['selected_score'],
                'all_scores': hybrid_decision['all_scores'],
                'num_candidates': hybrid_decision['num_candidates'],
                'candidates_summary': hybrid_decision.get('candidates_summary', []),
                'scoring_reasoning': hybrid_decision.get('scoring_reasoning', '')
            })

    return decisions


def check_selection_correctness(decisions):
    """Verify that selected candidate has highest score."""
    issues = []

    for i, dec in enumerate(decisions):
        scores = dec['all_scores']
        selected_idx = dec['selected_idx']
        selected_score = dec['selected_score']
        max_score = max(scores)
        max_idx = scores.index(max_score)

        # Check if selection is correct
        if selected_idx != max_idx:
            issues.append({
                'step': dec['step_num'],
                'selected_idx': selected_idx,
                'selected_score': selected_score,
                'correct_idx': max_idx,
                'correct_score': max_score,
                'all_scores': scores
            })

        # Also check for floating point precision issues
        if abs(selected_score - max_score) > 0.01:
            issues.append({
                'step': dec['step_num'],
                'issue': 'score_mismatch',
                'selected_score': selected_score,
                'max_score': max_score
            })

    return issues


def analyze_score_distribution(decisions):
    """Analyze distribution of ACTOR scores."""
    all_scores = []
    for dec in decisions:
        all_scores.extend(dec['all_scores'])

    return {
        'mean': np.mean(all_scores),
        'std': np.std(all_scores),
        'min': np.min(all_scores),
        'max': np.max(all_scores),
        'q25': np.percentile(all_scores, 25),
        'q50': np.percentile(all_scores, 50),
        'q75': np.percentile(all_scores, 75),
        'total_scores': len(all_scores)
    }


def analyze_candidate_diversity(decisions):
    """Measure diversity of candidates."""
    diversities = []

    for dec in decisions:
        scores = dec['all_scores']
        # Diversity measure: std dev of scores
        diversity = np.std(scores)
        diversities.append({
            'step': dec['step_num'],
            'diversity': diversity,
            'scores': scores,
            'num_candidates': dec['num_candidates']
        })

    return diversities


def analyze_by_environment(results_dir):
    """Analyze decisions grouped by environment."""
    results = defaultdict(lambda: {
        'episodes': [],
        'decisions': [],
        'test_results': [],
        'selection_issues': []
    })

    # Load all episodes
    for ep_file in sorted(results_dir.glob('*.json')):
        if 'config' in ep_file.name:
            continue

        with open(ep_file) as f:
            ep = json.load(f)

        env = ep.get('environment', 'unknown')
        agent_type = ep.get('agent_type', '')

        if agent_type != 'hybrid':
            continue

        # Extract decision data
        decisions = extract_decision_data(ep)

        # Check selection correctness
        issues = check_selection_correctness(decisions)

        results[env]['episodes'].append(ep)
        results[env]['decisions'].extend(decisions)
        results[env]['test_results'].extend(ep.get('test_results', []))
        results[env]['selection_issues'].extend([
            {**issue, 'episode': ep['episode_id']}
            for issue in issues
        ])

    return results


def analyze_planning_failures(env_results, environment_name):
    """Deep dive into planning question failures."""
    failures = []

    for ep in env_results['episodes']:
        ep_id = ep['episode_id']
        decisions = extract_decision_data(ep)

        # Get planning questions for this episode
        for result in ep.get('test_results', []):
            if result['query_type'] == 'planning':
                score = result.get('score', 0.0)
                correct = result.get('correct', False)

                if score < 0.5:  # Failed question
                    failures.append({
                        'episode': ep_id,
                        'question': result['query'],
                        'score': score,
                        'agent_answer': result.get('agent_answer', ''),
                        'confidence': result.get('confidence', 0.0),
                        'num_decisions': len(decisions)
                    })

    return failures


def main():
    parser = argparse.ArgumentParser(description='Diagnose hybrid agent decision-making')
    parser.add_argument('--results', required=True, help='Path to hybrid results directory')
    parser.add_argument('--output', help='Output file for report (default: stdout)')
    args = parser.parse_args()

    results_dir = Path(args.results)

    print("="*80)
    print("HYBRID AGENT DIAGNOSTIC ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing: {results_dir}\n")

    # Analyze by environment
    env_results = analyze_by_environment(results_dir)

    for env_name in sorted(env_results.keys()):
        data = env_results[env_name]

        print("\n" + "="*80)
        print(f"ENVIRONMENT: {env_name}")
        print("="*80)

        print(f"\nEpisodes: {len(data['episodes'])}")
        print(f"Total decisions: {len(data['decisions'])}")
        print(f"Test questions: {len(data['test_results'])}")

        # 1. Selection Correctness
        print("\n--- 1. SELECTION MECHANISM CORRECTNESS ---")
        if data['selection_issues']:
            print(f"⚠️  ISSUES FOUND: {len(data['selection_issues'])} decisions with selection problems")
            for issue in data['selection_issues'][:5]:  # Show first 5
                print(f"  Episode {issue['episode']}, Step {issue['step']}:")
                print(f"    Selected idx={issue['selected_idx']} (score={issue['selected_score']:.3f})")
                print(f"    Should be idx={issue['correct_idx']} (score={issue['correct_score']:.3f})")
        else:
            print("✅ All decisions correctly selected highest-scored candidate")

        # 2. Score Distribution
        print("\n--- 2. ACTOR SCORE DISTRIBUTION ---")
        score_stats = analyze_score_distribution(data['decisions'])
        print(f"  Mean score: {score_stats['mean']:.3f} ± {score_stats['std']:.3f}")
        print(f"  Range: [{score_stats['min']:.3f}, {score_stats['max']:.3f}]")
        print(f"  Quartiles: Q1={score_stats['q25']:.3f}, Median={score_stats['q50']:.3f}, Q3={score_stats['q75']:.3f}")

        # 3. Candidate Diversity
        print("\n--- 3. CANDIDATE DIVERSITY ---")
        diversities = analyze_candidate_diversity(data['decisions'])
        diversity_scores = [d['diversity'] for d in diversities]
        print(f"  Mean diversity (std of scores): {np.mean(diversity_scores):.3f}")
        print(f"  Low diversity steps (<0.1): {sum(1 for d in diversity_scores if d < 0.1)}/{len(diversity_scores)}")

        if np.mean(diversity_scores) < 0.1:
            print("  ⚠️  LOW DIVERSITY: Candidates may be too similar")

        # 4. Planning Question Performance
        print("\n--- 4. PLANNING QUESTION PERFORMANCE ---")
        planning_results = [r for r in data['test_results'] if r['query_type'] == 'planning']
        if planning_results:
            planning_scores = [r.get('score', r.get('correct', 0.0)) for r in planning_results]
            avg_score = np.mean(planning_scores)
            print(f"  Planning questions: {len(planning_results)}")
            print(f"  Average score: {avg_score:.3f}")
            print(f"  Success rate: {sum(1 for s in planning_scores if s > 0.5)/len(planning_scores)*100:.1f}%")

            # Show failures
            failures = analyze_planning_failures(data, env_name)
            if failures:
                print(f"\n  Planning failures: {len(failures)}")
                print("\n  Sample failures:")
                for failure in failures[:3]:
                    print(f"    Episode: {failure['episode']}")
                    print(f"    Question: {failure['question'][:80]}...")
                    print(f"    Score: {failure['score']:.2f}, Confidence: {failure['confidence']:.2f}")
                    print()

        # 5. Score vs Outcome Correlation
        print("\n--- 5. DECISION QUALITY ANALYSIS ---")
        # Extract: for each decision, what was the selected score?
        # Then look at episode outcome
        if data['episodes']:
            episode_scores = []
            for ep in data['episodes']:
                decisions = extract_decision_data(ep)
                if decisions:
                    avg_selected_score = np.mean([d['selected_score'] for d in decisions])

                    # Get episode performance
                    metrics = compute_all_metrics(ep)
                    overall_acc = metrics['overall_accuracy']

                    episode_scores.append({
                        'episode': ep['episode_id'],
                        'avg_selected_score': avg_selected_score,
                        'overall_accuracy': overall_acc
                    })

            if len(episode_scores) > 1:
                scores = [e['avg_selected_score'] for e in episode_scores]
                accs = [e['overall_accuracy'] for e in episode_scores]

                # Compute correlation
                correlation = np.corrcoef(scores, accs)[0, 1]

                print(f"  Correlation (ACTOR score vs episode accuracy): {correlation:.3f}")

                if correlation < 0:
                    print("  ❌❌ NEGATIVE CORRELATION: ACTOR scores are ANTI-correlated with success!")
                    print("     This suggests ACTOR is scoring strategies BACKWARDS")
                elif correlation < 0.3:
                    print("  ⚠️  WEAK CORRELATION: ACTOR scores may not be predictive of success")
                else:
                    print("  ✅ Positive correlation detected")

                print("\n  Episode breakdown:")
                for e in episode_scores:
                    print(f"    {e['episode']:30} avg_score={e['avg_selected_score']:.3f}  acc={e['overall_accuracy']:.3f}")

    # Overall Summary
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)

    for env_name, data in env_results.items():
        print(f"\n{env_name}:")

        if data['selection_issues']:
            print(f"  ❌ Selection mechanism has bugs ({len(data['selection_issues'])} issues)")
        else:
            print(f"  ✅ Selection mechanism working correctly")

        diversities = analyze_candidate_diversity(data['decisions'])
        avg_diversity = np.mean([d['diversity'] for d in diversities])
        if avg_diversity < 0.1:
            print(f"  ⚠️  Low candidate diversity (avg={avg_diversity:.3f})")

        if data['episodes'] and len(data['episodes']) > 1:
            episode_scores = []
            for ep in data['episodes']:
                decisions = extract_decision_data(ep)
                if decisions:
                    avg_selected_score = np.mean([d['selected_score'] for d in decisions])
                    metrics = compute_all_metrics(ep)
                    episode_scores.append((avg_selected_score, metrics['overall_accuracy']))

            if len(episode_scores) > 1:
                scores, accs = zip(*episode_scores)
                correlation = np.corrcoef(scores, accs)[0, 1]

                if correlation < 0:
                    print(f"  ❌❌ ACTOR scoring appears BROKEN (correlation={correlation:.3f})")
                    print(f"     → Recommendation: Debug ACTOR scoring logic for {env_name}")
                elif correlation < 0.3:
                    print(f"  ⚠️  ACTOR scoring not predictive (correlation={correlation:.3f})")

if __name__ == '__main__':
    main()
