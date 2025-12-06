#!/usr/bin/env python3
"""
Training Data Quality Audit Script

Analyzes data/training_pairs.json to identify issues that may cause
response quality degradation in the graduated world model.
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path
import random
import hashlib

def load_training_data():
    """Load training pairs from JSON file."""
    data_path = Path("data/training_pairs.json")
    with open(data_path, "r") as f:
        return json.load(f)

def compute_basic_statistics(pairs):
    """Compute basic statistics by domain."""
    stats = defaultdict(lambda: {
        "count": 0,
        "instruction_lengths": [],
        "response_lengths": [],
        "scores": [],
        "reliability": Counter()
    })

    for pair in pairs:
        domain = pair.get("domain", "unknown")
        stats[domain]["count"] += 1
        stats[domain]["instruction_lengths"].append(len(pair.get("instruction", "")))
        stats[domain]["response_lengths"].append(len(pair.get("response", "")))
        if "score" in pair:
            stats[domain]["scores"].append(pair["score"])
        if "reliability" in pair:
            stats[domain]["reliability"][pair["reliability"]] += 1

    # Compute aggregates
    result = {}
    for domain, data in stats.items():
        inst_lens = data["instruction_lengths"]
        resp_lens = data["response_lengths"]
        result[domain] = {
            "count": data["count"],
            "avg_instruction_len": sum(inst_lens) / len(inst_lens) if inst_lens else 0,
            "avg_response_len": sum(resp_lens) / len(resp_lens) if resp_lens else 0,
            "min_response_len": min(resp_lens) if resp_lens else 0,
            "max_response_len": max(resp_lens) if resp_lens else 0,
            "avg_score": sum(data["scores"]) / len(data["scores"]) if data["scores"] else None,
            "reliability": dict(data["reliability"])
        }
    return result

def find_conflicting_values(pairs):
    """Find pairs where similar instructions have contradictory responses."""
    conflicts = []

    # Group by normalized instruction
    instruction_groups = defaultdict(list)
    for pair in pairs:
        # Normalize instruction for comparison
        inst = pair.get("instruction", "").lower().strip()
        instruction_groups[inst].append(pair)

    # Check for conflicts within groups
    for inst, group in instruction_groups.items():
        if len(group) > 1:
            # Check if responses differ
            responses = set(p.get("response", "") for p in group)
            if len(responses) > 1:
                conflicts.append({
                    "instruction": inst,
                    "responses": list(responses),
                    "count": len(group),
                    "pairs": group
                })

    return conflicts

def check_format_leakage(pairs):
    """Check for format leakage in responses."""
    issues = []

    patterns = {
        "question_marks": r"\?",
        "instruction_marker": r"(?i)instruction:",
        "response_marker": r"(?i)response:",
        "format_artifacts": r"#{3,}",
        "embedded_qa": r"(?i)(Q:|A:|Question:|Answer:)",
        "self_reference": r"(?i)(I think|I believe|I would|Let me)",
        "continuation_markers": r"(?i)(continued|part \d|section \d)"
    }

    for pair in pairs:
        response = pair.get("response", "")
        found_issues = []

        for issue_type, pattern in patterns.items():
            if re.search(pattern, response):
                found_issues.append(issue_type)

        if found_issues:
            issues.append({
                "instruction": pair.get("instruction", ""),
                "response": response,
                "domain": pair.get("domain", "unknown"),
                "issues": found_issues
            })

    return issues

def check_repetition(pairs):
    """Check for duplicate or near-duplicate pairs and internal repetition."""
    duplicates = []
    internal_repetition = []
    boilerplate = Counter()

    # Check for exact duplicates
    seen = {}
    for i, pair in enumerate(pairs):
        key = (pair.get("instruction", ""), pair.get("response", ""))
        hash_key = hashlib.md5(f"{key[0]}|{key[1]}".encode()).hexdigest()

        if hash_key in seen:
            duplicates.append({
                "index1": seen[hash_key],
                "index2": i,
                "instruction": pair.get("instruction", ""),
                "domain": pair.get("domain", "unknown")
            })
        else:
            seen[hash_key] = i

    # Check for internal repetition in responses
    for pair in pairs:
        response = pair.get("response", "")
        # Look for repeated phrases (3+ words repeated)
        words = response.split()
        if len(words) > 10:
            # Check for repetition of 3-word phrases
            trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
            trigram_counts = Counter(trigrams)
            repeated = [t for t, c in trigram_counts.items() if c > 1]
            if repeated:
                internal_repetition.append({
                    "instruction": pair.get("instruction", ""),
                    "response": response,
                    "domain": pair.get("domain", "unknown"),
                    "repeated_phrases": repeated[:5]  # Limit to first 5
                })

        # Track common phrases for boilerplate detection
        sentences = response.split(". ")
        for sent in sentences:
            if len(sent.split()) >= 5:
                boilerplate[sent.strip()] += 1

    # Filter boilerplate to phrases appearing 3+ times
    common_boilerplate = {k: v for k, v in boilerplate.items() if v >= 3}

    return {
        "duplicates": duplicates,
        "internal_repetition": internal_repetition,
        "boilerplate": common_boilerplate
    }

def check_hallucination_patterns(pairs):
    """Look for suspiciously precise or unrealistic values."""
    issues = []

    # Patterns for suspicious values
    patterns = {
        "many_decimals": r"\d+\.\d{4,}",  # 4+ decimal places
        "unrealistic_temp": r"(?:temperature|temp)[:\s]+\d{4,}",  # 4+ digit temps
        "unrealistic_probability": r"(?:probability|prob)[:\s]+[01]\.\d{6,}",  # very precise probs
        "precise_percentages": r"\d+\.\d{3,}%"  # percentages with 3+ decimals
    }

    for pair in pairs:
        response = pair.get("response", "")
        instruction = pair.get("instruction", "")
        found_issues = []

        for issue_type, pattern in patterns.items():
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                found_issues.append({
                    "type": issue_type,
                    "matches": matches[:5]
                })

        if found_issues:
            issues.append({
                "instruction": instruction,
                "response": response,
                "domain": pair.get("domain", "unknown"),
                "issues": found_issues
            })

    return issues

def sample_quality_review(pairs, samples_per_domain=5):
    """Pull random samples from each domain for quality review."""
    by_domain = defaultdict(list)
    for pair in pairs:
        by_domain[pair.get("domain", "unknown")].append(pair)

    samples = {}
    random.seed(42)  # For reproducibility

    for domain, domain_pairs in by_domain.items():
        samples[domain] = random.sample(
            domain_pairs,
            min(samples_per_domain, len(domain_pairs))
        )

    return samples

def identify_cleanup_candidates(pairs, format_issues, conflicts, repetition):
    """Identify pairs that need cleanup."""
    to_remove = []
    to_fix = []
    patterns_to_filter = []

    # Pairs with format leakage - minor issues, can fix
    for issue in format_issues:
        if "question_marks" in issue["issues"] and len(issue["issues"]) == 1:
            # Only question mark, minor issue
            to_fix.append({
                "instruction": issue["instruction"][:80],
                "reason": "Contains question mark in response",
                "domain": issue["domain"]
            })
        elif len(issue["issues"]) > 2:
            # Multiple issues, remove
            to_remove.append({
                "instruction": issue["instruction"][:80],
                "reason": f"Multiple format issues: {', '.join(issue['issues'])}",
                "domain": issue["domain"]
            })

    # Exact duplicates - remove all but first
    for dup in repetition["duplicates"]:
        to_remove.append({
            "instruction": dup["instruction"][:80],
            "reason": f"Exact duplicate (indices {dup['index1']}, {dup['index2']})",
            "domain": dup["domain"]
        })

    # Internal repetition - fix or remove
    for rep in repetition["internal_repetition"]:
        to_fix.append({
            "instruction": rep["instruction"][:80],
            "reason": f"Internal repetition: {rep['repeated_phrases'][0][:40]}...",
            "domain": rep["domain"]
        })

    # Patterns to filter
    if repetition["boilerplate"]:
        top_boilerplate = sorted(repetition["boilerplate"].items(), key=lambda x: -x[1])[:5]
        for phrase, count in top_boilerplate:
            if count > 5:
                patterns_to_filter.append({
                    "pattern": phrase[:60],
                    "count": count,
                    "action": "Consider varying this boilerplate"
                })

    return {
        "to_remove": to_remove,
        "to_fix": to_fix,
        "patterns_to_filter": patterns_to_filter
    }

def generate_report(stats, conflicts, format_issues, repetition, hallucination,
                    samples, cleanup, total_pairs):
    """Generate the audit report in markdown format."""

    report = []
    report.append("# Training Data Audit Report\n")
    report.append(f"**Total pairs analyzed:** {total_pairs}\n")
    report.append(f"**Audit date:** Generated by audit_training_data.py\n\n")

    # Summary Statistics
    report.append("## Summary Statistics\n")
    report.append("| Metric | " + " | ".join(stats.keys()) + " | Total |")
    report.append("|--------|" + "|".join(["--------"] * len(stats)) + "|-------|")

    # Pairs row
    pairs_row = [str(stats[d]["count"]) for d in stats]
    total = sum(stats[d]["count"] for d in stats)
    report.append(f"| Pairs | {' | '.join(pairs_row)} | {total} |")

    # Avg instruction len
    inst_row = [f"{stats[d]['avg_instruction_len']:.1f}" for d in stats]
    avg_inst = sum(stats[d]['avg_instruction_len'] for d in stats) / len(stats)
    report.append(f"| Avg instruction len | {' | '.join(inst_row)} | {avg_inst:.1f} |")

    # Avg response len
    resp_row = [f"{stats[d]['avg_response_len']:.1f}" for d in stats]
    avg_resp = sum(stats[d]['avg_response_len'] for d in stats) / len(stats)
    report.append(f"| Avg response len | {' | '.join(resp_row)} | {avg_resp:.1f} |")

    # Min/Max response
    min_row = [str(stats[d]['min_response_len']) for d in stats]
    max_row = [str(stats[d]['max_response_len']) for d in stats]
    report.append(f"| Min response len | {' | '.join(min_row)} | - |")
    report.append(f"| Max response len | {' | '.join(max_row)} | - |")

    # Avg score
    score_row = [f"{stats[d]['avg_score']:.3f}" if stats[d]['avg_score'] else "N/A" for d in stats]
    report.append(f"| Avg score | {' | '.join(score_row)} | - |")

    report.append("")

    # Reliability breakdown
    report.append("### Reliability Distribution\n")
    for domain, data in stats.items():
        report.append(f"**{domain}:** {data['reliability']}")
    report.append("")

    # Issue 1: Conflicting Values
    report.append("## Issue 1: Conflicting Values\n")
    report.append(f"**{len(conflicts)} instruction(s) with conflicting responses found**\n")

    if conflicts:
        report.append("### Examples:\n")
        for conflict in conflicts[:5]:  # Show first 5
            report.append(f"#### Instruction: \"{conflict['instruction'][:80]}...\"")
            report.append(f"- **Occurrences:** {conflict['count']}")
            report.append("- **Responses:**")
            for resp in list(conflict['responses'])[:3]:
                report.append(f"  - \"{resp[:100]}...\"")
            report.append("")

    # Issue 2: Format Leakage
    report.append("## Issue 2: Format Leakage\n")

    # Count by issue type
    issue_counts = Counter()
    for issue in format_issues:
        for i in issue["issues"]:
            issue_counts[i] += 1

    report.append(f"**{len(format_issues)} responses with format issues**\n")
    report.append("### Issue breakdown:")
    for issue_type, count in issue_counts.most_common():
        report.append(f"- {issue_type}: {count}")
    report.append("")

    if format_issues:
        report.append("### Examples:\n")
        for issue in format_issues[:5]:
            report.append(f"- **Domain:** {issue['domain']}")
            report.append(f"  - **Instruction:** \"{issue['instruction'][:60]}...\"")
            report.append(f"  - **Issues:** {', '.join(issue['issues'])}")
            report.append(f"  - **Response:** \"{issue['response'][:100]}...\"")
            report.append("")

    # Issue 3: Repetition
    report.append("## Issue 3: Repetition Patterns\n")

    report.append(f"### Exact Duplicates: {len(repetition['duplicates'])}")
    if repetition['duplicates']:
        report.append("\nExamples:")
        for dup in repetition['duplicates'][:5]:
            report.append(f"- \"{dup['instruction'][:60]}...\" ({dup['domain']})")
    report.append("")

    report.append(f"### Internal Repetition: {len(repetition['internal_repetition'])}")
    if repetition['internal_repetition']:
        report.append("\nExamples:")
        for rep in repetition['internal_repetition'][:5]:
            report.append(f"- \"{rep['instruction'][:60]}...\"")
            report.append(f"  Repeated: {rep['repeated_phrases'][0][:50]}...")
    report.append("")

    report.append(f"### Boilerplate Phrases: {len(repetition['boilerplate'])}")
    if repetition['boilerplate']:
        report.append("\nTop repeated phrases:")
        for phrase, count in sorted(repetition['boilerplate'].items(), key=lambda x: -x[1])[:10]:
            report.append(f"- ({count}x) \"{phrase[:60]}...\"")
    report.append("")

    # Issue 4: Hallucination Patterns
    report.append("## Issue 4: Hallucination Patterns\n")
    report.append(f"**{len(hallucination)} responses with suspicious values**\n")

    if hallucination:
        report.append("### Examples:\n")
        for h in hallucination[:5]:
            report.append(f"- **Domain:** {h['domain']}")
            report.append(f"  - **Instruction:** \"{h['instruction'][:60]}...\"")
            for issue in h['issues']:
                report.append(f"  - **{issue['type']}:** {issue['matches'][:3]}")
    report.append("")

    # Sample Quality Review
    report.append("## Issue 5: Sample Quality Review\n")
    report.append("5 random samples from each domain for manual review:\n")

    for domain, domain_samples in samples.items():
        report.append(f"### {domain}\n")
        for i, sample in enumerate(domain_samples, 1):
            report.append(f"#### Sample {i}")
            report.append(f"- **Instruction:** {sample.get('instruction', 'N/A')}")
            report.append(f"- **Response:** {sample.get('response', 'N/A')[:200]}...")
            report.append(f"- **Reliability:** {sample.get('reliability', 'N/A')}")
            report.append(f"- **Score:** {sample.get('score', 'N/A')}")
            report.append("")

    # Cleanup Candidates
    report.append("## Cleanup Candidates\n")

    report.append(f"### Pairs to Remove ({len(cleanup['to_remove'])} total)\n")
    if cleanup['to_remove']:
        for item in cleanup['to_remove'][:10]:
            report.append(f"- [{item['domain']}] \"{item['instruction'][:50]}...\"")
            report.append(f"  - Reason: {item['reason']}")
    report.append("")

    report.append(f"### Pairs to Fix ({len(cleanup['to_fix'])} total)\n")
    if cleanup['to_fix']:
        for item in cleanup['to_fix'][:10]:
            report.append(f"- [{item['domain']}] \"{item['instruction'][:50]}...\"")
            report.append(f"  - Reason: {item['reason']}")
    report.append("")

    report.append(f"### Patterns to Filter ({len(cleanup['patterns_to_filter'])} total)\n")
    if cleanup['patterns_to_filter']:
        for item in cleanup['patterns_to_filter']:
            report.append(f"- ({item['count']}x) \"{item['pattern'][:50]}...\"")
            report.append(f"  - Action: {item['action']}")
    report.append("")

    # Recommendations
    report.append("## Recommendations\n")

    dups_count = len(repetition['duplicates'])
    format_count = len([i for i in format_issues if len(i['issues']) > 2])
    fix_count = len(cleanup['to_fix'])

    report.append("### Immediate Actions")
    if dups_count > 0:
        report.append(f"- [ ] Remove {dups_count} exact duplicate pairs")
    if format_count > 0:
        report.append(f"- [ ] Remove {format_count} pairs with severe format issues")
    if len(conflicts) > 0:
        report.append(f"- [ ] Resolve {len(conflicts)} instruction conflicts (keep most recent/highest score)")

    report.append("\n### Data Quality Improvements")
    if fix_count > 0:
        report.append(f"- [ ] Fix {fix_count} pairs with minor issues")
    if repetition['boilerplate']:
        report.append("- [ ] Add response variation to reduce boilerplate")
    if len(hallucination) > 0:
        report.append(f"- [ ] Review {len(hallucination)} pairs with suspicious values")

    report.append("\n### Structural Recommendations")

    # Check for domain imbalance
    counts = [stats[d]["count"] for d in stats]
    if max(counts) > 2 * min(counts):
        report.append("- [ ] Balance domain representation (significant imbalance detected)")

    report.append("- [ ] Consider deduplicating by instruction (keeping highest score)")
    report.append("- [ ] Add data augmentation for underrepresented scenarios")

    report.append("\n### Decision Matrix")
    report.append("| Action | Impact | Effort | Recommendation |")
    report.append("|--------|--------|--------|----------------|")

    if dups_count > total_pairs * 0.1:
        report.append("| Remove duplicates | High | Low | **Do immediately** |")
    else:
        report.append("| Remove duplicates | Low | Low | Do if convenient |")

    if len(conflicts) > 10:
        report.append("| Resolve conflicts | High | Medium | **Priority** |")
    else:
        report.append("| Resolve conflicts | Medium | Medium | Recommended |")

    if format_count > total_pairs * 0.05:
        report.append("| Fix format issues | High | Medium | **Priority** |")
    else:
        report.append("| Fix format issues | Low | Low | Optional |")

    report.append("")

    return "\n".join(report)

def main():
    print("Loading training data...")
    pairs = load_training_data()
    total_pairs = len(pairs)
    print(f"Loaded {total_pairs} pairs")

    print("\n1. Computing basic statistics...")
    stats = compute_basic_statistics(pairs)
    for domain, data in stats.items():
        print(f"  {domain}: {data['count']} pairs, avg response len: {data['avg_response_len']:.1f}")

    print("\n2. Checking for conflicting values...")
    conflicts = find_conflicting_values(pairs)
    print(f"  Found {len(conflicts)} instructions with conflicting responses")

    print("\n3. Checking for format leakage...")
    format_issues = check_format_leakage(pairs)
    print(f"  Found {len(format_issues)} responses with format issues")

    print("\n4. Checking for repetition patterns...")
    repetition = check_repetition(pairs)
    print(f"  Found {len(repetition['duplicates'])} exact duplicates")
    print(f"  Found {len(repetition['internal_repetition'])} with internal repetition")
    print(f"  Found {len(repetition['boilerplate'])} common boilerplate phrases")

    print("\n5. Checking for hallucination patterns...")
    hallucination = check_hallucination_patterns(pairs)
    print(f"  Found {len(hallucination)} responses with suspicious values")

    print("\n6. Sampling for quality review...")
    samples = sample_quality_review(pairs)
    for domain, domain_samples in samples.items():
        print(f"  {domain}: {len(domain_samples)} samples")

    print("\n7. Identifying cleanup candidates...")
    cleanup = identify_cleanup_candidates(pairs, format_issues, conflicts, repetition)
    print(f"  To remove: {len(cleanup['to_remove'])}")
    print(f"  To fix: {len(cleanup['to_fix'])}")
    print(f"  Patterns to filter: {len(cleanup['patterns_to_filter'])}")

    print("\n8. Generating report...")
    report = generate_report(
        stats, conflicts, format_issues, repetition, hallucination,
        samples, cleanup, total_pairs
    )

    report_path = Path("audit_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nReport written to: {report_path}")
    print("\n=== SUMMARY ===")
    print(f"Total pairs: {total_pairs}")
    print(f"Conflicts: {len(conflicts)}")
    print(f"Format issues: {len(format_issues)}")
    print(f"Duplicates: {len(repetition['duplicates'])}")
    print(f"Hallucination patterns: {len(hallucination)}")
    print(f"Cleanup candidates: {len(cleanup['to_remove']) + len(cleanup['to_fix'])}")

if __name__ == "__main__":
    main()
