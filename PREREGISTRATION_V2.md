# PREREGISTRATION V2: ACE with Cost and Token Accountability

**Document Version**: 2.0 (Extends V1)
**Date**: October 30, 2025
**Status**: LOCKED (after experiment begins)
**Previous Version**: PREREGISTRATION.md (2025-10-29)

---

## What's New in V2

This version **extends** the original preregistration with:
1. **Exact cost tracking** - USD costs with locked pricing snapshot
2. **Token breakdown by category** - Exploration, curation, evaluation, planning
3. **Updated evaluation system** - Exploration-dependent queries (v2)
4. **Cost-efficiency metrics** - Accuracy per dollar spent
5. **Token accounting validation** - Automated verification system

All hypotheses from V1 remain valid. New metrics added for transparency.

---

## V2 Additions to Primary Outcomes

### Cost Tracking (NEW)
**Definition**: Exact USD cost per episode using locked API pricing

**Pricing Snapshot** (locked October 30, 2025):
- **Claude Sonnet 4.5**: $3.00/1M input, $15.00/1M output
- **Source**: [Anthropic API Pricing](https://www.anthropic.com/api) as of 2025-10-30
- **Implementation**: `utils/cost_tracker.py`

**Measurement**:
```python
cost_usd = episode['cost']['total_cost_usd']
# Breakdown:
input_cost = (input_tokens / 1_000_000) × $3.00
output_cost = (output_tokens / 1_000_000) × $15.00
total_cost = input_cost + output_cost
```

**Aggregation**:
- Mean cost per agent (across 20 episodes)
- Total cost per agent
- Cost standard deviation

**Analysis Location**: `scripts/analyze_with_statistics.py::cost_analysis()`

---

### Token Breakdown by Category (NEW)
**Categories** (mutually exclusive, must sum to total):
1. **exploration** - Action selection based on observations
2. **curation** - Belief updates, playbook updates
3. **evaluation** - Test query answering
4. **planning** - Prior generation, reflection, lookahead

**Validation Requirement**:
```python
assert sum(breakdown[cat]['total'] for cat in categories) == total_tokens
```

**Measurement**:
```python
breakdown = episode['token_breakdown']['breakdown']
# Example:
{
  'exploration': {'input': 8000, 'output': 5000, 'total': 13000},
  'curation': {'input': 2000, 'output': 1500, 'total': 3500},
  'evaluation': {'input': 2000, 'output': 1500, 'total': 3500},
  'planning': {'input': 500, 'output': 300, 'total': 800},
  'totals': {'input': 12500, 'output': 8300, 'total': 20800}
}
```

**Per-Agent Expected Distributions**:
- **Observer**: evaluation ≥90% (no exploration/planning)
- **Actor**: exploration ≥60%, curation ≥10%, evaluation ≥15%, planning ≥5%
- **ACE**: exploration ≥50%, curation ≥15%, evaluation ≥15%, planning ≥10%

**Analysis Location**: `scripts/analyze_with_statistics.py` (token distribution plots)

---

### Cost-Efficiency (NEW)
**Definition**: Accuracy achieved per dollar spent

**Formula**:
```python
efficiency = mean_accuracy / mean_cost_usd
```

**Units**: Accuracy points per $1

**Expected Rankings**:
1. Observer (low cost, low accuracy → moderate efficiency)
2. Actor or ACE (high cost, high accuracy → best efficiency)
3. Model-Based (highest cost → lowest efficiency)

**Analysis**: Pareto frontier with cost on x-axis, accuracy on y-axis

**Analysis Location**: `scripts/analyze_with_statistics.py::cost_efficiency_analysis()`

---

## V2 Updated Hypotheses

### H-Token-Strategy (NEW)
**Hypothesis**: Agent architectures show distinct token allocation strategies.

**Predictions**:
- Observer: >85% evaluation tokens (no exploration)
- Actor: 60-70% exploration, 10-15% curation
- ACE: 50-60% exploration, 15-25% curation (due to playbook)

**Test**: Chi-square test for distribution differences, α=0.05

---

### H-Cost-Efficiency (NEW)
**Hypothesis**: Despite higher costs, ACE achieves best cost-efficiency (accuracy/cost_usd).

**Predictions**:
- ACE cost per episode: $0.20 - $0.40
- ACE efficiency: >2.0 accuracy points per $1
- ACE ranks #1 or #2 in efficiency

**Test**: Efficiency ranking + bootstrap CIs

---

## V2 Updated Experimental Design

### Sample Size (Unchanged)
- **n = 20** episodes per (agent, environment) pair
- **4 agents** × **2 environments** × **20 seeds** = **160 total episodes**
- **Seeds**: 1001-1020

### Evaluation System V2 (CRITICAL CHANGE)
**Problem Identified**: Original eval allowed Observer to score 70% via general knowledge

**Solution**: New evaluation system (`evaluation/tasks_exploration_v2.py`)
- Queries require specific measurements from episode trajectory
- Example: "What was the measured temperature at t=20s?" (must explore to answer)
- Ground truth extracted from actual episode steps

**Success Criteria**: Observer <40% accuracy (down from ~70%)

**Implementation**:
- `evaluation/tasks_exploration_v2.py` - New test queries
- `evaluation/trajectory_extraction.py` - Extract measurements from logs
- `scripts/upgrade_to_exploration_eval_v2.py` - Automated upgrade

---

## V2 Data Collection Schema

### Episode Log (Extended)
All episodes now include:

```json
{
  "episode_id": "ace_hotpot_seed1001",
  "cost": {
    "input_cost_usd": 0.0375,
    "output_cost_usd": 0.1245,
    "total_cost_usd": 0.1620,
    "model": "claude-sonnet-4-5-20250929",
    "pricing_snapshot": {
      "input_per_1M": 3.00,
      "output_per_1M": 15.00,
      "date": "2025-10-30"
    }
  },
  "token_breakdown": {
    "breakdown": {
      "exploration": {"input": 8000, "output": 5000, "total": 13000},
      "curation": {"input": 2000, "output": 1500, "total": 3500},
      "evaluation": {"input": 2000, "output": 1500, "total": 3500},
      "planning": {"input": 500, "output": 300, "total": 800},
      "totals": {"input": 12500, "output": 8300, "total": 20800}
    },
    "records": [
      {"category": "exploration", "input": 500, "output": 300, "metadata": {"step": 0}},
      ...
    ],
    "validation_passed": true
  }
}
```

---

## V2 Verification System

### Pre-Flight Checks (REQUIRED)
Before full experiment, run:

```bash
# 1. Cost tracker unit tests
python -m pytest tests/test_cost_tracker.py -v

# 2. Token accounting unit tests
python -m pytest tests/test_token_accounting.py -v

# 3. Integration test on pilot episode
python scripts/verify_token_accounting.py
```

**Success Criteria**:
- All unit tests pass (20/20 token tests, 9/9 cost tests)
- Pilot episode has valid token breakdown
- Validation passes (sum equals total)

**Implementation**: `scripts/verify_token_accounting.py`

---

## V2 Analysis Pipeline

### Automated Analysis (NEW)
All analyses run via single script:

```bash
python scripts/analyze_with_statistics.py results/ace_full_n20
```

**Outputs**:
1. `statistical_ttests.csv` - Paired t-tests
2. `statistical_confidence_intervals.csv` - Bootstrap CIs
3. `cost_analysis.csv` - Per-agent costs
4. `cost_efficiency.csv` - Efficiency rankings
5. `SUMMARY_STATEMENT.md` - Preregistration summary

**Key Methods**:
- `paired_t_tests()` - Accuracy comparisons
- `bootstrap_confidence_intervals()` - 95% CIs
- `cost_analysis()` - Cost statistics
- `cost_efficiency_analysis()` - Efficiency metrics
- `_generate_summary_statement()` - Final report

---

## V2 Validation Requirements

### Token Accounting Validation
Every episode MUST pass:
```python
recorded_input == episode['total_input_tokens']
recorded_output == episode['total_output_tokens']
```

**Tolerance**: 0 tokens (exact match required)

**Failure Handling**: Log warning, set `validation_passed=False`, continue experiment

**Post-Experiment**: Episodes with `validation_passed=False` excluded from token distribution analysis (but included in accuracy analysis)

---

## V2 Timeline Update

1. **Verification Run** (Oct 30, 2025) - Test new systems
   - 10 episodes (2 agents × 5 seeds)
   - Cost: ~$5
   - Duration: ~10 minutes
   - Verify Observer <40%, ACE >60%

2. **Full Experiment** (Oct 30-31, 2025) - 160 episodes
   - Cost: ~$60-80
   - Duration: ~4-6 hours
   - Run `scripts/run_experiment_parallel.py --workers 6`

3. **Analysis** (Oct 31, 2025)
   - Run `scripts/analyze_with_statistics.py`
   - Generate all CSV and markdown reports

4. **Reporting** (Oct 31, 2025)
   - Review `SUMMARY_STATEMENT.md`
   - Check all preregistered metrics reported

---

## V2 Commitments (Extended)

In addition to V1 commitments, we commit to:

1. **Cost Transparency**:
   - Report exact USD costs for all agents
   - Lock pricing snapshot (no updates mid-experiment)
   - Include cost in all figures and tables

2. **Token Accountability**:
   - Report token breakdown for all agents
   - Validate all episodes (log failures)
   - Publish complete records for reproducibility

3. **Evaluation Validity**:
   - Use only exploration-dependent queries
   - Verify Observer <40% (if not, evaluation invalid)
   - Document any query failures

4. **Reproducibility**:
   - All code, data, and analysis scripts public
   - Unit tests for all critical components
   - Verification script included

---

## V2 Changes from V1

| Aspect | V1 | V2 |
|--------|----|----|
| Cost tracking | Token counts only | Exact USD with locked pricing |
| Token breakdown | Total tokens | By category (4 categories) |
| Evaluation | Original queries | Exploration-dependent (v2) |
| Validation | Manual review | Automated validation system |
| Analysis | Manual scripts | Unified pipeline script |
| Cost metrics | Tokens per accuracy | Accuracy per dollar (efficiency) |

---

## V2 References

**Code Locations**:
- Cost tracking: `utils/cost_tracker.py`
- Token accounting: `utils/token_accounting.py`
- Runner integration: `experiments/runner.py` (lines 237-274)
- Analysis: `scripts/analyze_with_statistics.py`
- Verification: `scripts/verify_token_accounting.py`

**Unit Tests**:
- `tests/test_cost_tracker.py` (9 tests)
- `tests/test_token_accounting.py` (20 tests)

**Configuration**:
- Verification: `config_verification_v2.yaml`
- Full study: `config_ace_full_n20.yaml`

---

## V2 Signature

**Preregistration V2 locked on**: October 30, 2025
**Extends**: PREREGISTRATION.md (V1, Oct 29, 2025)
**Status**: Ready for verification run

By committing this V2 preregistration, we commit to:
1. All V1 commitments remain valid
2. Report all new V2 metrics (cost, token breakdown, efficiency)
3. Use updated evaluation system (exploration-dependent)
4. Run automated verification before full experiment
5. Publish complete cost and token accountability data

---

**END OF PREREGISTRATION V2**

**Next Steps**:
1. Run verification: `python scripts/verify_token_accounting.py`
2. If passes: Run full experiment via `scripts/run_preregistered_pipeline.sh`
3. Analyze: `python scripts/analyze_with_statistics.py results/ace_full_n20`
4. Report: Review `results/ace_full_n20/SUMMARY_STATEMENT.md`
