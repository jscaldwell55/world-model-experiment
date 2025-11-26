# Critical Fidelity Validation Experiments - Execution Plan

## Overview

This document outlines the execution plan for three critical experiments designed to validate whether Offline Consolidation (OC) helps or creates circular reasoning problems.

## Background

From `FIDELITY_CRITICAL_ISSUES.md`, we identified three critical issues:

1. **Circular Fidelity Scoring** - Synthetics are scored using the same model that generated them
2. **No Downstream Validation** - Haven't proven synthetics actually help with predictions
3. **Systematic Bias Risk** - Wrong model + high fidelity = amplified bias

## Experiments

### Experiment A: Do Synthetics Help?
**Goal:** Test if synthetic data from OC improves prediction accuracy

**Method:**
- Split episodes into train (67%) / test (33%)
- Condition 1: Train world model on real data only
- Condition 2: Train world model on real + synthetic data
- Compare accuracy on 50 held-out test queries with known ground truth

**Pass Criteria:** accuracy_synthetic > accuracy_real + 0.02 (2% improvement)

**What it tells us:**
- If synthetics help: OC is valuable, proceed with fidelity improvements
- If synthetics neutral: OC adds complexity without benefit
- If synthetics hurt: OC is actively harmful, stop immediately

---

### Experiment B: Cross-Validation of Synthetics
**Goal:** Validate that synthetics match held-out real data

**Method:**
- For each HIGH reliability episode:
  1. Hold it out
  2. Train world model on all OTHER episodes
  3. Generate synthetic matching held-out's conditions
  4. Compare synthetic vs real held-out outcomes
- Calculate mean error and pass rate

**Pass Criteria:** mean_error < 0.10 (10%) AND pass_rate > 0.80 (80%)

**What it tells us:**
- Whether synthetics are realistic
- If world model generalizes to counterfactuals
- Which types of synthetics to trust

---

### Experiment C: Wrong Model Detection
**Goal:** Test whether OC detects and rejects inaccurate world models

**Method:**
- Deliberately create wrong belief (train only on MIXED power episodes)
- Try to generate synthetics using OC
- Check if OC's quality gate catches the error
- Measure error on HIGH power ground truth queries

**Pass Criteria:** OC gate_status == 'FAIL' OR high error is flagged

**What it tells us:**
- Whether OC safeguards work
- If bias amplification is a real risk
- What thresholds are needed

---

## Execution Status

### Phase 1: Data Collection ✓ IN PROGRESS
**Goal:** Collect 30 HotPot episodes with persistent memory

**Config:** `config_fidelity_validation_30ep.yaml`
- 30 episodes with SimpleWorldModel agent
- Seeds: 100-129
- Persistent memory enabled
- Max steps: 15 per episode

**Requirements:**
- At least 10 total episodes for Experiment A
- At least 3 HIGH reliability episodes for Experiment B
- Mix of reliability levels for Experiment C

**Current Status:**
- Running in background with 5 workers
- Expected time: 10-20 minutes
- Output: `results/fidelity_data_30ep/`

### Phase 2: Experiment Execution (PENDING)
**Script:** `experiments/fidelity_validation.py`

**Command:**
```bash
python3 experiments/fidelity_validation.py \
  --domain hot_pot \
  --output results/fidelity_validation_results.json
```

**What to expect:**
- Experiment A: Tests real vs real+synthetic accuracy
- Experiment B: Cross-validates synthetics against held-out data
- Experiment C: Tests wrong model detection

**Output:**
- Console: Detailed results with pass/fail status
- JSON: `results/fidelity_validation_results.json`

### Phase 3: Analysis & Decision (PENDING)

**If all experiments PASS:**
✓ Offline consolidation is validated and safe
- Synthetics improve accuracy
- Synthetics match real data
- Quality gates detect bad models
→ **Action:** Proceed with OC → FTB integration, but fix fidelity scoring

**If any experiment FAILS:**

- **Experiment A fails:**
  - If synthetics hurt: STOP - disable OC immediately
  - If synthetics neutral: Reconsider - is OC worth the complexity?

- **Experiment B fails:**
  - Synthetics don't match reality
  - World model not generalizing
  → **Action:** Don't use synthetics for fine-tuning

- **Experiment C fails:**
  - Quality gates not working
  - Risk of bias amplification
  → **Action:** Add world model validation before generation

---

## Expected Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Data Collection | 10-20 min | 🔄 IN PROGRESS |
| Experiment Execution | 1-2 min | ⏸️ PENDING |
| Analysis & Reporting | 5 min | ⏸️ PENDING |
| **Total** | **~20-30 min** | |

---

## Key Files

### Data
- `memory/domains/hot_pot/playbook.json` - Current playbook (5 episodes, 1 HIGH)
- `memory/domains/hot_pot/episodes/*.json` - Raw episode data
- `results/fidelity_data_30ep/` - New 30-episode dataset

### Code
- `experiments/fidelity_validation.py` - All three experiments
- `config_fidelity_validation_30ep.yaml` - Data collection config
- `memory/offline_consolidation.py` - OC implementation
- `agents/simple_world_model.py` - World model agent

### Documentation
- `FIDELITY_CRITICAL_ISSUES.md` - Problem analysis
- `EXPERIMENT_PLAN.md` - This file

---

## Next Steps

1. ✅ Wait for data collection to complete (~10-20 min)
2. ⏸️ Run validation experiments
3. ⏸️ Analyze results
4. ⏸️ Make decision on OC → FTB integration

---

## Critical Success Factors

For experiments to be meaningful:

✓ **Sufficient data**
- Need 10+ total episodes
- Need 3+ HIGH reliability episodes
- Need mix of power settings

✓ **Diverse conditions**
- HIGH power (consistent heating)
- LOW power (consistent slower heating)
- MIXED power (averaged, LOW reliability)

✓ **Ground truth validation**
- Test queries with known correct answers
- HotPot ground truth: HIGH=2.5°C/s, LOW=0.5°C/s

---

## Risk Mitigation

**If data collection fails:**
- Reduce to 20 episodes for faster completion
- Run sequentially instead of parallel
- Check for API rate limits

**If experiments are inconclusive:**
- Collect more HIGH reliability episodes
- Adjust experiment parameters
- Run with synthetic ground truth first

**If all experiments fail:**
- Immediate decision: PAUSE OC integration
- Focus on fixing fidelity scoring
- Add world model validation
- Re-run experiments with improvements

---

## Contact

For questions about these experiments, see:
- Original experiment design in conversation history
- `FIDELITY_CRITICAL_ISSUES.md` for problem context
- `experiments/fidelity_validation.py` for implementation details
