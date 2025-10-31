# Preliminary Analysis Report: Token Prediction Experiment
**Date:** October 22, 2025
**Status:** Partial Data Analysis

---

## Executive Summary

**Overall Completion: ~50-60%**

- ✅ **Infrastructure:** 62% complete (8/13 tasks)
- ⚠️ **Data Collection:** Partial success with significant issues
- ❌ **Analysis:** Blocked due to missing belief surprisal data

---

## 1. Data Collection Status

### Available Data

#### Pilot Runs
- **pilot_token_fresh/**: 30 token prediction files (77 total in various pilot dirs)
  - 150 total prediction steps
  - Only Actor agent data (no Observer)
  - All 3 environments: HotPot (50), SwitchLight (50), ChemTile (50)

#### Full-Scale Runs
- **parallel_run_20251022_002708/**: 480 episode files
  - ❌ **No token prediction files** - token prediction wasn't enabled
  - ⚠️ Many API failures due to Anthropic overload (see `failed_episodes.json`)

### Critical Issue Identified

**🚨 BLOCKER: Belief surprisal not being computed correctly**

**Symptoms:**
- `belief_surprisal = None` or `-0.0` for all episodes
- Cannot compute coupling between token NLL and belief surprisal
- Core hypothesis (H-Token) cannot be tested with current data

**Root Cause (experiments/token_runner.py:124-134):**
```python
# INCORRECT CODE
if hasattr(agent, 'belief_state') and agent.belief_state is not None:
    if hasattr(agent.belief_state, 'log_likelihood'):  # ❌ This method doesn't exist
        log_likelihood = agent.belief_state.log_likelihood(next_obs, time_elapsed)
        belief_surprisal = -log_likelihood
```

**Correct Approach:**
```python
# SHOULD BE
if hasattr(agent, 'compute_surprisal'):
    belief_surprisal = agent.compute_surprisal(next_obs)
```

The Actor agent has a `compute_surprisal(observation)` method, but the token_runner is trying to access a non-existent `belief_state.log_likelihood()` method.

---

## 2. Analysis Results (Pilot Data)

### Token Prediction Quality
✅ **Token prediction IS working:**
- Successfully computing token NLL from OpenAI API
- Average token NLL: 3.5 (HotPot)
- Std deviation: 2.1
- Sample prediction: "The pot is no longer boiling. The stove indicator light is off." (NLL=1.38)

### Coupling Analysis (A1)
❌ **Cannot compute correlation:**
- Belief surprisal is constant (all zeros or None)
- Pearson/Spearman correlation: NaN (constant input)
- **Status:** Test cannot be performed until bug is fixed

### Other Analyses
- A2 (Surprise Detection): ❌ Blocked - needs valid surprisal
- A3 (Predictive Validity): ❌ Blocked - needs valid surprisal
- A4 (Calibration): ⚠️ Could partially run on token NLL alone
- A5 (Family Factor): ❌ Blocked - needs multiple model runs

---

## 3. Infrastructure Status

### ✅ Completed (8/13)
1. ✅ Negative control experiments - `textualization/negative_controls.py`
2. ✅ Advanced statistical metrics - MI, regression diagnostics, distance correlation
3. ✅ Theoretical framework - 2000+ word documentation
4. ✅ Token prediction system - OpenAI integration working
5. ✅ Textualization layers - All 3 environments
6. ✅ Token logging - JSON logs being saved
7. ✅ Basic analysis tools - `TokenAnalysis` class
8. ✅ Preregistration - Complete hypothesis specification

### 🚧 Remaining (5/13)
1. ⏳ **Fix belief surprisal extraction** - CRITICAL BLOCKER
2. ⏳ Update analysis scripts - Call new statistical methods
3. ⏳ Model-based agent baseline - Run experiments
4. ⏳ Discussion generator - Automated interpretation
5. ⏳ Testing - Unit tests for new functionality

---

## 4. Experiment Execution Status

### Preregistered Target
- 520 total episodes (130 × 4 agents)
- 50 episodes × 3 environments per agent type

### Actual Progress
| Agent | HotPot | SwitchLight | ChemTile | Total | Status |
|-------|--------|-------------|----------|-------|--------|
| Observer | 0/50 | 0/50 | 0/30 | 0/130 | ❌ Not run with tokens |
| Actor | 5/50 | 5/50 | 5/30 | 15/130 | ⚠️ Pilot only (no surprisal) |
| ModelBased | 0/50 | 0/50 | 0/30 | 0/130 | ❌ Not run |
| TextReader | - | - | - | 0/130 | ❌ Not in current runs |

**Total Progress: ~12% of preregistered sample**

### API Failures
Latest parallel run (Oct 22, 2025 00:27-03:22):
- 480 episodes attempted
- Multiple failures: "Error code: 500 - Overloaded" (Anthropic API)
- No token prediction data generated (not enabled)

---

## 5. Key Findings

### What's Working ✅
1. **Token prediction infrastructure** - OpenAI API integration functional
2. **Textualization** - Deterministic, no ground truth leakage
3. **Logging** - JSON format correct, parseable
4. **Analysis tools** - `TokenAnalysis` class loads data correctly

### What's Broken ❌
1. **Belief surprisal extraction** - Wrong method call in token_runner.py
2. **API rate limiting** - Anthropic overload errors blocking full runs
3. **Token prediction not enabled** - Latest parallel run didn't include token prediction
4. **Data completeness** - Only 15 usable episodes (vs 520 target)

### What's Unclear ❓
1. **Observer agent handling** - Should Observer have surprisal? (No belief state)
2. **Model-based baseline** - Not yet implemented/run
3. **Control experiments** - Shuffled/random textualization not tested

---

## 6. Immediate Action Items (Priority Order)

### 🔥 CRITICAL (Blocks all analysis)
1. **Fix belief surprisal extraction**
   - File: `experiments/token_runner.py:124-134`
   - Change: Use `agent.compute_surprisal(next_obs)` instead of `belief_state.log_likelihood()`
   - Test: Verify non-zero surprisal values in token logs

2. **Fix API rate limiting**
   - File: `scripts/run_experiment_parallel.py`
   - Options:
     - Reduce parallelism (fewer concurrent episodes)
     - Add exponential backoff with longer delays
     - Split into smaller batches

### ⚠️ HIGH (Needed for valid experiment)
3. **Re-run pilot with fixed surprisal**
   - Command: `python scripts/pilot_token_run.py --output-dir results/pilot_fixed`
   - Verify: Check that `belief_surprisal` has non-zero variance
   - Expected: HotPot coupling r > 0.3

4. **Enable token prediction in parallel runner**
   - Ensure token prediction flag is enabled for full-scale runs
   - Verify token logs are being saved alongside episode logs

### 📊 MEDIUM (Extends experiment)
5. **Run model-based baseline** - 30 episodes minimum
6. **Run negative controls** - Shuffled textualization
7. **Update analysis scripts** - Use new statistical methods (MI, distance correlation)

---

## 7. Hypothesis Status

| Hypothesis | Data Available | Can Test? | Preliminary Signal |
|------------|----------------|-----------|-------------------|
| **H-Token** | ❌ No | ❌ No | N/A - Missing surprisal |
| **H-Token1** (HotPot r>0.5) | ❌ No | ❌ No | N/A |
| **H-Token2** (Actor>Observer) | ❌ No | ❌ No | N/A |
| **H-Token3** (Environment gradient) | ❌ No | ❌ No | N/A |
| **H-Control** (Shuffled r<0.2) | ❌ No | ❌ No | N/A |
| **H-Baseline** (ModelBased best) | ❌ No | ❌ No | N/A |

**Verdict:** Zero hypotheses testable with current data.

---

## 8. Recommended Recovery Plan

### Phase 1: Fix Critical Bugs (2 hours)
1. Fix belief surprisal extraction in token_runner.py
2. Add better rate limiting / reduce parallelism
3. Test with 3 episodes (1 per environment)

### Phase 2: Validation Run (4 hours)
1. Run 30-episode pilot (10 per environment, Actor only)
2. Verify coupling r > 0.3 in at least HotPot
3. Generate diagnostic plots

### Phase 3: Full Experiment (8-12 hours)
1. Run full Actor experiments (130 episodes)
2. Run Observer experiments (130 episodes)
3. Run ModelBased baseline (30 episodes)
4. Run negative controls (30 episodes)

### Phase 4: Analysis (4 hours)
1. Compute all A1-A5 metrics
2. Test all 6 hypotheses
3. Generate figures
4. Write discussion

**Total Estimated Time to Completion: 18-22 hours**

---

## 9. Risk Assessment

### High Risk ⚠️
- **API costs:** Full experiment could cost $200-300 (OpenAI + Anthropic)
- **API failures:** Anthropic overload may continue
- **Weak coupling:** Even with fixed code, coupling might be < 0.3

### Medium Risk 📊
- **Time:** 18-22 hours assumes no major bugs
- **Compute:** May need to split runs across multiple days

### Low Risk ✅
- **Infrastructure:** Core systems are solid
- **Theory:** Well-documented, clear hypotheses
- **Reproducibility:** Seeds specified, deterministic

---

## 10. Data Quality Assessment

### Current Data Quality: ⭐⭐☆☆☆ (2/5 stars)

**Strengths:**
- Token predictions are clean and complete
- Logging format is correct
- No ground truth leakage detected

**Weaknesses:**
- Zero usable observations for coupling analysis
- Insufficient sample size (15 vs 520 episodes)
- Missing Observer and ModelBased agent data
- No control experiment data

**Recommendation:** Current data is **NOT suitable for publication**. Need to fix bugs and re-run experiments from scratch.

---

## Conclusion

The token prediction experiment is **well-designed but blocked by implementation bugs**. The infrastructure is 60% complete and the theoretical framework is solid, but:

1. **Critical bug:** Belief surprisal not being extracted correctly
2. **Data gap:** Only 12% of preregistered sample collected
3. **API issues:** Rate limiting blocking full-scale runs

**Next Step:** Fix the belief surprisal extraction bug in `token_runner.py` and re-run a small validation pilot (10 episodes) to verify coupling signal exists before investing in full-scale experiments.

**Estimated Time to Valid Results:** 2-3 days of focused work.

---

**Generated:** 2025-10-22
**Analyst:** Claude Code
**Data Sources:** results/pilot_token_fresh/, IMPLEMENTATION_STATUS.md
