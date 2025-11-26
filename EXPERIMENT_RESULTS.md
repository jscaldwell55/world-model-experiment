# Fidelity Validation Experiment Results

**Date**: 2025-11-17
**Status**: ✗ EXPERIMENTS FAILED - DO NOT PROCEED WITH OC → FTB
**Decision**: PAUSE offline consolidation integration until critical issues are fixed

---

## Executive Summary

Ran three critical experiments to validate whether offline consolidation (OC) helps or creates circular reasoning problems. **All experiments failed** except the safety gate test, confirming the concerns raised in `FIDELITY_CRITICAL_ISSUES.md`.

**Bottom line**: Offline consolidation in its current form **amplifies model bias** rather than correcting it. Synthetics provide zero benefit and fail to match real data.

---

## Data Collection

✅ Successfully collected **30 HotPot episodes**
- 11 HIGH reliability episodes (36.7%)
- 19 LOW reliability episodes (63.3%)
- Source: `results/fidelity_data_30ep/raw/`
- Playbook: `memory/domains/hot_pot/playbook_30ep.json`

---

## Experiment Results

### ❌ Experiment A: Do Synthetics Help?

**Goal**: Test if synthetic data from OC improves prediction accuracy

**Results**:
- Real-only accuracy: **70.5%**
- With synthetics: **70.3%**
- Improvement: **-0.2%** (actually worse!)
- Synthetics generated: 3
- Training episodes: 20

**Conclusion**: Synthetics provide **NO benefit**. OC adds complexity without value.

**Recommendations**:
- Check fidelity scoring for circular reasoning ✓ (found it!)
- Validate world model accuracy before generating synthetics
- Don't use these synthetics for fine-tuning

---

### ❌ Experiment B: Cross-Validation of Synthetics

**Goal**: Validate that synthetics match held-out real data

**Results**:
- Mean error: **23.9%** (threshold: 15%)
- Pass rate: **0%** (0/11 episodes passed)
- All synthetic predictions diverged significantly from reality

**Critical Finding**: Synthetic predictions were nearly **identical** despite varying real outcomes:
- 10/11 synthetics predicted: **37.96°C**
- 1/11 predicted: **39.11°C**
- Real temps ranged: **32°C to 50°C**

**Root Cause**: World model averages beliefs **without conditioning on context** (power setting):
```
Averaged heating rate ≈ 1.5°C/s
→ Predicts ~38°C for everything

But real outcomes vary:
- HIGH power: 50°C (error: 24%)
- LOW power: 32°C (error: 19%)
- OFF: 20°C (error: 90%)
```

**Conclusion**: World model is **not generalizing** to counterfactuals. Beliefs are context-agnostic.

**Recommendations**:
- Fix belief extraction to preserve conditional structure: `P(rate | power)` not `P(rate)`
- Update WorldModelSimulator to condition on context
- Verify SimpleWorldModel stores context-aware beliefs

---

### ✅ Experiment C: Wrong Model Detection

**Goal**: Test if OC detects and rejects inaccurate world models

**Results**:
- OC gate status: **FAIL** (correctly rejected)
- Mean error: **48.9%**
- Synthetics generated: 0

**Conclusion**: Quality gate is **working as intended**. Safety mechanisms caught the bad model.

---

## Overall Assessment

### Experiment Summary
| Experiment | Status | Key Finding |
|------------|--------|-------------|
| A: Synthetics Help? | ❌ FAIL | No accuracy improvement |
| B: Cross-Validation | ❌ FAIL | 24% error, 0% pass rate |
| C: Wrong Model Detection | ✅ PASS | Safety gate working |

**Overall**: ✗ CRITICAL FAILURES - DO NOT PROCEED

---

## Key Problems Identified

### 1. **No Performance Benefit**
Synthetics don't improve prediction accuracy. Adding them to training data made accuracy slightly worse (-0.2%).

### 2. **Poor Generalization**
24% error when validating against real data. Synthetics fail to capture true dynamics.

### 3. **Circular Reasoning Confirmed**
- Fidelity scores: **94.7%** (high!)
- But synthetics don't match reality (24% error)
- **Why?** Synthetics are generated AND scored by the same (potentially wrong) model

### 4. **Context Loss in Belief Extraction**
The critical architectural flaw:

**What's stored in playbook:**
```json
{
  "beliefs": {"heating_rate_mean": 2.5},
  "context": {"power_setting": "HIGH"}
}
```

**What offline consolidation does:**
```python
# Averages ALL heating rates → loses context
avg_rate = mean([2.5, 2.5, 1.0, 0.0, ...])  # ≈ 1.5
# Result: uniform predictions regardless of power setting
```

**What it should do:**
```python
# Condition on context
if power == "HIGH":
    rate = 2.5
elif power == "LOW":
    rate = 1.0
else:
    rate = 0.0
```

---

## Root Cause Analysis

### The Circular Reasoning Loop

```
1. Extract beliefs from episodes (potentially wrong)
     ↓
2. Generate synthetics using those beliefs
     ↓
3. Score synthetics with same beliefs
     ↓
4. High fidelity! ✓ (but meaningless)
     ↓
5. Synthetics don't match real data ✗
```

The fidelity metric measures **self-consistency**, not **accuracy**.

### Why World Model Fails

The experiments revealed that belief consolidation:
1. Averages beliefs across different contexts
2. Loses conditional structure: `P(heating_rate | power_setting)`
3. Produces context-agnostic predictions
4. Fails to generalize to counterfactual scenarios

---

## Decision: PAUSE OC Integration

**DO NOT PROCEED** with OC → Fine-Tuning Bridge until:

### Critical Fixes Required

1. **Fix Belief Extraction**
   - Preserve conditional structure from SimpleWorldModel
   - Don't extract from ground truth, use agent's actual beliefs
   - Maintain context dependencies: `P(parameter | context)`

2. **Implement External Validation**
   - Replace circular fidelity scoring
   - Validate against held-out real data
   - Use proper cross-validation with context conditioning

3. **Fix World Model Simulator**
   - Make predictions context-aware
   - Condition on power setting (or other context)
   - Don't average beliefs across different contexts

4. **Verify SimpleWorldModel**
   - Check if agent stores context-conditioned beliefs
   - If not, this needs fixing at the agent level
   - If yes, fix extraction to preserve this structure

### Alternative Approaches to Consider

1. **Direct Episode Replay** (no consolidation)
   - Use raw episodes for fine-tuning
   - Simpler, no belief averaging issues
   - Lose diversity benefits but avoid bias amplification

2. **Context-Stratified Consolidation**
   - Separate beliefs by context (HIGH/LOW/MIXED power)
   - Generate synthetics within each stratum
   - More accurate but more complex

3. **Ensemble Approach**
   - Train separate world models per context
   - Use appropriate model based on query context
   - More expensive but better generalization

---

## Files Generated

### Data
- `results/fidelity_data_30ep/raw/*.json` - 30 collected episodes
- `memory/domains/hot_pot/playbook_30ep.json` - Processed playbook
- `results/fidelity_validation_output.log` - Full experiment output

### Scripts
- `scripts/process_episodes_to_playbook.py` - Episode to observation converter
- `experiments/fidelity_validation.py` - Three validation experiments

### Documentation
- `EXPERIMENT_PLAN.md` - Original experiment design
- `FIDELITY_CRITICAL_ISSUES.md` - Problem analysis
- `EXPERIMENT_RESULTS.md` - This file

---

## Next Steps

### Immediate (Do Not Proceed with OC)
- [x] Document experiment results
- [x] Identify root causes
- [x] Make go/no-go decision → **NO GO**

### Before Resuming OC Work
1. Investigate SimpleWorldModel belief structure
2. Fix belief extraction to preserve context
3. Implement context-aware WorldModelSimulator
4. Add external validation (not circular fidelity)
5. Re-run validation experiments
6. Verify cross-validation error < 10%

### Alternative Path (If OC unfixable)
1. Implement direct episode replay for FTB
2. Skip consolidation layer entirely
3. Focus on episode selection/weighting instead

---

## Lessons Learned

1. **Fidelity ≠ Accuracy**: High self-consistency doesn't mean real-world validity
2. **Context Matters**: Averaging beliefs across contexts destroys predictive power
3. **Validate Externally**: Always test against held-out real data
4. **Safety Gates Work**: The quality gate correctly caught bad models
5. **Experiments Saved Us**: Without these experiments, we'd have shipped biased synthetics

---

## References

- `FIDELITY_CRITICAL_ISSUES.md` - Original problem identification
- `EXPERIMENT_PLAN.md` - Experiment design and methodology
- `memory/offline_consolidation.py` - OC implementation (lines 122-143 have the averaging bug)
- `experiments/fidelity_validation.py` - Validation experiments (lines 365-377 show context loss)
