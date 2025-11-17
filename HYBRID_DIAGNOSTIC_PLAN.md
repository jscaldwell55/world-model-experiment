# Hybrid Agent Diagnostic & Fix Implementation Plan

**Date:** November 15, 2025
**Status:** Phase 1 - Root Cause Analysis & Initial Fix

---

## Executive Summary

Based on the critical analysis of ACE Baseline vs Hybrid performance, I've identified the root causes and developed a systematic implementation plan to diagnose and fix the issues.

**Key Finding:** The hybrid agent's decision metadata was not being saved to episode logs due to a bug in how `get_belief_state()` was implemented. This prevented proper analysis of the selection mechanism and ACTOR scoring quality.

---

## Critical Issues Identified

### 1. ❌ Decision Metadata Not Saved (BLOCKING BUG - FIXED)
- **Problem:** Hybrid agent's decision metadata (candidates, scores, selection) was lost during episode logging
- **Root Cause:** Runner overwrites `agent_step.belief_state` with `agent.get_belief_state()` after step creation
- **Impact:** Unable to analyze selection correctness or ACTOR score quality
- **Status:** ✅ **FIXED** - Modified `hybrid_agent.py` to preserve decision metadata in `get_belief_state()`

### 2. ⚠️ HotPotLab Catastrophic Failure (-17 points)
- **Problem:** Hybrid performs worse than ACE in HotPotLab (66% vs 83%)
- **Theory:** ACTOR scoring may be anti-correlated with actual success
- **Status:** 🔍 **NEEDS DIAGNOSIS** - Requires re-running diagnostic with fixed agent

### 3. ⚠️ Missing ACTOR Baseline
- **Problem:** Cannot determine if ACTOR alone would beat Hybrid
- **Impact:** Don't know if selection mechanism adds value over pure ACTOR
- **Status:** 📋 **PLANNED** - Run 15 episodes with same seeds

### 4. ⚠️ Small Sample Size (N=5)
- **Problem:** Cannot claim statistical significance with only 5 episodes per environment
- **Impact:** Results may be due to random variance
- **Status:** 📋 **PLANNED** - Power analysis and expanded sample

---

## Implementation Plan

### Phase 1: Emergency Diagnostics (Current - Nov 15) ✅

**Objective:** Fix blocking bug and establish diagnostic infrastructure

#### Completed:
1. ✅ Created comprehensive diagnostic script (`scripts/diagnose_hybrid.py`)
2. ✅ Identified root cause: decision metadata not being preserved
3. ✅ Fixed `hybrid_agent.py` to store `_latest_decision` and include in `get_belief_state()`
4. ✅ Updated `get_belief_state()` to merge ACTOR belief with hybrid decision metadata

#### Changes Made:
```python
# agents/hybrid_agent.py

# Added instance variable to track latest decision
self._latest_decision: Optional[Dict] = None

# Modified get_belief_state() to include decision metadata
def get_belief_state(self) -> dict:
    belief_state = self.actor.get_belief_state()
    if self._latest_decision is not None:
        belief_state['hybrid_decision'] = self._latest_decision
    return belief_state

# Store decision metadata after selection
self._latest_decision = decision_metadata  # Line 384
```

### Phase 2: Validation & Deep Diagnosis (Next - Nov 15)

**Objective:** Verify fix and run comprehensive diagnostics on existing results

#### Steps:
1. ⏳ Run quick test episode to verify decision metadata is now saved
2. ⏳ Re-run diagnostic analysis on existing hybrid results (may need to re-run episodes with fixed agent)
3. ⏳ Analyze HotPotLab planning failures in detail:
   - What were the 5 ACE candidates for failed decisions?
   - What scores did ACTOR assign?
   - Which candidate was selected?
   - What would ACE have selected naturally?
4. ⏳ Compute ACTOR score correlation with episode outcomes:
   - Correlation > 0.3: ACTOR scoring is helpful
   - Correlation ≈ 0: ACTOR scoring is random/neutral
   - Correlation < 0: ACTOR scoring is **broken** (anti-correlated)
5. ⏳ Measure candidate diversity (std dev of scores per decision)

### Phase 3: ACTOR Baseline (Priority)

**Objective:** Establish whether ACTOR alone beats Hybrid

#### Execution Plan:
- Run ACTOR agent for 15 episodes (5 per environment)
- Use **same seeds** as ACE/Hybrid runs for fair comparison:
  - ChemTile: seeds [200, 201, 202, 203, 204]
  - HotPotLab: seeds [42, 43, 44, 45, 46]
  - SwitchLight: seeds [100, 101, 102, 103, 104]
- Cost estimate: ~$2-3 (similar to ACE baseline)
- Timeline: ~2 hours runtime

#### Expected Outcomes:
- **If ACTOR > Hybrid:** Selection mechanism is actively harmful
- **If ACTOR ≈ Hybrid:** Selection mechanism adds no value
- **If Hybrid > ACTOR:** Selection mechanism works, but has environment-specific issues

### Phase 4: Statistical Analysis

**Objective:** Determine if results are statistically significant

#### Analysis:
1. Compute p-values for ACE vs Hybrid differences (per environment)
2. Run power analysis: what N do we need for p < 0.05?
3. If needed, run additional episodes to reach statistical power

#### Current Sample Sizes:
- N=5 per environment (15 total per agent)
- Likely need N≥30 per environment for robust claims

### Phase 5: Question Difficulty Analysis

**Objective:** Explain why ACE got 62.5% on ChemTile planning instead of expected 12.3%

#### Investigation:
1. Extract planning questions from current episodes
2. Compare with planning questions from preregistered study
3. Check for:
   - Different question templates
   - Model improvements (Claude Sonnet 4.5 vs earlier)
   - Prompt changes
   - Measurement differences

### Phase 6: Targeted Fixes (Based on Diagnosis)

**Scenario A: ACTOR Scoring is Broken**
- Debug ACTOR's evaluation logic in `_score_with_actor()`
- Check if belief state updates are correct
- Verify ACTOR understands environment dynamics
- Consider using different scoring criteria

**Scenario B: Selection Mechanism Has Bug**
- Verify `max(scores)` selection is correct
- Add validation and error checking
- Add comprehensive unit tests

**Scenario C: Candidate Diversity Insufficient**
- Increase temperature variation in candidate generation
- Generate more candidates (N=10 instead of 5)
- Add diversity penalty to selection
- Use different sampling strategies

**Scenario D: Environment-Specific Issues**
- Investigate why ACTOR works in ChemTile but fails in HotPotLab
- May need environment-specific scoring logic
- Consider hybrid approach: ACE for planning, ACTOR for interventional

---

## Diagnostic Script Usage

### Run Diagnostics:
```bash
# After re-running hybrid episodes with fixed agent
python scripts/diagnose_hybrid.py --results results/hybrid_nov15/

# Output will show:
# - Selection mechanism correctness
# - ACTOR score distributions
# - Candidate diversity metrics
# - Score vs outcome correlations
# - Planning failure analysis
```

### Key Metrics to Watch:
1. **Selection Issues:** Should be 0 if mechanism is correct
2. **Score Correlation:** Should be positive if ACTOR scoring is useful
3. **Candidate Diversity:** Should be >0.1 (std dev of scores)
4. **Planning Success Rate:** Compare to ACE baseline

---

## Timeline & Priorities

### Immediate (Today - Nov 15):
- [x] Fix decision metadata preservation bug
- [ ] Run 1 test episode to verify fix
- [ ] Document findings

### High Priority (This Week):
- [ ] Run ACTOR baseline (15 episodes) - **CRITICAL FOR DIAGNOSIS**
- [ ] Re-run diagnostic analysis with fixed agent
- [ ] Identify root cause of HotPotLab failure
- [ ] Create diagnostic report with findings

### Medium Priority (Next Week):
- [ ] Implement targeted fixes based on diagnosis
- [ ] Run expanded sample (N=30 per environment)
- [ ] Statistical validation
- [ ] Question difficulty analysis

### Low Priority (Future):
- [ ] Environment-specific optimizations
- [ ] Hybrid architecture refinements
- [ ] Documentation and publication prep

---

## Success Criteria

### Phase 1 (Current):
- ✅ Decision metadata preservation bug fixed
- ✅ Diagnostic infrastructure in place
- ✅ Can extract and analyze decision data

### Phase 2 (Diagnosis):
- [ ] Identified root cause of HotPotLab failure
- [ ] Confirmed whether ACTOR scoring is helpful or harmful
- [ ] Measured candidate diversity
- [ ] Determined if selection mechanism is correct

### Phase 3 (Baseline):
- [ ] ACTOR baseline data collected (15 episodes)
- [ ] Can compare: ACE vs ACTOR vs Hybrid
- [ ] Understand value-add of hybrid selection

### Phase 4 (Fix):
- [ ] Hybrid performance restored or improved in all environments
- [ ] No catastrophic regressions (all deltas within ±5%)
- [ ] Cost-benefit justified (performance gain > cost increase)

---

## Open Questions

### Q1: Why Does Hybrid Destroy HotPotLab Performance?
**Hypotheses:**
- A) ACTOR scores are inverted (high score = bad strategy)
- B) Candidate diversity is too low (all similar strategies)
- C) Selection has a bug
- D) HotPotLab dynamics don't match ACTOR priors

**Answer Method:** Run diagnostic analysis with fixed agent

### Q2: Would ACTOR Alone Beat Hybrid?
**Answer Method:** Run ACTOR baseline (15 episodes)

### Q3: Are Results Statistically Significant?
**Answer Method:** Power analysis + p-value calculation

### Q4: Why Is ACE Planning 62.5% Instead of 12.3%?
**Answer Method:** Question difficulty comparison

---

## Risk Assessment

### High Risk Items:
1. ⚠️ **Results may not be reproducible** with larger N
2. ⚠️ **Fundamental architectural incompatibility** between ACE and ACTOR
3. ⚠️ **Cost may be prohibitive** for expanded sample (N=30 → ~$60)

### Mitigation Strategies:
1. Start with ACTOR baseline before expanding sample
2. Consider targeted fixes per environment
3. May need to revise hybrid architecture if fundamentally broken

---

## Next Steps (Immediate Actions)

1. **Run test episode** with fixed hybrid agent to verify decision metadata is saved
2. **If successful**, re-run full 15-episode hybrid benchmark with fixed agent
3. **Run diagnostic analysis** on new results
4. **Start ACTOR baseline** run (can run in parallel with step 2)
5. **Document findings** in comprehensive diagnostic report
6. **Decide on fixes** based on diagnostic results

---

## Notes

- All code changes are in `agents/hybrid_agent.py`
- Diagnostic script is `scripts/diagnose_hybrid.py`
- No changes needed to runner or other components
- Fix is backward-compatible (won't break existing agents)

---

**Contact:** Jay Caldwell | **Date:** November 15, 2025
**Status:** Diagnostic infrastructure complete, awaiting validation
