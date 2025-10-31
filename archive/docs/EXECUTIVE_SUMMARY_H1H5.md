# Executive Summary: Pilot H1-H5 Results

**Date:** October 23, 2025
**Experiment:** pilot_h1h5_fixed
**Episodes:** 40 (4 agents × 2 environments × 5 episodes)
**Model:** Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)

---

## Bottom Line

✅ **Active interaction matters:** Actor agents outperformed passive observers by 12.7%
⚠️ **Belief convergence is unreliable:** Only 40-60% success rate at identifying causal structure
❌ **Counterfactual reasoning fails:** All agents scored 36-50% (barely above chance)
💰 **Massive token cost difference:** Active agents use 3.5× more tokens for 8-12% accuracy gain

---

## Performance Rankings

| Rank | Agent       | Score  | Best For                          | Cost/Episode |
|------|-------------|--------|-----------------------------------|--------------|
| 🥇   | Actor       | 75.3%  | Physical environments, exploration| ~22K tokens  |
| 🥈   | Model-Based | 73.2%  | Planning tasks, explicit reasoning| ~22K tokens  |
| 🥉   | Text-Reader | 68.4%  | Budget-constrained applications   | ~6K tokens   |
| 4th  | Observer    | 66.8%  | Pure description-based reasoning  | ~6.5K tokens |

**Key Insight:** The 3.3× token efficiency of Observer/Text-Reader agents makes them competitive for cost-sensitive applications, trading 8-12% accuracy for 70% cost reduction.

---

## Hypothesis Test Results

### H1: Observer vs Actor (Passive vs Active)
**Status:** ❌ **REJECTED**
- Actor beats Observer by 12.7% (75.3% vs 66.8%)
- Gap is largest in HotPotLab (physical manipulation): +20.6%
- Conclusion: **Active interaction is crucial for world model formation**

### H2: Text-Reader vs Observer (Modality)
**Status:** ✅ **SUPPORTED (WEAKLY)**
- Text-Reader slightly ahead: 68.4% vs 66.8% (+2.4%)
- Suggests modality matters less than interaction capability

### H3: Model-Based vs Actor (Explicit vs Implicit Models)
**Status:** ~ **NEUTRAL**
- Model-Based: 73.2%, Actor: 75.3% (-2.8%)
- Difference too small to declare winner
- Both approaches roughly equivalent

### H4: Surprisal Utilization
**Status:** ⚠️ **PARTIALLY WORKING**
- Surprisal is computed correctly (mean: 1.1-3.6 nats)
- BUT: High surprisal doesn't reliably lead to correct belief updates
- Example: Actor ep004 has surprisal events but converges to wrong layout

### H5: Counterfactual Reasoning
**Status:** ❌ **MAJOR FAILURE**
- All agents: 36.7-50.0% (barely above chance)
- Actor: 50%, Model-Based: 43%, Observer: 40%, Text-Reader: 37%
- **This is a fundamental capability gap**

---

## Critical Findings

### 1. Belief Convergence Crisis (SwitchLight)

Even with NO mechanical failures, agents fail to identify correct causal structure:

| Agent       | Convergence Rate | Example Failure                              |
|-------------|------------------|----------------------------------------------|
| Actor       | 40% (2/5)        | ep004: 99% confident in WRONG layout         |
| Model-Based | 60% (3/5)        | ep004: 0% confidence in correct layout       |

**Implication:** Current Bayesian update logic has systematic errors.

### 2. The Medium Question Paradox (HotPotLab)

Medium questions (30-47%) are HARDER than hard questions (86.7-93.3%):

| Difficulty | Actor | Model-Based | Observer | Text-Reader |
|------------|-------|-------------|----------|-------------|
| Easy       | 100%  | 95%         | 75%      | 80%         |
| Medium     | 47%   | 43%         | 33%      | 30%         |
| Hard       | 93%   | 87%         | 93%      | 93%         |

**Likely cause:** Medium questions involve temporal extrapolation (heating rates over time), which is legitimately harder than conceptual understanding.

### 3. Token Efficiency Trade-off

Cost-effectiveness analysis (tokens per percentage point of accuracy):

- **Observer:** 98 tokens/% (most efficient)
- **Text-Reader:** 92 tokens/% (BEST efficiency)
- **Actor:** 293 tokens/% (3.2× more expensive)
- **Model-Based:** 305 tokens/% (3.3× more expensive)

**Recommendation:** Use Text-Reader for budget applications, Actor for accuracy-critical tasks.

### 4. Confidence Calibration Issues

| Agent       | Avg Confidence | Avg Accuracy | Overconfidence |
|-------------|---------------|--------------|----------------|
| Actor       | 83.9%         | 75.3%        | +8.6%          |
| Model-Based | 82.5%         | 73.2%        | +9.3%          |
| Observer    | 82.0%         | 66.8%        | **+15.2%**     |
| Text-Reader | 65.7%         | 68.4%        | -2.7% ✓        |

**Only Text-Reader is well-calibrated** (slightly underconfident). Observer is dangerously overconfident.

---

## Query Type Analysis

### Planning Questions: STRONG (80-90%)
- "What should we do to determine X?"
- "What evidence would tell us Y?"
- All agents perform well on strategic reasoning

### Interventional Questions: MODERATE (66-75%)
- "If we do X, what will happen?"
- Forward prediction from current state
- Active agents have 6-9% advantage

### Counterfactual Questions: WEAK (37-50%)
- "If we HAD done X instead of Y, what would have happened?"
- **All agents struggle fundamentally**
- Requires mentally simulating alternative timelines

---

## Environment-Specific Insights

### HotPotLab (Deceptive Labels Environment)

**Challenge:** Labels say "Boiling!" but pot is actually cold (20°C)

**Results:**
- Higher surprisal (mean: 3.2-3.6 nats)
- Actor performs best (82.0%)
- Observer struggles (68.0%) - can't verify labels through action

**Key Learning:** Deceptive perceptual cues require active validation

### SwitchLight (Causal Structure Discovery)

**Challenge:** Determine if switch→light is direct or inverted, with possible relay failure

**Results:**
- More variable performance (std dev: 0.40-0.44)
- Belief convergence only 40-60% success
- Agents get confused by faulty relay (100% failure when relay broken)

**Key Learning:** Separating causal structure from mechanism failure is hard

---

## Actionable Recommendations

### Immediate Fixes (High Priority)

1. **Audit Bayesian Update Logic**
   - Current belief updates lead to systematic errors
   - Focus on SwitchLight episodes where agents converge to wrong layout with high confidence

2. **Increase Action Budget for SwitchLight**
   - Current: 10 actions
   - Recommended: 15-20 actions
   - Agents need more exploration to disambiguate layouts

3. **Add Counterfactual Scaffolding**
   - Explicit "imagine alternative timeline" prompting
   - Provide examples of counterfactual reasoning
   - Consider causal graph extraction + intervention simulation

4. **Review Question Difficulty Labels**
   - Medium questions in HotPotLab appear harder than hard questions
   - May need separate "temporal reasoning" difficulty axis

### Strategic Improvements (Medium Priority)

5. **Hybrid Agent Architecture**
   - Start with Observer (cheap initial analysis)
   - Escalate to Actor if confidence < threshold
   - Could reduce token cost by 40-60% while maintaining accuracy

6. **Epistemic Uncertainty Tracking**
   - Separate "I'm confident in X" from "I don't know"
   - Add mechanism to detect when belief is unreliable
   - Consider "restart from scratch" when epistemic state is corrupted

7. **Environment-Specific Strategies**
   - HotPotLab: Emphasize label skepticism, active measurement
   - SwitchLight: Add explicit causal graph reasoning, increase exploration

### Research Questions (Low Priority)

8. **Meta-Learning Across Episodes**
   - Currently each episode is independent
   - Could agents improve from episode 1 to episode 5?

9. **Causal Discovery Algorithms**
   - Replace informal Bayesian updates with formal methods (PC algorithm, FCI)
   - May improve belief convergence

10. **Richer Environments**
    - Test on 3+ variable systems
    - Include cyclic causal structures
    - Add confounders and mediators

---

## Cost-Benefit Analysis

### For Budget-Constrained Applications
**Recommended:** Text-Reader
- 68.4% accuracy
- 92 tokens per percentage point
- ~6.3K tokens/episode
- Best efficiency ratio

### For Accuracy-Critical Applications
**Recommended:** Actor
- 75.3% accuracy
- 293 tokens per percentage point
- ~22K tokens/episode
- Best absolute performance

### For Planning-Heavy Tasks
**Recommended:** Model-Based
- 90% on planning questions
- Explicit reasoning traces
- ~22K tokens/episode

### For Research/Development
**Recommended:** Actor + Model-Based comparison
- Helps understand implicit vs explicit representations
- Both generate belief states and surprisal

---

## Red Flags 🚩

1. **Belief convergence <60%**: Fundamental issue with causal learning
2. **Counterfactual reasoning <50%**: Major capability gap
3. **Medium questions harder than hard**: Possible labeling error or temporal reasoning weakness
4. **Observer overconfidence**: 15% calibration gap is dangerous for deployment

---

## Green Flags ✅

1. **Active learning works**: Actor clearly outperforms Observer
2. **Token efficiency achievable**: Observer/Text-Reader viable for budget apps
3. **Planning reasoning strong**: 80-90% across all agents
4. **Well-calibrated option exists**: Text-Reader has good confidence calibration

---

## Next Milestone Criteria

Before moving to full-scale experiment:

- [ ] Belief convergence rate >80% on SwitchLight (currently 40-60%)
- [ ] Counterfactual reasoning >70% (currently 36-50%)
- [ ] Fix Medium question difficulty paradox
- [ ] Validate fixes on new 10-episode pilot
- [ ] Document token cost vs accuracy trade-off curves

---

## Files Generated

**Analysis:**
- `PILOT_H1H5_FIXED_ANALYSIS.md` - Full technical analysis
- `EXECUTIVE_SUMMARY_H1H5.md` - This document
- `analyze_pilot_h1h5_fixed.py` - Analysis script
- `pilot_summary_table.py` - Table generator

**Visualizations:**
- `results/pilot_h1h5_fixed/figures/1_overall_performance.png`
- `results/pilot_h1h5_fixed/figures/2_performance_by_environment.png`
- `results/pilot_h1h5_fixed/figures/3_performance_by_query_type.png`
- `results/pilot_h1h5_fixed/figures/4_performance_by_difficulty.png`
- `results/pilot_h1h5_fixed/figures/5_token_usage.png`
- `results/pilot_h1h5_fixed/figures/6_confidence_calibration.png`
- `results/pilot_h1h5_fixed/figures/7_belief_trajectories.png`

**Raw Data:**
- `results/pilot_h1h5_fixed/raw/*.json` - 40 episode files

---

**Contact:** Analysis generated automatically
**Version:** 1.0
**Last Updated:** October 23, 2025
