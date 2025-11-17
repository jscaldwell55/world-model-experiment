# Hybrid Agent vs Baseline Comparison Analysis

**Date**: November 15, 2025
**Experiment**: 15-episode validation run comparison
**Environments**: HotPotLab, SwitchLight, ChemTile

---

## Executive Summary

The hybrid agent shows **highly task-dependent performance**, ranging from catastrophic failure (HotPot: 0.40) to perfect execution (ChemTile: 1.00). At **2.4x the cost** of baseline agents with **1.82 score/$ ratio** (vs 4.91 for actor), the hybrid approach requires careful deployment consideration.

### Overall Rankings by Environment

| Environment | 🥇 Winner | 🥈 Second | 🥉 Third | Hybrid Rank |
|-------------|-----------|-----------|----------|-------------|
| **HotPot** | Text_Reader (0.83) | Actor (0.76) | ACE (0.76) | **5th (0.40)** ⚠️ |
| **SwitchLight** | Actor (0.85) | **Hybrid (0.80)** | Observer (0.79) | **2nd** ✓ |
| **ChemTile** | **Hybrid (1.00)** | Actor (0.87) | Text_Reader (0.69) | **1st** ✓✓ |

---

## Detailed Performance Comparison

### 1. Overall Performance by Agent

| Agent | Avg Score | Avg Cost | Time | Score/$ | Best Use Case |
|-------|-----------|----------|------|---------|---------------|
| **Text_Reader** | 0.76 | $0.067 | 115s | **11.34** 🏆 | Passive reasoning from descriptions |
| **Observer** | 0.70 | $0.074 | 128s | **9.46** | Single-shot observation tasks |
| **Actor** | 0.83 | $0.169 | 370s | **4.91** | Active exploration required |
| **ACE** | 0.72 | $0.158 | 253s | **4.56** | Counterfactual reasoning |
| **Hybrid** | 0.73 | $0.402 | 573s | **1.82** ⚠️ | Complex inference without noise |

### 2. Reasoning Capability Breakdown

| Query Type | Observer | Text_Reader | Actor | ACE | **Hybrid** | Best Agent |
|------------|----------|-------------|-------|-----|------------|------------|
| **Interventional** | 0.61 | 0.74 | **0.79** | 0.72 | 0.65 | Actor |
| **Planning** | 0.82 | **0.92** | 0.87 | 0.83 | 0.81 | Text_Reader |
| **Counterfactual** | 0.75 | 0.28 | **0.85** | 0.38 | 0.83 | Actor |

**Key Finding**: Hybrid does NOT excel at any specific reasoning type despite higher complexity.

---

## Critical Failure Analysis: HotPot

### Why Hybrid Failed (0.40 vs 0.83 Text_Reader)

**Hybrid Behavior**:
- 8/10 actions: `measure_temp()`
- 2/10 actions: `wait()`
- **Problem**: Could not detect stove was OFF
  - Ground truth: `heating_rate = 0.0°C/s`, `actual_temp = 20°C`
  - Agent belief: `heating_rate = 0.177°C/s` (from ±1°C noise)
  - Kept measuring 20°C ± noise without testing hypotheses

**Text_Reader Behavior**:
- 1 action: Read environment description
- Reasoned purely from text without noisy measurements
- Applied physics knowledge directly
- **Result**: 8.3/10 questions correct

### Root Cause

```
Hybrid relies on ACTIVE EXPERIMENTATION → vulnerable to SENSOR NOISE
Text_Reader relies on PASSIVE READING → immune to measurement errors
```

When sensor noise masks the true signal:
- ✗ Active agents get confused by conflicting evidence
- ✓ Passive agents reason from clean textual descriptions

---

## Success Case Analysis: ChemTile

### Why Hybrid Succeeded (1.00 vs 0.87 Actor)

**Hybrid Behavior**:
- 10/10 actions: `inspect()` (never mixed!)
- Strategy: Pure information gathering + theoretical reasoning
- **Result**: Perfect 10/10 score without experiments

**Actor Behavior**:
- Mixed inspections and reactions
- Made 2 errors on:
  - Temperature control strategy
  - Probabilistic risk assessment after single trial

### Root Cause

```
Hybrid's dual-system enables SAFE THEORETICAL REASONING
Actor must balance exploration vs exploitation → occasional mistakes
```

When inference is sufficient:
- ✓ Hybrid can reason without risky experiments
- ✗ Actor feels compelled to test hypotheses actively

---

## Strategic Recommendations

### ✅ **Use Hybrid Agent When:**

1. **High-stakes reasoning** where experiments are costly/dangerous
   - Example: ChemTile explosions, medical diagnosis

2. **Rich prior knowledge** allows theoretical inference
   - Can leverage ACE's knowledge synthesis

3. **Counterfactual reasoning** is critical
   - Hybrid scores 0.83 vs 0.38 for ACE baseline

4. **Budget permits** 2.4x cost over actor baseline
   - Justified only if accuracy gain is critical

### ❌ **Avoid Hybrid Agent When:**

1. **Noisy sensors** require statistical hypothesis testing
   - HotPot: Noise confused dual-system decision making

2. **Active exploration** is necessary and sufficient
   - SwitchLight: Hybrid ≈ Actor, but 2.5x more expensive

3. **Cost efficiency** is important
   - Text_Reader achieves 11.34 score/$ vs Hybrid's 1.82

4. **Simple interventional tasks**
   - Actor outperforms Hybrid 0.79 vs 0.65

---

## Technical Deep Dive

### Token Usage Breakdown (Hybrid)

| Component | HotPot | SwitchLight | ChemTile | Average |
|-----------|--------|-------------|----------|---------|
| **Exploration** (ACE) | 21,110 | 27,372 | 19,152 | 22,545 |
| **Planning** (Hybrid) | 12,336 | 14,506 | 10,773 | 12,538 |
| **Evaluation** | 4,582 | 3,807 | 4,225 | 4,205 |
| **Curation** | 1,627 | 4,212 | 1,588 | 2,476 |
| **Total** | 39,655 | 49,897 | 35,738 | 41,763 |

**Overhead**: 2.4x more tokens than actor baseline due to:
- Generating multiple ACE candidates (2-6 per step)
- Scoring each candidate with hybrid evaluator
- Iterative refinement (1-3 rounds)

### Action Selection Patterns

| Environment | Dominant Action | Action Diversity | Assessment |
|-------------|----------------|------------------|------------|
| HotPot | `measure_temp()` 8/10 | **Low** | ❌ Repetitive, ineffective |
| SwitchLight | `flip_switch()` 10/10 | **Low** | ✓ Appropriate for task |
| ChemTile | `inspect()` 10/10 | **Low** | ✓ Safe, successful |

**Concern**: Hybrid shows low action diversity across all environments.

---

## Comparative Agent Strengths

### Text_Reader (Best Cost-Efficiency: 11.34 score/$)
- **Strength**: Passive reasoning from environment descriptions
- **Best At**: Planning questions (0.92)
- **Weakness**: Counterfactuals (0.28) - can't simulate actions not taken
- **Use Case**: When environment description is complete and accurate

### Observer (High Efficiency: 9.46 score/$)
- **Strength**: Single observation → inference
- **Best At**: Balanced across all question types
- **Weakness**: Limited by what can be inferred from one observation
- **Use Case**: Quick assessments, preliminary screening

### Actor (Best Overall: 0.83 avg)
- **Strength**: Active exploration with Bayesian updating
- **Best At**: Interventional (0.79), Counterfactual (0.85)
- **Weakness**: Can make mistakes during exploration
- **Use Case**: Default choice for interactive environments

### ACE (Specialized Reasoning: 0.72 avg)
- **Strength**: Generates diverse candidate actions
- **Best At**: Complex planning scenarios
- **Weakness**: Poor counterfactual reasoning (0.38)
- **Use Case**: When multiple hypotheses need evaluation

### Hybrid (Task-Dependent: 0.40-1.00)
- **Strength**: Theoretical reasoning without experiments (ChemTile)
- **Best At**: Safe inference in high-stakes scenarios
- **Weakness**: Noise sensitivity (HotPot), high cost
- **Use Case**: Complex inference + rich priors + high stakes

---

## Conclusions

1. **Hybrid is NOT a universal improvement** over simpler baselines
   - Performance range: 0.40 (failure) to 1.00 (perfect)
   - Cost: 2.4x actor, 6x text_reader
   - Appropriate for <20% of tasks

2. **Text_Reader dominates** on cost-efficiency (11.34 score/$)
   - Consider passive reasoning before active exploration

3. **Actor remains the best general-purpose agent**
   - 0.83 average score, 4.91 score/$
   - Reliable across all environments

4. **Hybrid excels in niche scenarios**:
   - ✓ High-stakes reasoning (ChemTile)
   - ✓ Theoretical inference possible
   - ✗ Noisy measurements (HotPot)
   - ✗ Simple exploration tasks (SwitchLight)

---

## Next Steps / Recommendations

### Immediate Actions

1. **Fix HotPot failure mode**
   - Implement noise-robust hypothesis testing
   - Add statistical significance checks for heating rate
   - Prevent repetitive ineffective actions

2. **Optimize cost**
   - Reduce candidate generation from 2-6 to 2-3
   - Cache ACE responses for similar states
   - Add early stopping when confidence > 0.9

3. **Action diversity**
   - Penalize repetitive actions without belief change
   - Force hypothesis testing (e.g., toggle stove) vs passive measurement

### Research Questions

1. **When does dual-system reasoning help?**
   - Formalize conditions: noise level, prior knowledge, action cost

2. **Can we predict when to use which agent?**
   - Meta-learner to select agent based on task features

3. **Is hybrid overhead reducible?**
   - Can we get hybrid benefits at 1.5x cost instead of 2.4x?

---

## Appendix: Full Results Table

### HotPotLab (Temperature Measurement)

| Agent | Overall | Interventional | Planning | Counterfactual | Cost | Time |
|-------|---------|----------------|----------|----------------|------|------|
| Text_Reader | **0.83** 🥇 | 0.76 | **1.00** | 0.75 | **$0.068** | **108s** |
| Actor | 0.76 | 0.64 | 0.97 | 0.75 | $0.181 | 620s |
| ACE | 0.76 | 0.68 | 0.97 | 0.65 | $0.169 | 331s |
| Observer | 0.74 | 0.58 | 1.00 | 0.75 | $0.075 | 123s |
| **Hybrid** | **0.40** ⚠️ | 0.24 | 0.67 | 0.50 | $0.404 | 577s |

### SwitchLight (Wiring Diagnosis)

| Agent | Overall | Interventional | Planning | Counterfactual | Cost | Time |
|-------|---------|----------------|----------|----------------|------|------|
| Actor | **0.85** 🥇 | **0.85** | 0.80 | **1.00** | $0.183 | 276s |
| **Hybrid** | **0.80** 🥈 | 0.71 | 0.75 | **1.00** | $0.461 | 658s |
| Observer | 0.79 | 0.70 | 0.85 | **1.00** | **$0.074** | **130s** |
| Text_Reader | 0.75 | 0.80 | 0.88 | 0.00 | $0.067 | 115s |
| ACE | 0.75 | 0.76 | 0.88 | 0.20 | $0.158 | 212s |

### ChemTile (Chemical Synthesis)

| Agent | Overall | Interventional | Planning | Counterfactual | Cost | Time |
|-------|---------|----------------|----------|----------------|------|------|
| **Hybrid** | **1.00** 🥇 | **1.00** | **1.00** | **1.00** | $0.341 | 484s |
| Actor | 0.87 | 0.89 | 0.85 | 0.80 | **$0.143** | **213s** |
| Text_Reader | 0.69 | 0.66 | 0.88 | 0.10 | $0.067 | 121s |
| ACE | 0.64 | 0.71 | 0.65 | 0.30 | $0.147 | 217s |
| Observer | 0.58 | 0.56 | 0.62 | 0.50 | $0.074 | 131s |

---

**Generated**: 2025-11-15
**Experiment Config**: `config_hybrid_validated.yaml`
**Code SHA**: `212b75f`
