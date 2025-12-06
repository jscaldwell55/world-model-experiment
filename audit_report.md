# Training Data Audit Report

**Total pairs analyzed:** 1294
**Unique pairs:** 27 (2.1%)
**Audit date:** 2025-12-03

---

## CRITICAL FINDING: Severe Data Duplication

**The training dataset contains only 27 unique instruction-response pairs** repeated across 1294 entries. This is the root cause of the observed quality degradation:

- **Degenerate repetition**: The model sees the same responses hundreds of times
- **Self-questioning loops**: Lack of variety causes mode collapse
- **Hallucinated values**: With only 27 examples, the model cannot generalize
- **Contradictory probabilities**: Multiple values for the same parameter (e.g., failure rates of 1.0%, 1.5%, 2.0%)

### Root Cause Analysis

The issue originates in the **playbook data**, not the training data generator:

| Domain | Total Observations | Unique Belief Sets | Unique Instructions |
|--------|-------------------|-------------------|---------------------|
| hot_pot | 178 | 5 | 8 |
| chem_tile | 204 | 5 | 7 |
| switch_light | 205 | 5 | 4 |

**The 587 playbook observations contain only 15 unique belief configurations across all domains.**

### Belief Diversity Analysis

**hot_pot:**
- Heating rates: `0.0` (90x), `2.5` (81x), `1.0` (7x) - only 3 values
- Power settings: OFF (90), HIGH (81), LOW (7)

**switch_light:**
- Wiring layouts: `layout_A @ 50%` (203x), `layout_A @ 85%` (1x), `layout_B @ 100%` (1x)
- Failure probabilities: `0.02` (202x), `0.01` (2x), `0.015` (1x)

**chem_tile:**
- Only 2 reaction configurations covered

---

## Summary Statistics

| Metric | hot_pot | chem_tile | switch_light | Total |
|--------|---------|-----------|--------------|-------|
| Total Pairs | 343 | 399 | 552 | 1294 |
| Unique Pairs | 8 | 7 | 12 | 27 |
| Duplication Rate | 97.7% | 98.2% | 97.8% | 97.9% |
| Avg instruction len | 62.8 | 40.0 | 46.2 | 49.7 |
| Avg response len | 89.7 | 107.9 | 155.0 | 117.5 |
| Avg score | 0.787 | 0.780 | 0.687 | 0.75 |

### Reliability Distribution

- **hot_pot:** HIGH (341), SYNTHETIC_HIGH (2)
- **chem_tile:** HIGH (399)
- **switch_light:** HIGH (552)

All pairs are marked HIGH reliability, which is misleading given the lack of diversity.

---

## Issue 1: Conflicting Values

**4 instructions have contradictory responses**

These are not random conflicts - they represent the only variation in the switch_light domain:

| Instruction | Response Variations |
|-------------|-------------------|
| "What wiring layout is most likely?" | layout_A @ 85%, layout_A @ 50%, layout_B @ 100% |
| "How reliable is the system?" | 1.0%, 1.5%, 2.0% failure rates |
| "How can I determine the wiring?" | 50%, 85%, 100% confidence |
| "What if a light doesn't respond?" | 1.0%, 1.5%, 2.0% failure rates |

**Impact:** Model learns to hedge or produce contradictory outputs for switch_light queries.

---

## Issue 2: Format Leakage

**0 responses with format issues** - The templated responses are clean.

---

## Issue 3: Repetition Patterns

### Exact Duplicates: 1267

The same 27 pairs are repeated an average of 48 times each.

| Pair | Domain | Occurrences |
|------|--------|-------------|
| "What wiring layout is most likely?" | switch_light | 138 |
| "How can I determine the wiring layout?" | switch_light | 138 |
| "How reliable is the switch-light system?" | switch_light | 138 |
| "What should I do if a light doesn't respond?" | switch_light | 138 |
| "What is the heating rate at HIGH power?" | hot_pot | 81 |
| "How long to heat water at HIGH power?" | hot_pot | 81 |
| "Which power setting heats faster?" | hot_pot | 81 |
| "What temp after 30 seconds at HIGH?" | hot_pot | 81 |
| "What happens mixing A with B?" | chem_tile | 57 |
| "How do I create compound C?" | chem_tile | 57 |

### Internal Repetition: 0

No character-level repetition within responses (they're too short).

### Boilerplate Phrases: 31 phrases repeated 3+ times

Top repeated phrases (appearing in nearly every response of their domain):
- "To determine the wiring layout, systematically flip switches..." (138x)
- "Multiple switch tests increase certainty" (138x)
- "Exercise caution and ensure proper safety measures" (114x)

---

## Issue 4: Hallucination Patterns

**0 responses with suspicious numeric values**

The values are consistent with domain physics (e.g., 2.5°C/s heating rate is reasonable). However, this is because:
1. Only ~15 unique value configurations exist
2. All values are templated from playbook data

---

## Issue 5: Sample Quality Review

The 5 samples per domain are all duplicates of each other:

### hot_pot (5 samples = 2 unique pairs)
- 2x "What is the heating rate at HIGH power?"
- 3x "How long to heat water at HIGH power?"

### chem_tile (5 samples = 4 unique pairs)
- 2x "What happens mixing C with B?"
- 1x each of safety, creation, temperature questions

### switch_light (5 samples = 4 unique pairs)
- 2x "What wiring layout is most likely?"
- Minor variation in values

**Assessment:** The responses are:
- Well-formed (complete sentences)
- Factually consistent
- Clear instructions
- **But catastrophically lacking in diversity**

---

## Cleanup Candidates

### Pairs to Remove: 1267
All duplicates should be removed, keeping one instance of each unique pair.

### Pairs to Fix: 4
The conflicting switch_light pairs need resolution - either:
1. Keep only the highest-confidence version
2. Merge into a single response acknowledging uncertainty

### Patterns to Filter: 5
Boilerplate phrases that reduce response diversity.

---

## Root Cause Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE ISSUES                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Episode Runs (100s)                                          │
│        │                                                        │
│        ▼                                                        │
│   Belief Extraction ─────► Only 5 unique beliefs per domain    │
│        │                   (ground truth environments have     │
│        │                    limited parameter space)           │
│        ▼                                                        │
│   Playbook Storage ──────► 587 observations, 15 unique configs │
│        │                                                        │
│        ▼                                                        │
│   Training Data Gen ─────► 4-8 pairs per belief config         │
│        │                   × 587 observations = 1294 pairs     │
│        │                   but only 27 unique!                 │
│        ▼                                                        │
│   LoRA Training ─────────► Massive overfitting to 27 examples  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Recommendations

### Option A: Clean Existing Data (Quick Fix)

1. **Deduplicate immediately**
   ```bash
   python scripts/generate_training_data.py --min-reliability HIGH --output data/training_pairs_deduped.json
   ```
   This will produce 27 unique pairs.

2. **Resolve conflicts** by keeping highest-score version:
   - Final count: ~23-25 pairs

3. **Expected outcome:** Still insufficient for LoRA training (need 100-500+ diverse pairs)

### Option B: Regenerate Training Data (Recommended)

1. **Expand the domain parameter space:**
   - hot_pot: Add MEDIUM power, vary starting temperatures, add cooling scenarios
   - chem_tile: Add more compounds, reaction chains, temperature effects
   - switch_light: Add more layout configurations, multi-switch scenarios

2. **Diversify instruction templates:**
   - Current: 4-8 templates per domain
   - Target: 20-30 templates per domain with variations

3. **Add synthetic augmentation:**
   - Paraphrase existing instructions
   - Add edge case questions
   - Include negative examples ("What happens if I do X wrong?")

4. **Target:** 200-500 diverse, high-quality pairs per domain

### Option C: Hybrid Approach (Pragmatic)

1. **Clean existing 27 pairs** as a base
2. **Add synthetic pairs** using the training data generator with expanded templates
3. **Data augmentation:**
   - Instruction paraphrasing (3x multiplier)
   - Response variation templates
   - Edge case generation

---

## Decision Matrix

| Action | Impact | Effort | Priority |
|--------|--------|--------|----------|
| Deduplicate training data | Critical | Low | **Immediate** |
| Resolve value conflicts | High | Low | Immediate |
| Expand domain parameters | Critical | High | Required |
| Add instruction templates | High | Medium | Required |
| Synthetic augmentation | High | Medium | Recommended |
| Collect more episodes | High | High | Long-term |

---

## Conclusion

**The training data is fundamentally broken.** The 97.9% duplication rate means the model is essentially memorizing 27 examples rather than learning generalizable domain knowledge.

**Recommendation:** Do NOT proceed with graduation POC until:
1. Training data is deduplicated (1294 → 27 pairs)
2. At minimum 200+ diverse pairs are generated per domain
3. Instruction diversity is increased 5-10x

The current data will produce a model that:
- Repeats the same phrases verbatim
- Cannot handle novel questions
- Has inconsistent probability outputs
- Exhibits the exact symptoms described in the problem statement
