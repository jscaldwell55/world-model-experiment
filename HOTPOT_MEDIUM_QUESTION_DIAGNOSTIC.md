# HotPot Medium Question Collapse - Root Cause Analysis

## Executive Summary

**Issue:** HotPot medium questions achieved only 11.1% accuracy (1/9 correct) compared to 91.7% on easy and 88.9% on hard questions.

**Root Cause:** The agent learned an **incorrect heating rate** (1.08°C/s instead of the true 2.5°C/s) which caused it to systematically underestimate temperatures, leading to failures on temperature prediction questions.

**Critical Problem:** The memory consolidation system created a "bad belief trap" where:
1. First episodes learned very wrong values (~1.0°C/s)
2. This became the consolidated baseline
3. Later episodes with better estimates (1.45°C/s) were **rejected as outliers**
4. The wrong belief persisted and grew stronger with confidence

---

## The Numbers

### Consolidated Belief vs Ground Truth

| Parameter | Consolidated Belief | Ground Truth | Error |
|-----------|---------------------|--------------|-------|
| **heating_rate_mean** | **1.079°C/s** | **2.5°C/s** | **-57% (too low!)** |
| heating_rate_std | 0.216 | ~variable | - |
| measurement_noise | 2.0 | 2.0 | ✓ Correct |
| base_temp | 20.0 | 20.0 | ✓ Correct |

### Individual Episode Learning

| Episode | Learned Rate | Consolidated Status | Distance from Truth |
|---------|--------------|---------------------|---------------------|
| hot_pot_20251116_195624 | 0.998°C/s | ✓ **ACCEPTED** | -60% error |
| hot_pot_20251116_203426 | 1.096°C/s | ✓ **ACCEPTED** | -56% error |
| hot_pot_20251116_203846 | 1.449°C/s | ✗ **REJECTED** (outlier) | -42% error (best estimate!) |

**The Problem:** The episode with the *best* estimate (1.449°C/s, only 42% error) was rejected as an outlier because it differed too much from the *worse* baseline estimates!

---

## Why Did Episodes Learn Wrong Heating Rates?

### Problem: Mixed Observations During Exploration

The agent explores by toggling the stove between high and low settings, creating observations with different heating rates:

#### Example from Episode 1

```
Step 0 (t=1s):  Measure temp → 16.88°C
Step 1:         Wait 2s
Step 2 (t=4s):  Measure temp → 19.94°C  [Rate: +1.02°C/s, stove initially on]
Step 3 (t=5s):  Toggle stove → "dim" (LOW)
Step 4:         Wait 3s
Step 5 (t=9s):  Measure temp → 22.76°C  [Rate: +0.56°C/s, stove on LOW]
Step 6 (t=10s): Toggle stove → "bright" (HIGH)
Step 7:         Wait 2s
Step 8 (t=13s): Measure temp → 29.57°C  [Rate: +1.70°C/s, mixed LOW→HIGH]
Step 9 (t=14s): Measure temp → 37.82°C  [Rate: +8.25°C/s, noisy measurement]
```

**Observed Rates:**
- Stove on HIGH: ~2.5°C/s (true value, rarely observed cleanly)
- Stove on LOW: ~0.5-1.0°C/s
- **Mixed/Averaged: ~1.0-1.1°C/s** ← What the agent learns

### The Agent's Model Limitation

The SimpleWorldModel learns a **single** `heating_rate_mean` parameter, but the true model should be:

```
heating_rate_high ≈ 2.5°C/s
heating_rate_low  ≈ 0.5-1.0°C/s
cooling_rate      ≈ -0.1°C/s (when off)
```

By averaging across different power settings, the agent learns a value that doesn't represent *any* actual heating rate.

---

## Impact on Medium Questions

### Question Failure Pattern

All 3 episodes failed the same medium questions:

#### Question 2: "After toggling stove to low and waiting 40s, what temperature?"

| Episode | Agent Prediction | Score | Likely Correct Answer |
|---------|------------------|-------|-----------------------|
| 1 | 48°C | 0.30 | ~60-80°C? |
| 2 | 65°C | 0.30 | ~60-80°C? |
| 3 | 47°C | 0.30 | ~60-80°C? |

All predictions scored 0.30 (very low), indicating significant errors.

#### Question 3: "If stove on high for 1 minute, temp above 100°C?"

**With TRUE heating rate (2.5°C/s):**
```
Starting: 20°C
After 60s: 20 + (2.5 × 60) = 170°C
Answer: YES ✓
```

**With LEARNED heating rate (1.1°C/s):**
```
Starting: 20°C
After 60s: 20 + (1.1 × 60) = 86°C
Answer: NO ✗
```

| Episode | Agent Answer | Correct? | Score |
|---------|--------------|----------|-------|
| 1 | NO | ✗ | 0.50 |
| 2 | YES | ✓ | 1.00 |
| 3 | NO | ✗ | 0.50 |

**Result:** 2/3 episodes failed because they used the incorrect heating rate and predicted temperatures below 100°C.

---

## The "Bad Belief Trap"

### How It Happens

1. **Episode 1** learns heating_rate = 0.998°C/s
   - Mixed observations (high + low power)
   - This becomes the first consolidated belief

2. **Episode 2** learns heating_rate = 1.096°C/s
   - Similar mixed observations
   - Close enough to Episode 1, so it's accepted
   - Consolidated belief: (0.998 + 1.096) / 2 ≈ 1.047°C/s

3. **Episode 3** learns heating_rate = 1.449°C/s
   - Better estimate! (closer to true 2.5)
   - **REJECTED as outlier:** z-score = 8.25
   - Reason: "Differs too much from mean=1.047, std=0.049"
   - Consolidated belief stays at ~1.079°C/s

### Why This Is Critical

- **Confidence grows with each episode** (even with wrong beliefs)
- **Outlier detection becomes more aggressive** as confidence increases
- **Better observations are rejected** because they differ from the wrong baseline
- **The wrong belief becomes entrenched** and harder to correct

### Outlier Detection Parameters

From `memory/domain_memory.py:_is_outlier()`:
```python
# Reject if z-score > 2.5 standard deviations
threshold = 2.5
z_score = abs(new_value - mean) / std

if z_score > threshold:
    # Reject as outlier
```

**Problem:** With a very small std (0.049 from 2 similar wrong values), even slightly better estimates get huge z-scores and are rejected.

---

## Why Medium Questions Failed But Hard Questions Didn't

### Medium Questions (11.1% accuracy)
- Focus on **quantitative temperature predictions**
- Require accurate heating rate to calculate final temperatures
- Examples:
  - "After 40s on low, what temperature?" (requires precise rate)
  - "After 1 min on high, above 100°C?" (requires precise rate)

### Hard Questions (88.9% accuracy)
- Focus on **conceptual understanding** and **qualitative reasoning**
- Don't require precise numerical predictions
- Examples:
  - "What's the safest way to check if pot is hot?" → Use measure_temp()
  - "What causes burns?" → High temperature + contact duration

### Easy Questions (91.7% accuracy)
- Simple factual questions
- Don't require temperature calculations

**The incorrect heating rate doesn't affect conceptual questions, only numerical predictions!**

---

## Evidence Summary

### From Consolidated Beliefs
```json
{
  "heating_rate_mean": {
    "value": 1.0793,  // Should be ~2.5!
    "confidence": 0.3,
    "episode_count": 2,
    "source_episodes": [
      "hot_pot_20251116_195624",  // 0.998°C/s (wrong)
      "hot_pot_20251116_203426"   // 1.096°C/s (wrong)
    ],
    "excluded_observations": [
      {
        "value": 1.4487,  // Better estimate!
        "reason": "z-score=8.25 (mean=1.047, std=0.049)",
        "episode_id": "hot_pot_20251116_203846"  // REJECTED
      }
    ]
  }
}
```

### From Test Results
- **Total medium questions:** 9 (3 episodes × 3 questions each)
- **Correct:** 1/9 (11.1%)
- **Failed pattern:** Temperature prediction questions
- **Score pattern:** Consistently 0.30-0.50 (indicating systematic underestimation)

---

## Recommendations

### Immediate Fixes

1. **Relax Outlier Detection Threshold**
   - Current: z-score > 2.5
   - Suggested: z-score > 3.5 or 4.0
   - Or use percentage-based threshold instead of standard deviations

2. **Increase Minimum Sample Size for Outlier Detection**
   - Don't reject outliers until you have at least 5-10 episodes
   - 2 episodes is too few to establish a reliable distribution

3. **Context-Aware Heating Rate Learning**
   - Learn separate rates for different stove power levels
   - Model: `heating_rate_high`, `heating_rate_low`, `cooling_rate_off`
   - Don't average across different contexts

### Long-Term Improvements

4. **Belief Confidence Decay**
   - Reduce confidence over time to allow corrections
   - Don't let early wrong beliefs become too entrenched

5. **Meta-Learning Signal**
   - Track question performance by belief
   - If questions start failing, reduce confidence in related beliefs
   - Use test scores as feedback to adjust beliefs

6. **Outlier Detection Based on Ground Truth Distance**
   - When available, compare to ground truth
   - Don't reject observations that are closer to truth than current belief

7. **Exploration Strategy**
   - Keep stove at one power level during initial learning
   - Only toggle after basic heating rate is established
   - Avoid mixing observations from different contexts

---

## Files to Investigate

- **Memory consolidation:** `memory/domain_memory.py` (lines 93-228)
  - `_update_consolidated_beliefs()` - Where consolidation happens
  - `_is_outlier()` - Where good observations are rejected (lines 230-280)

- **Agent learning:** `agents/simple_world_model.py`
  - How heating_rate_mean is learned from observations
  - Whether it models power levels separately

- **Consolidated beliefs:** `memory/domains/hot_pot/consolidated/beliefs.json`
  - Current wrong beliefs (heating_rate: 1.079 instead of 2.5)

- **Episode beliefs:** `memory/domains/hot_pot/episodes/*.json`
  - Individual episode observations before consolidation

---

## Conclusion

The HotPot medium question collapse is **not a bug** in the traditional sense, but a **systemic failure** in the memory consolidation system:

1. ✗ Agent learns wrong heating rate due to mixed observations
2. ✗ Wrong belief becomes consolidated baseline
3. ✗ Better estimates are rejected as outliers
4. ✗ Wrong belief grows in confidence
5. ✗ Agent fails temperature prediction questions
6. ✗ Medium question accuracy collapses to 11.1%

**This is CRITICAL** because it demonstrates that the memory system can create self-reinforcing incorrect beliefs that:
- Get stronger over time (increasing confidence)
- Reject corrections (outlier detection)
- Cause systematic task failures (medium questions)

The memory bug fix you requested was essential, but this is a deeper architectural issue with how beliefs are learned and consolidated across episodes.
