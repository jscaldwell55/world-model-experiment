# CRITICAL ISSUES: Fidelity Scoring and World Model Bias

## Executive Summary

Investigation of the Offline Consolidation (OC) fidelity scoring revealed **three critical issues** that must be addressed before proceeding to Fine-Tuning Bridge integration:

1. **Fidelity is Circular** - High scores don't validate quality
2. **No Downstream Validation** - Haven't proven synthetics help
3. **Systematic Bias Risk** - Wrong model + high fidelity = amplified bias

**RECOMMENDATION: PAUSE OC → FTB integration** until validation experiments complete.

---

## Issue 1: Circular Fidelity Scoring

### The Problem

Synthetic observations are **generated** using the same model they're **scored** against:

**Generation (line 395-397):**
```python
predicted_temp = base_temp + heating_rate * time
noise = np.random.normal(0, measurement_noise)
measured_temp = predicted_temp + noise
```

**Fidelity Scoring (line 470-478):**
```python
predicted_temp = base_temp + heating_rate * time  # SAME FORMULA!
log_like = -0.5 * ((measured_temp - predicted_temp) / predictive_std)²
fidelity = exp(mean(log_likelihoods))
```

### Why This Matters

- Synthetics drawn from N(μ, σ²) will **always** have high likelihood under N(μ, σ²)
- Fidelity scores of 0.888, 0.918 are **expected**, not validating quality
- We're not measuring "how realistic is this?" but "does it match our belief?"

### Example

```
Learned (wrong) belief: heating_rate = 1.2°C/s
Ground truth: 2.5°C/s

Generate synthetic at time=12s:
  predicted_temp = 20 + 1.2*12 = 34.4°C
  measured_temp = 34.4 + noise(0, 2.0) = 36.1°C

Fidelity score:
  predicted_temp = 20 + 1.2*12 = 34.4°C (same!)
  log_like = -0.5 * ((36.1 - 34.4) / 2.0)² = -0.36
  fidelity = exp(-0.36) = 0.70 (high!)

Ground truth check:
  true_temp = 20 + 2.5*12 = 50.0°C
  error = 36.1 - 50.0 = -13.9°C (28% error!)
```

**High fidelity to wrong model = systematic bias**

---

## Issue 2: No Downstream Validation

### Current Status

❌ **NOT VALIDATED** that synthetics improve:
- Prediction accuracy on held-out tests
- Generalization to new scenarios
- Robustness to distribution shifts

### Critical Gap

We've implemented OC without testing the core hypothesis:
> "Synthetic data from world models improves downstream task performance"

This could be **false**. Synthetics might:
- Add no value (complexity without benefit)
- Hurt performance (amplify bias)
- Only help in specific conditions we haven't identified

### Required Validation Experiments

#### Experiment A: Prediction Accuracy
```
Baseline: Train on 5 real episodes
OC:       Train on 5 real + 1 synthetic
Metric:   Accuracy on 10 held-out test questions
Expected: If synthetics help, OC > Baseline
```

#### Experiment B: Data Efficiency
```
Baseline: Train on N real episodes
OC:       Train on N/2 real + N/2 synthetic
Metric:   Accuracy on test set
Expected: OC ≈ Baseline (same total, half collection cost)
```

#### Experiment C: Distribution Robustness
```
Baseline: Train on MIXED power (biased data)
OC:       Train on MIXED + synthetic HIGH power
Metric:   Accuracy on HIGH power test questions
Expected: OC > Baseline on undersampled contexts
```

#### Experiment D: Wrong Model Detection
```
Setup:    Inject systematic bias in beliefs
Baseline: Train on biased real data
OC:       Train on biased real + synthetics (to wrong model)
Metric:   Accuracy on test set
Expected: OC < Baseline (amplified bias)
```

**DO NOT proceed to FTB until these experiments run.**

---

## Issue 3: Systematic Bias Amplification

### The Danger

Current hot_pot playbook data:
- **4 LOW reliability episodes** (MIXED power, averaged beliefs)
- **1 HIGH reliability episode** (HIGH power, accurate beliefs)

If we generate synthetics from LOW reliability episodes:
1. Wrong beliefs (heating_rate = 1.2°C/s vs truth 2.5°C/s)
2. High fidelity to wrong model (0.7+)
3. Fine-tuning reinforces wrong belief
4. Agent becomes **more confident** in wrong answer
5. **Test accuracy decreases**

### Concrete Example

**Scenario:**
```
Real data: 4 MIXED power episodes
Learned: heating_rate = 1.2°C/s (averaged LOW/HIGH)
Truth (HIGH power): 2.5°C/s
Error: 52% underestimate
```

**Generate synthetics:**
```
Time    Synthetic   Ground Truth   Bias
3s      24.6°C      27.5°C        -10.6%
6s      26.9°C      35.0°C        -23.1%
9s      32.1°C      42.5°C        -24.5%
12s     37.4°C      50.0°C        -25.1%

Fidelity: 0.687 (HIGH despite systematic bias!)
```

**Training impact:**
```
Without synthetics:
  - 4 real × 0.3 weight = 1.2 effective episodes
  - Model uncertain, might default to prior

With synthetics:
  - 4 real × 0.3 + 1 synthetic × 0.8 = 2.0 effective episodes
  - ALL data reinforces heating_rate = 1.2°C/s
  - Model MORE CONFIDENT in WRONG belief
  - Test accuracy DECREASES
```

### Actual Data Check

From `memory/domains/hot_pot/playbook.json`:

| Episode | Power | Reliability | Learned Rate | Ground Truth | Error |
|---------|-------|-------------|--------------|--------------|-------|
| ep 042951 | MIXED | LOW | 1.13°C/s | Unknown | ⚠️ Averaged |
| ep 043409 | MIXED | LOW | 1.41°C/s | Unknown | ⚠️ Averaged |
| ep 043813 | MIXED | LOW | 1.05°C/s | Unknown | ⚠️ Averaged |
| ep 044224 | MIXED | LOW | 1.06°C/s | Unknown | ⚠️ Averaged |
| ep011 | **HIGH** | **HIGH** | **2.50°C/s** | **2.50°C/s** | **0.0%** ✓ |

**Good news:** The 1 HIGH reliability episode is **accurate** (0% error).

**Bad news:** If we had generated from LOW reliability episodes, we'd amplify bias.

---

## Proposed Safeguards

### 1. World Model Validation
```python
def validate_world_model(beliefs: dict, ground_truth: dict) -> bool:
    """Don't generate synthetics from inaccurate beliefs"""
    error_pct = abs(beliefs['heating_rate'] - ground_truth['heating_rate']) / ground_truth['heating_rate']
    return error_pct < 0.20  # < 20% error threshold
```

### 2. Cross-Validation
```python
def validate_synthetics(train_episodes, val_episodes):
    """Test synthetics on held-out real data"""
    synthetics = generate_counterfactuals(train_episodes)

    # Predict on validation set
    val_accuracy_baseline = evaluate(train_episodes, val_episodes)
    val_accuracy_oc = evaluate(train_episodes + synthetics, val_episodes)

    if val_accuracy_oc < val_accuracy_baseline:
        return None  # REJECT synthetics - they hurt
    else:
        return synthetics  # ACCEPT
```

### 3. Diversity Requirements
```python
def can_generate_synthetics(episodes: List[dict]) -> bool:
    """Only generate if we have HIGH reliability across contexts"""
    high_rel = [e for e in episodes if e['reliability'] == 'HIGH']

    if len(high_rel) < 2:
        return False  # Need multiple HIGH reliability episodes

    contexts = set(e['context']['power_setting'] for e in high_rel)
    if len(contexts) < 2:
        return False  # Need diversity across contexts

    return True
```

### 4. Fidelity Calibration
```python
def calculate_calibrated_fidelity(synthetic, real_episodes):
    """Fidelity = P(synthetic | real_world) not P(synthetic | belief)"""
    # Use held-out real data to calibrate
    # Compare synthetic to actual observations, not beliefs

    distances = []
    for real_ep in real_episodes:
        distance = trajectory_distance(synthetic, real_ep)
        distances.append(distance)

    # Fidelity = how close to nearest real episode
    min_distance = min(distances)
    calibrated_fidelity = exp(-min_distance)

    return calibrated_fidelity
```

### 5. Conservative Generation
```python
# Update thresholds
self.min_high_reliability_for_generation = 0.50  # 50% not 20%
self.max_synthetic_fraction = 0.10  # 10% not 30%
self.synthetic_weight = 0.5  # 0.5 not 0.8

# Only generate from HIGH reliability + accurate beliefs
def should_generate_from_episode(episode):
    return (
        episode['reliability'] == 'HIGH' and
        episode['world_model_error'] < 0.20  # < 20% error
    )
```

---

## Action Plan

### Immediate (Before FTB Integration)

1. **Run Validation Experiments A-D** ← CRITICAL
   - Measure actual impact on test accuracy
   - Detect if synthetics help or hurt
   - Estimated time: 2-3 days

2. **Add World Model Validation** ← HIGH PRIORITY
   - Compare learned beliefs to ground truth (when available)
   - Flag episodes with >20% error
   - Don't generate from inaccurate beliefs
   - Estimated time: 4 hours

3. **Implement Cross-Validation** ← HIGH PRIORITY
   - Hold out validation set
   - Test synthetics on real data
   - Reject if accuracy decreases
   - Estimated time: 6 hours

### Short-term (Next Week)

4. **Fix Fidelity Scoring**
   - Replace circular scoring with calibrated version
   - Use held-out real data as reference
   - Estimated time: 8 hours

5. **Add Conservative Limits**
   - Increase HIGH reliability threshold (50%)
   - Decrease synthetic fraction (10%)
   - Lower synthetic weights (0.5)
   - Estimated time: 2 hours

6. **Add Diversity Requirements**
   - Require multiple HIGH reliability episodes
   - Require coverage across contexts
   - Estimated time: 3 hours

### Long-term (Future Work)

7. **Adaptive Fidelity Thresholds**
   - Learn optimal thresholds from validation data
   - Domain-specific calibration

8. **Active Learning Integration**
   - Identify which contexts need more real data
   - Prioritize collection over generation

9. **Uncertainty-Aware Generation**
   - Generate synthetics that explore uncertainty
   - Not just high-likelihood trajectories

---

## Current Status

✅ **Fixed:** Race condition in ACE playbook (0% observation loss)

❌ **NOT READY:** OC → FTB integration due to:
1. Circular fidelity (always high, not validating)
2. No downstream validation (unknown if helps)
3. Bias amplification risk (wrong model = worse results)

✅ **Good Implementation:** Code structure is solid
✅ **Good Testing:** Test suite comprehensive
⚠️ **Missing Validation:** Core hypothesis untested

---

## Recommendation

**PAUSE** Fine-Tuning Bridge integration until:

1. ✅ Validation Experiment A completes (does it help?)
2. ✅ World model validation implemented (< 20% error)
3. ✅ Cross-validation safeguard added (reject if hurts)

**Timeline:** ~1 week to address critical gaps

**Alternative:** If urgent, proceed with **HIGH reliability only** generation:
- Current hot_pot: 1 HIGH reliability episode → 0-1 synthetics
- Switch_light: 4 HIGH reliability episodes → 1-2 synthetics
- Chem_tile: 0 HIGH reliability → 0 synthetics (SKIP)

This is **safe** but provides minimal value (very few synthetics).

---

## Conclusion

The OC system is **well-implemented** but **under-validated**.

**Core issue:** We built a system to generate synthetic data without first proving synthetic data helps.

This is **backwards**. We should:
1. First: Validate that perfect synthetics (hand-crafted) improve accuracy
2. Then: Build system to generate them automatically
3. Finally: Validate automated synthetics match hand-crafted quality

We skipped step 1.

**Next steps:** Run validation experiments before proceeding.
