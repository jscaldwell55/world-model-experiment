# Graduation POC: v1 vs v2 Comparison

## The Problem Diagnosed

The v1 POC test (`scripts/test_graduation_poc.py`) had a critical baseline contamination bug:

After LoRA training, it used `bridge.generate(..., use_adapter=False)` to get "baseline" responses. However, after PEFT wrapping and training, the model reference was contaminated - both "baseline" and "graduated" tests were effectively using the same trained model.

**Result**: 0% differentiation in v1 was a false signal. The test methodology was flawed.

## Methodology Comparison

| Aspect | v1 (Buggy) | v2 (Fixed) |
|--------|-----------|------------|
| Baseline capture timing | After training | **Before training** |
| Baseline method | `use_adapter=False` post-training | `capture_baseline()` pre-training |
| Contamination risk | HIGH - PEFT model wrapping | NONE - raw base model |
| Test questions | Exact match only | Exact match + Generalization |
| Cross-domain analysis | None | Full interference check |

## v2 Key Improvements

### 1. Pre-Training Baseline Capture

```python
# CORRECT FLOW in v2:
bridge = LoRABridge(...)

# Step 1: Load model
bridge._lazy_init()  # Called implicitly

# Step 2: Capture TRUE baseline BEFORE any adapter
true_baseline = bridge.capture_baseline(test_questions)

# Step 3: NOW create adapter and train
bridge.train(training_pairs, output_dir)

# Step 4: Compare against cached TRUE baseline
```

### 2. Generalization Questions

v2 adds rephrased and application-based questions to test actual understanding vs memorization:

**Exact Match** (training format):
- "What is the heating rate at HIGH power?"

**Generalization** (rephrased):
- "If I set the stove to maximum, how quickly will the water heat up?"

**Application**:
- "I have soup at 25C and need it at 80C. High power - time estimate?"

### 3. Cross-Domain Interference Detection

v2 tracks whether training on one domain affects others:
- **Trained domain** (hot_pot): Changes are EXPECTED
- **Non-trained domains** (chem_tile, switch_light): Changes indicate INTERFERENCE

## Files Modified

1. **`utils/lora_bridge.py`**
   - Added `capture_baseline()` method for pre-training baseline capture
   - Added `_cached_baselines` dict to store responses
   - Added `get_cached_baseline()` and `has_baseline_for()` helpers
   - Updated `generate()` to properly use `disable_adapter()` context

2. **`scripts/test_graduation_poc_v2.py`** (new)
   - Implements correct pre-training baseline methodology
   - Includes generalization test questions
   - Reports metrics separately for exact-match vs generalization
   - Assesses cross-domain interference

## Usage

```bash
# Quick test (1 epoch, 20 pairs)
USE_TF=0 python scripts/test_graduation_poc_v2.py --quick

# Full test
USE_TF=0 python scripts/test_graduation_poc_v2.py \
    --training-data data/training_pairs.json \
    --epochs 3 \
    --save-results results/graduation_poc_v2_results.json
```

## Expected Results

With correct methodology, we expect:
- **Trained domain (hot_pot)**: Some differentiation between baseline and graduated responses
- **Non-trained domains**: Minimal or no interference (changes would indicate unwanted model drift)
- **Generalization questions**: Demonstrates whether the model learned concepts or just memorized

## Conclusion

v1's 0% differentiation was a **methodology failure**, not a training failure. v2 fixes this by:
1. Capturing true baseline BEFORE any training contamination
2. Testing both exact-match and generalization capabilities
3. Monitoring cross-domain interference

The fix ensures we can now properly measure whether LoRA fine-tuning actually transfers world model knowledge.
