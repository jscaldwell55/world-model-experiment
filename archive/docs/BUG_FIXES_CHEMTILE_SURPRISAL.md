# Bug Fixes: ChemTile Execution & Surprisal

## Summary

Fixed two critical bugs that prevented proper experiment execution:

1. **ChemTile Quote Parsing Bug** - Actions like `mix('A', 'B')` failed due to quote handling
2. **Observation Step Lag Bug** (fixed separately) - Caused invalid surprisal calculations

## Bug #1: ChemTile Quote Parsing Failure

### Problem

The LLM generates actions with quotes: `mix('A', 'B')` and `inspect('A')`, but the environment's parsing code didn't strip quotes, causing:
- `mix('A', 'B')` → parsed as compound names `"'A'"` and `"'B'"` (with quotes!)
- These didn't match `available_compounds = ['A', 'B', 'B']`
- Result: "Compound 'A' not available" error even though A was present

### Root Cause

**File:** `environments/chem_tile.py`

**Line 193 (before fix):**
```python
parts = [p.strip() for p in compounds_str.split(",")]
```

This only stripped whitespace, not quotes. So `mix('A', 'B')` → `["'A'", "'B'"]`

**Line 356 (before fix):**
```python
compound = action.replace("inspect", "").strip("()")
```

Same issue - didn't strip quotes from compound name.

### Solution

**Lines 193 & 356 (after fix):**
```python
# For mix command
parts = [p.strip().strip("'\"") for p in compounds_str.split(",")]

# For inspect command
compound = action.replace("inspect", "").strip("()").strip("'\"")
```

Now strips both single and double quotes after parsing.

### Verification

**Test Results:**

Before fix:
```python
env.step("mix('A', 'B')")
# → "Compound 'A' not available"
```

After fix:
```python
env.step("mix('A', 'B')")
# → {'reaction': 'A+B', 'outcome': 'nothing', 'available_compounds': ['B'], ...}
# ✅ Compounds consumed, reaction executed
```

## Bug #2: Invalid Surprisal Values

### Problem

Surprisal values were incorrect across all environments:
- **ChemTile:** Always 0.0 (no learning signal)
- **HotPot:** Spikes to 313.88+ (impossibly high)
- **SwitchLight:** Moderate values 0.6-0.8 (seemed OK but still wrong)

### Root Cause

**Step observation lag bug** (fixed in separate task):
- Observations at step N were from step N-1's action
- Belief computed surprisal on WRONG observation
- Example: Expected temp at t=70s but got observation from t=0s

**File:** `experiments/runner.py` (fixed separately)

The episode loop was:
1. Agent chooses action based on old observation
2. Log step with old observation ❌
3. Execute `env.step(action)` to get new observation (too late!)

This caused:
- **HotPot:** Computing P(temp=173°C | t=0s) instead of P(temp=173°C | t=70s)
  - Z-score = (173-20)/2 = 76.5 → surprisal = 2927.74!

- **ChemTile:** Computing P(init_message | belief) = 0.0 (no reaction info)
  - Surprisal always 0 because observations had no reaction/outcome

### Solution

Restructured episode loop to:
1. Agent chooses action
2. Execute `env.step(action)` → get RESULT observation
3. Compute surprisal on RESULT observation ✅
4. Update belief with RESULT
5. Log step with correct action-observation pairing

### Verification

**HotPot Surprisal Test:**

With correct time-temperature pairing:
```python
belief = HotPotBelief(heating_rate_mean=2.19, ...)
obs = {'measured_temp': 173.0, 'time': 70.0}
# Expected: 20 + 2.19*70 = 173.3°C
surprisal = 1.62  # ✅ Reasonable!
```

With step lag bug (wrong pairing):
```python
obs = {'measured_temp': 173.0, 'time': 0.0}  # ❌ Wrong time
# Expected: 20°C but observed 173°C
surprisal = 2927.74  # ❌ HUGE!
```

**ChemTile Surprisal Test:**

With correct reaction observation:
```python
belief = ChemTileBelief(reaction_probs={'A+B': {'C': 0.8, ...}})
obs = {'reaction': 'A+B', 'outcome': 'C', ...}
surprisal = 0.223  # ✅ = -log(0.8)
```

With init message (no reaction info):
```python
obs = {'message': 'Lab initialized', ...}  # No reaction/outcome
surprisal = 0.0  # ✅ Correct - no info to evaluate
```

## Additional Improvements

### Debug Logging

Added optional debug output to ChemTile environment:

```python
# In environments/chem_tile.py, step() method
if os.environ.get('DEBUG_CHEMTILE'):
    print(f"[ChemTile.step] Action: {repr(action)}")
    print(f"[ChemTile.step] Available BEFORE: {self.state.available_compounds}")
    # ... execute action ...
    print(f"[ChemTile.step] Available AFTER: {self.state.available_compounds}")
    print(f"[ChemTile.step] Observation: {obs}")
```

Usage:
```bash
DEBUG_CHEMTILE=1 python scripts/run_experiment.py ...
```

## Testing

### Unit Tests

Created `test_chemtile_debug.py`:
- Tests quote handling: `mix('A', 'B')` vs `mix(A, B)`
- Tests inspect: `inspect('A')` vs `inspect(A)`
- Validates ChemTileBelief log_likelihood computation
- Verifies surprisal calculations

Created `test_hotpot_surprisal.py`:
- Tests HotPotBelief with correct/incorrect time-temp pairings
- Demonstrates step lag bug effect on surprisal
- Shows Z-scores and expected values

### Integration Tests

```bash
# Test ChemTile environment directly
python -c "
from environments.chem_tile import ChemTile
env = ChemTile(seed=200)
env.reset(200)
obs, _, _, _ = env.step(\"mix('A', 'B')\")
print(obs)  # Should show reaction executed
"

# Run full experiment
python scripts/run_experiment.py --num-episodes 1 --output-dir results/test_fixes_final
```

## Results

### Before Fixes

**ChemTile Actor:**
- ❌ 0/10 actions executed successfully
- ❌ All mix() commands failed with "Compound not available"
- ❌ Surprisal always 0.0
- ❌ No reactions completed

**HotPot Actor:**
- ❌ Surprisal spikes to 313.88+
- ❌ Belief updates based on wrong observations
- ❌ No learning signal

### After Fixes

**ChemTile Actor:**
- ✅ Actions parse correctly with/without quotes
- ✅ Reactions execute and consume compounds
- ✅ Observations reflect state changes
- ✅ Surprisal computed on reaction outcomes
- ✅ Belief updates with actual results

**HotPot Actor:**
- ✅ Surprisal values reasonable (~1.6 for expected observations)
- ✅ High surprisal only for truly unexpected observations
- ✅ Belief updates with correct time-temperature pairings
- ✅ Learning signal present

## Files Modified

1. `environments/chem_tile.py`
   - Lines 193: Strip quotes in mix() parsing
   - Lines 356: Strip quotes in inspect() parsing
   - Lines 111-140: Added debug logging

2. `experiments/runner.py` (from previous fix)
   - Lines 127-173: Restructured episode loop
   - Now logs action with RESULT observation

3. `agents/actor.py` (from previous fix)
   - Separated action selection from belief updates
   - Added public methods for surprisal computation

## Acceptance Criteria

### ChemTile Execution ✅
- ✅ `mix('A', 'B')` successfully consumes A and B from available_compounds
- ✅ Reaction produces outcome based on probabilities
- ✅ `inspect('A')` returns compound info without errors
- ✅ Observations accurately reflect state changes
- ✅ Actor can complete reactions

### Surprisal ✅
- ✅ ChemTile surprisal is non-zero for reaction outcomes
- ✅ HotPot surprisal values are reasonable (<10 for normal observations)
- ✅ Surprisal correlates with unexpected observations
- ✅ Computed on correct (post-action) observations

## Next Steps

1. Run full 5-episode experiment to collect statistics
2. Verify Actor agents now learn from interactive experience
3. Compare Actor vs Observer performance with fixed execution
4. Analyze if interactive experience helps beyond structured belief representation
