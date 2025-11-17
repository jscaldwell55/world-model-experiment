# Duplicate Run Diagnostics & Prevention Report

**Date:** 2025-11-12
**Issue:** Two identical experiment runs within the same hour, costing ~$0.78 extra

---

## Root Cause Analysis

### Primary Issue: Command-Line Parsing Failure

Your command had an unintended line break that caused zsh to split it into two separate commands:

```bash
# What you intended (single command):
python scripts/run_experiment.py --config config_hybrid_test.yaml --num-episodes 5 --output-dir results/hybrid_5ep

# What actually happened (TWO commands):
python scripts/run_experiment.py --config config_hybrid_test.yaml  ✓ EXECUTED
  --num-episodes 5 --output-dir results/hybrid_5ep  ✗ FAILED (command not found)
```

### Evidence

1. **Output showed:** `Episodes per condition: None` (should have been `5`)
2. **Error at end:** `zsh: command not found: --num-episodes`
3. **Only 2 episodes ran** (from config default), not the requested 5

### Secondary Issue: Duplicate Execution

Two identical runs occurred within the same hour:
- **Run 1:** 2025-11-12 16:02:40 (4:02 PM)
- **Run 2:** 2025-11-12 16:29:42 (4:29 PM)

Both ran the same config:
- Environment: `hot_pot`
- Agent: `hybrid`
- Episodes: 2 (seeds 42, 43)
- Model: `claude-sonnet-4-5-20250929`

---

## Financial Impact

### Cost Breakdown

**Episode 1:**
- Input tokens: 32,960
- Output tokens: 18,815
- Cost: **$0.381**

**Episode 2:**
- Input tokens: 32,779
- Output tokens: 19,709
- Cost: **$0.394**

**Total per run:** ~$0.78
**Duplicate cost:** ~$0.78

### Cost Calculation Formula
```
Input cost  = (tokens / 1,000,000) × $3.00
Output cost = (tokens / 1,000,000) × $15.00
Total       = Input + Output
```

---

## Safeguards Implemented

### 1. Duplicate Detection (scripts/run_experiment.py:34-83)

- Scans for similar runs in the last 2 hours
- Compares configs (agents, environments, episode counts)
- Shows warning with cost estimate
- Requires manual confirmation to proceed

**Example output:**
```
======================================================================
⚠️  WARNING: Potential duplicate run detected!
======================================================================
  - 2025-11-12 16:02:40: 2 episodes, hybrid agents

This could cost ~$0.78 again.

Continue anyway? [y/N]:
```

### 2. Pre-Run Cost Estimation (scripts/run_experiment.py:85-98)

- Calculates estimated cost range before running
- Based on agent type, episode count, and model pricing
- Requires confirmation to proceed

**Example output:**
```
======================================================================
💰 COST ESTIMATE
======================================================================
Total episodes: 2
Estimated cost: $0.60 - $1.20
  (Actual cost may vary based on agent behavior)
======================================================================

Proceed with experiment? [y/N]:
```

### 3. Post-Run Cost Summary (scripts/run_experiment.py:181-212)

- Calculates actual cost from episode logs
- Shows total tokens used
- Displays precise USD cost

**Example output:**
```
======================================================================
COMPLETED
======================================================================
Total episodes: 2
Failed episodes: 0
Results saved to: results/raw/20251112_162942

💰 ACTUAL COST:
  Total: $0.7751
  Input tokens: 65,739
  Output tokens: 38,524
======================================================================
```

### 4. Argument Validation (scripts/run_experiment.py:214-219)

- Warns if fewer episodes ran than requested
- Alerts if `--num-episodes` wasn't parsed correctly

**Example output:**
```
⚠️  WARNING: Fewer episodes ran than requested!
  Requested: 5 per agent
  Actual: 2 total
  Check if --num-episodes was parsed correctly.
```

### 5. Skip Confirmations Flag

For automated runs or CI/CD:
```bash
python scripts/run_experiment.py --config config.yaml --skip-confirmations
```

---

## Best Practices Going Forward

### ✓ DO

1. **Write commands on a single line:**
   ```bash
   python scripts/run_experiment.py --config config_hybrid_test.yaml --num-episodes 5 --output-dir results/hybrid_5ep
   ```

2. **Use backslash for multi-line:**
   ```bash
   python scripts/run_experiment.py \
     --config config_hybrid_test.yaml \
     --num-episodes 5 \
     --output-dir results/hybrid_5ep
   ```

3. **Check the output:**
   - Verify "Episodes per condition" shows your expected number
   - Watch for the cost estimate confirmation
   - Note the duplicate detection warnings

4. **Review before confirming:**
   - Read the cost estimate carefully
   - Check if duplicate warning appears
   - Only proceed if intentional

### ✗ DON'T

1. **Don't split commands accidentally:**
   ```bash
   # BAD - line break without backslash
   python scripts/run_experiment.py --config config.yaml
     --num-episodes 5
   ```

2. **Don't ignore warnings:**
   - Duplicate detection warnings are there to save money
   - Always check why a duplicate is being detected

3. **Don't assume arguments were parsed:**
   - Always check "Episodes per condition" in the output
   - Use the new validation warnings

---

## Testing the Safeguards

### Test 1: Duplicate Detection

```bash
# Run once
python scripts/run_experiment.py --config config_hybrid_test.yaml

# Immediately run again (should trigger warning)
python scripts/run_experiment.py --config config_hybrid_test.yaml
```

Expected: Warning about duplicate run detected within 2 hours.

### Test 2: Cost Estimation

```bash
# Run any experiment
python scripts/run_experiment.py --config config_hybrid_test.yaml
```

Expected: Cost estimate prompt before execution.

### Test 3: Argument Validation

```bash
# Request 5 episodes but let config default to 2
python scripts/run_experiment.py --config config_hybrid_test.yaml --num-episodes 5

# (When prompted, type 'n' to abort and fix the issue)
```

Expected: Warning at the end if only 2 episodes ran instead of 5.

---

## Quick Reference

### Command Template
```bash
python scripts/run_experiment.py \
  --config <config_file.yaml> \
  --num-episodes <N> \
  --output-dir <results/output_dir>
```

### Skip Confirmations (for automation)
```bash
python scripts/run_experiment.py \
  --config <config_file.yaml> \
  --skip-confirmations
```

### Check Recent Runs
```bash
ls -lt results/raw | head -10
```

### Calculate Cost of Recent Run
```bash
jq '.cost.total_cost_usd' results/raw/<timestamp>/*.json | \
  awk '{sum+=$1} END {print "Total: $"sum}'
```

---

## Summary

**Problem:** Accidental duplicate runs due to command parsing issues and no safeguards.

**Solution:** Four-layer protection system:
1. Duplicate detection (2-hour window)
2. Pre-run cost estimation
3. Post-run cost summary
4. Argument validation warnings

**Impact:** Should prevent future duplicate runs and provide cost transparency.

**Next Steps:**
1. Test the new safeguards (see Testing section above)
2. Use proper command formatting (single line or backslash continuation)
3. Always review warnings and confirmations before proceeding
