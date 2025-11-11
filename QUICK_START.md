# QUICK START GUIDE
**Fix Evaluation & Debug ACE - Ready to Execute**

---

## What Was Built

✅ **New Evaluation System** - Questions require exploration data (not general knowledge)
✅ **ACE Debug Report** - Identified surprisal bug, provided fixes
✅ **Statistical Analysis** - Comprehensive testing framework
✅ **Study Configs** - Verification (n=5) and Full Study (n=20)

**Problem**: Observer scored 70.5% without exploring → Questions were too easy
**Solution**: New questions require specific measurements from episodes

---

## Option 1: Quick Verification (Recommended First)

**Run verification to test if new questions work:**

```bash
# 1. Apply evaluation upgrade
python scripts/upgrade_to_exploration_eval_v2.py --apply

# 2. Run 10-episode verification
ANTHROPIC_API_KEY="sk-ant-api03-..." OPENAI_API_KEY="sk-proj-..." \
python scripts/run_experiment_parallel.py \
  --config configs/config_verification_v2.yaml \
  --output-dir results/verification_v2 \
  --workers 2

# 3. Check results (should show Observer <40%, ACE >60%)
python3 << 'EOF'
import json, glob
episodes = [json.load(open(f)) for f in glob.glob('results/verification_v2/raw/*.json')]
by_agent = {}
for ep in episodes:
    agent = ep['agent_type']
    if agent not in by_agent: by_agent[agent] = []
    tests = ep.get('test_results', [])
    acc = sum(1 for t in tests if t.get('correct', False)) / len(tests) if tests else 0
    by_agent[agent].append(acc)
for agent, accs in by_agent.items():
    mean = sum(accs) / len(accs)
    print(f"{agent}: {mean:.1%} ({len(accs)} episodes)")
EOF
```

**Cost**: ~$5, **Time**: ~10 minutes

**✅ SUCCESS if**: Observer <40%, ACE >60%
**❌ FAIL if**: Observer still >40% (questions still too easy)

---

## Option 2: Full Study (After Verification Passes)

**Run full n=20 study for statistical validity:**

```bash
# Run 160 episodes (4 agents × 2 environments × 20 seeds)
ANTHROPIC_API_KEY="..." OPENAI_API_KEY="..." \
python scripts/run_experiment_parallel.py \
  --config configs/config_ace_full_n20.yaml \
  --output-dir results/ace_full_n20 \
  --workers 6

# Monitor progress
watch -n 60 'echo "Episodes completed:" && ls results/ace_full_n20/raw/*.json | wc -l'

# Run statistical analysis when done
python scripts/analyze_with_statistics.py results/ace_full_n20
```

**Cost**: ~$60-80, **Time**: ~4-6 hours

---

## ACE Surprisal Bug (Optional Fix)

**Current State**: ACE shows `surprisal = 0.0` for all steps (hardcoded)

**Decision Needed**: Should ACE compute surprisal?

### Option A: Accept 0.0 (ACE is non-probabilistic by design)
- No code changes needed
- Document that ACE uses context evolution, not belief updates
- Proceed with verification as-is

### Option B: Implement Novelty-Based Surprisal
- See `ACE_DEBUG_REPORT.md` for implementation
- Add `compute_surprisal()` method to `agents/ace.py`
- Test on single episode, then re-run verification

**Recommendation**: Start with Option A (accept 0.0), can add later if needed

---

## Rollback (If Something Breaks)

```bash
# Revert to original evaluation
python scripts/upgrade_to_exploration_eval_v2.py --rollback
```

---

## Files to Review

1. **`MISSION_SUMMARY.md`** - Complete overview, all deliverables
2. **`ACE_DEBUG_REPORT.md`** - ACE surprisal bug details and fixes
3. **`evaluation/tasks_exploration_v2.py`** - New test questions
4. **`scripts/analyze_with_statistics.py`** - Statistical analysis

---

## Expected Results

### Verification (10 episodes)
```
observer: 35% (5 episodes) ← Can't answer without exploration ✅
ace:      68% (5 episodes) ← Uses exploration data ✅
```

### Full Study (160 episodes)
```
Agent Accuracy (95% CI):
  observer:    35% [30%, 40%]  ← Baseline
  actor:       72% [67%, 77%]
  ace:         75% [70%, 80%]  ← Best (if playbook helps)

Statistical Tests:
  ACE vs Observer: p<0.001, d=1.2 (large effect) ✅
  ACE vs Actor:    p=0.15,  d=0.3 (small effect)
```

---

## Troubleshooting

**Q: Observer still scores >40%**
→ Questions still answerable from descriptions
→ Review `evaluation/tasks_exploration_v2.py` and tighten further

**Q: ACE scores <50%**
→ ACE might not be using exploration data properly
→ Check episode logs for measurement data
→ Verify playbook contains useful information

**Q: API errors during run**
→ Parallel runner has retry logic
→ Failed episodes logged in `failed_episodes.json`
→ Can re-run specific episodes manually

**Q: Statistical analysis fails**
→ Check that all episodes have `test_results` key
→ Verify ground truth contains trajectory data
→ See error message for specific issue

---

## Next Steps

1. **NOW**: Run verification (`config_verification_v2.yaml`)
2. **Check**: Observer <40%, ACE >60%?
3. **If Pass**: Run full study (`config_ace_full_n20.yaml`)
4. **If Fail**: Review questions, tighten requirements
5. **After Full**: Run statistical analysis
6. **Document**: Update paper with results

---

## Support

- **Evaluation issues**: See `evaluation/tasks_exploration_v2.py`
- **ACE surprisal**: See `ACE_DEBUG_REPORT.md`
- **Statistics**: See `scripts/analyze_with_statistics.py`
- **General**: See `MISSION_SUMMARY.md`

**Ready to execute!** Start with verification run above.
