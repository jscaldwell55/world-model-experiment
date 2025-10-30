# World Model Experiments: ACE vs Interactive Learning

**Central Research Question:** Can comprehensive, evolved context substitute for interactive experience in LLM agents?

This project tests **Agentic Context Engineering (ACE)** against traditional interactive learning approaches to understand when rich context can replace expensive interaction.

---

## 🚨 **IMPORTANT: Evaluation System Upgraded (2025-10-30)**

**Previous pilot revealed critical flaw**: Observer (passive agent) scored 70% despite doing ZERO exploration!

**Root cause**: Test questions were answerable from general knowledge, not exploration data.

**Solution**: Complete evaluation overhaul with exploration-dependent questions.

**Status**: ✅ Fixed, ⚠️ Verification pending

**→ See [QUICK_START.md](QUICK_START.md) for execution instructions**

---

## Quick Start

### Recommended: Run Verification First

```bash
# 1. Set API keys
export ANTHROPIC_API_KEY="sk-ant-api03-..."
export OPENAI_API_KEY="sk-proj-..."

# 2. Apply evaluation upgrade (one-time)
python scripts/upgrade_to_exploration_eval_v2.py --apply

# 3. Run verification (10 episodes, ~$5, ~10 min)
python scripts/run_experiment_parallel.py \
  --config config_verification_v2.yaml \
  --output-dir results/verification_v2 \
  --workers 2

# 4. Check if Observer <40% (proves new questions work)
# See QUICK_START.md for analysis code
```

**Expected**: Observer drops from 70% → 35% (can't answer without exploration!)

### Full Study (After Verification Passes)

```bash
# Run full n=20 study (160 episodes, ~$70, ~5 hours)
python scripts/run_experiment_parallel.py \
  --config config_ace_full_n20.yaml \
  --output-dir results/ace_full_n20 \
  --workers 6

# Comprehensive statistical analysis
python scripts/analyze_with_statistics.py results/ace_full_n20
```

**See [QUICK_START.md](QUICK_START.md) for complete instructions with copy-paste commands.**

---

## Overview

### The Pivot: From Token Bridge to ACE

This project originally explored whether linguistic next-token prediction encodes the same learning signals as grounded world-model prediction (Token Prediction Bridge).

**Pilot Results (ACE V2, 40 episodes)** revealed a critical evaluation flaw:
- Observer: 70.5% accuracy @ 6.7K tokens (0 exploration actions!)
- ACE: 72% accuracy @ 19K tokens (10 exploration actions)
- Actor: 75.5% accuracy @ 22K tokens (10 exploration actions)
- Model-Based: 74% accuracy @ 22K tokens (10 exploration actions)

**Critical Discovery:** Observer matched active agents despite ZERO exploration!

**Root Cause:** Test questions were answerable from general knowledge:
- ❌ "Will touching boiling water burn you?" → Common sense: Yes
- ❌ "Should you verify the label?" → Scientific method: Always verify

**Solution (2025-10-30):** Complete evaluation overhaul:
- ✅ New exploration-dependent questions requiring specific measurements
- ✅ "What exact temperature at t=20s?" → Must have measured!
- ✅ "What is the heating rate in °C/s?" → Must have calculated!
- ✅ Expected: Observer drops to <40% (can't fake exploration)

**Key insight:** With valid evaluation, can ACE match Actor performance at lower token cost?

### The ACE Hypothesis

**H-ACE:** Agentic Context Engineering can achieve Actor-level performance (~75%) at Observer-level cost (~6-8K tokens/episode), providing 3× efficiency gain.

**How it works:**
- Instead of parametric belief updates (Actor), ACE maintains an evolving **"playbook"** of strategies
- After each episode, **Reflector** extracts insights from the trajectory
- **Curator** organizes insights into compact, actionable playbook items
- **Generator** uses the playbook to guide future actions
- Playbook persists across episodes, accumulating knowledge over time

**Based on:** "Agentic Context Engineering" (2024) - achieved +17% improvement on AppWorld task

---

## 🔧 Evaluation System V2 (Oct 2025)

### The Problem

Original test questions tested **general knowledge**, not **exploration learning**:

| Question Type | Example | Why It's Bad |
|--------------|---------|--------------|
| General Safety | "Will boiling water burn you?" | Common knowledge |
| Scientific Method | "Should you verify labels?" | Always verify |
| Physics Reasoning | "Does high heat for 50s make pot hot?" | Basic inference |

**Result**: Observer scored 70% by reasoning alone, making exploration worthless!

### The Solution

New **exploration-dependent questions** require actual measurements:

| Question Type | Example | Why It's Better |
|--------------|---------|-----------------|
| Specific Measurement | "What temperature at t=20s?" | Must have measured at that time |
| Calculated Dynamics | "What is heating rate in °C/s?" | Must calculate from multiple measurements |
| Temporal Tracking | "When did temp exceed 80°C?" | Must track progression over time |
| Action-Specific | "What temp after toggling stove?" | Must have performed that action |

### Implementation

**Files Created:**
- `evaluation/tasks_exploration_v2.py` - New test questions (20 questions, all exploration-dependent)
- `evaluation/trajectory_extraction.py` - Extract measurement data from episode logs
- `scripts/upgrade_to_exploration_eval_v2.py` - Automated upgrade with backup/rollback

**Usage:**
```bash
# One-time upgrade
python scripts/upgrade_to_exploration_eval_v2.py --apply

# Rollback if needed
python scripts/upgrade_to_exploration_eval_v2.py --rollback
```

### Expected Impact

| Agent | Old Accuracy | New Accuracy (Expected) | Reason |
|-------|--------------|------------------------|--------|
| Observer | 70% | **35%** ↓ | Can't answer without exploration data |
| ACE | 72% | **68%** → | Uses exploration, slight drop from question difficulty |
| Actor | 75% | **72%** → | Same, slight difficulty increase |
| Model-Based | 74% | **70%** → | Same, slight difficulty increase |

**Success Criteria**: Observer <40%, ACE >60%, Gap >20 percentage points

---

## 🐛 ACE Agent Debugging (Oct 2025)

### Issues Found

**1. Surprisal Always 0.0**
- **Location**: `agents/ace.py:135` - hardcoded `surprisal=0.0`
- **Cause**: No `compute_surprisal()` method implemented
- **Impact**: ACE can't do proper active inference

**2. No Belief Updating**
- ACE only tracks playbook size, not probabilistic beliefs
- Missing `update_belief_from_observation()` method
- By design: ACE uses context evolution, not parametric updates

**3. Playbook Usage** ✅
- **Verified working**: Playbook IS consulted for action selection
- **Location**: `agents/ace.py:237` - included in Generator prompt
- **Not a bug**: This is working correctly

### Fix Options

**See `ACE_DEBUG_REPORT.md` for:**
- Detailed root cause analysis
- 3 implementation options (novelty-based, LLM-based, accept-as-is)
- Code examples and testing procedures

**Recommendation**: Start with surprisal=0 (non-probabilistic by design), can enhance later if needed

---

## 📊 Statistical Analysis Framework

**New comprehensive analysis** with scientific rigor:

**File**: `scripts/analyze_with_statistics.py`

**Includes:**
- ✅ Paired t-tests between all agent pairs
- ✅ Bootstrap 95% confidence intervals (10,000 resamples)
- ✅ Cohen's d effect sizes (negligible/small/medium/large)
- ✅ Bonferroni correction for multiple comparisons
- ✅ Power analysis (80% power for d≥0.65 with n=20)
- ✅ Summary tables and CSV exports

**Usage:**
```bash
python scripts/analyze_with_statistics.py results/ace_full_n20
```

**Outputs:**
- `statistical_ttests.csv` - All pairwise comparisons
- `statistical_confidence_intervals.csv` - Bootstrap CIs
- `statistical_raw_data.csv` - Full dataset
- Console: Complete statistical report

---

## Agent Architectures

### 1. Observer (Baseline)
- **Capability:** Language-only reasoning, no interaction
- **Belief:** None (passive reasoning)
- **Learning:** None (flat surprisal)
- **Expected:** 60-70% accuracy, ~6K tokens/episode

### 2. Actor
- **Capability:** Interactive experimentation
- **Belief:** Parametric distributions (HotPotBelief, SwitchLightBelief, etc.)
- **Learning:** Bayesian updates from observations
- **Expected:** 75% accuracy, ~22K tokens/episode

### 3. Model-Based
- **Capability:** Actor + explicit transition model (MLP)
- **Belief:** Parametric + learned dynamics
- **Learning:** Bayesian updates + model predictions
- **Expected:** 73% accuracy, ~22K tokens/episode

### 4. ACE (Agentic Context Engineering) ✨ **NEW**
- **Capability:** Interactive experimentation with context evolution
- **Belief:** Structured playbook of strategies (non-parametric)
- **Learning:** Reflection → Curation → Playbook updates
- **Expected:** 70-75% accuracy, ~8-10K tokens/episode

---

## Environments

### Hot-Pot Lab
**Challenge:** Deceptive labels require intervention to discover true dynamics

**Setup:**
- Pot on stove with temperature sensor
- Label may say "Boiling!" when actually cold
- Must measure temperature to trust observations

**Test queries:**
- Interventional: "If I turn the stove on for 30s, what will the temperature be?"
- Counterfactual: "If I had turned it off earlier, would it still be hot?"

**Expected coupling:**
- Strong (labels actively mislead)
- Actor advantage: Can test hypotheses
- ACE advantage: Can learn "always verify temperature before trusting labels"

### Switch-Light
**Challenge:** Distinguish causation from correlation

**Setup:**
- 2 switches, 2 lights
- Unknown wiring (direct, crossed, OR-gate, etc.)
- Must intervene to determine structure

**Test queries:**
- Interventional: "If I flip switch 0, which lights change?"
- Structural: "What is the wiring configuration?"

**Expected coupling:**
- Moderate (requires intervention)
- Actor advantage: Can isolate causal effects
- ACE advantage: Can learn "test each switch individually"

### Chem-Tile (Optional)
**Challenge:** Compositional reasoning with safety constraints

**Setup:**
- Grid of chemical tiles
- Combining chemicals triggers reactions
- Some reactions are dangerous

**Test queries:**
- Compositional: "What happens if I combine A + B + C?"
- Safety: "Is this combination safe?"

**Expected coupling:**
- Weak (compositional patterns may be inferable)
- Actor advantage: Less clear
- ACE advantage: Can learn reaction rules

---

## ACE Implementation

### Architecture

```
Episode → Reflector → Curator → Playbook
  ↓                                 ↓
Generator ←──────────────────────────
```

**1. Generator** (`agents/ace.py:_choose_action`)
- Uses current playbook as context
- Decides next action based on strategies
- Similar to Actor but with playbook instead of belief state

**2. Reflector** (`agents/ace.py:_reflect_on_episode`)
- Analyzes episode trajectory after completion
- Identifies what worked, what failed, new insights
- Outputs structured JSON with insights

**3. Curator** (`agents/ace.py:_curate_insights`)
- Converts insights into compact playbook items
- Avoids redundancy with existing playbook
- Categorizes into sections (strategies, pitfalls, etc.)

**4. Playbook** (persistent across episodes)
```python
{
  'strategies_and_hard_rules': [
    {'id': 'abc123', 'content': 'Always verify temp before trusting labels'},
    ...
  ],
  'useful_code_snippets': [...],
  'troubleshooting_and_pitfalls': [...],
  'apis_to_use': [...],
  'verification_checklist': [...]
}
```

### Prompts

All prompts are versioned in `experiments/prompts.py`:

- **ACE_GENERATOR_TEMPLATE**: Action selection with playbook context
- **ACE_REFLECTOR_TEMPLATE**: Episode analysis and insight extraction
- **ACE_CURATOR_TEMPLATE**: Insight organization into playbook items
- **ACE_QUERY_TEMPLATE**: Test query answering with playbook

Prompts adapted from ACE paper (Appendix, Figures 9-11).

### Key Files

```
agents/ace.py                    # ACE agent implementation (530 lines)
experiments/prompts.py           # ACE prompts (lines 238-354)
experiments/runner.py            # Episode execution (calls update_playbook)
evaluation/metrics.py            # ACE-specific metrics (lines 385-577)
config_ace_pilot.yaml            # Pilot configuration (40 episodes)
config_ace_full.yaml             # Full configuration (600 episodes)
test_ace_agent.py                # Single episode test
analyze_ace_pilot.py             # Pilot analysis script
```

---

## Experimental Design

### Pilot Experiment (40 episodes)

**Configuration:** `config_ace_pilot.yaml`

**Episodes:**
- 2 environments (HotPot, SwitchLight)
- 4 agents (Observer, Actor, Model-Based, ACE)
- 5 seeds per combination
- **Total:** 2 × 4 × 5 = 40 episodes

**Budget:**
- Time: ~15-20 minutes (6 workers)
- Cost: ~$15-25

**Purpose:**
- Verify ACE implementation works
- Quick comparison to baselines
- Decision point for full experiment

### Full Experiment (600 episodes)

**Configuration:** `config_ace_full.yaml`

**Episodes:**
- 3 environments (HotPot, SwitchLight, ChemTile)
- 4 agents (Observer, Actor, Model-Based, ACE)
- 50 seeds per combination
- **Total:** 3 × 4 × 50 = 600 episodes

**Budget:**
- Time: ~2-3 hours (10 workers)
- Cost: ~$150-200

**Purpose:**
- Statistical power for hypothesis testing
- Publication-ready results
- Detailed analysis of ACE vs baselines

---

## Metrics

### Standard Metrics (All Agents)

**Accuracy:**
- Overall accuracy
- Interventional accuracy (causal queries)
- Counterfactual accuracy (hypothetical queries)
- By difficulty (easy/medium/hard)

**Efficiency:**
- Tokens per episode
- Tokens per % accuracy
- Actions taken / action budget
- API calls per episode

**Calibration:**
- Brier score
- Expected Calibration Error (ECE)
- Confidence vs correctness correlation

### ACE-Specific Metrics

**Playbook Growth:**
- Total bullets over time
- Growth rate (bullets per episode)
- Convergence (when growth stabilizes)

**Playbook Utilization:**
- % of items marked helpful
- % of items marked harmful
- Section breakdown (which sections used most)

**Context Efficiency:**
- Accuracy per playbook bullet
- Comparison to Actor's belief state size

---

## Running Experiments

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY="your-key-here"
```

### Test Single Episode

```bash
# Quick test (5 min, $0.20)
python test_ace_agent.py
```

**Expected output:**
- Episode completes successfully
- Playbook has 3-5 items
- Log saved to `/tmp/test_ace_agent/raw/test_ace_ep001.json`

### Run Pilot

```bash
# 40 episodes, 15-20 min, $15-25
python scripts/run_experiment_parallel.py \
  --config config_ace_pilot.yaml \
  --preregistration preregistration.yaml \
  --output-dir results/ace_pilot \
  --workers 6
```

**Monitor progress:**
```bash
# Count completed episodes
ls results/ace_pilot/raw/ | wc -l

# Watch real-time updates
tail -f results/ace_pilot/raw/*.json
```

### Analyze Results

```bash
python analyze_ace_pilot.py results/ace_pilot
```

**Output:**
- Comparison table (accuracy, tokens, efficiency)
- Detailed breakdown per agent
- ACE playbook statistics
- Decision guidance (proceed to full experiment?)

### Run Full Experiment

**Only after pilot succeeds!**

```bash
# 600 episodes, 2-3 hours, $150-200
python scripts/run_experiment_parallel.py \
  --config config_ace_full.yaml \
  --preregistration preregistration.yaml \
  --output-dir results/ace_full \
  --workers 10
```

---

## Expected Results (With V2 Evaluation)

### Success Criteria (Updated for Exploration-Dependent Questions)

| Result | Accuracy | Tokens/Ep | Tokens/% | Interpretation |
|--------|----------|-----------|----------|----------------|
| **Excellent** | 68-72% | 18-20K | 264-294 | **ACE matches Actor, playbook works** |
| **Good** | 62-68% | 18-20K | 265-323 | ACE competitive, room for improvement |
| **Mixed** | 55-62% | 18-20K | 290-364 | ACE underperforms, debug needed |
| **Weak** | <55% | >20K | >364 | Major issues with ACE implementation |

**Note**: Absolute accuracy expected lower with harder V2 questions. Focus on **relative** performance.

### Comparison to Baselines

**Pilot V2 Results (with flawed evaluation):**
- Observer: 70.5% @ 6.7K tokens (❌ shouldn't be this high!)
- ACE: 72% @ 19K tokens
- Actor: 75.5% @ 22K tokens
- Model-Based: 74% @ 22K tokens

**Expected with Fixed Evaluation:**
- Observer: **35-40%** @ 6.7K tokens (↓ drops due to exploration requirement)
- ACE: **68-72%** @ 19K tokens (→ maintains, uses exploration)
- Actor: **70-75%** @ 22K tokens (→ maintains, uses exploration)
- Model-Based: **68-73%** @ 22K tokens (→ maintains, uses exploration)

**Critical Test**: ACE should outperform Observer by **>25 percentage points** (proves playbook + exploration works)

### Statistical Requirements (n=20)

With n=20 per condition:
- **Power**: 80% to detect d≥0.65 (medium-large effect)
- **Significance**: α=0.05 with Bonferroni correction
- **Effect sizes**:
  - ACE vs Observer: Expect d>1.0 (large)
  - ACE vs Actor: Expect d=0.3-0.5 (small-medium)

**Validation Criteria:**
- ✅ Observer <40% (proves questions require exploration)
- ✅ ACE >60% (proves ACE can use exploration data)
- ✅ ACE within 5 points of Actor (proves playbook as effective as belief state)
- ✅ Statistical significance (p<0.05 after correction)

---

## Scientific Rigor

### Preregistration

**This study was preregistered on 2025-10-29 prior to data collection.**

- **Preregistration**: [preregistration.md](preregistration.md)
- **Git commit**: 0353080d7a675c6cebfec2fb2ad2ca20a3257113
- **Git tag**: prereg-v1.0
- **Experiments begin**: 2025-10-29 or later

**Primary Hypotheses**:
- H-ACE-vs-Belief: ACE achieves Actor-level accuracy at ≤50% token cost
- H-Budget: Diminishing returns for larger playbook caps
- H-Curation: Curated playbook outperforms append-only by ≥5 pts
- H-Shift: ACE recovers from distribution shifts within 10 episodes

### Provenance

Every episode log contains:
- Git SHA (code version)
- Config hash (settings)
- Prompt version (prompts.py:PROMPT_VERSION)
- Timestamp and seed
- Full trajectory and playbook state

### Reproducibility

- Deterministic environments with explicit seeds
- All prompts versioned in code
- No hidden magic strings
- Full episode logs for replay

### Guard Rails

- **No ground truth leakage:** Observations never contain hidden state
- **Programmatic injection:** Observations injected, never echoed by LLM
- **Validated observations:** All pass through Pydantic schemas
- **Isolated playbook updates:** Only after test queries, not during episode

---

## Project Structure

```
world-model-experiment/
├── README.md                           # This file ⭐ UPDATED
├── QUICK_START.md                      # 🆕 Copy-paste execution guide
├── MISSION_SUMMARY.md                  # 🆕 Complete mission overview
├── ACE_DEBUG_REPORT.md                 # 🆕 ACE surprisal debugging
├── requirements.txt                    # Dependencies
│
├── config_verification_v2.yaml         # 🆕 Verification run (n=5)
├── config_ace_full_n20.yaml            # 🆕 Full study (n=20)
├── config_ace_pilot_v2.yaml            # Pilot config (40 episodes)
├── preregistration.yaml                # Locked hypotheses
│
├── test_ace_agent.py                   # Single episode test
│
├── agents/
│   ├── base.py                         # Base Agent class
│   ├── observer.py                     # Passive reasoning
│   ├── actor.py                        # Interactive + beliefs
│   ├── model_based.py                  # Actor + MLP model
│   └── ace.py                          # ACE agent ⭐ (530 lines, surprisal=0 bug)
│
├── environments/
│   ├── hot_pot.py                      # Deceptive labels
│   ├── switch_light.py                 # Causal structure
│   └── chem_tile.py                    # Compositional reasoning
│
├── experiments/
│   ├── config.py                       # Config loading
│   ├── runner.py                       # Episode execution (updated for V2)
│   ├── prompts.py                      # All prompts
│   ├── provenance.py                   # Versioning
│   └── rate_limiter.py                 # API rate limiting
│
├── evaluation/
│   ├── tasks.py                        # Original test queries (V1)
│   ├── tasks_exploration_v2.py         # 🆕 Exploration-dependent questions
│   ├── trajectory_extraction.py        # 🆕 Extract measurement trajectories
│   ├── judge.py                        # Vendor-disjoint judging
│   └── metrics.py                      # Metrics computation
│
├── models/
│   ├── llm.py                          # LLM interface
│   ├── belief_state.py                 # Belief state models
│   └── tools.py                        # Environment tools
│
├── scripts/
│   ├── run_experiment_parallel.py      # Parallel runner
│   ├── upgrade_to_exploration_eval_v2.py  # 🆕 Evaluation upgrade script
│   └── analyze_with_statistics.py      # 🆕 Comprehensive statistical analysis
│
└── results/                            # Episode logs
    ├── ace_pilot_v2/                   # Pilot V2 results (40 episodes)
    │   ├── raw/*.json                  # Episode logs
    │   └── failed_episodes.json        # Errors (if any)
    ├── verification_v2/                # Verification results (10 episodes)
    └── ace_full_n20/                   # Full study (160 episodes)
        ├── raw/*.json                  # Episode logs
        ├── statistical_ttests.csv      # 🆕 T-test results
        ├── statistical_confidence_intervals.csv  # 🆕 Bootstrap CIs
        └── statistical_raw_data.csv    # 🆕 Full dataset
```

**⭐ = Updated for evaluation fix and ACE debugging**
**🆕 = New files created for mission**

---

## Troubleshooting

### Common Issues

**1. Empty playbooks**
- Check Reflector output in episode logs
- Verify Curator is creating delta items
- May need to adjust prompts

**2. High token usage**
- Check playbook isn't growing too large
- Verify deduplication is working
- Consider shorter playbook format

**3. Rate limit errors**
- Script auto-retries with backoff
- Reduce workers if persistent
- Uses 90% of limits as buffer

**4. Episode failures**
- Check `failed_episodes.json` for details
- Common: JSON parsing, action errors, timeouts
- Resume with `--resume-from results/dir`

### Debug Steps

1. Run single episode test first
2. Check episode log structure
3. Verify Reflector/Curator outputs
4. Analyze playbook content quality
5. Compare to Actor's successful strategies

**See ACE_RUN_INSTRUCTIONS.md for detailed troubleshooting.**

---

## References

### ACE Framework
- Paper: "Agentic Context Engineering" (2024)
- Key idea: Context as evolving playbook, not static prompt
- Innovation: Structured, itemized updates prevent "context collapse"
- Results: +17.1% on AppWorld task

### Theoretical Background
- Sutskever: "Rich context ≈ world model"
- Sutton: "Need interaction for true understanding"
- This experiment: Tests when each is right

### Prior Work
- Token Prediction Bridge (original direction)
- Pilot H1-H5 results (motivation for ACE)
- See Documentation/ for detailed theoretical framework

---

## Next Steps

### After Pilot (if successful)

1. **Run full experiment** (600 episodes)
2. **Statistical analysis:**
   - t-tests for accuracy differences
   - Effect sizes (Cohen's d)
   - Bootstrap confidence intervals
3. **Playbook analysis:**
   - Qualitative review of items
   - Compare to Actor's successful strategies
   - Identify what context captures vs misses
4. **Write-up:**
   - ACE vs baselines comparison
   - When context suffices vs when interaction needed
   - Cost-benefit analysis for practitioners

### Publication Targets

**Strong results (ACE ≥ 70%):**
- "Agentic Context Engineering Matches Interactive Learning at 1/3 Cost"
- Venue: NeurIPS, ICML, ICLR

**Moderate results (ACE 65-70%):**
- "Limits and Opportunities of Context Engineering for World Models"
- Venue: ACL, EMNLP, CoRL

**Weak results (ACE < 65%):**
- "When Context Cannot Replace Experience"
- Venue: Workshop (e.g., FMDM @ NeurIPS)

---

## 📚 Documentation

### Quick Reference
- **[QUICK_START.md](QUICK_START.md)** - Copy-paste commands to run experiments
- **[MISSION_SUMMARY.md](MISSION_SUMMARY.md)** - Complete overview of evaluation fix and ACE debugging
- **[ACE_DEBUG_REPORT.md](ACE_DEBUG_REPORT.md)** - Detailed ACE surprisal bug analysis and fixes

### Evaluation System
- **[evaluation/tasks_exploration_v2.py](evaluation/tasks_exploration_v2.py)** - New exploration-dependent questions
- **[evaluation/trajectory_extraction.py](evaluation/trajectory_extraction.py)** - Extract measurement data from episodes

### Configuration Files
- **[config_verification_v2.yaml](config_verification_v2.yaml)** - Verification run (10 episodes, ~$5)
- **[config_ace_full_n20.yaml](config_ace_full_n20.yaml)** - Full study (160 episodes, ~$70)

### Analysis
- **[scripts/analyze_with_statistics.py](scripts/analyze_with_statistics.py)** - Comprehensive statistical analysis

### Original Documentation
- **preregistration.md** - Locked hypotheses (pre-registered 2025-10-29)
- **Documentation/** - Theoretical framework and background

---

## Contact

For questions, issues, or collaboration:

**New Setup (V2 Evaluation):**
1. Start with [QUICK_START.md](QUICK_START.md) for execution
2. Review [MISSION_SUMMARY.md](MISSION_SUMMARY.md) for complete context
3. Check [ACE_DEBUG_REPORT.md](ACE_DEBUG_REPORT.md) for ACE-specific issues
4. See episode logs in `results/*/raw/*.json` for debugging

**Original Setup:**
- Review episode logs in results/*/raw/
- Examine prompts in experiments/prompts.py
- See agents/ace.py for implementation details

---

## License

MIT License - See LICENSE file

---

## Status

**Current State (2025-10-30):**
- ✅ ACE implementation complete
- ✅ Pilot V2 run complete (revealed evaluation flaw)
- ✅ Evaluation system overhauled (V2 questions)
- ✅ ACE debugging complete (surprisal bug documented)
- ✅ Statistical analysis framework ready
- ⏸️ Verification run pending (Observer <40% validation)
- ⏸️ Full study pending (n=20, after verification)

**Next Steps:**
1. Run verification (10 episodes) - See [QUICK_START.md](QUICK_START.md)
2. Validate Observer <40%, ACE >60%
3. If pass → Run full study (160 episodes)
4. Comprehensive statistical analysis
5. Results interpretation and write-up

**Last updated:** 2025-10-30 (Evaluation V2 + ACE debugging)
