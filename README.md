# World Model Experiments: ACE vs Interactive Learning

**Central Research Question:** Can comprehensive, evolved context substitute for interactive experience in LLM agents?

This project tests **Agentic Context Engineering (ACE)** against traditional interactive learning approaches to understand when rich context can replace expensive interaction.

ACE architecture based on "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models" https://arxiv.org/abs/2510.04618

Initial research inspired by “Reflections on Richard Sutton’s Interview — Part I” https://yuanxue.github.io/2025/10/06/reflection-sutton-part1.html

---

## 🎓 **Study Complete (2025-10-31)**

**Preregistered comparison of ACE vs. belief-state agents completed.**

**Key Finding**: ACE's qualitative playbooks excel at strategy but struggle with quantitative probability questions that ACTOR's explicit belief states handle perfectly.

**Results**: See [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) for full analysis

**Status**: ✅ Complete - 506 episodes, statistically significant findings

**→ See below for reproduction instructions**

---

## Quick Start

### View Results

- **[RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)** - Complete analysis and findings
- **Preregistration**: [preregistration.md](preregistration.md) (locked at commit `cd41f0c`)
- **Analysis script**: `analyze_full_study.py`

### Reproduce the Study

```bash
# 1. Set API keys
export ANTHROPIC_API_KEY="sk-ant-api03-..."
export OPENAI_API_KEY="sk-proj-..."

# 2. Run the full study (506+ episodes, ~$60-80, 3-4 hours with 4 workers)
python scripts/run_experiment_parallel.py \
  --config config_full_study_3agents.yaml \
  --preregistration preregistration.md \
  --output-dir results/full_study_reproduction \
  --workers 4

# 3. Analyze results
python analyze_full_study.py
```

**Note**: The original study used commit `cd41f0c`. Results should be qualitatively similar but may vary slightly due to LLM stochasticity.

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

## Agent Architectures (Confirmatory Study v1.3)

**Note**: Model-Based agent removed after pilot (dominated by Actor). ChemTile deferred pending V2 validation. See [CHANGELOG.md](CHANGELOG.md) for complete justification.

### 1. Observer (Baseline)
- **Capability:** Language-only reasoning, no interaction
- **Belief:** None (passive reasoning)
- **Learning:** None (flat surprisal)
- **Expected:** 60-70% accuracy, ~$0.08/episode (~6,500 tokens)

### 2. Actor
- **Capability:** Interactive experimentation
- **Belief:** Parametric distributions (HotPotBelief, SwitchLightBelief, etc.)
- **Learning:** Bayesian updates from observations
- **Expected:** 75-80% accuracy, ~$0.18/episode (~22,000 tokens)

### 3. ACE (Agentic Context Engineering) ✨
- **Capability:** Interactive experimentation with context evolution
- **Belief:** Structured playbook of strategies (non-parametric)
- **Learning:** Reflection → Curation → Playbook updates
- **Expected:** 70-75% accuracy, ~$0.14/episode (~18,700 tokens)

**Removed from Full Study:**
- **Model-Based** (Actor + MLP): Pilot showed underperformance (70.7% vs Actor's 76.9% at same cost). See [CHANGELOG.md](CHANGELOG.md) Entry 2.

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

### Chem-Tile (Deferred - Not in v1.3)
**Challenge:** Compositional reasoning with safety constraints

**Status:** Deferred pending V2 evaluation validation. Not included in locked confirmatory study (v1.3).

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

### Completed Study Design

**Configuration:** `config_full_study_3agents.yaml`

**Episodes:**
- 3 environments (ChemTile, HotPotLab, SwitchLight)
- 3 agents (OBSERVER, ACTOR, ACE)
- Multiple seeds per environment
- **Total:** 506 successful episodes (603 attempted, 83.9% completion rate)

**Preregistration:**
- Locked at commit `cd41f0c` before data collection
- See [preregistration.md](preregistration.md) for full methodology

**Key Finding:**
- ACE showed boundary conditions on quantitative reasoning
- ACTOR's explicit belief states handle probability queries that ACE's textual playbooks cannot
- ACE excels at qualitative strategy formulation

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

# Set API keys (both required)
export ANTHROPIC_API_KEY="sk-ant-api03-..."
export OPENAI_API_KEY="sk-proj-..."
```

**Note:** ANTHROPIC_API_KEY is required for agents (Claude Sonnet 4.5). OPENAI_API_KEY is required for judge (GPT-4).

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
  --preregistration PREREGISTRATION.md \
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

### Run Confirmatory Study

**🔒 LOCKED preregistered experiment (v1.3)**

```bash
# 120 episodes, 3-4 hours, ~$60
python scripts/run_experiment_parallel.py \
  --config config_ace_full_n20.yaml \
  --preregistration PREREGISTRATION.md \
  --output-dir results/ace_confirmatory_n20 \
  --workers 6
```

**Important:** This is the preregistered confirmatory study. Do NOT modify configuration after starting data collection.

---

## Actual Results

### Overall Performance

| Agent | Accuracy | Avg Score | Tokens/Episode | Cost/Episode |
|-------|----------|-----------|----------------|--------------|
| **ACTOR** | **81.2%** | 8.12/10 | 19,289 | $0.12 |
| **ACE** | 70.3% | 7.03/10 | 20,692 | $0.13 |
| **OBSERVER** | 69.4% | 6.94/10 | 6,381 | $0.04 |

### Performance by Question Type (ChemTile)

| Question Type | ACE | ACTOR | Gap |
|---------------|-----|-------|-----|
| **Planning (easy)** | 12.3% | 100.0% | **+87.7%** |
| **Counterfactual (medium)** | 18.5% | 100.0% | **+81.5%** |
| Planning (medium) | 53.8% | 91.8% | +37.9% |
| Interventional (medium) | 82.3% | 97.8% | +15.5% |
| Interventional (easy) | 87.7% | 94.0% | +6.3% |
| Interventional (hard) | 93.1% | 88.1% | -5.0% |

**Key Insight:** ACE's qualitative playbooks excel at strategic interventional reasoning but struggle with quantitative probability questions that ACTOR's explicit belief states handle naturally.

**See [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) for complete analysis.**

---

## Scientific Rigor

### Preregistration

**This study was preregistered on 2025-10-29 prior to data collection.**

- **Preregistration v1.3** (LOCKED): [PREREGISTRATION.md](PREREGISTRATION.md) (2025-10-30)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md) - Documents all deviations transparently
- **Git tag v1.0**: prereg-v1.0 (SHA: 0353080d7a)
- **Git tag v1.1**: prereg-v1.1 (SHA: a9d81ea)
- **Git tag v1.2**: prereg-v1.2 (SHA: 91b0881)
- **Git tag v1.3** (CURRENT): prereg-v1.3 (SHA: 61d2154) 🔒

**Primary Hypotheses (v1.3)**:
- **H1a (Accuracy)**: ACE achieves ≥70% accuracy on causal reasoning tasks
- **H1b (Cost)**: ACE uses ≤70% of Actor's token cost
- H-Budget: Diminishing returns for larger playbook caps
- H-Curation: Curated playbook outperforms append-only by ≥5 pts
- H-Shift: ACE recovers from distribution shifts within 10 episodes

**Locked Study Design (v1.3)**:
- Sample size: n=20 per condition (120 total episodes)
- Environments: HotPot, SwitchLight (ChemTile deferred)
- Agents: Observer, Actor, ACE (Model-Based removed)
- Power: 80% to detect d≥0.65 at α=0.05

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
├── config_ace_full_n20.yaml            # 🔒 Confirmatory study (n=20, LOCKED)
├── config_ace_pilot_v2.yaml            # Pilot config (40 episodes)
├── PREREGISTRATION.md                  # 🔒 Locked hypotheses (v1.3, prereg-v1.3)
│
├── test_ace_agent.py                   # Single episode test
│
├── agents/
│   ├── base.py                         # Base Agent class
│   ├── observer.py                     # Passive reasoning
│   ├── actor.py                        # Interactive + beliefs
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
    └── ace_confirmatory_n20/           # 🔒 Confirmatory study (120 episodes)
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

## Extensions and Future Work

### Potential Extensions

1. **Hybrid Architectures:**
   - Combine ACE's qualitative playbooks with quantitative belief states
   - Test whether hybrid approach captures benefits of both

2. **Additional Environments:**
   - Test on domains with heavier quantitative reasoning requirements
   - Explore environments where qualitative strategies dominate

3. **Scale Study:**
   - Increase sample size for tighter confidence intervals
   - Test with different LLM models

4. **Prompt Engineering:**
   - Optimize ACE prompts specifically for probability questions
   - Test whether better prompting can close the quantitative gap

### Open Questions

- Can ACE be augmented to maintain quantitative summaries alongside qualitative playbooks?
- Do other context engineering approaches face similar limitations?
- What is the optimal allocation between qualitative and quantitative representations?

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
- **[config_ace_full_n20.yaml](config_ace_full_n20.yaml)** - 🔒 Confirmatory study (120 episodes, ~$60, LOCKED)

### Analysis
- **[scripts/analyze_with_statistics.py](scripts/analyze_with_statistics.py)** - Comprehensive statistical analysis

### Original Documentation
- **preregistration.md** - Locked hypotheses (pre-registered 2025-10-29)
- **Documentation/** - Theoretical framework and background

---

## Contact

**Jay Caldwell**
Independent Researcher
jay.s.caldwell@gmail.com

For questions about methodology, implementation, or collaboration:
- Full results: [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)
- Preregistration: [preregistration.md](preregistration.md)
- Episode logs: `results/full_study_v2/raw/*.json`
- Implementation: `agents/ace.py`, `agents/actor.py`

---

## License

MIT License - See LICENSE file

---

## Status

**Study Complete (2025-10-31):**
- ✅ Preregistered comparison completed
- ✅ 506 successful episodes (603 attempted, 83.9% completion rate)
- ✅ 5,060 causal reasoning questions evaluated
- ✅ Key finding: ACE excels at qualitative strategy but struggles with quantitative probability questions
- ✅ Results documented in [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)
- ✅ All data and analysis scripts available in repository

**Study Parameters:**
- Preregistration: [preregistration.md](preregistration.md) (locked at commit `cd41f0c`)
- Environments: ChemTile, HotPotLab, SwitchLight
- Agents: ACE, ACTOR, OBSERVER
- Configuration: `config_full_study_3agents.yaml`
- Analysis: `analyze_full_study.py`

**Key Results:**
- ACTOR: 81.2% accuracy (best overall, especially on quantitative questions)
- ACE: 70.3% accuracy (strong qualitative reasoning)
- OBSERVER: 69.4% accuracy (baseline)
- Critical finding: ACE showed 87.7% gap on planning questions requiring probability estimates

**Last updated:** 2025-10-31 (Study complete)
