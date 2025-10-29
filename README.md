# World Model Experiments: ACE vs Interactive Learning

**Central Research Question:** Can comprehensive, evolved context substitute for interactive experience in LLM agents?

This project tests **Agentic Context Engineering (ACE)** against traditional interactive learning approaches to understand when rich context can replace expensive interaction.

---

## Quick Start

```bash
# 1. Set API key
export ANTHROPIC_API_KEY="your-key-here"

# 2. Test ACE agent (5 min, $0.20)
python test_ace_agent.py

# 3. Run pilot experiment (20 min, $20)
python scripts/run_experiment_parallel.py \
  --config config_ace_pilot.yaml \
  --output-dir results/ace_pilot \
  --workers 6

# 4. Analyze results
python analyze_ace_pilot.py results/ace_pilot
```

**See ACE_RUN_INSTRUCTIONS.md for detailed commands and troubleshooting.**

---

## Overview

### The Pivot: From Token Bridge to ACE

This project originally explored whether linguistic next-token prediction encodes the same learning signals as grounded world-model prediction (Token Prediction Bridge).

**Initial results** showed that interactive agents (Actor, Model-Based) outperformed passive reasoning (Observer) by ~8.5%, but at 3.5× token cost:
- Observer: 66.8% accuracy @ 6.5K tokens/episode
- Actor: 75.3% accuracy @ 22K tokens/episode
- Model-Based: 73.2% accuracy @ 22K tokens/episode

**Key insight:** Is this 3.5× cost necessary, or can we achieve similar performance through better context engineering?

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

## Expected Results

### Success Criteria

| Result | Accuracy | Tokens/Ep | Tokens/% | Interpretation |
|--------|----------|-----------|----------|----------------|
| **Excellent** | 70-75% | <8K | <111 | **Strong evidence for H-ACE** |
| **Good** | 65-70% | 8-10K | 111-154 | Moderate evidence, investigate |
| **Mixed** | 60-65% | 10-12K | 154-200 | Weak evidence, analyze failures |
| **Weak** | <60% | >12K | >200 | Debug implementation |

### Comparison to Baselines

**Current baselines (pilot_h1h5_fixed):**
- Observer: 66.8% @ 6.5K tokens = 97 tokens/%
- Actor: 75.3% @ 22K tokens = 292 tokens/%
- Model-Based: 73.2% @ 22K tokens = 305 tokens/%

**Target for ACE:**
- ACE: 70-75% @ 8-10K tokens = 111-143 tokens/%
- **2-3× more efficient than Actor**
- **Match or slightly below Actor accuracy**

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
├── README.md                           # This file
├── ACE_RUN_INSTRUCTIONS.md             # Detailed run commands ⭐
├── requirements.txt                    # Dependencies
├── config.yaml                         # Full experiment config
├── config_ace_pilot.yaml               # Pilot config (40 episodes)
├── config_ace_full.yaml                # Full config (600 episodes)
├── preregistration.yaml                # Locked hypotheses
│
├── test_ace_agent.py                   # Single episode test ⭐
├── analyze_ace_pilot.py                # Pilot analysis ⭐
│
├── agents/
│   ├── base.py                         # Base Agent class
│   ├── observer.py                     # Passive reasoning
│   ├── actor.py                        # Interactive + beliefs
│   ├── model_based.py                  # Actor + MLP model
│   └── ace.py                          # ACE agent ⭐ NEW (530 lines)
│
├── environments/
│   ├── hot_pot.py                      # Deceptive labels
│   ├── switch_light.py                 # Causal structure
│   └── chem_tile.py                    # Compositional reasoning
│
├── experiments/
│   ├── config.py                       # Config loading
│   ├── runner.py                       # Episode execution ⭐ (updated)
│   ├── prompts.py                      # All prompts ⭐ (updated)
│   ├── provenance.py                   # Versioning
│   └── rate_limiter.py                 # API rate limiting
│
├── evaluation/
│   ├── metrics.py                      # Metrics computation ⭐ (updated)
│   └── tasks.py                        # Test queries
│
├── models/
│   ├── llm.py                          # LLM interface
│   ├── belief_state.py                 # Belief state models
│   └── tools.py                        # Environment tools
│
├── scripts/
│   └── run_experiment_parallel.py      # Parallel runner ⭐ (updated)
│
└── results/                            # Episode logs
    ├── ace_pilot/                      # Pilot results
    │   ├── raw/*.json                  # Episode logs
    │   └── analysis_summary.json       # Aggregate stats
    └── ace_full/                       # Full results
        └── raw/*.json
```

**⭐ = Modified or new for ACE implementation**

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

## Contact

For questions, issues, or collaboration:
- Check ACE_RUN_INSTRUCTIONS.md for detailed commands
- Review episode logs in results/*/raw/
- Examine prompts in experiments/prompts.py
- See agents/ace.py for implementation details

---

## License

MIT License - See LICENSE file

---

**Status:** Implementation complete, ready for pilot experiment

**Last updated:** 2025-10-23 (ACE implementation)
