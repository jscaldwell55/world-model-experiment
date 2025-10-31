# Preregistration: ACE Cost-Aware Evaluation Study

**Preregistration Date**: 2025-10-29
**Study Start Date**: 2025-10-29 or later (MUST BE AFTER THIS DATE)
**Principal Investigator**: Jay Caldwell
**Affiliation**: Scale AI

## Study Overview

Evaluation of ACE (Agentic Context Engineering) framework against traditional interactive learning approaches, with focus on cost-efficiency and boundary conditions.

**Research Question**: Can comprehensive, evolved context (ACE playbook) substitute for expensive interactive experience at comparable accuracy?

## Primary Hypotheses

### H1a: ACE Accuracy Claim
ACE achieves clinically meaningful accuracy (≥70%) on causal reasoning tasks.

**Success Threshold**:
- ACE overall accuracy ≥ 70%

**Pilot Evidence**: 72.8% ± 6.7% (10 episodes, 95% CI: [66.1%, 79.5%])

**Statistical Test**: One-sample t-test, H₀: μ < 70%, α = 0.05

### H1b: ACE Cost Efficiency Claim
ACE achieves substantial cost savings (≤50% of Actor's USD cost) compared to interactive learning.

**Success Threshold**:
- ACE USD cost ≤ 0.5 × Actor USD cost

**Pilot Evidence**: ACE $0.14/ep vs Actor $0.18/ep = 78% (fails ≤50% threshold)

**Statistical Test**: Paired t-test across seeds, H₀: ACE cost ≥ 0.5 × Actor cost, α = 0.05

### Combined Interpretation (H1a × H1b)

| H1a (Accuracy) | H1b (Cost) | Outcome | Interpretation |
|----------------|------------|---------|----------------|
| ✅ Pass | ✅ Pass | **FULL SUCCESS** | ACE validated: accurate + efficient |
| ✅ Pass | ❌ Fail | **PARTIAL SUCCESS** | ACE accurate but not cost-efficient |
| ❌ Fail | ✅ Pass | **WEAK SUCCESS** | ACE efficient but inaccurate |
| ❌ Fail | ❌ Fail | **FAILURE** | ACE neither accurate nor efficient |

**Pilot Result**: Row 2 (Partial Success) - H1a supported, H1b not supported

**Pre-Commit Decision**:
- If H1a passes: Publish accuracy results, investigate cost optimization
- If H1b passes: Publish cost efficiency, investigate accuracy improvements
- If both pass: Publish full ACE validation
- If both fail: Publish boundary conditions and failure analysis

### H-Budget (Diminishing Returns)
Increasing playbook cap from 1k→2k tokens yields <50% of the gain from 512→1k.

**Success Threshold**:
- gain(1k→2k) < 0.5 × gain(512→1k)
- where gain = Δ accuracy in percentage points

**Statistical Test**: Linear regression, slope comparison

### H-Curation (Mechanism Check)
Curated ACE outperforms append-only (NoCurate) by ≥5 percentage points at same token cap.

**Success Threshold**:
- Curated_accuracy - NoCurate_accuracy ≥ 5 pts at 1k cap

**Statistical Test**: Paired t-test, Cohen's d ≥ 0.5

### H-Shift (Robustness)
Under distribution shift, ACE recovers to ≥95% of pre-shift accuracy within ≤10 episodes.

**Success Threshold**:
- post_shift_accuracy ≥ 0.95 × pre_shift_accuracy within 10 episodes

**Statistical Test**: Time-to-recovery analysis, survival curves

## Decision Rules (Pre-Commit to Interpretation)

### GREEN LIGHT (Publish as Validation)
- ACE sits on Pareto frontier in ≥2 of 3 environments
- Curated beats NoCurate by ≥5 pts at same token cap
- Total ops cost (tokens + API calls) ≤70% of Actor cost
- H-ACE-vs-Belief supported (within thresholds above)

**Interpretation**: ACE's advantages validated; context can substitute for interaction in these domains.

### AMBER LIGHT (Publish Hybrid Story)
- Pure ACE inconsistent across environments
- BUT shows ≥30% token savings in at least 1 environment
- OR Curation effect 3-5 pts (marginally significant)

**Interpretation**: ACE has value in specific domains; analyze boundary conditions.

### RED LIGHT (Publish Limits Paper)
- ACE advantages vanish across all environments (not on Pareto frontier)
- OR Curation effect <3 pts
- OR Ops costs negate token savings (total cost >70% of Actor)

**Interpretation**: ACE does not generalize; document failure modes and limitations.

## Environments

### 1. Hot-Pot Lab
**Challenge**: Deceptive labels require intervention to discover true dynamics

**Setup**:
- Pot on stove with temperature sensor
- Labels may say "Boiling!" when actually cold
- Must measure temperature to verify observations

**Test queries**:
- Interventional: "If I turn the stove on for 30s, what will the temperature be?"
- Counterfactual: "If I had turned it off earlier, would it still be hot?"

**Expected ACE advantage**: Can learn "always verify temperature before trusting labels"

**Distribution Shifts**:
- New label patterns (different deception strategies)
- Added sensor noise (+10-20%)
- Mid-run thermometer recalibration

### 2. Switch-Light
**Challenge**: Distinguish causation from correlation

**Setup**:
- 2 switches, 2 lights
- Unknown wiring (direct, crossed, OR-gate, etc.)
- Must intervene to determine structure

**Test queries**:
- Interventional: "If I flip switch 0, which lights change?"
- Structural: "What is the wiring configuration?"

**Expected ACE advantage**: Can learn "test each switch individually"

**Distribution Shifts**:
- New wiring families (XOR, AND gates)
- Observation noise (+15%)
- Mid-run wiring swap

### 3. Chem-Tile
**Challenge**: Compositional reasoning with safety constraints

**Setup**:
- Grid of chemical tiles
- Combining chemicals triggers reactions
- Some reactions are dangerous

**Test queries**:
- Compositional: "What happens if I combine A + B + C?"
- Safety: "Is this combination safe?"

**Expected ACE advantage**: Can learn reaction rules

**Distribution Shifts**:
- New chemical families
- Changed reaction rules
- Stochastic reaction outcomes (+20% variability)

## Agents & Ablations

### Core Agents (Main Study)

1. **Observer**: Passive baseline, no interaction/memory
   - Cost: ~$0.08/episode (~6,500 tokens)
   - Expected accuracy: 65-70%
   - Purpose: Measures text-only reasoning without environment interaction

2. **Actor**: Interactive + explicit belief updates
   - Cost: ~$0.18/episode (~22,000 tokens)
   - Expected accuracy: 75-80%
   - Purpose: Traditional interactive learning baseline
   - Features: Action selection, belief state tracking, memory updates

3. **ACE** (Agentic Context Engineering): Interactive + curated playbook
   - Cost: ~$0.14/episode (~18,700 tokens, pilot estimate)
   - Expected accuracy: 70-75%
   - Purpose: Test if curated context can substitute for expensive interaction
   - Features: Playbook curation, strategic exploration, knowledge synthesis

**Removed from Main Study:**
- **Model-Based**: Originally planned (Actor + MLP transition model)
- **Reason**: Pilot showed Model-Based underperforms Actor (70.7% vs 76.9%) at same cost ($0.174 vs $0.175)
- **Decision**: Dominated by Actor on accuracy; resources better spent on ACE ablations
- **Logged**: CHANGELOG.md entry 2025-10-30

### Ablation Controls (Need to implement)
4. **ACE-512**: Curated playbook, 512 token cap
5. **ACE-2k**: Curated playbook, 2k token cap
6. **ACE-NoCurate**: Append-only memory, 1k cap (tests curation value)
7. **ACE-RandomSubset**: Random bullet selection, 1k cap (tests selection vs curation)

## Experimental Design

### Pilot Study
- **Episodes**: 40 (2 envs × 4 core agents × 5 seeds)
- **Purpose**: Infrastructure validation, initial Pareto estimation
- **Environments**: HotPot, SwitchLight
- **Seeds**: [42, 43, 44, 45, 46] for HotPot, [100, 101, 102, 103, 104] for SwitchLight
- **Outputs**: Pareto plot, accuracy comparison, token analysis

### Full Study
- **Episodes**: 603 (3 envs × 3 core agents × 67 seeds)
- **Purpose**: Confirmatory hypothesis testing
- **Agents**: Observer, Actor, ACE (Model-Based removed after pilot)
- **Environments**: HotPot, SwitchLight, ChemTile
- **Seeds**:
  - HotPot: [42-108] (67 seeds)
  - SwitchLight: [100-166] (67 seeds)
  - ChemTile: [200-266] (67 seeds)
- **Estimated Cost**:
  - Observer: 603 × $0.08 = $48
  - Actor: 603 × $0.18 = $109
  - ACE: 603 × $0.14 = $84
  - **Total**: ~$241 (vs $300 with Model-Based)

### Ablation Study (After pilot if promising)
- **Episodes**: 120 (3 envs × 4 ablations × 10 seeds)
- **Purpose**: Test budget sweep and curation mechanisms
- **Agents**: ACE-512, ACE-1k, ACE-2k, ACE-NoCurate, ACE-RandomSubset

### Shift Study (If time permits)
- **Episodes**: 60 (3 envs × 4 agents × 5 seeds, pre/post shift)
- **Purpose**: Test robustness to distribution shifts

## Models & Configuration

### Agent Models
- **Model**: Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)
- **Temperature**: 1.0 (Generator), 0.0 (action selection)
- **Version**: Pinned (will not change mid-study)

### Judge Models
- **Programmatic Judge**: First priority (exact match, numeric tolerance)
- **LLM Judge**: GPT-4 (vendor-disjoint from agents)
  - Model: gpt-4-0125-preview (pinned)
  - Temperature: 0.0
  - Used only when programmatic judge insufficient

### Episode Parameters
- **Max steps per episode**: 10 (action budget)
- **Action budget**: 10 interactions per episode
- **Success threshold**: 70% accuracy

## Primary Metrics (Will Report All)

### Accuracy Metrics
1. **Overall accuracy**: % episodes with correct final answer
2. **Interventional accuracy**: % correct on interventional queries
3. **Counterfactual accuracy**: % correct on counterfactual queries
4. **Confidence interval**: Bootstrap 95% CI (1000 samples)

### Cost Metrics
1. **Tokens per episode**: Mean ± SD
2. **Total token budget**: Sum across all episodes
3. **Tokens per % accuracy**: Total tokens / accuracy (efficiency metric)
4. **Cost position**: Is agent on Pareto frontier?

### ACE-Specific Metrics
1. **Playbook growth**: Bullets added per episode
2. **Playbook size**: Total tokens in playbook
3. **Playbook utilization**: % bullets referenced in decisions
4. **Convergence**: When playbook growth stabilizes

### Pareto Analysis
1. **Pareto frontier**: Accuracy vs tokens curve
2. **Pareto position**: Is ACE on/near frontier?
3. **Dominated region**: Which configs are strictly dominated?

## Secondary Metrics (Exploratory)

1. **Calibration**: Brier score, ECE (Expected Calibration Error)
2. **Sample efficiency**: Episodes to 70% accuracy
3. **Helpful vs harmful bullets**: Classification from post-hoc analysis
4. **Curator agreement**: Inter-curator consistency
5. **Shift recovery time**: Episodes to recover after distribution shift

## Statistical Analysis Plan

### Primary Comparisons
- **ACE vs Actor**: Paired t-test across seeds, report Cohen's d
- **Curation ablations**: One-way ANOVA (Curated vs NoCurate vs RandomSubset)
- **Budget sweep**: Linear regression (accuracy ~ log(token_cap))

### Multiple Comparisons Correction
- Bonferroni correction for family-wise error rate
- Only applied to confirmatory hypotheses, not exploratory analyses

### Confidence Intervals
- Bootstrap 95% CI for all accuracy metrics (1000 resamples)
- Report CI width to assess precision

### Significance Threshold
- α = 0.05 for hypothesis tests
- Will report exact p-values, not just significant/not significant

## What We Will NOT Change Mid-Study

**Locked Parameters** (Cannot modify after experiments begin):
1. Hypotheses and success thresholds
2. Decision rules (Green/Amber/Red)
3. Environments and seeds for main study
4. Model versions (Sonnet 4.5, GPT-4)
5. Primary metrics (accuracy, tokens, Pareto position)
6. Episode budgets (40 pilot, 600 full)

**Allowed Modifications** (If discovered during study):
1. Bug fixes in implementation (logged with SHA)
2. Adding exploratory post-hoc analyses (clearly labeled as non-preregistered)
3. Clarifying ambiguous grading rules (documented in CHANGELOG.md)

All changes will be logged in `CHANGELOG.md` with timestamps and justification.

## Data Exclusion Criteria

**Episodes will be excluded from analysis if:**
1. API timeout/error (logged for cost accounting, excluded from accuracy)
2. Agent crashes mid-episode (implementation bug)
3. Programmatic judge cannot score (ambiguous output format)

**Excluded episodes must be <5% of total.** If >5%, report as study limitation.

**We will NOT exclude episodes based on:**
- Agent getting wrong answer (that's the measurement)
- Low confidence scores
- Unexpected strategies

## Artifacts & Reproducibility

### Required Outputs
1. `preregistration.md` (this file, committed before experiments)
2. `reproduce.sh` (one-command pilot run, ≤30 minutes)
3. `results/ace_pilot/aggregate_metrics.csv`
4. `results/ace_pilot/pareto_plot.png`
5. `results/ace_pilot/summary.json`
6. `CHANGELOG.md` (any deviations from preregistration)

### Provenance Logging (Per Episode)
- Git SHA at experiment start
- Config file hash
- Model IDs and versions
- Timestamp (ISO 8601)
- Random seed
- Environment variant
- Token counts (input + output)
- Action sequence
- Correctness (programmatic + judge if applicable)
- Confidence scores
- Playbook state (for ACE agents)

### Public Release
- All code, configs, results released on GitHub
- Tag release (e.g., v1.0-pilot) with DOI (Zenodo)
- Data under MIT license

## Limitations & Boundaries

**This study will NOT:**
1. Test on >3 core environments (scope constraint)
2. Run with >50 seeds per agent-env pair (compute constraint)
3. Test multi-agent scenarios (out of scope)
4. Test on proprietary/non-reproducible environments
5. Optimize hyperparameters (use published ACE values)

**Known Limitations:**
1. Environments are relatively simple (not real-world complexity)
2. Programmatic grading may miss nuanced reasoning
3. Single model family (Sonnet 4.5) for agents
4. Token costs measured, but not wall-clock time optimization

## Timeline

- **Day 0** (2025-10-29): Preregistration committed and tagged
- **Day 1**: Run 40-episode pilot
- **Day 2**: Analyze pilot + decide on full experiment
- **Day 3**: Run full experiment (if pilot successful)

**No results will be examined until experiments complete.**

## Preregistration Verification

This preregistration was:
- **Written on**: 2025-10-29
- **Committed**: 2025-10-29
- **Git SHA**: 0353080d7a675c6cebfec2fb2ad2ca20a3257113
- **Git Tag**: prereg-v1.0
- **Experiments begin**: 2025-10-29 or later

## Signature

By committing this preregistration, I commit to:
1. Running experiments as specified above
2. Reporting all preregistered metrics
3. Clearly labeling any exploratory analyses
4. Not cherry-picking results based on outcomes
5. Publishing regardless of whether hypotheses are supported

---

**Preregistration Status**: 🔒 LOCKED
**Next Step**: Commit this file, tag it, then run experiments
