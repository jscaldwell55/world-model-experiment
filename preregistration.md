# Preregistration: ACE (Agentic Context Engineering) Validation Study

**Date**: 2025-10-31
**Study Version**: 2.0 (Post-Pilot Confirmatory)
**Git SHA**: d2f2815a909a8971173c025f37358b17bdc1f797
**Git Tag**: prereg-v2.0
**Data Collection Status**: NOT STARTED

---

## Research Question

Can language model agents achieve competitive accuracy on causal reasoning tasks using **curated context** (playbook) instead of extensive interaction, at reduced computational cost?

**ACE Hypothesis**: Strategic curation of experience into reusable context can substitute for expensive online interaction while maintaining accuracy and reducing cost.

---

## Background & Motivation

### Pilot Study Findings (Informing This Study)

**Pilot Configuration**: 10 episodes per agent-environment pair across 3 environments (HotPot, SwitchLight, ChemTile)

**Key Results**:
1. **Actor baseline performance**: 76.9% accuracy at $0.175/episode
2. **ACE pilot performance**: 72.8% accuracy at $0.14/episode (78% of Actor cost)
3. **Model-Based underperformed**: Removed from main study (dominated by Actor)
4. **ACE implementation issues**: Original ACE implementation had bugs preventing faithful paper implementation

**Decisions Based on Pilot**:
- Remove Model-Based agent (not competitive)
- Fix ACE to match paper specification (multi-round reflection, feedback loops, retrieval)
- Increase sample size to n=67 per condition for robust effect detection
- Focus on 3-agent comparison: Observer (passive baseline), Actor (interactive baseline), ACE (curated context)

### This Study's Purpose

**Confirmatory validation** of ACE with faithful paper implementation:
- All ACE mechanisms enabled (feedback, retrieval, multi-round reflection, deduplication)
- Adequate power (n=67) to detect meaningful effects
- Preregistered hypotheses and decision rules
- Clean comparison to baselines with bugs fixed

---

## Agents (Experimental Conditions)

### 1. Observer (Passive Baseline)
**Description**: Language-only agent with no interaction capability

**Mechanism**:
- Receives initial environment description (t=0)
- No actions allowed (action_budget = 0)
- Answers queries based solely on initial text and reasoning

**Purpose**: Measures ceiling of pure language-based reasoning without interaction

**Expected Performance**:
- Accuracy: 40-50% (based on pilot: some environments solvable from description alone)
- Cost: ~$0.08/episode (~6,500 tokens)

---

### 2. Actor (Interactive Baseline)
**Description**: Interactive agent with explicit belief tracking

**Mechanism**:
- Takes up to 10 actions per episode
- Maintains probabilistic belief state (environment-specific priors)
- Bayesian updates after observations
- Memory of recent interactions (max 10 steps)
- Answers queries using updated beliefs + memory

**Belief Models**:
- **HotPot**: Gaussian model over heating rate (Bayesian linear regression updates)
- **SwitchLight**: Categorical distribution over wiring layouts (Bayesian updates)
- **ChemTile**: Categorical reaction outcome probabilities

**Purpose**: Standard interactive learning baseline (cost upper bound)

**Expected Performance**:
- Accuracy: 75-80% (pilot: 76.9%)
- Cost: ~$0.18/episode (~22,000 tokens)

---

### 3. ACE (Agentic Context Engineering)
**Description**: Interactive agent with curated playbook that learns across episodes

**Mechanism** (Faithful Paper Implementation):

**Per-Episode Cycle**:
1. **Generator**: Choose actions using playbook + top-k retrieval (5 bullets/section)
2. **Reflector**: Multi-round reflection (2 rounds) to extract insights from episode
3. **Curator**: Merge insights into playbook, maintain utility scores
4. **Refine**: Deduplicate (similarity > 0.8) and prune (if token cap set)
5. **Feedback**: Update helpful/harmful counts for used bullets based on success

**Playbook Structure**:
- 4 sections: "What to try", "What to avoid", "Environment dynamics", "Debugging strategies"
- Each bullet has: content, ID, helpful_count, harmful_count, last_used_step
- Retrieval: Embed bullets, retrieve top-5 most relevant per section per step
- Utility scoring: helpful_count - harmful_count

**ACE Configuration** (This Study):
- `use_retrieval: true` (top-k=5)
- `reflection_rounds: 2` (multi-round refinement)
- `max_epochs: 1` (single pass, no replay)
- `generator_temperature: 0.7` (same across Generator/Reflector/Curator)
- `curation_mode: "curated"` (LLM-based curation)
- `token_cap: null` (no budget constraints for validation)

**Purpose**: Test if curated context can match Actor accuracy at lower cost

**Expected Performance**:
- Accuracy: 70-75% (target ≥70% for validation)
- Cost: ≤$0.126/episode (≤70% of Actor cost for meaningful efficiency)

---

## Environments (Causal Reasoning Tasks)

### 1. HotPot Lab
**Causal Structure**: Continuous dynamics with hidden state

**Setup**: Control stove to heat pot to target temperature
- **Hidden state**: Actual pot temperature (not directly observable)
- **Actions**: measure_temp(), toggle_stove(), wait(t)
- **Observations**: Measured temp (noisy), stove light indicator, time
- **Causal challenge**: Infer heating rate, measurement noise, stove power from noisy observations

**Ground Truth Evaluation**:
- Q1: "What is the actual pot temperature?" (±2°C tolerance)
- Q2: "What power level is the stove at?" (off/low/medium/high)

---

### 2. SwitchLight
**Causal Structure**: Categorical hidden state with observational confound

**Setup**: Determine hidden wiring configuration
- **Hidden state**: Wiring layout (layout_A or layout_B, inverted logic)
- **Actions**: toggle_switch(), observe_light()
- **Observations**: Switch position, light state, time
- **Causal challenge**: Disambiguate wiring from observations (switch→light mapping is hidden)

**Ground Truth Evaluation**:
- Q1: "Which wiring layout is active?" (layout_A / layout_B)
- Q2: "What would happen if you toggle the switch now?" (light on/off)

---

### 3. ChemTile
**Causal Structure**: Stochastic reaction network

**Setup**: Discover reaction rules through experiments
- **Hidden state**: Reaction probability table (A+B→C?, C+B→D?, etc.)
- **Actions**: mix(chem_a, chem_b), measure(), reset_tile()
- **Observations**: Reaction outcomes (product or "nothing" or "explode"), time
- **Causal challenge**: Infer reaction probabilities from stochastic outcomes

**Ground Truth Evaluation**:
- Q1: "What happens when you mix A and B?" (product / nothing / explode)
- Q2: "What is the most reliable path to produce D?" (sequence of reactions)

---

## Experimental Design

### Sample Size
- **n = 67 seeds** per agent × environment combination
- **Total episodes**: 603 (3 agents × 3 environments × 67 seeds)

**Power Analysis**:
- Target effect size: d ≥ 0.4 (medium effect)
- Power: >80% for paired t-tests at α=0.05
- Sufficient to detect 5-10 percentage point accuracy differences

### Episode Structure
- **Action budget**: 10 actions per episode
- **Token budget**: 2000 tokens per LLM call
- **Models**: Claude Sonnet 4.5 (all agents), GPT-4 (evaluation judge)
- **Evaluation**: After episode ends, agent answers test queries WITHOUT ground truth access
- **Scoring**: Judge evaluates answers against ground truth (binary correct/incorrect per query)

### Experimental Controls
1. **Fixed seeds**: Ensures reproducibility and paired comparisons
2. **No ground truth leakage**: Validated programmatically (assertions in runner)
3. **Same model/temperature**: All agents use Claude Sonnet 4.5, ACE uses temp=0.7 across all roles
4. **Programmatic observation injection**: Prevents LLM hallucination of observations

---

## Primary Hypotheses

### H1a: ACE Accuracy Validation
**Hypothesis**: ACE achieves clinically meaningful accuracy (≥70%) on causal reasoning tasks.

**Operationalization**:
- Metric: Overall accuracy across all test queries (mean of per-query correctness)
- Success threshold: ACE accuracy ≥ 70%
- Statistical test: One-sample t-test vs 70% threshold
- Significance level: α = 0.025 (Bonferroni correction for H1a + H1b)

**Justification**: 70% represents meaningful performance on challenging causal tasks. Below this, ACE is not useful.

---

### H1b: ACE Cost Efficiency
**Hypothesis**: ACE achieves meaningful cost reduction (≤70% of Actor's cost) compared to interactive learning.

**Operationalization**:
- Metric: USD cost per episode (computed from input/output tokens × model pricing)
- Success threshold: ACE cost ≤ 0.70 × Actor cost
- Statistical test: Paired t-test (ACE vs Actor, paired by seed)
- Significance level: α = 0.025 (Bonferroni correction for H1a + H1b)

**Justification**: 30% cost reduction is operationally significant. Pilot showed ACE at 78% of Actor cost; optimization should achieve 70%.

---

### Combined Interpretation (H1a × H1b)

| H1a (Accuracy ≥70%) | H1b (Cost ≤70%) | Outcome | Interpretation |
|---------------------|-----------------|---------|----------------|
| ✅ Pass | ✅ Pass | **FULL SUCCESS** | ACE validated: accurate + efficient |
| ✅ Pass | ❌ Fail | **PARTIAL SUCCESS** | ACE accurate but not cost-efficient (publish accuracy story) |
| ❌ Fail | ✅ Pass | **WEAK SUCCESS** | ACE efficient but inaccurate (publish as negative result with lessons) |
| ❌ Fail | ❌ Fail | **FAILURE** | ACE neither accurate nor efficient (publish limits paper) |

---

## Secondary Hypotheses (Exploratory)

### H2: ACE vs Actor Comparison
**Question**: How does ACE accuracy compare to Actor accuracy?

**Analysis**:
- Paired t-test (ACE vs Actor accuracy, paired by seed)
- Report Cohen's d effect size
- 95% bootstrap CI on accuracy difference
- **No FWER correction** (exploratory)

**Prediction**: ACE accuracy within 5 percentage points of Actor (non-inferiority margin)

---

### H3: Interaction Effect
**Question**: Does ACE's advantage vary by environment?

**Analysis**:
- Two-way ANOVA: Agent × Environment interaction on accuracy
- Post-hoc comparisons if interaction significant
- **No FWER correction** (exploratory)

**Prediction**: ACE performs better on environments with learnable patterns (HotPot, ChemTile) vs pure exploration (SwitchLight)

---

### H4: Observer Baseline
**Question**: Does interaction improve over language-only reasoning?

**Analysis**:
- Paired t-tests: Actor vs Observer, ACE vs Observer
- Report effect sizes
- **No FWER correction** (exploratory)

**Expected**: Observer <50%, confirming interaction is necessary

---

## Metrics (Will Report All)

### Primary Metrics

**1. Accuracy** (per agent, per environment, overall)
- Overall accuracy: % correct across all test queries
- Per-environment accuracy: % correct within each environment
- Per-agent accuracy: Mean accuracy across environments
- 95% bootstrap CI for all accuracy estimates

**2. Cost** (per agent, per environment, overall)
- USD cost per episode: (input_tokens × $0.003 + output_tokens × $0.015) / 1000
- Total cost per agent: Sum across all episodes
- Cost efficiency: ACE cost / Actor cost (ratio)
- 95% bootstrap CI for cost estimates

**3. ACE-Specific Metrics**
- Playbook size: Number of bullets over time
- Utility distribution: helpful_count - harmful_count per bullet
- Retrieval precision: % of retrieved bullets actually used in actions
- Curation acceptance rate: % of Reflector insights accepted by Curator

---

### Secondary Metrics (Exploratory)

**4. Calibration**
- Brier score: Mean squared error of confidence vs correctness
- Expected Calibration Error (ECE): Binned calibration gap

**5. Token Breakdown** (preregistration requirement)
- Per-agent token usage: Input vs output tokens
- Per-phase token usage (ACE): Generator vs Reflector vs Curator
- Token accounting validation: Verify all token counts sum correctly

**6. Episode Dynamics**
- Action diversity: Unique actions per episode
- Action efficiency: Test accuracy vs number of actions
- Belief convergence (Actor): Variance reduction over time

---

## Statistical Analysis Plan

### Primary Comparisons

**H1a (ACE ≥ 70%)**:
- One-sample t-test: ACE accuracy vs 70%
- Bonferroni-corrected α = 0.025
- Report exact p-value, 95% CI, and effect size

**H1b (ACE cost ≤ 70% of Actor)**:
- Paired t-test: ACE cost vs (0.70 × Actor cost), paired by seed
- Bonferroni-corrected α = 0.025
- Report exact p-value, 95% CI, and effect size

---

### Secondary Comparisons (Exploratory)

**H2 (ACE vs Actor accuracy)**:
- Paired t-test, paired by seed
- Cohen's d effect size
- 95% bootstrap CI (1000 resamples)
- No multiple comparisons correction (exploratory)

**H3 (Interaction)**:
- Two-way ANOVA: Agent (3) × Environment (3)
- Post-hoc Tukey HSD if interaction significant
- Report η² effect sizes

**H4 (Observer baseline)**:
- Paired t-tests: Actor vs Observer, ACE vs Observer
- Cohen's d effect sizes
- No multiple comparisons correction (exploratory)

---

### Confidence Intervals
- **All accuracy metrics**: 95% bootstrap CI (1000 resamples, stratified by environment)
- **All cost metrics**: 95% bootstrap CI (1000 resamples, stratified by environment)
- **Rationale**: Bootstrap handles non-normal distributions and respects seed pairing

---

### Significance Threshold
- **Primary hypotheses (H1a, H1b)**: α = 0.025 each (Bonferroni correction for 2 tests)
- **Secondary hypotheses**: α = 0.05 (no FWER control, labeled exploratory)
- **Reporting**: Always report exact p-values, never "p < 0.05" without exact value

---

## Decision Rules (Pre-Commit to Interpretation)

### GREEN LIGHT (Publish as Validation)
**Criteria**: H1a AND H1b both pass
- ACE accuracy ≥ 70% (p < 0.025)
- ACE cost ≤ 70% of Actor (p < 0.025)

**Story**: "ACE validated: Curated context achieves competitive accuracy at reduced cost"

**Publication target**: ML conference (NeurIPS, ICLR, ICML)

---

### AMBER LIGHT (Publish Hybrid Story)
**Criteria**: H1a passes XOR H1b passes (one but not both)

**Case 1 - Accurate but Expensive** (H1a pass, H1b fail):
- Story: "ACE achieves target accuracy but needs optimization for cost efficiency"
- Focus: Accuracy results, ablations, cost optimization opportunities
- Publication target: Workshop or domain venue

**Case 2 - Efficient but Inaccurate** (H1a fail, H1b pass):
- Story: "ACE achieves cost efficiency but accuracy needs improvement"
- Focus: Cost reduction mechanisms, accuracy-cost tradeoffs, future work
- Publication target: Workshop or negative results track

---

### RED LIGHT (Publish Limits Paper)
**Criteria**: Both H1a AND H1b fail
- ACE accuracy < 70% (not useful)
- ACE cost > 70% of Actor (not efficient)

**Story**: "When does context curation fail? Lessons from ACE"

**Focus**:
- Where ACE fails (environment-specific analysis)
- Why curation didn't help (playbook analysis)
- What we learned about LLM limitations
- Comparison to Actor as upper bound

**Publication target**: Negative results track or workshop

---

## What We Will NOT Change Mid-Study

### Locked Parameters (Cannot Modify After Data Collection Begins)

**Agent configurations**:
- Action budgets (10 actions/episode)
- Token budgets (2000 tokens/call)
- Model versions (Claude Sonnet 4.5, GPT-4)
- ACE configuration (reflection_rounds=2, top_k=5, etc.)

**Experimental design**:
- Seeds (67 per condition, [42-108], [100-166], [200-266])
- Environments (HotPot, SwitchLight, ChemTile)
- Agents (Observer, Actor, ACE)
- Test queries (defined per environment)

**Statistical plan**:
- Primary hypotheses (H1a, H1b)
- Significance thresholds (α=0.025 for primary)
- Success criteria (70% accuracy, 70% cost ratio)

**Evaluation**:
- Ground truth never exposed to agents
- Judge model (GPT-4)
- Scoring rubrics (binary correct/incorrect)

---

### What We CAN Change (Pre-Specified)

**Allowed adjustments** (must be documented):
- Worker count for parallel execution (does not affect results)
- Rate limiting parameters (does not affect results)
- Bug fixes that don't change agent logic (must document and justify)
- Output format/visualization (does not affect data)

**Documentation requirement**: Any change must be logged with:
- Timestamp
- Description of change
- Justification (why necessary)
- Impact assessment (affects results? yes/no)

---

## Data Quality & Validation

### Pre-Flight Checks (Before Main Study)
1. ✅ Test ACE bug fix with mini-run (3 episodes)
2. ✅ Verify belief/surprise tracking works for Actor
3. ✅ Confirm token accounting sums correctly
4. ✅ Validate ground truth never leaks to agents
5. ✅ Check all ace_config parameters load correctly

### During-Study Monitoring
- Monitor failed episodes (retry up to 3 times for API errors)
- Log all exceptions with full traceback
- Save raw episode logs (not just aggregates)
- Track rate limiter stats (wait times, throttle events)

### Post-Study Validation
- Verify all 603 episodes completed successfully
- Check token accounting: total = sum of per-phase tokens
- Validate no ground truth in any agent observations (programmatic scan)
- Reproduce key results from raw logs (spot check 10% of episodes)

---

## Deviations from Pilot

### Changes Made Based on Pilot Learnings

**1. Removed Model-Based Agent**
- **Reason**: Pilot showed Model-Based (70.7%) underperformed Actor (76.9%) at same cost
- **Impact**: Focus resources on Observer vs Actor vs ACE comparison
- **Episodes saved**: 201 (3 envs × 67 seeds)

**2. Fixed ACE Implementation**
- **Issues found**:
  - Bug: test_results type mismatch crashed feedback updates
  - Missing: ace_config not specified (defaulted to single reflection round)
- **Fixes applied**:
  - Handle test_results as list in feedback loop
  - Add ace_config with reflection_rounds=2, retrieval=true
- **Impact**: ACE now matches paper specification

**3. Increased Sample Size**
- **Change**: n=50 → n=67 seeds per condition
- **Reason**: +34% more seeds for same total cost (removed Model-Based)
- **Power**: Improved detection of 5-10pp accuracy differences

**4. Verified Belief/Surprise Tracking**
- **Issue**: Suspected belief states not updating for Actor
- **Resolution**: Beliefs ARE updating correctly (only updates on observations with measurements)
- **Impact**: No changes needed, Actor working as designed

---

## Estimated Cost & Time

### Cost Breakdown (Based on Pilot Extrapolation)

| Agent | Episodes | Cost/Episode | Total Cost |
|-------|----------|--------------|------------|
| Observer | 201 | $0.08 | $16.08 |
| Actor | 201 | $0.18 | $36.18 |
| ACE | 201 | $0.14 | $28.14 |
| **Total** | **603** | - | **$80.40** |

**Note**: ACE cost assumes 2× reflection overhead. May be higher with faithful implementation.

**Contingency**: Budget $100 to account for variance and API retries

---

### Time Estimate

**Sequential**: ~30-40 hours (not practical)

**Parallel execution**:
- 4 workers: ~6-8 hours
- 6 workers: ~4-6 hours
- 10 workers: ~3-4 hours (recommended)
- 15 workers: ~2-3 hours (max, may hit rate limits)

**Recommendation**: Use 10 workers with conservative rate limiting

---

## Implementation Checklist

### Pre-Run Validation
- [ ] All code changes committed with descriptive messages
- [ ] Git tag created with SHA for preregistration
- [ ] Config file validated (ace_config present, all params correct)
- [ ] Mini test run (3 episodes) completed successfully
- [ ] Token accounting verified
- [ ] Ground truth leakage checks passing

### Execution
- [ ] Backup previous results directory
- [ ] Clear failed_episodes.json from previous runs
- [ ] Start experiment with --workers 10
- [ ] Monitor console for errors
- [ ] Track progress (episodes completed / total)

### Post-Run
- [ ] Verify 603 episodes completed (0 failures)
- [ ] Run validation scripts
- [ ] Archive raw results
- [ ] Compute all preregistered metrics
- [ ] Generate figures and tables
- [ ] Write results according to decision rules

---

## Commit & Sign-Off

**This preregistration will be committed to git with tag `prereg-v2.0` before data collection begins.**

**Signed** (via git commit):
- Date: (to be filled at commit)
- Git SHA: (to be filled at commit)
- Files locked: config_full_study_3agents.yaml, agents/*.py, environments/*.py, experiments/runner.py

**Data collection begins**: After all pre-flight checks pass

---

## Appendix: ACE Implementation Details

### Faithful Paper Implementation Checklist

✅ **Feedback Loop**:
- Track referenced_bullets during each episode
- Increment helpful_count on episode success
- Increment harmful_count on episode failure
- Pass utility scores (helpful - harmful) to Curator

✅ **Top-K Retrieval**:
- Embed playbook bullets using sentence-transformers (or TF-IDF fallback)
- Retrieve top-5 most relevant bullets per section
- Build context query from observation + recent memory
- Update embeddings when playbook changes

✅ **Multi-Round Reflection**:
- reflection_rounds = 2 (first round generates, second refines)
- Pass prior_insights to second round
- LLM temperature = 0.7 for all Reflector calls

✅ **Deduplication**:
- Semantic similarity threshold = 0.8
- Merge feedback counts when deduplicating (sum helpful/harmful)
- Keep bullet with higher utility score

✅ **Pruning**:
- token_cap = null (no pruning for validation study)
- If enabled: utility-based pruning (keep high helpful-harmful bullets)

✅ **Same Model/Temperature**:
- Generator: Claude Sonnet 4.5, temp=0.7
- Reflector: Claude Sonnet 4.5, temp=0.7
- Curator: Claude Sonnet 4.5, temp=0.7

✅ **Curation Mode**:
- Mode: "curated" (LLM-based merge with conflict resolution)
- Alternative modes available for ablations: "no_curate", "random", "greedy"

---

**END OF PREREGISTRATION**
