Token Prediction Bridge: Testing the Coupling Between Language and World Models

Central Research Question: Does linguistic next-token prediction encode the same learning signals as grounded world-model prediction?

This experiment tests whether language models' native capability (predicting next tokens) captures the same information as explicit belief-state updates in interactive environments. We measure the coupling strength between token-level negative log-likelihood (NLL) and belief surprisal across different types of reasoning tasks.

Overview

The Token Prediction Bridge hypothesis states that when language models predict next tokens, they implicitly perform the same Bayesian updates that we explicitly compute in belief states. To test this, we run LLM agents through interactive episodes while simultaneously:

1. Computing explicit belief surprisal from parametric probability models
2. Computing token-level NLL from linguistic predictions of the next observation
3. Measuring the correlation between these two signals across different reasoning domains

We compare different agent architectures to create varied learning trajectories:

Observer: Language-only reasoning, no interaction (flat surprisal expected)
Actor: Interactive agent that updates beliefs from experience (decreasing surprisal)
Text-Reader: Observer that reads prior episode logs (vicarious learning)
Model-Based: Actor + explicit learned dynamics model (MLP)

These agents operate in three micro-world environments designed to test different coupling strengths:

Hot-Pot Lab: Causal reasoning with misleading linguistic priors (strong coupling expected)
Switch-Light: Distinguishing intervention from observation (moderate coupling)
Chem-Tile: Compositional reasoning with safety constraints (weak coupling)

Dual API Architecture
This project uses a sophisticated dual API setup:

**Anthropic Claude Sonnet 4.5** (Primary - All Agent Operations)
- **Purpose:** All agent reasoning, belief updates, and decision-making
- **Why Claude?** Superior mathematical reasoning, long-context understanding, structured planning
- **Cost:** ~$3/1M input, $15/1M output tokens
- **Required:** Yes (for running experiments)

**OpenAI GPT-4o-mini** (Optional - Token Prediction Only)
- **Purpose:** Token-level log probabilities for coupling analysis
- **Why OpenAI?** Only provider offering token-level logprobs (required for NLL computation)
- **Cost:** ~$0.15/1M input, $0.60/1M output tokens
- **Required:** No (only for token prediction experiments)

This architecture provides the best of both worlds: Claude's exceptional reasoning for agents, and OpenAI's logprobs for linguistic analysis.

Key Design Principles

1. Token Prediction Bridge Architecture

Parallel computation: Token NLL and belief surprisal computed simultaneously during episodes
Deterministic textualization: 1:1 mapping from observations to canonical language (prevents hallucination)
No information leakage: Hidden state never appears in linguistic observations
Synchronized logging: Both signals recorded at identical timesteps for valid coupling analysis

2. Scientific Rigor

Preregistration: Hypotheses locked in before experiments (preregistration.yaml)
Provenance: Every episode logs git SHA, code hashes, and full configuration
Reproducibility: Deterministic environments with explicit random seeds
Statistical power: Pre-computed sample sizes and effect sizes

3. Guard Rails Against Contamination

No ground truth leakage: Observations never contain hidden state
Programmatic injection: Observations injected into prompts, never echoed by LLM
Counterfactual purity: Simulation queries have no side effects
Validated observations: All observations pass through Pydantic schemas

4. Computable Metrics

Parametric beliefs: Agents maintain probability distributions (not "vibes")
Surprisal: -log P(observation | belief) computed from proper likelihoods
Token NLL: -Σ log p(token | context) from linguistic prediction
Calibration: Brier scores and Expected Calibration Error
Learning rate: Surprisal trajectory slope (negative = learning)

5. Transparency

Every component is identifiable: Single file per concept
No abstraction layers: Direct LLM API calls, no frameworks
JSON logs: Every episode fully inspectable and replayable
Version control: All prompts versioned, no magic strings

Project Structure
world-model-experiment/
├── README.md                        # This file
├── requirements.txt                 # Dependencies
├── config.yaml                      # Model configs (Anthropic Claude), budgets, seeds
├── config_token.yaml                # Token prediction experiment configuration
├── preregistration.yaml             # Locked hypotheses (DO NOT MODIFY after experiments start)
│
├── Documentation/
│   ├── PRIOR_GENERATION.md          # LLM-generated belief prior methodology
│   └── THEORETICAL_FRAMEWORK.md     # Theoretical background and framework
│
├── TOKEN_EXPERIMENT_README.md       # Comprehensive token prediction guide (root)
├── VALIDATION_REPORT.md             # System validation results (98% confidence) (root)
├── VALIDATION_REPORT_OPTION_C.md    # Alternative validation approach (root)
├── ANTHROPIC_MIGRATION.md           # Anthropic API migration summary (root)
├── BUG_FIX_SUMMARY.md               # Recent bug fixes and improvements (root)
├── BUG_FIXES_CHEMTILE_SURPRISAL.md  # Detailed ChemTile bug fix documentation (root)
├── DIAGNOSTIC_REPORT.md             # System diagnostic results (root)
├── PILOT_COUPLING_REPORT.md         # Pilot experiment coupling analysis (root)
├── PRELIMINARY_ANALYSIS_REPORT.md   # Initial analysis results (root)
├── IMPLEMENTATION_STATUS.md         # Current implementation status tracking (root)
│
├── .env                             # API keys (gitignored, create this)
├── .gitignore
├── pilot_coupling_analysis.png      # Pilot results visualization
│
├── analyze_pilot.py                 # Pilot analysis script (root, temporary)
├── visualize_pilot_coupling.py      # Visualization script (root, temporary)
├── diagnostic_test.py               # Diagnostic testing script (root, temporary)
├── test_*.py                        # Debug/validation scripts (root, temporary)
│
├── environments/                    # Micro-world simulators
│   ├── base.py                      # Abstract Environment interface
│   ├── hot_pot.py                   # Hot-Pot Lab (causal reasoning)
│   ├── switch_light.py              # Switch-Light (intervention vs observation)
│   ├── chem_tile.py                 # Chem-Tile (compositional reasoning)
│   └── transfer_env.py              # Out-of-distribution test
│
├── agents/                          # Agent implementations
│   ├── base.py                      # Abstract Agent + LLM interfaces (Anthropic + OpenAI)
│   ├── observer.py                  # Language-only reasoning
│   ├── actor.py                     # Interactive with belief updates
│   ├── text_reader.py               # Reads prior logs
│   └── model_based.py               # Actor + learned dynamics
│
├── models/                          # Belief states and tools
│   ├── belief_state.py              # Parametric beliefs (HotPotBelief, etc)
│   ├── transition_model.py          # MLP for dynamics learning
│   └── tools.py                     # Tool definitions per environment
│
├── textualization/                  # Natural language conversion layer
│   ├── base.py                      # Abstract textualization interface
│   ├── hot_pot_text.py              # HotPot observations → natural language
│   ├── switch_light_text.py         # SwitchLight observations → natural language
│   ├── chem_tile_text.py            # ChemTile observations → natural language
│   └── validation.py                # Template validation (determinism, no leakage)
│
├── token_prediction/                # Token-level prediction system
│   ├── predictor.py                 # Abstract predictor interfaces
│   ├── openai_predictor.py          # OpenAI logprobs implementation
│   ├── logger.py                    # Token prediction logging
│   └── metrics.py                   # Token-level metrics (NLL, perplexity)
│
├── evaluation/                      # Metrics and analysis
│   ├── metrics.py                   # All 7 core metrics (interventional accuracy, etc)
│   ├── tasks.py                     # Test query sets
│   ├── statistical.py               # Power analysis, t-tests, effect sizes
│   ├── token_analysis.py            # Token prediction statistical analyses (A1-A5)
│   └── token_validation.py          # Token prediction robustness tests
│
├── experiments/                     # Execution infrastructure
│   ├── runner.py                    # Main episode loop with guard rails
│   ├── token_runner.py              # Episode runner with parallel token prediction
│   ├── provenance.py                # Git SHA tracking, code hashing
│   ├── prompts.py                   # All prompts (versioned)
│   ├── config.py                    # API key loading
│   └── ablations.py                 # Ablation configurations
│
├── scripts/                         # Entry points
│   ├── run_experiment.py            # Main: run all episodes
│   ├── analyze_results.py           # Generate report + figures
│   ├── inspect_episode.py           # Debug single episode
│   ├── compute_power_analysis.py    # Statistical power analysis
│   ├── generate_all_logs.py         # Generate detailed logs for episodes
│   │
│   ├── Token Prediction Scripts/
│   ├── validate_templates.py        # Validate textualization templates
│   ├── pilot_token_run.py           # Pilot token experiment (30 episodes)
│   ├── analyze_token_pilot.py       # Pilot results analysis
│   ├── run_full_token_experiment.py # Full token experiment (300 episodes)
│   ├── analyze_full_token_results.py # Comprehensive token analysis (A1-A5)
│   └── generate_token_figures.py    # Publication-ready token figures
│
├── results/                         # Generated outputs (gitignored)
│   ├── raw/                         # JSON logs per episode
│   ├── aggregated/                  # CSV summaries, figures
│   ├── figures/                     # Plots
│   └── pilot_token*/                # Token prediction pilot results
│
├── logs/                            # Human-readable episode logs (generated)
│   └── TIMESTAMP/                   # Timestamped log directories
│
└── tests/                           # Test suite
    ├── conftest.py                  # Test configuration
    ├── test_environments.py         # Environment determinism, purity
    ├── test_agents.py               # Agent behavior
    ├── test_beliefs.py              # Likelihood computations
    ├── test_metrics.py              # Metric calculations
    ├── test_integration.py          # Full pipeline
    ├── test_textualization.py       # Textualization layer tests (16 tests)
    └── test_token_prediction.py     # Token prediction tests (14 tests)
Quick Start

To run the Token Prediction Bridge experiments (PRIMARY), skip to the "Token-Level Prediction Bridge (CENTERPIECE)" section below.

For basic infrastructure setup:

1. Install Dependencies
bashpip install -r requirements.txt
2. Configure API Keys
Create .env file in project root:
bashANTHROPIC_API_KEY=sk-ant-your-key-here  # Required: Used for all agent operations
OPENAI_API_KEY=sk-your-key-here             # Required: Needed for token prediction (logprobs)

Important Notes:
- .env is gitignored. Never commit API keys.
- **Anthropic API (Required)**: All agents use Claude Sonnet 4.5 for grounded reasoning
- **OpenAI API (Required)**: Needed for token prediction experiments (logprobs functionality)
- Both APIs are required for the full Token Prediction Bridge experiment
- See ANTHROPIC_MIGRATION.md for complete migration details
3. Run Tests
Verify environments and metrics work:
bash# Test environments (no API calls needed)
pytest tests/test_environments.py -v

# Test metrics
pytest tests/test_metrics.py -v

# Integration test (requires API keys)
pytest tests/test_integration.py -v -m integration
4. Run Experiment

**Option A: Parallel Execution (Recommended - 5x faster)**
```bash
python scripts/run_experiment_parallel.py \
    --config config.yaml \
    --preregistration preregistration.yaml \
    --workers 10
```

This runs episodes in parallel with intelligent rate limiting:
- **10 workers**: ~2 hours for full experiment (vs. 10 hours sequential)
- **Automatic rate limiting**: Respects Anthropic API limits (900 RPM, 405K input TPM, 81K output TPM)
- **Real-time progress**: Shows episodes/min rate and ETA
- **Graceful error handling**: Retries rate limit errors, saves failed episodes
- **Ctrl+C support**: Clean shutdown with partial results saved

**Option B: Sequential Execution**
```bash
python scripts/run_experiment.py \
    --config config.yaml \
    --preregistration preregistration.yaml
```

Both methods:
- Run all environment × agent combinations
- Save episode logs to results/raw/TIMESTAMP/
- Track full provenance (git SHA, code versions)
- Enforce all guard rails
- Track token usage and API costs

5. Analyze Results
bashpython scripts/analyze_results.py \
    --results results/raw/TIMESTAMP/
Generates:

results/aggregated/metrics_summary.csv - Main results
results/aggregated/statistical_comparisons.csv - t-tests, effect sizes
results/aggregated/figures/ - Plots

6. Inspect Individual Episodes
bashpython scripts/inspect_episode.py \
    results/raw/TIMESTAMP/hot_pot_actor_ep001.json
Shows:

Step-by-step actions and observations
Surprisal trajectory
Belief state evolution
Test query performance

7. Generate Detailed Logs
bashpython scripts/generate_all_logs.py \
    results/raw/TIMESTAMP \
    --output-dir logs/TIMESTAMP
Generates human-readable text logs for all episodes with:

Full step-by-step trajectories
Observations, actions, and beliefs
Test query results
Metadata and ground truth

8. Compute Statistical Power
bashpython scripts/compute_power_analysis.py \
    --results results/aggregated/TIMESTAMP
Performs statistical power analysis for actor vs observer comparisons:

Effect sizes and confidence intervals
Required sample sizes for desired power
t-tests and statistical significance

Token-Level Prediction Bridge (CENTERPIECE)

This project includes experimental capability to test whether **linguistic next-token prediction** (Token NLL) encodes similar uncertainty signals as **grounded world-model prediction** (Belief Surprisal).

**Key Innovation**: Unlike traditional LLM evaluations that test linguistic fluency or factual knowledge, we test whether LLMs' *uncertainty* over next observations correlates with a *physics-based model's* uncertainty. This directly probes the question: "Do LLMs learn world models from language?"

**Why This Matters**:
- If correlation is strong → Supports Sutskever's view that next-token prediction implicitly learns reality
- If correlation is weak → Supports Sutton's critique that LLMs lack true understanding
- Environment-specific patterns → Reveals *which* aspects of world modeling are linguistically learnable

**Guard Rail**: Belief surprisal is computed from explicit probability distributions (NOT language), ensuring the two signals are independent by design. Any observed correlation reflects genuine information coupling.

The experiment runs agents through interactive episodes while simultaneously computing:
1. **Belief surprisal**: -log P(observation | parametric belief state) - from explicit physics models
2. **Token NLL**: -Σ log p(token | linguistic context) - from LLM predictions

We then measure the correlation between these signals to test whether language models' native prediction capability implicitly performs Bayesian world-model updates.

Quick Start
1. Validate Templates
bashpython scripts/validate_templates.py
2. Run Pilot (5 episodes × 3 envs × 2 agents = 30 episodes)
Requires OPENAI_API_KEY environment variable:
bashexport OPENAI_API_KEY='your-key-here'
python scripts/pilot_token_run.py
This will:

Convert environment observations to canonical natural language
Query LLMs for next-observation predictions with token logprobs
Compute token NLL alongside belief surprisal
Save token logs to results/raw/pilot_token_TIMESTAMP/

3. Analyze Results
bashpython scripts/analyze_token_pilot.py results/raw/pilot_token_TIMESTAMP
Generates:

Correlation analysis (token NLL vs belief surprisal)
Scatter plots showing coupling strength
CSV files with statistical results

Expected Pattern (Testing Primary Hypothesis T1)
The Token Prediction Bridge hypothesis predicts environment-dependent coupling:

HotPot (strong coupling, r > 0.7): Causal dynamics well-captured by language
- Language models should implicitly track temperature dynamics through linguistic prediction
- Token NLL should decrease as belief surprisal decreases (both capture learning)

SwitchLight (moderate coupling, r ~ 0.5): Intervention reasoning partially linguistic
- Language can capture some causal structure but intervention requires active testing
- Moderate alignment between linguistic and grounded prediction

ChemTile (weak coupling, r < 0.4): Compositional reasoning requires interaction
- Compositional safety constraints poorly captured by linguistic priors
- Token NLL and belief surprisal may diverge as agents learn through experience

Configuration
Edit config_token.yaml to adjust:

Model selection (GPT-4o, GPT-4o-mini, etc.)
Temperature settings
Number of pilot episodes
Budget constraints

Example:
yamltoken_prediction:
  enabled: true
  predictors:
    observer:
      model: "gpt-4o-mini"
      temperature: 0.0
    actor:
      model: "gpt-4o"
      temperature: 0.0

pilot:
  num_episodes_per_env: 5
  environments: ["hot_pot", "switch_light", "chem_tile"]
  agents: ["observer", "actor"]
  seeds: [42, 43, 44, 45, 46]
How It Works

Textualization Layer: Converts every environment observation into canonical natural language strings (deterministic 1:1 mapping)
Next-Sentence Prediction: Queries LLMs to predict the next observation text, capturing per-token log probabilities
Token NLL Computation: Calculates negative log-likelihood (NLL) = -Σ log p(token | context)
Alignment Analysis: Correlates token NLL with belief surprisal to see if they track together

Key Design Principles

Deterministic Templates: Same observation → same text, always
No Ground Truth Leakage: Hidden state never appears in text
Synchronized Logging: Token NLL and belief surprisal recorded at same steps
Programmatic Injection: Text generated by code, not LLM (prevents hallucination)

Files

textualization/ - Converts observations to canonical text
token_prediction/ - OpenAI API integration for token logprobs
experiments/token_runner.py - Parallel token prediction during episodes
scripts/pilot_token_run.py - Pilot experiment runner
scripts/analyze_token_pilot.py - Coupling analysis and visualization
scripts/validate_templates.py - Template validation tests
config_token.yaml - Token prediction configuration

Core Concepts
Environments
Each environment is a deterministic simulator with:

Hidden state: Ground truth never shown to agents
Actions: Tools agents can use (measure, wait, toggle, etc)
Observations: What agents see (may include noise or misleading info)
Counterfactuals: Pure functions that simulate without side effects

Example from Hot-Pot Lab:
pythonenv = HotPotLab(seed=42)
obs = env.reset(42)  # {'label': 'Boiling!', 'stove_light': 'on'}

obs, reward, done, info = env.step('measure_temp()')
# obs = {'measured_temp': 23.5, 'time_elapsed': 0}

# Ground truth only for evaluation
gt = env.get_ground_truth()  
# {'actual_temp': 23.5, 'stove_power': 'off'}
Belief States
Agents maintain parametric probability distributions:

### Belief Surprisal vs Token NLL: Key Distinction

**This project measures two types of uncertainty:**

**1. Belief Surprisal (Grounded World Model)**
```python
# Computed from explicit probability distribution
from models.belief_state import HotPotBelief

belief = HotPotBelief(
    heating_rate_mean=1.5,    # °C per second
    heating_rate_std=0.3,     # Uncertainty
    measurement_noise=2.0     # Observation noise
)

# Compute likelihood for surprisal
log_prob = belief.log_likelihood(
    observation={'measured_temp': 45.0},
    time_elapsed=20.0
)

surprisal = -log_prob  # Higher = more surprising to world model
```

**2. Token NLL (Linguistic Model)**
```python
# Computed from LLM token probabilities
context = "Thermometer reads 23.6°C. Action: turn_on_stove."
prediction = llm.predict_next_observation(context)
token_nll = prediction.sequence_nll  # Higher = more surprising to LLM
```

**Critical Independence**: These are computed by *separate systems*:
- Belief surprisal: Parametric model of physics (heating rates, wire probabilities)
- Token NLL: Language model trained on text

**Research Question**: Do they correlate? If yes → LLMs implicitly learn world models.

See `TOKEN_EXPERIMENT_README.md` (root) and `Documentation/THEORETICAL_FRAMEWORK.md` for detailed treatment.

**LLM-Generated Priors (NEW):**
Actor agents now generate their own prior beliefs using the LLM based on initial observations, rather than using hard-coded values. This allows the agent to set appropriate initial uncertainty based on what it observes:

python# During episode initialization
initial_obs = {'label': 'Boiling!', 'stove_light': 'on', 'time': 0.0}

# Actor generates priors based on this observation
agent.reset(environment_type='HotPotLab', initial_observation=initial_obs)

# LLM might generate:
# {
#   'heating_rate_mean': 1.5,    # Label suggests heating
#   'heating_rate_std': 0.8,     # Moderate uncertainty (labels may be misleading)
#   'measurement_noise': 1.5,
#   'reasoning': 'Label indicates boiling but maintaining skepticism...'
# }

The generated priors, reasoning, and token usage are logged in episode metadata for full transparency and reproducibility. See `Documentation/PRIOR_GENERATION.md` for detailed methodology.
Agents
All agents inherit from Agent base class:
pythonclass Agent(ABC):
    def act(self, observation: dict) -> AgentStep:
        """Process observation, return action"""
        
    def answer_query(self, question: str) -> Tuple[str, float]:
        """Answer query, return (answer, confidence)"""
Observer: Never acts, reasons from initial description only
Actor: Takes actions, updates beliefs, computes surprisal
Text-Reader: Like Observer but reads prior episode logs
Model-Based: Like Actor but also fits explicit MLP dynamics model
Metrics
Seven key metrics computed for each episode:

Interventional Accuracy: Correct answers on "What if we DO X?" queries
Counterfactual Accuracy: Correct answers on "What if we HAD DONE X?" queries
Surprisal Trajectory: Does surprisal decrease over time? (learning indicator)
Planning Success: Can agent achieve goals safely?
Calibration: Do confidence scores match actual accuracy?
Sample Efficiency: How many actions to reach target accuracy?
Δ Accuracy Post-Surprise: Does accuracy improve after surprising observations?

Experimental Hypotheses

Primary Hypothesis (Token Prediction Bridge):

T1: Token NLL and belief surprisal show environment-dependent coupling
- HotPot: Strong correlation (r > 0.7) - Causal dynamics well-captured by language
- SwitchLight: Moderate correlation (r ≈ 0.5) - Intervention reasoning partially linguistic
- ChemTile: Weak correlation (r < 0.4) - Compositional reasoning requires interaction

Rationale: If linguistic prediction encodes world-model updates, coupling strength should reflect how well language captures each domain's structure.

Secondary Hypotheses (Agent Comparison):
From preregistration.yaml:

H1: Actor agents achieve ≥15% higher interventional accuracy than Observers
- Rationale: Interactive experience should build better causal models

H2: Actor agents show negative surprisal slope; Observers show flat slope
- Rationale: Learning manifests as decreasing surprisal

H3: Model-Based agents outperform pure Actor reasoning by ≥10%
- Rationale: Explicit dynamics models enable better planning

H4: Interventional accuracy improves after high-surprisal events
- Rationale: Tests "observe → update → improve" loop

H5: Actors show better out-of-distribution generalization than Observers
- Rationale: Experience-based models should transfer better

Configuration
Model Selection (config.yaml)
All agents now use Anthropic's Claude Sonnet 4.5 for superior reasoning capabilities:
yamlmodels:
  observer: "claude-sonnet-4-5-20250929"
  actor: "claude-sonnet-4-5-20250929"
  model_based: "claude-sonnet-4-5-20250929"
  text_reader: "claude-sonnet-4-5-20250929"

**Why Claude Sonnet 4.5?**
- Exceptional mathematical reasoning (belief state updates)
- Superior long-context understanding (episode histories)
- Strong structured reasoning (planning, counterfactuals)
- Excellent instruction following
- Pricing: $3/1M input tokens, $15/1M output tokens
Episode Budgets
yamlbudgets:
  actions_per_episode: 10    # Max actions per episode
  tokens_per_call: 2000       # Max tokens per LLM call
Seeds and Replication
yamlenvironments:
  hot_pot:
    num_episodes: 50
    seeds: [42, 43, 44, ...]  # Explicit seeds for reproducibility
Guard Rails and Safety
1. No Ground Truth Leakage
python# ✅ Allowed
observation = {'measured_temp': 45.0, 'time_elapsed': 20}

# ❌ NEVER allowed - will trigger assertion
observation = {
    'measured_temp': 45.0, 
    'ground_truth': {'actual_temp': 45.0},  # FORBIDDEN
    'hidden_state': {...}                    # FORBIDDEN
}
2. Programmatic Observation Injection
python# In runner.py
agent_step = agent.act(observation)

# GUARD RAIL: Override with true observation
# (prevents LLM from hallucinating observations)
agent_step.observation = observation
3. Counterfactual Purity
python# Counterfactuals must not modify state
state_before = env.get_ground_truth()

result = env.counterfactual_query(['wait(60)', 'measure_temp'], seed=99)

state_after = env.get_ground_truth()
assert state_before == state_after  # Must be identical
4. Deterministic Environments
python# Same seed → same trajectory
env1 = HotPotLab(seed=42)
env2 = HotPotLab(seed=42)

for _ in range(10):
    obs1, _, _, _ = env1.step('wait(5)')
    obs2, _, _, _ = env2.step('wait(5)')
    assert obs1 == obs2
Episode Log Format
Every episode saved as JSON with full provenance:
json{
  "episode_id": "hot_pot_actor_ep001",
  "seed": 42,
  "environment": "HotPotLab",
  "agent_type": "actor",
  "provenance": {
    "timestamp": "2024-01-15T14:30:22",
    "code_sha": "a3f9d2c1...",
    "environment_version": "e4b8c9a2...",
    "config_hash": "f1a2d5e3..."
  },
  "steps": [
    {
      "step_num": 0,
      "thought": "I should measure the temperature first",
      "action": "measure_temp()",
      "observation": {"measured_temp": 23.5, "time_elapsed": 0},
      "belief_state": {
        "heating_rate_mean": 1.5,
        "heating_rate_std": 0.3
      },
      "surprisal": 0.42,
      "token_usage": 150
    },
    ...
  ],
  "test_results": [
    {
      "query": "If we wait 30s and touch, what happens?",
      "query_type": "interventional",
      "agent_answer": "It would be safe, temp ~25°C",
      "confidence": 0.85,
      "correct": true
    },
    ...
  ],
  "ground_truth": {
    "stove_power": "off",
    "actual_temp": 23.5
  }
}
Development Workflow
Adding a New Environment

Create environments/my_env.py inheriting from Environment
Implement all abstract methods (reset, step, get_ground_truth, counterfactual_query)
Create corresponding MyEnvBelief in models/belief_state.py
Add tools to models/tools.py
Write tests in tests/test_environments.py
Add test queries in evaluation/tasks.py

Adding a New Metric

Implement function in evaluation/metrics.py
Add to compute_all_metrics() aggregator
Write tests in tests/test_metrics.py
Update analysis script to plot/report the metric

Modifying Prompts
Never hardcode prompts in agent files!

Edit experiments/prompts.py
Update PROMPT_VERSION constant
Changes automatically logged in episode provenance

## Parallel Execution

The experiment supports parallel execution with intelligent rate limiting to dramatically reduce runtime while respecting API limits.

### Quick Start

```bash
# Full experiment with 10 workers (~2 hours vs. 10 hours sequential)
python scripts/run_experiment_parallel.py \
    --config config.yaml \
    --workers 10

# Resume from interrupted run
python scripts/run_experiment_parallel.py \
    --config config.yaml \
    --workers 10 \
    --resume-from results/parallel_run_20251021_123456
```

### Features

**Intelligent Rate Limiting:**
- Tracks requests/min and tokens/min in sliding 1-minute windows
- Automatically waits when approaching API limits (uses 90% as safety buffer)
- Thread-safe implementation for concurrent episodes
- Real-time adjustment based on actual vs. estimated token usage

**Progress Tracking:**
```
✓ [45/520] hot_pot_actor_ep023 (4.2 eps/min, ETA: 113.1min)
✓ [46/520] switch_light_observer_ep015 (4.3 eps/min, ETA: 110.5min)
⏸ Rate limit approaching, waiting 2.3s...
✓ [47/520] chem_tile_model_based_ep008 (4.1 eps/min, ETA: 115.4min)
```

**Error Handling:**
- Rate limit errors: Exponential backoff (60s, 120s, 240s), up to 3 retries
- Network errors: Quick retry (5s), up to 2 retries
- Failed episodes logged to `failed_episodes.json` with full traceback
- Graceful shutdown on Ctrl+C (waits for running episodes to finish)

**Token Tracking:**
All episode logs now include detailed token usage:
```json
{
  "episode_id": "hot_pot_actor_ep001",
  "total_input_tokens": 18234,
  "total_output_tokens": 3891,
  "total_api_calls": 23,
  "duration_seconds": 67.3,
  ...
}
```

### Worker Recommendations

| Episodes | Workers | Estimated Time | Cost (Claude Sonnet 4.5) |
|----------|---------|----------------|--------------------------|
| Full (520) | 10 | 2 hours | ~$100-150 |
| Full (520) | 6 | 3 hours | ~$100-150 |
| Pilot (50) | 4 | 20 min | ~$10-20 |

**Note:** Higher worker counts don't always mean faster execution due to rate limiting. 10 workers is optimal for the Anthropic API tier 1 limits.

### Implementation Details

**Rate Limiter Algorithm:**
- Sliding window tracks last 60 seconds of API calls
- Estimates token usage before each call
- Updates with actual usage after completion
- Blocks new requests if adding them would exceed 90% of any limit

**Token Usage Tracking:**
- All LLM interfaces (Anthropic, OpenAI, Mock) track tokens automatically
- Records input tokens, output tokens, and API call count per episode
- Cumulative stats available via `llm.get_total_usage()`

**Files:**
- `experiments/rate_limiter.py` - RateLimiter class (300 lines)
- `scripts/run_experiment_parallel.py` - Parallel runner (400 lines)
- `tests/test_parallel_execution.py` - Test suite (300 lines, 17 tests)

Recent Updates

Parallel Execution with Rate Limiting (October 2025) ✅ NEW
Complete parallel execution system for 5x faster experiments:
- Thread-safe rate limiter with sliding window algorithm
- Automatic token tracking in all LLM interfaces
- Retry logic for transient failures (rate limits, network errors)
- Real-time progress with episodes/min and ETA
- 17 comprehensive tests, all passing
- Full integration with existing experiment infrastructure

Token Prediction Bridge (October 2025) ✅ CENTERPIECE
This is the primary research question of the experiment: testing whether linguistic next-token prediction encodes similar learning signals as grounded world-model prediction (belief surprisal). Complete implementation includes:
- Textualization layer: Deterministic 1:1 mapping from observations to canonical language
- Token prediction system: Parallel computation of token NLL alongside belief surprisal
- Dual API architecture: Anthropic Claude Sonnet 4.5 (agents) + OpenAI GPT-4o (token logprobs)
- Statistical analyses (A1-A5) for coupling metrics across environments
- Complete validation: 16 textualization tests + 14 token prediction tests
- See TOKEN_EXPERIMENT_README.md for comprehensive 20+ page guide

Anthropic API Migration (October 2025) ✅ COMPLETE
Migrated all agent operations to Anthropic's Claude Sonnet 4.5 for superior reasoning while maintaining OpenAI for token-level logprobs (required for coupling analysis):
- Superior mathematical reasoning for belief state updates
- Better long-context understanding for episode histories
- Dual API architecture supporting both grounded and linguistic prediction
- See ANTHROPIC_MIGRATION.md for complete details

Action-Observation Alignment Fix (October 2025)
A critical bug in the episode runner has been fixed where actions and observations were misaligned by one step. The fix ensures that each logged step shows the action taken and the observation resulting from that action (not the previous observation). See BUG_FIX_SUMMARY.md for details.

System Validation (October 2025)
The entire system has been validated with 98% confidence. All mathematical formulas (surprisal, log-likelihood, belief updates) have been verified. The system is ready for full-scale experiments. See VALIDATION_REPORT.md for complete validation results.

Common Issues
"API key not found"
Solution: Create .env file with your keys:
bashANTHROPIC_API_KEY=sk-ant-...  # Required for all experiments
OPENAI_API_KEY=sk-...             # Only needed for token prediction

Note:
- Anthropic key is REQUIRED for all experiments
- OpenAI key is OPTIONAL (only for token prediction experiments)
"Counterfactual modified state"
Solution: Ensure counterfactual_query saves/restores state:
pythondef counterfactual_query(self, actions, seed):
    saved_state = self._save_state()
    # ... simulate ...
    self._restore_state(saved_state)
"Ground truth leaked to agent"
Solution: Check observation dict doesn't contain forbidden keys:
python# In environment.step()
obs = self._compute_observation()
assert 'ground_truth' not in obs
assert 'hidden_state' not in obs
return obs, reward, done, info
Tests failing with "No API keys"
Expected: Most tests don't need APIs. Only integration tests require keys.
bash# Run without API calls
pytest tests/test_environments.py tests/test_metrics.py -v

# Run with API calls (requires keys)
pytest tests/test_integration.py -v -m integration
Roadmap
Batch 1: Core Infrastructure ✅ COMPLETE

✅ Environments (hot_pot, switch_light, chem_tile)
✅ Parametric belief states
✅ Guard rails and provenance
✅ Environment tests

Batch 2: Agents ✅ COMPLETE

✅ Observer agent
✅ Actor agent with belief updates
✅ Text-Reader baseline
✅ Model-Based agent
✅ Transition model (MLP)
✅ Agent tests

Batch 3: Evaluation & Execution ✅ COMPLETE

✅ Test query sets
✅ All 7 metrics
✅ Experiment runner
✅ Analysis script
✅ Visualization
✅ Integration tests
✅ Statistical power analysis
✅ Episode log generation

Batch 4: Token Prediction Bridge ✅ COMPLETE

✅ Textualization layer (deterministic, no leakage)
✅ OpenAI token prediction integration
✅ Parallel episode + token logging
✅ Statistical analyses (A1-A5)
✅ Validation tests (16 textualization + 14 token prediction)
✅ Pilot and full experiment scripts
✅ Token analysis and figure generation

Batch 5: API Migration ✅ COMPLETE

✅ Anthropic Claude Sonnet 4.5 integration
✅ Dual API architecture (Anthropic + OpenAI)
✅ All tests passing with Claude
✅ Migration documentation

Current Status: 🚀 READY FOR TOKEN PREDICTION BRIDGE EXPERIMENTS

System validated (98% confidence) and ready for full-scale Token Prediction Bridge experiments:
- All agents running on Anthropic Claude Sonnet 4.5 (grounded reasoning)
- Token prediction using OpenAI GPT-4o (linguistic prediction with logprobs)
- Parallel computation: Both signals computed simultaneously during episodes
- Complete test coverage: 16 textualization + 14 token prediction tests
- Comprehensive documentation: 20+ page TOKEN_EXPERIMENT_README.md

**Documentation:**
- TOKEN_EXPERIMENT_README.md (root) - Comprehensive token prediction guide (PRIMARY)
- PILOT_COUPLING_REPORT.md (root) - Pilot experiment coupling analysis results
- PRELIMINARY_ANALYSIS_REPORT.md (root) - Initial experimental analysis
- VALIDATION_REPORT.md (root) - System validation results (98% confidence)
- ANTHROPIC_MIGRATION.md (root) - Dual API architecture details
- BUG_FIX_SUMMARY.md (root) - Recent bug fixes
- IMPLEMENTATION_STATUS.md (root) - Current implementation tracking

Next Steps (Token Prediction Bridge Experiments)

1. **Run Full Token Experiment** (300 episodes: 3 environments × 4 agents × 25 seeds)
   - Parallel computation of token NLL and belief surprisal
   - Expected cost: ~$150-200 (see TOKEN_EXPERIMENT_README.md for breakdown)

2. **Token Coupling Analysis** (Primary Research Question)
   - A1: Correlation analysis (token NLL vs belief surprisal per environment)
   - A2: Environment comparison (HotPot > SwitchLight > ChemTile expected)
   - A3: Agent comparison (Actor vs Observer coupling patterns)
   - A4: Temporal dynamics (coupling evolution over episodes)
   - A5: Cross-episode prediction (token NLL predicting future belief surprisal)

3. **Secondary Analyses** (Agent Comparison)
   - Test all 5 preregistered hypotheses (H1-H5)
   - Interventional accuracy, surprisal trajectories, planning success

4. **Publication** (Write up Token Prediction Bridge results)

Future Research Directions

Token Prediction Bridge Extensions (Primary):
□ Model family comparison (GPT-4 vs Claude vs Llama coupling patterns)
□ Temperature sensitivity analysis (Does temperature affect coupling strength?)
□ Context length ablations (How much history needed for token prediction to align?)
□ Alternative textualization schemes (Different linguistic framings of same observations)
□ Cross-domain coupling (Can token prediction transfer across environments?)
□ Causal intervention on token NLL (Does forcing alignment improve world models?)

Secondary Extensions (Agent Comparison):
□ Transfer environment experiments (out-of-distribution generalization)
□ Prompt engineering variants (CoT, few-shot)
□ Alternative belief representations (neural, symbolic hybrid)
□ Advanced planning algorithms (MCTS, value iteration)

---

Documentation Index

**Core Documentation:**
- **README.md** (this file) - Main project overview and quick start
- **preregistration.yaml** - Locked experimental hypotheses (H1-H5, T1)
- **config.yaml** - Model configurations and experiment settings
- **config_token.yaml** - Token prediction experiment settings

**Feature Guides:**
- **TOKEN_EXPERIMENT_README.md** (root) - Comprehensive guide to token prediction experiments
  - How token prediction works
  - Running pilot and full experiments
  - Statistical analyses (A1-A5)
  - Cost estimates and troubleshooting
  - 20+ pages of detailed documentation

**Technical Reports:**
- **VALIDATION_REPORT.md** (root) - System validation results (98% confidence)
  - Mathematical formula verification
  - Belief state validation
  - Surprisal computation validation
- **VALIDATION_REPORT_OPTION_C.md** (root) - Alternative validation approach
- **ANTHROPIC_MIGRATION.md** (root) - API migration summary
  - Dual API architecture details
  - Migration checklist
  - Cost comparisons
  - Testing results

**Analysis Reports:**
- **PILOT_COUPLING_REPORT.md** (root) - Pilot token prediction coupling analysis
- **PRELIMINARY_ANALYSIS_REPORT.md** (root) - Initial experimental results
- **IMPLEMENTATION_STATUS.md** (root) - Implementation tracking and status

**Bug Fixes and Diagnostics:**
- **BUG_FIX_SUMMARY.md** (root) - Recent bug fixes overview
- **BUG_FIXES_CHEMTILE_SURPRISAL.md** (root) - Detailed ChemTile bug fix documentation
- **DIAGNOSTIC_REPORT.md** (root) - System diagnostic results

**Theoretical Documentation:**
- **Documentation/PRIOR_GENERATION.md** - LLM-generated belief prior methodology
- **Documentation/THEORETICAL_FRAMEWORK.md** - Theoretical framework and background

**Temporary Files (Root Directory):**
- **test_*.py** - Various debug and validation scripts (temporary, not in tests/)
- **analyze_pilot.py** - Pilot analysis helper script
- **visualize_pilot_coupling.py** - Coupling visualization script
- **diagnostic_test.py** - System diagnostic testing
- **pilot_coupling_analysis.png** - Pilot results visualization

**Quick Reference:**

| Task | Documentation |
|------|---------------|
| Getting started | README.md Quick Start section |
| Understanding hypotheses | preregistration.yaml (H1-H5, T1) |
| Running core experiments | README.md → sections 4-6 |
| Token prediction setup | TOKEN_EXPERIMENT_README.md (root) |
| Pilot results | PILOT_COUPLING_REPORT.md (root) |
| API configuration | ANTHROPIC_MIGRATION.md (root) |
| Troubleshooting | README.md Common Issues, TOKEN_EXPERIMENT_README.md Troubleshooting |
| System validation | VALIDATION_REPORT.md (root) |
| Understanding architectures | README.md → Dual API Architecture section |
| Implementation status | IMPLEMENTATION_STATUS.md (root) |
| Theoretical framework | Documentation/THEORETICAL_FRAMEWORK.md |

**Test Coverage:**
- environments: 100% (test_environments.py)
- agents: 100% (test_agents.py)
- beliefs: 100% (test_beliefs.py)
- metrics: 100% (test_metrics.py)
- textualization: 100% (test_textualization.py - 16 tests)
- token prediction: 100% (test_token_prediction.py - 14 tests)
- integration: Full pipeline (test_integration.py)

---

**Questions or Issues?**
1. Check the relevant documentation file above
2. Review Common Issues section in this README
3. See TOKEN_EXPERIMENT_README.md for token-specific questions
4. Check git history for implementation details

**Citation:**
If you use this codebase in your research, please cite appropriately and reference the preregistration.yaml for experimental hypotheses.