World Model Experiment: Testing Interactive Learning in LLMs
Research Question: Can LLMs build better internal world models through interactive experience compared to pure language reasoning?
Overview
This project implements a controlled experiment comparing how different LLM agent architectures perform on prediction and planning tasks:

Observer: Language-only reasoning, no interaction
Actor: Interactive agent that updates beliefs from experience
Text-Reader: Observer that reads prior episode logs (vicarious learning)
Model-Based: Actor + explicit learned dynamics model (MLP)

We test these agents in three micro-world environments designed to isolate specific reasoning capabilities:

Hot-Pot Lab: Causal reasoning with misleading linguistic priors
Switch-Light: Distinguishing intervention from observation (do-calculus)
Chem-Tile: Compositional reasoning with safety constraints

Key Design Principles
1. Scientific Rigor

Preregistration: Hypotheses locked in before experiments (preregistration.yaml)
Provenance: Every episode logs git SHA, code hashes, and full configuration
Reproducibility: Deterministic environments with explicit random seeds
Statistical power: Pre-computed sample sizes and effect sizes

2. Guard Rails Against Contamination

No ground truth leakage: Observations never contain hidden state
Programmatic injection: Observations injected into prompts, never echoed by LLM
Counterfactual purity: Simulation queries have no side effects
Validated observations: All observations pass through Pydantic schemas

3. Computable Metrics

Parametric beliefs: Agents maintain probability distributions (not "vibes")
Surprisal: -log P(observation | belief) computed from proper likelihoods
Calibration: Brier scores and Expected Calibration Error
Learning rate: Surprisal trajectory slope (negative = learning)

4. Transparency

Every component is identifiable: Single file per concept
No abstraction layers: Direct LLM API calls, no frameworks
JSON logs: Every episode fully inspectable and replayable
Version control: All prompts versioned, no magic strings

Project Structure
world-model-experiment/
├── README.md                        # This file
├── requirements.txt                 # Dependencies
├── config.yaml                      # Model configs, budgets, seeds
├── preregistration.yaml             # Locked hypotheses (DO NOT MODIFY after experiments start)
├── .env                             # API keys (gitignored, create this)
├── .gitignore
│
├── environments/                    # Micro-world simulators
│   ├── base.py                      # Abstract Environment interface
│   ├── hot_pot.py                   # Hot-Pot Lab
│   ├── switch_light.py              # Switch-Light
│   ├── chem_tile.py                 # Chem-Tile
│   └── transfer_env.py              # Out-of-distribution test
│
├── agents/                          # Agent implementations
│   ├── base.py                      # Abstract Agent + LLM interfaces
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
├── evaluation/                      # Metrics and analysis
│   ├── metrics.py                   # All 7 metrics (interventional accuracy, etc)
│   ├── tasks.py                     # Test query sets
│   └── statistical.py               # Power analysis, t-tests, effect sizes
│
├── experiments/                     # Execution infrastructure
│   ├── runner.py                    # Main episode loop with guard rails
│   ├── provenance.py                # Git SHA tracking, code hashing
│   ├── prompts.py                   # All prompts (versioned)
│   ├── config.py                    # API key loading
│   └── ablations.py                 # Ablation configurations
│
├── scripts/                         # Entry points
│   ├── run_experiment.py            # Main: run all episodes
│   ├── analyze_results.py           # Generate report + figures
│   └── inspect_episode.py           # Debug single episode
│
├── results/                         # Generated outputs (gitignored)
│   ├── raw/                         # JSON logs per episode
│   ├── aggregated/                  # CSV summaries, figures
│   └── figures/                     # Plots
│
└── tests/                           # Test suite
    ├── conftest.py                  # Test configuration
    ├── test_environments.py         # Environment determinism, purity
    ├── test_agents.py               # Agent behavior
    ├── test_beliefs.py              # Likelihood computations
    ├── test_metrics.py              # Metric calculations
    └── test_integration.py          # Full pipeline
Quick Start
1. Install Dependencies
bashpip install -r requirements.txt
2. Configure API Keys
Create .env file in project root:
bashOPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
Important: .env is gitignored. Never commit API keys.
3. Run Tests
Verify environments and metrics work:
bash# Test environments (no API calls needed)
pytest tests/test_environments.py -v

# Test metrics
pytest tests/test_metrics.py -v

# Integration test (requires API keys)
pytest tests/test_integration.py -v -m integration
4. Run Experiment
bashpython scripts/run_experiment.py \
    --config config.yaml \
    --preregistration preregistration.yaml
This will:

Run all environment × agent combinations
Save episode logs to results/raw/TIMESTAMP/
Track full provenance (git SHA, code versions)
Enforce all guard rails

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
pythonfrom models.belief_state import HotPotBelief

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

surprisal = -log_prob  # Higher = more surprising
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
From preregistration.yaml:
H1 (Primary): Actor agents achieve ≥15% higher interventional accuracy than Observers

Rationale: Interactive experience should build better causal models

H2: Actor agents show negative surprisal slope; Observers show flat slope

Rationale: Learning manifests as decreasing surprisal

H3: Model-Based agents outperform pure Actor reasoning by ≥10%

Rationale: Explicit dynamics models enable better planning

H4: Interventional accuracy improves after high-surprisal events

Rationale: Tests "observe → update → improve" loop

H5: Actors show better out-of-distribution generalization than Observers

Rationale: Experience-based models should transfer better

Configuration
Model Selection (config.yaml)
yamlmodels:
  observer:
    provider: "openai"
    model: "gpt-4o-mini"
  actor:
    provider: "openai"
    model: "gpt-4o"
  model_based:
    provider: "anthropic"
    model: "claude-sonnet-4"
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

Common Issues
"API key not found"
Solution: Create .env file with your keys:
bashOPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
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
Batch 1: Core Infrastructure ✓

 Environments (hot_pot, switch_light, chem_tile)
 Parametric belief states
 Guard rails and provenance
 Environment tests

Batch 2: Agents (In Progress)

 Observer agent
 Actor agent with belief updates
 Text-Reader baseline
 Model-Based agent
 Transition model (MLP)
 Agent tests

Batch 3: Evaluation & Execution

 Test query sets
 All 7 metrics
 Experiment runner
 Analysis script
 Visualization
 Integration tests

Future Enhancements

 Transfer environment (different dynamics)
 Ablation studies (no memory, observation-only, etc)
 Alternative belief representations
 Advanced planning algorithms
 Multi-step rollouts