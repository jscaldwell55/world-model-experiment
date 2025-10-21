# Token-Level Prediction Bridge

**Comprehensive Guide to Token Prediction Experiments in LLM World Models**

---

## Table of Contents

1. [Overview](#overview)
2. [Research Question](#research-question)
3. [Quick Start](#quick-start)
4. [Architecture](#architecture)
5. [Configuration](#configuration)
6. [Expected Results](#expected-results)
7. [Output Files](#output-files)
8. [API Usage and Costs](#api-usage-and-costs)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)
11. [Citation](#citation)
12. [Next Steps](#next-steps)

---

## Overview

The **Token-Level Prediction Bridge** tests whether linguistic next-token prediction encodes similar uncertainty signals as grounded world-model prediction (belief surprisal). This system:

- **Textualizes** environment observations into deterministic natural language
- **Predicts** next observations using LLM token probabilities (logprobs)
- **Compares** token-level negative log-likelihood (NLL) with belief state surprisal
- **Analyzes** coupling strength, predictive validity, and calibration across environments

### Key Innovation

Traditional LLMs generate text autoregressively but don't explicitly model world dynamics. This system asks: **Do token prediction uncertainties (NLL) correlate with world-model uncertainties (belief surprisal)?** If yes, linguistic prediction may implicitly encode grounded prediction.

---

## Research Question

**H-Token (Primary):** Token NLL from next-sentence prediction correlates positively with belief surprisal from parametric world models.

**Sub-Hypotheses:**
- **H-Token1:** HotPot shows coupling r > 0.5 (deterministic physics)
- **H-Token2:** Actor agents show higher predictive validity than Observer agents
- **H-Token3:** Coupling strength: HotPot > SwitchLight > ChemTile (decreasing predictability)

---

## Quick Start

### 1. Validate Templates

Ensure textualization layers have no ground truth leakage and are deterministic:

```bash
python scripts/validate_templates.py
```

Expected output:
```
✓ All environments passed determinism test
✓ All environments passed no-leakage test
✓ All environments passed numerical precision test
```

### 2. Run Pilot Experiment

Run a small-scale pilot (30 episodes: 5 episodes × 3 environments × 2 agents):

```bash
export OPENAI_API_KEY='your-api-key'
python scripts/pilot_token_run.py --output-dir results/pilot_token
```

**Estimated time:** ~10-15 minutes
**Estimated cost:** ~$2-5 USD (using gpt-4o-mini)

### 3. Analyze Pilot Results

Compute coupling metrics and generate diagnostic plots:

```bash
python scripts/analyze_token_pilot.py results/pilot_token
```

Outputs:
- `coupling_metrics.csv` - Correlation coefficients by environment
- `coupling_scatter.png` - Scatter plots of token NLL vs belief surprisal
- `temporal_alignment.png` - Time-series of both metrics

### 4. Run Full Experiment (Optional)

Scale up to 300 episodes (50 × 3 × 2):

```bash
python scripts/run_full_token_experiment.py --output-dir results/full_token
```

**Estimated time:** ~2-3 hours
**Estimated cost:** ~$40-60 USD

### 5. Generate Publication-Ready Analysis

```bash
python scripts/analyze_full_token_results.py results/full_token
python scripts/generate_token_figures.py results/full_token
```

Outputs:
- All A1-A5 statistical analyses
- Hypothesis test results
- 4 publication-ready figures

---

## Architecture

### Data Flow

```
Environment Observation (dict)
         ↓
Textualization Layer (deterministic)
         ↓
Natural Language Text
         ↓
OpenAI API (ChatCompletion with logprobs)
         ↓
Token Probabilities → Token NLL
         ↓
Compare with Belief Surprisal
         ↓
Statistical Analysis (A1-A5)
```

### Components

#### 1. Textualization Layer (`textualization/`)

**Purpose:** Convert structured observations to natural language without leaking ground truth.

**Files:**
- `base.py` - Abstract interface with validation methods
- `hot_pot_text.py` - HotPot Lab (thermometer, stove, waiting)
- `switch_light_text.py` - Switch Light (switch flips, bulb observations)
- `chem_tile_text.py` - ChemTile (compound mixing, reactions)

**Key Properties:**
- **Deterministic:** Same observation → same text (always)
- **No Leakage:** Forbidden keys never appear in text (actual_temp, broken, reaction_probs)
- **Numerical Precision:** Consistent rounding (temp: 1 decimal, time: 0 decimals)

**Example:**
```python
# HotPot observation
obs = {'measured_temp': 23.567, 'time': 42.3, 'action': 'measure_temp'}

# Textualization
text = textualize_observation(obs)
# → "Thermometer reads 23.6°C. Time elapsed: 42 seconds."
```

#### 2. Token Prediction (`token_prediction/`)

**Purpose:** Use LLM logprobs to compute uncertainty over next observation.

**Files:**
- `predictor.py` - Abstract interfaces (NextSentencePredictor, TokenPrediction)
- `openai_predictor.py` - OpenAI ChatCompletion implementation
- `logger.py` - JSON logging (TokenLogger, TokenLogEntry)
- `metrics.py` - Token-level metrics (NLL, perplexity, calibration)

**API Call:**
```python
predictor = OpenAINextSentencePredictor(model='gpt-4o-mini')

context = """
Action taken: measure_temp
Thermometer reads 20.5°C. Time elapsed: 0 seconds.
Action taken: turn_on_stove
Stove is now ON.
Action taken: measure_temp
"""

prediction = predictor.predict_next_observation(context, temperature=0.0)

# prediction.sequence_nll = 3.45  (total -log P)
# prediction.per_token_nll = 0.38  (average per token)
# prediction.tokens = ['Therm', 'ometer', ' reads', ' 22', '.', '3', '°', 'C', ...]
# prediction.logprobs = [-0.01, -0.05, -0.02, -1.2, ...]
```

#### 3. Integration Layer (`experiments/`)

**Purpose:** Run token prediction in parallel with agent/environment loop.

**File:** `token_runner.py`

**Key Function:**
```python
def run_episode_with_tokens(
    env,
    agent,
    textualizer,
    predictor,
    seed,
    max_actions=10,
    save_dir=None
):
    """Run episode with parallel token prediction.

    Returns:
        test_results: Dict with episode metrics
        token_logger: TokenLogger with all predictions
    """
```

**Workflow:**
1. Agent observes environment
2. Agent selects action (using belief state)
3. Textualize action + build context
4. **Predict next observation** (token prediction)
5. Execute action in environment
6. Textualize true observation
7. Extract belief surprisal from agent's belief state
8. Log: context, prediction, true obs, token NLL, belief surprisal
9. Repeat until done

#### 4. Statistical Analysis (`evaluation/`)

**Purpose:** Test hypotheses about token-belief coupling.

**Files:**
- `token_analysis.py` - A1-A5 statistical analyses
- `token_validation.py` - Robustness tests (paraphrase, stopwords, ranking)

**Analyses:**

**A1: Coupling** - Pearson/Spearman correlation between token NLL and belief surprisal
```python
analysis = TokenAnalysis('results/full_token')
coupling = analysis.compute_coupling()
# → DataFrame with r, p-value, n_steps per environment
```

**A2: Surprise Detection** - Precision-Recall AUC for detecting high-surprisal events
```python
surprise = analysis.compute_surprise_detection(surprisal_threshold=2.0)
# → PR-AUC scores by environment
```

**A3: Predictive Validity** - Does low token NLL predict higher accuracy later?
```python
validity = analysis.compute_predictive_validity(lag=1)
# → Correlation between NLL(t) and Accuracy(t+1)
```

**A4: Calibration** - Brier score and Expected Calibration Error
```python
calibration = analysis.compute_token_calibration()
# → {'brier_score': 0.23, 'expected_calibration_error': 0.12, ...}
```

**A5: Family Factor** - Model-family × environment interactions (placeholder)

---

## Configuration

### File: `config_token.yaml`

```yaml
token_prediction:
  enabled: true

  predictors:
    # Observer agent predictor
    observer:
      model: "gpt-4o-mini"
      temperature: 0.0
      max_tokens: 100

    # Actor agent predictor
    actor:
      model: "gpt-4o"  # Can use different model per agent
      temperature: 0.0
      max_tokens: 100

# Full experiment settings
num_episodes_per_env: 50  # 50 × 3 envs × 2 agents = 300 total
environments:
  - hot_pot
  - switch_light
  - chem_tile
agents:
  - observer
  - actor
base_seed: 42
```

### Environment Variables

```bash
export OPENAI_API_KEY='sk-...'  # Required for OpenAI API
```

---

## Expected Results

### By Environment

#### **HotPot Lab** (Deterministic Physics)
- **Coupling (A1):** r = 0.6 - 0.8 (strong positive)
- **Surprise Detection (A2):** PR-AUC = 0.75 - 0.85
- **Mechanism:** Temperature follows deterministic heating curve. LLM learns physics → low NLL for expected temps, high NLL for anomalies

**Example:**
```
Step 3: measured_temp=22.3°C (expected ~22°C) → Token NLL=2.1, Surprisal=0.8
Step 5: measured_temp=45.2°C (expected ~24°C) → Token NLL=8.7, Surprisal=6.3
```

#### **SwitchLight** (Moderate Stochasticity)
- **Coupling (A1):** r = 0.4 - 0.6 (moderate positive)
- **Surprise Detection (A2):** PR-AUC = 0.65 - 0.75
- **Mechanism:** Some randomness in wire layout, but consistent patterns. LLM partially learns electrical rules.

#### **ChemTile** (High Stochasticity)
- **Coupling (A1):** r = 0.2 - 0.4 (weak positive)
- **Surprise Detection (A2):** PR-AUC = 0.55 - 0.65
- **Mechanism:** Reaction outcomes highly stochastic. LLM struggles to predict, leading to noisy correlation.

### By Agent Type

#### **Actor** (Has Belief State)
- **Predictive Validity (A3):** Higher correlation between NLL(t) and Accuracy(t+1)
- **Mechanism:** Actor's belief state provides better context → LLM predictions align better with future performance

#### **Observer** (No Belief State)
- **Predictive Validity (A3):** Lower correlation
- **Mechanism:** Observer has no internal model → LLM predictions less informative about future

---

## Output Files

### Pilot Experiment (`scripts/pilot_token_run.py`)

**Token Logs:** `results/pilot_token/`
```
HotPotLab_ObserverAgent_ep000_token.json
HotPotLab_ObserverAgent_ep001_token.json
...
ChemTile_ActorAgent_ep004_token.json
```

**Structure:**
```json
{
  "episode_id": "HotPotLab_ActorAgent_ep000",
  "entries": [
    {
      "step": 0,
      "context_text": "...",
      "true_observation": "Thermometer reads 23.6°C. Time elapsed: 42 seconds.",
      "predicted_text": "Thermometer reads 23.8°C. Time elapsed: 42 seconds.",
      "tokens": ["Therm", "ometer", " reads", ...],
      "logprobs": [-0.01, -0.05, -0.02, ...],
      "sequence_nll": 3.45,
      "per_token_nll": 0.38,
      "belief_surprisal": 1.2,
      "accuracy": 0.92
    },
    ...
  ]
}
```

### Pilot Analysis (`scripts/analyze_token_pilot.py`)

**Files:**
- `coupling_metrics.csv` - Correlation coefficients
```csv
environment,pearson_r,pearson_p,spearman_r,spearman_p,n_steps
HotPot,0.73,0.0001,0.68,0.0003,45
SwitchLight,0.52,0.0045,0.49,0.0067,38
ChemTile,0.31,0.0234,0.28,0.0456,42
```

- `coupling_scatter.png` - Scatter plots
- `temporal_alignment.png` - Time-series plots

### Full Experiment Analysis (`scripts/analyze_full_token_results.py`)

**Files in `results/full_token/`:**
- `analysis_report.txt` - Text summary
- `coupling_by_environment.csv` - A1 results
- `coupling_by_agent.csv` - A1 stratified by agent
- `surprise_detection.csv` - A2 results
- `predictive_validity.csv` - A3 results
- `calibration_metrics.json` - A4 results
- `hypothesis_tests.json` - H-Token1, H-Token2, H-Token3

**Example hypothesis_tests.json:**
```json
{
  "H-Token1": {
    "description": "HotPot coupling > 0.5",
    "result": "PASS",
    "statistic": "r=0.687, p=0.0000"
  },
  "H-Token2": {
    "description": "Actor > Observer predictive validity",
    "result": "PASS",
    "statistic": "Actor=0.42, Observer=0.28"
  },
  "H-Token3": {
    "description": "Coupling strength: HotPot > SwitchLight > ChemTile",
    "result": "See observed order",
    "observed_order": ["HotPot", "SwitchLight", "ChemTile"]
  }
}
```

### Figures (`scripts/generate_token_figures.py`)

**Files in `results/full_token/figures/`:**
- `figure1_coupling_scatter.png` - Token NLL vs Belief Surprisal (3 subplots)
- `figure2_coupling_heatmap.png` - Environment × Agent heatmap
- `figure3_temporal_alignment.png` - Time-series alignment
- `figure4_predictive_validity.png` - Box plots by environment/agent

---

## API Usage and Costs

### OpenAI API Pricing (as of 2024)

**gpt-4o-mini:**
- Input: $0.150 / 1M tokens
- Output: $0.600 / 1M tokens

**gpt-4o:**
- Input: $2.50 / 1M tokens
- Output: $10.00 / 1M tokens

### Estimated Usage

**Per Episode:**
- Context tokens: ~500-1000 (depends on episode length)
- Output tokens: ~50-100 (predicted observation)
- **Total per episode:** ~600-1100 tokens

**Pilot (30 episodes with gpt-4o-mini):**
- Total tokens: ~30,000
- **Cost:** ~$2-5 USD

**Full Experiment (300 episodes with gpt-4o-mini):**
- Total tokens: ~300,000
- **Cost:** ~$40-60 USD

**Full Experiment (mixed: observer=gpt-4o-mini, actor=gpt-4o):**
- 150 episodes × gpt-4o-mini: ~$20-30
- 150 episodes × gpt-4o: ~$150-200
- **Total Cost:** ~$170-230 USD

### Cost Reduction Strategies

1. **Use gpt-4o-mini for all agents** (recommended for initial experiments)
2. **Reduce num_episodes_per_env** from 50 to 20-30
3. **Limit max_tokens** to 50 instead of 100
4. **Cache repeated contexts** (not currently implemented)

---

## Troubleshooting

### Issue: "OPENAI_API_KEY environment variable not set"

**Solution:**
```bash
export OPENAI_API_KEY='sk-your-api-key-here'
```

Or add to `.bashrc` / `.zshrc`:
```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc
```

### Issue: Token logs contain NaN for belief_surprisal

**Cause:** Agent does not have a belief state (e.g., ObserverAgent)

**Expected Behavior:** ObserverAgent should have `belief_surprisal=None` for all steps. Only ActorAgent computes belief surprisal.

**Fix:** This is expected. Analysis functions automatically filter out NaN values:
```python
df_valid = self.df.dropna(subset=['token_nll', 'belief_surprisal'])
```

### Issue: Low or negative coupling coefficients

**Possible Causes:**
1. **Insufficient data:** Need at least 20-30 steps per environment
2. **API errors:** Check logs for failed predictions
3. **Environment-specific issues:** Some environments may have weak coupling (ChemTile expected to be low)

**Diagnostic:**
```bash
# Check number of valid data points
python -c "
from evaluation.token_analysis import TokenAnalysis
analysis = TokenAnalysis('results/pilot_token')
print(analysis.df.groupby('environment')['belief_surprisal'].describe())
"
```

### Issue: API rate limit errors

**Symptoms:** Errors like `RateLimitError: Rate limit exceeded`

**Solutions:**
1. Add retry logic (already implemented with exponential backoff)
2. Reduce parallelism (run fewer episodes concurrently)
3. Upgrade OpenAI account tier

### Issue: Validation tests fail

**Common failures:**
- **Determinism:** Check that textualization has no randomness
- **Leakage:** Ensure forbidden keys (actual_temp, broken, etc.) never appear in text
- **Numerical precision:** Verify consistent rounding (use Python's built-in `round()`)

**Debug:**
```bash
python scripts/validate_templates.py --verbose
```

---

## Advanced Usage

### Custom Textualizers

Create a new textualization layer for your environment:

```python
from textualization.base import TextualizationLayer

class MyEnvTextualization(TextualizationLayer):
    def __init__(self):
        super().__init__()
        self._forbidden_keys = {'ground_truth', 'hidden_state'}

    def textualize_observation(self, obs: dict) -> str:
        # Your deterministic template here
        return f"Sensor reading: {obs['sensor']:.2f}"

    def textualize_action(self, action: str) -> str:
        return f"Action taken: {action}"
```

Then register in `experiments/token_runner.py`:
```python
def create_textualizer(env):
    env_name = env.__class__.__name__
    if 'MyEnv' in env_name:
        return MyEnvTextualization()
    # ... existing logic
```

### Custom Predictors

Use a different LLM provider (e.g., Claude, Llama):

```python
from token_prediction.predictor import NextSentencePredictor, TokenPrediction

class ClaudePredictor(NextSentencePredictor):
    def predict_next_observation(self, context: str, **kwargs) -> TokenPrediction:
        # Call Claude API
        response = anthropic.completions.create(...)

        # Extract logprobs (if available)
        return TokenPrediction(
            predicted_text=response.text,
            tokens=...,
            logprobs=...,
            sequence_nll=...,
            per_token_nll=...
        )
```

### Validation Variants

Test robustness of token predictions:

```python
from evaluation.token_validation import ValidationVariants

validator = ValidationVariants(predictor, textualizer)

# Test 1: Paraphrase robustness
paraphrase_rules = [
    ("Thermometer reads", "Temperature is"),
    ("Time elapsed", "Time passed")
]
result = validator.test_paraphrase_robustness(obs, paraphrase_rules)

# Test 2: Candidate ranking
decoys = ["Thermometer reads 25.0°C", "Thermometer reads 30.0°C"]
ranking = validator.test_candidate_ranking(context, true_obs, decoys)

# Test 3: Action conditioning ablation
ablation = validator.test_action_conditioning(full_context, last_action)
```

---

## Citation

If you use this token prediction system in your research, please cite:

```bibtex
@software{token_prediction_bridge,
  title = {Token-Level Prediction Bridge for LLM World Models},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourname/world-model-experiment}
}
```

**Related Work:**

- **Language Models as World Models:** Hao et al. (2023)
- **Predictive Uncertainty in LLMs:** Kadavath et al. (2022)
- **Grounded Language Models:** Ahn et al. (2022)

---

## Next Steps

### Immediate (After Pilot)

1. ✓ Validate templates pass all tests
2. ✓ Run pilot experiment (30 episodes)
3. ✓ Check coupling metrics (expect r > 0.5 for HotPot)
4. Iterate on templates if coupling is weak
5. Run full experiment (300 episodes)

### Extended Research

1. **Model Families:** Test GPT-4 vs Claude vs Llama (A5 analysis)
2. **Context Length:** Vary context window size (full history vs last 3 steps)
3. **Temperature Ablation:** Test temperature=0 vs 0.5 vs 1.0
4. **Prompt Engineering:** Test different prompt formats (CoT, few-shot)
5. **Causal Analysis:** Does belief surprisal cause token NLL, or vice versa?

### Publication Targets

- **ACL/EMNLP:** Focus on linguistic aspects (textualization, token metrics)
- **NeurIPS/ICLR:** Focus on world modeling and uncertainty quantification
- **CoRL:** Focus on robotics applications (if extended to embodied agents)

---

## Appendix: File Manifest

**Core Textualization:**
- `textualization/__init__.py`
- `textualization/base.py`
- `textualization/hot_pot_text.py`
- `textualization/switch_light_text.py`
- `textualization/chem_tile_text.py`
- `textualization/validation.py`

**Token Prediction:**
- `token_prediction/__init__.py`
- `token_prediction/predictor.py`
- `token_prediction/openai_predictor.py`
- `token_prediction/logger.py`
- `token_prediction/metrics.py`

**Integration:**
- `experiments/token_runner.py`

**Scripts:**
- `scripts/validate_templates.py` - Template validation
- `scripts/pilot_token_run.py` - Pilot experiment (30 episodes)
- `scripts/analyze_token_pilot.py` - Pilot analysis
- `scripts/run_full_token_experiment.py` - Full experiment (300 episodes)
- `scripts/analyze_full_token_results.py` - Comprehensive analysis (A1-A5)
- `scripts/generate_token_figures.py` - Publication figures

**Evaluation:**
- `evaluation/token_analysis.py` - Statistical analyses (A1-A5)
- `evaluation/token_validation.py` - Robustness tests

**Tests:**
- `tests/test_textualization.py` - 16 textualization tests
- `tests/test_token_prediction.py` - 14 token prediction tests

**Configuration:**
- `config_token.yaml` - Token experiment configuration

**Documentation:**
- `TOKEN_EXPERIMENT_README.md` - This file
- `README.md` - Main project README (includes token section)

---

**Questions or Issues?**

Open an issue on GitHub or contact the maintainers.

**Happy Experimenting!** 🚀
