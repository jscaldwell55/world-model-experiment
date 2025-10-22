# LLM-Generated Prior Beliefs

## Overview

Actor agents in this experiment now generate their own prior beliefs using the LLM, based on initial observations from the environment. This replaces hard-coded priors with adaptive, observation-informed initial uncertainty estimates.

**Key Innovation:** The agent sets its own starting point for belief-state learning by reasoning about what the initial observation suggests about the environment's dynamics.

## Motivation

Previously, all Actor agents started with the same hard-coded prior parameters regardless of what they observed:

```python
# OLD: Hard-coded priors
belief = HotPotBelief(
    heating_rate_mean=1.5,   # Fixed value
    heating_rate_std=0.3,    # Fixed uncertainty
    measurement_noise=2.0    # Fixed noise estimate
)
```

This was problematic because:
1. **Ignored initial information**: The agent saw `{'label': 'Boiling!', 'stove_light': 'on'}` but didn't use it
2. **Unrealistic**: Real agents adjust initial beliefs based on context
3. **Missed opportunity**: Initial observations often contain valuable signals (even if misleading)

With LLM-generated priors:
```python
# NEW: LLM generates priors from observation
initial_obs = {'label': 'Boiling!', 'stove_light': 'on', 'time': 0.0}
agent.reset(environment_type='HotPotLab', initial_observation=initial_obs)

# LLM generates (example):
# {
#   'heating_rate_mean': 1.2,     # Label suggests heating
#   'heating_rate_std': 0.8,      # High uncertainty (labels may deceive)
#   'measurement_noise': 1.5,
#   'reasoning': 'Label says boiling and light is on, suggesting active heating.
#                 However, maintaining moderate uncertainty as labels could be misleading.'
# }
```

## Architecture

### Component Overview

```
Initial Observation → LLM Prior Generation → Validation → Belief Initialization
                            ↓
                    Logged in Episode Metadata
```

**Files Modified:**
- `experiments/prompts.py` - Prior generation prompt templates (v1.1.0)
- `agents/actor.py` - `_generate_priors()`, `_validate_priors()`, updated `reset()`
- `experiments/runner.py` - Passes environment type and observation to `agent.reset()`
- `tests/test_agents.py` - Test suite for prior generation (9 tests)

### Prior Generation Flow

1. **Episode Initialization** (`runner.py:113-116`)
   ```python
   agent.reset(
       environment_type='HotPotLab',
       initial_observation={'label': 'Boiling!', 'stove_light': 'on', 'time': 0.0}
   )
   ```

2. **Prior Generation** (`actor.py:433-533`)
   - Select environment-specific prompt template
   - Format prompt with initial observation (as JSON)
   - Query LLM with temperature=0.0 (for consistency)
   - Parse JSON response
   - Extract reasoning and priors

3. **Validation** (`actor.py:335-431`)
   - Check all required parameters present
   - Verify values are within reasonable ranges
   - Ensure probability distributions are valid

4. **Fallback on Failure**
   - If parsing fails: Retry with modified prompt
   - If retry fails: Use uninformative priors (high uncertainty)
   - Always log warnings and reasoning for failures

5. **Logging** (`runner.py:208-214`)
   ```python
   episode_log['prior_generation'] = {
       'initial_priors': {...},
       'prior_reasoning': "...",
       'prior_generation_tokens': 150
   }
   ```

## Prompt Design

### Design Principles

1. **Explicit Parameter Descriptions**: Each parameter clearly explained with ranges
2. **Encourage Uncertainty**: Prompts emphasize expressing genuine doubt
3. **Deterministic Output**: Temperature=0.0 ensures reproducibility
4. **JSON-Only Response**: No markdown, no explanations outside JSON
5. **Skepticism Built-In**: Warn that labels/observations may be misleading

### Example: HotPot Prior Generation

```python
HOTPOT_PRIOR_GENERATION_TEMPLATE = """You are initializing your beliefs about a laboratory environment based on the initial observation.

Initial Observation:
{initial_observation}

Environment Type: Hot-Pot Lab

Your task is to set initial beliefs (priors) about the environment's dynamics based on this observation.
You need to specify parameters for:

1. heating_rate_mean: Expected temperature change per second (°C/s)
   - Range: [-5.0, 5.0]
   - Positive = heating, Negative = cooling, Zero = stable
   - Consider: What does the observation suggest about the heating state?

2. heating_rate_std: Your uncertainty about the heating rate (°C/s)
   - Range: [0.1, 10.0]
   - Higher = more uncertain about the heating dynamics
   - Consider: How confident are you? Could labels be misleading?

3. measurement_noise: Expected noise in temperature measurements (°C)
   - Range: [0.1, 5.0]
   - Typical thermometers: 0.5-2.0°C
   - Consider: How precise do you expect measurements to be?

IMPORTANT GUIDELINES:
- Express genuine uncertainty - labels may be misleading
- Base your priors on what you observe, but maintain skepticism
- If the observation suggests heating but you're unsure, use a moderate mean with higher std

Respond with ONLY a JSON object in this exact format (no markdown, no explanations):
{"heating_rate_mean": <float>, "heating_rate_std": <float>, "measurement_noise": <float>, "reasoning": "<1-2 sentence explanation>"}

Your prior belief (JSON only):"""
```

**Key Features:**
- Explicit ranges prevent out-of-bounds values
- Multiple "Consider:" questions guide reasoning
- Warning about misleading labels (critical for HotPot deception)
- JSON-only format for reliable parsing

## Validation System

### Validation Ranges

**HotPot Lab:**
- `heating_rate_mean`: [-5.0, 5.0] °C/s
- `heating_rate_std`: [0.1, 10.0] °C/s
- `measurement_noise`: [0.1, 5.0] °C

**Switch-Light:**
- `connection_probs`: 2×2 matrix, all values in [0.0, 1.0]
- `uncertainty`: [0.0, 1.0]

**ChemTile:**
- `reaction_safety_priors`: dict with values in [0.0, 1.0]
- `reaction_outcome_uncertainty`: [0.0, 1.0]
- `temperature_effect_prior`: [0.0, 1.0]

### Error Handling

**Strategy:** Graceful degradation with full transparency

```python
try:
    # Attempt 1: Generate priors
    priors, reasoning, tokens = agent._generate_priors(obs, env_type)
except ParseError:
    # Attempt 2: Retry with modified prompt
    prompt += "\n\nPREVIOUS ATTEMPT FAILED: {error}\nPlease ensure valid JSON..."
    priors, reasoning, tokens = agent._generate_priors(obs, env_type)
except ValidationError:
    # Fallback: Use uninformative priors
    priors = {
        'heating_rate_mean': 0.0,   # No prior knowledge
        'heating_rate_std': 5.0,    # Maximum uncertainty
        'measurement_noise': 2.0
    }
    reasoning = f"Failed to generate priors: {error}. Using uninformative defaults."
    tokens = 0
```

**Logging:**
- All failures are logged with full error messages
- Fallback priors are clearly marked
- Token count is 0 for fallback (indicating no successful LLM call)

## Episode Metadata

### Logged Information

Every episode log now includes (when using Actor agents):

```json
{
  "episode_id": "hot_pot_actor_ep001",
  "environment": "HotPotLab",
  "agent_type": "actor",
  "prior_generation": {
    "initial_priors": {
      "heating_rate_mean": 1.2,
      "heating_rate_std": 0.8,
      "measurement_noise": 1.5
    },
    "prior_reasoning": "Label says boiling and light is on, suggesting active heating. However, maintaining moderate uncertainty as labels could be misleading.",
    "prior_generation_tokens": 147
  },
  "steps": [...],
  "test_results": [...]
}
```

### Token Budget Tracking

Prior generation adds ~100-200 tokens per episode:
- Input: ~150 tokens (prompt + observation)
- Output: ~50 tokens (JSON response)
- **Total cost**: ~$0.0003-0.0005 per episode (using Claude Sonnet 4.5)

This is negligible compared to episode action costs (typically 10 actions × 500 tokens = 5000 tokens).

## Reproducibility

### Determinism Guarantee

**Same observation + same seed → same priors**

```python
# Test
obs = {'label': 'Boiling!', 'stove_light': 'on', 'time': 0.0}

agent1 = ActorAgent(llm, action_budget=10, environment_name='HotPotLab')
agent1.reset(environment_type='HotPotLab', initial_observation=obs)
priors1 = agent1.belief_state.model_dump()

agent2 = ActorAgent(llm, action_budget=10, environment_name='HotPotLab')
agent2.reset(environment_type='HotPotLab', initial_observation=obs)
priors2 = agent2.belief_state.model_dump()

assert priors1 == priors2  # With temperature=0.0
```

**Why deterministic?**
- Temperature set to 0.0 in all prior generation calls
- No random seeds or sampling
- LLM should return identical output for identical input

**Caveats:**
- LLM API updates may change outputs over time
- Prompts are versioned (v1.1.0) and logged in provenance
- Episode logs contain exact priors used, ensuring full reproducibility

## Expected Behavior

### HotPot Lab (Misleading Labels)

**Initial Observation:**
```json
{'label': 'Boiling!', 'stove_light': 'on', 'time': 0.0}
```

**Expected Prior Generation:**
- `heating_rate_mean`: 0.5-2.0 (suggests heating but uncertain)
- `heating_rate_std`: 0.5-1.5 (moderate-high uncertainty due to potential deception)
- `measurement_noise`: 1.0-2.0 (typical thermometer)
- `reasoning`: Mentions label suggests heating but expresses skepticism

**After First Measurement** (`measure_temp() → 23°C`):
- Agent updates beliefs via Bayesian inference
- `heating_rate_mean`: shifts toward 0.0 (no heating detected)
- `heating_rate_std`: decreases (more certain)
- **This demonstrates learning!**

### Switch-Light (Ambiguous Wiring)

**Initial Observation:**
```json
{'switch_positions': [0, 0], 'light_states': [0, 0], 'time': 0.0}
```

**Expected Prior Generation:**
- `connection_probs`: [[0.5, 0.5], [0.5, 0.5]] (uniform - can't determine yet)
- `uncertainty`: 0.7-0.9 (high - no information yet)
- `reasoning`: No switch activations yet, uniform priors

### ChemTile (Unknown Compounds)

**Initial Observation:**
```json
{'available_compounds': ['A', 'B', 'C'], 'temperature': 'medium', 'time': 0.0}
```

**Expected Prior Generation:**
- `reaction_safety_priors`: {'A': 0.5, 'B': 0.5, 'C': 0.5} (cautious defaults)
- `reaction_outcome_uncertainty`: 0.6-0.8 (chemistry is unpredictable)
- `temperature_effect_prior`: 0.5-0.7 (typical chemistry)
- `reasoning`: No information about specific compounds, using cautious priors

## Testing

### Test Coverage

**9 tests in `tests/test_agents.py::TestPriorGeneration`:**

1. `test_validate_priors_hotpot_valid` - Accepts valid HotPot priors
2. `test_validate_priors_hotpot_invalid_ranges` - Rejects out-of-range values
3. `test_validate_priors_switchlight_valid` - Accepts valid SwitchLight priors
4. `test_validate_priors_switchlight_invalid_matrix` - Rejects malformed matrices
5. `test_validate_priors_chemtile_valid` - Accepts valid ChemTile priors
6. `test_generate_priors_mock_llm` - Fallback behavior with mock LLM
7. `test_reset_with_prior_generation` - Reset initializes with LLM priors
8. `test_reset_without_prior_generation` - Reset without params preserves belief
9. `test_prior_generation_metadata_structure` - Metadata has correct structure

**All tests passing** ✅

### Running Tests

```bash
# Test prior generation only
pytest tests/test_agents.py::TestPriorGeneration -v

# All actor tests
pytest tests/test_agents.py::TestActorAgent -v

# Full test suite
pytest tests/test_agents.py -v
```

## Comparison: Before vs After

### Before (Hard-Coded Priors)

```python
# agents/actor.py (old)
class ActorAgent(Agent):
    def reset(self):
        super().reset()
        # Belief state persists with original hard-coded values
```

**Problems:**
- All episodes start identically
- Initial observations ignored
- No adaptation to context
- Unrealistic agent behavior

### After (LLM-Generated Priors)

```python
# agents/actor.py (new)
class ActorAgent(Agent):
    def reset(self, environment_type=None, initial_observation=None):
        super().reset()

        if environment_type and initial_observation:
            # Generate priors from LLM based on observation
            priors, reasoning, tokens = self._generate_priors(
                initial_observation, environment_type
            )

            # Initialize belief with generated priors
            self.belief_state = BeliefClass(**priors)

            # Log for transparency
            self.prior_generation_metadata = {
                'priors': priors,
                'reasoning': reasoning,
                'token_count': tokens
            }
```

**Benefits:**
- Context-aware initialization
- Transparent reasoning
- Fully logged and reproducible
- More realistic agent behavior

## Future Enhancements

### Potential Improvements

1. **Multi-Agent Consensus**
   - Generate priors from multiple LLM calls
   - Use ensemble mean/variance
   - May improve robustness

2. **Prior Calibration Analysis**
   - Track how well initial priors predict final beliefs
   - Identify systematic over/under-confidence
   - Adjust prompts based on calibration data

3. **Environment-Specific Tuning**
   - Custom prompt templates per environment
   - Domain-specific guidance
   - Learned prompt optimization

4. **Adaptive Temperature**
   - Use higher temperature for genuinely ambiguous observations
   - Lower temperature when observation is clear
   - May improve prior quality

## Summary

**What Changed:**
- Actor agents now generate their own prior beliefs using LLM
- Initial observations inform starting uncertainty estimates
- Full transparency via logged priors, reasoning, and token usage

**Why It Matters:**
- More realistic agent behavior
- Better utilizes available information
- Maintains scientific rigor (logged, validated, reproducible)

**Cost:**
- Minimal (~$0.0005 per episode)
- ~150 tokens per episode initialization

**Testing:**
- 9 dedicated tests, all passing
- Validation ensures safe, bounded priors
- Graceful fallback on failure

**Documentation:**
- Prompts versioned (v1.1.0)
- Episode logs contain all prior generation metadata
- Full provenance maintained

---

**See Also:**
- `experiments/prompts.py` - Prior generation templates
- `agents/actor.py` - Implementation (lines 335-533)
- `tests/test_agents.py` - Test suite
- `README.md` - Project overview
