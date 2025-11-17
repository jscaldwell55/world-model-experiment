# Hybrid Agent Implementation

## Overview

The **HybridAgent** combines two existing agent architectures to leverage their complementary strengths:

- **ACE (Agentic Context Engineering)**: Generates qualitative strategies based on an evolved "playbook" of learned strategies
- **ACTOR**: Evaluates strategies quantitatively using probabilistic belief states

## Architecture

### Decision-Making Process

For each action decision, the HybridAgent follows these steps:

1. **Strategy Generation (ACE)**: Generate multiple candidate strategies using ACE's playbook
   - Uses temperature sampling to create diverse candidates (default: 5 candidates)
   - Each candidate includes reasoning (thought) and action

2. **Strategy Scoring (ACTOR)**: Score each candidate based on ACTOR's belief state
   - Evaluates expected information gain
   - Assesses probability of success
   - Considers risk based on current beliefs

3. **Strategy Selection**: Select the highest-scoring strategy
   - Logs decision metadata for debugging and analysis
   - Combines ACE's contextual reasoning with ACTOR's probabilistic evaluation

4. **Learning**: Both sub-agents learn independently
   - ACE updates its playbook after each episode
   - ACTOR updates its belief state after each observation

### Query Answering

When answering test queries, the HybridAgent:
- Gets answers from both ACE and ACTOR
- Uses the answer from the more confident agent
- Logs both answers for analysis

## Implementation Details

### File Structure

- **agents/hybrid_agent.py**: Main HybridAgent implementation
- **config_hybrid_test.yaml**: Test configuration for quick testing
- **test_hybrid_integration.py**: Integration test script

### Key Classes and Methods

#### HybridAgent

**Constructor Parameters:**
- `llm`: LLM interface (shared by both sub-agents)
- `action_budget`: Maximum actions per episode
- `environment_name`: Environment name for tool selection
- `num_candidates`: Number of strategies to generate (default: 5)
- `candidate_temperature`: Temperature for ACE generation (default: 0.9)
- `scoring_temperature`: Temperature for ACTOR scoring (default: 0.3)
- ACE-specific parameters: `use_retrieval`, `top_k`, `reflection_rounds`, etc.

**Key Methods:**
- `act(observation)`: Main decision-making method
- `answer_query(question)`: Answer test queries
- `update_playbook(outcome)`: Update ACE's playbook after episode
- `update_belief_from_observation(obs)`: Update ACTOR's belief state
- `get_token_breakdown()`: Get detailed token usage statistics

### Decision Metadata

Each decision includes metadata for analysis:
```python
{
    'strategy': 'hybrid',  # or 'fallback_to_ace' / 'fallback_to_actor'
    'num_candidates': 5,
    'selected_idx': 2,  # Which candidate was chosen
    'selected_score': 0.85,
    'all_scores': [0.7, 0.6, 0.85, 0.5, 0.8],
    'scoring_reasoning': "Strategy 3 scores highest because...",
    'candidates_summary': [
        {
            'action': 'measure_temp()',
            'score': 0.7,
            'thought': '...'
        },
        ...
    ]
}
```

## Configuration

### Basic Configuration

Create a YAML config file (see `config_hybrid_test.yaml`):

```yaml
models:
  hybrid:
    model: "gpt-4o-mini"  # or "claude-sonnet-4-5-20250929"

budgets:
  actions_per_episode: 10

# Hybrid-specific settings
hybrid_config:
  num_candidates: 5
  candidate_temperature: 0.9
  scoring_temperature: 0.3

# ACE configuration (for ACE sub-agent)
ace_config:
  use_retrieval: true
  top_k: 5
  reflection_rounds: 1
  curation_mode: "curated"

# Which agents to run
agents_to_run:
  - hybrid

# Environments to test
environments:
  hot_pot:
    num_episodes: 10
    seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
```

### Configuration Parameters

#### Hybrid-Specific
- **num_candidates** (default: 5): Number of candidate strategies ACE generates
- **candidate_temperature** (default: 0.9): Higher = more diverse strategies
- **scoring_temperature** (default: 0.3): Lower = more consistent scoring

#### ACE Sub-Agent
- **use_retrieval** (default: true): Use top-k retrieval from playbook
- **top_k** (default: 5): Number of bullets to retrieve per section
- **reflection_rounds** (default: 1): Number of reflection iterations
- **curation_mode**: "curated", "no_curate", "random", or "greedy"
- **token_cap**: Maximum playbook tokens (null = unlimited)

## Usage

### Running Experiments

**Quick test (2 episodes):**
```bash
python scripts/run_experiment.py --config config_hybrid_test.yaml
```

**Full experiment:**
```bash
python scripts/run_experiment.py --config config_hybrid_full.yaml --num-episodes 10
```

**Parallel execution:**
```bash
python scripts/run_experiment_parallel.py \
  --config config_hybrid.yaml \
  --workers 4 \
  --output-dir results/hybrid_study
```

### Integration Testing

Run the integration test to verify the implementation:
```bash
python test_hybrid_integration.py
```

This tests:
- Agent instantiation
- Action selection with hybrid decision-making
- Belief state updates
- Query answering
- Token accounting

## Analysis

### Accessing Decision Logs

Episode logs include hybrid-specific metadata:

```python
import json

# Load episode log
with open('results/.../episode.json') as f:
    episode = json.load(f)

# Access hybrid decisions
for step in episode['steps']:
    if 'hybrid_decision' in step['belief_state']:
        decision = step['belief_state']['hybrid_decision']
        print(f"Candidates: {decision['num_candidates']}")
        print(f"Selected: {decision['selected_idx']}")
        print(f"Scores: {decision['all_scores']}")
```

### Token Breakdown

The hybrid agent provides detailed token accounting:

```python
{
    'ace': {
        'by_category': {
            'exploration': {'input_tokens': 1000, 'output_tokens': 500},
            'planning': {'input_tokens': 2000, 'output_tokens': 1000},
            ...
        }
    },
    'actor': {
        'by_category': {
            'exploration': {'input_tokens': 800, 'output_tokens': 400},
            'curation': {'input_tokens': 500, 'output_tokens': 200},
            ...
        }
    },
    'hybrid_specific': {
        'by_category': {
            'planning': {'input_tokens': 500, 'output_tokens': 100}
        }
    },
    'total_by_category': {
        'exploration': {'input_tokens': 1800, 'output_tokens': 900},
        'planning': {'input_tokens': 2500, 'output_tokens': 1100},
        ...
    }
}
```

## Expected Performance

The hybrid agent should:
- Generate diverse strategies from ACE's learned playbook
- Select strategies based on ACTOR's probabilistic reasoning
- Perform between ACE and ACTOR baselines initially
- Potentially outperform both as playbook evolves and beliefs improve
- Show which sub-agent contributed what to each decision

## Troubleshooting

### Common Issues

**1. Import errors**
```bash
# Ensure you're in the project root
cd /Users/jaycaldwell/world-model-experiment
# Test imports
python -c "from agents.hybrid_agent import HybridAgent; print('OK')"
```

**2. Missing configuration**
- Ensure config has `models.hybrid` section
- Add `hybrid_config` for hybrid-specific parameters
- Add `ace_config` for ACE sub-agent parameters

**3. Token accounting validation fails**
- This can happen due to rounding differences
- Check the `token_breakdown` in episode logs for details
- Small discrepancies (<10 tokens) are acceptable

**4. Fallback to single agent**
- Check logs for "fallback_to_ace" or "fallback_to_actor" messages
- This happens when candidate generation or scoring fails
- Verify LLM is working correctly with test script

### Debugging

Enable verbose logging by checking decision metadata in each step:

```python
for step in episode['steps']:
    belief = step['belief_state']
    if 'hybrid_decision' in belief:
        print(f"Step {step['step_num']}:")
        print(f"  Strategy: {belief['hybrid_decision']['strategy']}")
        print(f"  Candidates: {belief['hybrid_decision']['candidates_summary']}")
```

## Future Enhancements

Potential improvements:
1. **Adaptive candidate count**: Vary number based on uncertainty
2. **Weighted combination**: Blend ACE and ACTOR strategies instead of selecting one
3. **Meta-learning**: Learn when to trust ACE vs ACTOR
4. **Parallel scoring**: Score candidates in parallel for efficiency
5. **Strategy caching**: Cache and reuse similar strategies

## References

- ACE implementation: `agents/ace.py`
- ACTOR implementation: `agents/actor.py`
- Base agent interface: `agents/base.py`
- Experiment runner: `experiments/runner.py`
