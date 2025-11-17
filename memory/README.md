# Domain-Specific Persistent Memory

This module implements domain-specific persistent memory for the Simple World Model agent, enabling learning across episodes without modifying the core algorithm.

## Overview

The memory system adds a persistent layer on top of the proven Simple World Model (81.1% accuracy). Each domain (ChemTile, HotPot, SwitchLight) maintains isolated memory that accumulates knowledge across episodes.

## Key Features

- **Domain Isolation**: Each environment has completely separate memory
- **Non-Invasive**: Core Bayesian algorithm remains unchanged
- **Adaptive Priors**: Prior strength increases with confidence (0.1 → 0.3 max)
- **Episode Learning**: Beliefs are saved after each episode and loaded at start
- **Performance Tracking**: Episode scores tracked to weight high-performing beliefs

## Directory Structure

```
memory/
├── domains/
│   ├── chem_tile/
│   │   ├── consolidated/
│   │   │   └── beliefs.json      # Aggregated beliefs across episodes
│   │   └── episodes/              # Individual episode data
│   ├── hot_pot/
│   │   ├── consolidated/
│   │   │   └── beliefs.json
│   │   └── episodes/
│   └── switch_light/
│       ├── consolidated/
│       │   └── beliefs.json
│       └── episodes/
├── domain_memory.py               # Core memory implementation
├── analyze_memory.py              # Analysis tool
└── README.md                      # This file
```

## Usage

### Running Experiments with Memory

```bash
# Run 10 episodes to test memory system
python scripts/run_experiment.py --config config_memory_test.yaml --output-dir results/memory_test

# Run 30 episodes to see performance improvement
python scripts/run_experiment.py --config config_memory_test.yaml --num-episodes 30 --output-dir results/memory_30ep
```

### Analyzing Memory

```bash
# Analyze all domains
python memory/analyze_memory.py

# Analyze specific domain
python memory/analyze_memory.py --domain hot_pot

# Custom memory path
python memory/analyze_memory.py --base-path /path/to/memory/domains
```

### Example Output

```
======================================================================
HOT POT DOMAIN MEMORY
======================================================================
Episodes completed: 15
Average score:      82.3%
Confidence level:   0.28
Prior strength:     0.28 (capped at 0.3)

Key Beliefs:
{
  "heating_rate_mean": 2.47,
  "heating_rate_std": 0.31,
  "measurement_noise": 0.95,
  "burn_threshold": 84.5
}

Episode History (15 episodes):
Episode                                      Score    Timestamp
----------------------------------------------------------------------
hot_pot_20251116_143022                      78.2%   2025-11-16 14:30
hot_pot_20251116_143145                      81.5%   2025-11-16 14:31
...
```

## How It Works

### 1. Episode Start (`start_episode`)

- Loads consolidated beliefs for the domain
- Initializes agent's belief state with prior knowledge
- Adjusts prior_strength based on confidence (caps at 0.3)

### 2. During Episode

- Agent operates normally using existing Bayesian updates
- No changes to core algorithm

### 3. Episode End (`end_episode`)

- Extracts key beliefs from final belief state
- Saves episode data with score
- Updates consolidated beliefs using weighted average:
  - High-scoring episodes have more influence
  - Confidence increases with more episodes (caps at 0.5)

### 4. Belief Consolidation

Consolidated beliefs are updated using:
```python
new_belief = (1 - weight) * old_belief + weight * new_belief
weight = episode_score / 100.0  # 0-1 range
```

Confidence increases with experience:
```python
confidence = min(0.5, 0.1 + (num_episodes * 0.02))
```

Prior strength for next episode:
```python
prior_strength = min(0.3, confidence)  # NEVER exceeds 0.3
```

## Critical Constraints

### DO NOT:

1. Change `prior_strength` default from 0.1
2. Allow `prior_strength` to exceed 0.3 (prevents 0.5 disaster)
3. Mix memories between domains
4. Modify the core Bayesian update logic

### DO:

1. Keep domain memories completely isolated
2. Save after every episode
3. Load priors only at episode start
4. Cap prior_strength at 0.3 maximum

## Expected Performance

With domain-specific memory across 30+ episodes:

| Domain      | Baseline | With Memory | Improvement |
|-------------|----------|-------------|-------------|
| ChemTile    | 90.9%    | 92-93%      | +1-2%       |
| HotPot      | 80.0%    | 82-83%      | +2-3%       |
| SwitchLight | 72.3%    | 74-75%      | +2%         |
| **Overall** | **81.1%**| **83-84%**  | **+2-3%**   |

Improvement comes from:
- Not relearning domain basics each episode
- Better initial priors after first few episodes
- Accumulated knowledge of edge cases
- Optimized exploration (know what to test)

## Files

### `domain_memory.py`

Core implementation:
- `DomainSpecificMemory`: Main class managing persistent memory
- `save_episode()`: Save episode beliefs and update consolidated beliefs
- `load_prior()`: Load consolidated beliefs for domain
- `get_prior_strength()`: Calculate adaptive prior strength

### `analyze_memory.py`

Analysis and monitoring tool:
- View consolidated beliefs per domain
- Track episode score trends
- Compare performance across domains
- Identify learning patterns

## Integration

The memory system integrates seamlessly:

1. **Agent**: `agents/simple_world_model.py`
   - Added `start_episode()` and `end_episode()` methods
   - Loads priors at episode start
   - Saves beliefs at episode end

2. **Runner**: `experiments/runner.py`
   - Calls `start_episode()` after agent reset
   - Calls `end_episode()` after evaluation

3. **No Changes Required**:
   - Core algorithm
   - Belief update logic
   - Evaluation system

## Success Criteria

Implementation succeeds if:

1. ✅ Memory files are created in correct structure
2. ✅ Prior beliefs are loaded at episode start
3. ✅ Performance improves by 2-3% after 20+ episodes
4. ✅ No degradation of baseline performance
5. ✅ Memories remain domain-isolated

## Troubleshooting

### No memory files created

Check that:
- `start_episode()` and `end_episode()` are called
- Agent has `SimpleWorldModel` type
- Experiment runner includes memory calls

### Memory not loading

Verify:
- Consolidated beliefs file exists
- Domain name mapping is correct
- Belief state attributes match saved beliefs

### Performance not improving

Consider:
- Need more episodes (>10) to see improvement
- Check if prior_strength is being applied
- Verify consolidated beliefs are sensible

## Future Enhancements

Potential improvements (not currently implemented):

1. **Cross-domain transfer**: Learn general physics across domains
2. **Forgetting mechanism**: Decay old beliefs over time
3. **Confidence intervals**: Track uncertainty in beliefs
4. **Active learning**: Prioritize episodes that maximize learning
5. **Meta-learning**: Learn optimal prior_strength per domain
