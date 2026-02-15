# World Model Experiments: Persistent Learning with ACE Memory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Current Focus:** Preventing belief traps in persistent world model learning through methodology-aware memory systems.

This project explores how LLM-based agents can learn persistent world models across episodes without falling into "belief traps"—situations where early incorrect beliefs prevent learning correct knowledge later.

**Status:** ✅ ACE Memory System Implemented & Validated (2025-11-17)

---

## Quick Links

- **[ACE Implementation](memory/ace_playbook.py)** - Core memory system implementation
- **[Quick Validation Script](scripts/quick_validation.py)** - Fast local validation workflow
- **[Research Preregistration](preregistration.md)** - Study design and hypotheses
- **[Preregistration](preregistration.md)** - Original study hypotheses (commit `cd41f0c`)

---

## Current System: SimpleWorldModel + ACE Memory

### The Problem: Belief Traps in Persistent Learning

**Original System (Consolidation-based):**
```
Episode 1-2: Mixed power settings → Learn heating_rate = 1.0°C/s (wrong!)
            Score ≥75% → Gets consolidated ✅
            High confidence because scores are good

Episode 3:   Consistent HIGH power → Learn heating_rate = 2.5°C/s (correct!)
            ❌ REJECTED as outlier (z-score > 2.5)

Result: System stuck with wrong belief forever
```

**Why it happens:**
- Episode score (answer quality) ≠ methodology quality
- High-scoring episodes can have flawed data collection
- Outlier detection rejects correct observations that differ from consolidated beliefs

### The Solution: ACE Memory System

**ACE (Agentic Context Engineering) Playbook:**
```
Episode 1-2: Store with LOW reliability tag
            "Power toggle detected - mixed contexts (averaged data)"

Episode 3:   Store with HIGH reliability tag
            "Consistent power setting - reliable measurement"
            ✅ NOT rejected despite 2x difference!

Result: Agent sees both observations with methodology warnings
```

**Key Innovation:** Separate **score** (answer quality) from **reliability** (methodology quality)

### Architecture

```
┌─────────────────────────────────────────┐
│         EPISODE RUNTIME                 │
│ 1. ACE Playbook provides context        │
│ 2. SimpleWorldModel initializes         │
│    (prior_strength=0.1 - weak priors!)  │
│ 3. Real-time Bayesian updates           │
│ 4. Episode completes                     │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│      AFTER EPISODE: ACE LEARNS          │
│ 1. Reflector analyzes trajectory        │
│    - Detects methodology issues          │
│    - Tags reliability (HIGH/MEDIUM/LOW)  │
│ 2. Curator generates delta updates      │
│ 3. Playbook updated (NOT consolidated!) │
└─────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│   NEXT EPISODE: CONTEXT PROVIDED        │
│ Agent sees:                              │
│ ✓ HIGH reliability observations          │
│ ⚠️ LOW reliability observations          │
│ 💡 Methodology warnings                  │
└─────────────────────────────────────────┘
```

### Components

**1. SimpleWorldModel Agent** (`agents/simple_world_model.py`)
- Evolution of ACTOR with persistent memory
- Real-time Bayesian belief updates (prior_strength=0.1)
- Statistical tracking for noise filtering
- Causal relationship learning
- **Unchanged from ACTOR:** Core Bayesian inference

**2. ACE Playbook** (`memory/ace_playbook.py`)
- Stores observations with context and methodology tags
- Reflects on trajectories to assess reliability
- Curates delta updates (no consolidation!)
- Generates natural language context with warnings
- **Key:** Never rejects observations as outliers

**3. Methodology Detection**
- Detects power toggles (HotPot) → LOW reliability
- Detects limited exploration (ChemTile) → LOW reliability
- Detects systematic exploration (SwitchLight) → HIGH reliability
- Tags reliability independently from episode score

---

## Validation Results

### ✅ Controlled Belief Trap Test (Priority 1)

**Test Scenario:**
```bash
python scripts/quick_validation.py
```

**Results:**
```
Phase 1: Episodes 1-2 (MIXED power)
  → 1.2-1.4°C/s learned, tagged LOW reliability ✅

Phase 2: Episode 3 (HIGH power) - CRITICAL TEST
  → 2.5°C/s learned, tagged HIGH reliability ✅
  → NOT rejected as outlier ✅

Phase 3: Episode 4 (HIGH power)
  → 2.6°C/s learned, HIGH reliability ✅

Phase 4: Episode 5 (LOW power)
  → 0.5°C/s learned, HIGH reliability ✅

✅ All 5 observations stored (no rejection)
✅ Reliability correctly tagged in 100% of cases
✅ Core value proposition VALIDATED
```

**What this proves:**
- ACE prevents belief traps by storing all observations
- Methodology quality tagged separately from score
- Correct observations not rejected even when very different from prior beliefs

### ✅ 9-Episode Validation Test

**Configuration:** 3 episodes per domain (HotPot, ChemTile, SwitchLight)

**Results:**
```
Overall accuracy: 84.6%
  - ChemTile:    95.0% (excellent!)
  - HotPot:      79.7% (all episodes had power toggles)
  - SwitchLight: 79.2% (improved from 69% → 84% across episodes)

Methodology Detection:
  - HotPot:      3/3 correctly tagged LOW (power toggles)
  - ChemTile:    3/3 tagged LOW (limited exploration)
  - SwitchLight: 3/3 correctly tagged HIGH (systematic exploration)

Accuracy: 100% in methodology classification
```

**Key findings:**
1. ACE correctly detects methodology issues in real episodes
2. Performance competitive with baselines (84.6% overall)
3. Learning progression visible (SwitchLight: 69% → 84%)
4. No observations rejected despite methodology diversity

---

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

### Run Validation Test

```bash
# Quick local validation
python scripts/quick_validation.py

# 9-episode validation (~$1.50, 16 minutes)
python scripts/run_experiment_parallel.py \
  --config config_ace_validation_9ep.yaml \
  --output-dir results/ace_validation_9ep \
  --workers 1

# 30-episode comprehensive validation (~$5, 2 hours)
python scripts/run_experiment_parallel.py \
  --config config_ace_validation_30ep.yaml \
  --output-dir results/ace_validation_30ep \
  --workers 1
```

### Analyze Results

```bash
# Analyze learning progression
python analyze_ace_learning.py --results-dir results/ace_validation_9ep

# View ACE playbooks
cat memory/domains/hot_pot/playbook.json | jq '.observations[] | {episode_id, reliability, reason}'
cat memory/domains/chem_tile/playbook.json | jq '.observations'
cat memory/domains/switch_light/playbook.json | jq '.observations'
```

---

## Project Structure

```
world-model-experiment/
├── README.md                           # This file
├── preregistration.md                  # Study hypotheses and design
├── verify_rollback.py                  # Rollback validation script
├── analyze_ace_learning.py             # Results analysis
│
├── config_ace_validation_9ep.yaml      # 9-episode test config
├── config_ace_validation_30ep.yaml     # 30-episode test config
│
├── agents/
│   ├── simple_world_model.py           # World model agent with ACE
│   ├── actor.py                        # Original ACTOR (Bayesian)
│   ├── observer.py                     # Baseline (no learning)
│   └── ace.py                          # Original ACE agent
│
├── memory/
│   ├── ace_playbook.py                 # NEW: ACE memory system
│   └── domain_memory.py                # OLD: Consolidation-based (deprecated)
│
├── models/
│   └── belief_state.py                 # Belief representations
│
├── environments/
│   ├── hot_pot.py                      # Temperature dynamics
│   ├── switch_light.py                 # Wiring inference
│   └── chem_tile.py                    # Chemical reactions
│
├── experiments/
│   ├── runner.py                       # Episode orchestration
│   ├── prompts.py                      # LLM prompts
│   └── provenance.py                   # Version tracking
│
└── memory/domains/                     # ACE playbooks (gitignored)
    ├── hot_pot/
    │   ├── playbook.json               # Observations + methodology tags
    │   ├── episodes/*.json             # Raw episode data
    │   └── metadata/stats.json
    ├── chem_tile/
    └── switch_light/
```

---

## Key Metrics

### Belief Trap Prevention
- **Observation retention:** 100% (no rejections)
- **Methodology detection accuracy:** 100% in validation tests
- **Reliability tagging:** HIGH/MEDIUM/LOW based on data collection quality

### Performance
- **Overall accuracy:** 84.6% (9-episode test)
- **ChemTile:** 95.0% (range: 92-100%)
- **HotPot:** 79.7% (range: 73-83%)
- **SwitchLight:** 79.2% (range: 69-84%)

### Efficiency
- **Tokens per episode:** ~23k (input + output)
- **Cost per episode:** ~$0.17 (Claude Sonnet 4.5)
- **Time per episode:** ~2 minutes

---

## Comparison to Consolidation-Based Memory

| Aspect | Consolidation (Old) | ACE Memory (New) |
|--------|-------------------|------------------|
| **Storage** | Averaged beliefs | Individual observations |
| **Quality Control** | Outlier rejection | Methodology tagging |
| **Score vs Reliability** | Conflated | Separated |
| **Belief Traps** | ❌ Vulnerable | ✅ Prevented |
| **Data Loss** | ❌ Yes (outliers rejected) | ✅ No (all stored) |
| **Context Type** | Consolidated values | Natural language warnings |
| **Prior Strength** | Adaptive (0.1-0.3) | Fixed (0.1) |

**Critical Difference:** ACE stores observations with context instead of consolidating to single values. This prevents rejection of correct but different observations.

---

## Research Context

### Original Study (Completed 2025-10-31)

This project originated from a preregistered study comparing ACE vs Interactive Learning:

**Key Findings:**
- **ACTOR (Bayesian):** 81.2% accuracy
- **ACE (Playbook):** 70.3% accuracy
- **Critical insight:** Qualitative playbooks struggle with quantitative probability questions

**Study record:** See [preregistration.md](preregistration.md) and commit history.

### Current Development (2025-11-17)

Focus shifted to **persistent learning** and **belief trap prevention**:

**Problem identified:** Consolidation-based memory creates belief traps when:
1. Early episodes have good scores but flawed methodology
2. Later episodes have better methodology but different observations
3. Outlier detection rejects the correct observations

**Solution implemented:** ACE memory system with methodology tracking

---

## Technical Details

### Methodology Detection (HotPot Example)

```python
# LOW Reliability (power toggles)
Actions: ['measure_temp', 'toggle_power', 'measure_temp', 'toggle_power']
Context: {'power_setting': 'MIXED'}
Reliability: LOW
Reason: "Multiple power toggles (2) - averaged across contexts"

# HIGH Reliability (consistent power)
Actions: ['measure_temp', 'wait', 'measure_temp', 'wait']
Context: {'power_setting': 'HIGH'}
Reliability: HIGH
Reason: "Consistent power setting - reliable measurement"
```

### Context Generation

```
=== HotPotLab KNOWLEDGE BASE ===

✓ HIGH-RELIABILITY OBSERVATIONS:
  • Episode ep003 (score: 88%): heating_rate ~2.50°C/s [power: HIGH]
    → Consistent power setting - reliable measurement

⚠️ LOW-RELIABILITY OBSERVATIONS (USE WITH CAUTION):
  • Episode ep001 (score: 85%): heating_rate ~1.20°C/s [power: MIXED]
    → Power toggle detected - mixed contexts (averaged data)

💡 RECOMMENDATION:
  Initialize with WEAK priors (prior_strength=0.1)
  Trust current observations over past averages
  Pay attention to context (settings, actions taken)
```

### Prior Strength

**Critical parameter:** `prior_strength = 0.1` (fixed)

- Weak priors ensure agent adapts quickly to current observations
- ACE context provides guidance without strong constraints
- Prevents over-reliance on potentially unreliable historical data

---

## Future Work

### Immediate Next Steps
1. ✅ **Controlled belief trap test** - COMPLETE
2. 🔄 **30-episode validation** - IN PROGRESS
3. ⏸️ **Compare to consolidation baseline** - Planned
4. ⏸️ **Long-term learning (100+ episodes)** - Planned

### Research Directions
1. **Continuous reliability scores** (vs. HIGH/MEDIUM/LOW)
2. **Cross-domain transfer learning**
3. **Offline consolidation** (Dream → NeSy → Fine-tuning)
4. **Exploration strategy optimization**

### Open Questions
- How many episodes before HIGH reliability data emerges naturally?
- Can ACE be extended to other learning domains?
- What is optimal playbook size (currently capped at 10 observations)?
- How to balance context length vs. information density?

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{caldwell2025worldmodel,
  title={Preventing Belief Traps in Persistent World Model Learning},
  author={Caldwell, Jay},
  year={2025},
  howpublished={\url{https://github.com/jaycald/world-model-experiment}},
  note={ACE-based memory system for methodology-aware learning}
}
```

For the original ACE vs Interactive Learning study:

```bibtex
@misc{caldwell2025ace,
  title={World Model Experiments: ACE vs Interactive Learning},
  author={Caldwell, Jay},
  year={2025},
  howpublished={\url{https://github.com/jaycald/world-model-experiment}},
  note={Preregistered study comparing context engineering vs. Bayesian learning}
}
```

---

## References

### ACE Framework
- **ACE Paper:** ["Agentic Context Engineering"](https://arxiv.org/abs/2510.04618) (2024)
- **Implementation:** Original ACE agent vs. new ACE memory system

### Theoretical Background
- **Belief trap problem:** High-scoring but flawed methodology prevents learning
- **Methodology tracking:** Separate data quality from task performance
- **Context vs. consolidation:** Natural language warnings vs. averaged values

---

## Contact

**Jay Caldwell**
Independent Researcher
jay.s.caldwell@gmail.com

For questions:
- **ACE Memory Implementation:** `memory/ace_playbook.py`
- **Validation Scripts:** `scripts/quick_validation.py`
- **Original Study:** [preregistration.md](preregistration.md)

---

## License

MIT License - See [LICENSE](LICENSE) file

Copyright (c) 2025 Jay Caldwell

---

**Last updated:** 2025-11-17 | ACE Memory System validated | Belief trap prevention confirmed
