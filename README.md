# World Model Experiments: ACE vs Interactive Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Central Research Question:** Can comprehensive, evolved context substitute for interactive experience in LLM agents?

This project compares **Agentic Context Engineering (ACE)** against traditional interactive learning approaches to understand when rich context can replace expensive interaction in causal reasoning tasks.

**Status:** ✅ Study Complete (2025-10-31) | 506 episodes | Statistically significant findings

---

## Quick Links

- **[Full Results](RESULTS_SUMMARY.md)** - Complete analysis and findings
- **[Preregistration](preregistration.md)** - Locked hypotheses (commit `cd41f0c`)
- **[Quick Start](QUICK_START.md)** - Copy-paste reproduction guide
- **[Changelog](CHANGELOG.md)** - All deviations documented

---

## Key Findings

| Agent | Accuracy | Tokens/Episode | Cost/Episode |
|-------|----------|----------------|--------------|
| **ACTOR** | **81.2%** | 19,289 | $0.12 |
| **ACE** | 70.3% | 20,692 | $0.13 |
| **OBSERVER** | 69.4% | 6,381 | $0.04 |

**Critical Discovery:** ACE's qualitative playbooks excel at strategic intervention planning but struggle with quantitative probability questions that ACTOR's explicit belief states handle naturally. ACE showed an 87.7% performance gap on planning questions requiring probability estimates.

**Implication:** Context engineering has architectural boundaries—qualitative strategies cannot fully substitute for quantitative representations in probabilistic reasoning tasks.

---

## Scientific Approach

### Preregistration

This study was preregistered prior to data collection:
- **Version:** v1.3 (locked at commit `cd41f0c`)
- **Date:** 2025-10-29
- **Git tags:** `prereg-v1.0`, `prereg-v1.1`, `prereg-v1.2`, `prereg-v1.3`
- **Deviations:** All documented in [CHANGELOG.md](CHANGELOG.md)

### Primary Hypotheses

- **H1a (Accuracy):** ACE achieves ≥70% accuracy on causal reasoning tasks
- **H1b (Cost):** ACE uses ≤70% of Actor's token cost
- **H-Budget:** Diminishing returns for larger playbook capacities
- **H-Curation:** Curated playbook outperforms append-only by ≥5 points
- **H-Shift:** ACE recovers from distribution shifts within 10 episodes

### Study Design

- **Sample size:** n=506 successful episodes (603 attempted, 83.9% completion rate)
- **Environments:** ChemTile, HotPotLab, SwitchLight
- **Agents:** Observer (baseline), Actor (Bayesian updates), ACE (context evolution)
- **Power:** 80% to detect effect size d≥0.65 at α=0.05

### Provenance

Every episode log contains:
- Git SHA (code version)
- Config hash (experiment settings)
- Prompt version identifier
- Timestamp and random seed
- Full trajectory and state information

---

## Architecture Comparison

### Observer (Baseline)
- **Method:** Language-only reasoning, no interaction
- **Belief:** None (passive inference)
- **Learning:** None
- **Expected:** 60-70% accuracy, ~$0.08/episode

### Actor
- **Method:** Interactive experimentation with Bayesian updates
- **Belief:** Parametric probability distributions
- **Learning:** Bayesian inference from observations
- **Expected:** 75-80% accuracy, ~$0.18/episode

### ACE (Agentic Context Engineering)
- **Method:** Interactive experimentation with context evolution
- **Belief:** Structured playbook of qualitative strategies
- **Learning:** Reflection → Curation → Playbook updates
- **Expected:** 70-75% accuracy, ~$0.14/episode

**ACE Architecture:**
```
Episode → Reflector → Curator → Playbook
  ↓                                 ↓
Generator ←──────────────────────────
```

**Based on:** ["Agentic Context Engineering"](https://arxiv.org/abs/2510.04618) (2024)

---

## Experimental Environments

### Hot-Pot Lab
**Challenge:** Deceptive labels require intervention to discover true dynamics
- Pot on stove with potentially misleading temperature labels
- Must measure actual temperature to verify observations
- Tests interventional and counterfactual reasoning

### Switch-Light
**Challenge:** Distinguish causation from correlation
- 2 switches, 2 lights, unknown wiring configuration
- Must intervene to determine causal structure
- Tests structural inference

### Chem-Tile
**Challenge:** Compositional reasoning with safety constraints
- Grid of chemical tiles with reaction dynamics
- Combining chemicals triggers various reactions
- Tests compositional inference and safety reasoning

---

## Reproduction Instructions

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys (both required)
export ANTHROPIC_API_KEY="sk-ant-api03-..."  # For agents (Claude Sonnet 4.5)
export OPENAI_API_KEY="sk-proj-..."          # For judge (GPT-4)
```

### Reproduce Full Study

```bash
# Run full study (506+ episodes, ~$60-80, 3-4 hours with 4 workers)
python scripts/run_experiment_parallel.py \
  --config configs/config_full_study_3agents.yaml \
  --preregistration preregistration.md \
  --output-dir results/reproduction \
  --workers 4

# Analyze results
python scripts/analyze_full_study.py results/reproduction
```

**Note:** Results should be qualitatively similar but may vary slightly due to LLM stochasticity. The original study used commit `cd41f0c`.

### Quick Test

```bash
# Single episode test (5 min, ~$0.20)
python scripts/run_experiment_parallel.py \
  --config configs/config.yaml \
  --output-dir results/test \
  --workers 1
```

---

## Project Structure

```
world-model-experiment/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── RESULTS_SUMMARY.md           # Complete analysis
├── QUICK_START.md               # Reproduction guide
├── CHANGELOG.md                 # Deviation tracking
├── preregistration.md           # Locked hypotheses
├── requirements.txt             # Dependencies
│
├── configs/                     # Experiment configurations
│   ├── config.yaml
│   └── config_full_study_3agents.yaml
│
├── agents/                      # Agent implementations
│   ├── observer.py              # Passive reasoning
│   ├── actor.py                 # Bayesian updates
│   └── ace.py                   # Context evolution
│
├── environments/                # Experimental environments
│   ├── hot_pot.py
│   ├── switch_light.py
│   └── chem_tile.py
│
├── evaluation/                  # Metrics and judging
│   ├── tasks.py                 # Test questions
│   ├── judge.py                 # Vendor-disjoint evaluation
│   └── metrics.py               # Metric computation
│
├── experiments/                 # Experiment framework
│   ├── runner.py                # Episode execution
│   ├── prompts.py               # Versioned prompts
│   └── provenance.py            # Version tracking
│
├── scripts/                     # Analysis and utilities
│   ├── run_experiment_parallel.py
│   ├── analyze_full_study.py
│   └── analyze_with_statistics.py
│
├── archive/                     # Historical artifacts
│   ├── configs/                 # Old configurations
│   ├── analysis/                # Pilot analysis scripts
│   └── docs/                    # Development documentation
│
└── results/                     # Experimental results (gitignored)
    └── full_study_final/        # Completed study data
```

---

## Metrics

### Standard Metrics
- **Accuracy:** Overall, interventional, counterfactual, by difficulty
- **Efficiency:** Tokens per episode, tokens per % accuracy
- **Calibration:** Brier score, Expected Calibration Error (ECE)

### ACE-Specific Metrics
- **Playbook Growth:** Total items over time, convergence analysis
- **Playbook Utilization:** Helpful vs. harmful item tracking
- **Context Efficiency:** Accuracy per playbook item

---

## Statistical Analysis

Full statistical analysis includes:
- Paired t-tests between all agent pairs
- Bootstrap 95% confidence intervals (10,000 resamples)
- Cohen's d effect sizes
- Bonferroni correction for multiple comparisons
- Power analysis (80% power for d≥0.65)

```bash
python scripts/analyze_with_statistics.py results/full_study_final
```

**Outputs:**
- `statistical_ttests.csv` - Pairwise comparisons
- `statistical_confidence_intervals.csv` - Bootstrap CIs
- `statistical_raw_data.csv` - Full dataset

---

## Extensions and Future Work

### Potential Extensions
1. **Hybrid Architectures:** Combine ACE's qualitative playbooks with quantitative belief states
2. **Additional Environments:** Test domains with varying quantitative/qualitative requirements
3. **Scale Study:** Increase sample size, test different LLM models
4. **Prompt Engineering:** Optimize ACE prompts for probability questions

### Open Questions
- Can ACE be augmented to maintain quantitative summaries alongside qualitative playbooks?
- Do other context engineering approaches face similar limitations?
- What is the optimal allocation between qualitative and quantitative representations?

---

## References

### Primary Work
- **ACE Framework:** ["Agentic Context Engineering"](https://arxiv.org/abs/2510.04618) (2024)
- **Theoretical Motivation:** ["Reflections on Richard Sutton's Interview"](https://yuanxue.github.io/2025/10/06/reflection-sutton-part1.html)

### Theoretical Background
- Sutskever: "Rich context ≈ world model"
- Sutton: "Need interaction for true understanding"
- This work: Tests when each perspective is correct

---

## Citation

If you use this work in your research, please cite:

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

## Contact

**Jay Caldwell**
Independent Researcher
jay.s.caldwell@gmail.com

For questions about methodology, implementation, or collaboration:
- Study design: [preregistration.md](preregistration.md)
- Implementation: `agents/ace.py`, `agents/actor.py`
- Results: [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)
- Episode logs: `results/full_study_final/raw/*.json`

---

## License

MIT License - See [LICENSE](LICENSE) file

Copyright (c) 2025 Jay Caldwell

---

**Last updated:** 2025-10-31 | Study complete | 506 episodes | Statistically significant findings
