# Theoretical Framework: LLMs as Implicit World Modelers

## 1. Conceptual Foundations

### 1.1 Linguistic Surprisal (Hale 2001, Levy 2008)

**Definition**: Linguistic surprisal quantifies the predictability of a word given its context:

```
Surprisal(word_t) = -log P(word_t | word_1...word_{t-1})
```

**Empirical Support**:
- Correl with reading time in eye-tracking studies (Rayner et al., 2004)
- Predicts N400 ERP amplitude in neurolinguistic experiments (Frank et al., 2015)
- Robust across languages and text domains (Smith & Levy, 2013)

**Key Insight**: The human language processing system maintains probabilistic expectations about upcoming words. Higher surprisal indicates violated expectations.

### 1.2 Belief Surprisal / Prediction Error (Friston 2010, Clark 2013)

**Definition**: In predictive processing frameworks, belief surprisal measures the unexpectedness of sensory observations relative to an internal generative model:

```
Belief_Surprisal(observation_t) = -log P(observation_t | belief_state)
```

**Theoretical Context**:
- **Free Energy Principle** (Friston, 2010): Biological systems minimize prediction error to maintain homeostasis
- **Predictive Coding** (Rao & Ballard, 1999): Neural circuits encode predictions and propagate errors
- **Active Inference** (Friston et al., 2012): Actions selected to reduce expected surprisal

**Applications**:
- Computational neuroscience (modeling sensory processing)
- Cognitive science (explaining perception, attention, learning)
- Robotics (model-based control and exploration)

### 1.3 Bridging Linguistic and Grounded Surprisal

**Research Question**: Do large language models, trained exclusively on next-token prediction, implicitly learn to model the dynamics of the physical/causal world described in text?

**Operationalization**:
1. **Linguistic Surprisal**: Token-level NLL from LLM predictions of textualized observations
2. **Grounded Belief Surprisal**: Probabilistic surprise from explicit world model (e.g., Gaussian belief over heating rates)
3. **Coupling Hypothesis**: If LLMs encode world models, linguistic and grounded surprisal should be correlated

**Theoretical Significance**:
- Tests **Sutskever's claim**: "Predicting text is indistinguishable from predicting reality" (if text is accurately describing reality)
- Challenges **Sutton's critique**: LLMs are "shallow pattern matchers" lacking genuine causal understanding
- Informs **scaling laws**: Does model capacity improve world modeling, or only surface statistics?

## 2. Research Question Formulation

### Central Question
**Do LLMs trained on next-token prediction implicitly implement predictive processing over world states?**

### Hypotheses

**H-Token1: Coupling Hypothesis**
> Linguistic surprisal (token NLL) and grounded belief surprisal (from explicit world model) are positively correlated across environments.

- **Strong support**: r > 0.6 (Pearson correlation)
- **Moderate support**: 0.3 < r < 0.6
- **Weak/No support**: r < 0.3

**H-Token2: Environment Gradient**
> Coupling strength varies systematically across environments based on determinism and observability:

- **Deterministic, fully observable** (HotPot): Highest coupling (easiest to model)
- **Stochastic, partially observable** (SwitchLight): Moderate coupling
- **Complex, multi-factor** (ChemTile): Lower coupling (harder to model from text)

**H-Token3: Semantic Validation**
> Negative controls (shuffled/random text) show significantly lower coupling than normal textualization.

- **Prediction**: r_normal > 0.5, r_shuffled < 0.2, r_random < 0.2
- **Implication**: High coupling in controls would indicate spurious correlation

**H-Token4: Agent Hierarchy**
> Coupling strength reflects quality of belief state:

- **Model-based** (explicit dynamics): Highest coupling (gold standard)
- **Actor** (parametric belief): Moderate coupling
- **Observer** (no belief): Lowest coupling

## 3. Expected Outcomes & Interpretations

### Decision Matrix

|                          | **Strong Coupling** (r>0.6) | **Weak Coupling** (r<0.3) |
|--------------------------|----------------------------|---------------------------|
| **Deterministic Env**<br>(HotPot) | **Interpretation 1**:<br>LLMs learn implicit world models from text<br>→ Support Sutskever view<br>→ "Predicting text ≈ predicting reality" | **Interpretation 2**:<br>LLMs fail even on simple physics<br>→ Support Sutton critique<br>→ Shallow pattern matching |
| **Stochastic Env**<br>(ChemTile) | **Interpretation 3** (**Surprising!**):<br>LLMs robust to uncertainty<br>→ Strong world modeling<br>→ Exceeds expectations | **Interpretation 4** (**Expected**):<br>LLMs limited to linguistic patterns<br>→ Cannot handle true uncertainty<br>→ Confirms limitations |

### Specific Predictions

**If Coupling is Strong (r > 0.6)**:
1. LLMs implicitly encode causal/physical dynamics from training data
2. Token prediction serves as lossy compression of world state transitions
3. Scaling to larger models should improve coupling further
4. **Practical implication**: LLMs can be used for approximate world modeling without explicit simulation

**If Coupling is Weak (r < 0.3)**:
1. LLMs rely primarily on surface-level statistical patterns
2. Text prediction does not require (or learn) genuine causal models
3. Explicit world models remain necessary for reliable prediction
4. **Practical implication**: Need hybrid architectures (LLM + explicit simulator)

**If Coupling is Moderate (0.3 < r < 0.6)**:
1. LLMs capture some aspects of world dynamics, but incompletely
2. May depend on training data coverage of specific domains
3. **Critical test**: Does fine-tuning on domain text improve coupling?

## 4. Key References

### Linguistic Surprisal & Psycholinguistics
```bibtex
@article{hale2001probabilistic,
  title={A probabilistic Earley parser as a psycholinguistic model},
  author={Hale, John},
  journal={Proceedings of NAACL},
  year={2001}
}

@article{levy2008expectation,
  title={Expectation-based syntactic comprehension},
  author={Levy, Roger},
  journal={Cognition},
  volume={106},
  number={3},
  pages={1126--1177},
  year={2008}
}

@article{frank2015erp,
  title={The ERP response to the amount of information conveyed by words in sentences},
  author={Frank, Stefan L and Otten, Leun J and Galli, Giulia and Vigliocco, Gabriella},
  journal={Brain and Language},
  volume={140},
  pages={1--11},
  year={2015}
}
```

### Predictive Processing & Free Energy
```bibtex
@article{friston2010free,
  title={The free-energy principle: a unified brain theory?},
  author={Friston, Karl},
  journal={Nature Reviews Neuroscience},
  volume={11},
  number={2},
  pages={127--138},
  year={2010}
}

@article{clark2013whatever,
  title={Whatever next? Predictive brains, situated agents, and the future of cognitive science},
  author={Clark, Andy},
  journal={Behavioral and Brain Sciences},
  volume={36},
  number={3},
  pages={181--204},
  year={2013}
}

@article{rao1999predictive,
  title={Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects},
  author={Rao, Rajesh PN and Ballard, Dana H},
  journal={Nature Neuroscience},
  volume={2},
  number={1},
  pages={79--87},
  year={1999}
}
```

### World Models & AI
```bibtex
@article{lake2017building,
  title={Building machines that learn and think like people},
  author={Lake, Brenden M and Ullman, Tomer D and Tenenbaum, Joshua B and Gershman, Samuel J},
  journal={Behavioral and Brain Sciences},
  volume={40},
  year={2017}
}

@book{sutton2018reinforcement,
  title={Reinforcement Learning: An Introduction},
  author={Sutton, Richard S and Barto, Andrew G},
  year={2018},
  publisher={MIT Press},
  edition={2nd},
  note={See Chapter 8: Planning and Learning with Tabular Methods}
}

@article{ha2018world,
  title={World models},
  author={Ha, David and Schmidhuber, J{\"u}rgen},
  journal={arXiv preprint arXiv:1803.10122},
  year={2018}
}
```

### LLMs & Emergent Capabilities
```bibtex
@article{wei2022emergent,
  title={Emergent abilities of large language models},
  author={Wei, Jason and Tay, Yi and Bommasani, Rishi and others},
  journal={Transactions on Machine Learning Research},
  year={2022}
}

@article{bubeck2023sparks,
  title={Sparks of artificial general intelligence: Early experiments with GPT-4},
  author={Bubeck, S{\'e}bastien and Chandrasekaran, Varun and Eldan, Ronen and others},
  journal={arXiv preprint arXiv:2303.12712},
  year={2023}
}
```

## 5. Methodological Safeguards

### 5.1 Negative Controls
To rule out spurious coupling:
- **Shuffled textualization**: Breaks semantics while preserving vocabulary
- **Random substitution**: Replaces observations with unrelated valid observations
- **Expected result**: r_control < 0.2 if coupling is genuinely semantic

### 5.2 Multiple Metrics
Beyond Pearson correlation:
- **Spearman rank correlation**: Robust to outliers, detects monotonic relationships
- **Mutual information**: Captures nonlinear dependencies
- **Distance correlation**: Detects all types of dependence (dcor library)
- **Regression diagnostics**: Test for nonlinearity via polynomial features

### 5.3 Agent Hierarchy
Compare:
- **Model-based** (explicit dynamics): Upper bound on achievable coupling
- **Actor** (parametric belief): LLM-based world model
- **Observer** (no belief): Lower bound (minimal world modeling)

This hierarchy validates that coupling reflects world modeling quality, not artifacts.

## 6. Potential Confounds & Mitigations

| **Confound** | **Description** | **Mitigation** |
|--------------|-----------------|----------------|
| Textualization artifacts | Specific templates bias LLM predictions | Test with alternative phrasings |
| Model family effects | GPT-4 vs Claude may differ | Run with multiple models (future work) |
| Training data memorization | LLM saw similar scenarios in training | Use novel environment parameterizations |
| Measurement noise | Noisy observations inflate surprisal | Control for observation uncertainty |
| Sample size | Small N inflates correlation estimates | Bootstrap confidence intervals |

## 7. Broader Impact

**If LLMs encode world models**:
- Advances understanding of what neural language models learn
- Suggests text corpora contain rich implicit knowledge of physical/causal dynamics
- Opens path to extracting world models from pre-trained LLMs
- Informs debate on AI alignment and interpretability

**If LLMs do not encode world models**:
- Clarifies limitations of pure language modeling
- Motivates hybrid architectures (language + simulation)
- Guides expectations for LLM capabilities in physical reasoning tasks
