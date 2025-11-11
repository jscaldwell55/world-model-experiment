# Contributing to World Model Experiments

Thank you for your interest in contributing to this research project!

## Overview

This repository contains a completed preregistered study comparing Agentic Context Engineering (ACE) against traditional interactive learning approaches. The study was completed on 2025-10-31 with 506 episodes.

## Ways to Contribute

### 1. Reproduce the Study

The most valuable contribution is reproducing our results:

```bash
# Run the full study reproduction
python scripts/run_experiment_parallel.py \
  --config configs/config_full_study_3agents.yaml \
  --preregistration preregistration.md \
  --output-dir results/your_reproduction \
  --workers 4
```

**Please share your results:** If you reproduce the study, please open an issue with:
- Your reproduction commit SHA
- Summary statistics (accuracy, tokens, cost)
- Any deviations from our findings
- System information (OS, Python version, API provider details)

### 2. Report Issues

Found a bug or have a question? Please open an issue with:
- **Bug reports:** Include error messages, environment details, and reproduction steps
- **Questions:** Reference specific files/lines where possible
- **Suggestions:** Explain the motivation and expected benefit

### 3. Propose Extensions

We welcome proposals for extending this work:
- New environments (with different coupling characteristics)
- Alternative agent architectures
- Additional analysis methods
- Improved statistical tests

**Before implementing:** Open an issue to discuss your proposal first.

### 4. Improve Documentation

Help make this work more accessible:
- Fix typos or unclear explanations
- Add examples or clarifications
- Improve code comments
- Create visualizations

## Development Guidelines

### Code Style

- **Python:** Follow PEP 8 conventions
- **Type hints:** Use type annotations for function signatures
- **Docstrings:** Include docstrings for public functions/classes
- **Comments:** Explain "why" not "what"

### Testing

Before submitting changes:

```bash
# Run unit tests
pytest tests/

# Run integration test
python scripts/run_experiment_parallel.py --config configs/config.yaml --output-dir /tmp/test --workers 1

# Verify no regressions in metrics
python scripts/verify_metric_correctness.py
```

### Commit Messages

Use clear, descriptive commit messages:
- Start with a verb (Add, Fix, Update, Remove)
- Reference issues when applicable (#123)
- Keep first line under 72 characters

Good examples:
```
Add support for new environment configurations
Fix belief update bug in Actor agent (#42)
Update statistical analysis with Bonferroni correction
```

### Pull Requests

1. **Fork** the repository
2. **Create a branch** from `master`: `git checkout -b feature/your-feature`
3. **Make your changes** with clear commits
4. **Test thoroughly** - all tests must pass
5. **Update documentation** if needed
6. **Submit a pull request** with:
   - Clear description of changes
   - Motivation for the changes
   - Any breaking changes or dependencies
   - Test results

## Scientific Integrity

This project emphasizes scientific rigor:

### Preregistration

The study was preregistered at commit `cd41f0c`. Changes after preregistration must:
- Be documented in [CHANGELOG.md](CHANGELOG.md)
- Include justification for deviations
- Maintain transparency about what was/wasn't preregistered

### Reproducibility

All changes must maintain reproducibility:
- Use explicit random seeds
- Version all prompts in code
- Log provenance information
- Avoid hidden state or configuration

### Data Integrity

- Never modify raw episode logs in `results/`
- Keep separate analysis and data collection code
- Document all data processing steps
- Maintain audit trail for analysis decisions

## Code Organization

```
agents/          # Agent implementations
environments/    # Experimental environments
evaluation/      # Metrics, judging, and tasks
experiments/     # Experiment runner and configuration
models/          # Belief states and world models
scripts/         # Analysis and utility scripts
tests/           # Unit and integration tests
```

### Adding New Components

**New Agent:**
1. Inherit from `agents.base.Agent`
2. Implement `choose_action()`, `update_belief()`, `answer_question()`
3. Add tests in `tests/test_agents.py`
4. Document architecture in docstring

**New Environment:**
1. Inherit from `environments.base.Environment`
2. Implement `reset()`, `step()`, `render()`
3. Add test questions in `evaluation/tasks.py`
4. Add tests in `tests/test_environments.py`

**New Metric:**
1. Add computation to `evaluation/metrics.py`
2. Update `Metrics` dataclass
3. Add tests in `tests/test_metrics.py`
4. Document interpretation

## Questions?

- **Methodology:** See [preregistration.md](preregistration.md)
- **Implementation:** Read source code docstrings
- **Results:** See [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)
- **Contact:** jay.s.caldwell@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping advance our understanding of context engineering vs. interactive learning in LLM agents!
