# Faithful ACE Implementation

This document describes the implementation of a faithful Agentic Context Engineering (ACE) system based on the ACE paper.

## Summary of Changes

The previous ACE implementation was missing several key mechanisms described in the paper. This update implements all core ACE features to enable proper comparison with the Observer and Actor baselines.

## Key Mechanisms Implemented

### 1. ✅ Bullet Feedback Loop

**What it does:** Tracks which playbook bullets are actually helpful or harmful based on episode outcomes.

**Implementation:**
- Each bullet now has `helpful_count` and `harmful_count` fields
- When an episode succeeds, all referenced bullets get `helpful_count += 1`
- When an episode fails, all referenced bullets get `harmful_count += 1`
- Utility score = `helpful_count - harmful_count`

**Files modified:**
- `agents/ace.py`: Added `_update_bullet_feedback()` method
- `agents/ace.py`: Modified `update_playbook()` to call feedback updates

**Code location:**
- agents/ace.py:621-647 (feedback update method)

### 2. ✅ Top-K Retrieval

**What it does:** Instead of showing the entire playbook to the Generator, retrieve only the most relevant bullets for each step.

**Implementation:**
- Created `utils/embeddings.py` with `BulletRetriever` class
- Uses sentence-transformers (or TF-IDF fallback) for semantic similarity
- Retrieves top-k most relevant bullets per section based on current observation + history
- Default k=5 bullets per section

**Files created:**
- `utils/embeddings.py`: New embedding and retrieval utilities

**Files modified:**
- `agents/ace.py`: Added `use_retrieval` and `top_k` parameters
- `agents/ace.py`: Modified `_choose_action()` to use retrieval
- `agents/ace.py`: Updated `_format_playbook()` to support filtered playbooks

**Code location:**
- utils/embeddings.py:103-167 (retrieval implementation)
- agents/ace.py:268-279 (retrieval usage in Generator)

### 3. ✅ Grow-and-Refine (Dedup + Pruning)

**What it does:** Removes redundant bullets and prunes low-utility items when playbook grows too large.

**Implementation:**
- **Deduplication:** Uses SequenceMatcher to detect >80% similar bullets, merges their feedback counts
- **Pruning:** When token cap is reached, keeps only highest-utility bullets (sorted by helpful - harmful)

**Files modified:**
- `agents/ace.py`: Implemented `_deduplicate_playbook()` (previously a TODO)
- `agents/ace.py`: Implemented `_prune_playbook_to_cap()` for token budget enforcement

**Code location:**
- agents/ace.py:653-695 (deduplication)
- agents/ace.py:697-732 (pruning)

### 4. ✅ Multi-Round Reflection

**What it does:** Allows the Reflector to iterate and refine insights before passing to Curator.

**Implementation:**
- Added `reflection_rounds` parameter (default=1, set to 2+ for faithful ACE)
- Reflector can see its previous round's insights and refine them
- Each round strengthens the quality of extracted insights

**Files modified:**
- `agents/ace.py`: Added `_reflect_on_episode_multi_round()` method
- `agents/ace.py`: Modified `_reflect_on_episode()` to accept `prior_insights`

**Code location:**
- agents/ace.py:321-349 (multi-round reflection)
- agents/ace.py:351-412 (reflection with prior insights)

### 5. ✅ Same-LLM Settings Across Roles

**What it does:** Ensures fair comparison by using the same model and temperature for Generator, Reflector, and Curator.

**Implementation:**
- Added separate temperature parameters: `generator_temperature`, `reflector_temperature`, `curator_temperature`
- Default all to 0.7 (as recommended in paper for fairness)
- Previous implementation used temperature=0.8 for Generator, 0.0 for Reflector/Curator

**Files modified:**
- `agents/ace.py`: Added temperature parameters to `__init__`
- `agents/ace.py`: Updated LLM calls to use role-specific temperatures

**Code location:**
- agents/ace.py:58-60 (temperature parameters)
- agents/ace.py:304 (Generator temperature)
- agents/ace.py:398 (Reflector temperature)
- agents/ace.py:472 (Curator temperature)

## Configuration

### Example: Faithful ACE Config

```yaml
# config_ace_faithful_test.yaml
ace_config:
  use_retrieval: true              # Enable top-k retrieval
  top_k: 5                          # Retrieve 5 most relevant bullets per section
  reflection_rounds: 2              # Use 2 reflection iterations per episode
  generator_temperature: 0.7        # Same temperature across all roles
  reflector_temperature: 0.7
  curator_temperature: 0.7
  curation_mode: "curated"          # Use LLM-based curation
  token_cap: null                   # No token cap (or set to 1000/2000/etc)
```

### Integration with Experiment Runner

The experiment runner (`experiments/runner.py`) has been updated to automatically pass ACE config parameters to the agent:

**Code location:**
- experiments/runner.py:121-139 (ACE config injection)

## Testing

A comprehensive test suite verifies all mechanisms:

```bash
python test_faithful_ace.py
```

**Test coverage:**
- ✅ ACE initialization with all parameters
- ✅ Playbook structure with feedback counts
- ✅ Retrieval mechanism with top-k filtering
- ✅ Deduplication with feedback merging
- ✅ Pruning with utility scoring

**Files created:**
- `test_faithful_ace.py`: Comprehensive test suite

## Usage

### Running a Faithful ACE Experiment

```bash
python scripts/run_experiment_parallel.py \
  --config config_ace_faithful_test.yaml \
  --output-dir results/ace_faithful \
  --workers 2
```

### Comparing Faithful ACE vs. Original Implementation

To compare:
1. **Faithful ACE:** Set `use_retrieval=true`, `reflection_rounds=2`, uniform temperatures
2. **Original ACE:** Set `use_retrieval=false`, `reflection_rounds=1`, default temperatures

## Expected Improvements

With these mechanisms active, ACE should now:

1. **Learn selectively:** Bullets with positive utility get prioritized
2. **Scale efficiently:** Top-k retrieval prevents context explosion
3. **Refine insights:** Multi-round reflection produces higher-quality strategies
4. **Stay focused:** Deduplication and pruning keep the playbook lean and relevant

## Migration Notes

**Backward compatibility:**
- All new parameters have sensible defaults
- Existing configs will work unchanged (uses default ACE behavior)
- To enable faithful ACE, add `ace_config` section to your YAML

**Breaking changes:**
- None - all changes are additive

## Next Steps

1. **Run confirmatory study:** Compare faithful ACE vs. Observer vs. Actor
2. **Tune hyperparameters:** Test different values of `top_k`, `reflection_rounds`, `token_cap`
3. **Multi-epoch experiments:** Test repeated sampling on same seeds for stronger learning
4. **Embedding upgrades:** Consider upgrading from TF-IDF to sentence-transformers for better retrieval

## Files Changed Summary

### Created Files
- `utils/embeddings.py` - Embedding and retrieval utilities
- `test_faithful_ace.py` - Test suite
- `config_ace_faithful_test.yaml` - Example faithful ACE config
- `FAITHFUL_ACE_IMPLEMENTATION.md` - This documentation

### Modified Files
- `agents/ace.py` - Core ACE agent with all mechanisms
- `experiments/runner.py` - Config injection for ACE parameters

## References

The implementation follows the Agentic Context Engineering paper's description of:
- Generator → Reflector → Curator → Playbook loop
- Bullet-level feedback and utility scoring
- Retrieval-augmented context for efficient scaling
- Iterative reflection for insight refinement
- Fair evaluation with consistent LLM settings
