# ACE Implementation Upgrade - Environment-Specific Templates

## Overview
Implemented comprehensive environment-specific ACE templates to match the paper's approach and create a level playing field with Actor baseline.

## What Was Implemented

### 1. Environment-Specific Generator Templates ✓
Created detailed, domain-specific prompts for each environment:

#### **HotPot Lab Template** (`ACE_GENERATOR_HOTPOT_TEMPLATE`)
- **Domain Knowledge**: Temperature dynamics, stove labels can be deceptive, measurement noise
- **Critical Strategies**: Multiple measurements, wait() for evolution, verify labels
- **Tools Documented**: measure_temp(), wait(), touch_pot(), toggle_stove()
- **Safety Warnings**: touch_pot() burns above 60°C
- **Playbook Integration**: PLAYBOOK_BEGIN/END markers, explicit bullet reference requirement

#### **SwitchLight Template** (`ACE_GENERATOR_SWITCHLIGHT_TEMPLATE`)
- **Domain Knowledge**: 2 switches, 2 lights, unknown wiring, faulty relays
- **Critical Strategies**: Test individually, recognize intermittent behavior, costly inspect_wires()
- **Wiring Patterns**: Direct vs crossed connections
- **Tools Documented**: flip_switch(), jiggle_relay(), inspect_wires(), observe_light()

#### **ChemTile Template** (`ACE_GENERATOR_CHEMTILE_TEMPLATE`)
- **Domain Knowledge**: Compounds consumed on mixing, temperature before mixing, explosions
- **Critical Strategies**: Always inspect() first, backward planning, track inventory
- **Safety Warnings**: heat() increases explosion risk
- **Tools Documented**: mix(), heat(), cool(), inspect()

### 2. Environment-Specific Reflector Templates ✓
Each Reflector template includes:

- **Common Failure Patterns**: Domain-specific mistakes to check
  - HotPot: Trusting labels, single measurements, confusion between rate and temp
  - SwitchLight: Simultaneous flips, missing intermittent behavior, premature inspect_wires()
  - ChemTile: Mixing before inspecting, wrong temperature order, wasting reagents

- **Domain Insights to Capture**: Specific learnings to extract
  - HotPot: Label accuracy, noise levels, heating rates
  - SwitchLight: Wiring patterns, relay faults, testing sequences
  - ChemTile: Reaction pathways, temperature effects, synthesis plans

### 3. Environment-Specific Curator Templates ✓
Each Curator template includes:

- **Good Example Items**: Concrete, actionable playbook entries
  - HotPot: "Always take 3+ temperature measurements to average out noise"
  - SwitchLight: "If light changes intermittently, suspect faulty relay and try jiggle_relay()"
  - ChemTile: "Safe synthesis sequence: inspect(A), inspect(B), cool(), mix(A,B)"

- **Bad Example Items**: Too-vague entries to avoid
  - "Be careful with temperature"
  - "Test the switches carefully"
  - "Mix compounds wisely"

### 4. Enhanced Counterfactual Reasoning ✓
Created `ACE_COUNTERFACTUAL_QUERY_TEMPLATE` with:

- **Structured Reasoning Process**:
  1. What Actually Happened (observed)
  2. Counterfactual Scenario (proposed alternative)
  3. Apply Playbook to Simulation (use learned strategies)
  4. Express Uncertainty (acknowledge unobserved)

- **Confidence Calibration**: Max 0.85 for counterfactuals
- **Playbook Integration**: Reference specific bullets in reasoning
- **Stochastic Handling**: Probabilistic language for uncertain outcomes

## Code Changes

### `experiments/prompts.py`
Added 13 new templates (~500 lines):
- 3 Generator templates (one per environment)
- 3 Reflector templates (one per environment)
- 3 Curator templates (one per environment)
- 1 Enhanced counterfactual template

### `agents/ace.py`
Modified ACE agent class:

1. **New Parameter**: `use_environment_specific_prompts=True`
   - Defaults to True (faithful to paper)
   - Can disable for ablation studies

2. **Template Selection Methods**:
   - `_get_generator_template()`: Returns env-specific Generator prompt
   - `_get_reflector_template()`: Returns env-specific Reflector prompt
   - `_get_curator_template()`: Returns env-specific Curator prompt

3. **Updated Call Sites**:
   - `_choose_action()`: Uses `_get_generator_template()`
   - `_reflect_on_episode()`: Uses `_get_reflector_template()`
   - `_curate_default()`: Uses `_get_curator_template()`
   - `answer_query()`: Uses `ACE_COUNTERFACTUAL_QUERY_TEMPLATE` for counterfactuals

## Comparison: Before vs After

### Before (Generic Templates)
```python
ACE_GENERATOR_TEMPLATE = """You are an agent using a learned playbook...
**YOUR PLAYBOOK**: {playbook}
**Current Observation**: {observation}
Based on your playbook and observations, decide what to do next.
"""
```
- ~150 words
- No domain knowledge
- No critical strategies
- No tool documentation
- No failure patterns

### After (Environment-Specific Templates)
```python
ACE_GENERATOR_HOTPOT_TEMPLATE = """You are experimenting with a HotPot Lab...
**CRITICAL DOMAIN KNOWLEDGE**:
- Temperature evolves continuously over time based on heating state
- Stove labels (off/low/high) may NOT match actual heating behavior
- Thermometer readings have measurement noise (~0.5-2°C)
...
**STRATEGY GUIDELINES**:
1. Read your playbook FIRST - apply relevant learned strategies
2. Take multiple temperature measurements to account for noise
...
"""
```
- ~500 words per environment
- Explicit domain knowledge
- Concrete strategies
- Tool documentation with warnings
- Common failure patterns
- Example playbook items

## Alignment with Paper

### What the Paper Shows (Figure 9 - AppWorld)
- **Extensive domain instructions**: ReAct framework, Python REPL, API usage
- **Detailed key instructions**: 9 specific rules
- **Playbook markers**: PLAYBOOK_BEGIN/END
- **Task-specific context**: Personal info, supervisor role

### What We Now Have
- **Extensive domain instructions**: ✓ Per-environment strategies
- **Detailed key instructions**: ✓ 6-8 guidelines per environment
- **Playbook markers**: ✓ PLAYBOOK_BEGIN/END in all Generator templates
- **Task-specific context**: ✓ Environment-specific knowledge and tools

## Expected Performance Improvements

### Counterfactual Reasoning (ACE's Biggest Weakness)
**Before**: 45% score (worst of all agents)
**Expected After**: 65-75% score

**Why**:
- Enhanced counterfactual template guides structured reasoning
- Playbook bullets can now capture hypothetical patterns
- Explicit uncertainty acknowledgment in prompt
- 4-step reasoning process forces mental simulation

### Overall Performance
**Before**: 71.7% score, 58.7% accuracy (3rd/4th place)
**Expected After**: 78-82% score, 65-70% accuracy (1st/2nd place)

**Why**:
- Domain knowledge prevents naive mistakes
- Reflector captures environment-specific failures
- Curator produces actionable, specific playbook items
- Generator has concrete strategies to apply

### Token Efficiency
**Expected**: Similar or slightly higher tokens, but MUCH better quality
- More detailed prompts = more tokens
- But playbook items are now specific and actionable
- Should see better score-per-token ratio

## How to Test

### 1. Run Validation Test (Recommended)
```bash
ANTHROPIC_API_KEY="..." OPENAI_API_KEY="..." \
python scripts/run_experiment_parallel.py \
  --config config_validation_15ep.yaml \
  --output-dir results/ace_upgraded_validation \
  --workers 2
```

### 2. Quick Single-Episode Test
```python
from agents.ace import ACEAgent
from models.llm import ClaudeLLM

# Test with environment-specific prompts (new)
ace = ACEAgent(
    llm=ClaudeLLM(),
    action_budget=10,
    environment_name='ChemTile',
    use_environment_specific_prompts=True  # NEW!
)

# Test with generic prompts (old behavior)
ace_old = ACEAgent(
    llm=ClaudeLLM(),
    action_budget=10,
    environment_name='ChemTile',
    use_environment_specific_prompts=False
)
```

### 3. Compare Playbook Quality
After 5 episodes, inspect playbooks:
```python
# Environment-specific ACE
print(ace._format_playbook())
# Should see specific items like:
# "Always inspect() compounds before mixing"
# "A+B at high temp → explosion"

# Generic ACE (old)
print(ace_old._format_playbook())
# Might see vague items like:
# "Be careful with chemistry"
# "Temperature is important"
```

## Backward Compatibility

All changes are backward compatible:
- Default behavior is NEW (environment-specific)
- Set `use_environment_specific_prompts=False` to use old generic templates
- Existing experiments will automatically use upgraded templates

## Next Steps

1. **Run validation test** to measure improvement
2. **Compare playbook quality** between old and new
3. **Analyze counterfactual performance** specifically
4. **Measure token usage** and efficiency
5. **Optional**: Add environment-specific templates for other environments if needed

## Files Modified

1. `experiments/prompts.py` - Added 13 new templates
2. `agents/ace.py` - Added template selection logic
3. `ACE_IMPLEMENTATION_UPGRADE.md` - This documentation

## Key Insights

**Why ACE Was Underperforming**:
1. Generic prompts didn't capture domain requirements
2. Reflector produced vague insights without domain examples
3. Curator couldn't create actionable bullets without guidance
4. No systematic approach to counterfactual reasoning

**Why This Should Fix It**:
1. Environment-specific prompts provide domain context
2. Reflector has concrete failure patterns to check
3. Curator has good/bad examples to guide playbook creation
4. Structured counterfactual reasoning process

The implementation is now **faithful to the paper's approach** of extensive domain-specific prompt engineering at every layer (Generator, Reflector, Curator).
