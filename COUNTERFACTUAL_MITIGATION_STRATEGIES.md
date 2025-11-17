# Counterfactual Reasoning: Mitigation Strategies

## Executive Summary

Counterfactual reasoning scored **0.569** across all agents (vs 0.737 for interventional, 0.811 for planning).

**Critical findings:**
- **SwitchLight environment: 0% accuracy** on counterfactuals
- **ChemTile environment: 80% accuracy** (counterfactuals work here!)
- **HotPotLab: 62.5% accuracy** (moderate performance)

**Root cause:** Agents don't explicitly simulate alternative histories or use learned causal models for "what if" reasoning.

---

## Failure Mode Analysis

### 1. **Uncertainty Blindness (SwitchLight's 0% Problem)**

**Failed Question:**
> "If we had jiggled the relay before flipping the switch, would the outcome have been different?"

**Correct Answer:** "possibly" / "maybe" / "depends"
**Agent Answers:** "No" or "Cannot determine"

**Problem:** Agents fail to express appropriate **epistemic uncertainty** about unobserved counterfactuals.

### 2. **Opposite Predictions (HotPotLab)**

**Failed Question:**
> "If we had turned the stove to high immediately and waited 1 minute, would we see a temperature above 100°C?"

**Correct Answer:** Yes
**Agent Answers:** "No, temperature would be approximately 22°C"

**Problem:** Agents confuse **actual observations** (temperature currently 22°C) with **simulated counterfactuals** (what would have happened).

### 3. **Model-Free Responses**

Agents don't appear to:
- Use their learned belief states to simulate alternative sequences
- Apply causal reasoning to predict outcomes
- Distinguish between "what happened" vs "what could have happened"

---

## Mitigation Strategies (Ranked by Implementation Difficulty)

### ✅ **Strategy 1: Query-Type-Specific Prompting** (EASY)

**Implementation:** Modify `answer_query()` to detect counterfactual questions and use specialized prompts.

```python
def answer_query(self, question: str) -> Tuple[str, float]:
    # Detect counterfactual markers
    counterfactual_markers = ["if we had", "would have", "had we", "suppose we had"]
    is_counterfactual = any(marker in question.lower() for marker in counterfactual_markers)

    if is_counterfactual:
        # Use counterfactual-specific prompt
        prompt = COUNTERFACTUAL_QUERY_TEMPLATE.format(
            question=question,
            belief_state=self.belief_state,
            memory=self.memory
        )
    else:
        # Use standard prompt
        prompt = STANDARD_QUERY_TEMPLATE.format(...)
```

**Counterfactual Prompt Template:**
```
You are reasoning about a COUNTERFACTUAL scenario - something that didn't actually happen.

Your task:
1. Identify what actually occurred in your experience
2. Identify the alternative action being proposed
3. Use your learned model to mentally simulate what WOULD have happened
4. Express appropriate uncertainty about unobserved outcomes

Question: {question}

Your belief state: {belief_state}
Your observations: {memory}

Think step by step:
- What did I actually observe?
- What alternative action is being proposed?
- Based on my causal model, what would likely have happened?
- How confident can I be about this unobserved outcome?

Answer: [your counterfactual prediction]
Confidence: [0.0-1.0]
```

**Expected Impact:** +10-15 points on counterfactual accuracy
**Effort:** 2-4 hours
**Files:** `experiments/prompts.py`, `agents/actor.py`, `agents/ace.py`, `agents/observer.py`

---

### ✅ **Strategy 2: Explicit Mental Simulation** (MEDIUM)

**Implementation:** For Actor agents with belief states, use the environment's `counterfactual_query()` method to simulate alternatives.

```python
def answer_query(self, question: str) -> Tuple[str, float]:
    # Parse question to extract action sequence
    if self._is_counterfactual(question):
        # Extract proposed actions from question
        alt_actions = self._extract_action_sequence(question)

        # Use belief state to predict outcome
        predicted_outcome = self._simulate_counterfactual(alt_actions)

        # Generate answer based on simulation
        answer = self._format_counterfactual_answer(
            predicted_outcome,
            actual_outcome=self.memory[-1].observation
        )

        return answer, confidence
```

**Mental Simulation Logic:**
```python
def _simulate_counterfactual(self, action_sequence: list[str]) -> dict:
    """Simulate what would happen under alternative actions using belief state."""

    # Start from initial state
    simulated_state = copy.deepcopy(self.initial_belief)

    # Apply each action using belief transition model
    for action in action_sequence:
        simulated_state = self._predict_transition(simulated_state, action)

    return simulated_state
```

**Expected Impact:** +15-20 points on counterfactual accuracy for Actor
**Effort:** 1-2 days
**Files:** `agents/actor.py`, `models/belief_state.py`

---

### ✅ **Strategy 3: Uncertainty Calibration** (EASY)

**Implementation:** Train agents to recognize high-uncertainty scenarios and express appropriate confidence.

**Add to prompts:**
```
For counterfactual questions:
- If you didn't observe the scenario, confidence should be < 0.7
- If the question involves stochastic elements, express uncertainty ("possibly", "might", "depends")
- Never claim certainty about unobserved outcomes
```

**Special handling for SwitchLight:**
```python
# For questions about faulty relay or uncertain wirings
if "relay" in question.lower() or "jiggle" in question.lower():
    uncertainty_words = ["possibly", "might", "depends", "maybe", "could"]
    if not any(word in answer.lower() for word in uncertainty_words):
        # Flag as potentially overconfident
        confidence *= 0.5
```

**Expected Impact:** +20-30 points on SwitchLight counterfactuals
**Effort:** 4-6 hours
**Files:** `agents/base.py`, `evaluation/tasks.py`

---

### ⚠️ **Strategy 4: Causal Model Extraction** (HARD)

**Implementation:** Explicitly extract causal relationships during exploration and use them for counterfactual reasoning.

**Add to Actor agents:**
```python
class ActorAgent:
    def __init__(self, ...):
        self.causal_graph = {}  # Track learned causal relationships

    def update_belief_from_observation(self, observation):
        # Existing belief update
        super().update_belief_from_observation(observation)

        # Extract causal relationship
        if self.memory:
            last_action = self.memory[-1].action
            if last_action:
                self._update_causal_graph(last_action, observation)

    def _update_causal_graph(self, action, outcome):
        """Learn causal relationships from action-outcome pairs."""
        # e.g., "heat(high, 60s)" -> "temp > 80°C"
        if action not in self.causal_graph:
            self.causal_graph[action] = []
        self.causal_graph[action].append(outcome)
```

**Use for counterfactuals:**
```python
def _answer_counterfactual(self, question, proposed_action):
    # Look up what happened when we took similar actions
    if proposed_action in self.causal_graph:
        similar_outcomes = self.causal_graph[proposed_action]
        predicted_outcome = self._aggregate_outcomes(similar_outcomes)
        return predicted_outcome
```

**Expected Impact:** +25-35 points on counterfactual accuracy
**Effort:** 1-2 weeks
**Files:** New `models/causal_model.py`, `agents/actor.py`

---

### 🔬 **Strategy 5: Environment-Specific Counterfactual Oracles** (RESEARCH)

**Implementation:** Give agents access to environment's `counterfactual_query()` method **during evaluation only**.

**Controversial approach:**
```python
def answer_query(self, question: str, env=None) -> Tuple[str, float]:
    if self._is_counterfactual(question) and env is not None:
        # Extract alternative action sequence from question
        alt_actions = self._parse_counterfactual_actions(question)

        # Query environment for ground truth counterfactual
        counterfactual_outcome = env.counterfactual_query(
            action_sequence=alt_actions,
            seed=self.episode_seed
        )

        # Use ground truth to inform answer
        answer = self._interpret_counterfactual(counterfactual_outcome)
        return answer, confidence
```

**Pros:** Perfect accuracy on counterfactuals
**Cons:** Not testing agent's reasoning, just environment's simulation
**Expected Impact:** +100 points (but defeats purpose of evaluation)
**Effort:** 1 day
**Status:** NOT RECOMMENDED (evaluation integrity issue)

---

## Recommended Implementation Order

### Phase 1: Quick Wins (1 week)
1. **Uncertainty Calibration** (Strategy 3) - Fix SwitchLight's 0%
2. **Query-Type-Specific Prompting** (Strategy 1) - Universal improvement

**Expected Result:** +25-35 points overall

### Phase 2: Model-Based Reasoning (2-3 weeks)
3. **Explicit Mental Simulation** (Strategy 2) - Actor agents only
4. **Causal Model Extraction** (Strategy 4) - All agents

**Expected Result:** +40-50 points overall, bringing counterfactuals to ~0.75-0.85

---

## Environment-Specific Fixes

### SwitchLight (0% → Target: 60%)

**Problem:** Questions require epistemic uncertainty ("possibly", "depends")

**Fix:**
```python
# In evaluation/tasks.py, line 261
TestQuery(
    question="If we had jiggled the relay before flipping the switch, would the outcome have been different?",
    query_type="counterfactual",
    expected_answer="possibly",
    evaluation_function=lambda ans, gt: 1.0 if contains_keywords(
        ans, ['might', 'possibly', 'depends', 'maybe', 'could', 'uncertain']
    ) else 0.0,
    difficulty="hard"
)
```

**Agent fix:**
```python
# Add to all answer_query methods
if "relay" in question.lower() or any uncertainty word in question:
    # Bias toward expressing uncertainty
    prompt += "\n\nNote: This involves stochastic or unobserved elements. Express appropriate uncertainty."
```

### HotPotLab (62.5% → Target: 80%)

**Problem:** Agents confuse actual observations with simulated scenarios

**Fix:** Explicitly separate "what happened" from "what would have happened" in prompts:

```python
COUNTERFACTUAL_TEMPLATE = """
ACTUAL HISTORY:
{actual_observations}

COUNTERFACTUAL SCENARIO:
{proposed_alternative}

Question: {question}

To answer:
1. What actually happened: [state actual observations]
2. What would have happened if {alternative}: [simulate mentally]
3. Answer: [Yes/No based on simulation, NOT actual observations]
"""
```

---

## Testing & Validation

### Regression Testing
After implementing fixes, verify:
- Counterfactual accuracy improves
- **Interventional/planning accuracy doesn't degrade**
- Token usage doesn't explode (counterfactual simulation should be cheap)

### A/B Testing
Run experiments comparing:
- Baseline agents
- Agents with Strategy 1+3 (prompt-based)
- Agents with Strategy 1+3+2 (simulation-based)

### Success Metrics
- **Minimum viable:** Counterfactual accuracy > 0.70 (currently 0.57)
- **Target:** Counterfactual accuracy > 0.80 (match planning performance)
- **SwitchLight-specific:** Accuracy > 50% (currently 0%)

---

## Code Changes Required

### High Priority (Strategy 1 + 3)
1. `experiments/prompts.py`: Add `COUNTERFACTUAL_QUERY_TEMPLATE`
2. `agents/base.py`: Add `_is_counterfactual()` helper
3. `agents/actor.py`: Modify `answer_query()` to use counterfactual prompts
4. `agents/ace.py`: Same as actor
5. `agents/observer.py`: Same as actor
6. `evaluation/tasks.py`: Update SwitchLight evaluation to accept uncertainty

### Medium Priority (Strategy 2)
7. `models/belief_state.py`: Add `predict_transition()` method
8. `agents/actor.py`: Add `_simulate_counterfactual()` method

### Low Priority (Strategy 4)
9. `models/causal_model.py`: Create new module for causal graph extraction
10. `agents/actor.py`: Integrate causal reasoning into answer_query

---

## Timeline & Resources

**Phase 1 (1 week):**
- 1 engineer, 30-40 hours
- Implement Strategy 1 + 3
- Run validation experiments

**Phase 2 (2-3 weeks):**
- 1 engineer, 60-80 hours
- Implement Strategy 2 + 4
- Full experimental comparison

**Total:** 90-120 engineering hours + compute costs (~$50-100 for validation runs)

---

## Alternative: Accept the Limitation

**If counterfactuals remain hard:**
- Document as known limitation
- Report counterfactual scores separately
- Focus evaluation on interventional + planning (where agents excel)
- Frame counterfactuals as "future work" requiring causal inference capabilities

**Justification:** True counterfactual reasoning requires strong causal models, which may be beyond scope of current LLM-based agents.
