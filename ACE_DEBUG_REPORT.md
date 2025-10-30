# ACE Agent Debug Report

**Date**: 2025-10-30
**Issue**: ACE agent shows `surprisal = 0.0` for all exploration steps

---

## Root Cause Analysis

### 1. **Hardcoded Surprisal = 0.0**

**Location**: `agents/ace.py:135`

```python
step = AgentStep(
    timestamp=time.time(),
    step_num=len(self.memory),
    thought=thought,
    action=action,
    observation=observation,
    belief_state={'playbook_size': self._get_playbook_size()},
    surprisal=0.0,  # ← HARDCODED TO ZERO!
    token_usage=0
)
```

### 2. **Missing Methods**

ACE agent does NOT implement:
- ❌ `compute_surprisal(observation)` - Not present
- ❌ `update_belief_from_observation(observation)` - Not present
- ❌ Any form of belief tracking beyond playbook size

**Comparison with Actor Agent** (which correctly computes surprisal):
- ✅ `compute_surprisal(observation)` implemented at line 203
- ✅ `update_belief_from_observation(observation)` implemented at line 214
- ✅ Uses `belief_state.log_likelihood()` to calculate surprisal
- ✅ Tracks parametric beliefs (e.g., HotPotBelief, SwitchLightBelief)

### 3. **How the Runner Handles This**

From `experiments/runner.py:184-191`:

```python
# Recompute surprisal on the RESULT observation (before belief update)
if hasattr(agent, 'compute_surprisal'):
    agent_step.surprisal = agent.compute_surprisal(new_observation)

# Update belief with RESULT observation (if actor agent)
if hasattr(agent, 'update_belief_from_observation'):
    agent.update_belief_from_observation(new_observation)
    agent_step.belief_state = agent.get_belief_state()
```

**Result**: Since ACE doesn't have `compute_surprisal()`, the hardcoded `0.0` is never overwritten.

---

## Design Question: SHOULD ACE Compute Surprisal?

### Option A: **ACE Shouldn't Have Surprisal** (By Design)

**Rationale:**
- ACE uses "context evolution" (playbook curation), not parametric belief updates
- The playbook is qualitative knowledge, not a probability distribution
- There's no clear P(observation | playbook) to compute

**Implications:**
- Surprisal = 0.0 is correct
- ACE fundamentally different from Actor (non-probabilistic)
- Should document this clearly

### Option B: **ACE Should Compute Surprisal** (Bug to Fix)

**Rationale:**
- Active inference requires measuring surprise to guide exploration
- Can compute surprisal using implicit belief from playbook:
  - Parse playbook for predictions/expectations
  - Compare with observations
  - Higher mismatch = higher surprisal
- Needed for fair comparison with Actor

**Implementations**:

#### B1. Simple Heuristic Surprisal
```python
def compute_surprisal(self, observation: dict) -> float:
    """
    Compute surprisal based on playbook expectations.

    If playbook contains explicit predictions that are violated,
    surprisal is high. Otherwise, surprisal reflects uncertainty.
    """
    if self._get_playbook_size() == 0:
        # No knowledge yet = high uncertainty but not "surprising"
        return 0.5

    # Check if observation contradicts playbook entries
    contradictions = self._check_playbook_contradictions(observation)

    if contradictions:
        # Observation violates known rules
        return 0.8 + 0.2 * len(contradictions)
    else:
        # Observation consistent with playbook
        return 0.1
```

#### B2. LLM-Based Surprisal
```python
def compute_surprisal(self, observation: dict) -> float:
    """
    Ask LLM to rate surprise given playbook and observation.
    """
    prompt = f"""
    Given your playbook:
    {self._format_playbook()}

    You just observed:
    {observation}

    On a scale of 0-1, how surprising is this observation?
    0 = completely expected
    1 = completely unexpected/contradictory

    Provide just a number.
    """

    response = self.llm.generate(prompt, temperature=0)
    return float(extract_number(response))
```

#### B3. Observation Novelty (No Playbook Needed)
```python
def compute_surprisal(self, observation: dict) -> float:
    """
    Compute surprisal as observation novelty compared to history.
    """
    # Check if we've seen similar observations before
    for past_step in self.memory:
        if self._observations_similar(past_step.observation, observation):
            return 0.2  # Seen similar before

    return 0.8  # Novel observation
```

---

## Recommended Action

### Phase 1: Document Current Behavior

1. **Add comment to ACE agent** explaining why surprisal = 0:
   ```python
   # ACE uses context evolution (playbook), not parametric beliefs
   # Surprisal is not computed as ACE lacks a probability distribution
   surprisal=0.0,  # Fixed: ACE is non-probabilistic
   ```

2. **Update evaluation** to note ACE and Actor are fundamentally different

### Phase 2: Implement Simple Surprisal (If Needed)

If you want ACE to guide exploration via surprisal:

1. Implement Option B3 (Observation Novelty) - simplest, no LLM calls needed
2. Test on verification run
3. Compare ACE performance with/without surprisal

### Phase 3: Advanced Surprisal (Future Work)

For true active inference in ACE:
- Parse playbook for explicit predictions
- Check for contradictions with observations
- Use LLM to assess surprise when needed

---

## Immediate Next Steps

**Before running full study:**

1. ✅ Decide: Should ACE compute surprisal or not?

2. If YES:
   - Implement `compute_surprisal()` in ACE agent
   - Test on verification run
   - Ensure surprisal > 0 for novel observations

3. If NO:
   - Document that ACE is non-probabilistic by design
   - Accept that surprisal = 0 is correct
   - Note this in paper/analysis

4. Either way:
   - Run verification with new evaluation questions
   - Check Observer <40%, ACE >60%

---

## Code Changes Required (If Implementing Surprisal)

### File: `agents/ace.py`

Add after `act()` method:

```python
def compute_surprisal(self, observation: dict) -> float:
    """
    Compute surprisal as observation novelty.

    ACE doesn't have parametric beliefs like Actor, but we can
    still measure surprise as whether we've seen similar observations.

    Returns:
        Surprisal value (0.0 = familiar, 1.0 = novel)
    """
    # Check memory for similar past observations
    for past_step in self.memory:
        past_obs = past_step.observation

        # Define similarity based on environment type
        if self._observations_similar(past_obs, observation):
            return 0.2  # Low surprisal - seen this before

    # Novel observation
    return 0.7

def _observations_similar(self, obs1: dict, obs2: dict) -> bool:
    """
    Check if two observations are similar.

    Args:
        obs1, obs2: Observation dictionaries

    Returns:
        True if observations are similar
    """
    # Simple similarity: check if actions and key values match
    if obs1.get('action') != obs2.get('action'):
        return False

    # Environment-specific checks
    if 'measured_temp' in obs1 and 'measured_temp' in obs2:
        # Temperatures within 5 degrees
        return abs(obs1['measured_temp'] - obs2['measured_temp']) < 5.0

    if 'light_on' in obs1 and 'light_on' in obs2:
        # Same light state
        return obs1['light_on'] == obs2['light_on']

    # Default: not similar if we can't compare
    return False
```

Change in `act()` method:
```python
# Before:
surprisal=0.0,

# After:
surprisal=0.0,  # Will be recomputed by runner if compute_surprisal() exists
```

---

## Testing the Fix

```bash
# 1. Apply fix to agents/ace.py

# 2. Run single episode
ANTHROPIC_API_KEY="..." python3 << 'EOF'
from agents.ace import ACEAgent
from agents.base import create_llm
from environments.hot_pot import HotPotLab

llm = create_llm("claude-sonnet-4-5-20250929")
agent = ACEAgent(llm, action_budget=5, environment_name="HotPotLab")
env = HotPotLab(seed=42)

obs = env.reset()
agent.reset()

# First action
step1 = agent.act(obs)
print(f"Step 1 surprisal: {step1.surprisal}")  # Should be 0.7 (novel)

# Execute and get new obs
new_obs, _, _, _ = env.step(step1.action)

# Recompute surprisal (like runner does)
if hasattr(agent, 'compute_surprisal'):
    surprisal = agent.compute_surprisal(new_obs)
    print(f"Recomputed surprisal: {surprisal}")  # Should be > 0

# Second action with similar observation
step2 = agent.act(new_obs)
surprisal2 = agent.compute_surprisal(new_obs)
print(f"Step 2 surprisal: {surprisal2}")  # Should be ~0.2 (familiar)
EOF

# 3. Check that surprisal varies (not always 0.0)
```

---

## Questions for Discussion

1. **Is ACE supposed to be probabilistic or not?**
   - If not: surprisal = 0 is correct by design
   - If yes: need to implement surprisal calculation

2. **Does surprisal affect ACE's action selection?**
   - Currently: NO (playbook guides actions, not surprisal)
   - Actor: YES (belief state includes surprisal consideration)

3. **Should ACE and Actor be compared on equal footing?**
   - If yes: both need surprisal for active exploration
   - If no: acknowledge they're fundamentally different approaches

---

## Summary

**Current State:**
- ❌ ACE surprisal hardcoded to 0.0
- ❌ No compute_surprisal() method
- ❌ No belief updating mechanism

**Recommended Fix:**
- ✅ Implement simple observation novelty-based surprisal
- ✅ Add compute_surprisal() method
- ✅ Test on verification run before full study

**Impact:**
- ACE can now show non-zero surprisal
- More meaningful comparison with Actor
- Active exploration can be guided by surprise

