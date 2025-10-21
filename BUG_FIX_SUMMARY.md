# Bug Fix: Action-Observation Misalignment

## Problem

Actions and observations were misaligned by one step in episode logs. Step N showed action N but observation from step N-1.

**Example of the bug:**
```
Step 1:
  Action: wait(10)
  Observation: {action: measure_temp, measured_temp: 20.99}  ← WRONG! This is from step 0

Step 2:
  Action: measure_temp()
  Observation: {time: 10.0, action: wait(10.0)}  ← This is from step 1
```

## Root Cause

In `experiments/runner.py`, the episode loop was:
1. Call `agent.act(observation)` → agent processes OLD observation, chooses action
2. Inject OLD observation into agent_step (guard rail)
3. Log the step
4. Execute `env.step(action)` → get NEW observation (too late!)

The guard rail on line 131 was injecting the WRONG observation - the input observation instead of the result observation.

## Solution

### Changes to `agents/actor.py`

1. **Simplified `act()` method** (lines 62-97):
   - Removed surprisal computation
   - Removed belief update
   - Now only chooses action based on current state
   - Returns AgentStep with placeholder values that will be updated by runner

2. **Added public methods** (lines 135-164):
   - `get_belief_state()`: Get current belief as dictionary
   - `compute_surprisal(observation)`: Compute surprisal on any observation
   - `update_belief_from_observation(observation)`: Update belief with new observation

### Changes to `experiments/runner.py`

**Restructured episode loop** (lines 127-173):

1. Agent chooses action based on current observation
2. **Execute action and get RESULT observation**
3. **Inject RESULT observation** (guard rail - line 147)
4. **Compute surprisal on RESULT observation** (line 152)
5. **Update belief with RESULT observation** (line 157)
6. Log the step with correct pairing

**Key improvements:**
- Guard rail now injects the correct observation (the result)
- Surprisal computed on action result (not action input)
- Belief updated with action result (not action input)
- Proper guards for observer agents (lines 151, 156)

## Correctness

After the fix, each step logs:
- **action**: The action taken at step N
- **observation**: The observation RESULTING from action N
- **belief_state**: The belief AFTER seeing the result
- **surprisal**: How surprising the result was (computed before belief update)

## Guard Rails Maintained

✓ **Programmatic observation injection**: Still present (line 147)
  - Now injects RESULT observation instead of input observation
  - Still prevents LLM hallucination

✓ **No ground truth leakage**: Still present (lines 119-125)
  - All validation checks maintained
  - Environment-specific checks still active

✓ **Deterministic environments**: Not affected
  - Only changed logging, not execution

## Testing

Run a test episode:
```bash
python scripts/run_experiment.py --num-episodes 1 --output-dir results/test_fix_v2
```

Verify logs show correct pairing:
```bash
jq '.steps[] | {step: .step_num, action: .action, obs_keys: (.observation | keys)}' results/test_fix_v2/*/hot_pot_actor_ep000.json
```

Expected: Step N's observation should contain results from executing step N's action.
