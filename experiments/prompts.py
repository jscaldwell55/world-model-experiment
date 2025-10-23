# experiments/prompts.py
"""
All prompts versioned in one place for reproducibility.
"""

PROMPT_VERSION = "v1.1.0"

# ============================================================================
# Prior Generation Prompts (Actor Agent)
# ============================================================================

HOTPOT_PRIOR_GENERATION_TEMPLATE = """You are initializing your beliefs about a laboratory environment based on the initial observation.

Initial Observation:
{initial_observation}

Environment Type: Hot-Pot Lab

Your task is to set initial beliefs (priors) about the environment's dynamics based on this observation.
You need to specify parameters for:

1. heating_rate_mean: Expected temperature change per second (°C/s)
   - Range: [-5.0, 5.0]
   - Positive = heating, Negative = cooling, Zero = stable
   - Consider: What does the observation suggest about the heating state?

2. heating_rate_std: Your uncertainty about the heating rate (°C/s)
   - Range: [0.1, 10.0]
   - Higher = more uncertain about the heating dynamics
   - Consider: How confident are you? Could labels be misleading?

3. measurement_noise: Expected noise in temperature measurements (°C)
   - Range: [0.1, 5.0]
   - Typical thermometers: 0.5-2.0°C
   - Consider: How precise do you expect measurements to be?

IMPORTANT GUIDELINES:
- Express genuine uncertainty - labels may be misleading
- Use temperature=0 for consistency
- Base your priors on what you observe, but maintain skepticism
- If the observation suggests heating but you're unsure, use a moderate mean with higher std

Respond with ONLY a JSON object in this exact format (no markdown, no explanations):
{{"heating_rate_mean": <float>, "heating_rate_std": <float>, "measurement_noise": <float>, "reasoning": "<1-2 sentence explanation>"}}

Your prior belief (JSON only):"""

SWITCHLIGHT_PRIOR_GENERATION_TEMPLATE = """You are initializing your beliefs about a switch-light environment based on the initial observation.

Initial Observation:
{initial_observation}

Environment Type: Switch-Light

Your task is to set initial beliefs (priors) about the wiring configuration.
You need to specify parameters for:

1. connection_probs: Probability matrix for switch → light connections
   - Format: [[P(switch_0 → light_0), P(switch_0 → light_1)],
              [P(switch_1 → light_0), P(switch_1 → light_1)]]
   - Each probability in range [0.0, 1.0]
   - Consider: What wiring patterns seem plausible?

2. uncertainty: Overall uncertainty about the wiring
   - Range: [0.0, 1.0]
   - Higher = less confident about connections
   - Consider: How much can you infer from this initial observation?

IMPORTANT GUIDELINES:
- If you can't determine connections yet, use uniform priors (0.5)
- Express genuine uncertainty in the uncertainty parameter
- Common patterns: direct (switch_i → light_i) or crossed (switch_0 → light_1, etc.)

Respond with ONLY a JSON object in this exact format (no markdown, no explanations):
{{"connection_probs": [[<float>, <float>], [<float>, <float>]], "uncertainty": <float>, "reasoning": "<1-2 sentence explanation>"}}

Your prior belief (JSON only):"""

CHEMTILE_PRIOR_GENERATION_TEMPLATE = """You are initializing your beliefs about a chemistry lab environment based on the initial observation.

Initial Observation:
{initial_observation}

Environment Type: Chemical Tile Lab

Your task is to set initial beliefs (priors) about reaction outcomes and safety.
You need to specify parameters for:

1. reaction_safety_priors: Prior beliefs about compound safety
   - Format: {{"compound_A": <float>, "compound_B": <float>, ...}}
   - Each value is P(safe) in range [0.0, 1.0]
   - Consider: Do any compounds seem particularly dangerous or safe?

2. reaction_outcome_uncertainty: How unpredictable reactions are
   - Range: [0.0, 1.0]
   - Higher = more chaotic/unpredictable chemistry
   - Consider: Does the environment seem controlled or volatile?

3. temperature_effect_prior: Expected influence of temperature on reactions
   - Range: [0.0, 1.0]
   - Higher = temperature has stronger effects
   - Consider: Typical chemistry labs have moderate-to-strong temperature effects

IMPORTANT GUIDELINES:
- If no specific compounds are mentioned, use generic cautious priors (0.5)
- Express genuine uncertainty - chemistry can be unpredictable
- Safety-first: when uncertain, assume moderate danger

Respond with ONLY a JSON object in this exact format (no markdown, no explanations):
{{"reaction_safety_priors": {{"compound_A": <float>, "compound_B": <float>}}, "reaction_outcome_uncertainty": <float>, "temperature_effect_prior": <float>, "reasoning": "<1-2 sentence explanation>"}}

Your prior belief (JSON only):"""

# ============================================================================
# Observer Agent Prompts
# ============================================================================

OBSERVER_QUERY_TEMPLATE = """You are analyzing an environment without being able to interact with it.

Initial Description:
{initial_description}

Question: {question}

Reason carefully about what would happen based on the description and your knowledge.
Think step-by-step about the underlying physics, causal relationships, and likely outcomes.

Provide your answer in this exact format:
ANSWER: <your prediction>
CONFIDENCE: <0.0 to 1.0>
REASONING: <brief explanation of your reasoning>
"""

# ============================================================================
# Actor Agent Prompts
# ============================================================================

ACTOR_ACTION_TEMPLATE = """You are conducting experiments to understand an environment's dynamics.

Current Belief State:
{belief_state}

Current Observation:
{observation}

Recent History:
{memory_summary}

{available_tools}

Actions Remaining: {actions_remaining}

Based on your observations so far, decide what to do next to improve your understanding.
Think about:
- What have you learned?
- What's still uncertain?
- What experiment would be most informative?

Provide your response in this format:
THOUGHT: <what have you learned? what's uncertain?>
ACTION: <tool_name(params)>
"""

ACTOR_QUERY_TEMPLATE = """You have been experimenting with an environment and building a model of its dynamics.

Your Belief State:
{belief_state}

Your Experience:
{memory_summary}

Question: {question}

Use your updated beliefs and experience to answer the question.
Reference specific observations that inform your answer.

Format:
ANSWER: <your prediction>
CONFIDENCE: <0.0 to 1.0>
REASONING: <explain using your experience>
"""

BELIEF_UPDATE_TEMPLATE = """Update your beliefs based on this new evidence.

Current Beliefs:
{current_belief}

New Observation:
{observation}

Time Elapsed: {time_elapsed}

Your Experience:
{memory_summary}

Reasoning:
- Did this observation match your prediction?
- Should you increase or decrease uncertainty?
- What have you learned about the dynamics?

CRITICAL INSTRUCTIONS:
- Respond with ONLY a JSON object
- Use double quotes ("), not single quotes (')
- No explanatory text before or after
- No markdown code blocks
- Just the raw JSON

Valid format example:
{{"heating_rate_mean": 1.5, "heating_rate_std": 0.3, "measurement_noise": 2.0}}

Your updated belief (JSON only):
"""

# ============================================================================
# Text Reader Agent Prompts
# ============================================================================

TEXT_READER_QUERY_TEMPLATE = """You have access to descriptions and prior episode logs from this environment.

Initial Description:
{initial_description}

Prior Experience from Other Episodes:
{prior_experience}

Question: {question}

Use both the description and the patterns you observe in prior episodes to answer.
Look for consistent patterns across episodes that reveal the true dynamics.

Format:
ANSWER: <your prediction>
CONFIDENCE: <0.0 to 1.0>
REASONING: <explain using prior episodes as evidence>
"""

# ============================================================================
# ACE Agent Prompts (Agentic Context Engineering)
# ============================================================================

ACE_GENERATOR_TEMPLATE = """You are an agent using a learned playbook to experiment with an environment.

**YOUR PLAYBOOK** (strategies learned from prior episodes):
{playbook}

**Current Observation:**
{observation}

**Recent History:**
{memory_summary}

**Available Tools:**
{available_tools}

**Actions Remaining:** {actions_remaining}

Based on your playbook and observations, decide what to do next.
- Use strategies from your playbook when relevant
- Focus on experiments that reduce uncertainty
- If playbook is empty, use general scientific reasoning

Provide your response in this format:
THOUGHT: <what does your playbook suggest? what's uncertain?>
ACTION: <tool_name(params)>
"""

ACE_REFLECTOR_TEMPLATE = """You are reflecting on an episode to extract insights for your playbook.

**Environment:** {environment_name}

**Your Current Playbook:**
{playbook}

**Episode Trajectory:**
{episode_steps}

**Test Results:**
{test_results}

Analyze what happened in this episode:
1. What strategies worked well?
2. What strategies failed or were misleading?
3. What new insights did you gain?
4. Which playbook items (if any) were helpful or harmful?

Respond with ONLY a JSON object in this format:
{{
  "what_worked": ["strategy 1", "strategy 2", ...],
  "what_failed": ["mistake 1", "mistake 2", ...],
  "new_insights": ["insight 1", "insight 2", ...],
  "helpful_bullets": ["bullet_id_1", "bullet_id_2", ...],
  "harmful_bullets": ["bullet_id_3", ...]
}}

Your reflection (JSON only):"""

ACE_CURATOR_TEMPLATE = """You are curating insights into structured playbook updates.

**Environment:** {environment_name}

**Current Playbook:**
{playbook}

**Insights from Reflector:**
{insights}

Your task: Convert insights into concise, actionable playbook items.

Guidelines:
- Each item should be specific and actionable
- Avoid redundancy with existing playbook
- Categorize into appropriate sections:
  - strategies_and_hard_rules: General approaches and rules
  - useful_code_snippets: Specific action patterns
  - troubleshooting_and_pitfalls: Common mistakes to avoid
  - apis_to_use: Recommended tools/functions
  - verification_checklist: Things to check

Respond with ONLY a JSON object in this format:
{{
  "delta_items": [
    {{
      "section": "strategies_and_hard_rules",
      "content": "Always verify temperature before trusting labels",
      "operation": "add"
    }},
    {{
      "section": "troubleshooting_and_pitfalls",
      "content": "Labels can be deceptive - measure directly",
      "operation": "add"
    }}
  ]
}}

Your curation (JSON only):"""

ACE_QUERY_TEMPLATE = """You have experimented with an environment and built a playbook of strategies.

**Your Playbook:**
{playbook}

**Your Experience:**
{memory_summary}

**Question:** {question}

Use your playbook and experience to answer the question.
Reference specific playbook items that inform your answer.

Format:
ANSWER: <your prediction>
CONFIDENCE: <0.0 to 1.0>
REASONING: <explain using playbook items>
"""

# ============================================================================
# Model-Based Agent Prompts
# ============================================================================

MODEL_BASED_PLANNING_TEMPLATE = """You have an explicit transition model learned from experience.

Current State:
{current_observation}

Your Belief State:
{belief_state}

Model Predictions for Each Action:
{action_predictions}

Recent Experience:
{memory_summary}

Actions Remaining: {actions_remaining}

Based on your model's predictions, choose the action that will be most informative.
Consider:
- Which predicted outcome would most reduce your uncertainty?
- Which action tests your most uncertain beliefs?

Format:
THOUGHT: <reasoning about model predictions>
ACTION: <tool_name(params)>
"""

# ============================================================================
# Common Utility Prompts
# ============================================================================

PARSE_CONFIDENCE_INSTRUCTION = """
When providing confidence, use this scale:
- 0.9-1.0: Very certain, strong evidence
- 0.7-0.9: Fairly confident, good evidence
- 0.5-0.7: Moderate confidence, some evidence
- 0.3-0.5: Low confidence, weak evidence
- 0.0-0.3: Very uncertain, little to no evidence
"""

ANSWER_FORMAT_INSTRUCTION = """
IMPORTANT: You must provide your answer in this exact format:

ANSWER: <your prediction or answer>
CONFIDENCE: <number between 0.0 and 1.0>
REASONING: <brief explanation>

Do not deviate from this format. The answer will be automatically parsed.
"""


def format_observation_history(steps: list, max_steps: int = 5) -> str:
    """
    Format recent observation history for prompts.

    Args:
        steps: List of AgentStep objects
        max_steps: Maximum number of steps to include

    Returns:
        Formatted string
    """
    if not steps:
        return "No observations yet."

    recent = steps[-max_steps:]
    lines = []

    for step in recent:
        action_str = step.action if step.action else "observed"
        obs_str = str(step.observation)[:100]  # Truncate long observations

        line = f"Step {step.step_num}: {action_str} -> {obs_str}"

        if hasattr(step, 'surprisal') and step.surprisal > 0:
            line += f" (surprisal: {step.surprisal:.2f})"

        lines.append(line)

    return "\n".join(lines)


def format_belief_state(belief: 'BeliefState') -> str:
    """
    Format belief state for prompts.

    Args:
        belief: BeliefState object

    Returns:
        Formatted JSON string
    """
    if belief is None:
        return "{}"

    try:
        return belief.model_dump_json(indent=2)
    except AttributeError:
        # Fallback for non-pydantic beliefs
        return str(belief)


def extract_answer_components(response: str) -> tuple[str, float, str]:
    """
    Extract answer, confidence, and reasoning from LLM response.

    Args:
        response: Raw LLM response text

    Returns:
        Tuple of (answer, confidence, reasoning)
    """
    import re

    # Extract answer
    answer_match = re.search(r'ANSWER:\s*(.+?)(?=CONFIDENCE:|REASONING:|$)', response, re.DOTALL | re.IGNORECASE)
    answer = answer_match.group(1).strip() if answer_match else response[:200]

    # Extract confidence
    conf_match = re.search(r'CONFIDENCE:\s*(\d*\.?\d+)', response, re.IGNORECASE)
    confidence = float(conf_match.group(1)) if conf_match else 0.5

    # Clamp confidence to [0, 1]
    confidence = max(0.0, min(1.0, confidence))

    # Extract reasoning
    reasoning_match = re.search(r'REASONING:\s*(.+?)(?=$)', response, re.DOTALL | re.IGNORECASE)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    return answer, confidence, reasoning


def extract_action(response: str) -> str:
    """
    Extract ACTION: tool_name(...) from LLM response.

    Args:
        response: Raw LLM response text

    Returns:
        Action string like "measure_temp()" or None
    """
    import re

    # Look for ACTION: followed by tool call
    match = re.search(r'ACTION:\s*(\w+\([^)]*\))', response, re.IGNORECASE)

    if match:
        return match.group(1).strip()

    # Fallback: look for any tool call pattern
    match = re.search(r'(\w+\([^)]*\))', response)

    return match.group(1).strip() if match else None


def extract_thought(response: str) -> str:
    """
    Extract THOUGHT: ... from LLM response.

    Args:
        response: Raw LLM response text

    Returns:
        Thought string
    """
    import re

    match = re.search(r'THOUGHT:\s*(.+?)(?=ACTION:|BELIEF|$)', response, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()

    # Fallback: return first 200 chars
    return response[:200].strip()
