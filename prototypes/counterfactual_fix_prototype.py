#!/usr/bin/env python3
"""
Prototype implementation of counterfactual-specific prompting (Strategy 1).

This demonstrates how to add counterfactual reasoning support to agents
with minimal code changes and maximum impact.
"""

# ============================================================================
# Step 1: Add counterfactual detection helper
# ============================================================================

def is_counterfactual_question(question: str) -> bool:
    """
    Detect if a question is asking about a counterfactual scenario.

    Counterfactual markers:
    - "if we had..." (past perfect)
    - "would have..." (conditional perfect)
    - "had we..." (inverted conditional)
    - "suppose we had..."
    """
    counterfactual_markers = [
        "if we had",
        "if i had",
        "would have",
        "had we",
        "had i",
        "suppose we had",
        "supposing we had",
        "imagine we had"
    ]

    question_lower = question.lower()
    return any(marker in question_lower for marker in counterfactual_markers)


# ============================================================================
# Step 2: Add counterfactual-specific prompts
# ============================================================================

COUNTERFACTUAL_QUERY_TEMPLATE = """You are answering a COUNTERFACTUAL question - reasoning about what WOULD have happened under different circumstances.

CRITICAL INSTRUCTIONS:
1. Distinguish between ACTUAL observations (what you experienced) and COUNTERFACTUAL scenarios (what you're being asked about)
2. Use your learned mental model to SIMULATE the alternative scenario
3. Express appropriate UNCERTAINTY - you didn't observe this scenario, so confidence should reflect that
4. For scenarios with stochastic elements (like faulty relays, random outcomes), use words like "possibly", "might", "depends"

YOUR EXPERIENCE:
{observation_history}

YOUR BELIEFS:
{belief_state}

COUNTERFACTUAL QUESTION:
{question}

REASONING PROCESS:
Step 1 - What actually happened:
[Describe your actual observations]

Step 2 - What alternative is being proposed:
[Describe the counterfactual scenario]

Step 3 - Mental simulation:
[Use your learned model to predict what would have happened]

Step 4 - Express uncertainty:
[Is this outcome certain, likely, possible, or unknown?]

ANSWER: [Your counterfactual prediction]
CONFIDENCE: [0.0-1.0, typically < 0.8 for unobserved scenarios]
"""

INTERVENTIONAL_QUERY_TEMPLATE = """You are answering an INTERVENTIONAL question - predicting what WILL or WOULD happen if we take certain actions.

YOUR EXPERIENCE:
{observation_history}

YOUR BELIEFS:
{belief_state}

QUESTION:
{question}

Use your learned model to predict the outcome of the proposed intervention.

ANSWER: [Your prediction]
CONFIDENCE: [0.0-1.0]
"""

PLANNING_QUERY_TEMPLATE = """You are answering a PLANNING question - determining the best course of action to achieve a goal.

YOUR EXPERIENCE:
{observation_history}

YOUR BELIEFS:
{belief_state}

QUESTION:
{question}

Reason about which actions would best achieve the stated goal, considering both effectiveness and safety.

ANSWER: [Your recommended plan]
CONFIDENCE: [0.0-1.0]
"""


# ============================================================================
# Step 3: Modified answer_query() for Actor agent
# ============================================================================

def actor_answer_query_with_counterfactual_support(self, question: str):
    """
    Enhanced answer_query that uses specialized prompts for different query types.

    This is a drop-in replacement for the existing answer_query method.
    """
    from experiments.prompts import format_observation_history, extract_answer_components

    # Detect query type
    if is_counterfactual_question(question):
        template = COUNTERFACTUAL_QUERY_TEMPLATE
    elif any(word in question.lower() for word in ["will", "would", "if we", "what happens"]):
        template = INTERVENTIONAL_QUERY_TEMPLATE
    elif any(word in question.lower() for word in ["how can", "what's the best", "should we", "safest"]):
        template = PLANNING_QUERY_TEMPLATE
    else:
        template = INTERVENTIONAL_QUERY_TEMPLATE  # Default

    # Format observation history
    obs_history = format_observation_history(self.memory, max_steps=10)

    # Format belief state
    belief_str = str(self.belief_state) if hasattr(self, 'belief_state') else "No explicit belief state"

    # Build prompt
    prompt = template.format(
        observation_history=obs_history,
        belief_state=belief_str,
        question=question
    )

    # Query LLM
    response = self.llm.generate(prompt, temperature=0.0)

    # Extract answer and confidence
    answer, confidence, reasoning = extract_answer_components(response)

    # Counterfactual confidence calibration
    if is_counterfactual_question(question):
        # Cap confidence for counterfactuals - we didn't actually observe this
        if confidence > 0.85:
            confidence = 0.85  # Maximum confidence for unobserved scenarios

        # Check for overconfident language
        overconfident_words = ["definitely", "certainly", "absolutely", "always", "never"]
        if any(word in answer.lower() for word in overconfident_words):
            confidence *= 0.8  # Penalize overconfidence

    # Record token usage
    input_tokens, output_tokens = self.llm.get_last_usage()
    if hasattr(self, 'token_accountant'):
        self.token_accountant.record(
            'evaluation',
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={'question': question[:50], 'is_counterfactual': is_counterfactual_question(question)}
        )

    return answer, confidence


# ============================================================================
# Step 4: Environment-specific fixes
# ============================================================================

def enhance_switchlight_uncertainty(answer: str, question: str) -> tuple[str, float]:
    """
    Special handling for SwitchLight counterfactuals that require uncertainty.

    SwitchLight questions often involve faulty relays and stochastic elements.
    Agents should express "possibly" / "might" rather than definitive answers.
    """
    # Check if question involves uncertainty
    uncertainty_triggers = ["relay", "jiggle", "faulty", "random", "might"]

    if any(trigger in question.lower() for trigger in uncertainty_triggers):
        # Check if answer expresses uncertainty
        uncertainty_words = ["possibly", "might", "maybe", "could", "depends", "uncertain"]

        if not any(word in answer.lower() for word in uncertainty_words):
            # Answer is too definitive - add uncertainty
            if answer.lower().startswith("yes"):
                answer = "Possibly yes, " + answer[4:]
            elif answer.lower().startswith("no"):
                answer = "Possibly no, " + answer[3:]
            else:
                answer = "It depends. " + answer

    return answer


# ============================================================================
# Step 5: Complete integration example
# ============================================================================

class EnhancedActorAgent:
    """
    Example of how to integrate counterfactual support into existing Actor agent.

    Changes required:
    1. Replace answer_query() method
    2. Add is_counterfactual_question() helper
    3. Add counterfactual-specific prompts to experiments/prompts.py
    """

    def answer_query(self, question: str):
        """Enhanced version with counterfactual support."""
        # Use the enhanced implementation
        answer, confidence = actor_answer_query_with_counterfactual_support(self, question)

        # Apply environment-specific fixes if needed
        if hasattr(self, 'environment_name') and self.environment_name == "SwitchLight":
            answer = enhance_switchlight_uncertainty(answer, question)

        return answer, confidence


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Test counterfactual detection
    test_questions = [
        ("If we had waited 60 seconds with the stove on high instead of measuring immediately, would the temperature be above 80°C?", True),
        ("If we increase temperature to high before mixing A and B, how does explosion risk change?", False),
        ("What's the safest path to produce compound D?", False),
        ("If we had cooled the system before mixing C and B, would explosion probability decrease?", True),
        ("If we flip the switch to ON, will the light definitely turn on?", False),
        ("If we had jiggled the relay before flipping the switch, would the outcome have been different?", True),
    ]

    print("Testing counterfactual detection:")
    print("=" * 80)
    for question, expected in test_questions:
        detected = is_counterfactual_question(question)
        status = "✓" if detected == expected else "✗"
        print(f"{status} {question[:70]}...")
        print(f"   Expected: {expected}, Detected: {detected}\n")

    # Sample prompt generation
    print("\n" + "=" * 80)
    print("Sample Counterfactual Prompt:")
    print("=" * 80)
    sample_prompt = COUNTERFACTUAL_QUERY_TEMPLATE.format(
        observation_history="Step 1: mix(A, B) -> nothing\nStep 2: inspect(B) -> 'Catalyst compound'",
        belief_state="reaction_probs: {A+B: {C: 0.78, explode: 0.10, nothing: 0.12}}",
        question="If we had cooled the system before mixing A and B, would explosion probability decrease?"
    )
    print(sample_prompt[:500] + "...\n")
