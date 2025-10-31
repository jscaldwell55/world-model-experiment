# agents/observer.py
import time
from typing import Tuple, Optional
from agents.base import Agent, AgentStep, LLMInterface
from experiments.prompts import OBSERVER_QUERY_TEMPLATE, extract_answer_components
from utils.token_accounting import TokenAccountant


class ObserverAgent(Agent):
    """
    Language-only agent - no interaction allowed.

    Observer agents receive initial environment description and reason
    about it without taking any actions. This tests pure language-based
    reasoning without interaction.
    """

    def __init__(self, llm: LLMInterface, action_budget: int):
        """
        Initialize Observer agent.

        Args:
            llm: LLM interface
            action_budget: Not used (Observer takes no actions)
        """
        super().__init__(llm, action_budget)
        self.initial_description = None
        self.token_accountant = TokenAccountant()  # Track token breakdown

    def act(self, observation: dict) -> AgentStep:
        """
        Observer doesn't act - just stores initial observation.

        Args:
            observation: Environment observation

        Returns:
            AgentStep with action=None
        """
        # Store initial description on first observation
        if self.initial_description is None:
            self.initial_description = observation

        # Observer just reasons, doesn't take actions
        step = AgentStep(
            timestamp=time.time(),
            step_num=len(self.memory),
            thought="Observer: reasoning from description only, no actions",
            action=None,  # Observer never takes actions
            observation=observation,
            belief_state={},
            surprisal=0.0,
            token_usage=0
        )

        self.memory.append(step)
        return step

    def answer_query(self, question: str) -> Tuple[str, float]:
        """
        Answer using only initial description and reasoning.

        Args:
            question: Question to answer

        Returns:
            Tuple of (answer, confidence)
        """
        if self.initial_description is None:
            return "No observations yet", 0.0

        # Format prompt with initial description
        prompt = OBSERVER_QUERY_TEMPLATE.format(
            initial_description=str(self.initial_description),
            question=question
        )

        # Generate answer
        response = self.llm.generate(prompt)

        # Record token usage for evaluation
        input_tokens, output_tokens = self.llm.get_last_usage()
        self.token_accountant.record(
            'evaluation',
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={'question': question[:50]}
        )

        # Parse answer and confidence
        answer, confidence, reasoning = extract_answer_components(response)

        return answer, confidence

    def reset(self):
        """Reset agent state for new episode"""
        super().reset()
        self.initial_description = None
        self.token_accountant.reset()

    # ========================================================================
    # Token Accounting
    # ========================================================================

    def get_token_breakdown(self) -> dict:
        """
        Get token breakdown by category.

        Returns:
            Dictionary with token breakdown and validation status
        """
        return self.token_accountant.to_dict()

    def validate_token_accounting(self, total_input: int, total_output: int) -> bool:
        """
        Validate that token breakdown matches totals.

        Args:
            total_input: Expected total input tokens
            total_output: Expected total output tokens

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
        """
        return self.token_accountant.validate(total_input, total_output)
