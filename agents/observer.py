# agents/observer.py
import time
from typing import Tuple, Optional
from agents.base import Agent, AgentStep, LLMInterface
from experiments.prompts import OBSERVER_QUERY_TEMPLATE, extract_answer_components


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

        # Parse answer and confidence
        answer, confidence, reasoning = extract_answer_components(response)

        return answer, confidence

    def reset(self):
        """Reset agent state for new episode"""
        super().reset()
        self.initial_description = None
