# agents/actor.py
import time
import json
import re
from typing import Tuple, Optional, Any
from agents.base import Agent, AgentStep, LLMInterface
from experiments.prompts import (
    ACTOR_ACTION_TEMPLATE,
    ACTOR_QUERY_TEMPLATE,
    BELIEF_UPDATE_TEMPLATE,
    extract_answer_components,
    extract_action,
    extract_thought,
    format_observation_history
)
from models.tools import get_tools_for_environment


class ActorAgent(Agent):
    """
    Interactive agent with belief state updates.

    Actor agents maintain a parametric belief state and update it
    based on observations. They actively choose actions to reduce
    uncertainty and improve their world model.
    """

    def __init__(
        self,
        llm: LLMInterface,
        action_budget: int,
        environment_name: Optional[str] = None
    ):
        """
        Initialize Actor agent.

        Args:
            llm: LLM interface
            action_budget: Maximum number of actions allowed
            environment_name: Name of environment (for tool selection)
        """
        super().__init__(llm, action_budget)
        self.belief_state = None
        self.environment_name = environment_name
        self.tools_class = None

        if environment_name:
            try:
                self.tools_class = get_tools_for_environment(environment_name)
            except ValueError:
                print(f"Warning: No tools found for {environment_name}")

    def set_belief_state(self, belief: 'BeliefState'):
        """
        Initialize belief state for environment.

        Args:
            belief: BeliefState instance (e.g., HotPotBelief)
        """
        self.belief_state = belief

    def act(self, observation: dict) -> AgentStep:
        """
        Choose next action based on current observation.

        Note: This method does NOT update belief or compute final surprisal.
        The runner will do that after executing the action and getting the result.

        Args:
            observation: Environment observation

        Returns:
            AgentStep with action (observation/belief/surprisal will be updated by runner)
        """
        # Choose action if budget allows
        if self.action_count < self.action_budget:
            thought, action = self._choose_action(observation)
            self.action_count += 1
        else:
            thought = "Action budget exhausted"
            action = None

        # Create step record (observation, belief, and surprisal will be set by runner)
        # We use dummy values here that will be overwritten
        step = AgentStep(
            timestamp=time.time(),
            step_num=len(self.memory),
            thought=thought,
            action=action,
            observation=observation,  # Placeholder - will be overwritten with result
            belief_state=self._serialize_belief(),  # Current belief before action
            surprisal=0.0,  # Placeholder - will be computed on result
            token_usage=0  # TODO: track from API
        )

        self.memory.append(step)
        return step

    def answer_query(self, question: str) -> Tuple[str, float]:
        """
        Answer query using updated belief state and experience.

        Args:
            question: Question to answer

        Returns:
            Tuple of (answer, confidence)
        """
        prompt = ACTOR_QUERY_TEMPLATE.format(
            belief_state=self._serialize_belief(),
            memory_summary=format_observation_history(self.memory, max_steps=10),
            question=question
        )

        response = self.llm.generate(prompt)
        answer, confidence, reasoning = extract_answer_components(response)

        return answer, confidence

    def reset(self):
        """Reset agent state for new episode"""
        super().reset()
        # Note: belief_state is NOT reset - it persists across episodes
        # This allows learning across multiple runs

    def get_belief_state(self) -> dict:
        """
        Get current belief state as dictionary.

        Returns:
            Belief state dictionary
        """
        return self._serialize_belief()

    def compute_surprisal(self, observation: dict) -> float:
        """
        Public method to compute surprisal on an observation.

        Args:
            observation: Environment observation

        Returns:
            Surprisal value
        """
        return self._compute_surprisal(observation)

    def update_belief_from_observation(self, observation: dict):
        """
        Public method to update belief based on observation.

        Args:
            observation: Environment observation
        """
        time_elapsed = observation.get('time', observation.get('time_elapsed', 0))
        self._update_belief(observation, time_elapsed)

    # Private helper methods

    def _compute_surprisal(self, observation: dict) -> float:
        """
        Compute surprisal from observation given current belief.

        Surprisal = -log P(observation | belief)

        Args:
            observation: Environment observation

        Returns:
            Surprisal value (higher = more surprising)
        """
        if not self.belief_state:
            return 0.0

        try:
            time_elapsed = observation.get('time', observation.get('time_elapsed', 0))

            # Different belief types have different signatures
            if hasattr(self.belief_state, 'log_likelihood'):
                # Try with time parameter first
                try:
                    log_likelihood = self.belief_state.log_likelihood(observation, time_elapsed)
                except TypeError:
                    # Fallback: try without time parameter
                    log_likelihood = self.belief_state.log_likelihood(observation)

                return -log_likelihood

        except Exception as e:
            print(f"Warning: Failed to compute surprisal: {e}")

        return 0.0

    def _parse_belief_update(self, llm_response: str) -> dict:
        """Extract JSON from LLM response with robust parsing."""

        # Try 1: Direct JSON parse
        try:
            return json.loads(llm_response)
        except json.JSONDecodeError:
            pass

        # Try 2: Extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try 3: Find first {...} block
        json_match = re.search(r'\{[^{}]*\}', llm_response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Try 4: Replace single quotes with double quotes (common issue)
        try:
            # Simple replacement - works for most cases
            fixed = llm_response.replace("'", '"')
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # All parsing failed - return None and log
        print(f"ERROR: Could not parse belief update from: {llm_response}")
        return None

    def _update_belief(self, observation: dict, time_elapsed: float):
        """
        Update belief parameters based on observation.

        First tries programmatic update() method if available (for deterministic updates),
        then optionally uses LLM for more sophisticated reasoning.

        Args:
            observation: New observation
            time_elapsed: Time since episode start
        """
        if not self.belief_state:
            return

        # First, try programmatic update if belief has an update() method
        if hasattr(self.belief_state, 'update') and callable(self.belief_state.update):
            try:
                # Call the belief's update method
                self.belief_state = self.belief_state.update(observation, time_elapsed)
                # Programmatic update succeeded - no need for LLM update
                return
            except Exception as e:
                print(f"Warning: Programmatic belief update failed: {e}")
                # Fall through to LLM-based update

        # Fallback to LLM-based update for beliefs without programmatic update
        prompt = BELIEF_UPDATE_TEMPLATE.format(
            current_belief=self._serialize_belief(),
            observation=str(observation),
            time_elapsed=time_elapsed,
            memory_summary=format_observation_history(self.memory, max_steps=3)
        )

        try:
            response = self.llm.generate(prompt, temperature=0.7)

            # Parse with robust error handling
            parsed_belief = self._parse_belief_update(response)

            if parsed_belief is not None:
                # Update succeeded - merge with existing parameters
                current_params = self.belief_state.model_dump()
                current_params.update(parsed_belief)

                # Create new belief instance with updated params
                self.belief_state = type(self.belief_state)(**current_params)
            else:
                # Update failed - keep old belief
                print(f"Warning: Using previous belief state due to parse failure")

        except Exception as e:
            # Don't crash on belief update failure
            print(f"Warning: Belief update failed: {e}")

    def _choose_action(self, observation: dict) -> Tuple[str, Optional[str]]:
        """
        Decide next action based on belief and observations.

        Args:
            observation: Current observation

        Returns:
            Tuple of (thought, action_string)
        """
        # Get tool descriptions
        if self.tools_class and hasattr(self.tools_class, 'get_tool_descriptions'):
            available_tools = self.tools_class.get_tool_descriptions()
        else:
            available_tools = "No tools available"

        prompt = ACTOR_ACTION_TEMPLATE.format(
            belief_state=self._serialize_belief(),
            observation=str(observation),
            memory_summary=format_observation_history(self.memory, max_steps=3),
            available_tools=available_tools,
            actions_remaining=self.action_budget - self.action_count
        )

        response = self.llm.generate(prompt, temperature=0.8)

        # Extract thought and action from response
        thought = extract_thought(response)
        action = extract_action(response)

        return thought, action

    def _serialize_belief(self) -> dict:
        """
        Serialize belief state to dictionary.

        Returns:
            Belief state as dict
        """
        if not self.belief_state:
            return {}

        try:
            return self.belief_state.model_dump()
        except AttributeError:
            # Fallback for non-pydantic beliefs
            return {'belief': str(self.belief_state)}
