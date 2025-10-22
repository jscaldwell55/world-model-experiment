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
    HOTPOT_PRIOR_GENERATION_TEMPLATE,
    SWITCHLIGHT_PRIOR_GENERATION_TEMPLATE,
    CHEMTILE_PRIOR_GENERATION_TEMPLATE,
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

    def reset(
        self,
        environment_type: Optional[str] = None,
        initial_observation: Optional[dict] = None
    ):
        """
        Reset agent state for new episode.

        Args:
            environment_type: Type of environment (e.g., 'HotPotLab')
            initial_observation: Initial observation to generate priors from

        Note: If environment_type and initial_observation are provided,
              generates new priors. Otherwise, belief_state persists.
        """
        super().reset()

        # Store prior generation metadata for logging
        self.prior_generation_metadata = None

        # Generate new priors if environment type and observation are provided
        if environment_type and initial_observation and self.belief_state:
            try:
                # Generate priors using LLM
                priors, reasoning, token_count = self._generate_priors(
                    initial_observation,
                    environment_type
                )

                # Create new belief state with generated priors
                from models.belief_state import HotPotBelief, SwitchLightBelief, ChemTileBelief

                belief_mapping = {
                    'HotPotLab': HotPotBelief,
                    'SwitchLight': SwitchLightBelief,
                    'ChemTile': ChemTileBelief
                }

                if environment_type in belief_mapping:
                    belief_class = belief_mapping[environment_type]

                    # Initialize belief with generated priors
                    # For HotPotBelief, we need base_temp from observation or default
                    if environment_type == 'HotPotLab':
                        # Set base_temp to measured temp if available, else default
                        base_temp = initial_observation.get('measured_temp', 20.0)
                        self.belief_state = belief_class(
                            **priors,
                            base_temp=base_temp
                        )
                    else:
                        self.belief_state = belief_class(**priors)

                    # Store metadata for logging
                    self.prior_generation_metadata = {
                        'priors': priors,
                        'reasoning': reasoning,
                        'token_count': token_count,
                        'environment_type': environment_type
                    }

                    print(f"Initialized {environment_type} belief with LLM-generated priors")

            except Exception as e:
                print(f"Warning: Failed to generate priors: {e}")
                print(f"Keeping existing belief state")
                # belief_state remains unchanged

        # Note: If no environment_type provided, belief_state persists across episodes
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

    def _validate_priors(self, priors: dict, environment_type: str) -> bool:
        """
        Validate that generated priors are within reasonable ranges.

        Args:
            priors: Generated prior parameters
            environment_type: Type of environment (HotPotLab, SwitchLight, ChemTile)

        Returns:
            True if valid, raises ValueError if invalid

        Raises:
            ValueError: If priors are invalid with clear error message
        """
        if environment_type == "HotPotLab":
            # Validate heating_rate_mean
            if 'heating_rate_mean' not in priors:
                raise ValueError("Missing required parameter: heating_rate_mean")
            if not (-5.0 <= priors['heating_rate_mean'] <= 5.0):
                raise ValueError(
                    f"heating_rate_mean={priors['heating_rate_mean']} "
                    f"out of range [-5.0, 5.0]"
                )

            # Validate heating_rate_std
            if 'heating_rate_std' not in priors:
                raise ValueError("Missing required parameter: heating_rate_std")
            if not (0.1 <= priors['heating_rate_std'] <= 10.0):
                raise ValueError(
                    f"heating_rate_std={priors['heating_rate_std']} "
                    f"out of range [0.1, 10.0]"
                )

            # Validate measurement_noise
            if 'measurement_noise' not in priors:
                raise ValueError("Missing required parameter: measurement_noise")
            if not (0.1 <= priors['measurement_noise'] <= 5.0):
                raise ValueError(
                    f"measurement_noise={priors['measurement_noise']} "
                    f"out of range [0.1, 5.0]"
                )

        elif environment_type == "SwitchLight":
            # Validate connection_probs
            if 'connection_probs' not in priors:
                raise ValueError("Missing required parameter: connection_probs")

            conn_probs = priors['connection_probs']
            if not isinstance(conn_probs, list) or len(conn_probs) != 2:
                raise ValueError("connection_probs must be 2x2 matrix")

            for i, row in enumerate(conn_probs):
                if not isinstance(row, list) or len(row) != 2:
                    raise ValueError(f"connection_probs row {i} must have 2 elements")
                for j, prob in enumerate(row):
                    if not (0.0 <= prob <= 1.0):
                        raise ValueError(
                            f"connection_probs[{i}][{j}]={prob} "
                            f"out of range [0.0, 1.0]"
                        )

            # Validate uncertainty
            if 'uncertainty' not in priors:
                raise ValueError("Missing required parameter: uncertainty")
            if not (0.0 <= priors['uncertainty'] <= 1.0):
                raise ValueError(
                    f"uncertainty={priors['uncertainty']} "
                    f"out of range [0.0, 1.0]"
                )

        elif environment_type == "ChemTile":
            # Validate reaction_safety_priors
            if 'reaction_safety_priors' in priors:
                for compound, safety in priors['reaction_safety_priors'].items():
                    if not (0.0 <= safety <= 1.0):
                        raise ValueError(
                            f"reaction_safety_priors[{compound}]={safety} "
                            f"out of range [0.0, 1.0]"
                        )

            # Validate reaction_outcome_uncertainty
            if 'reaction_outcome_uncertainty' in priors:
                if not (0.0 <= priors['reaction_outcome_uncertainty'] <= 1.0):
                    raise ValueError(
                        f"reaction_outcome_uncertainty={priors['reaction_outcome_uncertainty']} "
                        f"out of range [0.0, 1.0]"
                    )

            # Validate temperature_effect_prior
            if 'temperature_effect_prior' in priors:
                if not (0.0 <= priors['temperature_effect_prior'] <= 1.0):
                    raise ValueError(
                        f"temperature_effect_prior={priors['temperature_effect_prior']} "
                        f"out of range [0.0, 1.0]"
                    )

        return True

    def _generate_priors(
        self,
        initial_observation: dict,
        environment_type: str
    ) -> Tuple[dict, str, int]:
        """
        Generate prior beliefs using LLM based on initial observation.

        Args:
            initial_observation: Initial observation from environment
            environment_type: Type of environment (HotPotLab, SwitchLight, ChemTile)

        Returns:
            Tuple of (priors_dict, reasoning, token_count)

        Raises:
            ValueError: If LLM fails to generate valid priors after retry
        """
        # Select appropriate prompt template
        prompt_templates = {
            'HotPotLab': HOTPOT_PRIOR_GENERATION_TEMPLATE,
            'SwitchLight': SWITCHLIGHT_PRIOR_GENERATION_TEMPLATE,
            'ChemTile': CHEMTILE_PRIOR_GENERATION_TEMPLATE
        }

        if environment_type not in prompt_templates:
            raise ValueError(f"Unknown environment type: {environment_type}")

        template = prompt_templates[environment_type]

        # Format prompt with initial observation
        prompt = template.format(
            initial_observation=json.dumps(initial_observation, indent=2)
        )

        # Try to generate priors (with one retry on failure)
        max_attempts = 2
        last_error = None

        for attempt in range(max_attempts):
            try:
                # Generate with temperature=0 for consistency
                response = self.llm.generate(prompt, temperature=0.0)

                # Parse response (reuse robust parsing from belief updates)
                parsed = self._parse_belief_update(response)

                if parsed is None:
                    raise ValueError(f"Failed to parse LLM response: {response}")

                # Extract reasoning
                reasoning = parsed.pop('reasoning', 'No reasoning provided')

                # Validate priors
                self._validate_priors(parsed, environment_type)

                # Success! Return priors, reasoning, and token estimate
                token_count = len(response.split())  # Rough estimate
                print(f"Generated priors for {environment_type}: {parsed}")
                print(f"Reasoning: {reasoning}")

                return parsed, reasoning, token_count

            except Exception as e:
                last_error = e
                print(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")

                if attempt < max_attempts - 1:
                    # Modify prompt to emphasize constraints
                    prompt = prompt + f"\n\nPREVIOUS ATTEMPT FAILED: {e}\nPlease ensure your response is valid JSON with values in the specified ranges."

        # All attempts failed - fall back to uninformative priors
        print(f"WARNING: Prior generation failed after {max_attempts} attempts: {last_error}")
        print(f"Falling back to uninformative (high uncertainty) priors")

        # Return uninformative priors based on environment type
        if environment_type == "HotPotLab":
            fallback_priors = {
                'heating_rate_mean': 0.0,  # No prior knowledge
                'heating_rate_std': 5.0,    # High uncertainty
                'measurement_noise': 2.0    # Moderate noise assumption
            }
            reasoning = f"Failed to generate priors (error: {last_error}). Using uninformative defaults."
        elif environment_type == "SwitchLight":
            fallback_priors = {
                'connection_probs': [[0.5, 0.5], [0.5, 0.5]],  # Uniform
                'uncertainty': 0.9  # High uncertainty
            }
            reasoning = f"Failed to generate priors (error: {last_error}). Using uniform defaults."
        elif environment_type == "ChemTile":
            fallback_priors = {
                'reaction_safety_priors': {},
                'reaction_outcome_uncertainty': 0.8,
                'temperature_effect_prior': 0.5
            }
            reasoning = f"Failed to generate priors (error: {last_error}). Using cautious defaults."
        else:
            fallback_priors = {}
            reasoning = f"Failed to generate priors (error: {last_error}). No fallback available."

        return fallback_priors, reasoning, 0
