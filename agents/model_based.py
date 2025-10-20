# agents/model_based.py
from typing import Tuple, Optional
from agents.actor import ActorAgent
from agents.base import AgentStep, LLMInterface
from models.transition_model import SimpleTransitionModel
from models.tools import get_tools_for_environment
from experiments.prompts import MODEL_BASED_PLANNING_TEMPLATE, extract_thought, extract_action
import time


class ModelBasedAgent(ActorAgent):
    """
    Actor + explicit learned transition model.

    ModelBased agents maintain both a parametric belief state (like Actor)
    and an explicit learned dynamics model. They use the model for
    planning by simulating potential actions.
    """

    def __init__(
        self,
        llm: LLMInterface,
        action_budget: int,
        environment_name: Optional[str] = None
    ):
        """
        Initialize ModelBased agent.

        Args:
            llm: LLM interface
            action_budget: Maximum number of actions
            environment_name: Environment name for tools
        """
        super().__init__(llm, action_budget, environment_name)

        self.transition_model: Optional[SimpleTransitionModel] = None
        self.model_fitted = False
        self.min_trajectories = 3  # Minimum data before fitting

    def act(self, observation: dict) -> AgentStep:
        """
        Act with model-based planning.

        Steps:
        1. Update belief (like Actor)
        2. Fit/update transition model if enough data
        3. Use model for planning
        4. Execute planned action

        Args:
            observation: Environment observation

        Returns:
            AgentStep with action and metadata
        """
        # Standard belief update from Actor
        step = super().act(observation)

        # Fit model after collecting enough trajectories
        if len(self.memory) >= self.min_trajectories and not self.model_fitted:
            self._fit_transition_model()
            self.model_fitted = True

        # Re-plan using model if available and budget allows
        if self.transition_model and self.action_count < self.action_budget:
            thought, action = self._plan_with_model(observation)

            # Update the step with model-based decision
            step.thought = f"Model-based: {thought}"
            step.action = action

        return step

    def _fit_transition_model(self):
        """
        Fit MLP transition model to collected trajectories.

        Extracts (state, action, next_state) tuples from memory
        and trains the model.
        """
        # Extract trajectories from memory
        trajectories = []

        for i in range(len(self.memory) - 1):
            curr_step = self.memory[i]
            next_step = self.memory[i + 1]

            # Only include steps where an action was taken
            if curr_step.action:
                trajectories.append((
                    curr_step.observation,
                    curr_step.action,
                    next_step.observation
                ))

        if not trajectories:
            print("Warning: No valid trajectories to fit model")
            return

        # Initialize model with inferred dimensions
        state_dim = self._get_state_dim()
        action_dim = self._get_action_dim()

        self.transition_model = SimpleTransitionModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=32
        )

        # Fit model to trajectories
        try:
            self.transition_model.fit(trajectories, epochs=50, verbose=False)
            print(f"Fitted transition model on {len(trajectories)} trajectories")
        except Exception as e:
            print(f"Warning: Model fitting failed: {e}")
            self.transition_model = None

    def _plan_with_model(self, current_obs: dict) -> Tuple[str, Optional[str]]:
        """
        Use transition model for 1-step lookahead planning.

        Simulates each possible action and chooses the one with
        highest expected information gain.

        Args:
            current_obs: Current observation

        Returns:
            Tuple of (thought, action)
        """
        if not self.transition_model:
            # Fallback to Actor's planning
            return super()._choose_action(current_obs)

        # Get possible actions from tools
        try:
            tools = get_tools_for_environment(self.environment_name)
            possible_actions = self._get_possible_actions(tools)
        except Exception:
            # Fallback if tools not available
            return super()._choose_action(current_obs)

        # Evaluate each action
        best_action = None
        best_info_gain = -float('inf')
        action_predictions = []

        for action_name in possible_actions[:5]:  # Limit to 5 actions for efficiency
            # Predict next state
            try:
                predicted_obs = self.transition_model.predict(current_obs, action_name)

                # Compute expected information gain
                info_gain = self._compute_info_gain(predicted_obs)

                action_predictions.append({
                    'action': action_name,
                    'predicted_obs': predicted_obs,
                    'info_gain': info_gain
                })

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_action = action_name

            except Exception as e:
                print(f"Warning: Failed to evaluate action {action_name}: {e}")
                continue

        # If no valid action found, fallback
        if best_action is None:
            return super()._choose_action(current_obs)

        thought = f"Model predicts best info gain: {best_info_gain:.2f}"

        return thought, best_action

    def _compute_info_gain(self, predicted_obs: dict) -> float:
        """
        Estimate information gain from predicted observation.

        Higher surprisal under current belief = more information.

        Args:
            predicted_obs: Predicted next observation

        Returns:
            Information gain estimate
        """
        if not self.belief_state:
            # Default: prefer diverse observations
            return sum(v for v in predicted_obs.values() if isinstance(v, (int, float)))

        # Compute expected surprisal
        try:
            surprisal = self._compute_surprisal(predicted_obs)
            return surprisal  # Higher surprisal = more informative
        except Exception:
            return 1.0  # Default moderate value

    def _get_possible_actions(self, tools_class) -> list[str]:
        """
        Extract list of possible action names from tools class.

        Args:
            tools_class: Tools class with static methods

        Returns:
            List of action names (e.g., ['measure_temp', 'wait'])
        """
        actions = []

        for name in dir(tools_class):
            if not name.startswith('_') and name != 'get_tool_descriptions':
                # Add with empty parens for simple actions
                actions.append(f"{name}()")

        return actions

    def _get_state_dim(self) -> int:
        """
        Infer state dimensionality from observations.

        Returns:
            State dimension
        """
        if not self.memory:
            return 10  # Default

        # Count numerical features in first observation
        sample_obs = self.memory[0].observation
        dim = sum(
            1 for v in sample_obs.values()
            if isinstance(v, (int, float, bool))
        )

        return max(dim, 5)  # At least 5 dimensions

    def _get_action_dim(self) -> int:
        """
        Infer action dimensionality.

        Returns:
            Action dimension
        """
        # Simple one-hot encoding for common actions
        return 8  # Enough for most environments

    def reset(self):
        """Reset agent state for new episode"""
        super().reset()
        # Note: transition_model is NOT reset - it persists for continual learning
