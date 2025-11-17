# agents/simple_world_model.py
"""
Simple World Model Agent - Minimal Viable World Model

Evolved from ACTOR agent with persistent beliefs that accumulate across episodes.
Key improvements over ACTOR:
- Persistent belief states (don't reset between episodes)
- Observation history accumulation
- Statistical tracking for noise filtering
- Causal relationship learning
- Confidence-based exploration

Removes all hybrid complexity while maintaining ACTOR's probabilistic foundation.
"""

import time
import json
import re
import numpy as np
from typing import Tuple, Optional, Any, List, Dict
from collections import defaultdict
from datetime import datetime

from agents.base import Agent, AgentStep, LLMInterface
from memory.ace_playbook import ACEPlaybook
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
from utils.token_accounting import TokenAccountant


# Utility functions for query type detection
def is_counterfactual_question(question: str) -> bool:
    """Detect if question is counterfactual (asks about alternative past scenarios)"""
    markers = [
        "if we had", "would have", "had we", "suppose we had",
        "what if we had", "could we have", "should we have"
    ]
    return any(marker in question.lower() for marker in markers)


def detect_query_type(question: str) -> str:
    """
    Detect query type: counterfactual, interventional, planning, or observational.

    Returns:
        'counterfactual', 'interventional', 'planning', or 'observational'
    """
    q_lower = question.lower()

    # Counterfactual: past hypotheticals
    if is_counterfactual_question(question):
        return 'counterfactual'

    # Interventional: future predictions with actions
    interventional_markers = ["if we", "what if", "will it", "would it"]
    if any(marker in q_lower for marker in interventional_markers):
        return 'interventional'

    # Planning: multi-step action sequences
    planning_markers = ["how to", "what steps", "plan", "sequence", "procedure"]
    if any(marker in q_lower for marker in planning_markers):
        return 'planning'

    # Default: observational
    return 'observational'


class SimpleWorldModel(Agent):
    """
    Simple World Model - extends ACTOR with persistent beliefs and statistical learning.

    Key differences from ACTOR:
    1. Beliefs persist across episodes (accumulate knowledge)
    2. Tracks observation statistics for noise filtering
    3. Learns causal action-outcome relationships
    4. Implements confidence-based exploration
    5. Maintains full observation history
    """

    def __init__(
        self,
        llm: LLMInterface,
        action_budget: int,
        environment_name: Optional[str] = None,
        prior_strength: float = 0.1,
        exploration_temperature: float = 0.1,
        confidence_threshold: float = 0.7,
        enable_persistence: bool = True
    ):
        """
        Initialize Simple World Model.

        Args:
            llm: LLM interface
            action_budget: Maximum number of actions allowed
            environment_name: Name of environment (for tool selection)
            prior_strength: Strength of prior beliefs (lower = more adaptable)
            exploration_temperature: Temperature for exploration bonus
            confidence_threshold: Minimum confidence to exploit vs explore
            enable_persistence: If True, beliefs persist across episodes
        """
        super().__init__(llm, action_budget)

        self.belief_state = None
        self.environment_name = environment_name
        self.tools_class = None
        self.token_accountant = TokenAccountant()

        # World model parameters
        self.prior_strength = prior_strength
        self.exploration_temperature = exploration_temperature
        self.confidence_threshold = confidence_threshold
        self.enable_persistence = enable_persistence

        # NEW: Persistent belief components that accumulate across episodes
        self.observation_history: List[Dict] = []  # Full history across all episodes
        self.episode_step = 0  # Current step within episode
        self.total_steps = 0  # Total steps across all episodes

        # NEW: Statistical tracking for noise filtering (fixes HotPot)
        self.observation_statistics: Dict[str, Dict] = defaultdict(lambda: {
            'values': [],
            'mean': 0.0,
            'variance': 0.0,
            'n_obs': 0
        })

        # NEW: Causal belief tracking (action -> outcome relationships)
        self.causal_relationships: Dict[Tuple, Dict] = defaultdict(lambda: {
            'success': 0,
            'total': 0,
            'outcomes': []
        })

        # NEW: Action outcome history
        self.action_outcomes: Dict[str, Dict] = defaultdict(lambda: {
            'success': 0,
            'total': 0,
            'rewards': []
        })

        # Environment-specific parameters learned over time
        self.environment_parameters: Dict[str, Any] = {}

        # NEW: ACE-based persistent memory (replaces consolidation)
        self.ace_playbook = None  # Will be initialized in start_episode
        self.current_domain = None
        self.episode_id = None

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
        Choose next action based on current observation and accumulated beliefs.

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

        # Create step record
        step = AgentStep(
            timestamp=time.time(),
            step_num=len(self.memory),
            thought=thought,
            action=action,
            observation=observation,
            belief_state=self._serialize_belief(),
            surprisal=0.0,
            token_usage=0
        )

        self.memory.append(step)
        return step

    def answer_query(self, question: str) -> Tuple[str, float]:
        """
        Answer query using accumulated beliefs and experience.

        Uses query-type-specific prompts and accumulated observation history.

        Args:
            question: Question to answer

        Returns:
            Tuple of (answer, confidence)
        """
        # Detect query type for confidence calibration
        query_type = detect_query_type(question)

        # Use standard ACTOR query template for all types
        template = ACTOR_QUERY_TEMPLATE

        # Build prompt with accumulated history (not just recent steps)
        belief_context = self._serialize_belief_with_statistics()

        prompt = template.format(
            belief_state=belief_context,
            memory_summary=format_observation_history(self.memory, max_steps=10),
            question=question
        )

        response = self.llm.generate(prompt, temperature=0.0)

        # Record token usage for evaluation
        input_tokens, output_tokens = self.llm.get_last_usage()
        self.token_accountant.record(
            'evaluation',
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={
                'question': question[:50],
                'query_type': query_type
            }
        )

        answer, confidence, reasoning = extract_answer_components(response)

        # Detect difficulty (heuristic based on question length and complexity markers)
        difficulty = self._estimate_question_difficulty(question)

        # Counterfactual confidence calibration
        if is_counterfactual_question(question):
            if confidence > 0.85:
                confidence = 0.85

            # Check for overconfident language
            overconfident_words = ["definitely", "certainly", "absolutely", "always", "never"]
            if any(word in answer.lower() for word in overconfident_words):
                confidence *= 0.8

        # NEW V2: Medium interventional question calibration
        if query_type == 'interventional' and difficulty == 'medium':
            answer, confidence = self._calibrate_interventional_answer(answer, confidence, question)

        # SwitchLight uncertainty enhancement
        if self.environment_name == "SwitchLight" and is_counterfactual_question(question):
            answer = self._enhance_switchlight_uncertainty(answer, question)

        return answer, confidence

    def _enhance_switchlight_uncertainty(self, answer: str, question: str) -> str:
        """
        Add uncertainty markers for SwitchLight stochastic scenarios.

        Args:
            answer: Original answer
            question: Question text

        Returns:
            Enhanced answer with appropriate uncertainty
        """
        uncertainty_triggers = ["relay", "jiggle", "faulty"]

        if any(trigger in question.lower() for trigger in uncertainty_triggers):
            uncertainty_words = ["possibly", "might", "maybe", "could", "depends", "uncertain"]

            if not any(word in answer.lower() for word in uncertainty_words):
                if answer.lower().startswith("yes"):
                    answer = "Possibly yes, " + answer[3:].lstrip()
                elif answer.lower().startswith("no"):
                    answer = "Possibly no, " + answer[2:].lstrip()
                else:
                    answer = "It depends. " + answer

        return answer

    def _estimate_question_difficulty(self, question: str) -> str:
        """
        Estimate question difficulty heuristically.

        NEW V2: Used for confidence calibration.

        Args:
            question: Question text

        Returns:
            'easy', 'medium', or 'hard'
        """
        # Heuristic markers
        hard_markers = [
            "if we had", "would have", "probability", "overall",
            "calculate", "exactly", "precisely"
        ]
        easy_markers = [
            "is it safe", "will it", "should we", "what happens if"
        ]

        question_lower = question.lower()

        # Check for hard markers
        if any(marker in question_lower for marker in hard_markers):
            return 'hard'

        # Check for easy markers
        if any(marker in question_lower for marker in easy_markers):
            return 'easy'

        # Default to medium
        return 'medium'

    def _calibrate_interventional_answer(self, answer: str, confidence: float,
                                        question: str) -> Tuple[str, float]:
        """
        Calibrate confidence for medium interventional questions.

        NEW V2: Addresses 40-57% accuracy issue on medium interventional Qs.

        Args:
            answer: Original answer
            confidence: Original confidence
            question: Question text

        Returns:
            Tuple of (calibrated_answer, calibrated_confidence)
        """
        # Get current belief uncertainty
        avg_confidence = self._compute_average_confidence()

        # If model is uncertain, express it in the answer
        if confidence < 0.7 or avg_confidence < 0.7:
            # Add uncertainty quantification
            uncertainty_phrases = ["likely", "probably", "approximately", "around"]

            # Check if answer already has uncertainty markers
            has_uncertainty = any(phrase in answer.lower() for phrase in uncertainty_phrases)

            if not has_uncertainty and not any(word in answer.lower() for word in
                                             ["definitely", "certainly", "exactly"]):
                # Add "likely" or "probably" prefix
                if question.lower().startswith("will"):
                    answer = f"Likely {answer[0].lower()}{answer[1:]}"
                elif question.lower().startswith("what"):
                    answer = f"Probably {answer[0].lower()}{answer[1:]}"
                else:
                    answer = f"Likely: {answer}"

            # Reduce confidence to reflect uncertainty
            confidence = min(confidence, 0.75)

        elif confidence < 0.9:
            # Moderate confidence - add "probably"
            if not any(word in answer.lower() for word in
                      ["probably", "likely", "possibly", "definitely", "certainly"]):
                answer = f"Probably {answer[0].lower()}{answer[1:]}"

        # Quantitative questions need ranges
        if any(word in question.lower() for word in
              ["temperature", "how many", "how much", "what percent"]):
            # Check if answer has a number
            import re
            numbers = re.findall(r'\d+\.?\d*', answer)
            if numbers and confidence < 0.85:
                # Add uncertainty range if not present
                if '±' not in answer and 'range' not in answer.lower():
                    # Estimate uncertainty from model std
                    if hasattr(self.belief_state, 'heating_rate_std'):
                        std = self.belief_state.heating_rate_std
                        answer += f" (±{std:.1f}°C uncertainty)"

        return answer, confidence

    def reset(
        self,
        environment_type: Optional[str] = None,
        initial_observation: Optional[dict] = None
    ):
        """
        Reset agent state for new episode.

        KEY DIFFERENCE FROM ACTOR: If enable_persistence=True, beliefs persist!

        Args:
            environment_type: Type of environment (e.g., 'HotPotLab')
            initial_observation: Initial observation to generate priors from
        """
        # Reset per-episode counters
        self.action_count = 0
        self.memory = []
        self.episode_step = 0

        # Reset token accounting for new episode
        self.token_accountant.reset()

        # If persistence is disabled, clear accumulated data
        if not self.enable_persistence:
            self.observation_history = []
            self.observation_statistics = defaultdict(lambda: {
                'values': [],
                'mean': 0.0,
                'variance': 0.0,
                'n_obs': 0
            })
            self.causal_relationships = defaultdict(lambda: {
                'success': 0,
                'total': 0,
                'outcomes': []
            })
            self.action_outcomes = defaultdict(lambda: {
                'success': 0,
                'total': 0,
                'rewards': []
            })
            self.total_steps = 0

        # Generate new priors if this is the first episode or persistence is disabled
        if environment_type and initial_observation:
            if not self.enable_persistence or self.belief_state is None:
                # First episode or non-persistent mode: generate fresh priors
                try:
                    priors, reasoning, token_count = self._generate_priors(
                        initial_observation,
                        environment_type
                    )

                    from models.belief_state import HotPotBelief, SwitchLightBelief, ChemTileBelief

                    belief_mapping = {
                        'HotPotLab': HotPotBelief,
                        'SwitchLight': SwitchLightBelief,
                        'ChemTile': ChemTileBelief
                    }

                    if environment_type in belief_mapping:
                        belief_class = belief_mapping[environment_type]

                        if environment_type == 'HotPotLab':
                            base_temp = initial_observation.get('measured_temp', 20.0)
                            self.belief_state = belief_class(
                                **priors,
                                base_temp=base_temp
                            )
                        else:
                            self.belief_state = belief_class(**priors)

                        print(f"Initialized {environment_type} belief with LLM-generated priors")

                except Exception as e:
                    print(f"Warning: Failed to generate priors: {e}")
            else:
                # Persistent mode: beliefs carry over from previous episodes
                print(f"Using persistent beliefs from previous episodes (total_steps={self.total_steps})")

    def get_belief_state(self) -> dict:
        """
        Get current belief state with statistics.

        Returns:
            Belief state dictionary with accumulated statistics
        """
        return self._serialize_belief_with_statistics()

    def compute_surprisal(self, observation: dict) -> float:
        """
        Compute surprisal on an observation.

        Args:
            observation: Environment observation

        Returns:
            Surprisal value
        """
        return self._compute_surprisal(observation)

    def update_belief_from_observation(self, observation: dict):
        """
        Update beliefs based on observation.

        NEW: Also updates statistical tracking and causal beliefs.

        Args:
            observation: Environment observation

        Note: Compatible with runner interface. Action and reward are extracted
              from memory if available.
        """
        time_elapsed = observation.get('time', observation.get('time_elapsed', 0))

        # Update core belief state (from ACTOR)
        self._update_belief(observation, time_elapsed)

        # NEW: Update accumulated statistics
        self._update_observation_statistics(observation)

        # NEW: Extract action and reward from last memory step for causal learning
        action = None
        reward = 0.0
        if self.memory and len(self.memory) > 0:
            last_step = self.memory[-1]
            action = last_step.action
            # Reward is typically in observation or we infer success from environment
            reward = observation.get('reward', 0.0)

            # Infer reward from environment signals if not explicit
            if reward == 0.0:
                # For HotPot: temperature change indicates progress
                if 'measured_temp' in observation and 'target_temp' in observation:
                    temp_diff = abs(observation['measured_temp'] - observation['target_temp'])
                    if temp_diff < 2.0:  # Close to target
                        reward = 1.0
                # For SwitchLight: bulb state matches expectation
                elif 'bulb_state' in observation:
                    reward = 0.5  # Neutral progress
                # For ChemTile: successful reaction
                elif 'reaction_result' in observation:
                    if observation['reaction_result'] != 'explosion':
                        reward = 1.0

        # NEW: Update causal beliefs if action extracted
        if action is not None:
            self._update_causal_beliefs(action, observation, reward)

        # NEW: Add to persistent observation history
        self.observation_history.append({
            'step': self.total_steps,
            'episode_step': self.episode_step,
            'observation': observation,
            'action': action,
            'reward': reward,
            'timestamp': time.time()
        })

        self.episode_step += 1
        self.total_steps += 1

    # ========================================================================
    # NEW: Statistical Tracking Methods (fixes HotPot noise issues)
    # ========================================================================

    def _update_observation_statistics(self, observation: dict):
        """
        Track running statistics for noise filtering.

        Critical for HotPot Lab where noise destroyed hybrid performance.

        Args:
            observation: New observation to incorporate
        """
        for key, value in observation.items():
            # Only track numeric values
            if not isinstance(value, (int, float)):
                continue

            stats = self.observation_statistics[key]
            stats['values'].append(value)
            stats['n_obs'] += 1

            # Incremental mean update (Welford's algorithm)
            delta = value - stats['mean']
            stats['mean'] += delta / stats['n_obs']

            # Update variance
            if stats['n_obs'] > 1:
                stats['variance'] = np.var(stats['values'])

    def _update_causal_beliefs(self, action: str, observation: dict, reward: float):
        """
        Learn action-outcome relationships over time.

        Args:
            action: Action taken
            observation: Resulting observation
            reward: Reward received
        """
        # Extract action type (e.g., "measure_temp" from "measure_temp()")
        action_type = action.split('(')[0] if '(' in action else action

        # Update action outcome tracking
        self.action_outcomes[action_type]['total'] += 1
        self.action_outcomes[action_type]['rewards'].append(reward)
        if reward > 0:
            self.action_outcomes[action_type]['success'] += 1

        # Track causal relationships (action + context -> outcome)
        # For now, simplified: just track action -> success rate
        outcome_category = 'success' if reward > 0 else 'failure'
        key = (action_type, outcome_category)

        self.causal_relationships[key]['total'] += 1
        if reward > 0:
            self.causal_relationships[key]['success'] += 1

        # Store outcome for pattern learning
        self.causal_relationships[key]['outcomes'].append({
            'observation': observation,
            'reward': reward
        })

    def _should_explore(self) -> bool:
        """
        Decide when to explore vs exploit based on confidence.

        Returns:
            True if should explore, False if should exploit
        """
        # Always explore initially
        if self.episode_step < 3:
            return True

        # Check uncertainty in key parameters
        avg_confidence = self._compute_average_confidence()

        # Explore if confidence is low
        return avg_confidence < self.confidence_threshold

    def _compute_average_confidence(self) -> float:
        """
        Compute average confidence across tracked statistics.

        Returns:
            Average confidence score [0, 1]
        """
        if not self.observation_statistics:
            return 0.0

        confidences = []
        for key, stats in self.observation_statistics.items():
            if stats['n_obs'] >= 3:
                # Confidence based on stability (low variance = high confidence)
                if stats['variance'] > 0:
                    # Normalize by coefficient of variation
                    cv = np.sqrt(stats['variance']) / (abs(stats['mean']) + 1e-6)
                    confidence = 1.0 / (1.0 + cv)
                    confidences.append(confidence)

        if not confidences:
            return 0.5  # Neutral if no data

        return np.mean(confidences)

    def _add_exploration_bonus(self, action_probs: Dict[str, float]) -> Dict[str, float]:
        """
        Add exploration bonus to action probabilities.

        Args:
            action_probs: Current action probability distribution

        Returns:
            Modified probabilities with exploration bonus
        """
        # Add temperature-based exploration
        total = sum(action_probs.values())
        if total > 0:
            normalized = {k: v / total for k, v in action_probs.items()}

            # Apply softmax with exploration temperature
            exp_probs = {
                k: np.exp(np.log(v + 1e-10) / self.exploration_temperature)
                for k, v in normalized.items()
            }

            total_exp = sum(exp_probs.values())
            return {k: v / total_exp for k, v in exp_probs.items()}

        return action_probs

    # ========================================================================
    # Core Methods (from ACTOR, mostly unchanged)
    # ========================================================================

    def _compute_surprisal(self, observation: dict) -> float:
        """
        Compute surprisal from observation given current belief.

        Args:
            observation: Environment observation

        Returns:
            Surprisal value (higher = more surprising)
        """
        if not self.belief_state:
            return 0.0

        try:
            time_elapsed = observation.get('time', observation.get('time_elapsed', 0))

            if hasattr(self.belief_state, 'log_likelihood'):
                try:
                    log_likelihood = self.belief_state.log_likelihood(observation, time_elapsed)
                except TypeError:
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

        # Try 4: Replace single quotes with double quotes
        try:
            fixed = llm_response.replace("'", '"')
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        print(f"ERROR: Could not parse belief update from: {llm_response}")
        return None

    def _update_belief(self, observation: dict, time_elapsed: float):
        """
        Update belief parameters based on observation.

        Args:
            observation: New observation
            time_elapsed: Time since episode start
        """
        if not self.belief_state:
            return

        # Try programmatic update if available
        if hasattr(self.belief_state, 'update') and callable(self.belief_state.update):
            try:
                self.belief_state = self.belief_state.update(observation, time_elapsed)
                return
            except Exception as e:
                print(f"Warning: Programmatic belief update failed: {e}")

        # Fallback to LLM-based update
        prompt = BELIEF_UPDATE_TEMPLATE.format(
            current_belief=self._serialize_belief(),
            observation=str(observation),
            time_elapsed=time_elapsed,
            memory_summary=format_observation_history(self.memory, max_steps=3)
        )

        try:
            response = self.llm.generate(prompt, temperature=0.7)

            input_tokens, output_tokens = self.llm.get_last_usage()
            self.token_accountant.record(
                'curation',
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata={'phase': 'belief_update'}
            )

            parsed_belief = self._parse_belief_update(response)

            if parsed_belief is not None:
                current_params = self.belief_state.model_dump()
                current_params.update(parsed_belief)
                self.belief_state = type(self.belief_state)(**current_params)
            else:
                print(f"Warning: Using previous belief state due to parse failure")

        except Exception as e:
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

        # Build context with accumulated statistics
        belief_context = self._serialize_belief_with_statistics()

        prompt = ACTOR_ACTION_TEMPLATE.format(
            belief_state=belief_context,
            observation=str(observation),
            memory_summary=format_observation_history(self.memory, max_steps=3),
            available_tools=available_tools,
            actions_remaining=self.action_budget - self.action_count
        )

        response = self.llm.generate(prompt, temperature=0.8)

        # Record token usage for exploration
        input_tokens, output_tokens = self.llm.get_last_usage()
        self.token_accountant.record(
            'exploration',
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={'action_count': self.action_count}
        )

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
            return {'belief': str(self.belief_state)}

    def _serialize_belief_with_statistics(self) -> dict:
        """
        Serialize belief state with accumulated statistics.

        NEW: Includes observation statistics and causal beliefs.

        Returns:
            Enhanced belief state dictionary
        """
        base_belief = self._serialize_belief()

        # Add accumulated statistics
        base_belief['_statistics'] = {
            'observation_stats': {
                key: {
                    'mean': stats['mean'],
                    'variance': stats['variance'],
                    'n_obs': stats['n_obs']
                }
                for key, stats in self.observation_statistics.items()
            },
            'action_outcomes': {
                action: {
                    'success_rate': outcomes['success'] / outcomes['total'] if outcomes['total'] > 0 else 0.0,
                    'total_trials': outcomes['total']
                }
                for action, outcomes in self.action_outcomes.items()
            },
            'total_steps': self.total_steps,
            'episode_step': self.episode_step,
            'average_confidence': self._compute_average_confidence()
        }

        return base_belief

    def _validate_priors(self, priors: dict, environment_type: str) -> bool:
        """Validate generated priors (same as ACTOR)"""
        if environment_type == "HotPotLab":
            if 'heating_rate_mean' not in priors:
                raise ValueError("Missing required parameter: heating_rate_mean")
            if not (-5.0 <= priors['heating_rate_mean'] <= 5.0):
                raise ValueError(
                    f"heating_rate_mean={priors['heating_rate_mean']} "
                    f"out of range [-5.0, 5.0]"
                )

            if 'heating_rate_std' not in priors:
                raise ValueError("Missing required parameter: heating_rate_std")
            if not (0.1 <= priors['heating_rate_std'] <= 10.0):
                raise ValueError(
                    f"heating_rate_std={priors['heating_rate_std']} "
                    f"out of range [0.1, 10.0]"
                )

            if 'measurement_noise' not in priors:
                raise ValueError("Missing required parameter: measurement_noise")
            if not (0.1 <= priors['measurement_noise'] <= 5.0):
                raise ValueError(
                    f"measurement_noise={priors['measurement_noise']} "
                    f"out of range [0.1, 5.0]"
                )

        elif environment_type == "SwitchLight":
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

            if 'uncertainty' not in priors:
                raise ValueError("Missing required parameter: uncertainty")
            if not (0.0 <= priors['uncertainty'] <= 1.0):
                raise ValueError(
                    f"uncertainty={priors['uncertainty']} "
                    f"out of range [0.0, 1.0]"
                )

        elif environment_type == "ChemTile":
            if 'reaction_safety_priors' in priors:
                for compound, safety in priors['reaction_safety_priors'].items():
                    if not (0.0 <= safety <= 1.0):
                        raise ValueError(
                            f"reaction_safety_priors[{compound}]={safety} "
                            f"out of range [0.0, 1.0]"
                        )

            if 'reaction_outcome_uncertainty' in priors:
                if not (0.0 <= priors['reaction_outcome_uncertainty'] <= 1.0):
                    raise ValueError(
                        f"reaction_outcome_uncertainty={priors['reaction_outcome_uncertainty']} "
                        f"out of range [0.0, 1.0]"
                    )

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
        """Generate prior beliefs using LLM (same as ACTOR)"""
        prompt_templates = {
            'HotPotLab': HOTPOT_PRIOR_GENERATION_TEMPLATE,
            'SwitchLight': SWITCHLIGHT_PRIOR_GENERATION_TEMPLATE,
            'ChemTile': CHEMTILE_PRIOR_GENERATION_TEMPLATE
        }

        if environment_type not in prompt_templates:
            raise ValueError(f"Unknown environment type: {environment_type}")

        template = prompt_templates[environment_type]

        prompt = template.format(
            initial_observation=json.dumps(initial_observation, indent=2)
        )

        max_attempts = 2
        last_error = None

        for attempt in range(max_attempts):
            try:
                response = self.llm.generate(prompt, temperature=0.0)

                input_tokens, output_tokens = self.llm.get_last_usage()
                self.token_accountant.record(
                    'planning',
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    metadata={'phase': 'prior_generation', 'environment': environment_type}
                )

                parsed = self._parse_belief_update(response)

                if parsed is None:
                    raise ValueError(f"Failed to parse LLM response: {response}")

                reasoning = parsed.pop('reasoning', 'No reasoning provided')

                self._validate_priors(parsed, environment_type)

                token_count = len(response.split())
                print(f"Generated priors for {environment_type}: {parsed}")
                print(f"Reasoning: {reasoning}")

                return parsed, reasoning, token_count

            except Exception as e:
                last_error = e
                print(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")

                if attempt < max_attempts - 1:
                    prompt = prompt + f"\n\nPREVIOUS ATTEMPT FAILED: {e}\nPlease ensure your response is valid JSON with values in the specified ranges."

        # Fallback to uninformative priors
        print(f"WARNING: Prior generation failed after {max_attempts} attempts: {last_error}")
        print(f"Falling back to uninformative (high uncertainty) priors")

        if environment_type == "HotPotLab":
            fallback_priors = {
                'heating_rate_mean': 1.5,
                'heating_rate_std': 0.3,
                'measurement_noise': 2.0
            }
            reasoning = f"Failed to generate priors (error: {last_error}). Using default physical priors."
        elif environment_type == "SwitchLight":
            fallback_priors = {
                'connection_probs': [[0.5, 0.5], [0.5, 0.5]],
                'uncertainty': 0.9
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

    # ========================================================================
    # Domain-Specific Memory Methods
    # ========================================================================

    def start_episode(self, environment: str):
        """
        Called at the beginning of each episode to load ACE context.

        ACE CHANGE: Instead of loading consolidated beliefs, we get natural language
        context with methodology warnings.

        Args:
            environment: Environment name (e.g., 'ChemTile', 'HotPotLab', 'SwitchLight')
        """
        print(f"\n{'='*70}")
        print(f"ACE: start_episode() for environment: {environment}")
        print(f"{'='*70}")

        self.current_domain = self._map_env_to_domain(environment)
        self.episode_id = f"{self.current_domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"ACE: Domain: {self.current_domain}")
        print(f"ACE: Episode ID: {self.episode_id}")

        # Initialize ACE playbook
        self.ace_playbook = ACEPlaybook(self.current_domain)

        # Get natural language context instead of consolidated beliefs
        context = self.ace_playbook.get_context(
            task_info={'environment': environment}
        )

        print(f"ACE: Generated context ({len(context)} chars)")
        print(f"ACE Context:\n{context}")

        # Parse context to initialize belief state (if we have one)
        if self.belief_state:
            self._initialize_from_ace_context(context)
            print(f"✓ Initialized beliefs from ACE context")
        else:
            print(f"⚠ No belief state to initialize")

        # KEEP prior_strength at 0.1 (ACE relies on weak priors + rich context)
        print(f"✓ Using prior_strength={self.prior_strength:.2f} (unchanged)")
        print(f"{'='*70}\n")

    def end_episode(self, final_score: float):
        """
        Called at the end of each episode to save learned beliefs.

        ACE CHANGE: Uses reflection → curation → playbook update instead of consolidation.

        Args:
            final_score: Episode performance score (0-100)
        """
        print(f"\n{'='*70}")
        print(f"ACE: end_episode() CALLED")
        print(f"ACE: Domain: {self.current_domain}")
        print(f"ACE: Episode ID: {self.episode_id}")
        print(f"ACE: Final score: {final_score:.1f}")
        print(f"{'='*70}")

        if not self.current_domain or not self.episode_id or not self.ace_playbook:
            print(f"⚠ WARNING: NOT SAVING - missing domain/episode/playbook")
            print(f"{'='*70}\n")
            return

        # Extract final beliefs
        final_beliefs = self._extract_key_beliefs()

        # Extract action sequence for methodology analysis
        actions = [step.action for step in self.memory if step.action]
        observations = [step.observation for step in self.memory if step.observation]

        # Build trajectory for ACE reflection
        trajectory = {
            'episode_id': self.episode_id,
            'observations': observations,
            'actions': actions,
            'final_beliefs': final_beliefs,
            'context_used': self.ace_playbook.current_context
        }

        # Build outcome
        outcome = {
            'score': final_score / 100.0,  # Normalize to 0-1
            'test_results': []  # Not available here, but ACE doesn't need it for core functionality
        }

        print(f"ACE: Running reflection...")
        # ACE Learning Pipeline
        # 1. Reflect: Analyze methodology and extract insights
        insights = self.ace_playbook.reflect(trajectory, outcome)
        print(f"ACE: Methodology quality = {insights['methodology_quality']}")
        print(f"ACE: Reason = {insights['reliability_reason']}")

        # 2. Curate: Generate delta updates
        deltas = self.ace_playbook.curate(insights, trajectory, outcome)
        print(f"ACE: Generated {len(deltas)} delta updates")

        # 3. Merge: Update playbook
        self.ace_playbook.merge_deltas(deltas)

        # 4. Save playbook and episode
        self.ace_playbook.save_playbook()
        self.ace_playbook.save_episode(self.episode_id, trajectory, outcome)

        print(f"✓ ACE: Saved episode {self.episode_id} (score: {final_score:.1f}, quality: {insights['methodology_quality']})")
        print(f"{'='*70}\n")

    def _map_env_to_domain(self, environment: str) -> str:
        """
        Map environment names to domain names.

        Args:
            environment: Environment name (various formats)

        Returns:
            Normalized domain name
        """
        mapping = {
            'ChemTile': 'chem_tile',
            'HotPotLab': 'hot_pot',
            'HotPot Lab': 'hot_pot',
            'SwitchLight': 'switch_light',
            'Switch Light': 'switch_light'
        }
        return mapping.get(environment, environment.lower().replace(' ', '_'))

    def _initialize_from_ace_context(self, context: str):
        """
        Initialize belief state from ACE natural language context.

        ACE CHANGE: Parse natural language context instead of structured beliefs.
        Uses simple heuristics to extract key values from context text.

        Args:
            context: Natural language context string from ACE
        """
        if not self.belief_state:
            return

        # Simple heuristic parsing of context
        # Look for patterns like "heating_rate ~2.5°C/s"

        if self.current_domain == 'hot_pot':
            # Extract heating rate from context
            import re
            heating_match = re.search(r'heating_rate ~?(\d+\.?\d*)°C/s', context)
            if heating_match:
                heating_rate = float(heating_match.group(1))
                # Use this as a hint, but keep uncertainty high
                self.belief_state.heating_rate_mean = heating_rate
                self.belief_state.heating_rate_std = 0.5  # Keep uncertainty high
                print(f"ACE: Parsed heating_rate ~{heating_rate}°C/s from context")
            else:
                print(f"ACE: No heating_rate found in context - using defaults")

        # For other domains, keep default initialization
        # The weak prior (0.1) means the agent will quickly adapt to observations

        print(f"ACE: Initialized with weak priors (prior_strength={self.prior_strength})")

    def _extract_key_beliefs(self) -> Dict:
        """
        Extract key beliefs to persist (domain-specific) with confidence tracking.

        Returns:
            Dictionary of key beliefs for the current domain
        """
        beliefs = {}

        if not self.belief_state:
            return beliefs

        try:
            # Get full belief state
            full_belief = self.belief_state.model_dump()

            # Helper function to wrap beliefs with confidence metadata
            def wrap_belief(value, obs_key: Optional[str] = None):
                """
                Wrap a belief value with confidence tracking.

                Args:
                    value: The belief value
                    obs_key: Optional observation key to compute confidence from statistics

                Returns:
                    Dict with value, confidence, episode_count, last_updated
                """
                # Calculate confidence based on observation statistics if available
                obs_confidence = 0.5  # Default moderate confidence
                n_obs = 0  # Total observations for this belief

                if obs_key and obs_key in self.observation_statistics:
                    stats = self.observation_statistics[obs_key]
                    n_obs = stats.get('n_obs', 0)
                    variance = stats.get('variance', float('inf'))

                    # Average confidence from all observations
                    avg_confidence = 0.5
                    for key, obs_stats in self.observation_statistics.items():
                        obs_n = obs_stats.get('n_obs', 0)
                        if obs_n > 0:
                            # Confidence increases with observations, decreases with variance
                            obs_var = obs_stats.get('variance', float('inf'))
                            if obs_var < float('inf') and obs_var > 0:
                                variance_confidence = 1.0 / (1.0 + obs_var / 100.0)
                                obs_confidence = min(0.95, 0.5 + (obs_n / 50.0) * 0.3)
                                obs_confidence = (obs_confidence + variance_confidence) / 2
                    else:
                        # Use average confidence if no specific stats
                        obs_confidence = avg_confidence

                # OBSERVATION MINIMUM: Reduce confidence if we have sparse data
                OBSERVATION_MINIMUM = 8
                if n_obs < OBSERVATION_MINIMUM and n_obs > 0:
                    penalty = 0.5  # Halve confidence for sparse data
                    obs_confidence *= penalty
                    print(f"  ⚠️ Only {n_obs} observations for {obs_key or 'belief'} (< {OBSERVATION_MINIMUM}) - reducing confidence to {obs_confidence:.3f}")

                return {
                    'value': value,
                    'confidence': float(obs_confidence),
                    'observation_count': int(n_obs) if n_obs > 0 else 1,  # For enhanced memory system
                    'episode_count': 1,  # Will be aggregated in consolidation
                    'last_updated': datetime.now().isoformat()
                }

            if self.current_domain == 'hot_pot':
                # Key beliefs for temperature dynamics with confidence tracking
                beliefs['heating_rate_mean'] = wrap_belief(
                    full_belief.get('heating_rate_mean', 1.5),
                    obs_key='measured_temp'
                )
                beliefs['heating_rate_std'] = wrap_belief(
                    full_belief.get('heating_rate_std', 0.3),
                    obs_key='measured_temp'
                )
                beliefs['measurement_noise'] = wrap_belief(
                    full_belief.get('measurement_noise', 2.0),
                    obs_key='measured_temp'
                )
                beliefs['base_temp'] = wrap_belief(
                    full_belief.get('base_temp', 20.0),
                    obs_key='measured_temp'
                )

            elif self.current_domain == 'chem_tile':
                # Key beliefs for chemistry
                reaction_probs = full_belief.get('reaction_probs', {
                    'A+B': {'C': 0.8, 'explode': 0.1, 'nothing': 0.1},
                    'C+B': {'D': 0.7, 'explode': 0.2, 'nothing': 0.1}
                })
                beliefs['reaction_probs'] = wrap_belief(reaction_probs)

                temperature = full_belief.get('temperature', 'medium')
                beliefs['temperature'] = wrap_belief(temperature)

            elif self.current_domain == 'switch_light':
                # Key beliefs for wiring
                wiring_probs = full_belief.get('wiring_probs', {'layout_A': 0.5, 'layout_B': 0.5})
                beliefs['wiring_probs'] = wrap_belief(wiring_probs, obs_key='light_on')

                failure_prob = full_belief.get('failure_prob', 0.1)
                beliefs['failure_prob'] = wrap_belief(failure_prob, obs_key='light_on')

            # NEW: Persist accumulated statistics across episodes
            beliefs['_accumulated_stats'] = {
                'total_steps': self.total_steps,
                'observation_statistics': {
                    key: {
                        'mean': stats['mean'],
                        'variance': stats['variance'],
                        'n_obs': stats['n_obs']
                    }
                    for key, stats in self.observation_statistics.items()
                },
                'action_outcomes': {
                    action: {
                        'success': outcomes['success'],
                        'total': outcomes['total']
                    }
                    for action, outcomes in self.action_outcomes.items()
                }
            }

        except Exception as e:
            print(f"Warning: Failed to extract beliefs: {e}")

        return beliefs

    # ========================================================================
    # Token Accounting
    # ========================================================================

    def get_token_breakdown(self) -> dict:
        """Get token breakdown by category"""
        return self.token_accountant.to_dict()

    def validate_token_accounting(self, total_input: int, total_output: int) -> bool:
        """Validate token breakdown matches totals"""
        return self.token_accountant.validate(total_input, total_output)
