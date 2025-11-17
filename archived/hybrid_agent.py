# agents/hybrid_agent.py
"""
Hybrid Agent combining ACE and ACTOR architectures.

Combines:
- ACE: Qualitative strategy generation from evolved context (playbook)
- ACTOR: Quantitative strategy evaluation using probabilistic belief states

The hybrid agent uses ACE to generate diverse candidate strategies and
ACTOR to score and select the most promising one based on belief states.
"""

import time
import json
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from agents.base import Agent, AgentStep, LLMInterface, detect_query_type
from agents.ace import ACEAgent
from agents.actor import ActorAgent
from agents.text_reader import TextReaderAgent
from experiments.prompts import (
    format_observation_history,
    extract_answer_components
)
from utils.token_accounting import TokenAccountant


# Hybrid-specific prompt template for scoring strategies
HYBRID_SCORING_TEMPLATE = """You are helping evaluate action strategies based on a probabilistic belief state.

CURRENT BELIEF STATE:
{belief_state}

RECENT OBSERVATIONS:
{memory_summary}

CANDIDATE STRATEGIES:
{candidates}

TASK: Score each candidate strategy from 0.0 to 1.0 based on:
1. Expected information gain (how much will this reduce uncertainty?)
2. Probability of success (how likely is this to work given our beliefs?)
3. Risk assessment (what could go wrong?)

Return a JSON object with scores for each candidate:
{{
  "scores": [score1, score2, score3, ...],
  "reasoning": "Brief explanation of scoring rationale"
}}

Example:
{{
  "scores": [0.8, 0.6, 0.9, 0.5, 0.7],
  "reasoning": "Strategy 3 scores highest because it directly tests our most uncertain belief..."
}}
"""


class HybridAgent(Agent):
    """
    Hybrid agent combining ACE's context evolution with ACTOR's belief-based reasoning.

    Architecture:
    1. ACE generates multiple candidate strategies from its evolved playbook
    2. ACTOR scores each candidate based on probabilistic belief state
    3. Highest-scoring strategy is selected
    4. Both sub-agents learn and evolve (ACE updates playbook, ACTOR updates beliefs)
    """

    def __init__(
        self,
        llm: LLMInterface,
        action_budget: int,
        environment_name: Optional[str] = None,
        num_candidates: int = 5,
        candidate_temperature: float = 0.9,
        scoring_temperature: float = 0.3,
        # ACE-specific parameters
        use_retrieval: bool = True,
        top_k: int = 5,
        reflection_rounds: int = 1,
        # Cost optimization parameters
        enable_cost_optimization: bool = False,
        num_prescreening_candidates: int = 5,
        num_final_candidates: int = 2,
        prior_logs: Optional[List[Dict]] = None,
        # Weighted selection parameters
        selection_weights: Optional[Dict[str, float]] = None,
        # Pass through additional ACE/ACTOR parameters
        **kwargs
    ):
        """
        Initialize Hybrid agent.

        Args:
            llm: LLM interface (shared by both sub-agents)
            action_budget: Maximum number of actions per episode
            environment_name: Name of environment (for tool selection)
            num_candidates: Number of candidate strategies to generate from ACE
            candidate_temperature: Temperature for ACE candidate generation (higher = more diverse)
            scoring_temperature: Temperature for ACTOR scoring (lower = more consistent)
            use_retrieval: Whether ACE should use top-k retrieval
            top_k: Number of bullets ACE retrieves per section
            reflection_rounds: Number of reflection rounds for ACE
            enable_cost_optimization: Enable three-stage selection with text_reader pre-screening
            num_prescreening_candidates: Number of candidates for text_reader to score (stage 2)
            num_final_candidates: Number of candidates for ACTOR to score (stage 3)
            prior_logs: Prior episode logs for text_reader agent
            selection_weights: Dictionary of weights for environment/question type combinations
            **kwargs: Additional parameters passed to sub-agents
        """
        super().__init__(llm, action_budget)

        self.environment_name = environment_name
        self.num_candidates = num_candidates
        self.candidate_temperature = candidate_temperature
        self.scoring_temperature = scoring_temperature
        self.enable_cost_optimization = enable_cost_optimization
        self.num_prescreening_candidates = num_prescreening_candidates
        self.num_final_candidates = num_final_candidates

        # Default selection weights (can be overridden via config)
        self.selection_weights = selection_weights or {
            'hotpot_planning': {'ace_weight': 0.7, 'actor_weight': 0.3},
            'hotpot_counterfactual': {'ace_weight': 0.6, 'actor_weight': 0.4},
            'hotpot_interventional': {'ace_weight': 0.3, 'actor_weight': 0.7},
            'chemtile_any': {'ace_weight': 0.35, 'actor_weight': 0.65},
            'switchlight_counterfactual': {'ace_weight': 0.6, 'actor_weight': 0.4},
            'switchlight_interventional': {'ace_weight': 0.4, 'actor_weight': 0.6},
            'default': {'ace_weight': 0.5, 'actor_weight': 0.5}
        }

        # Initialize sub-agents with shared LLM
        # ACE: Strategy generation from evolved playbook
        self.ace = ACEAgent(
            llm=llm,
            action_budget=action_budget,
            environment_name=environment_name,
            use_retrieval=use_retrieval,
            top_k=top_k,
            reflection_rounds=reflection_rounds,
            generator_temperature=candidate_temperature,
            **{k: v for k, v in kwargs.items() if k in [
                'curation_mode', 'token_cap', 'reflector_temperature',
                'curator_temperature', 'max_epochs'
            ]}
        )

        # ACTOR: Belief-based strategy evaluation
        self.actor = ActorAgent(
            llm=llm,
            action_budget=action_budget,
            environment_name=environment_name
        )

        # TEXT_READER: Cost-effective pre-screening (optional, for cost optimization)
        self.text_reader = None
        if enable_cost_optimization and prior_logs:
            self.text_reader = TextReaderAgent(
                llm=llm,
                action_budget=action_budget,
                prior_logs=prior_logs
            )

        # Decision log for analysis and debugging
        self.decision_log: List[Dict] = []

        # Track current environment context for weighted selection
        self.current_environment_type: Optional[str] = None
        self.current_question_type: Optional[str] = None

        # Latest decision metadata (for get_belief_state())
        self._latest_decision: Optional[Dict] = None

        # Token accounting for hybrid-specific operations
        self.token_accountant = TokenAccountant()

        # Thinking efficiency metrics (for optimization analysis)
        self.thinking_metrics = {
            "candidates_generated": [],      # Number of candidates generated per action
            "candidates_evaluated": [],      # Number of candidates scored per action
            "generation_tokens": [],         # Tokens used for candidate generation
            "scoring_tokens": [],            # Tokens used for scoring
            "thinking_tokens": [],           # Total internal deliberation tokens
            "action_tokens": [],             # Tokens for final action execution
            "selection_confidence": [],      # max_score - mean_score
            "iterations_to_solution": [],    # Number of batches needed
            "early_stops": 0,                # Count of early terminations
            "max_score_per_action": [],      # Best score achieved per action
        }

        # Iterative refinement parameters
        self.iterative_batch_size = 2
        self.iterative_max_candidates = 8
        self.iterative_confidence_threshold = 0.75

    def set_belief_state(self, belief: Any):
        """
        Initialize belief state for ACTOR sub-agent.

        Args:
            belief: BeliefState instance (e.g., HotPotBelief)
        """
        self.actor.set_belief_state(belief)

    def reset(
        self,
        environment_type: Optional[str] = None,
        initial_observation: Optional[dict] = None
    ):
        """
        Reset agent state for new episode.

        Resets both sub-agents and decision log.

        Args:
            environment_type: Type of environment (for ACTOR prior generation)
            initial_observation: Initial observation (for ACTOR prior generation)
        """
        super().reset()

        # Track environment type for weighted selection
        self.current_environment_type = environment_type

        # Reset sub-agents
        self.ace.reset()
        if environment_type and initial_observation:
            self.actor.reset(environment_type, initial_observation)
        else:
            self.actor.reset()

        # Reset text_reader if enabled
        if self.text_reader:
            self.text_reader.reset()

        # Reset decision log
        self.decision_log = []
        self._latest_decision = None
        self.current_question_type = None

        # Reset token accounting
        self.token_accountant.reset()

        # Reset thinking metrics for new episode
        # Note: We keep cumulative metrics across episodes, just track per-action
        # The efficiency report will show episode-level stats

    def act(self, observation: dict) -> AgentStep:
        """
        Choose action by combining ACE strategy generation with ACTOR evaluation.

        Process:
        1. Generate multiple candidate strategies from ACE
        2. Score each candidate using ACTOR's belief state
        3. Select highest-scoring strategy
        4. Update ACTOR's belief based on observation

        Args:
            observation: Environment observation

        Returns:
            AgentStep with selected action and metadata
        """
        # Choose action if budget allows
        if self.action_count < self.action_budget:
            thought, action, decision_metadata = self._hybrid_choose_action(observation)
            self.action_count += 1
        else:
            thought = "Action budget exhausted"
            action = None
            decision_metadata = {}

        # Create step record
        step = AgentStep(
            timestamp=time.time(),
            step_num=len(self.memory),
            thought=thought,
            action=action,
            observation=observation,
            belief_state=self._serialize_hybrid_state(),
            surprisal=0.0,  # Will be computed by runner after action execution
            token_usage=0  # Tracked at episode level
        )

        # Add decision metadata to step
        if decision_metadata:
            step.belief_state['hybrid_decision'] = decision_metadata

        self.memory.append(step)
        return step

    def answer_query(self, question: str) -> Tuple[str, float]:
        """
        Answer query by combining insights from both ACE and ACTOR.

        Strategy:
        - Detect query type (counterfactual, planning, interventional)
        - Use selection weights to choose between agents
        - Combine their answers with weighted confidence

        Args:
            question: Question to answer

        Returns:
            Tuple of (answer, confidence)
        """
        # Get answers from both sub-agents
        ace_answer, ace_confidence = self.ace.answer_query(question)
        actor_answer, actor_confidence = self.actor.answer_query(question)

        # Detect query type for weighted selection
        query_type = detect_query_type(question)

        # Get selection weights for this query type
        # selection_weights format: {query_type: [actor_weight, ace_weight]}
        weights = self.selection_weights.get(query_type, self.selection_weights.get('default', [0.5, 0.5]))
        actor_weight, ace_weight = weights

        # Compute weighted confidence scores
        actor_weighted = actor_confidence * actor_weight
        ace_weighted = ace_confidence * ace_weight

        # Select agent based on weighted confidence
        if actor_weighted >= ace_weighted:
            final_answer = actor_answer
            final_confidence = actor_confidence
            reasoning = f"ACTOR selected (weighted: {actor_weighted:.3f} vs ACE: {ace_weighted:.3f}, type: {query_type}, weights: {weights})"
        else:
            final_answer = ace_answer
            final_confidence = ace_confidence
            reasoning = f"ACE selected (weighted: {ace_weighted:.3f} vs ACTOR: {actor_weighted:.3f}, type: {query_type}, weights: {weights})"

        # Log decision
        self.decision_log.append({
            'type': 'query',
            'question': question,
            'query_type': query_type,
            'ace_answer': ace_answer,
            'ace_confidence': ace_confidence,
            'actor_answer': actor_answer,
            'actor_confidence': actor_confidence,
            'selection_weights': weights,
            'actor_weighted_score': actor_weighted,
            'ace_weighted_score': ace_weighted,
            'final_answer': final_answer,
            'final_confidence': final_confidence,
            'reasoning': reasoning
        })

        return final_answer, final_confidence

    def compute_surprisal(self, observation: dict) -> float:
        """
        Compute surprisal using ACTOR's belief state.

        Args:
            observation: Environment observation

        Returns:
            Surprisal value
        """
        return self.actor.compute_surprisal(observation)

    def update_belief_from_observation(self, observation: dict):
        """
        Update ACTOR's belief state based on observation.

        Args:
            observation: Environment observation
        """
        self.actor.update_belief_from_observation(observation)

    def get_belief_state(self) -> dict:
        """
        Get current hybrid belief state (ACTOR belief + latest decision metadata).

        Returns:
            Belief state dictionary with hybrid_decision if available
        """
        belief_state = self.actor.get_belief_state()

        # Add latest hybrid decision metadata if available
        if self._latest_decision is not None:
            belief_state['hybrid_decision'] = self._latest_decision

        return belief_state

    def update_playbook(self, outcome: dict):
        """
        Update ACE's playbook after episode completion.

        This allows ACE to learn from experience across episodes.

        Args:
            outcome: Episode outcome including test results
        """
        self.ace.update_playbook(outcome)

    # ========================================================================
    # Hybrid Decision Making (Core Algorithm)
    # ========================================================================

    def _get_selection_weights(self, observation: dict) -> Dict[str, float]:
        """
        Get environment and question-type specific weights for selection.

        Args:
            observation: Current observation (may contain question for context)

        Returns:
            Dictionary with 'ace_weight' and 'actor_weight'
        """
        # Try to detect question type from observation if available
        # (this is a heuristic - actual question type is known during answer_query)
        question_type = 'interventional'  # default for actions

        # Check if we're in a specific environment
        env_type = self.current_environment_type or 'unknown'

        # Normalize environment name
        if env_type:
            env_type = env_type.lower()
            if 'hotpot' in env_type:
                env_type = 'hotpot'
            elif 'chemtile' in env_type or 'chem' in env_type:
                env_type = 'chemtile'
            elif 'switch' in env_type or 'light' in env_type:
                env_type = 'switchlight'

        # Build lookup key
        lookup_keys = [
            f"{env_type}_{question_type}",  # e.g., "hotpot_interventional"
            f"{env_type}_any",               # e.g., "chemtile_any"
            "default"                         # fallback
        ]

        # Find first matching weight configuration
        for key in lookup_keys:
            if key in self.selection_weights:
                weights = self.selection_weights[key]
                # Convert list format [actor_weight, ace_weight] to dict
                if isinstance(weights, list):
                    return {'actor_weight': weights[0], 'ace_weight': weights[1]}
                else:
                    return weights

        # Ultimate fallback
        return {'ace_weight': 0.5, 'actor_weight': 0.5}

    def _weighted_selection(
        self,
        candidates: List[Dict[str, str]],
        scores: List[float],
        observation: dict
    ) -> int:
        """
        Select candidate using weighted combination of ACTOR scores and ACE ranking.

        Combines:
        1. ACTOR's numerical scores (normalized)
        2. ACE's implicit ranking (earlier candidates ranked higher)
        3. Environment-specific weights

        Args:
            candidates: List of candidate strategies from ACE
            scores: ACTOR scores for each candidate
            observation: Current observation (for environment context)

        Returns:
            Index of selected candidate
        """
        # Get environment-specific weights
        weights = self._get_selection_weights(observation)
        ace_weight = weights['ace_weight']
        actor_weight = weights['actor_weight']

        # Normalize ACTOR scores to [0, 1]
        if max(scores) > min(scores):
            normalized_scores = [
                (s - min(scores)) / (max(scores) - min(scores))
                for s in scores
            ]
        else:
            normalized_scores = [0.5] * len(scores)

        # ACE ranking: earlier candidates are preferred
        # Convert to scores: first candidate = 1.0, last = 0.0
        ace_scores = [
            1.0 - (i / max(1, len(candidates) - 1))
            for i in range(len(candidates))
        ]

        # Combine scores with weights
        combined_scores = [
            actor_weight * actor_score + ace_weight * ace_score
            for actor_score, ace_score in zip(normalized_scores, ace_scores)
        ]

        # Select highest combined score
        best_idx = combined_scores.index(max(combined_scores))

        return best_idx

    def _hybrid_choose_action(
        self,
        observation: dict
    ) -> Tuple[str, Optional[str], Dict]:
        """
        Hybrid action selection: ACE generates candidates, ACTOR scores them.

        Uses iterative refinement: generates candidates in batches, scores them,
        and stops early if a high-confidence candidate is found.

        Args:
            observation: Current observation

        Returns:
            Tuple of (thought, action_string, decision_metadata)
        """
        # Track token usage at start
        tokens_before = self.llm.get_total_tokens() if hasattr(self.llm, 'get_total_tokens') else 0

        # Step 1: Generate candidate strategies from ACE with iterative refinement
        candidates, all_scores, iterations = self._generate_candidates_iterative(observation)

        # Track generation tokens
        tokens_after_generation = self.llm.get_total_tokens() if hasattr(self.llm, 'get_total_tokens') else 0
        generation_tokens = tokens_after_generation - tokens_before

        if not candidates:
            # Fallback: If ACE fails, use ACTOR directly
            thought, action = self.actor._choose_action(observation)
            return thought, action, {
                'strategy': 'fallback_to_actor',
                'reason': 'ACE candidate generation failed'
            }

        # Note: all_scores already contains scores from iterative refinement
        # We use these directly without re-scoring (unless cost optimization is enabled)

        # Step 2: Pre-screen with TEXT_READER if cost optimization is enabled
        prescreening_scores = None
        prescreening_reasoning = ""
        candidates_to_score = candidates  # Default: use all candidates
        actor_scores = all_scores  # Use scores from iterative refinement

        if self.enable_cost_optimization and self.text_reader:
            # Three-stage pipeline: ACE -> TEXT_READER -> ACTOR
            prescreening_scores, prescreening_reasoning = self._prescreen_with_text_reader(
                observation, candidates
            )

            if prescreening_scores and len(prescreening_scores) == len(candidates):
                # Select top N candidates based on text_reader scores
                scored_candidates = list(zip(candidates, prescreening_scores, range(len(candidates))))
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                top_candidates = scored_candidates[:self.num_final_candidates]

                # Extract candidates and their original indices
                candidates_to_score = [c for c, _, _ in top_candidates]
                prescreening_indices = [idx for _, _, idx in top_candidates]

                # Re-score with ACTOR for final selection
                actor_scores, scoring_reasoning = self._score_with_actor(observation, candidates_to_score)
            else:
                prescreening_indices = list(range(len(candidates)))
        else:
            prescreening_indices = list(range(len(candidates)))
            scoring_reasoning = "Used scores from iterative refinement"

        if not actor_scores or len(actor_scores) != len(candidates_to_score):
            # Fallback: If scoring fails, use first ACE candidate
            best_candidate = candidates[0]
            return (
                best_candidate['thought'],
                best_candidate['action'],
                {
                    'strategy': 'fallback_to_ace',
                    'reason': 'ACTOR scoring failed',
                    'candidates': len(candidates),
                    'cost_optimization_enabled': self.enable_cost_optimization
                }
            )

        # Step 4: Select using weighted combination of ACTOR scores and ACE ranking
        # Note: We apply weighted selection on the prescreened candidates
        local_best_idx = self._weighted_selection(candidates_to_score, actor_scores, observation)
        best_candidate = candidates_to_score[local_best_idx]

        # Map back to original candidate index
        if self.enable_cost_optimization and prescreening_scores:
            best_idx = prescreening_indices[local_best_idx]
        else:
            best_idx = local_best_idx

        # Get weights used for decision logging
        weights = self._get_selection_weights(observation)

        # Step 5: Log decision for analysis
        decision_metadata = {
            'strategy': 'hybrid_weighted' if not self.enable_cost_optimization else 'hybrid_optimized',
            'num_candidates': len(candidates),
            'selected_idx': best_idx,
            'selected_score': actor_scores[local_best_idx],
            'all_actor_scores': actor_scores,
            'scoring_reasoning': scoring_reasoning,
            'cost_optimization_enabled': self.enable_cost_optimization,
            'selection_weights': weights,
            'candidates_summary': [
                {
                    'action': c['action'],
                    'actor_score': s,
                    'thought': c['thought'][:100]  # Truncate
                }
                for c, s in zip(candidates_to_score, actor_scores)
            ]
        }

        # Add prescreening data if available
        if self.enable_cost_optimization and prescreening_scores:
            decision_metadata['prescreening'] = {
                'num_prescreened': len(candidates),
                'num_actor_scored': len(candidates_to_score),
                'prescreening_scores': prescreening_scores,
                'prescreening_reasoning': prescreening_reasoning,
                'prescreened_indices': prescreening_indices
            }

        self.decision_log.append(decision_metadata)

        # Store latest decision for get_belief_state()
        self._latest_decision = decision_metadata

        # Track thinking efficiency metrics
        tokens_after_scoring = self.llm.get_total_tokens() if hasattr(self.llm, 'get_total_tokens') else tokens_after_generation
        scoring_tokens = tokens_after_scoring - tokens_after_generation
        total_thinking_tokens = tokens_after_scoring - tokens_before

        self.thinking_metrics["candidates_generated"].append(len(candidates))
        self.thinking_metrics["candidates_evaluated"].append(len(candidates_to_score))
        self.thinking_metrics["generation_tokens"].append(generation_tokens)
        self.thinking_metrics["scoring_tokens"].append(scoring_tokens)
        self.thinking_metrics["thinking_tokens"].append(total_thinking_tokens)
        self.thinking_metrics["iterations_to_solution"].append(iterations)
        self.thinking_metrics["max_score_per_action"].append(max(actor_scores))

        # Calculate selection confidence (spread between best and average)
        if len(actor_scores) > 1:
            confidence = max(actor_scores) - np.mean(actor_scores)
        else:
            confidence = 0.0
        self.thinking_metrics["selection_confidence"].append(confidence)

        # Construct hybrid thought combining ACE's reasoning with ACTOR's scoring
        hybrid_thought = (
            f"[HYBRID] Selected strategy {best_idx+1}/{len(candidates)} "
            f"(score={actor_scores[local_best_idx]:.2f}, iterations={iterations}): {best_candidate['thought']}"
        )

        return hybrid_thought, best_candidate['action'], decision_metadata

    def _generate_candidates_iterative(
        self,
        observation: dict
    ) -> Tuple[List[Dict[str, str]], List[float], int]:
        """
        Generate candidates iteratively with early stopping.

        Process:
        1. Generate batch of 2 candidates
        2. Score them with ACTOR
        3. If max(score) >= threshold, stop (early termination)
        4. Otherwise, generate 2 more informed by previous scores
        5. Repeat until max_candidates reached or confidence achieved

        Args:
            observation: Current observation

        Returns:
            Tuple of (all_candidates, all_scores, num_iterations)
        """
        all_candidates = []
        all_scores = []
        iteration = 0
        max_iterations = self.iterative_max_candidates // self.iterative_batch_size

        for iteration in range(1, max_iterations + 1):
            # Generate batch of candidates
            batch = self._generate_candidate_batch(
                observation,
                batch_size=self.iterative_batch_size,
                previous_scores=all_scores if all_scores else None
            )

            if not batch:
                break  # Generation failed

            # Score the batch
            batch_scores, _ = self._score_with_actor(observation, batch)

            if not batch_scores or len(batch_scores) != len(batch):
                # Scoring failed, use neutral scores
                batch_scores = [0.5] * len(batch)

            # Add to running lists
            all_candidates.extend(batch)
            all_scores.extend(batch_scores)

            # Check for early stopping
            max_score = max(batch_scores)
            if max_score >= self.iterative_confidence_threshold:
                # Found confident candidate, stop early
                self.thinking_metrics["early_stops"] += 1
                break

            # Check for diminishing returns
            if iteration > 1:
                # Compare best score in current batch vs previous best
                prev_best = max(all_scores[:-len(batch)])
                if max_score - prev_best < 0.05:
                    # Less than 5% improvement, stop
                    break

        return all_candidates, all_scores, iteration

    def _generate_candidate_batch(
        self,
        observation: dict,
        batch_size: int,
        previous_scores: Optional[List[float]] = None
    ) -> List[Dict[str, str]]:
        """
        Generate a batch of candidate strategies.

        If previous_scores are provided, uses them to inform ACE about
        what didn't work well (lower-scoring strategies).

        Args:
            observation: Current observation
            batch_size: Number of candidates to generate
            previous_scores: Scores from previous batches (for informed generation)

        Returns:
            List of candidate dictionaries with 'thought' and 'action' keys
        """
        candidates = []

        # Store original temperature
        original_temp = self.ace.generator_temperature

        try:
            # If we have previous scores, we can inform generation
            # For now, we just vary temperature more aggressively
            # Future enhancement: Could inject feedback into ACE prompt
            temp_boost = 0.2 if previous_scores and max(previous_scores) < 0.6 else 0.0

            # Generate candidates with varied temperature for diversity
            for i in range(batch_size):
                # Vary temperature for diversity
                temp_variation = 0.1 * (i - batch_size // 2) + temp_boost
                self.ace.generator_temperature = min(1.0, max(0.1, original_temp + temp_variation))

                # Generate candidate
                thought, action = self.ace._choose_action(observation)

                if action:  # Only include valid actions
                    candidates.append({
                        'thought': thought,
                        'action': action,
                        'temperature': self.ace.generator_temperature
                    })

        finally:
            # Restore original temperature
            self.ace.generator_temperature = original_temp

        return candidates

    def _generate_ace_candidates(self, observation: dict) -> List[Dict[str, str]]:
        """
        Generate multiple candidate strategies from ACE.

        Uses temperature sampling to generate diverse candidates from ACE's playbook.

        Args:
            observation: Current observation

        Returns:
            List of candidate dictionaries with 'thought' and 'action' keys
        """
        candidates = []

        # Store original temperature
        original_temp = self.ace.generator_temperature

        try:
            # Generate candidates with varied temperature for diversity
            for i in range(self.num_candidates):
                # Vary temperature slightly for each candidate
                # This encourages diversity while staying near ACE's preferred range
                temp_variation = 0.1 * (i - self.num_candidates // 2)
                self.ace.generator_temperature = min(1.0, max(0.1, original_temp + temp_variation))

                # Generate candidate
                thought, action = self.ace._choose_action(observation)

                if action:  # Only include valid actions
                    candidates.append({
                        'thought': thought,
                        'action': action,
                        'temperature': self.ace.generator_temperature
                    })

        finally:
            # Restore original temperature
            self.ace.generator_temperature = original_temp

        return candidates

    def _score_with_actor(
        self,
        observation: dict,
        candidates: List[Dict[str, str]]
    ) -> Tuple[List[float], str]:
        """
        Score each candidate strategy using ACTOR's belief state.

        Uses ACTOR's LLM to evaluate how well each strategy aligns with
        the current belief state and expected outcomes.

        Args:
            observation: Current observation
            candidates: List of candidate strategies

        Returns:
            Tuple of (scores_list, reasoning_string)
        """
        # Format candidates for prompt
        candidates_text = "\n".join([
            f"Strategy {i+1}: {c['action']}\n  Reasoning: {c['thought']}"
            for i, c in enumerate(candidates)
        ])

        # Build scoring prompt
        prompt = HYBRID_SCORING_TEMPLATE.format(
            belief_state=json.dumps(self.actor.get_belief_state(), indent=2),
            memory_summary=format_observation_history(self.memory, max_steps=3),
            candidates=candidates_text
        )

        # Query ACTOR's LLM for scores
        try:
            response = self.llm.generate(prompt, temperature=self.scoring_temperature)

            # Record token usage for planning (hybrid scoring)
            input_tokens, output_tokens = self.llm.get_last_usage()
            self.token_accountant.record(
                'planning',
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata={'phase': 'hybrid_scoring', 'num_candidates': len(candidates)}
            )

            # Parse scores
            parsed = self._parse_json_response(response)
            scores = parsed.get('scores', [])
            reasoning = parsed.get('reasoning', 'No reasoning provided')

            # Validate scores
            if len(scores) != len(candidates):
                print(f"Warning: Score count mismatch ({len(scores)} vs {len(candidates)})")
                # Fallback: uniform scores
                scores = [0.5] * len(candidates)

            # Normalize scores to [0, 1]
            scores = [max(0.0, min(1.0, float(s))) for s in scores]

            return scores, reasoning

        except Exception as e:
            print(f"Warning: Scoring failed: {e}")
            # Fallback: uniform scores
            return [0.5] * len(candidates), f"Scoring failed: {e}"

    def _prescreen_with_text_reader(
        self,
        observation: dict,
        candidates: List[Dict[str, str]]
    ) -> Tuple[List[float], str]:
        """
        Pre-screen candidates using TEXT_READER for cost optimization.

        TEXT_READER is cheaper than ACTOR but provides reasonable estimates
        of candidate quality based on prior episode experience.

        Args:
            observation: Current observation
            candidates: List of candidate strategies

        Returns:
            Tuple of (scores_list, reasoning_string)
        """
        if not self.text_reader:
            return [], "Text reader not initialized"

        # Format candidates as evaluation questions
        # We ask text_reader to score each candidate's likelihood of success
        scores = []

        for i, candidate in enumerate(candidates):
            # Create a question that asks about the candidate's effectiveness
            question = (
                f"How effective would this action be: {candidate['action']}? "
                f"Rate from 0.0 (ineffective) to 1.0 (very effective). "
                f"Reasoning: {candidate['thought']}"
            )

            try:
                # Get text_reader's evaluation
                answer, confidence = self.text_reader.answer_query(question)

                # Extract numerical score from answer
                # Try to parse a float from the answer
                score = confidence  # Use confidence as proxy for effectiveness

                # Alternative: try to extract explicit score from answer
                import re
                score_match = re.search(r'(\d+\.?\d*)\s*(?:/\s*1\.0|out of 1)', answer.lower())
                if score_match:
                    score = float(score_match.group(1))

                scores.append(max(0.0, min(1.0, score)))

            except Exception as e:
                print(f"Warning: Text reader prescreening failed for candidate {i}: {e}")
                scores.append(0.5)  # Neutral score on failure

        # Record token usage for prescreening
        if self.text_reader:
            # Text reader already records its own tokens, we just need to note the phase
            input_tokens, output_tokens = self.llm.get_last_usage()
            self.token_accountant.record(
                'planning',
                input_tokens=input_tokens * len(candidates),  # Approximate total
                output_tokens=output_tokens * len(candidates),
                metadata={'phase': 'text_reader_prescreening', 'num_candidates': len(candidates)}
            )

        reasoning = f"Pre-screened {len(candidates)} candidates with text_reader"
        return scores, reasoning

    def _serialize_hybrid_state(self) -> dict:
        """
        Serialize hybrid state combining ACE and ACTOR state.

        Returns:
            Dictionary with hybrid state
        """
        return {
            'ace_playbook_size': self.ace._get_playbook_size(),
            'actor_belief': self.actor.get_belief_state(),
            'decision_count': len(self.decision_log)
        }

    def _parse_json_response(self, response: str) -> dict:
        """
        Parse JSON from LLM response with robust error handling.

        Args:
            response: Raw LLM response

        Returns:
            Parsed JSON dictionary
        """
        # Reuse ACE's robust JSON parser
        return self.ace._parse_json_response(response)

    # ========================================================================
    # Token Accounting
    # ========================================================================

    def get_token_breakdown(self) -> dict:
        """
        Get token breakdown combining both sub-agents and hybrid-specific operations.

        Returns:
            Dictionary with token breakdown
        """
        # Get breakdowns from sub-agents
        ace_breakdown = self.ace.get_token_breakdown()
        actor_breakdown = self.actor.get_token_breakdown()
        hybrid_breakdown = self.token_accountant.to_dict()

        # Combine breakdowns
        combined = {
            'ace': ace_breakdown,
            'actor': actor_breakdown,
            'hybrid_specific': hybrid_breakdown,
            'total_by_category': {}
        }

        # Sum up totals by category across all agents
        # Note: sub-agent breakdowns have structure {'breakdown': {category: {'input': X, 'output': Y, 'total': Z}}}
        ace_cats = ace_breakdown.get('breakdown', {})
        actor_cats = actor_breakdown.get('breakdown', {})
        hybrid_cats = hybrid_breakdown.get('breakdown', {})

        categories = set()
        categories.update(ace_cats.keys())
        categories.update(actor_cats.keys())
        categories.update(hybrid_cats.keys())
        # Remove 'totals' if present (it's a summary, not a category)
        categories.discard('totals')

        for category in categories:
            combined['total_by_category'][category] = {
                'input_tokens': (
                    ace_cats.get(category, {}).get('input', 0) +
                    actor_cats.get(category, {}).get('input', 0) +
                    hybrid_cats.get(category, {}).get('input', 0)
                ),
                'output_tokens': (
                    ace_cats.get(category, {}).get('output', 0) +
                    actor_cats.get(category, {}).get('output', 0) +
                    hybrid_cats.get(category, {}).get('output', 0)
                )
            }

        return combined

    def validate_token_accounting(self, total_input: int, total_output: int) -> bool:
        """
        Validate that token breakdown matches totals.

        Args:
            total_input: Expected total input tokens
            total_output: Expected total output tokens

        Returns:
            True if validation passes
        """
        # Get combined breakdown
        breakdown = self.get_token_breakdown()

        # Sum up all input/output tokens from all sources
        computed_input = sum(
            cat.get('input_tokens', 0)
            for cat in breakdown['total_by_category'].values()
        )
        computed_output = sum(
            cat.get('output_tokens', 0)
            for cat in breakdown['total_by_category'].values()
        )

        # Allow small discrepancy due to rounding
        input_match = abs(computed_input - total_input) <= 10
        output_match = abs(computed_output - total_output) <= 10

        if not (input_match and output_match):
            raise ValueError(
                f"Token accounting mismatch: "
                f"expected ({total_input}, {total_output}), "
                f"got ({computed_input}, {computed_output})"
            )

        return True

    def get_efficiency_report(self) -> dict:
        """
        Generate thinking efficiency report for analysis and optimization.

        Returns comprehensive metrics about candidate generation, scoring,
        token usage, and decision quality.

        Returns:
            Dictionary with efficiency metrics and analysis
        """
        metrics = self.thinking_metrics

        # Handle empty metrics (no actions taken yet)
        if not metrics["candidates_generated"]:
            return {
                "status": "no_data",
                "message": "No actions taken yet - metrics not available"
            }

        # Calculate aggregate statistics
        total_actions = len(metrics["candidates_generated"])

        report = {
            "episode_summary": {
                "total_actions": total_actions,
                "total_candidates_generated": sum(metrics["candidates_generated"]),
                "total_candidates_evaluated": sum(metrics["candidates_evaluated"]),
                "early_stops": metrics["early_stops"],
                "early_stop_rate": metrics["early_stops"] / total_actions if total_actions > 0 else 0,
            },

            "per_action_averages": {
                "avg_candidates_generated": np.mean(metrics["candidates_generated"]),
                "avg_candidates_evaluated": np.mean(metrics["candidates_evaluated"]),
                "avg_iterations": np.mean(metrics["iterations_to_solution"]),
                "avg_selection_confidence": np.mean(metrics["selection_confidence"]),
                "avg_max_score": np.mean(metrics["max_score_per_action"]),
            },

            "token_efficiency": {
                "total_thinking_tokens": sum(metrics["thinking_tokens"]),
                "total_generation_tokens": sum(metrics["generation_tokens"]),
                "total_scoring_tokens": sum(metrics["scoring_tokens"]),
                "avg_thinking_tokens_per_action": np.mean(metrics["thinking_tokens"]),
                "generation_percentage": (
                    sum(metrics["generation_tokens"]) / sum(metrics["thinking_tokens"]) * 100
                    if sum(metrics["thinking_tokens"]) > 0 else 0
                ),
                "scoring_percentage": (
                    sum(metrics["scoring_tokens"]) / sum(metrics["thinking_tokens"]) * 100
                    if sum(metrics["thinking_tokens"]) > 0 else 0
                ),
                "tokens_per_candidate_generated": (
                    sum(metrics["generation_tokens"]) / sum(metrics["candidates_generated"])
                    if sum(metrics["candidates_generated"]) > 0 else 0
                ),
                "tokens_per_candidate_scored": (
                    sum(metrics["scoring_tokens"]) / sum(metrics["candidates_evaluated"])
                    if sum(metrics["candidates_evaluated"]) > 0 else 0
                ),
            },

            "optimization_insights": {
                "avg_candidates_saved": (
                    self.iterative_max_candidates - np.mean(metrics["candidates_generated"])
                ),
                "token_savings_from_early_stop": (
                    sum(metrics["thinking_tokens"]) * metrics["early_stops"] / total_actions * 0.3
                    if total_actions > 0 else 0
                ),
                "low_confidence_actions": sum(
                    1 for conf in metrics["selection_confidence"] if conf < 0.2
                ),
                "high_confidence_actions": sum(
                    1 for conf in metrics["selection_confidence"] if conf >= 0.5
                ),
            },

            "per_action_details": [
                {
                    "action_num": i + 1,
                    "candidates_generated": metrics["candidates_generated"][i],
                    "candidates_evaluated": metrics["candidates_evaluated"][i],
                    "iterations": metrics["iterations_to_solution"][i],
                    "thinking_tokens": metrics["thinking_tokens"][i],
                    "max_score": metrics["max_score_per_action"][i],
                    "confidence": metrics["selection_confidence"][i],
                }
                for i in range(total_actions)
            ]
        }

        return report

    def print_efficiency_report(self):
        """Print human-readable efficiency report to console."""
        report = self.get_efficiency_report()

        if report.get("status") == "no_data":
            print(report["message"])
            return

        print("\n" + "=" * 70)
        print("HYBRID AGENT THINKING EFFICIENCY REPORT")
        print("=" * 70)

        # Episode Summary
        summary = report["episode_summary"]
        print(f"\n📊 Episode Summary:")
        print(f"  Total actions: {summary['total_actions']}")
        print(f"  Total candidates generated: {summary['total_candidates_generated']}")
        print(f"  Total candidates evaluated: {summary['total_candidates_evaluated']}")
        print(f"  Early stops: {summary['early_stops']} ({summary['early_stop_rate']:.1%})")

        # Averages
        avg = report["per_action_averages"]
        print(f"\n📈 Per-Action Averages:")
        print(f"  Candidates generated: {avg['avg_candidates_generated']:.1f}")
        print(f"  Candidates evaluated: {avg['avg_candidates_evaluated']:.1f}")
        print(f"  Iterations to solution: {avg['avg_iterations']:.1f}")
        print(f"  Selection confidence: {avg['avg_selection_confidence']:.3f}")
        print(f"  Max score achieved: {avg['avg_max_score']:.3f}")

        # Token Efficiency
        tokens = report["token_efficiency"]
        print(f"\n💰 Token Efficiency:")
        print(f"  Total thinking tokens: {tokens['total_thinking_tokens']:,}")
        print(f"  Average per action: {tokens['avg_thinking_tokens_per_action']:.0f}")
        print(f"  Generation: {tokens['generation_percentage']:.1f}% ({tokens['total_generation_tokens']:,} tokens)")
        print(f"  Scoring: {tokens['scoring_percentage']:.1f}% ({tokens['total_scoring_tokens']:,} tokens)")
        print(f"  Tokens per candidate generated: {tokens['tokens_per_candidate_generated']:.0f}")
        print(f"  Tokens per candidate scored: {tokens['tokens_per_candidate_scored']:.0f}")

        # Optimization Insights
        insights = report["optimization_insights"]
        print(f"\n💡 Optimization Insights:")
        print(f"  Avg candidates saved (vs max {self.iterative_max_candidates}): {insights['avg_candidates_saved']:.1f}")
        print(f"  Estimated token savings: {insights['token_savings_from_early_stop']:.0f}")
        print(f"  Low confidence actions (< 0.2): {insights['low_confidence_actions']}")
        print(f"  High confidence actions (>= 0.5): {insights['high_confidence_actions']}")

        print("=" * 70 + "\n")
