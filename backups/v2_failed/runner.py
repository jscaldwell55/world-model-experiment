# experiments/runner.py
from typing import Optional, TYPE_CHECKING
from pathlib import Path
import json
import time

from experiments.provenance import ProvenanceLog
from agents.base import create_llm
from evaluation.tasks import get_test_queries
from utils.cost_tracker import CostTracker

if TYPE_CHECKING:
    from experiments.rate_limiter import RateLimiter


class ExperimentRunner:
    """Main experiment loop with guard rails"""

    def __init__(self, config: dict, environment_cls, agent_cls, shared_agent=None):
        """
        Initialize experiment runner.

        Args:
            config: Configuration dictionary
            environment_cls: Environment class
            agent_cls: Agent class
            shared_agent: Optional pre-existing agent instance to reuse (for multi-epoch ACE)
        """
        self.config = config
        self.environment_cls = environment_cls
        self.agent_cls = agent_cls
        self.shared_agent = shared_agent
        self._last_agent = None  # Track agent for sharing

        # Provenance
        import environments
        import agents as agent_module
        from experiments.config import load_config

        # Use provided config or load default
        full_config = config if config else load_config()

        self.provenance = ProvenanceLog(full_config, environments, agent_module)

    def run_episode(
        self,
        episode_id: str,
        seed: int,
        save_dir: Path,
        rate_limiter: Optional['RateLimiter'] = None
    ) -> dict:
        """
        Run single episode with strict guard rails:
        1. Observations injected programmatically (never echoed)
        2. Ground truth never exposed to agent
        3. All steps validated

        Args:
            episode_id: Unique episode identifier
            seed: Random seed
            save_dir: Directory to save episode log

        Returns:
            Episode log dictionary
        """
        # Track episode start time
        episode_start_time = time.time()

        # Initialize environment
        env = self.environment_cls(seed=seed)
        observation = env.reset(seed=seed)

        # Initialize agent with appropriate LLM
        # Convert PascalCase to snake_case: TextReaderAgent -> text_reader
        import re
        class_name = self.agent_cls.__name__.replace('Agent', '')
        agent_type = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()

        # Get model config - no defaults, must be in config
        if 'models' not in self.config:
            raise ValueError("Config must contain 'models' section")
        if agent_type not in self.config['models']:
            raise ValueError(f"Config must specify model for agent type '{agent_type}'")

        model_config = self.config['models'][agent_type]

        # Ensure model is specified
        if 'model' not in model_config:
            raise ValueError(f"Model config for '{agent_type}' must contain 'model' field")

        model_name = model_config['model']

        # Log which model is being used
        print(f"  Creating {agent_type} agent with model: {model_name}")

        # Create LLM - never use mocks in production
        # Mock LLM should only be used in unit tests via explicit mock=True
        use_mock = self.config.get('use_mock_llm', False)
        llm = create_llm(model_name, mock=use_mock)

        # Set rate limiter if provided
        if rate_limiter:
            llm.set_rate_limiter(rate_limiter)

        # Create agent with environment name if applicable
        agent_kwargs = {
            'llm': llm,
            'action_budget': self.config.get('budgets', {}).get('actions_per_episode', 10)
        }

        # Add environment_name for agents that need it
        if hasattr(self.agent_cls, '__init__'):
            import inspect
            sig = inspect.signature(self.agent_cls.__init__)
            if 'environment_name' in sig.parameters:
                agent_kwargs['environment_name'] = env.__class__.__name__

            # Add prior_logs for TextReaderAgent
            if 'prior_logs' in sig.parameters:
                # Note: TextReaderAgent not used in main study.
                # Future work: load prior episode logs for few-shot learning.
                agent_kwargs['prior_logs'] = []

            # Add ACE-specific parameters if this is an ACEAgent
            if agent_type == 'a_c_e' and 'ace_config' in self.config:
                ace_config = self.config['ace_config']

                # Map config to agent parameters
                ace_params = {
                    'use_retrieval': 'use_retrieval',
                    'top_k': 'top_k',
                    'reflection_rounds': 'reflection_rounds',
                    'generator_temperature': 'generator_temperature',
                    'reflector_temperature': 'reflector_temperature',
                    'curator_temperature': 'curator_temperature',
                    'curation_mode': 'curation_mode',
                    'token_cap': 'token_cap',
                    'max_epochs': 'max_epochs'
                }

                for config_key, param_name in ace_params.items():
                    if config_key in ace_config and param_name in sig.parameters:
                        agent_kwargs[param_name] = ace_config[config_key]

        # Use shared agent if provided (for multi-epoch ACE), otherwise create new one
        if self.shared_agent is not None:
            agent = self.shared_agent
            print(f"  Reusing shared agent (playbook size: {agent._get_playbook_size() if hasattr(agent, '_get_playbook_size') else 'unknown'})")
        else:
            agent = self.agent_cls(**agent_kwargs)

        # Store agent for potential sharing
        self._last_agent = agent

        # Set environment-specific belief if Actor
        if hasattr(agent, 'set_belief_state'):
            belief_cls = self._get_belief_class(env.__class__.__name__)
            if belief_cls:
                agent.set_belief_state(belief_cls())

        # Reset agent with prior generation (for Actor agents)
        # Pass environment type and initial observation for LLM-based prior generation
        if hasattr(agent, 'reset'):
            if hasattr(agent, 'set_belief_state'):
                # Actor agent - generate priors from initial observation
                agent.reset(
                    environment_type=env.__class__.__name__,
                    initial_observation=observation
                )
            else:
                # Other agent types - standard reset
                agent.reset()

        # Episode loop
        steps = []
        done = False
        step_num = 0
        max_steps = self.config.get('budgets', {}).get('actions_per_episode', 10)

        while not done and step_num < max_steps:
            # GUARD RAIL: Validate observation
            observation = self._validate_observation(observation)

            # GUARD RAIL: Never expose ground truth
            assert 'ground_truth' not in observation, "Ground truth leaked in observation!"
            assert 'hidden_state' not in observation, "Hidden state leaked in observation!"

            # Additional environment-specific checks
            if env.__class__.__name__ == 'HotPotLab':
                assert 'actual_temp' not in observation, "Actual temp leaked!"
                assert 'stove_power' not in observation, "Stove power leaked!"

            # Agent chooses action based on current observation
            # Note: At this point, agent computes surprisal/updates belief based on OLD observation
            # We'll recompute with the RESULT observation below
            agent_step = agent.act(observation)

            # Environment step - execute action and get RESULT observation
            if agent_step.action is not None:
                try:
                    # Normalize action: remove empty parentheses for simple actions
                    # e.g., "measure_temp()" -> "measure_temp"
                    # but preserve "wait(10)" as "wait(10)"
                    normalized_action = agent_step.action
                    if normalized_action.endswith('()'):
                        normalized_action = normalized_action[:-2]

                    new_observation, reward, done, info = env.step(normalized_action)
                    new_observation = self._validate_observation(new_observation)

                    # GUARD RAIL: Programmatically inject RESULT observation
                    # This prevents LLM from hallucinating observations
                    agent_step.observation = new_observation

                    # Recompute surprisal on the RESULT observation (before belief update)
                    # Surprisal measures how surprising the result was given prior belief
                    if hasattr(agent, 'compute_surprisal'):
                        agent_step.surprisal = agent.compute_surprisal(new_observation)

                    # Update belief with RESULT observation (if actor agent)
                    # This happens for ALL steps including step 0, since we're updating based on action results
                    if hasattr(agent, 'update_belief_from_observation'):
                        agent.update_belief_from_observation(new_observation)
                        agent_step.belief_state = agent.get_belief_state()

                    observation = new_observation

                except Exception as e:
                    # Invalid action - log and continue
                    print(f"Warning: Invalid action '{agent_step.action}': {e}")
                    observation = {'error': str(e), 'time': env.get_time_elapsed()}
                    agent_step.observation = observation
                    done = True
            else:
                # Observer or budget exhausted
                # No action taken, so observation doesn't change
                agent_step.observation = observation
                done = True

            steps.append(agent_step)
            step_num += 1

        # Evaluation (ground truth used ONLY here)
        ground_truth = env.get_ground_truth()
        test_queries = get_test_queries(env.__class__.__name__)
        test_results = self._evaluate_agent(agent, env, test_queries, ground_truth)

        # Update playbook for ACE agents (after test queries)
        if hasattr(agent, 'update_playbook'):
            agent.update_playbook({
                'test_results': test_results,
                'episode_steps': steps
            })

        # Get token usage from LLM
        llm_usage = llm.get_total_usage()

        # Calculate episode duration
        episode_duration = time.time() - episode_start_time

        # Calculate cost (preregistration requirement)
        cost_tracker = CostTracker()
        cost_info = cost_tracker.compute_cost(
            input_tokens=llm_usage['total_input_tokens'],
            output_tokens=llm_usage['total_output_tokens'],
            model_name=model_name
        )

        # Get token breakdown from agent (preregistration requirement)
        token_breakdown = None
        if hasattr(agent, 'get_token_breakdown'):
            token_breakdown = agent.get_token_breakdown()

            # Validate token breakdown matches totals
            if hasattr(agent, 'validate_token_accounting'):
                try:
                    agent.validate_token_accounting(
                        llm_usage['total_input_tokens'],
                        llm_usage['total_output_tokens']
                    )
                    token_breakdown['validation_passed'] = True
                except ValueError as e:
                    print(f"Warning: Token accounting validation failed: {e}")
                    token_breakdown['validation_passed'] = False
                    token_breakdown['validation_error'] = str(e)

        # Package episode log
        episode_log = {
            'episode_id': episode_id,
            'seed': seed,
            'environment': env.__class__.__name__,
            'agent_type': agent_type,
            'provenance': self.provenance.to_dict(),
            'steps': [step.to_dict() for step in steps],
            'test_results': test_results,
            'ground_truth': ground_truth,  # For evaluation only
            'timestamp': time.time(),
            # Token usage tracking
            'total_input_tokens': llm_usage['total_input_tokens'],
            'total_output_tokens': llm_usage['total_output_tokens'],
            'total_api_calls': llm_usage['total_api_calls'],
            'duration_seconds': episode_duration,
            # Cost tracking (preregistration requirement)
            'cost': cost_info,
            # Token breakdown by category (preregistration requirement)
            'token_breakdown': token_breakdown
        }

        # Add prior generation metadata if available (Actor agents)
        if hasattr(agent, 'prior_generation_metadata') and agent.prior_generation_metadata:
            episode_log['prior_generation'] = {
                'initial_priors': agent.prior_generation_metadata['priors'],
                'prior_reasoning': agent.prior_generation_metadata['reasoning'],
                'prior_generation_tokens': agent.prior_generation_metadata['token_count']
            }

        # Add playbook metadata if available (ACE agents)
        if hasattr(agent, 'playbook') and hasattr(agent, '_get_playbook_size'):
            episode_log['playbook'] = {
                'final_playbook': agent.playbook,
                'total_bullets': agent._get_playbook_size(),
                'sections': {
                    section: len(bullets)
                    for section, bullets in agent.playbook.items()
                }
            }
            # Add growth trajectory if episode_history available
            if hasattr(agent, 'episode_history') and agent.episode_history:
                last_episode = agent.episode_history[-1]
                if 'delta_items' in last_episode:
                    episode_log['playbook']['delta_items_added'] = len(last_episode['delta_items'])
                if 'insights' in last_episode:
                    episode_log['playbook']['insights'] = last_episode['insights']

        # Save
        save_path = save_dir / f"{episode_id}.json"
        with open(save_path, 'w') as f:
            json.dump(episode_log, f, indent=2)

        return episode_log

    def _validate_observation(self, obs: dict) -> dict:
        """
        Ensure observation structure is valid.

        Args:
            obs: Observation dictionary

        Returns:
            Validated observation
        """
        # Basic validation
        assert isinstance(obs, dict), "Observation must be dict"

        # Ensure no forbidden keys
        forbidden = ['ground_truth', 'hidden_state', 'actual_temp', 'stove_power',
                     'wire_layout', 'faulty_relay']
        for key in forbidden:
            if key in obs:
                # Remove forbidden key
                obs = {k: v for k, v in obs.items() if k not in forbidden}
                break

        return obs

    def _get_belief_class(self, env_name: str):
        """
        Get appropriate belief class for environment.

        Args:
            env_name: Environment class name

        Returns:
            Belief class or None
        """
        from models.belief_state import HotPotBelief, SwitchLightBelief, ChemTileBelief

        mapping = {
            'HotPotLab': HotPotBelief,
            'SwitchLight': SwitchLightBelief,
            'ChemTile': ChemTileBelief,
        }
        return mapping.get(env_name)

    def _evaluate_agent(
        self,
        agent,
        env,
        test_queries: list,
        ground_truth: dict
    ) -> list:
        """
        Evaluate agent on test queries.

        Agent answers WITHOUT ground truth access.
        Evaluation uses ground truth for scoring ONLY.

        Args:
            agent: Agent instance
            env: Environment instance
            test_queries: List of TestQuery objects
            ground_truth: Ground truth from environment

        Returns:
            List of test result dictionaries
        """
        results = []

        for query in test_queries:
            # Agent answers (no ground truth access)
            try:
                answer, confidence = agent.answer_query(query.question)
            except Exception as e:
                print(f"Warning: Query failed: {e}")
                answer = "Error"
                confidence = 0.0

            # Evaluate using query's evaluation function
            try:
                score = query.evaluate(answer, ground_truth)
            except Exception as e:
                print(f"Warning: Evaluation failed: {e}")
                score = 0.0

            # Convert score to boolean correct
            if isinstance(score, bool):
                correct = score
                score_value = 1.0 if score else 0.0
            elif isinstance(score, (int, float)):
                score_value = float(score)
                correct = score_value > 0.5
            else:
                score_value = 0.0
                correct = False

            results.append({
                'query': query.question,
                'query_type': query.query_type,
                'difficulty': query.difficulty,
                'agent_answer': answer,
                'confidence': confidence,
                'correct': correct,
                'score': score_value
            })

        return results
