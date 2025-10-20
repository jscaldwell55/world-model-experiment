# experiments/runner.py
from typing import Optional
from pathlib import Path
import json
import time

from experiments.provenance import ProvenanceLog
from agents.base import create_llm
from evaluation.tasks import get_test_queries


class ExperimentRunner:
    """Main experiment loop with guard rails"""

    def __init__(self, config: dict, environment_cls, agent_cls):
        """
        Initialize experiment runner.

        Args:
            config: Configuration dictionary
            environment_cls: Environment class
            agent_cls: Agent class
        """
        self.config = config
        self.environment_cls = environment_cls
        self.agent_cls = agent_cls

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
        save_dir: Path
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
        # Initialize environment
        env = self.environment_cls(seed=seed)
        observation = env.reset(seed=seed)

        # Initialize agent with appropriate LLM
        agent_type = self.agent_cls.__name__.lower().replace('agent', '')
        model_config = self.config.get('models', {}).get(agent_type,
                                                          self.config.get('models', {}).get('actor',
                                                                                             {'model': 'gpt-4o-mini'}))

        # Create LLM - never use mocks in production
        # Mock LLM should only be used in unit tests via explicit mock=True
        use_mock = self.config.get('use_mock_llm', False)
        llm = create_llm(model_config.get('model', 'gpt-4o-mini'), mock=use_mock)

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

        agent = self.agent_cls(**agent_kwargs)

        # Set environment-specific belief if Actor
        if hasattr(agent, 'set_belief_state'):
            belief_cls = self._get_belief_class(env.__class__.__name__)
            if belief_cls:
                agent.set_belief_state(belief_cls())

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

            # Agent acts
            agent_step = agent.act(observation)

            # GUARD RAIL: Programmatically inject observation (override any LLM hallucination)
            agent_step.observation = observation

            steps.append(agent_step)

            # Environment step
            if agent_step.action is not None:
                try:
                    # Normalize action: remove empty parentheses for simple actions
                    # e.g., "measure_temp()" -> "measure_temp"
                    # but preserve "wait(10)" as "wait(10)"
                    normalized_action = agent_step.action
                    if normalized_action.endswith('()'):
                        normalized_action = normalized_action[:-2]

                    observation, reward, done, info = env.step(normalized_action)
                except Exception as e:
                    # Invalid action - log and continue
                    print(f"Warning: Invalid action '{agent_step.action}': {e}")
                    observation = {'error': str(e), 'time': env.get_time_elapsed()}
                    done = True
            else:
                done = True  # Observer or budget exhausted

            step_num += 1

        # Evaluation (ground truth used ONLY here)
        ground_truth = env.get_ground_truth()
        test_queries = get_test_queries(env.__class__.__name__)
        test_results = self._evaluate_agent(agent, env, test_queries, ground_truth)

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
            'timestamp': time.time()
        }

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
