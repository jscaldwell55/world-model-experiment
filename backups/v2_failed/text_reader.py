# agents/text_reader.py
from typing import Tuple, Optional
from agents.observer import ObserverAgent
from agents.base import LLMInterface
from experiments.prompts import TEXT_READER_QUERY_TEMPLATE, extract_answer_components


class TextReaderAgent(ObserverAgent):
    """
    Observer that reads prior episode logs.

    TextReader extends Observer by having access to formatted logs from
    previous episodes. This tests whether reading about interactions
    (without doing them) improves predictions compared to pure reasoning.
    """

    def __init__(
        self,
        llm: LLMInterface,
        action_budget: int,
        prior_logs: list[dict]
    ):
        """
        Initialize TextReader agent.

        Args:
            llm: LLM interface
            action_budget: Not used (TextReader takes no actions like Observer)
            prior_logs: List of episode logs to learn from
        """
        super().__init__(llm, action_budget)
        self.prior_experience = self._format_logs(prior_logs)

    def _format_logs(self, logs: list[dict]) -> str:
        """
        Convert episode logs to natural language for context.

        Args:
            logs: List of episode dictionaries with 'steps' and optional results

        Returns:
            Formatted string describing prior episodes
        """
        if not logs:
            return "No prior experience available."

        formatted = []

        # Use up to 10 most recent episodes
        for i, log in enumerate(logs[:10]):
            episode_summary = [f"\n=== Episode {i+1} ==="]

            # Format steps
            steps = log.get('steps', [])
            if steps:
                episode_summary.append("Actions taken:")

                for step in steps[:15]:  # Limit steps per episode
                    action = step.get('action')
                    observation = step.get('observation', {})

                    if action:
                        # Format observation concisely
                        obs_str = self._format_observation(observation)
                        episode_summary.append(f"  {action} -> {obs_str}")

            # Add test results if available
            if 'test_results' in log:
                results = log['test_results']
                if results:
                    correct_count = sum(1 for r in results if r.get('correct', False))
                    total = len(results)
                    accuracy = correct_count / total if total > 0 else 0.0
                    episode_summary.append(f"\nTest Accuracy: {accuracy:.1%} ({correct_count}/{total})")

            # Add ground truth if available (for learning)
            if 'ground_truth' in log:
                gt = log['ground_truth']
                episode_summary.append(f"Ground Truth: {gt}")

            formatted.append("\n".join(episode_summary))

        return "\n\n".join(formatted)

    def _format_observation(self, obs: dict) -> str:
        """
        Format observation dictionary concisely.

        Args:
            obs: Observation dictionary

        Returns:
            Concise string representation
        """
        # Extract key information
        parts = []

        for key, value in obs.items():
            if key in ['ground_truth', 'hidden_state']:
                continue  # Skip ground truth

            if isinstance(value, (int, float)):
                parts.append(f"{key}={value:.1f}")
            elif isinstance(value, bool):
                parts.append(f"{key}={value}")
            elif isinstance(value, str) and len(value) < 50:
                parts.append(f"{key}='{value}'")

        return ", ".join(parts[:5])  # Limit to 5 key-value pairs

    def answer_query(self, question: str) -> Tuple[str, float]:
        """
        Answer using initial description AND prior experience.

        Args:
            question: Question to answer

        Returns:
            Tuple of (answer, confidence)
        """
        if self.initial_description is None:
            return "No observations yet", 0.0

        prompt = TEXT_READER_QUERY_TEMPLATE.format(
            initial_description=str(self.initial_description),
            prior_experience=self.prior_experience,
            question=question
        )

        response = self.llm.generate(prompt)
        answer, confidence, reasoning = extract_answer_components(response)

        return answer, confidence

    def reset(self):
        """Reset agent state for new episode"""
        super().reset()
        # Note: prior_experience is NOT reset - it's fixed context
