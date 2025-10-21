"""Logging infrastructure for token predictions."""

from dataclasses import dataclass, asdict
from typing import List, Optional
import json
from pathlib import Path


@dataclass
class TokenLogEntry:
    """Single step of token prediction log.

    Attributes:
        step: Step number in episode
        context_text: Full transcript up to this step
        true_observation: Actual observation from environment
        predicted_text: Predicted observation from LLM
        tokens: List of tokens in prediction
        logprobs: List of log probabilities for each token
        sequence_nll: Sequence-level negative log-likelihood
        per_token_nll: Per-token normalized NLL
        belief_surprisal: Optional belief surprisal from agent
        accuracy: Optional accuracy metric for this prediction
    """
    step: int
    context_text: str
    true_observation: str
    predicted_text: str
    tokens: List[str]
    logprobs: List[float]
    sequence_nll: float
    per_token_nll: float
    belief_surprisal: Optional[float] = None
    accuracy: Optional[float] = None


class TokenLogger:
    """Logs token predictions for an episode.

    This class maintains a log of all token-level predictions made during
    an episode, including the context, true observation, predicted text,
    and token-level log probabilities.
    """

    def __init__(self, episode_id: str):
        """Initialize token logger.

        Args:
            episode_id: Unique identifier for this episode
        """
        self.episode_id = episode_id
        self.entries: List[TokenLogEntry] = []

    def log_step(self, entry: TokenLogEntry) -> None:
        """Add entry for a step.

        Args:
            entry: TokenLogEntry to add to log
        """
        self.entries.append(entry)

    def save(self, filepath: str) -> None:
        """Save log as JSON.

        Args:
            filepath: Path to save JSON file

        Raises:
            IOError: If file cannot be written
        """
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(filepath, 'w') as f:
                json.dump({
                    'episode_id': self.episode_id,
                    'num_steps': len(self.entries),
                    'entries': [asdict(e) for e in self.entries]
                }, f, indent=2)
        except Exception as e:
            raise IOError(f"Failed to save token log to {filepath}: {str(e)}") from e

    @staticmethod
    def load(filepath: str) -> 'TokenLogger':
        """Load logger from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            TokenLogger instance with loaded data

        Raises:
            IOError: If file cannot be read
            ValueError: If JSON format is invalid
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            episode_id = data.get('episode_id', 'unknown')
            logger = TokenLogger(episode_id)

            # Reconstruct entries
            for entry_dict in data.get('entries', []):
                entry = TokenLogEntry(**entry_dict)
                logger.entries.append(entry)

            return logger

        except FileNotFoundError:
            raise IOError(f"Token log file not found: {filepath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in token log: {str(e)}") from e
        except Exception as e:
            raise IOError(f"Failed to load token log from {filepath}: {str(e)}") from e

    def get_sequence_nlls(self) -> List[float]:
        """Get sequence NLL values for all steps.

        Returns:
            List of sequence NLL values
        """
        return [entry.sequence_nll for entry in self.entries]

    def get_per_token_nlls(self) -> List[float]:
        """Get per-token NLL values for all steps.

        Returns:
            List of per-token NLL values
        """
        return [entry.per_token_nll for entry in self.entries]

    def get_belief_surprisals(self) -> List[Optional[float]]:
        """Get belief surprisal values for all steps.

        Returns:
            List of belief surprisal values (may contain None)
        """
        return [entry.belief_surprisal for entry in self.entries]

    def compute_alignment_correlation(self) -> Optional[float]:
        """Compute correlation between token NLL and belief surprisal.

        Returns:
            Pearson correlation coefficient, or None if insufficient data

        Raises:
            ImportError: If numpy is not installed
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy required for correlation computation")

        # Extract non-None values
        nlls = []
        surprisals = []

        for entry in self.entries:
            if entry.belief_surprisal is not None:
                nlls.append(entry.sequence_nll)
                surprisals.append(entry.belief_surprisal)

        if len(nlls) < 2:
            return None  # Need at least 2 points for correlation

        # Compute Pearson correlation
        correlation = np.corrcoef(nlls, surprisals)[0, 1]

        return float(correlation)

    def __len__(self) -> int:
        """Get number of logged steps.

        Returns:
            Number of entries
        """
        return len(self.entries)

    def __repr__(self) -> str:
        """Get string representation.

        Returns:
            String representation of logger
        """
        return f"TokenLogger(episode_id={self.episode_id}, num_steps={len(self.entries)})"
