"""Basic token-level metrics."""

from typing import List
import numpy as np


def compute_sequence_nll(logprobs: List[float]) -> float:
    """Compute sequence-level negative log-likelihood.

    Args:
        logprobs: List of log probabilities for each token

    Returns:
        Negative sum of log probabilities
    """
    if not logprobs:
        return 0.0
    return -sum(logprobs)


def compute_per_token_nll(logprobs: List[float]) -> float:
    """Compute per-token normalized NLL.

    Args:
        logprobs: List of log probabilities for each token

    Returns:
        Normalized NLL (sequence_nll / number of tokens)
    """
    if not logprobs:
        return 0.0
    return -sum(logprobs) / len(logprobs)


def compute_bits_per_word(logprobs: List[float], num_words: int) -> float:
    """Compute bits per word (normalized by word count).

    This converts NLL from nats to bits and normalizes by word count,
    which is a common metric in language modeling.

    Args:
        logprobs: List of log probabilities for each token
        num_words: Number of words in the sequence

    Returns:
        Bits per word
    """
    if num_words <= 0:
        return 0.0

    nll = -sum(logprobs)
    bits = nll / np.log(2)  # Convert nats to bits
    return bits / num_words


def compute_perplexity(logprobs: List[float]) -> float:
    """Compute perplexity from token log probabilities.

    Perplexity is exp(average negative log-likelihood), a common
    metric for language model performance.

    Args:
        logprobs: List of log probabilities for each token

    Returns:
        Perplexity value
    """
    if not logprobs:
        return float('inf')

    avg_nll = -sum(logprobs) / len(logprobs)
    return np.exp(avg_nll)


def compute_token_entropy(logprobs: List[float]) -> float:
    """Compute average entropy per token.

    This is just the average NLL, but named to emphasize the
    information-theoretic interpretation.

    Args:
        logprobs: List of log probabilities for each token

    Returns:
        Average entropy in nats per token
    """
    return compute_per_token_nll(logprobs)


def detect_high_surprisal_tokens(
    logprobs: List[float],
    tokens: List[str],
    threshold: float = -5.0
) -> List[tuple]:
    """Detect tokens with high surprisal (low probability).

    Args:
        logprobs: List of log probabilities for each token
        tokens: List of token strings
        threshold: Log probability threshold (tokens below this are high surprisal)

    Returns:
        List of (token_index, token, logprob) tuples for high-surprisal tokens
    """
    if len(logprobs) != len(tokens):
        raise ValueError("logprobs and tokens must have same length")

    high_surprisal = []
    for i, (token, logprob) in enumerate(zip(tokens, logprobs)):
        if logprob < threshold:
            high_surprisal.append((i, token, logprob))

    return high_surprisal


def compute_calibration_error(
    predicted_nlls: List[float],
    actual_accuracies: List[float],
    num_bins: int = 10
) -> float:
    """Compute calibration error between predicted uncertainty and actual accuracy.

    This bins predictions by NLL and compares the average NLL in each bin
    to the actual accuracy (fraction correct) in that bin.

    Args:
        predicted_nlls: List of predicted NLL values
        actual_accuracies: List of accuracy values (0 or 1 typically)
        num_bins: Number of bins for calibration analysis

    Returns:
        Expected calibration error (lower is better)
    """
    if len(predicted_nlls) != len(actual_accuracies):
        raise ValueError("predicted_nlls and actual_accuracies must have same length")

    if not predicted_nlls:
        return 0.0

    # Create bins
    min_nll = min(predicted_nlls)
    max_nll = max(predicted_nlls)
    bin_edges = np.linspace(min_nll, max_nll, num_bins + 1)

    total_error = 0.0
    total_count = 0

    for i in range(num_bins):
        # Find samples in this bin
        in_bin = [
            (nll, acc)
            for nll, acc in zip(predicted_nlls, actual_accuracies)
            if bin_edges[i] <= nll < bin_edges[i + 1]
        ]

        if not in_bin:
            continue

        # Compute average NLL and accuracy in bin
        avg_nll = np.mean([nll for nll, _ in in_bin])
        avg_acc = np.mean([acc for _, acc in in_bin])

        # Convert NLL to probability: p = exp(-NLL)
        # For calibration, we want: -log(avg_acc) ≈ avg_nll
        expected_nll = -np.log(max(avg_acc, 1e-10))  # Avoid log(0)

        # Accumulate weighted error
        error = abs(avg_nll - expected_nll)
        total_error += error * len(in_bin)
        total_count += len(in_bin)

    if total_count == 0:
        return 0.0

    return total_error / total_count


def compute_alignment_metrics(
    token_nlls: List[float],
    belief_surprisals: List[float]
) -> dict:
    """Compute alignment metrics between token NLL and belief surprisal.

    Args:
        token_nlls: List of token NLL values
        belief_surprisals: List of belief surprisal values

    Returns:
        Dictionary with alignment metrics:
        - pearson_r: Pearson correlation coefficient
        - spearman_r: Spearman rank correlation
        - mse: Mean squared error
        - mae: Mean absolute error
    """
    if len(token_nlls) != len(belief_surprisals):
        raise ValueError("token_nlls and belief_surprisals must have same length")

    if len(token_nlls) < 2:
        return {
            'pearson_r': None,
            'spearman_r': None,
            'mse': None,
            'mae': None
        }

    # Compute correlations
    pearson_r = np.corrcoef(token_nlls, belief_surprisals)[0, 1]

    from scipy.stats import spearmanr
    spearman_r, _ = spearmanr(token_nlls, belief_surprisals)

    # Compute errors
    errors = np.array(token_nlls) - np.array(belief_surprisals)
    mse = np.mean(errors ** 2)
    mae = np.mean(np.abs(errors))

    return {
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
        'mse': float(mse),
        'mae': float(mae)
    }
