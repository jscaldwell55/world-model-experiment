"""
Cost tracking for LLM API usage.

Preregistration requirement: All costs must be tracked with exact pricing
from API providers as of October 2025.

This module provides precise cost calculations for experimental transparency
and reproducibility.
"""

from typing import Dict, Any
from datetime import date


class CostTracker:
    """
    Track and compute costs for LLM API usage.

    Pricing is snapshot from October 30, 2025 (Anthropic API pricing).
    All costs in USD.

    Preregistration commitment: This pricing snapshot is locked for the
    duration of the experiment. If pricing changes mid-experiment, we will
    note discrepancies but use these rates for consistency.
    """

    # Pricing snapshot: October 30, 2025
    # Source: https://www.anthropic.com/api (pricing as of this date)
    PRICING = {
        "claude-sonnet-4-5-20250929": {
            "input_per_1M": 3.00,   # $3.00 per 1M input tokens
            "output_per_1M": 15.00,  # $15.00 per 1M output tokens
            "snapshot_date": "2025-10-30"
        },
        "claude-sonnet-3-5-20241022": {
            "input_per_1M": 3.00,
            "output_per_1M": 15.00,
            "snapshot_date": "2025-10-30"
        },
        "gpt-4-0125-preview": {
            "input_per_1M": 10.00,   # $10.00 per 1M input tokens
            "output_per_1M": 30.00,  # $30.00 per 1M output tokens
            "snapshot_date": "2025-10-30"
        }
    }

    def __init__(self):
        """
        Initialize cost tracker.

        No state needed - all methods are stateless calculations.
        """
        pass

    def compute_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Compute cost for LLM API usage.

        Args:
            input_tokens: Number of input tokens (prompt)
            output_tokens: Number of output tokens (completion)
            model_name: Model identifier (e.g., "claude-sonnet-4-5-20250929")

        Returns:
            Dictionary with:
            - input_cost_usd: Cost of input tokens in USD
            - output_cost_usd: Cost of output tokens in USD
            - total_cost_usd: Total cost in USD
            - model: Model name used
            - pricing_snapshot: Pricing details and date

        Raises:
            ValueError: If model_name not in pricing table

        Example:
            >>> tracker = CostTracker()
            >>> cost = tracker.compute_cost(20000, 15000, "claude-sonnet-4-5-20250929")
            >>> print(f"Total: ${cost['total_cost_usd']:.3f}")
            Total: $0.285

        Preregistration note:
            - All calculations use exact pricing from snapshot date
            - Rounding to 6 decimal places for USD amounts
            - Total cost = input_cost + output_cost (no hidden fees)
        """
        if model_name not in self.PRICING:
            available = ", ".join(self.PRICING.keys())
            raise ValueError(
                f"Model '{model_name}' not in pricing table. "
                f"Available models: {available}"
            )

        pricing = self.PRICING[model_name]

        # Calculate costs (tokens / 1M * price_per_1M)
        input_cost_usd = (input_tokens / 1_000_000) * pricing["input_per_1M"]
        output_cost_usd = (output_tokens / 1_000_000) * pricing["output_per_1M"]
        total_cost_usd = input_cost_usd + output_cost_usd

        return {
            "input_cost_usd": round(input_cost_usd, 6),
            "output_cost_usd": round(output_cost_usd, 6),
            "total_cost_usd": round(total_cost_usd, 6),
            "model": model_name,
            "pricing_snapshot": {
                "input_per_1M": pricing["input_per_1M"],
                "output_per_1M": pricing["output_per_1M"],
                "date": pricing["snapshot_date"]
            }
        }

    def get_pricing_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get pricing information for a model.

        Args:
            model_name: Model identifier

        Returns:
            Dictionary with pricing details

        Raises:
            ValueError: If model_name not found
        """
        if model_name not in self.PRICING:
            raise ValueError(f"Model '{model_name}' not in pricing table")

        return self.PRICING[model_name].copy()

    def estimate_episode_cost(
        self,
        num_actions: int,
        tokens_per_action: int,
        model_name: str,
        include_evaluation: bool = True
    ) -> Dict[str, Any]:
        """
        Estimate cost for a full episode.

        Useful for budgeting before running experiments.

        Args:
            num_actions: Number of actions in episode (e.g., 10)
            tokens_per_action: Average tokens per action (input + output)
            model_name: Model to use
            include_evaluation: Whether to include evaluation overhead (~3K tokens)

        Returns:
            Cost breakdown dictionary

        Example:
            >>> tracker = CostTracker()
            >>> est = tracker.estimate_episode_cost(10, 2000, "claude-sonnet-4-5-20250929")
            >>> print(f"Estimated cost: ${est['total_cost_usd']:.2f}")
        """
        # Estimate input/output split (typically 60/40 for our prompts)
        total_tokens = num_actions * tokens_per_action

        if include_evaluation:
            total_tokens += 3000  # Evaluation overhead

        input_tokens = int(total_tokens * 0.6)
        output_tokens = int(total_tokens * 0.4)

        cost = self.compute_cost(input_tokens, output_tokens, model_name)
        cost["estimation_params"] = {
            "num_actions": num_actions,
            "tokens_per_action": tokens_per_action,
            "total_tokens_estimated": total_tokens,
            "include_evaluation": include_evaluation
        }

        return cost


# Convenience function for quick calculations
def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """
    Quick cost calculation (returns float USD).

    Args:
        input_tokens: Input token count
        output_tokens: Output token count
        model: Model name

    Returns:
        Total cost in USD (float)

    Example:
        >>> cost = calculate_cost(10000, 5000, "claude-sonnet-4-5-20250929")
        >>> print(f"${cost:.3f}")
        $0.105
    """
    tracker = CostTracker()
    result = tracker.compute_cost(input_tokens, output_tokens, model)
    return result["total_cost_usd"]


# Export main class and convenience function
__all__ = ["CostTracker", "calculate_cost"]
