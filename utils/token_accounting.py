"""
Token accounting system for detailed usage breakdown.

Preregistration requirement: Track token usage by category to understand
where computational resources are spent.

Categories:
- exploration: Tokens used for interacting with environment (action selection)
- curation: Tokens used for belief updates, playbook updates (ACE)
- evaluation: Tokens used for answering test queries
- planning: Tokens used for prior generation, planning (Actor, ACE)

Validation: Sum of all categories must equal total tokens (within rounding).
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class TokenRecord:
    """Record of token usage for a single API call"""
    category: str  # exploration, curation, evaluation, planning
    input_tokens: int
    output_tokens: int
    total_tokens: int
    metadata: Optional[Dict] = None  # Additional context (step_num, action, etc.)

    def __post_init__(self):
        # Validate that total matches sum
        expected_total = self.input_tokens + self.output_tokens
        if self.total_tokens != expected_total:
            raise ValueError(
                f"Total tokens ({self.total_tokens}) != input + output "
                f"({expected_total})"
            )


class TokenAccountant:
    """
    Track and validate token usage across categories.

    Usage:
        accountant = TokenAccountant()
        accountant.record('exploration', input_tokens=100, output_tokens=50, metadata={'step': 0})
        accountant.record('curation', input_tokens=200, output_tokens=100, metadata={'step': 0})
        breakdown = accountant.get_breakdown()
        accountant.validate(total_input=300, total_output=150)  # Passes
    """

    VALID_CATEGORIES = {'exploration', 'curation', 'evaluation', 'planning'}

    def __init__(self):
        self.records: List[TokenRecord] = []

    def record(
        self,
        category: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict] = None
    ):
        """
        Record token usage for a category.

        Args:
            category: One of {exploration, curation, evaluation, planning}
            input_tokens: Input tokens for this API call
            output_tokens: Output tokens for this API call
            metadata: Optional metadata (step_num, action, etc.)

        Raises:
            ValueError: If category is invalid or tokens are negative
        """
        if category not in self.VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. "
                f"Must be one of {self.VALID_CATEGORIES}"
            )

        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts cannot be negative")

        total_tokens = input_tokens + output_tokens

        record = TokenRecord(
            category=category,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            metadata=metadata or {}
        )

        self.records.append(record)

    def get_breakdown(self) -> Dict[str, Dict[str, int]]:
        """
        Get token breakdown by category.

        Returns:
            Dictionary with structure:
            {
                'exploration': {'input': X, 'output': Y, 'total': Z},
                'curation': {...},
                'evaluation': {...},
                'planning': {...},
                'totals': {'input': sum, 'output': sum, 'total': sum}
            }
        """
        breakdown = {}

        # Initialize all categories to zero
        for category in self.VALID_CATEGORIES:
            breakdown[category] = {
                'input': 0,
                'output': 0,
                'total': 0
            }

        # Aggregate by category
        for record in self.records:
            breakdown[record.category]['input'] += record.input_tokens
            breakdown[record.category]['output'] += record.output_tokens
            breakdown[record.category]['total'] += record.total_tokens

        # Compute totals
        total_input = sum(cat['input'] for cat in breakdown.values())
        total_output = sum(cat['output'] for cat in breakdown.values())
        total_all = sum(cat['total'] for cat in breakdown.values())

        breakdown['totals'] = {
            'input': total_input,
            'output': total_output,
            'total': total_all
        }

        return breakdown

    def validate(
        self,
        total_input: int,
        total_output: int,
        tolerance: int = 0
    ) -> bool:
        """
        Validate that breakdown matches reported totals.

        Args:
            total_input: Expected total input tokens
            total_output: Expected total output tokens
            tolerance: Allowed difference (default 0 for exact match)

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails with detailed error message
        """
        breakdown = self.get_breakdown()
        recorded_input = breakdown['totals']['input']
        recorded_output = breakdown['totals']['output']

        # Check input tokens
        input_diff = abs(recorded_input - total_input)
        if input_diff > tolerance:
            raise ValueError(
                f"Input token mismatch: recorded {recorded_input}, "
                f"expected {total_input} (diff: {input_diff})"
            )

        # Check output tokens
        output_diff = abs(recorded_output - total_output)
        if output_diff > tolerance:
            raise ValueError(
                f"Output token mismatch: recorded {recorded_output}, "
                f"expected {total_output} (diff: {output_diff})"
            )

        return True

    def get_category_percentage(self, category: str) -> float:
        """
        Get percentage of total tokens used by a category.

        Args:
            category: Category name

        Returns:
            Percentage (0-100)
        """
        if category not in self.VALID_CATEGORIES:
            raise ValueError(f"Invalid category '{category}'")

        breakdown = self.get_breakdown()
        category_total = breakdown[category]['total']
        overall_total = breakdown['totals']['total']

        if overall_total == 0:
            return 0.0

        return (category_total / overall_total) * 100

    def get_summary(self) -> Dict[str, any]:
        """
        Get human-readable summary of token usage.

        Returns:
            Dictionary with breakdown and percentages
        """
        breakdown = self.get_breakdown()
        total = breakdown['totals']['total']

        summary = {
            'breakdown': breakdown,
            'percentages': {},
            'total_tokens': total,
            'total_records': len(self.records)
        }

        for category in self.VALID_CATEGORIES:
            if total > 0:
                pct = (breakdown[category]['total'] / total) * 100
                summary['percentages'][category] = round(pct, 2)
            else:
                summary['percentages'][category] = 0.0

        return summary

    def to_dict(self) -> Dict:
        """
        Export accountant state to dictionary (for episode logs).

        Returns:
            Dictionary with breakdown and all records
        """
        return {
            'breakdown': self.get_breakdown(),
            'records': [asdict(r) for r in self.records],
            'validation_passed': True  # Will be set by validate()
        }

    def reset(self):
        """Clear all records (for new episode)"""
        self.records.clear()


# Convenience function for quick validation
def validate_token_breakdown(
    breakdown: Dict[str, Dict[str, int]],
    total_input: int,
    total_output: int,
    tolerance: int = 0
) -> bool:
    """
    Validate a token breakdown dictionary.

    Args:
        breakdown: Breakdown dict from get_breakdown()
        total_input: Expected total input tokens
        total_output: Expected total output tokens
        tolerance: Allowed difference

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    recorded_input = breakdown['totals']['input']
    recorded_output = breakdown['totals']['output']

    input_diff = abs(recorded_input - total_input)
    output_diff = abs(recorded_output - total_output)

    if input_diff > tolerance:
        raise ValueError(
            f"Input token mismatch: {recorded_input} != {total_input} "
            f"(diff: {input_diff})"
        )

    if output_diff > tolerance:
        raise ValueError(
            f"Output token mismatch: {recorded_output} != {total_output} "
            f"(diff: {output_diff})"
        )

    return True


# Export main classes
__all__ = ["TokenAccountant", "TokenRecord", "validate_token_breakdown"]
