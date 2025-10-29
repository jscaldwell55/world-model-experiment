# evaluation/judge.py
"""
Vendor-disjoint judge for evaluating agent answers.

Two-tier approach:
1. Programmatic Judge: Exact match, numeric tolerance, structural checks (preferred)
2. LLM Judge: GPT-4 for semantic evaluation (vendor-disjoint from Sonnet agents)

Following preregistration requirements:
- LLM judge must use GPT-4 (vendor-disjoint from Claude Sonnet agents)
- Temperature = 0.0 for determinism
- Calibration set of 50 items for validation
"""

import re
import json
import os
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class JudgeResult:
    """Result from judging an answer"""
    score: float  # 0.0 to 1.0
    correct: bool  # True if score >= threshold
    reasoning: str  # Explanation of judgment
    judge_type: str  # "programmatic" or "llm"
    confidence: float = 1.0  # Judge's confidence in the decision


class ProgrammaticJudge:
    """
    Programmatic judge using exact match and numeric tolerance.

    Preferred method when ground truth is well-defined.
    """

    def __init__(self, numeric_tolerance: float = 0.01):
        """
        Initialize programmatic judge.

        Args:
            numeric_tolerance: Tolerance for numeric comparisons (default: 1%)
        """
        self.numeric_tolerance = numeric_tolerance

    def judge(
        self,
        answer: str,
        ground_truth: Any,
        context: Optional[Dict] = None
    ) -> JudgeResult:
        """
        Judge answer using programmatic rules.

        Args:
            answer: Agent's answer (string)
            ground_truth: Ground truth value (can be string, number, dict, etc.)
            context: Optional context (e.g., query type, environment state)

        Returns:
            JudgeResult with score, correctness, reasoning
        """
        # Normalize answer
        answer_normalized = self._normalize(answer)

        # Handle different ground truth types
        if isinstance(ground_truth, (int, float)):
            return self._judge_numeric(answer_normalized, ground_truth)
        elif isinstance(ground_truth, str):
            return self._judge_string(answer_normalized, ground_truth)
        elif isinstance(ground_truth, dict):
            return self._judge_structured(answer_normalized, ground_truth)
        elif isinstance(ground_truth, list):
            return self._judge_list(answer_normalized, ground_truth)
        else:
            return JudgeResult(
                score=0.0,
                correct=False,
                reasoning=f"Unknown ground truth type: {type(ground_truth)}",
                judge_type="programmatic",
                confidence=0.0
            )

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison"""
        return text.strip().lower()

    def _judge_numeric(self, answer: str, ground_truth: float) -> JudgeResult:
        """Judge numeric answer with tolerance"""
        # Extract number from answer
        numbers = re.findall(r'-?\d+\.?\d*', answer)

        if not numbers:
            return JudgeResult(
                score=0.0,
                correct=False,
                reasoning="No numeric value found in answer",
                judge_type="programmatic"
            )

        # Take first number found
        answer_value = float(numbers[0])

        # Check within tolerance
        if ground_truth == 0:
            # Absolute tolerance for zero
            within_tolerance = abs(answer_value) <= self.numeric_tolerance
        else:
            # Relative tolerance otherwise
            relative_error = abs(answer_value - ground_truth) / abs(ground_truth)
            within_tolerance = relative_error <= self.numeric_tolerance

        if within_tolerance:
            return JudgeResult(
                score=1.0,
                correct=True,
                reasoning=f"Answer {answer_value} matches ground truth {ground_truth} within tolerance",
                judge_type="programmatic"
            )
        else:
            return JudgeResult(
                score=0.0,
                correct=False,
                reasoning=f"Answer {answer_value} does not match ground truth {ground_truth} (tolerance: {self.numeric_tolerance})",
                judge_type="programmatic"
            )

    def _judge_string(self, answer: str, ground_truth: str) -> JudgeResult:
        """Judge string answer with exact match"""
        ground_truth_normalized = self._normalize(ground_truth)

        # Exact match
        if answer == ground_truth_normalized:
            return JudgeResult(
                score=1.0,
                correct=True,
                reasoning=f"Answer matches ground truth exactly",
                judge_type="programmatic"
            )

        # Substring match (ground truth in answer)
        if ground_truth_normalized in answer:
            return JudgeResult(
                score=0.8,
                correct=True,
                reasoning=f"Ground truth found in answer",
                judge_type="programmatic",
                confidence=0.8
            )

        # No match
        return JudgeResult(
            score=0.0,
            correct=False,
            reasoning=f"Answer does not match ground truth",
            judge_type="programmatic"
        )

    def _judge_structured(self, answer: str, ground_truth: Dict) -> JudgeResult:
        """Judge structured answer (e.g., wiring configuration)"""
        # Try to parse answer as JSON
        try:
            answer_parsed = json.loads(answer)
            if answer_parsed == ground_truth:
                return JudgeResult(
                    score=1.0,
                    correct=True,
                    reasoning="Structured answer matches ground truth",
                    judge_type="programmatic"
                )
        except json.JSONDecodeError:
            pass

        # Partial match: check if all ground truth items mentioned
        matches = 0
        total = len(ground_truth)

        for key, value in ground_truth.items():
            if str(key) in answer and str(value) in answer:
                matches += 1

        score = matches / total if total > 0 else 0.0

        return JudgeResult(
            score=score,
            correct=score >= 0.8,
            reasoning=f"Matched {matches}/{total} ground truth items",
            judge_type="programmatic",
            confidence=0.8
        )

    def _judge_list(self, answer: str, ground_truth: List) -> JudgeResult:
        """Judge list answer (e.g., multiple items)"""
        # Count how many ground truth items are mentioned
        matches = sum(1 for item in ground_truth if str(item).lower() in answer)

        score = matches / len(ground_truth) if ground_truth else 0.0

        return JudgeResult(
            score=score,
            correct=score >= 0.8,
            reasoning=f"Matched {matches}/{len(ground_truth)} ground truth items",
            judge_type="programmatic",
            confidence=0.8
        )


class LLMJudge:
    """
    LLM-based judge using GPT-4 (vendor-disjoint from Claude agents).

    Used when programmatic judging is insufficient (e.g., nuanced reasoning).
    Following preregistration: Temperature = 0.0, model = gpt-4-0125-preview
    """

    def __init__(
        self,
        model: str = "gpt-4-0125-preview",
        temperature: float = 0.0,
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM judge.

        Args:
            model: GPT-4 model version (vendor-disjoint from Claude)
            temperature: Temperature for generation (0.0 for determinism)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        self.model = model
        self.temperature = temperature

        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set. Required for vendor-disjoint judge.")

        self.client = openai.OpenAI(api_key=api_key)

        # Calibration stats (to be filled by calibrate method)
        self.calibration_accuracy = None
        self.calibration_size = 0

    def judge(
        self,
        answer: str,
        ground_truth: Any,
        context: Optional[Dict] = None
    ) -> JudgeResult:
        """
        Judge answer using GPT-4.

        Args:
            answer: Agent's answer
            ground_truth: Ground truth value
            context: Optional context (query, environment description, etc.)

        Returns:
            JudgeResult with score, correctness, reasoning
        """
        # Build prompt
        prompt = self._build_judge_prompt(answer, ground_truth, context)

        # Call GPT-4
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert judge evaluating agent responses. Be fair, precise, and explain your reasoning."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"}
            )

            # Parse response
            result_text = response.choices[0].message.content
            result_json = json.loads(result_text)

            score = float(result_json.get("score", 0.0))
            reasoning = result_json.get("reasoning", "No reasoning provided")
            confidence = float(result_json.get("confidence", 1.0))

            return JudgeResult(
                score=score,
                correct=score >= 0.5,
                reasoning=reasoning,
                judge_type="llm",
                confidence=confidence
            )

        except Exception as e:
            return JudgeResult(
                score=0.0,
                correct=False,
                reasoning=f"LLM judge error: {str(e)}",
                judge_type="llm",
                confidence=0.0
            )

    def _build_judge_prompt(
        self,
        answer: str,
        ground_truth: Any,
        context: Optional[Dict]
    ) -> str:
        """Build prompt for LLM judge"""
        context_str = ""
        if context:
            if "query" in context:
                context_str += f"Query: {context['query']}\n\n"
            if "environment" in context:
                context_str += f"Environment: {context['environment']}\n\n"

        prompt = f"""Grade the following agent answer against the ground truth.

{context_str}Agent Answer:
{answer}

Ground Truth:
{ground_truth}

Provide a score from 0.0 (completely wrong) to 1.0 (completely correct).
Consider:
- Factual correctness
- Logical reasoning
- Completeness of answer

Respond with JSON:
{{
    "score": <float 0.0-1.0>,
    "reasoning": "<explanation of your grading>",
    "confidence": <float 0.0-1.0, your confidence in this judgment>
}}
"""
        return prompt

    def calibrate(self, calibration_set: List[Dict]) -> Dict[str, float]:
        """
        Calibrate judge on known examples.

        Args:
            calibration_set: List of dicts with "answer", "ground_truth", "expected_score"

        Returns:
            Calibration statistics (accuracy, mean error, etc.)
        """
        if len(calibration_set) < 10:
            raise ValueError("Calibration set should have at least 10 items (target: 50)")

        errors = []
        agreements = 0

        for item in calibration_set:
            result = self.judge(
                answer=item["answer"],
                ground_truth=item["ground_truth"],
                context=item.get("context")
            )

            expected_score = item.get("expected_score", 1.0 if item.get("correct", True) else 0.0)
            error = abs(result.score - expected_score)
            errors.append(error)

            # Check agreement on binary decision
            expected_correct = expected_score >= 0.5
            if result.correct == expected_correct:
                agreements += 1

        self.calibration_accuracy = agreements / len(calibration_set)
        self.calibration_size = len(calibration_set)

        return {
            "accuracy": self.calibration_accuracy,
            "mean_absolute_error": sum(errors) / len(errors),
            "calibration_size": self.calibration_size,
            "model": self.model
        }


class HybridJudge:
    """
    Hybrid judge: Try programmatic first, fall back to LLM if needed.

    This is the recommended approach for the study.
    """

    def __init__(
        self,
        numeric_tolerance: float = 0.01,
        use_llm_fallback: bool = True,
        llm_model: str = "gpt-4-0125-preview"
    ):
        """
        Initialize hybrid judge.

        Args:
            numeric_tolerance: Tolerance for programmatic numeric comparisons
            use_llm_fallback: Whether to use LLM when programmatic judge uncertain
            llm_model: GPT-4 model for LLM fallback
        """
        self.programmatic = ProgrammaticJudge(numeric_tolerance=numeric_tolerance)
        self.use_llm_fallback = use_llm_fallback

        if use_llm_fallback:
            try:
                self.llm = LLMJudge(model=llm_model)
            except (ImportError, ValueError) as e:
                print(f"Warning: LLM judge not available: {e}")
                print("Falling back to programmatic judge only")
                self.use_llm_fallback = False
                self.llm = None
        else:
            self.llm = None

    def judge(
        self,
        answer: str,
        ground_truth: Any,
        context: Optional[Dict] = None
    ) -> JudgeResult:
        """
        Judge using programmatic first, LLM fallback if uncertain.

        Args:
            answer: Agent's answer
            ground_truth: Ground truth
            context: Optional context

        Returns:
            JudgeResult
        """
        # Try programmatic first
        prog_result = self.programmatic.judge(answer, ground_truth, context)

        # Use programmatic if confident
        if prog_result.confidence >= 0.9:
            return prog_result

        # Fall back to LLM if available and uncertain
        if self.use_llm_fallback and self.llm:
            llm_result = self.llm.judge(answer, ground_truth, context)

            # Combine reasoning
            combined_reasoning = (
                f"Programmatic: {prog_result.reasoning} (confidence: {prog_result.confidence:.2f})\n"
                f"LLM: {llm_result.reasoning} (confidence: {llm_result.confidence:.2f})"
            )

            # Use LLM result but note both were used
            return JudgeResult(
                score=llm_result.score,
                correct=llm_result.correct,
                reasoning=combined_reasoning,
                judge_type="hybrid",
                confidence=llm_result.confidence
            )

        # Return programmatic result if no LLM fallback
        return prog_result


# Convenience function for default judge
def create_judge(judge_type: str = "hybrid") -> Any:
    """
    Create judge instance.

    Args:
        judge_type: "programmatic", "llm", or "hybrid" (recommended)

    Returns:
        Judge instance
    """
    if judge_type == "programmatic":
        return ProgrammaticJudge()
    elif judge_type == "llm":
        return LLMJudge()
    elif judge_type == "hybrid":
        return HybridJudge()
    else:
        raise ValueError(f"Unknown judge type: {judge_type}")
