# agents/base.py
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any
from dataclasses import dataclass, field
import time


@dataclass
class AgentStep:
    """Single agent step with full tracking"""
    timestamp: float
    step_num: int
    thought: str
    action: Optional[str]
    observation: dict
    belief_state: dict
    surprisal: float
    token_usage: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'step_num': self.step_num,
            'thought': self.thought,
            'action': self.action,
            'observation': self.observation,
            'belief_state': self.belief_state,
            'surprisal': self.surprisal,
            'token_usage': self.token_usage
        }


class Agent(ABC):
    """Base agent interface"""

    def __init__(self, llm: 'LLMInterface', action_budget: int):
        self.llm = llm
        self.action_budget = action_budget
        self.action_count = 0
        self.memory: list[AgentStep] = []

        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.api_call_count = 0

    @abstractmethod
    def act(self, observation: dict) -> AgentStep:
        """
        Process observation and return next step.

        Args:
            observation: Environment observation

        Returns:
            AgentStep with action and metadata
        """
        pass

    @abstractmethod
    def answer_query(self, question: str) -> Tuple[str, float]:
        """
        Answer query about environment.

        Args:
            question: Question to answer

        Returns:
            Tuple of (answer, confidence)
        """
        pass

    def reset(self):
        """Reset agent state for new episode"""
        self.action_count = 0
        self.memory = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.api_call_count = 0


class LLMInterface(ABC):
    """Abstract LLM interface for different providers"""

    def __init__(self):
        """Initialize token tracking."""
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_api_calls = 0
        self.rate_limiter = None  # Optional rate limiter

    def set_rate_limiter(self, rate_limiter):
        """
        Set rate limiter for API calls.

        Args:
            rate_limiter: RateLimiter instance
        """
        self.rate_limiter = rate_limiter

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Token count
        """
        pass

    def get_last_usage(self) -> Tuple[int, int]:
        """
        Get token usage from last API call.

        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        return (self.last_input_tokens, self.last_output_tokens)

    def get_total_usage(self) -> dict:
        """
        Get cumulative token usage stats.

        Returns:
            Dict with usage statistics
        """
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_api_calls': self.total_api_calls
        }


class OpenAILLM(LLMInterface):
    """OpenAI API wrapper"""

    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        Initialize OpenAI LLM.

        Args:
            model: Model name (e.g., 'gpt-4o-mini', 'gpt-4o')
            api_key: API key (if None, loads from config)
        """
        super().__init__()

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        from experiments.config import get_api_key

        self.api_key = api_key or get_api_key('openai')
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API"""
        # Estimate token usage for rate limiting
        estimated_input = self.count_tokens(prompt)
        estimated_output = kwargs.get('max_tokens', 2000) // 2  # Conservative estimate

        # Wait if rate limiter is set
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed(estimated_input, estimated_output)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2000),
                **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
            )

            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                self.last_input_tokens = response.usage.prompt_tokens
                self.last_output_tokens = response.usage.completion_tokens
                self.total_input_tokens += self.last_input_tokens
                self.total_output_tokens += self.last_output_tokens
                self.total_api_calls += 1

                # Update rate limiter with actual usage
                if self.rate_limiter:
                    self.rate_limiter.record_actual_usage(
                        self.last_input_tokens,
                        self.last_output_tokens
                    )
            else:
                # Fallback: estimate tokens
                self.last_input_tokens = self.count_tokens(prompt)
                self.last_output_tokens = self.count_tokens(response.choices[0].message.content)
                self.total_input_tokens += self.last_input_tokens
                self.total_output_tokens += self.last_output_tokens
                self.total_api_calls += 1

                # Update rate limiter with estimates
                if self.rate_limiter:
                    self.rate_limiter.record_actual_usage(
                        self.last_input_tokens,
                        self.last_output_tokens
                    )

            return response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Estimate token count (simplified)"""
        # Rough estimate: ~4 chars per token
        return len(text) // 4


class AnthropicLLM(LLMInterface):
    """Anthropic API wrapper"""

    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        Initialize Anthropic LLM.

        Args:
            model: Model name (e.g., 'claude-3-5-sonnet-20241022')
            api_key: API key (if None, loads from config)
        """
        super().__init__()

        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        from experiments.config import get_api_key

        self.api_key = api_key or get_api_key('anthropic')
        self.client = Anthropic(api_key=self.api_key)
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic API"""
        # Estimate token usage for rate limiting
        estimated_input = self.count_tokens(prompt)
        estimated_output = kwargs.get('max_tokens', 2000) // 2  # Conservative estimate

        # Wait if rate limiter is set
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed(estimated_input, estimated_output)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.pop('max_tokens', 2000),
                temperature=kwargs.pop('temperature', 0.7),
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )

            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                self.last_input_tokens = response.usage.input_tokens
                self.last_output_tokens = response.usage.output_tokens
                self.total_input_tokens += self.last_input_tokens
                self.total_output_tokens += self.last_output_tokens
                self.total_api_calls += 1

                # Update rate limiter with actual usage
                if self.rate_limiter:
                    self.rate_limiter.record_actual_usage(
                        self.last_input_tokens,
                        self.last_output_tokens
                    )
            else:
                # Fallback: estimate tokens
                self.last_input_tokens = self.count_tokens(prompt)
                self.last_output_tokens = self.count_tokens(response.content[0].text)
                self.total_input_tokens += self.last_input_tokens
                self.total_output_tokens += self.last_output_tokens
                self.total_api_calls += 1

                # Update rate limiter with estimates
                if self.rate_limiter:
                    self.rate_limiter.record_actual_usage(
                        self.last_input_tokens,
                        self.last_output_tokens
                    )

            return response.content[0].text

        except Exception as e:
            print(f"Anthropic API error: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Estimate token count (simplified)"""
        # Rough estimate: ~4 chars per token
        return len(text) // 4


class MockLLM(LLMInterface):
    """Mock LLM for testing without API calls"""

    def __init__(self, mock_responses: Optional[list[str]] = None):
        """
        Initialize mock LLM.

        Args:
            mock_responses: List of canned responses to cycle through
        """
        super().__init__()

        self.mock_responses = mock_responses or [
            "ANSWER: Unknown\nCONFIDENCE: 0.5\nREASONING: Mock response",
            "THOUGHT: Testing\nACTION: measure_temp()"
        ]
        self.call_count = 0

    def generate(self, prompt: str, **kwargs) -> str:
        """Return canned response"""
        response = self.mock_responses[self.call_count % len(self.mock_responses)]
        self.call_count += 1

        # Track mock token usage
        self.last_input_tokens = self.count_tokens(prompt)
        self.last_output_tokens = self.count_tokens(response)
        self.total_input_tokens += self.last_input_tokens
        self.total_output_tokens += self.last_output_tokens
        self.total_api_calls += 1

        return response

    def count_tokens(self, text: str) -> int:
        """Count tokens"""
        return len(text) // 4


def create_llm(
    model_name: str,
    api_key: Optional[str] = None,
    mock: bool = False
) -> LLMInterface:
    """
    Factory to create LLM interface.

    Args:
        model_name: Model identifier
        api_key: Optional API key
        mock: If True, return MockLLM for testing

    Returns:
        LLMInterface instance

    Raises:
        ValueError: If model name not recognized
    """
    if mock:
        return MockLLM()

    model_lower = model_name.lower()

    if model_lower.startswith('gpt') or model_lower.startswith('o1'):
        return OpenAILLM(model_name, api_key)
    elif model_lower.startswith('claude'):
        return AnthropicLLM(model_name, api_key)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported: gpt-*, o1-*, claude-*"
        )
