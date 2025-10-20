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


class LLMInterface(ABC):
    """Abstract LLM interface for different providers"""

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


class OpenAILLM(LLMInterface):
    """OpenAI API wrapper"""

    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        Initialize OpenAI LLM.

        Args:
            model: Model name (e.g., 'gpt-4o-mini', 'gpt-4o')
            api_key: API key (if None, loads from config)
        """
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
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2000),
                **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
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
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.pop('max_tokens', 2000),
                temperature=kwargs.pop('temperature', 0.7),
                messages=[{"role": "user", "content": prompt}],
                **kwargs
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
        self.mock_responses = mock_responses or [
            "ANSWER: Unknown\nCONFIDENCE: 0.5\nREASONING: Mock response",
            "THOUGHT: Testing\nACTION: measure_temp()"
        ]
        self.call_count = 0

    def generate(self, prompt: str, **kwargs) -> str:
        """Return canned response"""
        response = self.mock_responses[self.call_count % len(self.mock_responses)]
        self.call_count += 1
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
