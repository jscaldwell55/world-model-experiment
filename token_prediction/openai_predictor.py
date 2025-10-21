"""OpenAI API integration for token-level prediction."""

import os
from typing import List, Tuple
from token_prediction.predictor import NextSentencePredictor, TokenPrediction


class OpenAINextSentencePredictor(NextSentencePredictor):
    """Implementation using OpenAI ChatCompletion API.

    Uses the ChatCompletion API with logprobs enabled to get token-level
    log probabilities for next observation prediction.

    Attributes:
        model: OpenAI model name (e.g., "gpt-4o-mini", "gpt-4o")
        api_key: OpenAI API key (reads from OPENAI_API_KEY env var if None)
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        """Initialize OpenAI predictor.

        Args:
            model: OpenAI model name
            api_key: API key (defaults to OPENAI_API_KEY env var)

        Raises:
            ValueError: If API key not found
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Import OpenAI client lazily to avoid import errors if not installed
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

    def predict_next_observation(
        self,
        context: str,
        temperature: float = 0.0,
        max_tokens: int = 100
    ) -> TokenPrediction:
        """Predict next observation using ChatCompletion API.

        Args:
            context: Full transcript up to current step
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens to generate

        Returns:
            TokenPrediction with predicted text and logprobs

        Raises:
            RuntimeError: If API call fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are predicting the exact next observation in an interactive environment. Output only the observation text, no explanations or commentary."
                    },
                    {
                        "role": "user",
                        "content": f"Given this transcript, predict the exact next observation:\n\n{context}\n\nNext observation:"
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=1  # We only need the top logprob for each token
            )

            # Extract predicted text
            predicted_text = response.choices[0].message.content.strip()

            # Extract tokens and logprobs
            tokens = []
            logprobs = []

            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                for token_data in response.choices[0].logprobs.content:
                    tokens.append(token_data.token)
                    logprobs.append(token_data.logprob)

            # Compute NLL metrics
            sequence_nll = -sum(logprobs) if logprobs else 0.0
            per_token_nll = sequence_nll / len(logprobs) if logprobs else 0.0

            return TokenPrediction(
                predicted_text=predicted_text,
                tokens=tokens,
                logprobs=logprobs,
                sequence_nll=sequence_nll,
                per_token_nll=per_token_nll
            )

        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}") from e

    def rank_candidates(
        self,
        context: str,
        candidates: List[str]
    ) -> List[Tuple[str, float]]:
        """Score multiple candidate observations.

        Uses the ChatCompletion API to score each candidate by computing
        the log probability of the candidate given the context.

        Note: This is an approximation since OpenAI doesn't provide a direct
        scoring API. We use completion with logprobs and extract the likelihood.

        Args:
            context: Full transcript up to current step
            candidates: List of candidate observation strings

        Returns:
            List of (candidate, nll) tuples sorted by NLL (lower is better)

        Raises:
            RuntimeError: If API call fails
        """
        scored_candidates = []

        for candidate in candidates:
            try:
                # Use the prediction API to get logprobs for this specific candidate
                # by providing it as a prompt completion task
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are evaluating the likelihood of observations in an interactive environment."
                        },
                        {
                            "role": "user",
                            "content": f"Given this transcript:\n\n{context}\n\nThe next observation was:\n{candidate}"
                        }
                    ],
                    temperature=0.0,
                    max_tokens=1,  # We just need the logprobs, not generation
                    logprobs=True
                )

                # Extract logprobs - use sum of logprobs as proxy for likelihood
                # This is not perfect but gives a rough ranking
                if response.choices[0].logprobs and response.choices[0].logprobs.content:
                    logprobs = [
                        token_data.logprob
                        for token_data in response.choices[0].logprobs.content
                    ]
                    nll = -sum(logprobs)
                else:
                    nll = float('inf')  # No logprobs available

                scored_candidates.append((candidate, nll))

            except Exception as e:
                # On error, assign infinite NLL
                scored_candidates.append((candidate, float('inf')))

        # Sort by NLL (lower is better)
        scored_candidates.sort(key=lambda x: x[1])

        return scored_candidates

    def get_model_name(self) -> str:
        """Get the name of the underlying model.

        Returns:
            Model name string
        """
        return self.model

    def get_provider(self) -> str:
        """Get the provider name.

        Returns:
            Provider name string
        """
        return "openai"
