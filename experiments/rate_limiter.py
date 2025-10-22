# experiments/rate_limiter.py
"""
Thread-safe rate limiter for API calls using sliding window algorithm.

Tracks requests per minute (RPM) and tokens per minute (TPM) to stay
under API rate limits with a safety buffer.
"""
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Tuple


class RateLimiter:
    """
    Thread-safe rate limiter using sliding window algorithm.

    Maintains sliding 1-minute windows for:
    - Requests per minute (RPM)
    - Input tokens per minute (TPM)
    - Output tokens per minute (TPM)

    Blocks API calls if they would exceed configured limits.
    """

    def __init__(
        self,
        rpm: int = 1000,
        input_tpm: int = 450000,
        output_tpm: int = 90000,
        safety_factor: float = 0.9
    ):
        """
        Initialize rate limiter.

        Args:
            rpm: Requests per minute limit
            input_tpm: Input tokens per minute limit
            output_tpm: Output tokens per minute limit
            safety_factor: Fraction of limits to use (0.9 = 90%)
        """
        # Apply safety buffer to all limits
        self.rpm_limit = int(rpm * safety_factor)
        self.input_tpm_limit = int(input_tpm * safety_factor)
        self.output_tpm_limit = int(output_tpm * safety_factor)

        # Sliding windows: deques of (timestamp, value) tuples
        self.requests = deque()
        self.input_tokens = deque()
        self.output_tokens = deque()

        # Thread safety
        self.lock = threading.Lock()

        # Statistics
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_wait_time = 0.0
        self.wait_count = 0

    def wait_if_needed(
        self,
        estimated_input: int = 1000,
        estimated_output: int = 200
    ) -> float:
        """
        Block if adding this request would exceed any limit.

        Args:
            estimated_input: Estimated input tokens for this request
            estimated_output: Estimated output tokens for this request

        Returns:
            Seconds waited (0.0 if no wait needed)
        """
        with self.lock:
            now = datetime.now()
            self._expire_old_entries(now)

            # Calculate wait time needed for each limit
            wait_times = []

            # Check RPM limit
            if len(self.requests) >= self.rpm_limit:
                wait_times.append(self._time_until_window_frees(self.requests))

            # Check input TPM limit
            current_input = sum(tokens for _, tokens in self.input_tokens)
            if current_input + estimated_input > self.input_tpm_limit:
                wait_times.append(self._time_until_tokens_free(
                    self.input_tokens,
                    current_input + estimated_input - self.input_tpm_limit
                ))

            # Check output TPM limit
            current_output = sum(tokens for _, tokens in self.output_tokens)
            if current_output + estimated_output > self.output_tpm_limit:
                wait_times.append(self._time_until_tokens_free(
                    self.output_tokens,
                    current_output + estimated_output - self.output_tpm_limit
                ))

            # Wait if needed
            wait_seconds = 0.0
            if wait_times:
                wait_seconds = max(wait_times) + 0.5  # Small buffer
                self.total_wait_time += wait_seconds
                self.wait_count += 1

                # Release lock while waiting
                self.lock.release()
                try:
                    time.sleep(wait_seconds)
                finally:
                    self.lock.acquire()

                # Expire entries again after waiting
                now = datetime.now()
                self._expire_old_entries(now)

            # Record this request with estimates
            self._record_request(now, estimated_input, estimated_output)

            return wait_seconds

    def record_actual_usage(
        self,
        actual_input: int,
        actual_output: int
    ):
        """
        Update the most recent request with actual token usage.

        This adjusts the token counts from estimates to actuals.

        Args:
            actual_input: Actual input tokens used
            actual_output: Actual output tokens used
        """
        with self.lock:
            if not self.requests:
                return

            # Get the most recent request timestamp
            last_timestamp = self.requests[-1][0]

            # Find and update corresponding token entries
            if self.input_tokens and self.input_tokens[-1][0] == last_timestamp:
                estimated_input = self.input_tokens[-1][1]
                self.input_tokens[-1] = (last_timestamp, actual_input)
                self.total_input_tokens += (actual_input - estimated_input)

            if self.output_tokens and self.output_tokens[-1][0] == last_timestamp:
                estimated_output = self.output_tokens[-1][1]
                self.output_tokens[-1] = (last_timestamp, actual_output)
                self.total_output_tokens += (actual_output - estimated_output)

    def get_current_usage(self) -> dict:
        """
        Get current usage stats.

        Returns:
            Dict with current RPM and TPM usage
        """
        with self.lock:
            now = datetime.now()
            self._expire_old_entries(now)

            return {
                'requests_per_minute': len(self.requests),
                'input_tokens_per_minute': sum(t for _, t in self.input_tokens),
                'output_tokens_per_minute': sum(t for _, t in self.output_tokens),
                'rpm_limit': self.rpm_limit,
                'input_tpm_limit': self.input_tpm_limit,
                'output_tpm_limit': self.output_tpm_limit,
                'rpm_utilization': len(self.requests) / self.rpm_limit if self.rpm_limit > 0 else 0,
                'input_tpm_utilization': sum(t for _, t in self.input_tokens) / self.input_tpm_limit if self.input_tpm_limit > 0 else 0,
                'output_tpm_utilization': sum(t for _, t in self.output_tokens) / self.output_tpm_limit if self.output_tpm_limit > 0 else 0,
            }

    def get_total_stats(self) -> dict:
        """
        Get total cumulative statistics.

        Returns:
            Dict with total usage stats
        """
        with self.lock:
            return {
                'total_requests': self.total_requests,
                'total_input_tokens': self.total_input_tokens,
                'total_output_tokens': self.total_output_tokens,
                'total_wait_time': self.total_wait_time,
                'wait_count': self.wait_count,
                'avg_wait_time': self.total_wait_time / self.wait_count if self.wait_count > 0 else 0
            }

    def _expire_old_entries(self, now: datetime):
        """Remove entries older than 1 minute from all windows."""
        cutoff = now - timedelta(minutes=1)

        while self.requests and self.requests[0][0] < cutoff:
            self.requests.popleft()

        while self.input_tokens and self.input_tokens[0][0] < cutoff:
            self.input_tokens.popleft()

        while self.output_tokens and self.output_tokens[0][0] < cutoff:
            self.output_tokens.popleft()

    def _time_until_window_frees(self, window: deque) -> float:
        """
        Calculate seconds until oldest entry expires.

        Args:
            window: Deque of (timestamp, value) tuples

        Returns:
            Seconds to wait
        """
        if not window:
            return 0.0

        oldest_timestamp = window[0][0]
        expire_time = oldest_timestamp + timedelta(minutes=1)
        now = datetime.now()

        wait_seconds = (expire_time - now).total_seconds()
        return max(0.0, wait_seconds)

    def _time_until_tokens_free(self, window: deque, tokens_needed: int) -> float:
        """
        Calculate seconds until enough tokens are freed.

        Args:
            window: Deque of (timestamp, tokens) tuples
            tokens_needed: Number of tokens that need to be freed

        Returns:
            Seconds to wait
        """
        if not window or tokens_needed <= 0:
            return 0.0

        # Find when we'll have freed enough tokens
        freed = 0
        target_timestamp = None

        for timestamp, tokens in window:
            freed += tokens
            if freed >= tokens_needed:
                target_timestamp = timestamp
                break

        if target_timestamp is None:
            # Not enough tokens in window, wait for oldest to expire
            target_timestamp = window[0][0]

        expire_time = target_timestamp + timedelta(minutes=1)
        now = datetime.now()

        wait_seconds = (expire_time - now).total_seconds()
        return max(0.0, wait_seconds)

    def _record_request(
        self,
        timestamp: datetime,
        input_tokens: int,
        output_tokens: int
    ):
        """
        Record a new request in all windows.

        Args:
            timestamp: When the request was made
            input_tokens: Input tokens (estimate or actual)
            output_tokens: Output tokens (estimate or actual)
        """
        self.requests.append((timestamp, 1))
        self.input_tokens.append((timestamp, input_tokens))
        self.output_tokens.append((timestamp, output_tokens))

        self.total_requests += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens


# Token usage estimates per agent type (based on empirical data)
TOKEN_ESTIMATES = {
    'observer': {
        'input': 8000,
        'output': 1200
    },
    'actor': {
        'input': 20000,
        'output': 4000
    },
    'model_based': {
        'input': 28000,
        'output': 5000
    },
    'text_reader': {
        'input': 12000,
        'output': 2000
    }
}


def get_token_estimate(agent_type: str) -> Tuple[int, int]:
    """
    Get token usage estimate for agent type.

    Args:
        agent_type: Type of agent (observer, actor, etc.)

    Returns:
        Tuple of (estimated_input, estimated_output) tokens
    """
    estimates = TOKEN_ESTIMATES.get(agent_type, {'input': 15000, 'output': 3000})
    return estimates['input'], estimates['output']
