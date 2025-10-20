# experiments/config.py
import os
from typing import Optional
from pathlib import Path


def get_api_key(provider: str) -> str:
    """
    Load API key from environment or .env file.

    Args:
        provider: 'openai' or 'anthropic'

    Returns:
        API key string

    Raises:
        ValueError: If API key not found
    """
    # Try environment variable first
    env_var_name = f"{provider.upper()}_API_KEY"
    api_key = os.environ.get(env_var_name)

    if api_key:
        return api_key

    # Try loading from .env file
    env_file = Path.cwd() / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith(env_var_name):
                    # Parse KEY=value format
                    if '=' in line:
                        api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                        if api_key:
                            return api_key

    raise ValueError(
        f"API key for {provider} not found. "
        f"Please set {env_var_name} environment variable or add to .env file."
    )


def load_config() -> dict:
    """
    Load experiment configuration.

    Returns:
        Configuration dictionary
    """
    return {
        'default_model': 'gpt-4o-mini',
        'max_tokens': 2000,
        'temperature': 0.7,
        'seed': 42,
        'action_budget': 10,
        'episode_timeout': 300,  # seconds
    }
