# experiments/config.py
import os
import yaml
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


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load experiment configuration from YAML file.

    Args:
        config_path: Path to config.yaml file (default: config.yaml in project root)

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
    """
    if config_path is None:
        config_path = Path.cwd() / 'config.yaml'
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Please create a config.yaml file with model specifications."
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    if 'models' not in config:
        raise ValueError("Config must contain 'models' section")

    required_models = ['observer', 'actor', 'text_reader']
    for model_type in required_models:
        if model_type not in config['models']:
            raise ValueError(f"Config must specify model for '{model_type}'")

    return config
