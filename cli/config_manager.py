"""
Configuration manager for pull-assist CLI.

Manages user settings stored in ~/.pull-assist/config.json:
  - LLM server endpoint URL
  - API key for the proxy layer
  - GitHub personal access token
  - Model name

This replaces the .env file friction for end users.
"""

import json
from pathlib import Path
from typing import Optional

# Default config directory — follows XDG convention on macOS/Linux
CONFIG_DIR = Path.home() / ".pull-assist"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Default values (used if config file doesn't exist yet)
DEFAULTS = {
    "server": "",
    "api_key": "",
    "github_token": "",
    "model": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "registry_gist_id": "",
}


def _ensure_config_dir():
    """Create config directory if it doesn't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    """
    Load configuration from disk.
    Falls back to environment variables, then defaults.
    """
    import os

    config = dict(DEFAULTS)

    # Layer 1: config file on disk
    if CONFIG_FILE.exists():
        try:
            file_config = json.loads(CONFIG_FILE.read_text())
            for k, v in file_config.items():
                if v:  # don't overwrite defaults with empty strings
                    config[k] = v
        except (json.JSONDecodeError, OSError):
            pass  # corrupted config — fall through to env vars

    # Layer 2: environment variables override config file
    env_map = {
        "PA_SERVER":       "server",
        "LLM_BASE_URL":    "server",
        "PA_API_KEY":      "api_key",
        "LLM_API_KEY":     "api_key",
        "GITHUB_TOKEN":    "github_token",
        "PA_GITHUB_TOKEN": "github_token",
        "LLM_MODEL":      "model",
        "PA_MODEL":        "model",
    }
    for env_var, config_key in env_map.items():
        val = os.getenv(env_var)
        if val:
            config[config_key] = val

    return config


def save_config(config: dict):
    """Persist configuration to disk."""
    _ensure_config_dir()

    # Save all known keys
    all_keys = set(DEFAULTS.keys()) | set(config.keys())
    saveable = {k: config.get(k, "") for k in all_keys}
    CONFIG_FILE.write_text(json.dumps(saveable, indent=2) + "\n")


def get_value(key: str) -> Optional[str]:
    """Get a single config value."""
    config = load_config()
    return config.get(key)


def set_value(key: str, value: str):
    """Set a single config value and persist."""
    config = load_config()
    config[key] = value
    save_config(config)


def apply_config_to_env():
    """
    Push CLI config values into environment variables
    so the existing config/settings.py module picks them up
    without any code changes.

    Call this before importing any module that reads from env.
    """
    import os
    config = load_config()

    env_map = {
        "server":       "LLM_BASE_URL",
        "api_key":      "LLM_API_KEY",
        "github_token": "GITHUB_TOKEN",
        "model":        "LLM_MODEL",
    }

    for config_key, env_var in env_map.items():
        val = config.get(config_key)
        if val and not os.getenv(env_var):
            os.environ[env_var] = val
