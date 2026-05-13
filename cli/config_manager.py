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
    so the existing config/settings.py module picks them up.

    For the server URL, auto-discovers via the registry (GitHub Gist)
    so agents always connect to the current GPU server, even when
    the IP changes. This is the same registry that `pa status` uses.

    CLI config always takes precedence over .env file values,
    because .env is for local development while CLI config
    represents the user's explicit settings.
    """
    import os
    config = load_config()

    # ── Resolve server URL via registry (same as pa status) ───────────────
    try:
        from cli.registry import fetch_server_url
        registry = fetch_server_url()
        registry_url = registry.get("server_url")
        if registry_url and registry.get("active", True):
            config["server"] = registry_url
    except Exception:
        pass  # fall through to whatever server is in config

    env_map = {
        "server":       "LLM_BASE_URL",
        "api_key":      "LLM_API_KEY",
        "github_token": "GITHUB_TOKEN",
        "model":        "LLM_MODEL",
    }

    for config_key, env_var in env_map.items():
        val = config.get(config_key)
        if val:
            os.environ[env_var] = val

    # Reload settings.py module-level variables that were read at import time
    try:
        import config.settings as settings
        settings.LLM_BASE_URL = os.getenv("LLM_BASE_URL", settings.LLM_BASE_URL)
        settings.LLM_API_KEY = os.getenv("LLM_API_KEY", settings.LLM_API_KEY)
        settings.LLM_MODEL = os.getenv("LLM_MODEL", settings.LLM_MODEL)
        settings.GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", settings.GITHUB_TOKEN)
        if os.getenv("LLM_MAX_TOKENS"):
            settings.LLM_MAX_TOKENS = int(os.environ["LLM_MAX_TOKENS"])
        if os.getenv("USE_NATIVE_TOOL_CALLING") is not None:
            settings.USE_NATIVE_TOOL_CALLING = os.getenv(
                "USE_NATIVE_TOOL_CALLING", "false"
            ).lower() in ("1", "true", "yes")
    except ImportError:
        pass  # settings not yet imported — env vars will be read on first import
