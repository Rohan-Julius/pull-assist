"""
`pa config` subcommand.

Lets users set their GitHub token, LLM endpoint, API key,
and model without touching a .env file manually.

Config is stored at ~/.pull-assist/config.json.
"""

import sys
import click
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


@click.group("config")
def config():
    """Manage pull-assist configuration.

    \b
    EXAMPLES:
      pa config show                      Show current config
      pa config set --token ghp_...       Set GitHub token
      pa config set --server http://...   Set LLM server endpoint
      pa config set --key pa-abc123       Set API key
      pa config reset                     Reset to defaults
    """
    pass


@config.command("show")
def config_show():
    """Show current configuration (tokens are masked)."""
    from cli.banner import print_mini_banner
    print_mini_banner(console)

    from cli.config_manager import load_config, CONFIG_FILE

    cfg = load_config()

    table = Table(
        title="[bold]Current Configuration[/bold]",
        box=box.ROUNDED,
        show_lines=False,
    )
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value")
    table.add_column("Source", style="dim")

    import os

    for key, value in cfg.items():
        # Mask sensitive values
        display_value = _mask(value) if key in ("github_token", "api_key") else value

        # Determine source (env var, config file, or default)
        env_map = {
            "server": ["PA_SERVER", "LLM_BASE_URL"],
            "api_key": ["PA_API_KEY", "LLM_API_KEY"],
            "github_token": ["PA_GITHUB_TOKEN", "GITHUB_TOKEN"],
            "model": ["PA_MODEL", "LLM_MODEL"],
        }
        source = "default"
        for env_var in env_map.get(key, []):
            if os.getenv(env_var):
                source = f"env: {env_var}"
                break
        if CONFIG_FILE.exists():
            import json
            try:
                file_cfg = json.loads(CONFIG_FILE.read_text())
                if key in file_cfg and file_cfg[key]:
                    if source == "default":
                        source = f"~/.pull-assist/config.json"
            except (json.JSONDecodeError, OSError):
                pass

        table.add_row(key, display_value, source)

    console.print(table)
    console.print(f"\n[dim]Config file: {CONFIG_FILE}[/dim]")


@config.command("set")
@click.option("--key", type=str, help="API key (get from admin).")
@click.option("--token", "github_token", type=str, help="GitHub personal access token.")
@click.option("--model", type=str, help="LLM model name.", hidden=True)
@click.option("--server", type=str, help="Override server URL directly.", hidden=True)
def config_set(key, github_token, model, server):
    """Set configuration values.

    \b
    EXAMPLES:
      pa config set --key pa-abc123 --token ghp_abc123
      pa config set --token ghp_abc123
      pa config set --key pa-abc123
    """
    from cli.config_manager import load_config, save_config

    if not any([key, github_token, model, server]):
        console.print("[yellow]No values provided. Use --help to see options.[/yellow]")
        return

    cfg = load_config()
    changes = []

    if key:
        cfg["api_key"] = key
        changes.append(("api_key", _mask(key)))
    if github_token:
        cfg["github_token"] = github_token
        changes.append(("github_token", _mask(github_token)))
    if model:
        cfg["model"] = model
        changes.append(("model", model))
    if server:
        cfg["server"] = server
        changes.append(("server", server))

    save_config(cfg)

    console.print("[green]✓ Configuration updated:[/green]")
    for k, v in changes:
        console.print(f"  {k} = {v}")
    console.print(f"\n[dim]Run 'pa status' to verify connectivity.[/dim]")


@config.command("reset")
@click.confirmation_option(prompt="Reset all configuration to defaults?")
def config_reset():
    """Reset configuration to defaults."""
    from cli.config_manager import DEFAULTS, save_config

    save_config(dict(DEFAULTS))
    console.print("[green]✓ Configuration reset to defaults.[/green]")


def _mask(value: str) -> str:
    """Mask a sensitive value, showing only first 4 and last 4 chars."""
    if not value or len(value) <= 8:
        return "****"
    return value[:4] + "•" * (len(value) - 8) + value[-4:]
