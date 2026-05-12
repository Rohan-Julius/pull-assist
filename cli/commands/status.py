"""
`pa status` subcommand.

Checks connectivity and configuration:
  - GitHub token valid?
  - LLM endpoint reachable?
  - Memory store accessible?
  - Python version / dependencies OK?

Critical quality-of-life feature for onboarding.
"""

import sys
import time
import click
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.command("status")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Show detailed output for each check.")
def status(verbose):
    """Check connectivity: GitHub token, LLM endpoint, memory store, deps.

    \b
    Run this after 'pa config set' to verify everything works.
    """
    from cli.banner import print_mini_banner
    print_mini_banner(console)

    from cli.config_manager import apply_config_to_env, load_config
    apply_config_to_env()

    cfg = load_config()
    all_ok = True

    console.print("[bold]Running connectivity checks...[/bold]\n")

    # ── 1. GitHub Token ───────────────────────────────────────────────────────
    github_ok = _check_github(cfg, verbose)
    all_ok = all_ok and github_ok

    # ── 2. LLM Endpoint ──────────────────────────────────────────────────────
    llm_ok = _check_llm(cfg, verbose)
    all_ok = all_ok and llm_ok

    # ── 3. Memory Store ──────────────────────────────────────────────────────
    memory_ok = _check_memory(verbose)
    all_ok = all_ok and memory_ok

    # ── 4. Dependencies ──────────────────────────────────────────────────────
    deps_ok = _check_deps(verbose)
    all_ok = all_ok and deps_ok

    # ── 5. Python version ────────────────────────────────────────────────────
    python_ok = _check_python(verbose)
    all_ok = all_ok and python_ok

    # ── Summary ──────────────────────────────────────────────────────────────
    console.print()
    if all_ok:
        console.print(Panel(
            "[bold green]All checks passed ✓[/bold green]\n\n"
            "You're ready to run:\n"
            "  [cyan]pa review https://github.com/owner/repo/pull/123[/cyan]",
            border_style="green",
        ))
    else:
        console.print(Panel(
            "[bold red]Some checks failed ✗[/bold red]\n\n"
            "Fix the issues above, then run [cyan]pa status[/cyan] again.\n"
            "Use [cyan]pa config set --help[/cyan] to configure settings.",
            border_style="red",
        ))
        sys.exit(1)


def _check_github(cfg: dict, verbose: bool) -> bool:
    """Check if the GitHub token is valid."""
    token = cfg.get("github_token", "")

    if not token:
        _status("GitHub Token", False, "Not configured. Run: pa config set --token ghp_...")
        return False

    try:
        import requests
        start = time.time()
        resp = requests.get(
            "https://api.github.com/user",
            headers={"Authorization": f"token {token}"},
            timeout=10,
        )
        elapsed = time.time() - start

        if resp.status_code == 200:
            user = resp.json().get("login", "unknown")
            _status("GitHub Token", True, f"Authenticated as [green]{user}[/green] ({elapsed:.1f}s)")
            if verbose:
                scopes = resp.headers.get("X-OAuth-Scopes", "none")
                rate_remaining = resp.headers.get("X-RateLimit-Remaining", "?")
                console.print(f"    [dim]Scopes: {scopes}[/dim]")
                console.print(f"    [dim]Rate limit remaining: {rate_remaining}[/dim]")
            return True
        elif resp.status_code == 401:
            _status("GitHub Token", False, "Invalid or expired token")
            return False
        else:
            _status("GitHub Token", False, f"Unexpected status: {resp.status_code}")
            return False
    except requests.exceptions.Timeout:
        _status("GitHub Token", False, "Connection timeout (>10s)")
        return False
    except requests.exceptions.ConnectionError:
        _status("GitHub Token", False, "Cannot reach api.github.com")
        return False
    except Exception as e:
        _status("GitHub Token", False, str(e))
        return False


def _check_llm(cfg: dict, verbose: bool) -> bool:
    """Check if the LLM endpoint is reachable."""
    server = cfg.get("server", "")

    if not server:
        _status("LLM Endpoint", False, "Not configured. Run: pa config set --server http://...")
        return False

    try:
        import requests
        # vLLM exposes /v1/models — try that first
        models_url = server.rstrip("/")
        if not models_url.endswith("/models"):
            models_url += "/models"

        start = time.time()
        resp = requests.get(
            models_url,
            headers={"Authorization": f"Bearer {cfg.get('api_key', '')}"},
            timeout=10,
        )
        elapsed = time.time() - start

        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            model_names = [m.get("id", "?") for m in models]
            _status("LLM Endpoint", True,
                    f"Reachable ({elapsed:.1f}s) — {len(models)} model(s) available")
            if verbose and model_names:
                for m in model_names[:5]:
                    console.print(f"    [dim]• {m}[/dim]")

            # Check if configured model is available
            configured_model = cfg.get("model", "")
            if configured_model and configured_model not in model_names:
                console.print(f"    [yellow]⚠ Configured model '{configured_model}' not found on server[/yellow]")

            return True
        else:
            _status("LLM Endpoint", False, f"Server returned {resp.status_code}")
            if verbose:
                console.print(f"    [dim]URL: {models_url}[/dim]")
            return False

    except requests.exceptions.Timeout:
        _status("LLM Endpoint", False, f"Connection timeout (>10s) — {server}")
        return False
    except requests.exceptions.ConnectionError:
        _status("LLM Endpoint", False, f"Cannot reach {server}")
        if verbose:
            console.print("    [dim]Is your vLLM server running?[/dim]")
        return False
    except Exception as e:
        _status("LLM Endpoint", False, str(e))
        return False


def _check_memory(verbose: bool) -> bool:
    """Check if the memory store is accessible."""
    try:
        from memory.store import MemoryStore
        memory = MemoryStore()

        # Quick read test
        rows = memory.conn.execute("SELECT COUNT(*) as cnt FROM pr_analyses").fetchone()
        count = rows["cnt"] if rows else 0
        memory.close()

        _status("Memory Store", True, f"SQLite OK — {count} analysis record(s)")
        if verbose:
            from config.settings import MEMORY_DB_PATH
            console.print(f"    [dim]Path: {MEMORY_DB_PATH}[/dim]")
        return True

    except Exception as e:
        _status("Memory Store", False, str(e))
        return False


def _check_deps(verbose: bool) -> bool:
    """Check that required Python packages are importable."""
    required = [
        ("rich", "rich"),
        ("click", "click"),
        ("langchain", "langchain"),
        ("langchain_openai", "langchain_openai"),
        ("langgraph", "langgraph"),
        ("requests", "requests"),
        ("pydantic", "pydantic"),
        ("dotenv", "python-dotenv"),
    ]

    missing = []
    for module_name, pip_name in required:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        _status("Dependencies", False, f"Missing: {', '.join(missing)}")
        console.print(f"    [dim]Run: pip install {' '.join(missing)}[/dim]")
        return False
    else:
        _status("Dependencies", True, f"All {len(required)} required packages found")
        return True


def _check_python(verbose: bool) -> bool:
    """Check Python version."""
    import platform
    version = platform.python_version()
    major, minor = sys.version_info[:2]

    if major >= 3 and minor >= 10:
        _status("Python", True, f"v{version}")
        return True
    else:
        _status("Python", False, f"v{version} — requires Python 3.10+")
        return False


def _status(name: str, ok: bool, detail: str = ""):
    """Print a single status line."""
    if ok:
        console.print(f"  [green]✓[/green] [bold]{name}[/bold]  {detail}")
    else:
        console.print(f"  [red]✗[/red] [bold]{name}[/bold]  [red]{detail}[/red]")
