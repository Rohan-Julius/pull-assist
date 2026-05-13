"""
pull-assist CLI application.

This is the main entry point that Click resolves when you run `pa` or `pullassist`.
It wires up all subcommands and displays the startup banner.

When run from VS Code, it automatically opens a new integrated terminal
so `pa` feels like a standalone app (no (venv)(base) prompt clutter).

Usage:
  pa review <PR_URL>
  pa review --diff <file> --repo <owner/repo>
  pa review --diff <file> --local <path>
  pa history [<repo>]
  pa config show | set | reset
  pa status
"""

import os
import sys
import subprocess
import click
from cli import __version__


def _launch_in_new_vscode_terminal(args: list[str]):
    """
    Open a new VS Code integrated terminal and re-run `pa` inside it.
    Uses osascript to send Cmd+Shift+` (new terminal shortcut) to VS Code,
    then types the command.

    Returns True if launch succeeded, False if not in VS Code.
    """
    term_program = os.environ.get("TERM_PROGRAM", "")
    if term_program != "vscode":
        return False

    # Find the pa executable path
    pa_bin = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "venv", "bin", "pa")
    if not os.path.exists(pa_bin):
        # Fallback: use whatever `pa` is in PATH
        pa_bin = "pa"

    # Build the command string
    if args:
        cmd_str = f"{pa_bin} {' '.join(args)}"
    else:
        cmd_str = pa_bin

    # Escape for osascript
    escaped_cmd = cmd_str.replace('"', '\\"')

    try:
        # Open new VS Code terminal
        subprocess.run([
            "osascript", "-e",
            'tell application "System Events" to tell process "Code" '
            'to keystroke "`" using {command down, shift down}'
        ], capture_output=True, timeout=3)

        import time
        time.sleep(0.6)

        # Type the command into the new terminal
        # Set a clean prompt and run the pa command
        subprocess.run([
            "osascript", "-e",
            f'tell application "System Events" to tell process "Code" '
            f'to keystroke "clear && export PS1=\'pull-assist> \' && PA_LAUNCHED=1 {escaped_cmd}" & return'
        ], capture_output=True, timeout=3)

        return True
    except Exception:
        return False


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="pull-assist")
@click.pass_context
def cli(ctx):
    """pull-assist — AI-powered PR impact analysis."""
    # Only show banner when no subcommand is given (bare `pa`)
    if ctx.invoked_subcommand is None:
        import shutil
        from cli.banner import print_banner
        from rich.console import Console
        console = Console()
        print_banner(console, compact=False)

        tw = shutil.get_terminal_size((80, 24)).columns

        help_lines = [
            "",
            "Commands:",
            "  review       Analyze a PR for risk and impact",
            "  history      View past analysis results",
            "  config       Manage configuration (tokens, endpoints)",
            "  status       Check connectivity and setup",
            "",
            "Quick Start:",
            "  pa config set --token ghp_... --server http://your-server:8000/v1",
            "  pa status",
            "  pa review https://github.com/owner/repo/pull/123",
            "",
            f"Version: {__version__}  |  github.com/Rohan-Julius/pull-assist",
        ]
        # Center as a block: pad all lines based on the LONGEST line
        max_len = max(len(l) for l in help_lines)
        block_pad = max(0, (tw - max_len) // 2)
        for line in help_lines:
            console.print(" " * block_pad + line)


# ── Register subcommands ──────────────────────────────────────────────────────

def _register_commands():
    """Import and register all subcommands."""
    from cli.commands.review import review
    from cli.commands.history import history
    from cli.commands.config_cmd import config
    from cli.commands.status import status

    cli.add_command(review)
    cli.add_command(history)
    cli.add_command(config)
    cli.add_command(status)


_register_commands()


def main():
    """
    Entry point called by the `pa` / `pullassist` console scripts.

    If running inside VS Code and PA_LAUNCHED is not set, opens a
    new integrated terminal and re-runs the command there.
    Otherwise, runs the CLI normally.
    """
    # Check if we should launch in a new terminal
    if not os.environ.get("PA_LAUNCHED"):
        # Get the original arguments (skip the script name)
        args = sys.argv[1:]
        if _launch_in_new_vscode_terminal(args):
            # Successfully launched in new terminal — exit this one
            sys.exit(0)

    # Normal execution (either PA_LAUNCHED=1 or not in VS Code)
    cli()


if __name__ == "__main__":
    main()
