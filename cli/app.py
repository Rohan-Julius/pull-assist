"""
pull-assist CLI application.

This is the main entry point that Click resolves when you run `pa` or `pullassist`.
It wires up all subcommands and displays the startup banner.

Usage:
  pa review <PR_URL>
  pa review --diff <file> --repo <owner/repo>
  pa review --diff <file> --local <path>
  pa history [<repo>]
  pa config show | set | reset
  pa status
"""

import click
from cli import __version__



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
        from rich.text import Text
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
    This is what pyproject.toml's [project.scripts] points to.
    """
    cli()


if __name__ == "__main__":
    main()
