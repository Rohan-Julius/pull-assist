"""
`pa review` subcommand.

The primary CLI entry point for running PR impact analysis.
Supports three modes:
  1. pa review <PR_URL>                           — GitHub PR URL
  2. pa review --diff <file> --repo <owner/repo>  — Local diff + GitHub context
  3. pa review --diff <file> --local <path>        — Fully offline
"""

import sys
import click
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.command("review")
@click.argument("pr_url", required=False, default=None)
@click.option("--diff", "diff_file", type=click.Path(exists=True),
              help="Path to a local diff/patch file.")
@click.option("--repo", "repo_name", type=str,
              help="GitHub repo (owner/repo) for tool calls when using --diff.")
@click.option("--local", "local_path", type=click.Path(exists=True),
              help="Path to local git repo for fully offline analysis.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Show full error tracebacks.")
@click.option("--full", is_flag=True, default=False,
              help="Show the full detailed report instead of the compact summary.")
@click.option("--watch", "-w", is_flag=True, default=False,
              help="Tail the agent log in real time as each agent completes.")
@click.option("--day1-only", is_flag=True, default=False,
              help="Run only the data pipeline, skip agents (no GPU needed).")
def review(pr_url, diff_file, repo_name, local_path, verbose, full, watch, day1_only):
    """Analyze a PR for risk, blast radius, test gaps, and deployment strategy.

    \b
    MODES:
      pa review <PR_URL>                              GitHub PR URL
      pa review --diff changes.patch --repo owner/repo   Local diff + GitHub
      pa review --diff changes.patch --local ./          Fully offline
    """
    from cli.banner import print_mini_banner
    print_mini_banner(console)

    # ── Validate input combinations ───────────────────────────────────────────
    if not pr_url and not diff_file:
        console.print("[red]Error:[/red] Provide a PR URL or use --diff <file>")
        console.print("[dim]  pa review https://github.com/owner/repo/pull/123[/dim]")
        console.print("[dim]  pa review --diff changes.patch --repo owner/repo[/dim]")
        sys.exit(1)

    if pr_url and diff_file:
        console.print("[red]Error:[/red] Cannot use both a PR URL and --diff. Choose one.")
        sys.exit(1)

    if diff_file and not repo_name and not local_path:
        console.print("[red]Error:[/red] --diff requires either --repo or --local")
        sys.exit(1)

    if pr_url and (repo_name or local_path):
        console.print("[red]Error:[/red] --repo and --local are only used with --diff")
        sys.exit(1)

    # ── Apply CLI config to environment ───────────────────────────────────────
    from cli.config_manager import apply_config_to_env
    apply_config_to_env()

    # ── Build context ─────────────────────────────────────────────────────────
    try:
        if pr_url:
            from main import build_context
            context = build_context(pr_url, verbose=verbose)
        else:
            from main import build_context_from_diff
            context = build_context_from_diff(
                diff_path=diff_file,
                repo=repo_name,
                local_repo=local_path,
                verbose=verbose,
            )

        if day1_only:
            console.print("\n[bold green]✓ Data pipeline complete[/bold green] — context built successfully.\n")
            return

        # ── Run agents ────────────────────────────────────────────────────────
        console.print("\n[bold magenta]Starting analysis pipeline...[/bold magenta]\n")

        from main import run_agents, save_outputs
        final_state = run_agents(context)

        # ── Output ────────────────────────────────────────────────────────────
        if full:
            # Full detailed report (same as original main.py output)
            save_outputs(final_state)
        else:
            # Compact summary panel first
            from output.report_builder import build_report, build_memory_record
            from cli.ui import print_summary_panel

            report = build_report(final_state)
            print_summary_panel(report)

            # Save reports silently
            from output.formatter import save_markdown, save_json
            md_path = save_markdown(report)
            json_path = save_json(report)

            console.print(f"\n[dim]Reports saved:[/dim]")
            console.print(f"  [dim]Markdown: {md_path}[/dim]")
            console.print(f"  [dim]JSON:     {json_path}[/dim]")

            # Save to memory store
            memory = final_state.get("_memory_store")
            if memory:
                record = build_memory_record(report)
                memory.save_analysis(record)

        console.print("\n[bold green]Analysis complete.[/bold green]\n")

    except ValueError as e:
        console.print(f"\n[bold red]Input error:[/bold red] {e}\n")
        sys.exit(1)
    except RuntimeError as e:
        console.print(f"\n[bold red]API error:[/bold red] {e}\n")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
