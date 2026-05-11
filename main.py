#!/usr/bin/env python3
"""
PR Impact Analyzer — main entry point

Usage (3 input modes):

  Mode 1: GitHub PR URL (original)
    python main.py --pr https://github.com/owner/repo/pull/123
    python main.py --pr https://github.com/owner/repo/pull/123 --verbose

  Mode 2: Local diff file + remote GitHub repo
    python main.py --diff changes.patch --repo owner/repo
    python main.py --diff changes.patch --repo owner/repo --verbose

  Mode 3: Local diff file + local git repo (fully offline, no GitHub API needed)
    python main.py --diff changes.patch --local-repo /path/to/repo
    python main.py --diff changes.patch --local-repo . --verbose

  Other flags:
    --day1-only   Run only the data pipeline, skip agents (no GPU needed)
    --verbose     Show full error tracebacks
"""

import argparse
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from github.client import GitHubClient, parse_pr_url
from github.diff_parser import parse_diff
from memory.store import MemoryStore

console = Console()


# ── Mode 1: GitHub PR URL ─────────────────────────────────────────────────────

def build_context(pr_url: str, verbose: bool = False) -> dict:
    """Build analysis context from a GitHub PR URL."""
    console.print("\n[bold cyan]Step 1/5[/bold cyan] Parsing PR URL...")
    repo, pr_number = parse_pr_url(pr_url)
    console.print(f"  Repo: [green]{repo}[/green]  PR: [green]#{pr_number}[/green]")

    console.print("\n[bold cyan]Step 2/5[/bold cyan] Fetching PR metadata...")
    client = GitHubClient(repo=repo)
    metadata = client.fetch_pr_metadata(pr_number)
    console.print(f"  Title: [green]{metadata['title']}[/green]  Author: [green]{metadata['author']}[/green]")
    console.print(f"  Changes: [green]+{metadata['additions']} / -{metadata['deletions']} across {metadata['changed_files']} files[/green]")

    console.print("\n[bold cyan]Step 3/5[/bold cyan] Fetching raw diff...")
    raw_diff = client.fetch_pr_diff(pr_number)
    console.print(f"  Diff size: [green]{len(raw_diff):,} chars[/green]")

    console.print("\n[bold cyan]Step 4/5[/bold cyan] Parsing diff...")
    parsed = parse_diff(raw_diff)
    console.print(f"  Files: [green]{len(parsed.changed_files)}[/green]  "
                  f"Languages: [green]{', '.join(parsed.languages) or 'unknown'}[/green]  "
                  f"Symbols: [green]{', '.join(parsed.all_changed_symbols[:6]) or 'none'}[/green]  "
                  f"Tests in PR: [green]{'Yes' if parsed.has_test_changes else 'No'}[/green]")

    console.print("\n[bold cyan]Step 5/5[/bold cyan] Querying memory store...")
    memory = MemoryStore()
    repo_ctx = memory.get_repo_context(repo)
    if repo_ctx:
        console.print(f"  Found [green]{repo_ctx.total_prs_analyzed}[/green] prior analyses")
    else:
        console.print("  [dim]No prior history for this repo[/dim]")

    history_prompt = memory.format_context_for_prompt(repo)

    context = {
        "repo": repo, "pr_number": pr_number,
        "pr_url": pr_url, "pr_title": metadata["title"],
        "pr_author": metadata["author"], "base_branch": metadata["base_branch"],
        "pr_html_url": metadata["html_url"],
        "diff_summary": parsed.summary,
        "total_additions": parsed.total_additions,
        "total_deletions": parsed.total_deletions,
        "languages": parsed.languages,
        "changed_files": [f.path for f in parsed.changed_files],
        "source_files": parsed.source_files_changed,
        "test_files": parsed.test_files_changed,
        "changed_symbols": parsed.all_changed_symbols,
        "has_test_changes": parsed.has_test_changes,
        "per_file_context": parsed.to_agent_context()["per_file"],
        "raw_diff": raw_diff,
        "repo_history": history_prompt,
        "_github_client": client,
        "_memory_store": memory,
        "_parsed_diff": parsed,
        "_analyzed_at": datetime.now(timezone.utc).isoformat(),
    }

    console.print(Panel(
        f"[bold]PR:[/bold]      [link={context['pr_html_url']}]{context['pr_html_url']}[/link]\n"
        f"[bold]Title:[/bold]   {context['pr_title']}\n"
        f"[bold]Diff:[/bold]    {context['diff_summary']}\n"
        f"[bold]Tests:[/bold]   {'✓ PR includes test changes' if context['has_test_changes'] else '✗ No test files in diff'}",
        title="[bold green]✓ Context Built (GitHub PR)[/bold green]",
        border_style="green",
    ))
    return context


# ── Mode 2 & 3: Local diff file ──────────────────────────────────────────────

def build_context_from_diff(
    diff_path: str,
    repo: str = None,
    local_repo: str = None,
    verbose: bool = False,
) -> dict:
    """
    Build analysis context from a local diff file.

    Two sub-modes:
      - repo="owner/repo"         → tool calls go to GitHub API
      - local_repo="/path/to/dir" → tool calls use local git commands (fully offline)
    """
    diff_file = Path(diff_path)
    if not diff_file.is_file():
        raise ValueError(f"Diff file not found: {diff_path}")

    # ── Step 1: Read diff ─────────────────────────────────────────────────────
    console.print("\n[bold cyan]Step 1/4[/bold cyan] Reading diff file...")
    raw_diff = diff_file.read_text(errors="replace")
    console.print(f"  File: [green]{diff_file.name}[/green]  Size: [green]{len(raw_diff):,} chars[/green]")

    # ── Step 2: Parse diff ────────────────────────────────────────────────────
    console.print("\n[bold cyan]Step 2/4[/bold cyan] Parsing diff...")
    parsed = parse_diff(raw_diff)
    console.print(f"  Files: [green]{len(parsed.changed_files)}[/green]  "
                  f"Languages: [green]{', '.join(parsed.languages) or 'unknown'}[/green]  "
                  f"Symbols: [green]{', '.join(parsed.all_changed_symbols[:6]) or 'none'}[/green]  "
                  f"Tests in diff: [green]{'Yes' if parsed.has_test_changes else 'No'}[/green]")

    # ── Step 3: Build client ──────────────────────────────────────────────────
    console.print("\n[bold cyan]Step 3/4[/bold cyan] Setting up repo client...")

    if local_repo:
        # Mode 3: fully local
        from github.local_client import LocalGitClient
        resolved_path = str(Path(local_repo).resolve())
        client = LocalGitClient(repo_path=resolved_path)
        repo_name = Path(resolved_path).name  # use folder name as repo identifier
        mode_label = f"Local repo: {resolved_path}"
        console.print(f"  Mode: [green]Local git repo[/green]")
        console.print(f"  Path: [green]{resolved_path}[/green]")
    elif repo:
        # Mode 2: diff is local, tools call GitHub
        client = GitHubClient(repo=repo)
        repo_name = repo
        mode_label = f"GitHub repo: {repo}"
        console.print(f"  Mode: [green]Local diff + GitHub repo[/green]")
        console.print(f"  Repo: [green]{repo}[/green]")
    else:
        raise ValueError("Must provide either --repo or --local-repo with --diff")

    # ── Step 4: Memory ────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Step 4/4[/bold cyan] Querying memory store...")
    memory = MemoryStore()
    repo_ctx = memory.get_repo_context(repo_name)
    if repo_ctx:
        console.print(f"  Found [green]{repo_ctx.total_prs_analyzed}[/green] prior analyses")
    else:
        console.print("  [dim]No prior history for this repo[/dim]")

    history_prompt = memory.format_context_for_prompt(repo_name)

    # Synthesize metadata that would normally come from the GitHub PR API
    pr_title = f"Local diff: {diff_file.name}"
    context = {
        "repo": repo_name,
        "pr_number": 0,  # no PR number for local diffs
        "pr_url": f"file://{diff_file.resolve()}",
        "pr_title": pr_title,
        "pr_author": os.environ.get("USER", "local"),
        "base_branch": "main",
        "pr_html_url": f"file://{diff_file.resolve()}",
        "diff_summary": parsed.summary,
        "total_additions": parsed.total_additions,
        "total_deletions": parsed.total_deletions,
        "languages": parsed.languages,
        "changed_files": [f.path for f in parsed.changed_files],
        "source_files": parsed.source_files_changed,
        "test_files": parsed.test_files_changed,
        "changed_symbols": parsed.all_changed_symbols,
        "has_test_changes": parsed.has_test_changes,
        "per_file_context": parsed.to_agent_context()["per_file"],
        "raw_diff": raw_diff,
        "repo_history": history_prompt,
        "_github_client": client,
        "_memory_store": memory,
        "_parsed_diff": parsed,
        "_analyzed_at": datetime.now(timezone.utc).isoformat(),
    }

    console.print(Panel(
        f"[bold]Source:[/bold]  {mode_label}\n"
        f"[bold]Diff:[/bold]    {diff_file.name}\n"
        f"[bold]Summary:[/bold] {context['diff_summary']}\n"
        f"[bold]Tests:[/bold]   {'✓ Diff includes test changes' if context['has_test_changes'] else '✗ No test files in diff'}",
        title="[bold green]✓ Context Built (Local Diff)[/bold green]",
        border_style="green",
    ))
    return context


# ── Agent + Output pipeline ───────────────────────────────────────────────────

def run_agents(context: dict) -> dict:
    from agents.orchestrator import run_analysis
    return run_analysis(context)


def save_outputs(final_state: dict):
    from output.report_builder import build_report, build_memory_record
    from output.formatter import print_report, save_markdown, save_json

    report = build_report(final_state)
    print_report(report)

    md_path = save_markdown(report)
    json_path = save_json(report)
    console.print(f"\n[bold green]Reports saved:[/bold green]")
    console.print(f"  Markdown: [cyan]{md_path}[/cyan]")
    console.print(f"  JSON:     [cyan]{json_path}[/cyan]")

    memory = final_state.get("_memory_store")
    if memory:
        record = build_memory_record(report)
        memory.save_analysis(record)
        console.print(f"  Memory:   [cyan]Saved to memory store ✓[/cyan]")

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PR Impact Analyzer — Analyze the impact of code changes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a GitHub PR
  python main.py --pr https://github.com/expressjs/express/pull/7171

  # Analyze a local diff against a remote GitHub repo
  python main.py --diff changes.patch --repo expressjs/express

  # Analyze a local diff against a local git repo (fully offline)
  python main.py --diff changes.patch --local-repo /path/to/express

  # Generate just a diff file from git
  git diff main..feature-branch > changes.patch
  python main.py --diff changes.patch --local-repo .
""",
    )

    # Input mode arguments (mutually exclusive groups)
    mode_group = parser.add_argument_group("Input modes (choose one)")
    mode_group.add_argument(
        "--pr", metavar="URL",
        help="GitHub PR URL (e.g. https://github.com/owner/repo/pull/123)",
    )
    mode_group.add_argument(
        "--diff", metavar="FILE",
        help="Path to a local diff/patch file",
    )

    # Repo source (used with --diff)
    repo_group = parser.add_argument_group("Repo source (used with --diff)")
    repo_group.add_argument(
        "--repo", metavar="OWNER/REPO",
        help="GitHub repo for tool calls (e.g. expressjs/express)",
    )
    repo_group.add_argument(
        "--local-repo", metavar="PATH",
        help="Path to local git repo for fully offline analysis",
    )

    # Flags
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full error tracebacks")
    parser.add_argument("--day1-only", action="store_true",
                        help="Run only the data pipeline, skip agents (no GPU needed)")

    args = parser.parse_args()

    # ── Validate input combinations ───────────────────────────────────────────
    if not args.pr and not args.diff:
        parser.error("Must provide either --pr URL or --diff FILE")

    if args.pr and args.diff:
        parser.error("Cannot use both --pr and --diff. Choose one input mode.")

    if args.diff and not args.repo and not args.local_repo:
        parser.error("--diff requires either --repo OWNER/REPO or --local-repo PATH")

    if args.pr and (args.repo or args.local_repo):
        parser.error("--repo and --local-repo are only used with --diff, not --pr")

    # ── Run ───────────────────────────────────────────────────────────────────
    try:
        if args.pr:
            context = build_context(args.pr, verbose=args.verbose)
        else:
            context = build_context_from_diff(
                diff_path=args.diff,
                repo=args.repo,
                local_repo=args.local_repo,
                verbose=args.verbose,
            )

        if args.day1_only:
            console.print("\n[bold green]Day 1 checkpoint PASSED[/bold green] — context ready.\n")
            return

        console.print("\n[bold magenta]Starting agent pipeline...[/bold magenta]")
        final_state = run_agents(context)
        save_outputs(final_state)
        console.print("\n[bold green]Analysis complete.[/bold green]\n")

    except ValueError as e:
        console.print(f"\n[bold red]Input error:[/bold red] {e}\n")
        sys.exit(1)
    except RuntimeError as e:
        console.print(f"\n[bold red]API error:[/bold red] {e}\n")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        if args.verbose:
            import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()