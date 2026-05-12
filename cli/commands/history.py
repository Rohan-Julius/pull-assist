"""
`pa history` subcommand.

Shows past analyses from the memory store, letting users see
risk trends for repos they've analyzed before.
"""

import sys
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


@click.command("history")
@click.argument("repo", required=False, default=None)
@click.option("--limit", "-n", type=int, default=20,
              help="Maximum number of records to show.")
@click.option("--high-risk", is_flag=True, default=False,
              help="Show only high-risk PRs (score >= 7.0).")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Output as JSON instead of a table.")
def history(repo, limit, high_risk, as_json):
    """Show past PR analysis history from the memory store.

    \b
    EXAMPLES:
      pa history                          Show all analyzed repos
      pa history expressjs/express        Show analyses for a specific repo
      pa history --high-risk              Show only high-risk analyses
      pa history -n 5                     Show last 5 analyses
    """
    from cli.banner import print_mini_banner
    print_mini_banner(console)

    from cli.config_manager import apply_config_to_env
    apply_config_to_env()

    from memory.store import MemoryStore

    try:
        memory = MemoryStore()
    except Exception as e:
        console.print(f"[red]Error opening memory store:[/red] {e}")
        sys.exit(1)

    if repo:
        _show_repo_history(memory, repo, limit, high_risk, as_json)
    else:
        _show_all_repos(memory, as_json)


def _show_all_repos(memory, as_json: bool):
    """Show a summary of all analyzed repos."""
    rows = memory.conn.execute(
        "SELECT * FROM repo_stats ORDER BY last_updated DESC"
    ).fetchall()

    if not rows:
        console.print("[dim]No analysis history found. Run 'pa review' first.[/dim]")
        return

    if as_json:
        import json
        data = []
        for r in rows:
            data.append({
                "repo": r["repo"],
                "total_prs": r["total_prs_analyzed"],
                "avg_risk": r["avg_risk_score"],
                "high_risk_prs": r["high_risk_prs"],
                "last_updated": r["last_updated"],
            })
        console.print(json.dumps(data, indent=2))
        return

    table = Table(
        title="[bold]Analyzed Repositories[/bold]",
        box=box.ROUNDED,
        show_lines=False,
        pad_edge=True,
    )
    table.add_column("Repository", style="cyan", no_wrap=True)
    table.add_column("PRs", justify="right")
    table.add_column("Avg Risk", justify="right")
    table.add_column("High Risk", justify="right")
    table.add_column("Last Updated", style="dim")

    for r in rows:
        avg = r["avg_risk_score"]
        avg_color = "green" if avg < 4 else "yellow" if avg < 7 else "red"
        hr = r["high_risk_prs"]
        hr_style = f"[red]{hr}[/red]" if hr > 0 else f"[dim]{hr}[/dim]"

        table.add_row(
            r["repo"],
            str(r["total_prs_analyzed"]),
            f"[{avg_color}]{avg:.1f}[/{avg_color}]",
            hr_style,
            (r["last_updated"] or "")[:19].replace("T", " "),
        )

    console.print(table)
    console.print(f"\n[dim]Run 'pa history <repo>' to see individual PR analyses.[/dim]")


def _show_repo_history(memory, repo: str, limit: int, high_risk: bool, as_json: bool):
    """Show analysis history for a specific repo."""

    # Show repo stats first
    ctx = memory.get_repo_context(repo)
    if not ctx:
        console.print(f"[dim]No history found for '{repo}'.[/dim]")
        console.print("[dim]Available repos:[/dim]")
        repos = memory.conn.execute("SELECT repo FROM repo_stats").fetchall()
        for r in repos:
            console.print(f"  • {r['repo']}")
        return

    # Fetch individual PR records
    query = "SELECT * FROM pr_analyses WHERE repo = ?"
    params = [repo]
    if high_risk:
        query += " AND overall_risk_score >= 7.0"
    query += " ORDER BY analyzed_at DESC LIMIT ?"
    params.append(limit)

    rows = memory.conn.execute(query, params).fetchall()

    if as_json:
        import json
        data = {
            "repo": repo,
            "stats": {
                "total_prs": ctx.total_prs_analyzed,
                "avg_risk": ctx.avg_risk_score,
                "high_risk_prs": ctx.high_risk_prs,
                "most_touched_files": ctx.most_touched_files,
                "recurring_test_gaps": ctx.recurring_test_gap_files,
            },
            "analyses": [],
        }
        for r in rows:
            data["analyses"].append({
                "pr_number": r["pr_number"],
                "pr_title": r["pr_title"],
                "risk_score": r["overall_risk_score"],
                "risk_level": r["risk_level"],
                "blast_radius": r["blast_radius_count"],
                "test_gaps": bool(r["had_test_gaps"]),
                "critic_verdict": r["critic_verdict"],
                "analyzed_at": r["analyzed_at"],
            })
        console.print(json.dumps(data, indent=2))
        return

    # Repo summary panel
    trend_color = {
        "IMPROVING": "green", "STABLE": "yellow", "WORSENING": "red"
    }
    console.print(Panel(
        f"[bold]PRs Analyzed:[/bold]  {ctx.total_prs_analyzed}\n"
        f"[bold]Avg Risk:[/bold]      {ctx.avg_risk_score:.1f}/10\n"
        f"[bold]High-Risk:[/bold]     {ctx.high_risk_prs}\n"
        f"[bold]Hot Files:[/bold]     {', '.join(ctx.most_touched_files[:4]) or 'none'}\n"
        f"[bold]Test Gaps:[/bold]     {', '.join(ctx.recurring_test_gap_files[:3]) or 'none'}",
        title=f"[bold cyan]{repo}[/bold cyan]",
        border_style="cyan",
    ))

    # PR history table
    if not rows:
        console.print("[dim]No matching analyses found.[/dim]")
        return

    table = Table(
        box=box.SIMPLE,
        show_lines=False,
        pad_edge=True,
    )
    table.add_column("PR", style="cyan", justify="right", width=8)
    table.add_column("Title", no_wrap=False, max_width=40)
    table.add_column("Risk", justify="center", width=12)
    table.add_column("Blast", justify="right", width=6)
    table.add_column("Gaps", justify="center", width=5)
    table.add_column("Critic", width=18)
    table.add_column("Date", style="dim", width=12)

    RISK_EMOJI = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🚨"}
    RISK_COLORS = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red", "CRITICAL": "bold red"}

    for r in rows:
        level = r["risk_level"]
        rc = RISK_COLORS.get(level, "white")
        re = RISK_EMOJI.get(level, "❓")
        score = r["overall_risk_score"]

        table.add_row(
            f"#{r['pr_number']}",
            (r["pr_title"] or "")[:40],
            f"[{rc}]{re} {score:.1f}[/{rc}]",
            str(r["blast_radius_count"]),
            "[red]✗[/red]" if r["had_test_gaps"] else "[green]✓[/green]",
            r["critic_verdict"] or "AGREE",
            (r["analyzed_at"] or "")[:10],
        )

    console.print(table)
    console.print(f"\n[dim]Showing {len(rows)} of {ctx.total_prs_analyzed} analyses.[/dim]")
