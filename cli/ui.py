"""
Terminal UI utilities for pull-assist CLI.

Provides:
  - Progressive spinner during agent execution
  - Color-coded risk level formatting
  - Compact summary panel for quick glance
  - Interactive prompt to show full report
"""

import time
from contextlib import contextmanager
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.rule import Rule
from rich import box

console = Console()

# ── Risk color mapping ─────────────────────────────────────────────────────────

RISK_COLORS = {
    "LOW":      "green",
    "MEDIUM":   "yellow",
    "HIGH":     "red",
    "CRITICAL": "bold red",
    "UNKNOWN":  "dim",
}

RISK_EMOJI = {
    "LOW":      "🟢",
    "MEDIUM":   "🟡",
    "HIGH":     "🔴",
    "CRITICAL": "🚨",
    "UNKNOWN":  "❓",
}

STRATEGY_EMOJI = {
    "DIRECT_MERGE":       "🟢",
    "FEATURE_FLAG":       "🚩",
    "CANARY_ROLLOUT":     "🐤",
    "BLUE_GREEN":         "🔵",
    "STAGED_ROLLOUT":     "📊",
    "MAINTENANCE_WINDOW": "🔧",
}


def risk_color(level: str) -> str:
    """Get rich color tag for a risk level."""
    return RISK_COLORS.get(level, "white")


def risk_emoji(level: str) -> str:
    """Get emoji for a risk level."""
    return RISK_EMOJI.get(level, "❓")


# ── Agent pipeline spinner ─────────────────────────────────────────────────────

AGENT_NAMES = [
    ("Dependency Mapper",          "Mapping blast radius..."),
    ("Change Simulator",           "Simulating runtime behaviour..."),
    ("Test Gap Analyzer",          "Checking test coverage..."),
    ("Rollback Advisor",           "Assessing rollback strategy..."),
    ("Business Impact + Risk",     "Scoring risk dimensions..."),
    ("Critic Agent",               "Cross-checking agent findings..."),
    ("Graph Layer",                "Building evidence graph & deployment advice..."),
]


@contextmanager
def agent_progress():
    """
    Context manager that provides a live progress display for the agent pipeline.
    Usage:
        with agent_progress() as update:
            update(0, "running")   # Dependency Mapper running
            ...
            update(0, "done")      # Dependency Mapper done
            update(1, "running")   # Change Simulator running
    """
    progress = Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold]{task.description}[/bold]"),
        TextColumn("{task.fields[status]}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )

    tasks = []
    with progress:
        for name, desc in AGENT_NAMES:
            task_id = progress.add_task(
                f"  {name}",
                total=1,
                status="[dim]pending[/dim]",
            )
            tasks.append(task_id)

        def update(index: int, status: str):
            if index >= len(tasks):
                return
            if status == "running":
                progress.update(tasks[index], status=f"[cyan]⏳ {AGENT_NAMES[index][1]}[/cyan]")
            elif status == "done":
                progress.update(tasks[index], status="[green]✓ done[/green]", completed=1)
            elif status == "failed":
                progress.update(tasks[index], status="[red]✗ failed[/red]", completed=1)
            elif status == "skipped":
                progress.update(tasks[index], status="[dim]— skipped[/dim]", completed=1)
            else:
                progress.update(tasks[index], status=f"[yellow]{status}[/yellow]")

        yield update


# ── Summary panel (compact output after analysis) ──────────────────────────────

def print_summary_panel(report):
    """
    Print a compact summary panel showing the most important results at a glance.
    This is what users see first — the full report is optional.
    """
    risk = report.risk_assessment
    level = report.risk_level
    score = report.overall_risk_score
    rc = risk_color(level)
    re = risk_emoji(level)

    # Deployment strategy
    deploy = getattr(report, 'deployment_advice', {}) or {}
    strategy = deploy.get('strategy', 'N/A')
    strategy_em = deploy.get('emoji', STRATEGY_EMOJI.get(strategy, ''))

    # Build the summary content
    lines = [
        f"[bold {rc}]{re}  RISK: {score:.1f}/10 — {level}[/bold {rc}]",
        "",
    ]

    # Risk dimensions as a compact table
    dims = risk.get("dimension_scores", {})
    if dims:
        lines.append("[bold]Risk Dimensions:[/bold]")
        lines.append(f"  Blast Radius   {_bar(dims.get('blast_radius_score', 0))}  {dims.get('blast_radius_score', '?')}/10")
        lines.append(f"  Test Coverage  {_bar(dims.get('test_coverage_score', 0))}  {dims.get('test_coverage_score', '?')}/10")
        lines.append(f"  Runtime Risk   {_bar(dims.get('runtime_risk_score', 0))}  {dims.get('runtime_risk_score', '?')}/10")
        lines.append(f"  Complexity     {_bar(dims.get('complexity_score', 0))}  {dims.get('complexity_score', '?')}/10")
        lines.append("")

    # Deployment strategy
    if strategy != 'N/A':
        lines.append(f"[bold]Deploy:[/bold] {strategy_em} {strategy.replace('_', ' ').title()}")
        if deploy.get('description'):
            lines.append(f"  [dim]{deploy['description'][:80]}[/dim]")
        lines.append("")

    # Top concerns (max 3)
    concerns = report.top_concerns[:3] if report.top_concerns else []
    if concerns:
        lines.append("[bold]Top Concerns:[/bold]")
        for c in concerns:
            lines.append(f"  [yellow]⚠[/yellow]  {c}")
        lines.append("")

    # Rollback difficulty
    ra = report.rollback_advice
    if ra:
        diff = ra.get("rollback_difficulty", "?")
        diff_c = RISK_COLORS.get(diff, "white")
        lines.append(f"[bold]Rollback:[/bold] [{diff_c}]{diff}[/{diff_c}]")

    console.print(Panel(
        "\n".join(lines),
        title=f"[bold]PR #{report.pr_number} — {report.pr_title[:60]}[/bold]",
        subtitle="[dim]pa review --full for detailed report[/dim]",
        border_style=rc,
        padding=(1, 2),
    ))


def _bar(score, max_score=10, width=15) -> str:
    """Generate a mini progress bar for a score."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "[dim]" + "░" * width + "[/dim]"

    filled = int((s / max_score) * width)
    empty = width - filled

    if s >= 7:
        color = "red"
    elif s >= 4:
        color = "yellow"
    else:
        color = "green"

    return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"


# ── Status checks ─────────────────────────────────────────────────────────────

def print_status_check(name: str, ok: bool, detail: str = ""):
    """Print a single status check line."""
    if ok:
        console.print(f"  [green]✓[/green] {name}  {detail}")
    else:
        console.print(f"  [red]✗[/red] {name}  [red]{detail}[/red]")


# ── Watch mode ─────────────────────────────────────────────────────────────────

def print_watch_log(agent_name: str, message: str, level: str = "info"):
    """Print a watch-mode log line with timestamp."""
    ts = time.strftime("%H:%M:%S")
    if level == "error":
        console.print(f"  [dim]{ts}[/dim] [red]✗[/red] [bold]{agent_name}:[/bold] {message}")
    elif level == "warn":
        console.print(f"  [dim]{ts}[/dim] [yellow]⚠[/yellow] [bold]{agent_name}:[/bold] {message}")
    else:
        console.print(f"  [dim]{ts}[/dim] [green]▸[/green] [bold]{agent_name}:[/bold] {message}")
