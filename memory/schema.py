from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PRAnalysisRecord:
    """
    Everything we persist to the memory store after analyzing a PR.
    This becomes the historical context injected into future PR analyses
    for the same repo.
    """
    # Identity
    repo: str
    pr_number: int
    pr_title: str
    analyzed_at: str            # ISO8601 timestamp

    # What changed
    files_touched: list[str]    # list of file paths
    symbols_changed: list[str]  # list of function/class names
    languages: list[str]
    additions: int
    deletions: int

    # What we found
    overall_risk_score: float   # 0.0–10.0
    risk_level: str             # LOW / MEDIUM / HIGH / CRITICAL
    blast_radius_count: int     # number of files in blast radius
    had_test_gaps: bool
    top_concerns: list[str]     # top 3 concern strings

    # Agent disagreement tracking
    critic_verdict: str         # AGREE / MINOR_ISSUES / SIGNIFICANT_ISSUES
    conflict_rounds: int        # how many re-run rounds were needed

    # Optional: used when a PR is later confirmed to have caused an incident
    confirmed_incident: Optional[bool] = None
    incident_notes: Optional[str] = None


@dataclass
class RepoContext:
    """
    Aggregated view of a repo's PR history — injected into agent prompts
    as background context so agents know: "this repo tends to have X pattern."
    """
    repo: str
    total_prs_analyzed: int
    avg_risk_score: float
    high_risk_prs: int
    most_touched_files: list[str]      # top 5 most frequently changed files
    recurring_test_gap_files: list[str] # files that consistently lack tests
    recent_summaries: list[str]        # last N PR one-liners for context