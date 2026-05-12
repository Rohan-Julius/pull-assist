"""
Report Builder 

"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from memory.schema import PRAnalysisRecord


@dataclass
class ReportData:
    # Identity
    repo: str
    pr_number: int
    pr_title: str
    pr_url: str
    pr_author: str
    analyzed_at: str

    # Diff overview
    diff_summary: str
    total_additions: int
    total_deletions: int
    languages: list
    changed_files: list
    changed_symbols: list
    has_test_changes: bool

    # Agent outputs
    blast_radius: dict
    runtime_risks: dict
    test_gaps: dict
    risk_assessment: dict
    objections: dict

    # Enhancement outputs
    rollback_advice: dict
    business_impacts: list
    impact_summary: str
    severity_domains: list
    historical_context: dict

    # Graph layer outputs
    evidence_graph: dict
    propagation_chains: list
    deployment_advice: dict

    # Conflict metadata
    rerun_count: int
    conflict_log: list

    # Derived convenience fields
    overall_risk_score: float = 0.0
    risk_level: str = "UNKNOWN"
    top_concerns: list = field(default_factory=list)
    recommended_actions: list = field(default_factory=list)


def build_report(final_state: dict) -> ReportData:
    risk = final_state.get("risk_assessment", {})

    return ReportData(
        repo=final_state.get("repo", ""),
        pr_number=final_state.get("pr_number", 0),
        pr_title=final_state.get("pr_title", ""),
        pr_url=final_state.get("pr_html_url", ""),
        pr_author=final_state.get("pr_author", ""),
        analyzed_at=final_state.get("_analyzed_at", datetime.now(timezone.utc).isoformat()),

        diff_summary=final_state.get("diff_summary", ""),
        total_additions=final_state.get("total_additions", 0),
        total_deletions=final_state.get("total_deletions", 0),
        languages=final_state.get("languages", []),
        changed_files=final_state.get("changed_files", []),
        changed_symbols=final_state.get("changed_symbols", []),
        has_test_changes=final_state.get("has_test_changes", False),

        blast_radius=final_state.get("blast_radius", {}),
        runtime_risks=final_state.get("runtime_risks", {}),
        test_gaps=final_state.get("test_gaps", {}),
        risk_assessment=risk,
        objections=final_state.get("objections", {}),

        # Enhancements
        rollback_advice=final_state.get("rollback_advice", {}),
        business_impacts=final_state.get("business_impacts", []),
        impact_summary=final_state.get("impact_summary", ""),
        severity_domains=final_state.get("severity_domains", []),
        historical_context=final_state.get("historical_context", {}),

        # Graph layer
        evidence_graph=final_state.get("evidence_graph", {}),
        propagation_chains=final_state.get("propagation_chains", []),
        deployment_advice=final_state.get("deployment_advice", {}),

        rerun_count=final_state.get("rerun_count", 0),
        conflict_log=final_state.get("conflict_log", []),

        overall_risk_score=risk.get("overall_risk_score", 0.0),
        risk_level=risk.get("risk_level", "UNKNOWN"),
        top_concerns=risk.get("top_concerns", []),
        recommended_actions=risk.get("recommended_actions", []),
    )


def build_memory_record(report: ReportData) -> PRAnalysisRecord:
    blast = report.blast_radius
    direct = blast.get("direct_dependents", [])
    indirect = blast.get("indirect_dependents", [])
    blast_count = len(direct) + len(indirect)
    uncovered = report.test_gaps.get("uncovered_functions", [])

    return PRAnalysisRecord(
        repo=report.repo,
        pr_number=report.pr_number,
        pr_title=report.pr_title,
        analyzed_at=report.analyzed_at,
        files_touched=report.changed_files,
        symbols_changed=report.changed_symbols,
        languages=report.languages,
        additions=report.total_additions,
        deletions=report.total_deletions,
        overall_risk_score=report.overall_risk_score,
        risk_level=report.risk_level,
        blast_radius_count=blast_count,
        had_test_gaps=len(uncovered) > 0,
        top_concerns=report.top_concerns[:3],
        critic_verdict=report.objections.get("verdict", "AGREE"),
        conflict_rounds=report.rerun_count,
    )
