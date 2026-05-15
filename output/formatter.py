"""
Formatter 

"""

import json
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich import box

from agents.orchestrator_patch import (
    compute_adjudication_summary,
    objection_claims_empty_breaking_but_evidence_exists,
    objection_confuses_test_coverage_with_runtime,
)

console = Console()

RISK_COLORS = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red", "CRITICAL": "bold red", "UNKNOWN": "dim"}
RISK_EMOJI  = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🚨", "UNKNOWN": "❓"}
ROLLBACK_COLORS = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}


def _methodology_markdown_lines() -> list[str]:
    """Framing for merge-readiness — aligns report with scoring contract."""
    return [
        "## How to read this analysis",
        "",
        "This pipeline emits an **evidence-weighted merge signal** (multi-agent + graph checks), not a substitute for domain policy or legal sign-off:",
        "",
        "- **Runtime risk** reflects **`runtime_risks.breaking_scenarios`** (including **`_verified_from_diff`** entries from the patch). Other scenarios should cite **`fetch_file`** when blast radius lists callers.",
        "- **Test gaps** drive the **Test coverage** score and the Test Gaps section; they are **not** promoted into runtime risk without a proven caller break.",
        "- **Blast radius** is from repo-wide symbol search — **disambiguate** near-names (e.g. npm `on-finished` vs PR symbol `onFinish`) using the diff.",
        "",
    ]


def _blast_radius_homonym_footnote(report) -> str | None:
    """Warn when common Express/npm naming collisions may inflate dependents."""
    syms = [str(s).lower() for s in (report.analysis_symbols or [])]
    if "onfinish" not in syms:
        return None
    direct = report.blast_radius.get("direct_dependents") or []
    indirect = report.blast_radius.get("indirect_dependents") or []
    blob = " ".join(str(d.get("reason", "")) for d in list(direct) + list(indirect)).lower()
    if any(x in blob for x in ("on-finished", "onfinished", "onfinish(err", "function onfinish")):
        return (
            "_Search-hit note: mentions of **`on-finished`** / **`onfinished`** / **`onfinish(`** may be the "
            "npm **`on-finished`** package or an internal callback name, not the PR **`onFinish`** symbol — "
            "confirm against the diff before treating as dependents._"
        )
    return None


def print_report(report):
    rc = RISK_COLORS.get(report.risk_level, "white")
    re = RISK_EMOJI.get(report.risk_level, "?")

    # ── Header ────────────────────────────────────────────────────────────────
    console.print(Rule("[bold]PR Impact Analysis Report[/bold]"))
    console.print(Panel(
        f"[bold]PR:[/bold]      [link={report.pr_url}]{report.pr_url}[/link]\n"
        f"[bold]Title:[/bold]   {report.pr_title}\n"
        f"[bold]Author:[/bold]  {report.pr_author}\n"
        f"[bold]Diff:[/bold]    +{report.total_additions} / -{report.total_deletions} | "
        f"{len(report.changed_files)} files | {', '.join(report.languages)}\n"
        f"[bold]Symbols:[/bold] {', '.join(report.analysis_symbols[:8]) or 'none'}",
        title="Pull Request", border_style="blue",
    ))

    # ── Business Impact (prominent placement) ─────────────────────────────────
    if report.business_impacts:
        sev_tag = ""
        if report.severity_domains:
            sev_tag = f"  [bold red]CRITICAL DOMAINS: {', '.join(report.severity_domains)}[/bold red]"
        console.print(Panel(
            f"[bold]{report.impact_summary}[/bold]\n{sev_tag}\n\n"
            + "\n".join(f"  • {b}" for b in report.business_impacts),
            title="[bold yellow]Business Impact[/bold yellow]",
            border_style="yellow",
        ))

    # ── Risk score ─────────────────────────────────────────────────────────────
    dims = report.risk_assessment.get("dimension_scores", {})
    working = report.risk_assessment.get("score_working", "")
    score_corrected = report.risk_assessment.get("_score_corrected", False)
    correction_note = " [dim](server-corrected)[/dim]" if score_corrected else ""

    console.print(Panel(
        f"[bold {rc}]{re}  RISK: {report.overall_risk_score:.1f}/10  —  {report.risk_level}[/bold {rc}]{correction_note}\n\n"
        f"  Blast radius   ({int(0.30*100)}%): {dims.get('blast_radius_score', '?')}/10\n"
        f"  Test coverage  ({int(0.30*100)}%): {dims.get('test_coverage_score', '?')}/10\n"
        f"  Runtime risk   ({int(0.25*100)}%): {dims.get('runtime_risk_score', '?')}/10\n"
        f"  Complexity     ({int(0.15*100)}%): {dims.get('complexity_score', '?')}/10\n"
        + (f"\n  [dim]Working: {working}[/dim]" if working else ""),
        title="Risk Assessment", border_style=rc,
    ))

    # ── Historical context ─────────────────────────────────────────────────────
    hist = report.historical_context
    if hist and hist.get("past_high_risk_prs", 0) > 0:
        trend_color = {"IMPROVING": "green", "STABLE": "yellow", "WORSENING": "red"}.get(
            hist.get("historical_risk_trend", "STABLE"), "white"
        )
        overlap_warn = ""
        if hist.get("files_overlapping_past_high_risk"):
            overlap_warn = f"\n  [red]⚠ Files in this PR appeared in past high-risk PRs:[/red] " + \
                           ", ".join(hist["files_overlapping_past_high_risk"][:3])
        console.print(Panel(
            f"  Past high-risk PRs: [red]{hist['past_high_risk_prs']}[/red]  |  "
            f"Avg risk: {hist['avg_risk_score']:.1f}/10  |  "
            f"Trend: [{trend_color}]{hist['historical_risk_trend']}[/{trend_color}]\n"
            + (f"  Hot modules: {', '.join(hist.get('frequently_affected_modules', [])[:4])}\n" if hist.get('frequently_affected_modules') else "")
            + (f"  Recurring test gaps: {', '.join(hist.get('files_with_recurring_issues', [])[:3])}\n" if hist.get('files_with_recurring_issues') else "")
            + overlap_warn,
            title="Historical Context", border_style="dim",
        ))

    # ── Top concerns ──────────────────────────────────────────────────────────
    if report.top_concerns:
        console.print("\n[bold red]Top Concerns[/bold red]")
        for c in report.top_concerns:
            console.print(f"  ⚠  {c}")

    # ── Blast radius ──────────────────────────────────────────────────────────
    direct   = report.blast_radius.get("direct_dependents", [])
    indirect = report.blast_radius.get("indirect_dependents", [])
    if direct or indirect:
        console.print("\n[bold]Blast Radius[/bold]")
        t = Table(show_lines=False, box=None, pad_edge=False)
        t.add_column("File", style="cyan", no_wrap=False)
        t.add_column("Type", width=10)
        t.add_column("Conf", width=8)
        t.add_column("Reason", style="dim")
        for d in direct[:8]:
            t.add_row(d.get("file",""), "Direct", f"[green]{d.get('confidence','?')}[/green]", d.get("reason",""))
        for d in indirect[:4]:
            t.add_row(d.get("file",""), "Indirect", f"[yellow]{d.get('confidence','?')}[/yellow]", d.get("reason",""))
        console.print(t)
        console.print(f"  [dim]{report.blast_radius.get('blast_radius_summary','')}[/dim]")

    # ── Runtime risks with evidence ───────────────────────────────────────────
    scenarios = report.runtime_risks.get("breaking_scenarios", [])
    if scenarios:
        console.print("\n[bold red]Runtime Breaking Scenarios[/bold red]")
        for s in scenarios[:5]:
            sev_c = "red" if s.get("severity") == "HIGH" else "yellow"
            console.print(f"  [{sev_c}]●[/{sev_c}] {s.get('caller_file','?')} ~line {s.get('line_approx','?')}")
            console.print(f"    Mode: {s.get('failure_mode','')}  →  {s.get('failure_description', s.get('failure_mode',''))}")
            evidence = s.get("evidence", [])
            if evidence:
                console.print(f"    [dim]Evidence:[/dim]")
                for ev in evidence[:3]:
                    console.print(f"      [dim]• {ev}[/dim]")
        sim_conf = report.runtime_risks.get("confidence", 5)
        if sim_conf < 3:
            console.print(f"  [yellow]⚠ Simulator confidence: {sim_conf}/5 — treat scenarios with caution[/yellow]")

    # ── Test gaps ─────────────────────────────────────────────────────────────
    uncovered = report.test_gaps.get("uncovered_functions", [])
    coverage  = report.test_gaps.get("overall_coverage_assessment", "?")
    console.print(f"\n[bold]Test Coverage:[/bold] {coverage}")
    if uncovered:
        for u in uncovered[:5]:
            rc2 = "red" if u.get("risk") == "HIGH" else "yellow"
            console.print(f"  [{rc2}]✗[/{rc2}] [bold]{u.get('function','?')}[/bold] — {u.get('missing_scenario','')}")
    else:
        console.print("  [green]✓ No critical test gaps detected[/green]")

    # ── Rollback advice ───────────────────────────────────────────────────────
    ra = report.rollback_advice
    if ra:
        diff = ra.get("rollback_difficulty", "MEDIUM")
        diff_c = ROLLBACK_COLORS.get(diff, "white")
        data_risk = " [red]⚠ DATA SIDE EFFECTS[/red]" if ra.get("data_side_effects") else ""
        ff_note = " [green]✓ Feature flag rollback possible[/green]" if ra.get("feature_flag_possible") else ""
        console.print(f"\n[bold]Rollback:[/bold] [{diff_c}]{diff}[/{diff_c}]{data_risk}{ff_note}")
        console.print(f"  {ra.get('rollback_summary','')}")
        for risk in ra.get("rollback_risks", [])[:3]:
            console.print(f"  [yellow]•[/yellow] {risk}")
        steps = ra.get("rollback_steps", [])
        if steps:
            console.print("  [dim]Steps:[/dim]")
            for step in steps[:4]:
                console.print(f"    [dim]{step}[/dim]")

    # ── Propagation chains ────────────────────────────────────────────────────
    chains = getattr(report, 'propagation_chains', []) or []
    if chains:
        console.print("\n[bold]Failure Propagation Chains[/bold]")
        for c in chains[:3]:
            risk_c = RISK_COLORS.get(c.get('chain_risk_level', 'LOW'), 'white')
            console.print(f"  [{risk_c}]●[/{risk_c}] [bold]{c.get('symbol','')}[/bold]: {c.get('arrow_diagram','')}")
            if c.get('narrative'):
                console.print(f"    [dim]{c['narrative']}[/dim]")

    # ── Deployment advice ─────────────────────────────────────────────────────
    deploy = getattr(report, 'deployment_advice', {}) or {}
    if deploy and deploy.get('strategy'):
        strategy = deploy['strategy']
        emoji = deploy.get('emoji', '')
        console.print(Panel(
            f"[bold]{emoji} {strategy.replace('_',' ').title()}[/bold]\n"
            f"{deploy.get('description', '')}\n\n"
            + (f"[bold]Reasons:[/bold]\n" + "\n".join(f"  • {r}" for r in deploy.get('reasons', [])[:4]) + "\n\n" if deploy.get('reasons') else "")
            + (f"[bold]Conditions:[/bold]\n" + "\n".join(f"  • {c}" for c in deploy.get('conditions', [])[:4]) + "\n\n" if deploy.get('conditions') else "")
            + (f"[bold]Monitoring:[/bold]\n" + "\n".join(f"  📊 {m}" for m in deploy.get('monitoring_hints', [])[:3]) if deploy.get('monitoring_hints') else ""),
            title="[bold magenta]Deployment Strategy[/bold magenta]",
            border_style="magenta",
        ))

    # ── Conflict log ──────────────────────────────────────────────────────────
    verdict = report.objections.get("verdict", "AGREE")
    if verdict != "AGREE" or report.rerun_count > 0:
        console.print(f"\n[bold]Critic:[/bold] {verdict} (re-runs: {report.rerun_count})")
        shown = 0
        for obj in report.objections.get("objections", []):
            if not isinstance(obj, dict):
                continue
            if objection_confuses_test_coverage_with_runtime(
                obj, report.runtime_risks, report.risk_assessment
            ):
                continue
            if objection_claims_empty_breaking_but_evidence_exists(obj, report.runtime_risks):
                continue
            if shown >= 3:
                break
            console.print(f"  [yellow]⚡[/yellow] [{obj.get('target_agent','?')}] {obj.get('claim','')}")
            shown += 1

    # ── Actions ───────────────────────────────────────────────────────────────
    if report.recommended_actions:
        console.print("\n[bold green]Recommended Actions[/bold green]")
        for i, a in enumerate(report.recommended_actions[:5], 1):
            console.print(f"  {i}. {a}")

    console.print(Rule())


def save_markdown(report, output_dir: str = "reports") -> str:
    Path(output_dir).mkdir(exist_ok=True)
    fp = Path(output_dir) / f"pr-{report.pr_number}-report.md"
    re = RISK_EMOJI.get(report.risk_level, "?")
    dims = report.risk_assessment.get("dimension_scores", {})
    working = report.risk_assessment.get("score_working", "")
    direct   = report.blast_radius.get("direct_dependents", [])
    indirect = report.blast_radius.get("indirect_dependents", [])
    scenarios = report.runtime_risks.get("breaking_scenarios", [])
    uncovered = report.test_gaps.get("uncovered_functions", [])
    ra = report.rollback_advice
    hist = report.historical_context

    lines = [
        f"# PR Impact Analysis: #{report.pr_number}",
        f"",
        f"**{re} RISK: {report.overall_risk_score:.1f}/10 — {report.risk_level}**",
        f"",
        f"| Field | Value |",
        f"|---|---|",
        f"| PR | [{report.pr_title}]({report.pr_url}) |",
        f"| Author | {report.pr_author} |",
        f"| Diff | +{report.total_additions} / -{report.total_deletions} across {len(report.changed_files)} files |",
        f"| Languages | {', '.join(report.languages)} |",
        f"| Symbols | {', '.join(report.analysis_symbols[:8])} |",
        f"| Test changes in PR | {'Yes' if report.has_test_changes else 'No'} |",
        f"| Analyzed | {report.analyzed_at[:19].replace('T',' ')} UTC |",
        f"",
    ]
    lines += _methodology_markdown_lines()

    # Business impact
    if report.business_impacts:
        lines += ["## Business Impact", "", f"_{report.impact_summary}_", ""]
        for b in report.business_impacts:
            lines.append(f"- {b}")
        if report.severity_domains:
            lines.append(f"\n**Critical domains:** {', '.join(report.severity_domains)}")
        lines.append("")

    # Historical context
    if hist and hist.get("past_high_risk_prs", 0) > 0:
        lines += [
            "## Historical Context", "",
            f"| Metric | Value |", "|---|---|",
            f"| Past high-risk PRs | {hist['past_high_risk_prs']} |",
            f"| Avg risk score | {hist['avg_risk_score']:.1f}/10 |",
            f"| Risk trend | {hist['historical_risk_trend']} |",
        ]
        if hist.get("frequently_affected_modules"):
            lines.append(f"| Hot modules | {', '.join(hist['frequently_affected_modules'][:4])} |")
        if hist.get("files_overlapping_past_high_risk"):
            lines.append(f"| ⚠ Files in past high-risk PRs | {', '.join(hist['files_overlapping_past_high_risk'][:3])} |")
        lines.append("")

    # Risk breakdown
    lines += [
        "## Risk Breakdown", "",
        "| Dimension | Weight | Score |", "|---|---|---|",
        f"| Blast Radius | 30% | {dims.get('blast_radius_score','?')}/10 |",
        f"| Test Coverage | 30% | {dims.get('test_coverage_score','?')}/10 |",
        f"| Runtime Risk | 25% | {dims.get('runtime_risk_score','?')}/10 |",
        f"| Complexity | 15% | {dims.get('complexity_score','?')}/10 |",
        f"| **Overall** | | **{report.overall_risk_score:.1f}/10** |",
        "",
    ]
    if working:
        lines += [f"_Score working: {working}_", ""]

    # Top concerns
    if report.top_concerns:
        lines += ["## Top Concerns", ""]
        for c in report.top_concerns:
            lines.append(f"- ⚠ {c}")
        lines.append("")

    # Blast radius
    if direct or indirect:
        lines += ["## Blast Radius", "", "| File | Type | Confidence | Reason |", "|---|---|---|---|"]
        for d in direct[:8]:
            lines.append(f"| `{d.get('file','')}` | Direct | {d.get('confidence','')} | {d.get('reason','')} |")
        for d in indirect[:4]:
            lines.append(f"| `{d.get('file','')}` | Indirect | {d.get('confidence','')} | {d.get('reason','')} |")
        lines += ["", f"_{report.blast_radius.get('blast_radius_summary','')}_", ""]
        hom = _blast_radius_homonym_footnote(report)
        if hom:
            lines += [hom, ""]

    # Runtime risks with evidence
    if scenarios:
        lines += ["## Runtime Breaking Scenarios", ""]
        for s in scenarios[:5]:
            lines += [
                f"**`{s.get('caller_file','?')}`** ~line {s.get('line_approx','?')} — {s.get('failure_mode','')} — severity: {s.get('severity','')}",
                f"> {s.get('failure_description', s.get('failure_mode',''))}",
                "",
            ]
            evidence = s.get("evidence", [])
            if evidence:
                lines.append("Evidence:")
                for ev in evidence[:3]:
                    lines.append(f"- `{ev}`")
            lines.append("")

    # Test gaps
    if uncovered:
        lines += [f"## Test Gaps ({report.test_gaps.get('overall_coverage_assessment','?')})", ""]
        for u in uncovered[:5]:
            lines += [
                f"**`{u.get('function','?')}`** — risk: {u.get('risk','')}",
                f"> {u.get('missing_scenario','')}",
                "",
            ]

    # Rollback advisor
    if ra:
        diff_label = ra.get("rollback_difficulty", "MEDIUM")
        data_flag  = " ⚠ DATA SIDE EFFECTS" if ra.get("data_side_effects") else ""
        ff_flag    = " ✓ Feature flag rollback possible" if ra.get("feature_flag_possible") else ""
        lines += [
            f"## Rollback Assessment: {diff_label}{data_flag}{ff_flag}", "",
            f"_{ra.get('rollback_summary','')}_", "",
        ]
        if ra.get("rollback_risks"):
            lines += ["**Rollback risks:**"]
            for r in ra["rollback_risks"][:3]:
                lines.append(f"- {r}")
            lines.append("")
        if ra.get("rollback_steps"):
            lines += ["**Rollback steps:**"]
            for step in ra["rollback_steps"][:5]:
                lines.append(f"- {step}")
            lines.append("")

    # Propagation chains
    chains = getattr(report, 'propagation_chains', []) or []
    if chains:
        lines += ["## Failure Propagation Chains", ""]
        for c in chains[:3]:
            lines += [
                f"**`{c.get('symbol','')}`** — {c.get('chain_risk_level','LOW')} risk",
                f"> {c.get('arrow_diagram','')}",
                "",
            ]
            if c.get('narrative'):
                lines.append(f"_{c['narrative']}_")
                lines.append("")
            if c.get('file_chain'):
                lines.append(f"Files: `{c['file_chain']}`")
                lines.append("")

    # Deployment strategy
    deploy = getattr(report, 'deployment_advice', {}) or {}
    if deploy and deploy.get('strategy'):
        emoji = deploy.get('emoji', '')
        strategy = deploy['strategy']
        lines += [
            f"## Deployment Strategy: {emoji} {strategy.replace('_',' ').title()}", "",
            f"_{deploy.get('description', '')}_", "",
        ]
        if deploy.get('reasons'):
            lines += ["**Reasons:**"]
            for r in deploy['reasons'][:4]:
                lines.append(f"- {r}")
            lines.append("")
        if deploy.get('conditions'):
            lines += ["**Pre-deploy conditions:**"]
            for c in deploy['conditions'][:4]:
                lines.append(f"- {c}")
            lines.append("")
        if deploy.get('monitoring_hints'):
            lines += ["**Monitoring:**"]
            for m in deploy['monitoring_hints'][:3]:
                lines.append(f"- 📊 {m}")
            lines.append("")
        if deploy.get('estimated_blast_radius'):
            lines.append(f"_Blast radius: {deploy['estimated_blast_radius']}_")
            lines.append("")

    # Critic
    verdict = report.objections.get("verdict", "AGREE")
    adj = compute_adjudication_summary(
        report.conflict_log,
        report.risk_assessment,
        report.rerun_count,
        runtime_risks=report.runtime_risks,
        risk_assessment=report.risk_assessment,
        verdict=verdict,
    )
    lines += [f"## Critic Verdict: {verdict} (re-runs: {report.rerun_count})", "", adj, ""]
    if verdict != "AGREE":
        shown = 0
        for obj in report.objections.get("objections", []):
            if not isinstance(obj, dict):
                continue
            if objection_confuses_test_coverage_with_runtime(
                obj, report.runtime_risks, report.risk_assessment
            ):
                continue
            if objection_claims_empty_breaking_but_evidence_exists(obj, report.runtime_risks):
                continue
            if shown >= 3:
                break
            lines += [
                f"**[{obj.get('target_agent','?')}]** {obj.get('claim','')}",
                f"> {obj.get('reason','')}",
                f"> Suggested: {obj.get('suggested_correction','')}",
                "",
            ]
            shown += 1

    # Actions
    if report.recommended_actions:
        lines += ["## Recommended Actions", ""]
        for i, a in enumerate(report.recommended_actions[:5], 1):
            lines.append(f"{i}. {a}")
        lines.append("")

    lines += ["---", "_PR Impact Analyzer — evidence-weighted multi-agent merge signal._"]
    fp.write_text("\n".join(lines))
    return str(fp)


def save_json(report, output_dir: str = "reports") -> str:
    Path(output_dir).mkdir(exist_ok=True)
    fp = Path(output_dir) / f"pr-{report.pr_number}-report.json"

    payload = {
        "meta": {
            "repo": report.repo, "pr_number": report.pr_number,
            "pr_title": report.pr_title, "pr_url": report.pr_url,
            "pr_author": report.pr_author, "analyzed_at": report.analyzed_at,
        },
        "risk": {
            "overall_score": report.overall_risk_score,
            "level": report.risk_level,
            "dimensions": report.risk_assessment.get("dimension_scores", {}),
            "score_working": report.risk_assessment.get("score_working", ""),
            "score_corrected": report.risk_assessment.get("_score_corrected", False),
            "top_concerns": report.top_concerns,
            "recommended_actions": report.recommended_actions,
        },
        "business_impact": {
            "impacts": report.business_impacts,
            "summary": report.impact_summary,
            "severity_domains": report.severity_domains,
        },
        "historical_context": report.historical_context,
        "blast_radius": report.blast_radius,
        "runtime_risks": report.runtime_risks,
        "test_gaps": report.test_gaps,
        "rollback": report.rollback_advice,
        "critic": {
            "verdict": report.objections.get("verdict", "AGREE"),
            "objections": report.objections.get("objections", []),
            "missed_impacts": report.objections.get("missed_impacts", []),
            "rerun_count": report.rerun_count,
        },
        "evidence_graph": getattr(report, 'evidence_graph', {}),
        "propagation_chains": getattr(report, 'propagation_chains', []),
        "deployment_advice": getattr(report, 'deployment_advice', {}),
    }

    fp.write_text(json.dumps(payload, indent=2, default=str))
    return str(fp)
