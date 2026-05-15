"""
Conflict loop hardening utilities.

Provides two improvements to the orchestrator's conflict resolution:
  1. Adjudication summary — after conflict resolution, produces a plain-English
     explanation of what changed and why. Shown in the final report.
  2. Score delta guard — if the re-run doesn't meaningfully change the score
     (delta < 0.5), exit the loop immediately rather than burning another round.
"""

from rich.console import Console

console = Console()

SCORE_DELTA_THRESHOLD = 0.5   # minimum score change to justify another loop


def objection_confuses_test_coverage_with_runtime(
    obj: dict,
    runtime_risks: dict | None,
    risk_assessment: dict | None,
) -> bool:
    """
    True if a Critic objection asks to raise runtime_risk_score for test-only
    reasons while the pipeline has no verified breaking scenarios — a
    methodology violation for this product (test gaps → test_coverage dimension).
    """
    if not isinstance(obj, dict) or not runtime_risks or not risk_assessment:
        return False
    if runtime_risks.get("is_breaking_change"):
        return False
    if runtime_risks.get("breaking_scenarios"):
        return False
    try:
        rt = float(
            (risk_assessment.get("dimension_scores") or {}).get("runtime_risk_score", 99)
        )
    except (TypeError, ValueError):
        rt = 99.0
    if rt > 2.0:
        return False
    sc = (obj.get("suggested_correction") or "").lower()
    wants_runtime = any(
        p in sc
        for p in (
            "runtime_risk_score",
            "runtime risk score",
            "increase runtime",
            "raise runtime",
            "higher runtime",
        )
    )
    if not wants_runtime:
        return False
    reason = (obj.get("reason") or "").lower()
    claim = (obj.get("claim") or "").lower()
    if any(
        k in reason or k in claim
        for k in ("test gap", "test coverage", "coverage gap", "uncovered", "no test")
    ):
        return True
    return False


def objection_claims_empty_breaking_but_evidence_exists(
    obj: dict,
    runtime_risks: dict | None,
) -> bool:
    """Filter Critic noise that says breaking_scenarios is empty when it is not."""
    if not isinstance(obj, dict) or not runtime_risks:
        return False
    scenarios = runtime_risks.get("breaking_scenarios") or []
    if not scenarios:
        return False
    blob = ((obj.get("reason") or "") + " " + (obj.get("claim") or "")).lower()
    markers = (
        "empty breaking",
        "breaking_scenarios is empty",
        "breaking scenarios is empty",
        "no breaking scenario",
        "empty breaking_scenarios",
    )
    return any(m in blob for m in markers)


def compute_adjudication_summary(
    conflict_log: list,
    final_risk_assessment: dict,
    rerun_count: int,
    runtime_risks: dict | None = None,
    risk_assessment: dict | None = None,
    verdict: str = "AGREE",
) -> str:
    """
    Produces a human-readable summary of the conflict resolution process.
    Injected into the final report under the "Critic Verdict" section.
    """
    if rerun_count == 0:
        v = (verdict or "AGREE").upper()
        if v == "AGREE":
            return "Critic reviewed all agent findings and found them consistent. No re-runs required."
        return (
            f"Critic verdict on first pass: **{v}**. No re-runs were executed (rerun budget not used). "
            "Review any objections below; some may be filtered when they contradict diff-grounded evidence."
        )

    lines = [f"Conflict resolution ran {rerun_count} round(s)."]
    suppressed_total = 0

    for i, entry in enumerate(conflict_log, 1):
        verdict = entry.get("verdict", "AGREE")
        objections = entry.get("objections", [])
        summary = entry.get("critic_summary", "")
        snapshot = entry.get("_score_snapshot", None)
        lines.append(f"\nRound {i}: Critic verdict = {verdict}")
        if snapshot is not None:
            lines.append(f"  Score at this point: {snapshot:.1f}/10")
        if summary:
            lines.append(f"  Summary: {summary}")
        shown = 0
        for obj in objections:
            if not isinstance(obj, dict):
                continue
            if objection_confuses_test_coverage_with_runtime(
                obj, runtime_risks, risk_assessment or final_risk_assessment
            ):
                suppressed_total += 1
                continue
            if shown < 2:
                lines.append(f"  ⚡ [{obj.get('target_agent', '?')}] {obj.get('claim', '')}")
                lines.append(f"     → {obj.get('suggested_correction', '')}")
                shown += 1

    final_score = final_risk_assessment.get("overall_risk_score", 0)
    lines.append(f"\nFinal risk score after resolution: {final_score:.1f}/10")
    if suppressed_total and runtime_risks is not None:
        lines.append(
            "\n_Scoring contract: objections that asked to raise **runtime** risk solely "
            "for **test-coverage** gaps were not applied — runtime risk reflects verified "
            "`breaking_scenarios` only; test gaps are scored under **Test coverage**._"
        )
    return "\n".join(lines)


def should_exit_early(
    previous_risk_score: float,
    current_risk_score: float,
    rerun_count: int,
) -> bool:
    """
    Returns True if the score delta is too small to justify another loop.
    Prevents wasting tokens when agents just repeat themselves.
    """
    delta = abs(current_risk_score - previous_risk_score)
    if delta < SCORE_DELTA_THRESHOLD and rerun_count >= 1:
        console.print(
            f"  [dim]Score delta {delta:.2f} < {SCORE_DELTA_THRESHOLD} threshold. "
            f"Exiting conflict loop early.[/dim]"
        )
        return True
    return False
