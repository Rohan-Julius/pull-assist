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


def compute_adjudication_summary(
    conflict_log: list,
    final_risk_assessment: dict,
    rerun_count: int,
) -> str:
    """
    Produces a human-readable summary of the conflict resolution process.
    Injected into the final report under the "Critic Verdict" section.
    """
    if rerun_count == 0:
        return "Critic reviewed all agent findings and found them consistent. No re-runs required."

    lines = [f"Conflict resolution ran {rerun_count} round(s)."]

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
        for obj in objections[:2]:
            lines.append(f"  ⚡ [{obj.get('target_agent', '?')}] {obj.get('claim', '')}")
            lines.append(f"     → {obj.get('suggested_correction', '')}")

    final_score = final_risk_assessment.get("overall_risk_score", 0)
    lines.append(f"\nFinal risk score after resolution: {final_score:.1f}/10")
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
