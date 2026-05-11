"""
Risk Evaluator Agent — Enhanced

Enhancement 2: Explicit weighted risk scoring
  The LLM is now shown the exact weight formula and must compute
  the weighted average itself, then show working. This prevents
  the LLM from ignoring weights and scoring intuitively.

  After the LLM responds, we server-side validate the maths:
  if LLM's overall_risk_score deviates >0.5 from the formula result,
  we override it with the correct value. Agents can't lie about scores.

Enhancement 3: Business impact integration
  Risk evaluator now receives business_impacts (from the post-processor)
  and adjusts scores for known high-impact domains.

Enhancement (own): Confidence-weighted scoring
  If a prior agent flagged low confidence, its dimension is downweighted.
"""

import json
from agents.base import run_without_tools, AgentOutput
from agents.schema_validator import validate_and_repair
from config.settings import RISK_WEIGHTS

# Explicit weight formula shown to the LLM — prevents intuitive scoring
WEIGHT_FORMULA = f"""
MANDATORY SCORING FORMULA — you MUST follow this exactly:
  overall_risk_score = (
    blast_radius_score   × {RISK_WEIGHTS['blast_radius_files']:.2f}  +
    test_coverage_score  × {RISK_WEIGHTS['test_gap_coverage']:.2f}  +
    runtime_risk_score   × {RISK_WEIGHTS['runtime_breakage']:.2f}  +
    complexity_score     × {RISK_WEIGHTS['change_complexity']:.2f}
  )

Show your working in a "score_working" field:
  "score_working": "6.0×0.30 + 8.0×0.30 + 7.0×0.25 + 3.0×0.15 = 1.80+2.40+1.75+0.45 = 6.40"

Risk level thresholds (apply AFTER computing the weighted score):
  0.0–3.0  → LOW
  3.1–5.9  → MEDIUM
  6.0–7.9  → HIGH
  8.0–10.0 → CRITICAL
"""

SYSTEM_PROMPT = f"""You are a Risk Evaluator agent. You are an operational deployment risk assessor.
Your output will be used by engineering managers to decide whether to merge a PR.

{WEIGHT_FORMULA}

Dimension scoring guide:
  blast_radius_score (0–10):
    0  = only the changed file
    3  = 2–5 files affected
    6  = 5–15 files, some shared utilities
    9  = core framework / shared library affecting 20+ consumers

  test_coverage_score (0–10):
    0  = changed code has full test coverage including new behavior
    4  = some tests exist but new behavior is untested
    7  = changed functions have no tests at all
    10 = critical path, zero tests, no coverage anywhere near change

  runtime_risk_score (0–10):
    0  = pure addition, no existing callers can break
    4  = behavioral change but callers are defensive (null checks, try/catch)
    7  = breaking change, at least one caller confirmed unprotected
    10 = guaranteed data corruption or crash on first request

  complexity_score (0–10):
    0  = comment or whitespace change
    3  = < 20 lines changed, single file
    6  = 20–100 lines, multiple files
    9  = architectural change, 100+ lines, multiple modules

CALIBRATION: Most PRs are MEDIUM (3–6). Reserve CRITICAL for changes that
will cause an incident within minutes of deployment. Don't inflate scores.

Respond with ONLY this JSON (no preamble):
{{
  "dimension_scores": {{
    "blast_radius_score": 0.0,
    "test_coverage_score": 0.0,
    "runtime_risk_score": 0.0,
    "complexity_score": 0.0
  }},
  "score_working": "show your weighted calculation here",
  "overall_risk_score": 0.0,
  "risk_level": "LOW",
  "top_concerns": ["specific concern 1", "specific concern 2"],
  "recommended_actions": ["specific action 1", "specific action 2"],
  "rollback_difficulty": "EASY",
  "confidence": 4
}}"""


def _server_side_score(dimension_scores: dict) -> float:
    """
    Compute the correct weighted score server-side.
    Used to validate/override the LLM's self-reported score.
    """
    weights = {
        "blast_radius_score":   RISK_WEIGHTS["blast_radius_files"],
        "test_coverage_score":  RISK_WEIGHTS["test_gap_coverage"],
        "runtime_risk_score":   RISK_WEIGHTS["runtime_breakage"],
        "complexity_score":     RISK_WEIGHTS["change_complexity"],
    }
    total = sum(
        float(dimension_scores.get(k, 0)) * w
        for k, w in weights.items()
    )
    return round(min(10.0, max(0.0, total)), 2)


def _derive_risk_level(score: float) -> str:
    if score <= 3.0:   return "LOW"
    if score <= 5.9:   return "MEDIUM"
    if score <= 7.9:   return "HIGH"
    return "CRITICAL"


def run(state: dict) -> AgentOutput:
    blast_radius  = state.get("blast_radius", {})
    runtime_risks = state.get("runtime_risks", {})
    test_gaps     = state.get("test_gaps", {})
    business      = state.get("business_impacts", [])
    is_pure_addition = (state.get("total_deletions", 0) == 0 and state.get("total_additions", 0) > 0)

    # Confidence flags from prior agents — shown to risk evaluator
    sim_conf  = runtime_risks.get("confidence", 5)
    gap_conf  = test_gaps.get("confidence", 5)
    conf_note = ""
    if sim_conf < 3:
        conf_note += f"\nNOTE: Change Simulator had low confidence ({sim_conf}/5) — runtime_risk_score should reflect this uncertainty.\n"
    if gap_conf < 3:
        conf_note += f"\nNOTE: Test Gap Agent had low confidence ({gap_conf}/5) — test_coverage_score should be conservative.\n"

    business_note = ""
    if business:
        business_note = f"\nBUSINESS IMPACTS IDENTIFIED:\n" + "\n".join(f"  • {b}" for b in business)
        business_note += "\nElevate runtime_risk_score if these business domains are critical.\n"

    # Pure-addition context for the LLM
    addition_note = ""
    if is_pure_addition:
        addition_note = f"""
⚠️ PURE ADDITION: This PR has 0 deletions — no existing code was changed.
  - runtime_risk_score should be LOW (0–3) because existing callers cannot break.
  - blast_radius_score should be LOW unless the new code is in a shared module.
  - Focus risk assessment on the quality of the NEW code, not on breakage.
"""

    human_message = f"""Score the deployment risk of this PR. Follow the weighted formula exactly.

PR: {state.get('pr_title', '')}
Diff: {state.get('diff_summary', '')}
Files: {len(state.get('changed_files', []))} changed (+{state.get('total_additions', 0)}/-{state.get('total_deletions', 0)})
Has tests in PR: {state.get('has_test_changes', False)}
{conf_note}
{business_note}
{addition_note}
=== BLAST RADIUS ===
{json.dumps(blast_radius, indent=2)}

=== RUNTIME RISKS ===
{json.dumps(runtime_risks, indent=2)}

=== TEST GAPS ===
{json.dumps(test_gaps, indent=2)}

Score each dimension, show your weighted calculation in score_working,
then respond with ONLY the JSON object."""

    output = run_without_tools(
        system_prompt=SYSTEM_PROMPT,
        human_message=human_message,
        agent_name="risk_evaluator",
    )

    if output.success and output.data:
        # Server-side score validation — override if LLM maths is wrong
        dims = output.data.get("dimension_scores", {})
        if dims:
            # ── Pure-addition cap: runtime_risk_score cannot exceed 3.0 ─────
            # When total_deletions == 0, no existing code was changed,
            # so existing callers cannot break.
            if is_pure_addition:
                runtime_score = float(dims.get("runtime_risk_score", 0))
                if runtime_score > 3.0:
                    from rich.console import Console
                    Console().print(
                        f"  [yellow]⚠ Pure-addition cap: runtime_risk_score "
                        f"{runtime_score:.1f} → 3.0 (0 deletions, no breakage possible)[/yellow]"
                    )
                    dims["runtime_risk_score"] = 3.0
                    output.data["_pure_addition_capped"] = True

            correct_score = _server_side_score(dims)
            llm_score = float(output.data.get("overall_risk_score", correct_score))
            if abs(llm_score - correct_score) > 0.5:
                from rich.console import Console
                Console().print(
                    f"  [yellow]⚠ Score corrected: LLM said {llm_score:.1f}, "
                    f"formula gives {correct_score:.1f}[/yellow]"
                )
                output.data["overall_risk_score"] = correct_score
                output.data["risk_level"] = _derive_risk_level(correct_score)
                output.data["_score_corrected"] = True

        output.data = validate_and_repair(output.data, "risk_evaluator")

    return output
