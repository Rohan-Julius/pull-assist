"""
Risk Evaluator Agent 
  Weighted scoring with explicit formula:
  overall_risk_score = (
    blast_radius_score   × 0.30  +
    test_coverage_score  × 0.30  +        
    runtime_risk_score   × 0.25  +
    complexity_score     × 0.15
  )
  
  Risk level thresholds (after computing score):
    0.0–3.0  → LOW
    3.1–5.9  → MEDIUM
    6.0–7.9  → HIGH
    8.0–10.0 → CRITICAL   


"""

import json
import re
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

CONSISTENCY: top_concerns MUST match dimension_scores. If runtime_risk_score is 0–1,
do not claim existing callers will crash or that production runtime errors are likely.

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


# ── CVE detection ──────────────────────────────────────────────────────────────
_CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)

# Risk floor when CVEs are present — a CVE fix is never LOW risk
_CVE_RISK_FLOOR = 7.0


def _detect_cves(state: dict) -> list[str]:
    """Extract CVE identifiers from the PR title, description, and diff."""
    text = " ".join([
        state.get("pr_title", ""),
        state.get("pr_description", ""),
        state.get("raw_diff", "")[:5000],  # check first 5k chars of diff
    ])
    return list(set(_CVE_PATTERN.findall(text)))


# If runtime_risk_score is this low, top_concerns must not imply caller/runtime breakage.
_RUNTIME_CALLER_CONCERN_FRAGMENTS = (
    "runtime error",
    "existing caller",
    "existing callers",
    "potential runtime",
    "caller will",
    "callers will",
    "caller may",
    "callers may",
    "caller ",
    "will break callers",
    "breaking existing",
    "missing_method",
    "silent error",
    "silent errors",
    "lack of null check",
    "lack of null checks",
    "null check",
    "null checks",
)


def _synthetic_concerns_from_state(state: dict, limit: int) -> list[str]:
    """Deterministic fallbacks when LLM concerns contradict scores."""
    out: list[str] = []
    tg = state.get("test_gaps") or {}
    for u in (tg.get("uncovered_functions") or [])[:3]:
        if isinstance(u, dict):
            fn = u.get("function", "?")
            risk = u.get("risk", "")
            out.append(f"Test gap: `{fn}` — add coverage ({risk} risk)".strip())
        elif isinstance(u, str):
            out.append(f"Test gap: {u}")
        if len(out) >= limit:
            return out
    br = state.get("blast_radius") or {}
    summ = (br.get("blast_radius_summary") or "").strip()
    if summ and len(out) < limit:
        out.append(f"Blast radius: {summ[:180]}")
    if not out:
        out.append("Review test coverage and monitoring for new code paths.")
    return out[:limit]


def _reconcile_top_concerns_with_scores(data: dict, state: dict) -> None:
    """
    Remove top_concerns that contradict dimension_scores (e.g. runtime caller
    doom when runtime_risk_score is ~0 after pure-addition pipeline).
    Refill from test gaps / blast summary when needed.
    """
    concerns = data.get("top_concerns")
    if not isinstance(concerns, list):
        return

    dims = data.get("dimension_scores") or {}
    try:
        rt = float(dims.get("runtime_risk_score", 0))
    except (TypeError, ValueError):
        rt = 0.0

    runtime_risks = state.get("runtime_risks") or {}
    is_breaking = bool(runtime_risks.get("is_breaking_change"))
    pure_override = bool(runtime_risks.get("_pure_addition_override"))
    is_pure = state.get("total_deletions", 0) == 0 and state.get("total_additions", 0) > 0
    no_runtime_proof = not (runtime_risks.get("breaking_scenarios") or [])
    verified_diff_runtime = any(
        isinstance(s, dict) and s.get("_verified_from_diff")
        for s in (runtime_risks.get("breaking_scenarios") or [])
    )

    strict_filter = (
        not verified_diff_runtime
        and (
            (pure_override and not is_breaking)
            or (is_pure and not is_breaking and no_runtime_proof)
            or (rt <= 1.0 and not is_breaking and no_runtime_proof)
        )
    )

    filtered: list[str] = []
    for c in concerns:
        if not isinstance(c, str):
            continue
        low = c.lower()
        drop = False
        if strict_filter and any(frag in low for frag in _RUNTIME_CALLER_CONCERN_FRAGMENTS):
            drop = True
        if rt <= 0.75 and not is_breaking:
            # Near-zero runtime: drop generic "runtime" / "caller" breakage language
            if any(
                x in low
                for x in (
                    "runtime",
                    "caller",
                    "break on deploy",
                    "production crash",
                )
            ):
                drop = True
        if not drop:
            filtered.append(c)

    if len(filtered) < len(concerns):
        data["_top_concerns_reconciled"] = True
        data["top_concerns"] = filtered

    # Drop stale "test coverage gap" hype when diff proves channel tests exist
    tg = state.get("test_gaps") or {}
    if tg.get("_diff_static_test_coverage") and not (tg.get("uncovered_functions") or []):
        cur = list(data.get("top_concerns") or [])
        dropped = False
        out_tc: list[str] = []
        for c in cur:
            if not isinstance(c, str):
                out_tc.append(c)
                continue
            cl = c.lower()
            if any(
                p in cl
                for p in (
                    "test coverage gap",
                    "test gaps",
                    "coverage gap",
                    "significant test coverage",
                    "uncovered changed",
                )
            ):
                dropped = True
                continue
            out_tc.append(c)
        if dropped:
            data["_top_concerns_test_gaps_reconciled"] = True
            data["top_concerns"] = out_tc

    # Ensure at least two concerns for managers when score is non-trivial
    need = 2 if float(data.get("overall_risk_score", 0)) > 2.5 else 1
    cur = data.get("top_concerns") or []
    if isinstance(cur, list) and len(cur) < need:
        extras = _synthetic_concerns_from_state(state, limit=need - len(cur))
        for ex in extras:
            if ex not in cur:
                cur.append(ex)
        data["top_concerns"] = cur[:5]


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
    diff_static_rt = bool(runtime_risks.get("_diff_static_runtime"))
    if is_pure_addition:
        if diff_static_rt:
            addition_note = """
⚠️ PURE ADDITION: This PR has 0 deletions — unchanged symbols and existing call sites
cannot break due to signature removal.

HOWEVER: runtime_risks may include diff-verified breaking_scenarios (_verified_from_diff)
for bugs in NEW code (e.g. instrumentation contracts). When those exist with HIGH severity,
runtime_risk_score MUST reflect real deployment impact (typically 5–8 for SILENT_WRONG
that skews APM or success metrics), not a near-zero score.
"""
        else:
            addition_note = f"""
⚠️ PURE ADDITION: This PR has 0 deletions — no existing code was changed.
  - runtime_risk_score should be LOW (0–3) because existing callers cannot break.
  - blast_radius_score should be LOW unless the new code is in a shared module.
  - Focus risk assessment on the quality of the NEW code, not on breakage.
"""

    # CVE context — security-critical PRs need elevated scoring
    cve_ids = _detect_cves(state)
    cve_note = ""
    if cve_ids:
        cve_note = f"""\n🔒 SECURITY: This PR references CVE(s): {', '.join(cve_ids)}
  - This is a security fix. The overall risk score MUST be at least {_CVE_RISK_FLOOR:.0f}/10.
  - blast_radius_score should reflect the scope of the vulnerability (typically 6+).
  - runtime_risk_score should reflect the behavioral change needed to mitigate the CVE.
  - DO NOT underrate security fixes. CVE fixes affect all deployments.
"""

    # Review comments context — what human reviewers flagged
    review_note = ""
    review_comments = state.get("review_comments", [])
    if review_comments:
        review_snippets = []
        for rc in review_comments[:8]:
            snippet = f"  [{rc.get('user', '?')}] on {rc.get('path', '?')}: {rc.get('body', '')[:150]}"
            review_snippets.append(snippet)
        review_note = "\n=== HUMAN REVIEWER COMMENTS (ground truth) ===\n" + "\n".join(review_snippets) + "\n"
        review_note += "Consider reviewer concerns when scoring dimensions.\n"

    human_message = f"""Score the deployment risk of this PR. Follow the weighted formula exactly.

PR: {state.get('pr_title', '')}
Description: {state.get('pr_description', '')[:500]}
Diff: {state.get('diff_summary', '')}
Files: {len(state.get('changed_files', []))} changed (+{state.get('total_additions', 0)}/-{state.get('total_deletions', 0)})
Has tests in PR: {state.get('has_test_changes', False)}
{conf_note}
{business_note}
{addition_note}
{cve_note}
{review_note}
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
                verified_diff_runtime = any(
                    isinstance(s, dict) and s.get("_verified_from_diff")
                    for s in (runtime_risks.get("breaking_scenarios") or [])
                )
                if not verified_diff_runtime:
                    runtime_score = float(dims.get("runtime_risk_score", 0))
                    if runtime_score > 3.0:
                        from rich.console import Console
                        Console().print(
                            f"  [yellow]⚠ Pure-addition cap: runtime_risk_score "
                            f"{runtime_score:.1f} → 3.0 (0 deletions, no breakage possible)[/yellow]"
                        )
                        dims["runtime_risk_score"] = 3.0
                        output.data["_pure_addition_capped"] = True

                    # Align with simulator: no breaking scenarios + not breaking → low runtime
                    if not runtime_risks.get("is_breaking_change", False) and not (
                        runtime_risks.get("breaking_scenarios") or []
                    ):
                        rt_cur = float(dims.get("runtime_risk_score", 0))
                        capped = min(rt_cur, 1.0)
                        if capped < rt_cur - 0.01:
                            dims["runtime_risk_score"] = round(capped, 2)
                            output.data["_runtime_aligned_no_breaking_scenarios"] = True

            # Diff-grounded test gap cleanup: empty uncovered → low test_coverage_score
            tg = state.get("test_gaps") or {}
            if tg.get("_diff_static_test_coverage") and not (tg.get("uncovered_functions") or []):
                try:
                    tcs = float(dims.get("test_coverage_score", 99))
                except (TypeError, ValueError):
                    tcs = 99.0
                if tcs > 3.0:
                    dims["test_coverage_score"] = 3.0
                    output.data["_test_coverage_reconciled_diff"] = True

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

            _reconcile_top_concerns_with_scores(output.data, state)

            # ── CVE floor: security fixes never score below _CVE_RISK_FLOOR ──
            if cve_ids:
                final_score = float(output.data.get("overall_risk_score", 0))
                if final_score < _CVE_RISK_FLOOR:
                    from rich.console import Console
                    Console().print(
                        f"  [yellow]⚠ CVE floor: score {final_score:.1f} → {_CVE_RISK_FLOOR:.1f} "
                        f"(PR references {', '.join(cve_ids)})[/yellow]"
                    )
                    output.data["overall_risk_score"] = _CVE_RISK_FLOOR
                    output.data["risk_level"] = _derive_risk_level(_CVE_RISK_FLOOR)
                    output.data["_cve_floor_applied"] = True
                    output.data["_cve_ids"] = cve_ids
                # Always tag CVE metadata
                output.data["_cve_ids"] = cve_ids

            # ── Docs/cosmetic ceiling: non-code PRs cannot be MEDIUM+ ────
            _NON_CODE_INDICATORS = (
                "docs/", "doc/", ".md", ".css", ".scss", ".less", ".txt",
                ".html", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".ico",
                ".yaml", ".yml", ".json", ".toml",
            )
            changed_files = state.get("changed_files", [])
            if changed_files:
                all_non_code = all(
                    any(ind in f.lower() for ind in _NON_CODE_INDICATORS)
                    for f in changed_files
                )
                analysis_symbols = state.get("analysis_symbols", state.get("changed_symbols", []))
                if all_non_code and not analysis_symbols and not cve_ids:
                    cur = float(output.data.get("overall_risk_score", 0))
                    _COSMETIC_CEILING = 2.0
                    if cur > _COSMETIC_CEILING:
                        from rich.console import Console
                        Console().print(
                            f"  [yellow]⚠ Cosmetic ceiling: {cur:.1f} → {_COSMETIC_CEILING:.1f} "
                            f"(all files are docs/CSS/config, no code symbols)[/yellow]"
                        )
                        output.data["overall_risk_score"] = _COSMETIC_CEILING
                        output.data["risk_level"] = "LOW"
                        output.data["_cosmetic_ceiling_applied"] = True

        output.data = validate_and_repair(output.data, "risk_evaluator")

    return output
