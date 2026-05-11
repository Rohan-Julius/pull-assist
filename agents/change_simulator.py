"""
Change Simulator Agent — Enhanced

Enhancement 1: Evidence-backed runtime risks
  Every breaking_scenario now requires an 'evidence' array — specific lines
  or file references proving the caller actually uses the changed symbol.
  Forces the LLM to cite fetched file content, not hallucinate callers.

Enhancement (own): Confidence self-reporting
  Agent outputs a 'confidence' score (1–5). Low confidence automatically
  primes the Critic to scrutinise this output more closely.

Enhancement (own): Severity-weighted blast radius integration
  Caller files on HIGH_SEVERITY_PATH_PATTERNS automatically flag as critical.
"""

from agents.base import run_with_tools, AgentOutput
from agents.schema_validator import validate_and_repair
from tools.context_budget import budget_per_file_diff, budget_history
from config.settings import HIGH_SEVERITY_PATH_PATTERNS

SYSTEM_PROMPT = """You are a Change Simulator agent. You reason about ACTUAL runtime behavior.

Your job is to predict what WILL BREAK when this code change is deployed.
You are NOT a code reviewer. You are simulating production failure modes.

STRICT RULES:
1. Only report a breaking scenario if you have EVIDENCE from a file you fetched.
2. Every breaking_scenario MUST include an 'evidence' array with specific proof.
3. If you cannot find evidence of a caller breaking, report is_breaking_change: false.
4. Do not hallucinate callers. If search returns nothing, say so.
5. Self-report your confidence (1=guessing, 5=high certainty from direct evidence).

Evidence examples (be this specific):
  "lib/router/index.js imports 'handle' from express at line 12"
  "lib/application.js calls router.handle(req, res) at line ~220"
  "test/app.js asserts handle() throws on invalid input — this now silently returns null"

Failure mode taxonomy — use these exact terms:
  NULL_DEREF      — caller dereferences a value that can now be null/undefined
  TYPE_ERROR      — caller expects type A but receives type B
  MISSING_METHOD  — caller calls method that was removed or renamed
  SILENT_WRONG    — no crash but behavior is now incorrect (worst kind)
  ASYNC_MISMATCH  — caller treats async as sync or vice versa
  SCHEMA_BREAK    — API response shape changed, consumers will fail parsing
  SIDE_EFFECT     — new side effect introduced that breaks caller assumptions

Methodology:
1. Read the diff — what EXACTLY changed? (return type? signature? behavior?)
2. Fetch 1-2 caller files from the blast radius list provided
3. Quote the specific line in the caller that will break
4. Rate confidence honestly

Respond ONLY with this JSON (no preamble):
{
  "before_behavior": "exact description of old behavior",
  "after_behavior": "exact description of new behavior",
  "behavior_delta": "one-sentence diff between the two",
  "breaking_scenarios": [
    {
      "caller_file": "lib/router/index.js",
      "line_approx": 87,
      "failure_mode": "NULL_DEREF",
      "failure_description": "caller does user.email but getUserById now returns null",
      "severity": "HIGH",
      "evidence": [
        "lib/router/index.js imports getUserById at line 3",
        "lib/router/index.js calls getUserById(req.params.id) at line ~87",
        "line 88: res.json({ email: user.email }) — no null check"
      ]
    }
  ],
  "is_breaking_change": true,
  "simulator_summary": "one-line summary for the report header",
  "confidence": 4
}"""


def _flag_critical_callers(blast_radius: dict) -> list[str]:
    """
    Elevates callers that match HIGH_SEVERITY_PATH_PATTERNS to fetch first.
    Returns caller file paths sorted by criticality (critical paths first).
    """
    if not blast_radius:
        return []

    direct = blast_radius.get("direct_dependents", [])
    critical, normal = [], []

    for dep in direct[:5]:
        path = dep.get("file", "").lower()
        is_critical = any(pat in path for pat in HIGH_SEVERITY_PATH_PATTERNS)
        if is_critical:
            critical.append(dep["file"])
        else:
            normal.append(dep["file"])

    return critical + normal


def run(state: dict, tools: list, blast_radius: dict = None) -> AgentOutput:
    diff_content = budget_per_file_diff(
        state.get("_parsed_diff").changed_files if state.get("_parsed_diff") else []
    )
    history_text = budget_history(state.get("repo_history", "No prior history."))
    is_pure_addition = (state.get("total_deletions", 0) == 0 and state.get("total_additions", 0) > 0)

    # Build caller hints — critical paths first
    caller_hints = ""
    if blast_radius and blast_radius.get("direct_dependents"):
        prioritised = _flag_critical_callers(blast_radius)
        caller_hints = "CALLER FILES TO FETCH (priority order — critical paths first):\n"
        for path in prioritised[:3]:
            # Find the reason from blast radius
            reason = next(
                (d.get("reason", "") for d in blast_radius["direct_dependents"] if d.get("file") == path),
                ""
            )
            is_crit = any(pat in path.lower() for pat in HIGH_SEVERITY_PATH_PATTERNS)
            flag = " [CRITICAL PATH]" if is_crit else ""
            caller_hints += f"  - {path}{flag}: {reason}\n"
        caller_hints += (
            "\nFetch these files using: filepath::symbol_name\n"
            "Quote the EXACT LINE that will break as evidence."
        )

    # Pure-addition guard — tell the LLM explicitly
    addition_guard = ""
    if is_pure_addition:
        addition_guard = """
⚠️ CRITICAL CONTEXT — PURE ADDITION:
This PR has ZERO deletions. No existing code was modified, renamed, or removed.
All changes are new additions only (+{additions}/-0).

IMPLICATIONS:
- Existing callers CANNOT break because nothing they depend on was changed.
- Do NOT confuse existing npm packages / modules with newly added functions.
- If a file like lib/response.js already uses an npm package (e.g. 'on-finished'),
  that is UNRELATED to new functions added in this PR.
- is_breaking_change should almost certainly be false for pure additions.
- Only report breaking scenarios if the NEW code itself has a bug (not existing callers breaking).
""".format(additions=state.get('total_additions', 0))

    human_message = f"""Simulate the runtime impact of this code change. Only report what you can prove.

PR: {state.get('pr_title', '')}
Changed symbols: {', '.join(state.get('changed_symbols', []))}
Diff stats: +{state.get('total_additions', 0)} / -{state.get('total_deletions', 0)}
{addition_guard}
DIFF:
{diff_content}

{caller_hints}

REPO HISTORY:
{history_text}

Instructions:
1. Fetch 1-2 caller files and find the specific lines that will break
2. Include exact evidence quotes in every breaking scenario
3. If you find no evidence of breakage, set is_breaking_change: false
4. For pure additions: existing callers cannot break — focus on whether the new code itself is correct
5. Be honest about your confidence (1-5)

Respond with ONLY the JSON object."""

    output = run_with_tools(
        system_prompt=SYSTEM_PROMPT,
        human_message=human_message,
        tools=tools,
        agent_name="change_simulator",
    )

    # Validate and repair output schema
    if output.success:
        output.data = validate_and_repair(output.data, "change_simulator")

        # ── Pure-addition post-processing guard ───────────────────────────
        # If total_deletions == 0, the LLM cannot claim existing code breaks.
        # Override hallucinated breaking changes and cap severity.
        if is_pure_addition and output.data:
            was_breaking = output.data.get("is_breaking_change", False)
            if was_breaking:
                from rich.console import Console
                Console().print(
                    "  [yellow]⚠ Pure-addition guard: overriding is_breaking_change "
                    "→ false (0 deletions, no existing code modified)[/yellow]"
                )
                output.data["is_breaking_change"] = False
                output.data["_pure_addition_override"] = True

                # Downgrade scenario severities — pure additions can't cause
                # HIGH-severity breakage in existing callers
                for scenario in output.data.get("breaking_scenarios", []):
                    if scenario.get("severity") == "HIGH":
                        scenario["severity"] = "LOW"
                        scenario["_downgraded"] = "pure addition — existing callers unaffected"

    return output
