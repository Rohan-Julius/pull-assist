"""
Change Simulator Agent
This agent simulates the runtime impact of code changes, predicting what will break in production.
It uses evidence from the codebase to identify specific breaking scenarios, their failure modes, and severity.
Strict rules prevent hallucination of breakage without evidence. A pure-addition guard ensures accurate assessment.

"""

from agents.base import run_with_tools, AgentOutput
from agents.schema_validator import validate_and_repair
from tools.context_budget import budget_per_file_diff, budget_history
from config.settings import HIGH_SEVERITY_PATH_PATTERNS
from github.diff_static_risks import augment_runtime_risks_with_diff

SYSTEM_PROMPT = """You are a Change Simulator agent. You reason about ACTUAL runtime behavior.

Your job is to predict what WILL BREAK when this code change is deployed.
You are NOT a code reviewer. You are simulating production failure modes.

STRICT RULES:
1. Only report a breaking scenario if you have EVIDENCE from a file you FETCHED via fetch_file (Observation in your trace).
2. Every breaking_scenario MUST include an 'evidence' array with specific proof from fetched file content.
3. If you cannot find evidence of a caller breaking, report is_breaking_change: false.
4. Do not hallucinate callers. If search returns nothing, say so.
5. Self-report your confidence (1=guessing, 5=high certainty from direct evidence).
6. You MUST NOT output the final JSON until you have called fetch_file at least once whenever the prompt lists CALLER FILES TO FETCH — otherwise set breaking_scenarios to [] and confidence to 1.

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
Changed symbols: {', '.join(state.get('analysis_symbols', state.get('changed_symbols', [])))}
Diff stats: +{state.get('total_additions', 0)} / -{state.get('total_deletions', 0)}
{addition_guard}
DIFF:
{diff_content}

{caller_hints}

REPO HISTORY:
{history_text}

Instructions:
1. If CALLER FILES TO FETCH is non-empty: call fetch_file at least once BEFORE your Final Answer JSON.
2. Include exact evidence quotes in every breaking scenario (only from fetched content).
3. If you find no evidence of breakage, set is_breaking_change: false
4. For pure additions: existing callers cannot break — focus on whether the new code itself is correct
5. Be honest about your confidence (1-5)

Respond with ONLY the JSON object after completing any required tool calls."""

    output = run_with_tools(
        system_prompt=SYSTEM_PROMPT,
        human_message=human_message,
        tools=tools,
        agent_name="change_simulator",
    )

    # Validate and repair output schema
    if output.success:
        output.data = validate_and_repair(output.data, "change_simulator")

        prioritised = _flag_critical_callers(blast_radius or {})
        if prioritised and output.tool_calls_made == 0:
            from rich.console import Console

            Console().print(
                "  [yellow]⚠ Change simulator: 0 fetch_file calls but blast radius lists "
                "callers — discarding unverified breaking_scenarios[/yellow]"
            )
            output.data["breaking_scenarios"] = []
            output.data["is_breaking_change"] = False
            try:
                c = int(output.data.get("confidence", 5))
            except (TypeError, ValueError):
                c = 5
            output.data["confidence"] = min(c, 2)
            prev = (output.data.get("simulator_summary") or "").strip()
            suf = " [fetch_file was not invoked — no verified file evidence for scenarios]"
            output.data["simulator_summary"] = (prev + suf).strip()
            output.data["_no_fetch_evidence"] = True
            if is_pure_addition:
                output.data["_pure_addition_override"] = True

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

                # Ground truth: with 0 deletions, existing callers cannot break on
                # this diff shape. Prior scenarios that alleged caller breakage are
                # invalid — remove them instead of leaving contradictory narratives.
                prior = output.data.get("breaking_scenarios") or []
                n_prior = len(prior)
                output.data["breaking_scenarios"] = []
                summary = (output.data.get("simulator_summary") or "").strip()
                suffix = (
                    f" Pure addition (+{state.get('total_additions', 0)}/-0): "
                    "no verified breaking scenarios for existing callers."
                )
                if n_prior:
                    suffix += f" ({n_prior} speculative scenario(s) discarded.)"
                output.data["simulator_summary"] = (summary + suffix).strip()

        # Diff-grounded runtime risks (e.g. on-finished + unconditional finish publish)
        output.data = augment_runtime_risks_with_diff(
            output.data,
            state.get("raw_diff", ""),
            int(state.get("total_deletions", 0) or 0),
        )

    return output
