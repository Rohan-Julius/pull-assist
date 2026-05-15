"""
Critic Agent

Job: Challenge all four prior agents' findings. Find inconsistencies,
     understated risks, and things nobody mentioned.

This is the agent that makes the system non-linear. If it raises
SIGNIFICANT_ISSUES, the Orchestrator sends objections back to the
flagged agents for a second round.

No GitHub tool calls — works only from text inputs.

Output written to state["objections"]:
  {
    "objections": [{target_agent, claim, reason, suggested_correction}],
    "missed_impacts": [str],
    "verdict": "AGREE|MINOR_ISSUES|SIGNIFICANT_ISSUES",
    "critic_summary": str
  }
"""

import json
from agents.base import run_without_tools, AgentOutput

SYSTEM_PROMPT = """You are a Critic agent. Your job is to find flaws in other agents' analyses.

You are skeptical, precise, and constructive. You do NOT repeat what agents said correctly.
You only raise issues when you have a specific, evidence-based reason.

What to look for:
1. SCORE INCONSISTENCY: e.g. Dependency Mapper found 8 affected files but Risk Evaluator gave blast_radius_score=2
2. LOGICAL CONFLICT: e.g. Change Simulator says "no breaking change" but the diff removes a function
3. MISSED IMPACT: e.g. a shared utility was changed but nobody mentioned it affects the test suite itself
4. UNDERSTATED RISK: e.g. the change touches authentication but runtime_risk_score is LOW
5. OVERSTATED RISK: e.g. the change is a comment update but risk_level is HIGH
6. TEST VS RUNTIME: If change simulator reports is_breaking_change false and empty breaking_scenarios
   (especially pure additions with 0 deletions), do NOT demand a higher runtime_risk_score for
   "lack of tests" — that belongs in test_coverage_score, not runtime_risk_score.
7. If runtime_risks.breaking_scenarios is NON-EMPTY, never claim it is "empty" or missing.

GROUNDING RULES (critical):
- ONLY reference symbols, files, and facts that appear in the data provided below.
- Do NOT fabricate or reference symbols that are not listed in CHANGED SYMBOLS or PER-FILE SYMBOLS.
- Do NOT reference projects, libraries, or PRs other than the one being analyzed.
- If you're unsure about a fact, do NOT assert it — omit it instead.

Verdict guide:
  AGREE              = all agents' findings are internally consistent and well-reasoned
  MINOR_ISSUES       = small gaps or inconsistencies that don't change the overall risk level
  SIGNIFICANT_ISSUES = major inconsistency or missed risk that should change the risk score

Only raise SIGNIFICANT_ISSUES if you are confident. False alarms waste re-run cycles.

Always respond with ONLY a JSON object:
{
  "objections": [
    {
      "target_agent": "risk_evaluator",
      "claim": "overall_risk_score of 2.0 is too low",
      "reason": "Dependency Mapper found onMessage used in 5 production files, none of which have tests",
      "suggested_correction": "risk_score should be at least 5.0 given zero test coverage on affected files"
    }
  ],
  "missed_impacts": ["list any impacts none of the agents mentioned"],
  "verdict": "AGREE",
  "critic_summary": "One sentence overall verdict"
}"""


def run(state: dict) -> AgentOutput:
    blast_radius = state.get("blast_radius", {})
    runtime_risks = state.get("runtime_risks", {})
    test_gaps = state.get("test_gaps", {})
    risk_assessment = state.get("risk_assessment", {})

    # Build per-file symbol summary for ground truth verification
    per_file = state.get("analysis_per_file_context", state.get("per_file_context", []))
    file_symbols_text = ""
    if per_file:
        lines = []
        for f in per_file:
            syms = f.get("symbols", [])
            sym_str = ", ".join(syms) if syms else "(no symbols detected)"
            lines.append(f"  {f.get('path', '?')} [{f.get('language', '?')}] +{f.get('additions', 0)}/-{f.get('deletions', 0)}: {sym_str}")
        file_symbols_text = "\n".join(lines)

    # All changed files (source + test) for blast radius verification
    all_changed = state.get("changed_files", [])
    source_files = state.get("source_files", [])
    test_files = state.get("test_files", [])

    # PR description (may contain CVE refs, design rationale)
    pr_desc = (state.get("pr_description") or "")[:600]

    # Human reviewer comments (ground truth)
    review_comments = state.get("review_comments", [])
    review_text = ""
    if review_comments:
        snippets = []
        for rc in review_comments[:10]:
            snippets.append(f"  [{rc.get('user', '?')}] {rc.get('path', '')}:{rc.get('line', '')} — {rc.get('body', '')[:200]}")
        review_text = "\n=== HUMAN REVIEWER COMMENTS (ground truth — use these to check agent blind spots) ===\n" + "\n".join(snippets) + "\n"

    human_message = f"""Review all agent findings for this PR and identify any issues.

PR: {state.get('pr_title', '')}
Description: {pr_desc}
Diff: {state.get('diff_summary', '')}
Changed symbols: {', '.join(state.get('changed_symbols', []))}

=== GROUND TRUTH: CHANGED FILES ({len(all_changed)} total) ===
Source files ({len(source_files)}): {', '.join(source_files)}
Test files ({len(test_files)}): {', '.join(test_files)}

=== PER-FILE SYMBOLS (from diff parser) ===
{file_symbols_text}
{review_text}
=== DEPENDENCY MAPPER OUTPUT ===
{json.dumps(blast_radius, indent=2)}

=== CHANGE SIMULATOR OUTPUT ===
{json.dumps(runtime_risks, indent=2)}

=== TEST GAP AGENT OUTPUT ===
{json.dumps(test_gaps, indent=2)}

=== RISK EVALUATOR OUTPUT ===
{json.dumps(risk_assessment, indent=2)}

Critically review these findings. Look for inconsistencies, missed impacts,
and under/over-stated risks.

KEY CHECKS:
1. Does blast_radius_summary match the actual number of changed files above?
2. Are all important symbols from PER-FILE SYMBOLS tracked by agents?
3. If human reviewers flagged concerns, are they reflected in the risk assessment?
4. Does test_gap assessment match whether test files were actually changed in the PR?

Remember: test coverage gaps are primarily reflected in test_coverage_score.
Do not flag SIGNIFICANT_ISSUES solely because runtime_risk_score is low while
breaking_scenarios is empty and is_breaking_change is false.

Respond with ONLY the JSON object."""

    return run_without_tools(
        system_prompt=SYSTEM_PROMPT,
        human_message=human_message,
        agent_name="critic",
    )


def build_rerun_context(original_output: dict, objection: dict) -> str:
    """
    Builds the extra context injected into a re-run prompt when the Critic
    raised an objection about a specific agent.
    """
    return f"""
=== CRITIC'S OBJECTION (re-run requested) ===
The Critic agent challenged your previous output:
  Claim:       {objection.get('claim', '')}
  Reason:      {objection.get('reason', '')}
  Correction:  {objection.get('suggested_correction', '')}

Your previous output was:
{json.dumps(original_output, indent=2)}

Please reconsider your analysis in light of this objection.
If the Critic is correct, revise your output. If you disagree, explain why in your JSON.
Add a "rerun_notes" field to your JSON response explaining what changed (or why you maintain your position).
"""