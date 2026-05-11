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

    human_message = f"""Review all agent findings for this PR and identify any issues.

PR: {state.get('pr_title', '')}
Diff: {state.get('diff_summary', '')}
Changed symbols: {', '.join(state.get('changed_symbols', []))}

=== DEPENDENCY MAPPER OUTPUT ===
{json.dumps(blast_radius, indent=2)}

=== CHANGE SIMULATOR OUTPUT ===
{json.dumps(runtime_risks, indent=2)}

=== TEST GAP AGENT OUTPUT ===
{json.dumps(test_gaps, indent=2)}

=== RISK EVALUATOR OUTPUT ===
{json.dumps(risk_assessment, indent=2)}

Critically review these findings. Look for inconsistencies, missed impacts,
and under/over-stated risks. Respond with ONLY the JSON object."""

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