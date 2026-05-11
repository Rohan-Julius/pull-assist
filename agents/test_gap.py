"""
Test Gap Agent

Job: Identify which changed functions have no test coverage.
     If no test files were in the diff, actively search for existing tests.

Tools available:
  - find_test_files:     searches for test files referencing a symbol
  - fetch_file_from_main: fetches a test file to see its coverage

Output written to state["test_gaps"]:
  {
    "covered_functions": [str],
    "uncovered_functions": [{function, missing_scenario, risk}],
    "overall_coverage_assessment": "POOR|PARTIAL|ADEQUATE",
    "test_gap_summary": str
  }
"""

from agents.base import run_with_tools, AgentOutput
from tools.context_budget import format_symbols_for_prompt, budget_history

SYSTEM_PROMPT = """You are a Test Gap Agent. You identify missing test coverage.

Your job is to find which changed functions have NO tests.

Methodology:
1. Note whether test files were included in the diff (if yes, still check coverage)
2. Use find_test_files to search for existing tests for the changed symbols
3. If you find a test file, use fetch_file_from_main to read it
4. Identify specific MISSING scenarios — not just "needs more tests" but exactly what case is missing

Be specific. Don't say "getUserById needs tests". Say:
"getUserById has no test for the case where the user ID doesn't exist and null is returned"

Severity guide:
  HIGH   = changed function has ZERO tests AND is in a critical path (auth, payments, data writes)
  MEDIUM = changed function has tests but the new behavior introduced by the diff is untested
  LOW    = changed function has good coverage, minor edge case missing

CRITICAL: You MUST actually call the tools and wait for the real Observation before drawing conclusions.
Do NOT make up or hallucinate tool results. Do NOT write fake Observations yourself.
You MUST call find_test_files or fetch_file_from_main and wait for the actual result.

After you have gathered results from the tools, provide your final answer as a JSON object.
The JSON format must be:
{
  "covered_functions": ["functionA", "functionB"],
  "uncovered_functions": [
    {
      "function": "onMessage",
      "missing_scenario": "no test for when the diagnostic channel listener throws an error",
      "risk": "MEDIUM"
    }
  ],
  "overall_coverage_assessment": "POOR",
  "test_gap_summary": "Brief summary of coverage situation"
}"""


def run(state: dict, tools: list) -> AgentOutput:
    symbols_text = format_symbols_for_prompt(
        state.get("changed_symbols", []),
        state.get("per_file_context", []),
    )
    history_text = budget_history(state.get("repo_history", "No prior history."))
    has_test_changes = state.get("has_test_changes", False)
    test_files_in_diff = state.get("test_files", [])

    test_context = ""
    if has_test_changes:
        test_context = f"NOTE: The diff INCLUDES test file changes: {', '.join(test_files_in_diff)}\nStill verify these tests actually cover the new behavior."
    else:
        test_context = "NOTE: The diff does NOT include any test file changes. Search for existing tests."

    human_message = f"""Identify test coverage gaps for this PR.

PR: {state.get('pr_title', '')}
Diff summary: {state.get('diff_summary', '')}

CHANGED SYMBOLS:
{symbols_text}

{test_context}

REPO HISTORY:
{history_text}

Step 1: Use find_test_files for the most important changed symbols.
Step 2: Fetch any test files found to check their actual coverage.
Step 3: Identify specific missing scenarios.
Step 4: Respond with the JSON format specified."""

    return run_with_tools(
        system_prompt=SYSTEM_PROMPT,
        human_message=human_message,
        tools=tools,
        agent_name="test_gap",
    )