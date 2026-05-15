"""
Dependency Mapper Agent

Job: Given the changed symbols, find which other files in the repo
     import or call those symbols. Produces the "blast radius".

Tools available:
  - search_symbol_in_repo: searches for symbol usage across the repo
  - get_repo_file_tree:    gets the file structure for context

Output written to state["blast_radius"]:
  {
    "direct_dependents": [{file, reason, confidence}],
    "indirect_dependents": [{file, reason, confidence}],
    "blast_radius_summary": str
  }
"""

from agents.base import run_with_tools, AgentOutput
from tools.context_budget import format_symbols_for_prompt, budget_history

SYSTEM_PROMPT = """You are a Dependency Mapper agent analyzing a code change.

Your ONLY job is to find which files in this repository are affected by the changed symbols.
Work methodically:
1. Use search_symbol_in_repo for each important changed symbol (max 3 searches)
2. Use get_repo_file_tree ONCE if you need to understand the project structure
3. Based on search results, identify direct and indirect dependents

Be precise — only list files that genuinely import or call the changed symbols.
Do not guess. If search returns no results, report zero dependents with confidence LOW.

CRITICAL: You MUST actually call the tools and wait for the real Observation before drawing conclusions.
Do NOT make up or hallucinate tool results. Do NOT skip the tool-calling step.

SYMBOL PRECISION:
- Treat search hits as valid only when they reference the exact changed symbol name,
  not homonyms (e.g. onFinished / on-finished vs onFinish).

After you have gathered results from the tools, provide your final answer as a JSON object.
The JSON format must be:
{
  "direct_dependents": [
    {"file": "path/to/file.js", "reason": "calls onMessage on line ~45", "confidence": "HIGH"}
  ],
  "indirect_dependents": [
    {"file": "path/to/file.js", "reason": "imports from module that uses onMessage", "confidence": "MEDIUM"}
  ],
  "blast_radius_summary": "2 files directly affected, 1 indirectly"
}"""


def run(state: dict, tools: list) -> AgentOutput:
    """
    Entry point called by the Orchestrator.
    Reads from state, returns AgentOutput, does NOT write to state.
    The Orchestrator writes results back.
    """
    symbols_text = format_symbols_for_prompt(
      state.get("analysis_symbols", state.get("changed_symbols", [])),
      state.get("analysis_per_file_context", state.get("per_file_context", [])),
    )
    history_text = budget_history(state.get("repo_history", "No prior history."))
    diff_summary = state.get("diff_summary", "")
    base_branch = state.get("base_branch", "main")

    human_message = f"""Analyze this pull request and find all files that depend on the changed symbols.

PR: {state.get('pr_title', '')}
Base branch: {base_branch}
Diff summary: {diff_summary}

CHANGED SYMBOLS (by file):
{symbols_text}

REPO HISTORY CONTEXT:
{history_text}

Use the search tool to find which other files in the repo reference these symbols.
Focus on the most important symbols first (public functions, exported classes).
SKIP local variable names and single-word common names — only search for function, type, or constant names.

SYMBOL MATCHING (critical):
- Only list a file as dependent if the hit is for THIS EXACT symbol token (e.g. onFinish),
  not a different name (e.g. onFinished package, function onfinish, or unrelated strings).
- If search snippets look like a different identifier, use confidence LOW or omit the file.
- In Go, only exported symbols (capitalized) create cross-file dependencies. Skip lowercase locals.

After searching, respond with the JSON format specified."""

    return run_with_tools(
        system_prompt=SYSTEM_PROMPT,
        human_message=human_message,
        tools=tools,
        agent_name="dependency_mapper",
    )