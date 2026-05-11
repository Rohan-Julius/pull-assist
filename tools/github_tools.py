"""
LangChain @tool wrappers around GitHubClient.

Each tool is a plain function decorated with @tool.
They are injected into agents that need them — not all agents get all tools.

Tool assignment:
  dependency_mapper  → search_symbol_in_repo, get_repo_file_tree
  change_simulator   → fetch_file_from_main
  test_gap_agent     → fetch_file_from_main, find_test_files
  risk_evaluator     → (no GitHub tools — works from prior agent outputs only)
  critic             → (no GitHub tools — works from prior agent outputs only)
"""

from langchain.tools import tool
from config.settings import GITHUB_MAX_TOOL_CALLS_PER_AGENT


def make_github_tools(github_client):
    """
    Factory — returns a dict of LangChain tools bound to a specific GitHubClient.

    Usage:
        tools = make_github_tools(client)
        dependency_tools = [tools["search_symbol"], tools["file_tree"]]
        simulator_tools  = [tools["fetch_file"]]
    """

    # Per-run call counter — enforces GITHUB_MAX_TOOL_CALLS_PER_AGENT cap
    call_counts: dict[str, int] = {}

    def _check_cap(tool_name: str) -> str | None:
        """Returns an error message if cap exceeded, None otherwise."""
        call_counts[tool_name] = call_counts.get(tool_name, 0) + 1
        if call_counts[tool_name] > GITHUB_MAX_TOOL_CALLS_PER_AGENT:
            return (
                f"STOP — tool call limit reached. '{tool_name}' has been called "
                f"{call_counts[tool_name]} times (max {GITHUB_MAX_TOOL_CALLS_PER_AGENT}). "
                "You MUST now provide your Final Answer using the information you "
                "already gathered. Do NOT call any more tools."
            )
        return None

    # ── Tool 1: Symbol search ──────────────────────────────────────────────────

    @tool
    def search_symbol_in_repo(symbol: str) -> str:
        """
        Search the repository for all files that reference a given symbol
        (function name, class name, variable name, or import).

        Use this to find which files depend on a changed symbol.
        Input: the exact symbol name as a string (e.g. 'getUserById').
        Returns: list of file paths and code snippets where the symbol appears.
        """
        cap_msg = _check_cap("search_symbol_in_repo")
        if cap_msg:
            return cap_msg
        results = github_client.search_symbol(symbol)
        if not results:
            return f"No files found referencing '{symbol}' in this repository."

        lines = [f"Files referencing '{symbol}':"]
        for r in results:
            lines.append(f"\n  PATH: {r['path']}")
            if r.get("snippet"):
                snippet = r["snippet"][:300].replace("\n", " ")
                lines.append(f"  SNIPPET: {snippet}")
        return "\n".join(lines)

    # ── Tool 2: Fetch file from main ───────────────────────────────────────────

    @tool
    def fetch_file_from_main(filepath_and_symbol: str) -> str:
        """
        Fetch the content of a file from the main/master branch.

        Use this to see code that was NOT in the diff but is needed to understand
        what breaks — for example, a caller of a changed function.

        Input format (TWO options):
          "src/services/userService.js"
              → fetches first 200 lines of the file

          "src/services/userService.js::getUserById"
              → fetches ~100 lines centred on 'getUserById' in the file

        Always prefer the '::symbol' form to stay within context limits.
        """
        cap_msg = _check_cap("fetch_file_from_main")
        if cap_msg:
            return cap_msg

        if "::" in filepath_and_symbol:
            filepath, symbol = filepath_and_symbol.split("::", 1)
        else:
            filepath, symbol = filepath_and_symbol, None

        filepath = filepath.strip()
        symbol = symbol.strip() if symbol else None

        content = github_client.fetch_file(filepath.strip(), symbol=symbol)
        return f"[FILE: {filepath}]\n{content}"

    # ── Tool 3: Get file tree ──────────────────────────────────────────────────

    @tool
    def get_repo_file_tree(depth: str = "2") -> str:
        """
        Get a flat list of file paths in the repository (up to 2 directories deep).

        Use this to understand the overall repo structure — e.g. to find where
        test files live, or to confirm whether a file exists before fetching it.

        Input: depth as a string ("1" or "2"). Default is "2".
        Returns: newline-separated list of file paths.
        """
        cap_msg = _check_cap("get_repo_file_tree")
        if cap_msg:
            return cap_msg
        try:
            d = int(depth)
        except ValueError:
            d = 2
        paths = github_client.get_file_tree(depth=d)
        if not paths:
            return "Repository file tree is empty or could not be fetched."
        return "Repository file tree:\n" + "\n".join(paths[:150])

    # ── Tool 4: Find test files ────────────────────────────────────────────────

    @tool
    def find_test_files(symbol_or_filename: str) -> str:
        """
        Search for test files related to a given symbol or source filename.

        Use this when the diff did NOT include test changes and you need to check
        whether tests already exist for the changed code.

        Input: either a function/class name (e.g. 'getUserById') or a source
        filename (e.g. 'authService.js').
        Returns: list of matching test file paths and relevant snippets.
        """
        cap_msg = _check_cap("find_test_files")
        if cap_msg:
            return cap_msg

        # First try searching by the symbol directly
        results = github_client.search_symbol(symbol_or_filename)

        # Filter to test files only
        test_results = [
            r for r in results
            if any(seg in r["path"].lower() for seg in [
                "test", "spec", "__tests__", ".test.", ".spec."
            ])
        ]

        if not test_results:
            return (
                f"No test files found referencing '{symbol_or_filename}'. "
                "This is a test gap — the changed symbol has no direct test coverage."
            )

        lines = [f"Test files referencing '{symbol_or_filename}':"]
        for r in test_results:
            lines.append(f"\n  PATH: {r['path']}")
            if r.get("snippet"):
                snippet = r["snippet"][:300].replace("\n", " ")
                lines.append(f"  SNIPPET: {snippet}")
        return "\n".join(lines)

    return {
        "search_symbol":  search_symbol_in_repo,
        "fetch_file":     fetch_file_from_main,
        "file_tree":      get_repo_file_tree,
        "find_test_files": find_test_files,
    }