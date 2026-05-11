"""
Context budget management.

Agents receive chunks of diff and file content as text.
If those chunks are too large they silently truncate the LLM's context,
causing hallucinations or missed findings.

Rules enforced here:
  - Per-file diff:    max 300 lines
  - File fetch:       max 200 lines  (already capped in client, this is a safety net)
  - Symbol search:    max 8 results × 300 chars snippet
  - Full diff:        max 500 lines sent to any single agent
  - History context:  max 800 chars
"""

MAX_DIFF_LINES_PER_FILE = 300
MAX_FULL_DIFF_LINES = 500
MAX_FILE_LINES = 200
MAX_HISTORY_CHARS = 800
MAX_SEARCH_RESULTS = 8
MAX_SNIPPET_CHARS = 300


def budget_diff(raw_diff: str, max_lines: int = MAX_FULL_DIFF_LINES) -> str:
    """
    Truncate a full diff to `max_lines` lines.
    Appends a notice if truncated so the LLM knows it's seeing a subset.
    """
    lines = raw_diff.splitlines()
    if len(lines) <= max_lines:
        return raw_diff
    truncated = lines[:max_lines]
    truncated.append(
        f"\n[TRUNCATED — showing {max_lines} of {len(lines)} lines. "
        "Focus analysis on the lines shown above.]"
    )
    return "\n".join(truncated)


def budget_file(file_content: str, max_lines: int = MAX_FILE_LINES) -> str:
    """Truncate a fetched file to `max_lines` lines."""
    lines = file_content.splitlines()
    if len(lines) <= max_lines:
        return file_content
    truncated = lines[:max_lines]
    truncated.append(
        f"\n[TRUNCATED — showing {max_lines} of {len(lines)} lines.]"
    )
    return "\n".join(truncated)


def budget_history(history_text: str) -> str:
    """Truncate repo history context to MAX_HISTORY_CHARS."""
    if len(history_text) <= MAX_HISTORY_CHARS:
        return history_text
    return history_text[:MAX_HISTORY_CHARS] + "\n[... history truncated]"


def budget_per_file_diff(changed_files: list, max_lines_each: int = MAX_DIFF_LINES_PER_FILE) -> str:
    """
    Build a compact diff summary for the Change Simulator.
    Shows each file's raw_diff up to max_lines_each lines, separated by headers.
    """
    parts = []
    for f in changed_files:
        if not f.raw_diff:
            continue
        lines = f.raw_diff.splitlines()
        header = f"\n{'='*60}\nFILE: {f.path}  [{f.language}]  +{f.additions}/-{f.deletions}\n{'='*60}"
        body_lines = lines[:max_lines_each]
        if len(lines) > max_lines_each:
            body_lines.append(f"[... {len(lines)-max_lines_each} more lines truncated]")
        parts.append(header + "\n" + "\n".join(body_lines))
    return "\n".join(parts) if parts else "[No diff content available]"


def format_symbols_for_prompt(symbols: list[str], per_file: list[dict]) -> str:
    """
    Format changed symbols into a concise prompt-ready string.
    Groups symbols by file for readability.
    """
    lines = []
    for file_info in per_file:
        if file_info.get("symbols"):
            lines.append(f"  {file_info['path']} ({file_info['language']}):")
            for sym in file_info["symbols"]:
                lines.append(f"    - {sym}")
    if not lines:
        lines = [f"  {s}" for s in symbols[:20]]
    return "\n".join(lines) if lines else "  (no symbols detected)"