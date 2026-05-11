"""
Local Git Client

Mirrors the GitHubClient interface but uses local filesystem + git commands.
Enables the analyzer to work on local repos without GitHub API calls.

Used in two modes:
  1. --diff FILE --local-repo PATH  →  fully offline analysis
  2. --diff FILE --repo owner/repo  →  diff is local, tool calls go to GitHub
     (handled by using GitHubClient for tools but skipping diff fetch)
"""

import os
import subprocess
from pathlib import Path
from config.settings import (
    GITHUB_SEARCH_MAX_RESULTS,
    GITHUB_FETCH_MAX_LINES,
    GITHUB_CONTEXT_WINDOW_LINES,
)


class LocalGitClient:
    """
    Drop-in replacement for GitHubClient that reads from a local git repo.

    Implements the same public API:
      - search_symbol(symbol)  → git grep
      - fetch_file(filepath)   → read from disk
      - get_file_tree(depth)   → os.walk
    """

    def __init__(self, repo_path: str):
        self.repo_path = str(Path(repo_path).resolve())
        if not Path(self.repo_path).is_dir():
            raise ValueError(f"Local repo path does not exist: {self.repo_path}")

        # Verify it's a git repo
        git_dir = Path(self.repo_path) / ".git"
        if not git_dir.exists():
            raise ValueError(
                f"Not a git repository: {self.repo_path}\n"
                "Make sure you point to the root of a git project."
            )

        self.repo = self.repo_path  # used for display/logging
        self._cache: dict = {}

    def _run_git(self, *args, **kwargs) -> subprocess.CompletedProcess:
        """Run a git command in the repo directory."""
        return subprocess.run(
            ["git", *args],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            timeout=30,
            **kwargs,
        )

    def search_symbol(self, symbol: str) -> list[dict]:
        """
        Search for files referencing `symbol` using git grep.
        Returns same format as GitHubClient: [{path, html_url, snippet}]
        """
        cache_key = f"search|{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            result = self._run_git("grep", "-rn", "--max-count=3", "-I", symbol)
        except subprocess.TimeoutExpired:
            return []

        if result.returncode != 0:
            # grep returns 1 when no matches found
            self._cache[cache_key] = []
            return []

        results = []
        seen_files = set()
        for line in result.stdout.strip().split("\n"):
            if not line or ":" not in line:
                continue
            # Format: filepath:line_number:content
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            filepath = parts[0]
            snippet = parts[2].strip()

            if filepath in seen_files:
                continue
            seen_files.add(filepath)

            results.append({
                "path": filepath,
                "html_url": f"file://{self.repo_path}/{filepath}",
                "snippet": snippet,
            })

            if len(results) >= GITHUB_SEARCH_MAX_RESULTS:
                break

        self._cache[cache_key] = results
        return results

    def fetch_file(self, filepath: str, ref: str = None, symbol: str = None) -> str:
        """
        Read a file from the local repo.
        If `symbol` is given, returns lines centred around it.
        The `ref` parameter is accepted for interface compatibility but ignored
        (reads from working tree).
        """
        cache_key = f"file|{filepath}|{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        full_path = Path(self.repo_path) / filepath
        if not full_path.is_file():
            return f"[FILE NOT FOUND: {filepath}]"

        try:
            content = full_path.read_text(errors="replace")
        except Exception as e:
            return f"[ERROR READING FILE: {filepath} — {e}]"

        lines = content.splitlines()

        if symbol:
            # Find the first line containing the symbol and slice around it
            symbol_line = next(
                (i for i, line in enumerate(lines) if symbol in line),
                None,
            )
            if symbol_line is not None:
                half = GITHUB_CONTEXT_WINDOW_LINES // 2
                start = max(0, symbol_line - half)
                end = min(len(lines), symbol_line + half)
                sliced = lines[start:end]
                header = f"# [Lines {start+1}–{end} of {filepath} (centred on '{symbol}')]\n"
                result = header + "\n".join(sliced)
            else:
                result = "\n".join(lines[:GITHUB_FETCH_MAX_LINES])
        else:
            result = "\n".join(lines[:GITHUB_FETCH_MAX_LINES])

        self._cache[cache_key] = result
        return result

    def get_file_tree(self, ref: str = None, depth: int = 2) -> list[str]:
        """
        Get file paths in the repo up to `depth` directories deep.
        Uses os.walk instead of GitHub API.
        """
        cache_key = f"tree|{depth}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        paths = []
        repo = Path(self.repo_path)

        for root, dirs, files in os.walk(repo):
            # Skip hidden dirs (.git, .venv, etc.) and node_modules
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".")
                and d not in ("node_modules", "__pycache__", "venv", ".venv", "dist", "build")
            ]

            rel_root = Path(root).relative_to(repo)
            current_depth = len(rel_root.parts)
            if current_depth >= depth:
                dirs.clear()  # don't descend further
                continue

            for f in files:
                if f.startswith("."):
                    continue
                rel_path = str(rel_root / f) if str(rel_root) != "." else f
                paths.append(rel_path)

        self._cache[cache_key] = paths
        return paths
