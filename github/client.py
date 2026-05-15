import base64
import time
import requests
from config.settings import (
    GITHUB_TOKEN,
    GITHUB_API_BASE,
    GITHUB_SEARCH_MAX_RESULTS,
    GITHUB_FETCH_MAX_LINES,
    GITHUB_CONTEXT_WINDOW_LINES,
)


class GitHubClient:
    """
    Wrapper around the GitHub REST API.
    Three capabilities:
      1. fetch_pr_diff      — get raw unified diff for a PR
      2. fetch_file         — get a file from main branch (with context slicing)
      3. search_symbol      — find files in the repo referencing a symbol
    """

    def __init__(self, repo: str, token: str = GITHUB_TOKEN):
        if not token:
            raise ValueError(
                "GITHUB_TOKEN is not set. Add it to your .env file.\n"
                "Generate one at: https://github.com/settings/tokens\n"
                "Required scope: repo (read-only is fine for public repos)"
            )
        self.repo = repo          # "owner/repo-name"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self._cache: dict = {}    # simple in-memory cache to avoid duplicate API calls
        self._default_branch: str = None  # lazily cached

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get(self, url: str, params: dict = None, accept_override: str = None) -> requests.Response:
        """GET with auth headers and basic rate-limit handling."""
        headers = self.headers.copy()
        if accept_override:
            headers["Accept"] = accept_override

        response = requests.get(url, headers=headers, params=params, timeout=15)

        # Handle rate limiting gracefully
        if response.status_code == 403:
            reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset_time - int(time.time()), 1)
            raise RuntimeError(
                f"GitHub API rate limit hit. Resets in {wait}s. "
                "Consider adding a GITHUB_TOKEN with higher limits."
            )

        if response.status_code == 401:
            raise RuntimeError(
                "GitHub token is invalid or expired. "
                "Check your GITHUB_TOKEN in .env"
            )

        return response

    def _cache_key(self, *args) -> str:
        return "|".join(str(a) for a in args)

    def _get_default_branch(self) -> str:
        """
        Get the default branch for this repo (e.g., 'main', 'master').
        Cached to avoid repeated API calls.
        """
        if self._default_branch:
            return self._default_branch

        url = f"{GITHUB_API_BASE}/repos/{self.repo}"
        response = self._get(url)
        response.raise_for_status()
        data = response.json()
        self._default_branch = data.get("default_branch", "main")
        return self._default_branch

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_pr_diff(self, pr_number: int) -> str:
        """
        Fetch the raw unified diff for a pull request.
        Returns the full diff as a string.
        """
        cache_key = self._cache_key("diff", self.repo, pr_number)
        if cache_key in self._cache:
            return self._cache[cache_key]

        url = f"{GITHUB_API_BASE}/repos/{self.repo}/pulls/{pr_number}"
        response = self._get(url, accept_override="application/vnd.github.v3.diff")

        if response.status_code == 404:
            raise ValueError(
                f"PR #{pr_number} not found in {self.repo}. "
                "Check the repo name and PR number."
            )

        response.raise_for_status()
        diff = response.text
        self._cache[cache_key] = diff
        return diff

    def fetch_pr_metadata(self, pr_number: int) -> dict:
        """
        Fetch PR metadata: title, description, author, base branch, created_at.
        """
        cache_key = self._cache_key("meta", self.repo, pr_number)
        if cache_key in self._cache:
            return self._cache[cache_key]

        url = f"{GITHUB_API_BASE}/repos/{self.repo}/pulls/{pr_number}"
        response = self._get(url)
        response.raise_for_status()

        data = response.json()
        metadata = {
            "pr_number":    pr_number,
            "title":        data.get("title", ""),
            "description":  data.get("body", "") or "",
            "author":       data.get("user", {}).get("login", "unknown"),
            "base_branch":  data.get("base", {}).get("ref", "main"),
            "head_branch":  data.get("head", {}).get("ref", ""),
            "created_at":   data.get("created_at", ""),
            "changed_files":data.get("changed_files", 0),
            "additions":    data.get("additions", 0),
            "deletions":    data.get("deletions", 0),
            "state":        data.get("state", ""),
            "html_url":     data.get("html_url", ""),
        }

        self._cache[cache_key] = metadata
        return metadata

    def fetch_pr_reviews(self, pr_number: int) -> list[dict]:
        """
        Fetch review comments (inline code comments) from a PR.
        Returns a list of dicts: {user, body, path, line, state}
        """
        cache_key = self._cache_key("reviews", self.repo, pr_number)
        if cache_key in self._cache:
            return self._cache[cache_key]

        url = f"{GITHUB_API_BASE}/repos/{self.repo}/pulls/{pr_number}/comments"
        response = self._get(url, params={"per_page": 50})

        if response.status_code == 404:
            return []

        response.raise_for_status()
        items = response.json()

        reviews = []
        for item in items[:50]:
            reviews.append({
                "user": item.get("user", {}).get("login", "unknown"),
                "body": (item.get("body") or "")[:500],
                "path": item.get("path", ""),
                "line": item.get("original_line") or item.get("line", 0),
                "created_at": item.get("created_at", ""),
            })

        self._cache[cache_key] = reviews
        return reviews

    def fetch_pr_review_states(self, pr_number: int) -> list[dict]:
        """
        Fetch top-level review states (APPROVED, CHANGES_REQUESTED, COMMENTED).
        Returns a list of dicts: {user, state, body}
        """
        cache_key = self._cache_key("review_states", self.repo, pr_number)
        if cache_key in self._cache:
            return self._cache[cache_key]

        url = f"{GITHUB_API_BASE}/repos/{self.repo}/pulls/{pr_number}/reviews"
        response = self._get(url, params={"per_page": 30})

        if response.status_code == 404:
            return []

        response.raise_for_status()
        items = response.json()

        states = []
        for item in items[:30]:
            body = (item.get("body") or "")[:300]
            states.append({
                "user": item.get("user", {}).get("login", "unknown"),
                "state": item.get("state", ""),
                "body": body,
            })

        self._cache[cache_key] = states
        return states

    def fetch_file(self, filepath: str, ref: str = None, symbol: str = None) -> str:
        """
        Fetch a file from the repo at a given ref (branch/commit).
        If ref is None, uses the repo's default branch.

        If `symbol` is provided, returns only GITHUB_CONTEXT_WINDOW_LINES lines
        centred around the first occurrence of that symbol — keeping the context
        window small and focused.

        If no symbol is given, returns up to GITHUB_FETCH_MAX_LINES from the top.
        """
        if ref is None:
            ref = self._get_default_branch()
        cache_key = self._cache_key("file", self.repo, filepath, ref, symbol)
        if cache_key in self._cache:
            return self._cache[cache_key]

        url = f"{GITHUB_API_BASE}/repos/{self.repo}/contents/{filepath}"
        response = self._get(url, params={"ref": ref})

        if response.status_code == 404:
            return f"[FILE NOT FOUND: {filepath} on {ref}]"

        response.raise_for_status()
        data = response.json()

        # GitHub returns base64-encoded content
        raw_content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        lines = raw_content.splitlines()

        if symbol:
            # Find the first line containing the symbol and slice around it
            symbol_line = next(
                (i for i, line in enumerate(lines) if symbol in line),
                None
            )
            if symbol_line is not None:
                half = GITHUB_CONTEXT_WINDOW_LINES // 2
                start = max(0, symbol_line - half)
                end = min(len(lines), symbol_line + half)
                sliced = lines[start:end]
                header = f"# [Lines {start+1}–{end} of {filepath} (centred on '{symbol}')]\n"
                result = header + "\n".join(sliced)
            else:
                # Symbol not found in file — return the top portion
                result = "\n".join(lines[:GITHUB_FETCH_MAX_LINES])
        else:
            result = "\n".join(lines[:GITHUB_FETCH_MAX_LINES])

        self._cache[cache_key] = result
        return result

    def search_symbol(self, symbol: str) -> list[dict]:
        """
        Search the repo for files referencing `symbol`.
        Returns a list of dicts: {path, html_url, snippet}

        Note: GitHub code search requires authentication and has stricter
        rate limits (10 requests/min for unauthenticated, 30/min authenticated).
        Results are cached to avoid re-hitting the same symbol.
        """
        cache_key = self._cache_key("search", self.repo, symbol)
        if cache_key in self._cache:
            return self._cache[cache_key]

        url = f"{GITHUB_API_BASE}/search/code"
        params = {
            "q": f"{symbol} repo:{self.repo}",
            "per_page": GITHUB_SEARCH_MAX_RESULTS,
        }
        # Code search requires a slightly different Accept header
        response = self._get(
            url,
            params=params,
            accept_override="application/vnd.github.v3.text-match+json",
        )

        if response.status_code == 422:
            # GitHub rejects queries that are too short or contain only operators
            return []

        response.raise_for_status()
        items = response.json().get("items", [])

        results = []
        for item in items[:GITHUB_SEARCH_MAX_RESULTS]:
            text_matches = item.get("text_matches", [])
            snippet = text_matches[0].get("fragment", "") if text_matches else ""
            results.append({
                "path":     item["path"],
                "html_url": item["html_url"],
                "snippet":  snippet,
            })

        self._cache[cache_key] = results
        return results

    def get_file_tree(self, ref: str = None, depth: int = 2) -> list[str]:
        """
        Get a flat list of file paths in the repo up to `depth` directories deep.
        If ref is None, uses the repo's default branch.
        Used by the Dependency Mapper to understand the repo layout.
        """
        if ref is None:
            ref = self._get_default_branch()
        cache_key = self._cache_key("tree", self.repo, ref, depth)
        if cache_key in self._cache:
            return self._cache[cache_key]

        url = f"{GITHUB_API_BASE}/repos/{self.repo}/git/trees/{ref}"
        response = self._get(url, params={"recursive": "1"})

        if response.status_code == 409:
            # Empty repo
            return []

        response.raise_for_status()
        tree = response.json().get("tree", [])

        # Filter to only files (not trees/dirs) within requested depth
        paths = [
            item["path"]
            for item in tree
            if item["type"] == "blob"
            and item["path"].count("/") < depth
        ]

        self._cache[cache_key] = paths
        return paths


def parse_pr_url(url: str) -> tuple[str, int]:
    """
    Parse a GitHub PR URL into (repo, pr_number).

    Accepts:
      https://github.com/owner/repo/pull/123
      https://github.com/owner/repo/pull/123/files
    """
    import re
    pattern = r"github\.com/([^/]+/[^/]+)/pull/(\d+)"
    match = re.search(pattern, url)
    if not match:
        raise ValueError(
            f"Could not parse PR URL: {url}\n"
            "Expected format: https://github.com/owner/repo/pull/123"
        )
    repo = match.group(1)
    pr_number = int(match.group(2))
    return repo, pr_number