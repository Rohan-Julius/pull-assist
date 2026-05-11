"""
Tests for github/client.py

These tests hit the real GitHub API for the PUBLIC test repo:
  https://github.com/expressjs/express

No token is needed for public repos on GET /contents, but the search
endpoint requires auth. Set GITHUB_TOKEN in .env before running.

Run with:  python -m pytest tests/test_github_client.py -v
"""

import pytest
from github.client import GitHubClient, parse_pr_url


# ── parse_pr_url ───────────────────────────────────────────────────────────────

class TestParsePrUrl:

    def test_standard_url(self):
        repo, pr = parse_pr_url("https://github.com/expressjs/express/pull/5944")
        assert repo == "expressjs/express"
        assert pr == 5944

    def test_url_with_trailing_slash(self):
        repo, pr = parse_pr_url("https://github.com/owner/my-repo/pull/99/files")
        assert repo == "owner/my-repo"
        assert pr == 99

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Could not parse"):
            parse_pr_url("https://github.com/owner/repo")

    def test_non_github_url_raises(self):
        with pytest.raises(ValueError):
            parse_pr_url("https://gitlab.com/owner/repo/merge_requests/1")


# ── GitHubClient — live API tests ──────────────────────────────────────────────
# These use a well-known public PR on expressjs/express that will never change.
# PR #7171: https://github.com/expressjs/express/pull/7171

TEST_REPO = "expressjs/express"
TEST_PR   = 7171


class TestGitHubClientMetadata:

    @pytest.fixture(scope="class")
    def client(self):
        return GitHubClient(repo=TEST_REPO)

    def test_fetch_metadata_returns_dict(self, client):
        meta = client.fetch_pr_metadata(TEST_PR)
        assert isinstance(meta, dict)

    def test_metadata_has_title(self, client):
        meta = client.fetch_pr_metadata(TEST_PR)
        assert "title" in meta and len(meta["title"]) > 0

    def test_metadata_has_author(self, client):
        meta = client.fetch_pr_metadata(TEST_PR)
        assert "author" in meta and len(meta["author"]) > 0

    def test_metadata_pr_number_matches(self, client):
        meta = client.fetch_pr_metadata(TEST_PR)
        assert meta["pr_number"] == TEST_PR

    def test_metadata_has_additions_deletions(self, client):
        meta = client.fetch_pr_metadata(TEST_PR)
        assert isinstance(meta["additions"], int)
        assert isinstance(meta["deletions"], int)

    def test_metadata_cached_on_second_call(self, client):
        """Second call should hit cache — same object returned."""
        meta1 = client.fetch_pr_metadata(TEST_PR)
        meta2 = client.fetch_pr_metadata(TEST_PR)
        assert meta1 is meta2   # same dict object from cache


class TestGitHubClientDiff:

    @pytest.fixture(scope="class")
    def client(self):
        return GitHubClient(repo=TEST_REPO)

    @pytest.fixture(scope="class")
    def raw_diff(self, client):
        return client.fetch_pr_diff(TEST_PR)

    def test_diff_is_string(self, raw_diff):
        assert isinstance(raw_diff, str)

    def test_diff_not_empty(self, raw_diff):
        assert len(raw_diff) > 100

    def test_diff_contains_unified_markers(self, raw_diff):
        assert "diff --git" in raw_diff
        assert "@@" in raw_diff

    def test_diff_cached_on_second_call(self, client):
        diff1 = client.fetch_pr_diff(TEST_PR)
        diff2 = client.fetch_pr_diff(TEST_PR)
        assert diff1 is diff2


class TestGitHubClientFetchFile:

    @pytest.fixture(scope="class")
    def client(self):
        return GitHubClient(repo=TEST_REPO)

    def test_fetch_existing_file(self, client):
        content = client.fetch_file("package.json")
        assert "express" in content.lower()

    def test_fetch_nonexistent_file_returns_not_found(self, client):
        content = client.fetch_file("this/does/not/exist.js")
        assert "NOT FOUND" in content

    def test_fetch_with_symbol_slices_content(self, client):
        content = client.fetch_file("lib/express.js", symbol="createApplication")
        # Should include a header showing line range
        assert "Lines" in content or "createApplication" in content

    def test_fetch_result_within_line_limit(self, client):
        from config.settings import GITHUB_FETCH_MAX_LINES
        content = client.fetch_file("README.md")
        line_count = len(content.splitlines())
        assert line_count <= GITHUB_FETCH_MAX_LINES + 5  # +5 for header line


class TestGitHubClientFileTree:

    @pytest.fixture(scope="class")
    def client(self):
        return GitHubClient(repo=TEST_REPO)

    def test_file_tree_returns_list(self, client):
        tree = client.get_file_tree()
        assert isinstance(tree, list)

    def test_file_tree_not_empty(self, client):
        tree = client.get_file_tree()
        assert len(tree) > 0

    def test_file_tree_contains_package_json(self, client):
        tree = client.get_file_tree()
        assert "package.json" in tree

    def test_file_tree_depth_respected(self, client):
        tree = client.get_file_tree(depth=1)
        for path in tree:
            assert "/" not in path, f"Depth-1 tree should have no subdirs: {path}"


class TestGitHubClientErrors:

    def test_invalid_token_raises_runtime_error(self):
        client = GitHubClient(repo=TEST_REPO, token="invalid_token_xyz")
        with pytest.raises(RuntimeError, match="invalid or expired"):
            client.fetch_pr_metadata(TEST_PR)

    def test_missing_token_raises_value_error(self):
        with pytest.raises(ValueError, match="GITHUB_TOKEN is not set"):
            GitHubClient(repo=TEST_REPO, token="")

    def test_nonexistent_pr_raises_value_error(self):
        client = GitHubClient(repo=TEST_REPO)
        with pytest.raises(ValueError, match="not found"):
            client.fetch_pr_diff(9999999)