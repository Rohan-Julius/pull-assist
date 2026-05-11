"""
Tests for memory/store.py

Run with:  python -m pytest tests/test_memory_store.py -v
"""

import pytest
import os
from datetime import datetime, UTC
from memory.store import MemoryStore
from memory.schema import PRAnalysisRecord


TEST_DB = "memory/test_pr_history.db"


@pytest.fixture(autouse=True)
def clean_db():
    """Remove test DB before and after each test."""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


@pytest.fixture
def store():
    return MemoryStore(db_path=TEST_DB)


def make_record(**overrides) -> PRAnalysisRecord:
    defaults = dict(
        repo="owner/repo",
        pr_number=1,
        pr_title="Fix auth bug",
        analyzed_at=datetime.now(UTC).isoformat(),
        files_touched=["src/auth.py", "src/user.py"],
        symbols_changed=["getUserById", "verifyToken"],
        languages=["python"],
        additions=45,
        deletions=12,
        overall_risk_score=7.5,
        risk_level="HIGH",
        blast_radius_count=5,
        had_test_gaps=True,
        top_concerns=["null return not handled", "no test for edge case"],
        critic_verdict="MINOR_ISSUES",
        conflict_rounds=1,
    )
    defaults.update(overrides)
    return PRAnalysisRecord(**defaults)


class TestSaveAndRetrieve:

    def test_save_and_get_record(self, store):
        record = make_record()
        store.save_analysis(record)
        retrieved = store.get_pr_record("owner/repo", 1)
        assert retrieved is not None
        assert retrieved.pr_number == 1

    def test_risk_score_persisted(self, store):
        store.save_analysis(make_record(overall_risk_score=8.2))
        r = store.get_pr_record("owner/repo", 1)
        assert r.overall_risk_score == 8.2

    def test_files_touched_persisted(self, store):
        store.save_analysis(make_record())
        r = store.get_pr_record("owner/repo", 1)
        assert "src/auth.py" in r.files_touched

    def test_symbols_persisted(self, store):
        store.save_analysis(make_record())
        r = store.get_pr_record("owner/repo", 1)
        assert "getUserById" in r.symbols_changed

    def test_had_test_gaps_persisted(self, store):
        store.save_analysis(make_record(had_test_gaps=True))
        r = store.get_pr_record("owner/repo", 1)
        assert r.had_test_gaps is True

    def test_nonexistent_record_returns_none(self, store):
        r = store.get_pr_record("owner/repo", 999)
        assert r is None

    def test_upsert_updates_existing(self, store):
        store.save_analysis(make_record(overall_risk_score=5.0))
        store.save_analysis(make_record(overall_risk_score=9.0))  # same pr_number
        r = store.get_pr_record("owner/repo", 1)
        assert r.overall_risk_score == 9.0


class TestRepoContext:

    def test_no_history_returns_none(self, store):
        ctx = store.get_repo_context("owner/repo")
        assert ctx is None

    def test_context_after_one_pr(self, store):
        store.save_analysis(make_record())
        ctx = store.get_repo_context("owner/repo")
        assert ctx is not None
        assert ctx.total_prs_analyzed == 1

    def test_avg_risk_score_calculated(self, store):
        store.save_analysis(make_record(pr_number=1, overall_risk_score=6.0))
        store.save_analysis(make_record(pr_number=2, overall_risk_score=8.0))
        ctx = store.get_repo_context("owner/repo")
        assert ctx.avg_risk_score == 7.0

    def test_high_risk_count(self, store):
        store.save_analysis(make_record(pr_number=1, overall_risk_score=8.0))
        store.save_analysis(make_record(pr_number=2, overall_risk_score=4.0))
        ctx = store.get_repo_context("owner/repo")
        assert ctx.high_risk_prs == 1

    def test_most_touched_files_populated(self, store):
        store.save_analysis(make_record(pr_number=1, files_touched=["src/auth.py", "src/user.py"]))
        store.save_analysis(make_record(pr_number=2, files_touched=["src/auth.py", "src/order.py"]))
        ctx = store.get_repo_context("owner/repo")
        assert "src/auth.py" in ctx.most_touched_files

    def test_recent_summaries_populated(self, store):
        store.save_analysis(make_record(pr_number=1, pr_title="Fix auth"))
        ctx = store.get_repo_context("owner/repo")
        assert any("Fix auth" in s for s in ctx.recent_summaries)

    def test_different_repos_isolated(self, store):
        store.save_analysis(make_record(repo="owner/repo-a", pr_number=1))
        ctx = store.get_repo_context("owner/repo-b")
        assert ctx is None


class TestPromptFormatting:

    def test_format_no_history(self, store):
        result = store.format_context_for_prompt("owner/repo")
        assert "No prior analysis" in result

    def test_format_with_history(self, store):
        store.save_analysis(make_record())
        result = store.format_context_for_prompt("owner/repo")
        assert "PRs analyzed" in result
        assert "Avg risk" in result

    def test_format_string_not_empty(self, store):
        store.save_analysis(make_record())
        result = store.format_context_for_prompt("owner/repo")
        assert len(result) > 50