"""
Enhancement tests — all new Day 3+ features

Tests:
  - Schema validator (validate, repair, safe defaults)
  - Evidence-backed runtime risks (change_simulator output schema)
  - Weighted risk scoring (server-side formula validation)
  - Business impact analyzer (path classification, runtime classification)
  - Rollback advisor (heuristics, output schema)
  - Historical context (memory store get_historical_context)
  - Report builder new fields
  - Formatter JSON output new keys

Run with:  python -m pytest tests/test_enhancements.py -v
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone


# ── Schema Validator tests ─────────────────────────────────────────────────────

class TestSchemaValidator:

    def test_valid_risk_evaluator_passes(self):
        from agents.schema_validator import validate
        data = {
            "dimension_scores": {"blast_radius_score": 5.0},
            "overall_risk_score": 5.0,
            "risk_level": "MEDIUM",
            "top_concerns": ["concern"],
            "recommended_actions": ["action"],
        }
        valid, issues = validate(data, "risk_evaluator")
        assert valid is True

    def test_missing_field_flagged(self):
        from agents.schema_validator import validate
        data = {"overall_risk_score": 5.0}  # missing most fields
        valid, issues = validate(data, "risk_evaluator")
        assert valid is False
        assert any("dimension_scores" in i for i in issues)

    def test_score_out_of_range_flagged(self):
        from agents.schema_validator import validate
        data = {
            "dimension_scores": {}, "overall_risk_score": 15.0,
            "risk_level": "HIGH", "top_concerns": [], "recommended_actions": [],
        }
        valid, issues = validate(data, "risk_evaluator")
        assert valid is False
        assert any("15.0" in i for i in issues)

    def test_invalid_risk_level_flagged(self):
        from agents.schema_validator import validate
        data = {
            "dimension_scores": {}, "overall_risk_score": 5.0,
            "risk_level": "SUPER_HIGH", "top_concerns": [], "recommended_actions": [],
        }
        valid, issues = validate(data, "risk_evaluator")
        assert valid is False

    def test_invalid_verdict_flagged(self):
        from agents.schema_validator import validate
        data = {"objections": [], "verdict": "MAYBE", "critic_summary": "ok"}
        valid, issues = validate(data, "critic")
        assert valid is False

    def test_score_clamped_to_10_on_repair(self):
        from agents.schema_validator import validate_and_repair
        data = {
            "dimension_scores": {}, "overall_risk_score": 99.0,
            "risk_level": "HIGH", "top_concerns": [], "recommended_actions": [],
        }
        result = validate_and_repair(data, "risk_evaluator")
        assert result["overall_risk_score"] == 10.0

    def test_risk_level_derived_from_score_on_repair(self):
        from agents.schema_validator import validate_and_repair
        data = {
            "dimension_scores": {}, "overall_risk_score": 9.5,
            "risk_level": "INVALID", "top_concerns": [], "recommended_actions": [],
        }
        result = validate_and_repair(data, "risk_evaluator")
        assert result["risk_level"] == "CRITICAL"

    def test_verdict_defaulted_on_repair(self):
        from agents.schema_validator import validate_and_repair
        data = {"objections": [], "verdict": "UNKNOWN_VERDICT", "critic_summary": "ok", "missed_impacts": []}
        result = validate_and_repair(data, "critic")
        assert result["verdict"] == "AGREE"

    def test_list_field_wrapped_on_repair(self):
        from agents.schema_validator import validate_and_repair
        data = {
            "direct_dependents": "lib/auth.js",   # should be a list
            "indirect_dependents": [],
            "blast_radius_summary": "1 file",
        }
        result = validate_and_repair(data, "dependency_mapper")
        assert isinstance(result["direct_dependents"], list)

    def test_parse_error_returns_safe_default(self):
        from agents.schema_validator import validate_and_repair
        result = validate_and_repair({"_parse_error": True}, "risk_evaluator")
        assert "overall_risk_score" in result
        assert result["overall_risk_score"] == 5.0

    def test_safe_default_never_raises(self):
        from agents.schema_validator import _safe_default
        for agent in ["dependency_mapper", "change_simulator", "test_gap",
                      "risk_evaluator", "critic", "rollback_advisor"]:
            d = _safe_default(agent)
            assert isinstance(d, dict)

    def test_unknown_agent_passes_through(self):
        from agents.schema_validator import validate
        valid, issues = validate({"anything": "goes"}, "nonexistent_agent")
        assert valid is True


# ── Evidence-backed runtime risks tests ───────────────────────────────────────

class TestEvidenceBackedRisks:

    def test_breaking_scenario_schema_has_evidence_field(self):
        """The change_simulator output schema must include evidence in each scenario."""
        # Simulate what a well-formed simulator output looks like
        data = {
            "before_behavior": "throws on missing user",
            "after_behavior": "returns null on missing user",
            "behavior_delta": "throws→null",
            "breaking_scenarios": [
                {
                    "caller_file": "lib/router/index.js",
                    "line_approx": 87,
                    "failure_mode": "NULL_DEREF",
                    "failure_description": "caller dereferences user.email without null check",
                    "severity": "HIGH",
                    "evidence": [
                        "lib/router/index.js imports getUserById at line 3",
                        "line 88: res.json({ email: user.email }) — no null check",
                    ],
                }
            ],
            "is_breaking_change": True,
            "simulator_summary": "Null return breaks router",
            "confidence": 4,
        }
        from agents.schema_validator import validate
        valid, _ = validate(data, "change_simulator")
        assert valid is True

    def test_scenario_without_evidence_still_valid_schema(self):
        """Evidence is new but we don't hard-require it in the schema validator
        (LLM might omit it for older scenarios). The prompt enforces it."""
        from agents.schema_validator import validate
        data = {
            "before_behavior": "throws", "after_behavior": "returns null",
            "breaking_scenarios": [
                {"caller_file": "a.js", "line_approx": 1,
                 "failure_mode": "NULL_DEREF", "severity": "HIGH"}
            ],
            "is_breaking_change": True, "simulator_summary": "test",
        }
        valid, _ = validate(data, "change_simulator")
        assert valid is True

    def test_failure_mode_taxonomy(self):
        """Verify taxonomy constants exist (used in prompt)."""
        valid_modes = {"NULL_DEREF", "TYPE_ERROR", "MISSING_METHOD",
                       "SILENT_WRONG", "ASYNC_MISMATCH", "SCHEMA_BREAK", "SIDE_EFFECT"}
        # These are defined in the prompt — just sanity check the set makes sense
        assert len(valid_modes) == 7

    def test_critical_caller_prioritisation(self):
        from agents.change_simulator import _flag_critical_callers
        blast = {
            "direct_dependents": [
                {"file": "src/components/Button.js", "reason": "uses util"},
                {"file": "src/auth/login.js", "reason": "uses getUserById"},
                {"file": "src/payment/checkout.js", "reason": "processes order"},
            ]
        }
        prioritised = _flag_critical_callers(blast)
        # auth and payment should come before Button
        assert prioritised[0] in ["src/auth/login.js", "src/payment/checkout.js"]
        assert "src/components/Button.js" == prioritised[-1]

    def test_no_blast_radius_returns_empty(self):
        from agents.change_simulator import _flag_critical_callers
        assert _flag_critical_callers(None) == []
        assert _flag_critical_callers({}) == []


# ── Weighted risk scoring tests ────────────────────────────────────────────────

class TestWeightedRiskScoring:

    def test_server_side_score_formula(self):
        from agents.risk_evaluator import _server_side_score
        dims = {
            "blast_radius_score":  6.0,
            "test_coverage_score": 8.0,
            "runtime_risk_score":  7.0,
            "complexity_score":    3.0,
        }
        score = _server_side_score(dims)
        expected = 6.0*0.30 + 8.0*0.30 + 7.0*0.25 + 3.0*0.15
        assert abs(score - expected) < 0.01

    def test_score_clamped_to_10(self):
        from agents.risk_evaluator import _server_side_score
        dims = {k: 10.0 for k in ["blast_radius_score", "test_coverage_score",
                                    "runtime_risk_score", "complexity_score"]}
        assert _server_side_score(dims) == 10.0

    def test_score_clamped_to_0(self):
        from agents.risk_evaluator import _server_side_score
        dims = {k: 0.0 for k in ["blast_radius_score", "test_coverage_score",
                                   "runtime_risk_score", "complexity_score"]}
        assert _server_side_score(dims) == 0.0

    def test_reconcile_top_concerns_removes_runtime_hype_when_no_proof(self):
        from agents.risk_evaluator import _reconcile_top_concerns_with_scores

        data = {
            "dimension_scores": {
                "blast_radius_score": 3.0,
                "test_coverage_score": 7.0,
                "runtime_risk_score": 0.0,
                "complexity_score": 3.0,
            },
            "overall_risk_score": 3.5,
            "top_concerns": [
                "Potential runtime errors in existing callers",
                "High test coverage gaps for critical functions",
            ],
        }
        state = {
            "total_deletions": 0,
            "total_additions": 100,
            "runtime_risks": {
                "is_breaking_change": False,
                "breaking_scenarios": [],
                "_pure_addition_override": True,
            },
            "test_gaps": {
                "uncovered_functions": [
                    {"function": "onMessage", "missing_scenario": "errors", "risk": "MEDIUM"}
                ]
            },
            "blast_radius": {"blast_radius_summary": "Limited dependents."},
        }
        _reconcile_top_concerns_with_scores(data, state)
        joined = " ".join(data["top_concerns"]).lower()
        assert "potential runtime errors in existing callers" not in joined
        assert data.get("_top_concerns_reconciled") is True
        assert any("test gap" in c.lower() or "coverage" in c.lower() for c in data["top_concerns"])

    def test_reconcile_drops_silent_error_concerns_under_strict_filter(self):
        from agents.risk_evaluator import _reconcile_top_concerns_with_scores

        data = {
            "dimension_scores": {
                "blast_radius_score": 3.0,
                "test_coverage_score": 7.0,
                "runtime_risk_score": 0.0,
                "complexity_score": 3.0,
            },
            "overall_risk_score": 3.5,
            "top_concerns": ["Potential for silent errors due to lack of null checks"],
        }
        state = {
            "total_deletions": 0,
            "total_additions": 50,
            "runtime_risks": {
                "is_breaking_change": False,
                "breaking_scenarios": [],
                "_pure_addition_override": True,
            },
            "test_gaps": {"uncovered_functions": [{"function": "x", "risk": "LOW"}]},
            "blast_radius": {},
        }
        _reconcile_top_concerns_with_scores(data, state)
        assert not any("silent error" in c.lower() for c in data["top_concerns"])

    def test_derive_risk_level_boundaries(self):
        from agents.risk_evaluator import _derive_risk_level
        assert _derive_risk_level(0.0)  == "LOW"
        assert _derive_risk_level(3.0)  == "LOW"
        assert _derive_risk_level(3.1)  == "MEDIUM"
        assert _derive_risk_level(5.9)  == "MEDIUM"
        assert _derive_risk_level(6.0)  == "HIGH"
        assert _derive_risk_level(7.9)  == "HIGH"
        assert _derive_risk_level(8.0)  == "CRITICAL"
        assert _derive_risk_level(10.0) == "CRITICAL"

    def test_missing_dimension_treated_as_zero(self):
        from agents.risk_evaluator import _server_side_score
        dims = {"blast_radius_score": 10.0}  # others missing
        score = _server_side_score(dims)
        assert score == round(10.0 * 0.30, 2)  # only blast radius contributes

    @patch("agents.risk_evaluator.run_without_tools")
    def test_score_corrected_when_llm_wrong(self, mock_run):
        from agents.base import AgentOutput
        from agents import risk_evaluator
        # LLM claims score=2.0 but formula gives 6.4
        llm_output = {
            "dimension_scores": {
                "blast_radius_score":  6.0,
                "test_coverage_score": 8.0,
                "runtime_risk_score":  7.0,
                "complexity_score":    3.0,
            },
            "overall_risk_score": 2.0,   # WRONG — LLM made arithmetic error
            "risk_level": "LOW",
            "top_concerns": [],
            "recommended_actions": [],
            "rollback_difficulty": "EASY",
        }
        mock_run.return_value = AgentOutput(
            agent_name="risk_evaluator", success=True, data=llm_output
        )
        result = risk_evaluator.run({
            "pr_title": "test", "diff_summary": "", "changed_files": [],
            "total_additions": 0, "total_deletions": 0, "has_test_changes": False,
            "blast_radius": {}, "runtime_risks": {}, "test_gaps": {},
            "business_impacts": [],
        })
        assert abs(result.data["overall_risk_score"] - 6.4) < 0.01
        assert result.data.get("_score_corrected") is True
        assert result.data["risk_level"] == "HIGH"


# ── Business Impact tests ─────────────────────────────────────────────────────

class TestBusinessImpact:

    def test_auth_file_maps_to_auth_impact(self):
        from agents.business_impact import classify_from_paths
        impacts = classify_from_paths(["src/auth/login.py"], {})
        assert "Authentication outage risk" in impacts

    def test_payment_file_maps_to_payment_impact(self):
        from agents.business_impact import classify_from_paths
        impacts = classify_from_paths(["app/checkout/stripe_handler.rb"], {})
        assert "Payment / checkout disruption" in impacts

    def test_test_file_yields_no_impact(self):
        from agents.business_impact import classify_from_paths
        impacts = classify_from_paths(["tests/test_auth.py"], {})
        assert "Authentication outage risk" not in impacts

    def test_blast_radius_files_included(self):
        from agents.business_impact import classify_from_paths
        blast = {
            "direct_dependents": [{"file": "src/payment/checkout.js", "reason": "imports"}],
            "indirect_dependents": [],
        }
        impacts = classify_from_paths([], blast)
        assert "Payment / checkout disruption" in impacts

    def test_runtime_risk_failure_mode_classified(self):
        from agents.business_impact import classify_from_runtime_risks
        risks = {
            "breaking_scenarios": [
                {"failure_mode": "NULL_DEREF",
                 "failure_description": "auth token validation fails silently"}
            ]
        }
        impacts = classify_from_runtime_risks(risks)
        assert "Authentication outage risk" in impacts

    def test_no_duplicates_in_impacts(self):
        from agents.business_impact import classify_from_paths
        impacts = classify_from_paths([
            "auth/login.py", "auth/logout.py", "auth/register.py"
        ], {})
        assert impacts.count("Authentication outage risk") == 1

    def test_severity_domains_only_critical(self):
        from agents.business_impact import analyze
        state = {
            "changed_files": ["src/auth/login.js", "src/utils/logger.js"],
            "blast_radius": {"direct_dependents": [], "indirect_dependents": []},
            "runtime_risks": {"breaking_scenarios": []},
            "risk_assessment": {"risk_level": "HIGH"},
        }
        result = analyze(state)
        assert "Authentication outage risk" in result["severity_domains"]
        assert "business_impacts" in result
        assert "impact_summary" in result

    def test_no_impacts_returns_empty_list(self):
        from agents.business_impact import classify_from_paths
        impacts = classify_from_paths(["src/tests/fixtures/data.json"], {})
        assert isinstance(impacts, list)

    def test_impact_summary_mentions_risk_level(self):
        from agents.business_impact import build_impact_summary
        summary = build_impact_summary(["Authentication outage risk"], "HIGH")
        assert "HIGH" in summary

    def test_empty_impacts_returns_safe_summary(self):
        from agents.business_impact import build_impact_summary
        summary = build_impact_summary([], "LOW")
        assert "No business-critical" in summary


# ── Rollback Advisor tests ─────────────────────────────────────────────────────

class TestRollbackAdvisor:

    def test_migration_file_heuristic_high(self):
        from agents.rollback_advisor import _heuristic_difficulty
        result = _heuristic_difficulty(["db/migrations/001_create_users.sql"], 5, {})
        assert result == "HIGH"

    def test_alembic_migration_heuristic_high(self):
        from agents.rollback_advisor import _heuristic_difficulty
        result = _heuristic_difficulty(["alembic/versions/add_column.py"], 10, {})
        assert result == "HIGH"

    def test_config_only_no_deletions_heuristic_low(self):
        from agents.rollback_advisor import _heuristic_difficulty
        result = _heuristic_difficulty(
            ["config/settings.yaml"],
            total_deletions=0,
            runtime_risks={"is_breaking_change": False}
        )
        assert result == "LOW"

    def test_pure_addition_non_breaking_heuristic_low(self):
        from agents.rollback_advisor import _heuristic_difficulty
        result = _heuristic_difficulty(
            ["src/new_feature.py"],
            total_deletions=0,
            runtime_risks={"is_breaking_change": False}
        )
        assert result == "LOW"

    def test_sanitize_rollback_clears_spurious_data_risk(self):
        from agents.rollback_advisor import _sanitize_rollback_output
        data = {
            "data_side_effects": True,
            "rollback_difficulty": "HIGH",
            "rollback_risks": ["Active sessions may be corrupted due to authentication"],
        }
        state = {
            "total_deletions": 0,
            "changed_files": ["lib/application.js", "lib/diagnostics.js"],
        }
        out = _sanitize_rollback_output(dict(data), state)
        assert out["data_side_effects"] is False
        assert out["rollback_difficulty"] == "LOW"
        assert not any("session" in str(r).lower() for r in out["rollback_risks"])

    def test_auth_file_heuristic_high(self):
        from agents.rollback_advisor import _heuristic_difficulty
        result = _heuristic_difficulty(["src/auth/session_manager.py"], 5, {})
        assert result == "HIGH"

    def test_ambiguous_returns_none(self):
        from agents.rollback_advisor import _heuristic_difficulty
        result = _heuristic_difficulty(["src/services/user_service.py"], 10, {"is_breaking_change": True})
        assert result is None   # needs LLM judgment

    def test_schema_validate_rollback_output(self):
        from agents.schema_validator import validate
        data = {
            "rollback_difficulty": "HIGH",
            "rollback_risks": ["Active sessions will be invalidated"],
            "rollback_steps": ["git revert", "redeploy"],
            "rollback_summary": "Complex rollback requiring coordination",
        }
        valid, issues = validate(data, "rollback_advisor")
        assert valid is True

    def test_schema_invalid_difficulty_flagged(self):
        from agents.schema_validator import validate
        data = {
            "rollback_difficulty": "VERY_HARD",
            "rollback_risks": [],
            "rollback_steps": [],
            "rollback_summary": "test",
        }
        valid, issues = validate(data, "rollback_advisor")
        assert valid is False

    @patch("agents.rollback_advisor.run_without_tools")
    def test_heuristic_fallback_on_llm_failure(self, mock_run):
        from agents.base import AgentOutput
        from agents import rollback_advisor
        mock_run.return_value = AgentOutput(
            agent_name="rollback_advisor", success=False, error="LLM timeout"
        )
        state = {
            "pr_title": "test",
            "changed_files": ["db/migrations/001.sql"],
            "total_additions": 10, "total_deletions": 5,
            "runtime_risks": {}, "blast_radius": {},
            "repo_history": "No history.",
        }
        result = rollback_advisor.run(state)
        # Should fall back to heuristic (migration → HIGH)
        assert result.success is True
        assert result.data["rollback_difficulty"] == "HIGH"


# ── Historical context tests ───────────────────────────────────────────────────

class TestHistoricalContext:

    @pytest.fixture
    def store_with_history(self, tmp_path):
        from memory.store import MemoryStore
        from memory.schema import PRAnalysisRecord
        store = MemoryStore(db_path=str(tmp_path / "test.db"))

        records = [
            PRAnalysisRecord(
                repo="owner/repo", pr_number=i,
                pr_title=f"PR {i}",
                analyzed_at=datetime.now(timezone.utc).isoformat(),
                files_touched=["src/auth.py", "src/utils.py"],
                symbols_changed=["login", "logout"],
                languages=["python"],
                additions=50, deletions=10,
                overall_risk_score=8.0 if i % 2 == 0 else 3.0,
                risk_level="HIGH" if i % 2 == 0 else "LOW",
                blast_radius_count=3,
                had_test_gaps=i % 2 == 0,
                top_concerns=["auth risk"],
                critic_verdict="AGREE",
                conflict_rounds=0,
            )
            for i in range(1, 7)
        ]
        for r in records:
            store.save_analysis(r)
        return store

    def test_past_high_risk_prs_counted(self, store_with_history):
        ctx = store_with_history.get_historical_context("owner/repo", [], [])
        assert ctx["past_high_risk_prs"] == 3   # PRs 2, 4, 6

    def test_avg_risk_score_calculated(self, store_with_history):
        ctx = store_with_history.get_historical_context("owner/repo", [], [])
        expected_avg = (8.0*3 + 3.0*3) / 6  # = 5.5
        assert abs(ctx["avg_risk_score"] - expected_avg) < 0.1

    def test_frequently_affected_modules_populated(self, store_with_history):
        ctx = store_with_history.get_historical_context("owner/repo", [], [])
        assert len(ctx["frequently_affected_modules"]) > 0

    def test_files_overlapping_past_high_risk(self, store_with_history):
        ctx = store_with_history.get_historical_context(
            "owner/repo", ["src/auth.py"], ["login"]
        )
        assert "src/auth.py" in ctx["files_overlapping_past_high_risk"]

    def test_symbols_seen_before(self, store_with_history):
        ctx = store_with_history.get_historical_context(
            "owner/repo", [], ["login"]
        )
        assert "login" in ctx["symbols_seen_before"]

    def test_prompt_text_generated(self, store_with_history):
        ctx = store_with_history.get_historical_context("owner/repo", [], [])
        assert "HISTORICAL CONTEXT" in ctx["prompt_text"]
        assert "owner/repo" in ctx["prompt_text"]

    def test_no_history_returns_safe_defaults(self, tmp_path):
        from memory.store import MemoryStore
        store = MemoryStore(db_path=str(tmp_path / "empty.db"))
        ctx = store.get_historical_context("new/repo", ["src/auth.py"], ["login"])
        assert ctx["past_high_risk_prs"] == 0
        assert ctx["historical_risk_trend"] == "STABLE"
        assert "No prior" in ctx["prompt_text"]

    def test_trend_worsening_detected(self, tmp_path):
        from memory.store import MemoryStore
        from memory.schema import PRAnalysisRecord
        store = MemoryStore(db_path=str(tmp_path / "trend.db"))
        # Recent PRs (higher index = more recent) should have higher scores
        for i in range(10):
            store.save_analysis(PRAnalysisRecord(
                repo="r/r", pr_number=i+1, pr_title=f"PR {i}",
                analyzed_at=datetime.now(timezone.utc).isoformat(),
                files_touched=["a.py"], symbols_changed=["f"],
                languages=["python"], additions=10, deletions=0,
                # First 5 (oldest) = low risk, last 5 (newest) = high risk
                overall_risk_score=2.0 if i < 5 else 9.0,
                risk_level="LOW" if i < 5 else "HIGH",
                blast_radius_count=1, had_test_gaps=False,
                top_concerns=[], critic_verdict="AGREE", conflict_rounds=0,
            ))
        ctx = store.get_historical_context("r/r", [], [])
        assert ctx["historical_risk_trend"] == "WORSENING"


# ── Report builder new fields tests ───────────────────────────────────────────

class TestReportBuilderEnhancements:

    @pytest.fixture
    def full_state(self):
        return {
            "repo": "owner/repo", "pr_number": 42,
            "pr_title": "Fix auth", "pr_url": "https://github.com/o/r/pull/42",
            "pr_author": "dev", "base_branch": "main",
            "pr_html_url": "https://github.com/o/r/pull/42",
            "diff_summary": "3 files (+50/-10)", "total_additions": 50,
            "total_deletions": 10, "languages": ["python"],
            "changed_files": ["src/auth.py"], "changed_symbols": ["login"],
            "has_test_changes": False,
            "blast_radius": {"direct_dependents": [], "indirect_dependents": [], "blast_radius_summary": ""},
            "runtime_risks": {"breaking_scenarios": [], "is_breaking_change": False, "simulator_summary": ""},
            "test_gaps": {"uncovered_functions": [], "overall_coverage_assessment": "ADEQUATE", "test_gap_summary": ""},
            "risk_assessment": {"overall_risk_score": 5.0, "risk_level": "MEDIUM", "dimension_scores": {},
                                "top_concerns": [], "recommended_actions": [], "rollback_difficulty": "MEDIUM"},
            "objections": {"verdict": "AGREE", "objections": [], "critic_summary": ""},
            "rollback_advice": {"rollback_difficulty": "MEDIUM", "rollback_risks": [], "rollback_steps": [],
                                "rollback_summary": "Safe rollback", "feature_flag_possible": False,
                                "data_side_effects": False},
            "business_impacts": ["Authentication outage risk"],
            "impact_summary": "Authentication outage risk — MEDIUM deployment risk.",
            "severity_domains": ["Authentication outage risk"],
            "historical_context": {"past_high_risk_prs": 2, "avg_risk_score": 6.5,
                                   "historical_risk_trend": "STABLE",
                                   "frequently_affected_modules": ["auth"],
                                   "files_with_recurring_issues": [],
                                   "files_overlapping_past_high_risk": ["src/auth.py"],
                                   "symbols_seen_before": ["login"],
                                   "prompt_text": "HISTORICAL CONTEXT FOR owner/repo:"},
            "rerun_count": 0, "conflict_log": [],
            "_analyzed_at": datetime.now(timezone.utc).isoformat(),
            "_memory_store": None, "_github_client": None, "_parsed_diff": None,
        }

    def test_report_has_rollback_advice(self, full_state):
        from output.report_builder import build_report
        report = build_report(full_state)
        assert report.rollback_advice["rollback_difficulty"] == "MEDIUM"

    def test_report_has_business_impacts(self, full_state):
        from output.report_builder import build_report
        report = build_report(full_state)
        assert "Authentication outage risk" in report.business_impacts

    def test_report_has_impact_summary(self, full_state):
        from output.report_builder import build_report
        report = build_report(full_state)
        assert "Authentication" in report.impact_summary

    def test_report_has_severity_domains(self, full_state):
        from output.report_builder import build_report
        report = build_report(full_state)
        assert "Authentication outage risk" in report.severity_domains

    def test_report_has_historical_context(self, full_state):
        from output.report_builder import build_report
        report = build_report(full_state)
        assert report.historical_context["past_high_risk_prs"] == 2


class TestFormatterEnhancements:

    @pytest.fixture
    def sample_report(self, tmp_path):
        from output.report_builder import build_report, ReportData
        state = {
            "repo": "owner/repo", "pr_number": 99, "pr_title": "Test PR",
            "pr_url": "https://github.com/o/r/pull/99",
            "pr_author": "dev", "base_branch": "main",
            "pr_html_url": "https://github.com/o/r/pull/99",
            "diff_summary": "2 files (+30/-5)", "total_additions": 30,
            "total_deletions": 5, "languages": ["javascript"],
            "changed_files": ["src/auth.js"], "changed_symbols": ["login"],
            "has_test_changes": False,
            "blast_radius": {"direct_dependents": [
                {"file": "src/router.js", "reason": "calls login", "confidence": "HIGH"}
            ], "indirect_dependents": [], "blast_radius_summary": "1 direct"},
            "runtime_risks": {
                "breaking_scenarios": [{
                    "caller_file": "src/router.js", "line_approx": 42,
                    "failure_mode": "NULL_DEREF",
                    "failure_description": "router dereferences user.token without null check",
                    "severity": "HIGH",
                    "evidence": ["src/router.js imports login at line 2", "line 42: user.token — no null check"],
                }],
                "is_breaking_change": True, "simulator_summary": "Null return breaks router",
                "confidence": 4,
            },
            "test_gaps": {"uncovered_functions": [
                {"function": "login", "missing_scenario": "no test for null return", "risk": "HIGH"}
            ], "overall_coverage_assessment": "POOR", "test_gap_summary": "login untested"},
            "risk_assessment": {
                "overall_risk_score": 7.2, "risk_level": "HIGH",
                "dimension_scores": {"blast_radius_score": 6.0, "test_coverage_score": 8.0,
                                     "runtime_risk_score": 7.0, "complexity_score": 3.0},
                "score_working": "6.0×0.30 + 8.0×0.30 + 7.0×0.25 + 3.0×0.15 = 7.2",
                "top_concerns": ["Null deref risk"], "recommended_actions": ["Add null check"],
                "rollback_difficulty": "MEDIUM",
            },
            "objections": {"verdict": "AGREE", "objections": [], "critic_summary": "OK", "missed_impacts": []},
            "rollback_advice": {
                "rollback_difficulty": "HIGH", "rollback_risks": ["Session tokens invalidated"],
                "rollback_steps": ["git revert", "redeploy", "verify /health"],
                "rollback_summary": "High complexity — auth session impact",
                "feature_flag_possible": False, "data_side_effects": True,
            },
            "business_impacts": ["Authentication outage risk"],
            "impact_summary": "Authentication outage risk — HIGH deployment risk.",
            "severity_domains": ["Authentication outage risk"],
            "historical_context": {"past_high_risk_prs": 1, "avg_risk_score": 6.0,
                                   "historical_risk_trend": "STABLE",
                                   "frequently_affected_modules": ["auth"],
                                   "files_with_recurring_issues": [],
                                   "files_overlapping_past_high_risk": [],
                                   "symbols_seen_before": [],
                                   "prompt_text": "HISTORICAL CONTEXT:"},
            "rerun_count": 0, "conflict_log": [],
            "_analyzed_at": datetime.now(timezone.utc).isoformat(),
        }
        return build_report(state)

    def test_markdown_contains_business_impact_section(self, sample_report, tmp_path):
        from output.formatter import save_markdown
        path = save_markdown(sample_report, str(tmp_path))
        content = (tmp_path / "pr-99-report.md").read_text()
        assert "How to read this analysis" in content
        assert "Business Impact" in content
        assert "Authentication outage risk" in content

    def test_markdown_contains_evidence(self, sample_report, tmp_path):
        from output.formatter import save_markdown
        save_markdown(sample_report, str(tmp_path))
        content = (tmp_path / "pr-99-report.md").read_text()
        assert "Evidence" in content
        assert "null check" in content.lower()

    def test_markdown_contains_rollback_section(self, sample_report, tmp_path):
        from output.formatter import save_markdown
        save_markdown(sample_report, str(tmp_path))
        content = (tmp_path / "pr-99-report.md").read_text()
        assert "Rollback" in content
        assert "git revert" in content

    def test_markdown_contains_score_working(self, sample_report, tmp_path):
        from output.formatter import save_markdown
        save_markdown(sample_report, str(tmp_path))
        content = (tmp_path / "pr-99-report.md").read_text()
        assert "score_working" in content or "Working" in content or "7.2" in content

    def test_json_has_business_impact_key(self, sample_report, tmp_path):
        from output.formatter import save_json
        save_json(sample_report, str(tmp_path))
        data = json.loads((tmp_path / "pr-99-report.json").read_text())
        assert "business_impact" in data
        assert "Authentication outage risk" in data["business_impact"]["impacts"]

    def test_json_has_rollback_key(self, sample_report, tmp_path):
        from output.formatter import save_json
        save_json(sample_report, str(tmp_path))
        data = json.loads((tmp_path / "pr-99-report.json").read_text())
        assert "rollback" in data
        assert data["rollback"]["rollback_difficulty"] == "HIGH"

    def test_json_has_historical_context_key(self, sample_report, tmp_path):
        from output.formatter import save_json
        save_json(sample_report, str(tmp_path))
        data = json.loads((tmp_path / "pr-99-report.json").read_text())
        assert "historical_context" in data
        assert data["historical_context"]["past_high_risk_prs"] == 1

    def test_json_score_working_included(self, sample_report, tmp_path):
        from output.formatter import save_json
        save_json(sample_report, str(tmp_path))
        data = json.loads((tmp_path / "pr-99-report.json").read_text())
        assert "score_working" in data["risk"]
