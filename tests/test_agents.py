"""
Day 3 — Full agent test suite. No real LLM or GitHub token needed.
Run: python -m pytest tests/test_agents.py -v
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

def _make_state(**overrides):
    base = {
        "repo": "expressjs/express", "pr_number": 7171,
        "pr_title": "feat: add diagnostic channels for request lifecycle",
        "pr_url": "https://github.com/expressjs/express/pull/7171",
        "pr_author": "OussemaNehdi", "base_branch": "master",
        "pr_html_url": "https://github.com/expressjs/express/pull/7171",
        "diff_summary": "3 files changed (+287/-0), languages: javascript",
        "total_additions": 287, "total_deletions": 0,
        "languages": ["javascript"],
        "changed_files": ["lib/application.js", "lib/request.js", "test/app.diagnostic.js"],
        "source_files": ["lib/application.js", "lib/request.js"],
        "test_files": ["test/app.diagnostic.js"],
        "changed_symbols": ["onMessage", "onStart", "onFinish"],
        "has_test_changes": True,
        "per_file_context": [{"path": "lib/application.js", "language": "javascript",
             "change_type": "modified", "additions": 140, "deletions": 0,
             "symbols": ["onMessage", "onStart"], "is_test": False}],
        "raw_diff": "diff --git a/lib/application.js b/lib/application.js\n+function onMessage() {}",
        "repo_history": "No prior analysis history for this repository.",
        "blast_radius": {}, "runtime_risks": {}, "test_gaps": {},
        "risk_assessment": {}, "objections": {},
        "rerun_count": 0, "conflict_log": [],
        "_github_client": MagicMock(), "_memory_store": MagicMock(),
        "_parsed_diff": MagicMock(), "_analyzed_at": datetime.now(timezone.utc).isoformat(),
    }
    base.update(overrides)
    return base

def _mock_llm(content):
    llm = MagicMock()
    resp = MagicMock()
    resp.content = content
    llm.invoke.return_value = resp
    llm.bind_tools.return_value = llm
    return llm

BLAST = json.dumps({
    "direct_dependents": [
        {"file": "lib/router/index.js", "reason": "imports onMessage", "confidence": "HIGH"},
        {"file": "lib/response.js", "reason": "calls onFinish on send()", "confidence": "HIGH"},
    ],
    "indirect_dependents": [
        {"file": "lib/express.js", "reason": "re-exports router", "confidence": "MEDIUM"},
    ],
    "blast_radius_summary": "2 direct, 1 indirect",
})
RUNTIME = json.dumps({
    "before_behavior": "no diagnostic channels",
    "after_behavior": "emits onMessage/onStart/onFinish events",
    "breaking_scenarios": [], "is_breaking_change": False,
    "simulator_summary": "Purely additive change.",
})
GAPS = json.dumps({
    "covered_functions": ["onStart", "onFinish"],
    "uncovered_functions": [{"function": "onMessage",
        "missing_scenario": "listener throws an error", "risk": "MEDIUM"}],
    "overall_coverage_assessment": "PARTIAL",
    "test_gap_summary": "onMessage error-path missing",
})
RISK = json.dumps({
    "dimension_scores": {"blast_radius_score": 2.5, "test_coverage_score": 4.0,
                         "runtime_risk_score": 1.5, "complexity_score": 3.0},
    "overall_risk_score": 2.8, "risk_level": "LOW",
    "top_concerns": ["onMessage error path untested", "2 production files depend on new symbols"],
    "recommended_actions": ["Add test for onMessage listener error"],
    "rollback_difficulty": "EASY",
})
CRITIC_AGREE = json.dumps({
    "objections": [], "missed_impacts": [], "verdict": "AGREE",
    "critic_summary": "All findings consistent.",
})
CRITIC_SIG = json.dumps({
    "objections": [{"target_agent": "risk_evaluator",
        "claim": "score of 2.8 too low",
        "reason": "2 production files in core routing affected",
        "suggested_correction": "blast_radius_score should be 5.0+"}],
    "missed_impacts": ["lib/express.js re-exports router"],
    "verdict": "SIGNIFICANT_ISSUES",
    "critic_summary": "Risk score understates blast radius.",
})
CRITIC_MINOR = json.dumps({
    "objections": [{"target_agent": "test_gap", "claim": "PARTIAL too lenient",
        "reason": "onMessage in hot path", "suggested_correction": "flag as POOR"}],
    "missed_impacts": [], "verdict": "MINOR_ISSUES",
    "critic_summary": "Minor coverage concern.",
})

# ── 1. parse_json_output ──────────────────────────────────────────────────────
class TestParseJsonOutput:
    def test_clean_json(self):
        from agents.base import parse_json_output
        assert parse_json_output('{"k": "v"}', "t")["k"] == "v"
    def test_markdown_fence(self):
        from agents.base import parse_json_output
        assert parse_json_output('```json\n{"s": 7.5}\n```', "t")["s"] == 7.5
    def test_preamble_prose(self):
        from agents.base import parse_json_output
        assert parse_json_output('Analysis:\n{"verdict": "HIGH"}\nEnd.', "t")["verdict"] == "HIGH"
    def test_malformed_returns_parse_error(self):
        from agents.base import parse_json_output
        assert parse_json_output("not json", "t").get("_parse_error") is True
    def test_empty_string_parse_error(self):
        from agents.base import parse_json_output
        assert parse_json_output("", "t").get("_parse_error") is True
    def test_nested_json(self):
        from agents.base import parse_json_output
        r = parse_json_output('{"s": {"a": 1}, "l": [1,2]}', "t")
        assert r["s"]["a"] == 1 and r["l"] == [1, 2]
    def test_raw_preserved_on_error(self):
        from agents.base import parse_json_output
        r = parse_json_output("unparseable", "my_agent")
        assert r.get("_agent") == "my_agent" and "unparseable" in r.get("_raw", "")
    def test_float_preserved(self):
        from agents.base import parse_json_output
        assert parse_json_output('{"score": 7.25}', "t")["score"] == 7.25
    def test_bool_preserved(self):
        from agents.base import parse_json_output
        r = parse_json_output('{"a": false, "b": true}', "t")
        assert r["a"] is False and r["b"] is True
    def test_double_fence_still_parsed(self):
        from agents.base import parse_json_output
        assert parse_json_output('```json\n```json\n{"k":"v"}\n```\n```', "t").get("k") == "v"

# ── 2. context_budget ─────────────────────────────────────────────────────────
class TestContextBudget:
    def test_diff_short_unchanged(self):
        from tools.context_budget import budget_diff
        s = "\n".join(["line"]*10)
        assert budget_diff(s, max_lines=50) == s
    def test_diff_long_truncated(self):
        from tools.context_budget import budget_diff
        r = budget_diff("\n".join(["x"]*600), max_lines=100)
        assert len(r.splitlines()) <= 102 and "TRUNCATED" in r
    def test_diff_exact_limit_no_truncation(self):
        from tools.context_budget import budget_diff
        assert "TRUNCATED" not in budget_diff("\n".join(["x"]*100), max_lines=100)
    def test_file_short_unchanged(self):
        from tools.context_budget import budget_file
        s = "\n".join(["x"]*50)
        assert budget_file(s, max_lines=200) == s
    def test_file_long_truncated(self):
        from tools.context_budget import budget_file
        assert "TRUNCATED" in budget_file("\n".join(["x"]*300), max_lines=200)
    def test_history_short_unchanged(self):
        from tools.context_budget import budget_history
        assert budget_history("short") == "short"
    def test_history_long_truncated(self):
        from tools.context_budget import budget_history
        assert len(budget_history("x"*2000)) <= 850
    def test_format_symbols_filepath(self):
        from tools.context_budget import format_symbols_for_prompt
        r = format_symbols_for_prompt(["onMessage"], [{"path":"lib/app.js","language":"js","symbols":["onMessage"]}])
        assert "lib/app.js" in r and "onMessage" in r
    def test_format_symbols_fallback(self):
        from tools.context_budget import format_symbols_for_prompt
        assert "funcA" in format_symbols_for_prompt(["funcA"], [])
    def test_per_file_diff_empty(self):
        from tools.context_budget import budget_per_file_diff
        assert "No diff content" in budget_per_file_diff([])
    def test_per_file_diff_with_file(self):
        from tools.context_budget import budget_per_file_diff
        f = MagicMock()
        f.path, f.language, f.additions, f.deletions = "lib/a.js", "js", 5, 0
        f.raw_diff = "+line 1\n+line 2"
        assert "lib/a.js" in budget_per_file_diff([f])

# ── 3. Risk Evaluator ─────────────────────────────────────────────────────────
class TestRiskEvaluator:
    @patch("agents.base.get_llm")
    def test_success(self, m):
        m.return_value = _mock_llm(RISK)
        from agents.risk_evaluator import run
        r = run(_make_state())
        assert r.agent_name == "risk_evaluator" and r.success
    @patch("agents.base.get_llm")
    def test_overall_score_present(self, m):
        m.return_value = _mock_llm(RISK)
        from agents.risk_evaluator import run
        assert "overall_risk_score" in run(_make_state()).data
    @patch("agents.base.get_llm")
    def test_risk_level_valid(self, m):
        m.return_value = _mock_llm(RISK)
        from agents.risk_evaluator import run
        assert run(_make_state()).data.get("risk_level") in {"LOW","MEDIUM","HIGH","CRITICAL"}
    @patch("agents.base.get_llm")
    def test_all_dimension_scores(self, m):
        m.return_value = _mock_llm(RISK)
        from agents.risk_evaluator import run
        dims = run(_make_state()).data.get("dimension_scores", {})
        for k in ("blast_radius_score","test_coverage_score","runtime_risk_score","complexity_score"):
            assert k in dims
    @patch("agents.base.get_llm")
    def test_rollback_difficulty_present(self, m):
        m.return_value = _mock_llm(RISK)
        from agents.risk_evaluator import run
        assert "rollback_difficulty" in run(_make_state()).data
    @patch("agents.base.get_llm")
    def test_recommended_actions_list(self, m):
        m.return_value = _mock_llm(RISK)
        from agents.risk_evaluator import run
        assert isinstance(run(_make_state()).data.get("recommended_actions"), list)
    @patch("agents.base.get_llm")
    def test_llm_error_returns_failed(self, m):
        lm = MagicMock(); lm.invoke.side_effect = RuntimeError("LLM connection refused")
        m.return_value = lm
        from agents.risk_evaluator import run
        r = run(_make_state())
        assert not r.success and "connection refused" in r.error.lower()
    @patch("agents.base.get_llm")
    def test_malformed_response_no_raise(self, m):
        m.return_value = _mock_llm("I cannot score this.")
        from agents.risk_evaluator import run
        assert run(_make_state()).agent_name == "risk_evaluator"

# ── 4. Critic Agent ───────────────────────────────────────────────────────────
class TestCriticAgent:
    @patch("agents.base.get_llm")
    def test_agree(self, m):
        m.return_value = _mock_llm(CRITIC_AGREE)
        from agents.critic import run
        r = run(_make_state(blast_radius=json.loads(BLAST), runtime_risks=json.loads(RUNTIME),
                            test_gaps=json.loads(GAPS), risk_assessment=json.loads(RISK)))
        assert r.success and r.data.get("verdict") == "AGREE"
    @patch("agents.base.get_llm")
    def test_significant_issues(self, m):
        m.return_value = _mock_llm(CRITIC_SIG)
        from agents.critic import run
        assert run(_make_state()).data.get("verdict") == "SIGNIFICANT_ISSUES"
    @patch("agents.base.get_llm")
    def test_minor_issues(self, m):
        m.return_value = _mock_llm(CRITIC_MINOR)
        from agents.critic import run
        assert run(_make_state()).data.get("verdict") == "MINOR_ISSUES"
    @patch("agents.base.get_llm")
    def test_objections_is_list(self, m):
        m.return_value = _mock_llm(CRITIC_SIG)
        from agents.critic import run
        assert isinstance(run(_make_state()).data.get("objections"), list)
    @patch("agents.base.get_llm")
    def test_objection_required_fields(self, m):
        m.return_value = _mock_llm(CRITIC_SIG)
        from agents.critic import run
        obj = run(_make_state()).data["objections"][0]
        for f in ("target_agent","claim","reason","suggested_correction"):
            assert f in obj
    @patch("agents.base.get_llm")
    def test_agree_empty_objections(self, m):
        m.return_value = _mock_llm(CRITIC_AGREE)
        from agents.critic import run
        assert run(_make_state()).data.get("objections") == []
    @patch("agents.base.get_llm")
    def test_critic_summary_present(self, m):
        m.return_value = _mock_llm(CRITIC_AGREE)
        from agents.critic import run
        r = run(_make_state())
        assert len(r.data.get("critic_summary","")) > 0
    def test_rerun_context_has_claim(self):
        from agents.critic import build_rerun_context
        ctx = build_rerun_context({"overall_risk_score": 2.8},
                                  {"claim": "score too low", "reason": "blast radius undercounted", "suggested_correction": "raise to 5.0"})
        assert "score too low" in ctx and "undercounted" in ctx
    def test_rerun_context_has_original_output(self):
        from agents.critic import build_rerun_context
        ctx = build_rerun_context({"overall_risk_score": 2.8},
                                  {"claim":"","reason":"","suggested_correction":""})
        assert "2.8" in ctx
    def test_rerun_context_has_correction(self):
        from agents.critic import build_rerun_context
        ctx = build_rerun_context({}, {"claim":"c","reason":"r","suggested_correction":"raise to 6.0"})
        assert "6.0" in ctx

# ── 5. Orchestrator routing ───────────────────────────────────────────────────
class TestOrchestratorGraph:
    def test_compiles(self):
        from agents.orchestrator import build_graph
        assert build_graph() is not None
    def test_all_nodes(self):
        from agents.orchestrator import build_graph
        nodes = set(build_graph().nodes)
        for n in ("dependency_mapper","change_simulator","test_gap",
                  "rollback_advisor","risk_evaluator_with_business","critic","rerun_with_objections"):
            assert n in nodes
    def test_agree_done(self):
        from agents.orchestrator import should_rerun
        assert should_rerun(_make_state(objections={"verdict":"AGREE"}, rerun_count=0)) == "done"
    def test_sig_round0_rerun(self):
        from agents.orchestrator import should_rerun
        assert should_rerun(_make_state(objections={"verdict":"SIGNIFICANT_ISSUES"}, rerun_count=0)) == "rerun"
    def test_sig_round1_rerun(self):
        from agents.orchestrator import should_rerun
        assert should_rerun(_make_state(objections={"verdict":"SIGNIFICANT_ISSUES"}, rerun_count=1)) == "rerun"
    def test_sig_round2_done(self):
        from agents.orchestrator import should_rerun
        assert should_rerun(_make_state(objections={"verdict":"SIGNIFICANT_ISSUES"}, rerun_count=2)) == "done"
    def test_minor_done(self):
        from agents.orchestrator import should_rerun
        assert should_rerun(_make_state(objections={"verdict":"MINOR_ISSUES"}, rerun_count=0)) == "done"
    def test_empty_verdict_done(self):
        from agents.orchestrator import should_rerun
        assert should_rerun(_make_state(objections={}, rerun_count=0)) == "done"
    def test_rerun_table(self):
        from agents.orchestrator import should_rerun
        for count, exp in [(0,"rerun"),(1,"rerun"),(2,"done"),(5,"done")]:
            s = _make_state(objections={"verdict":"SIGNIFICANT_ISSUES"}, rerun_count=count)
            assert should_rerun(s) == exp, f"count={count}"
    def test_state_key_mapping(self):
        from agents.orchestrator import _agent_state_key
        assert _agent_state_key("dependency_mapper") == "blast_radius"
        assert _agent_state_key("change_simulator") == "runtime_risks"
        assert _agent_state_key("test_gap") == "test_gaps"
        assert _agent_state_key("risk_evaluator") == "risk_assessment"

# ── 6. Report Builder ─────────────────────────────────────────────────────────
class TestReportBuilder:
    def _state(self, **extra):
        return _make_state(blast_radius=json.loads(BLAST), runtime_risks=json.loads(RUNTIME),
                           test_gaps=json.loads(GAPS), risk_assessment=json.loads(RISK),
                           objections=json.loads(CRITIC_AGREE), **extra)
    def test_returns_report_data(self):
        from output.report_builder import build_report, ReportData
        assert isinstance(build_report(self._state()), ReportData)
    def test_risk_score(self):
        from output.report_builder import build_report
        assert build_report(self._state()).overall_risk_score == 2.8
    def test_risk_level(self):
        from output.report_builder import build_report
        assert build_report(self._state()).risk_level == "LOW"
    def test_pr_title(self):
        from output.report_builder import build_report
        assert "diagnostic" in build_report(self._state()).pr_title.lower()
    def test_pr_number(self):
        from output.report_builder import build_report
        assert build_report(self._state()).pr_number == 7171
    def test_changed_files(self):
        from output.report_builder import build_report
        assert "lib/application.js" in build_report(self._state()).changed_files
    def test_top_concerns_non_empty(self):
        from output.report_builder import build_report
        assert len(build_report(self._state()).top_concerns) > 0
    def test_defaults_when_empty(self):
        from output.report_builder import build_report
        r = build_report(_make_state())
        assert r.overall_risk_score == 0.0 and r.risk_level == "UNKNOWN"
    def test_memory_record_repo_pr(self):
        from output.report_builder import build_report, build_memory_record
        rec = build_memory_record(build_report(self._state()))
        assert rec.repo == "expressjs/express" and rec.pr_number == 7171
    def test_memory_record_risk(self):
        from output.report_builder import build_report, build_memory_record
        assert build_memory_record(build_report(self._state())).overall_risk_score == 2.8
    def test_memory_record_had_gaps_true(self):
        from output.report_builder import build_report, build_memory_record
        assert build_memory_record(build_report(self._state())).had_test_gaps is True
    def test_memory_record_had_gaps_false(self):
        from output.report_builder import build_report, build_memory_record
        s = _make_state(blast_radius=json.loads(BLAST), runtime_risks=json.loads(RUNTIME), test_gaps={"uncovered_functions":[]}, risk_assessment=json.loads(RISK), objections=json.loads(CRITIC_AGREE))
        assert build_memory_record(build_report(s)).had_test_gaps is False
    def test_memory_record_blast_count(self):
        from output.report_builder import build_report, build_memory_record
        assert build_memory_record(build_report(self._state())).blast_radius_count == 3
    def test_top_concerns_capped(self):
        from output.report_builder import build_report, build_memory_record
        s = self._state(); s["risk_assessment"]["top_concerns"] = ["a","b","c","d","e"]
        assert len(build_memory_record(build_report(s)).top_concerns) <= 3
    def test_critic_verdict_in_record(self):
        from output.report_builder import build_report, build_memory_record
        assert build_memory_record(build_report(self._state())).critic_verdict == "AGREE"
    def test_safe_with_all_empty(self):
        from output.report_builder import build_report
        assert build_report(_make_state()) is not None

# ── 7. Formatter ──────────────────────────────────────────────────────────────
class TestFormatter:
    def _report(self, **extra):
        from output.report_builder import build_report
        return build_report(_make_state(
            blast_radius=json.loads(BLAST), runtime_risks=json.loads(RUNTIME),
            test_gaps=json.loads(GAPS), risk_assessment=json.loads(RISK),
            objections=json.loads(CRITIC_AGREE), **extra))

    def test_md_file_created(self, tmp_path):
        from output.formatter import save_markdown
        r = self._report()
        save_markdown(r, output_dir=str(tmp_path))
        assert (tmp_path/f"pr-{r.pr_number}-report.md").exists()
    def test_md_risk_score(self, tmp_path):
        from output.formatter import save_markdown
        assert "2.8" in open(save_markdown(self._report(), output_dir=str(tmp_path))).read()
    def test_md_risk_level(self, tmp_path):
        from output.formatter import save_markdown
        assert "LOW" in open(save_markdown(self._report(), output_dir=str(tmp_path))).read()
    def test_md_pr_title(self, tmp_path):
        from output.formatter import save_markdown
        assert "diagnostic" in open(save_markdown(self._report(), output_dir=str(tmp_path))).read().lower()
    def test_md_blast_radius_section(self, tmp_path):
        from output.formatter import save_markdown
        assert "Blast Radius" in open(save_markdown(self._report(), output_dir=str(tmp_path))).read()
    def test_md_test_gaps_section(self, tmp_path):
        from output.formatter import save_markdown
        assert "Test Gap" in open(save_markdown(self._report(), output_dir=str(tmp_path))).read()
    def test_md_recommended_actions(self, tmp_path):
        from output.formatter import save_markdown
        assert "Recommended" in open(save_markdown(self._report(), output_dir=str(tmp_path))).read()
    def test_md_author(self, tmp_path):
        from output.formatter import save_markdown
        assert "OussemaNehdi" in open(save_markdown(self._report(), output_dir=str(tmp_path))).read()
    def test_md_critical_risk(self, tmp_path):
        from output.formatter import save_markdown
        from output.report_builder import build_report
        s = _make_state(blast_radius=json.loads(BLAST), runtime_risks=json.loads(RUNTIME),
                        test_gaps=json.loads(GAPS), objections=json.loads(CRITIC_AGREE),
                        risk_assessment={**json.loads(RISK), "overall_risk_score":8.5, "risk_level":"CRITICAL"})
        path = save_markdown(build_report(s), output_dir=str(tmp_path))
        c = open(path).read()
        assert "CRITICAL" in c and "8.5" in c
    def test_md_no_crash_empty_blast(self, tmp_path):
        from output.formatter import save_markdown
        from output.report_builder import build_report
        r = build_report(_make_state(risk_assessment=json.loads(RISK), objections=json.loads(CRITIC_AGREE)))
        assert open(save_markdown(r, output_dir=str(tmp_path))).read() != ""
    def test_json_file_created(self, tmp_path):
        from output.formatter import save_json
        r = self._report()
        save_json(r, output_dir=str(tmp_path))
        assert (tmp_path/f"pr-{r.pr_number}-report.json").exists()
    def test_json_valid(self, tmp_path):
        from output.formatter import save_json
        assert isinstance(json.load(open(save_json(self._report(), output_dir=str(tmp_path)))), dict)
    def test_json_all_sections(self, tmp_path):
        from output.formatter import save_json
        d = json.load(open(save_json(self._report(), output_dir=str(tmp_path))))
        for s in ("meta","risk","blast_radius","runtime_risks","test_gaps","critic"):
            assert s in d
    def test_json_risk_score(self, tmp_path):
        from output.formatter import save_json
        assert json.load(open(save_json(self._report(), output_dir=str(tmp_path))))["risk"]["overall_score"] == 2.8
    def test_json_meta_pr(self, tmp_path):
        from output.formatter import save_json
        m = json.load(open(save_json(self._report(), output_dir=str(tmp_path))))["meta"]
        assert m["pr_number"] == 7171 and m["repo"] == "expressjs/express"
    def test_json_critic_verdict(self, tmp_path):
        from output.formatter import save_json
        assert json.load(open(save_json(self._report(), output_dir=str(tmp_path))))["critic"]["verdict"] == "AGREE"
    def test_json_rerun_count(self, tmp_path):
        from output.formatter import save_json
        assert "rerun_count" in json.load(open(save_json(self._report(), output_dir=str(tmp_path))))["critic"]
    def test_json_uncovered_functions(self, tmp_path):
        from output.formatter import save_json
        d = json.load(open(save_json(self._report(), output_dir=str(tmp_path))))
        assert len(d["test_gaps"]["uncovered_functions"]) == 1
    def test_json_dimension_scores(self, tmp_path):
        from output.formatter import save_json
        dims = json.load(open(save_json(self._report(), output_dir=str(tmp_path))))["risk"]["dimensions"]
        assert "blast_radius_score" in dims and "test_coverage_score" in dims