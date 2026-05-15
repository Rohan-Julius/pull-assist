"""Tests for conflict-loop adjudication helpers."""

from agents.orchestrator_patch import (
    objection_confuses_test_coverage_with_runtime,
    compute_adjudication_summary,
    objection_claims_empty_breaking_but_evidence_exists,
)


def test_objection_flags_test_gap_as_runtime_inflation():
    obj = {
        "target_agent": "risk_evaluator",
        "claim": "score too low",
        "reason": "Test Gap Agent found no tests for onMessage",
        "suggested_correction": "Increase runtime_risk_score to 5.0",
    }
    rr = {"is_breaking_change": False, "breaking_scenarios": []}
    risk = {"dimension_scores": {"runtime_risk_score": 0.0}}
    assert objection_confuses_test_coverage_with_runtime(obj, rr, risk)


def test_objection_keeps_real_runtime_concern():
    obj = {
        "target_agent": "risk_evaluator",
        "claim": "runtime too low",
        "reason": "Simulator found NULL_DEREF in production caller",
        "suggested_correction": "Raise runtime_risk_score to 8.0",
    }
    rr = {"is_breaking_change": True, "breaking_scenarios": [{"failure_mode": "NULL_DEREF"}]}
    risk = {"dimension_scores": {"runtime_risk_score": 2.0}}
    assert not objection_confuses_test_coverage_with_runtime(obj, rr, risk)


def test_adjudication_summary_suppresses_note_when_filtered():
    log = [
        {
            "verdict": "SIGNIFICANT_ISSUES",
            "critic_summary": "x",
            "objections": [
                {
                    "target_agent": "risk_evaluator",
                    "claim": "low",
                    "reason": "Test gap for foo",
                    "suggested_correction": "Increase runtime_risk_score",
                }
            ],
            "_score_snapshot": 3.0,
        }
    ]
    rr = {"is_breaking_change": False, "breaking_scenarios": []}
    risk = {"overall_risk_score": 3.0, "dimension_scores": {"runtime_risk_score": 0.0}}
    text = compute_adjudication_summary(log, risk, 1, runtime_risks=rr, risk_assessment=risk)
    assert "Scoring contract" in text
    assert "Increase runtime_risk_score" not in text


def test_adjudication_first_pass_agree_vs_minor():
    risk = {"overall_risk_score": 4.5}
    t_agree = compute_adjudication_summary([], risk, 0, verdict="AGREE")
    assert "consistent" in t_agree.lower()
    t_minor = compute_adjudication_summary([], risk, 0, verdict="MINOR_ISSUES")
    assert "MINOR_ISSUES" in t_minor
    assert "consistent" not in t_minor.lower()


def test_objection_empty_breaking_filtered():
    obj = {
        "claim": "score low",
        "reason": "especially given the empty breaking_scenarios",
        "suggested_correction": "raise score",
    }
    rr = {"breaking_scenarios": [{"failure_mode": "SILENT_WRONG"}]}
    assert objection_claims_empty_breaking_but_evidence_exists(obj, rr)
