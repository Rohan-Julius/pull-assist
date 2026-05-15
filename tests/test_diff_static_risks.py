"""Unit tests for diff-grounded symbol prioritization and static risk signals."""

from github.diff_parser import parse_diff
from github.diff_static_risks import (
    augment_runtime_risks_with_diff,
    augment_test_gaps_with_diff,
    detect_onfinished_finish_without_err_guard,
    diff_indicates_observability_channel_tests,
    extract_diagnostics_channel_names,
    extract_js_export_property_keys,
    prioritize_changed_symbols,
)

EXPRESS_7171_SNIPPET = r"""
diff --git a/lib/application.js b/lib/application.js
--- a/lib/application.js
+++ b/lib/application.js
@@ -174,6 +174,12 @@ app.handle = function handle(req, res, callback) {
+    onFinished(res, function (err) {
+      if (err && diagnostics.requestError.hasSubscribers) {
+        diagnostics.requestError.publish({ req: req, res: res, error: err });
+      }
+
+      if (diagnostics.requestFinish.hasSubscribers) {
+        diagnostics.requestFinish.publish({ req: req, res: res });
+      }
+    });
diff --git a/lib/diagnostics.js b/lib/diagnostics.js
new file mode 100644
--- /dev/null
+++ b/lib/diagnostics.js
@@ -0,0 +1,6 @@
+module.exports = {
+  requestStart: dc.channel('express.request.start'),
+  requestFinish: dc.channel('express.request.finish'),
+  requestError: dc.channel('express.request.error')
+};
diff --git a/test/diagnostics-channel.js b/test/diagnostics-channel.js
new file mode 100644
--- /dev/null
+++ b/test/diagnostics-channel.js
@@ -0,0 +1,8 @@
+describe('diagnostics channels', function () {
+  describe('express.request.start', function () {
+      function onMessage(message) {
+        return message;
+      }
+  });
+});
"""


def test_extract_channel_names_and_exports():
    names = extract_diagnostics_channel_names(EXPRESS_7171_SNIPPET)
    assert "express.request.start" in names
    assert "express.request.finish" in names
    assert "express.request.error" in names
    keys = extract_js_export_property_keys(EXPRESS_7171_SNIPPET)
    assert "requestStart" in keys
    assert "requestFinish" in keys
    assert "requestError" in keys


def test_prioritize_symbols_prefers_channels_over_test_callbacks():
    parsed = parse_diff(EXPRESS_7171_SNIPPET)
    syms = prioritize_changed_symbols(parsed, EXPRESS_7171_SNIPPET)
    assert syms[0].startswith("express.request.")
    assert "requestStart" in syms
    assert "onMessage" not in syms


def test_detect_onfinished_finish_bug():
    scenarios = detect_onfinished_finish_without_err_guard(EXPRESS_7171_SNIPPET)
    assert len(scenarios) == 1
    assert scenarios[0]["failure_mode"] == "SILENT_WRONG"
    assert scenarios[0]["_verified_from_diff"] is True


def test_augment_runtime_merges_static():
    base = {"breaking_scenarios": [], "is_breaking_change": False, "simulator_summary": ""}
    out = augment_runtime_risks_with_diff(base, EXPRESS_7171_SNIPPET, 0)
    assert out["is_breaking_change"] is True
    assert len(out["breaking_scenarios"]) == 1


def test_diff_indicates_observability_tests():
    assert diff_indicates_observability_channel_tests(EXPRESS_7171_SNIPPET) is True


def test_augment_test_gaps_clears_noise_callbacks():
    gaps = {
        "uncovered_functions": [
            {"function": "onMessage", "missing_scenario": "x", "risk": "MEDIUM"},
            {"function": "requestFinish", "missing_scenario": "y", "risk": "HIGH"},
        ],
        "overall_coverage_assessment": "POOR",
        "test_gap_summary": "bad",
    }
    out = augment_test_gaps_with_diff(gaps, EXPRESS_7171_SNIPPET)
    funcs = [u["function"] for u in out["uncovered_functions"]]
    assert "onMessage" not in funcs
    assert "requestFinish" in funcs
    assert out["overall_coverage_assessment"] == "PARTIAL"


def test_guarded_finish_not_flagged():
    safe = """
onFinished(res, function (err) {
  if (!err && diagnostics.requestFinish.hasSubscribers) {
    diagnostics.requestFinish.publish({ req: req, res: res });
  }
});
"""
    assert detect_onfinished_finish_without_err_guard(safe) == []


def test_reconcile_keeps_runtime_language_when_diff_verified():
    from agents.risk_evaluator import _reconcile_top_concerns_with_scores

    data = {
        "dimension_scores": {
            "blast_radius_score": 3.0,
            "test_coverage_score": 2.0,
            "runtime_risk_score": 6.0,
            "complexity_score": 3.0,
        },
        "overall_risk_score": 4.0,
        "top_concerns": ["Potential runtime errors in APM subscribers"],
    }
    state = {
        "total_deletions": 0,
        "total_additions": 100,
        "runtime_risks": {
            "is_breaking_change": True,
            "breaking_scenarios": [{"_verified_from_diff": True, "failure_mode": "SILENT_WRONG"}],
            "_pure_addition_override": True,
        },
        "test_gaps": {},
        "blast_radius": {},
    }
    _reconcile_top_concerns_with_scores(data, state)
    assert "APM" in " ".join(data["top_concerns"])
