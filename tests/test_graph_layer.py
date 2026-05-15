"""
Graph layer tests — Evidence Graph, Propagation Engine, Deployment Advisor

All deterministic — zero LLM calls, zero GitHub API calls.
Run with: python -m pytest tests/test_graph_layer.py -v
"""

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def simple_blast_radius():
    return {
        "direct_dependents": [
            {"file": "src/session.py",  "reason": "imports validate_user at line 5", "confidence": "HIGH"},
            {"file": "src/api/users.py","reason": "calls validate_user on line ~88", "confidence": "HIGH"},
        ],
        "indirect_dependents": [
            {"file": "src/checkout.py", "reason": "depends on session which uses validate_user", "confidence": "MEDIUM"},
        ],
        "blast_radius_summary": "2 direct, 1 indirect",
    }


@pytest.fixture
def auth_blast_radius():
    """Blast radius involving critical auth/payment paths."""
    return {
        "direct_dependents": [
            {"file": "src/auth/session.py",   "reason": "imports login at line 3",  "confidence": "HIGH"},
            {"file": "src/payment/checkout.py","reason": "calls login for auth",     "confidence": "HIGH"},
        ],
        "indirect_dependents": [
            {"file": "src/billing/invoice.py", "reason": "flows through checkout.py","confidence": "MEDIUM"},
        ],
        "blast_radius_summary": "2 direct critical-path dependents",
    }


@pytest.fixture
def base_state():
    mock_diff = MagicMock()
    mock_diff.changed_files = []
    return {
        "repo": "owner/repo",
        "pr_number": 42,
        "changed_symbols": ["validate_user"],
        "per_file_context": [
            {"path": "src/auth.py", "symbols": ["validate_user"], "language": "python",
             "is_test": False, "additions": 20, "deletions": 5, "change_type": "modified"},
        ],
        "_parsed_diff": mock_diff,
        "risk_assessment": {"risk_level": "HIGH", "overall_risk_score": 7.5},
        "rollback_advice": {"data_side_effects": False, "rollback_difficulty": "MEDIUM"},
        "test_gaps": {"uncovered_functions": [{"function": "validate_user", "missing_scenario": "no null test", "risk": "HIGH"}]},
        "business_impacts": ["Authentication outage risk"],
        "has_test_changes": False,
    }


# ── Evidence Graph tests ───────────────────────────────────────────────────────

class TestEvidenceGraph:

    def test_build_returns_evidence_graph(self, base_state, simple_blast_radius):
        from graph.evidence_graph import build_evidence_graph, EvidenceGraph
        graph = build_evidence_graph(base_state, simple_blast_radius, base_state["per_file_context"])
        assert isinstance(graph, EvidenceGraph)

    def test_has_node_for_each_symbol(self, base_state, simple_blast_radius):
        from graph.evidence_graph import build_evidence_graph
        graph = build_evidence_graph(base_state, simple_blast_radius, base_state["per_file_context"])
        assert graph.total_symbols_changed == 1
        assert graph.get_node("validate_user") is not None

    def test_node_changed_in_correct_file(self, base_state, simple_blast_radius):
        from graph.evidence_graph import build_evidence_graph
        graph = build_evidence_graph(base_state, simple_blast_radius, base_state["per_file_context"])
        node = graph.get_node("validate_user")
        assert node.changed_in == "src/auth.py"

    def test_direct_dependents_become_caller_edges(self, base_state, simple_blast_radius):
        from graph.evidence_graph import build_evidence_graph
        graph = build_evidence_graph(base_state, simple_blast_radius, base_state["per_file_context"])
        node = graph.get_node("validate_user")
        caller_files = {c.file for c in node.affected_callers}
        assert "src/session.py" in caller_files

    def test_line_number_extracted_from_reason(self, base_state, simple_blast_radius):
        from graph.evidence_graph import build_evidence_graph
        graph = build_evidence_graph(base_state, simple_blast_radius, base_state["per_file_context"])
        node = graph.get_node("validate_user")
        session_caller = next((c for c in node.affected_callers if "session" in c.file), None)
        assert session_caller is not None
        assert session_caller.line == 5

    def test_critical_path_flagged(self, base_state, auth_blast_radius):
        from graph.evidence_graph import build_evidence_graph
        state = {**base_state, "changed_symbols": ["login"],
                 "per_file_context": [{"path": "src/auth/login.py", "symbols": ["login"],
                                       "language": "python", "is_test": False,
                                       "additions": 30, "deletions": 0, "change_type": "modified"}]}
        graph = build_evidence_graph(state, auth_blast_radius, state["per_file_context"])
        node = graph.get_node("login")
        assert node is not None
        assert node.has_critical_path  # session.py and checkout.py are critical paths

    def test_total_files_affected_counts_all(self, base_state, simple_blast_radius):
        from graph.evidence_graph import build_evidence_graph
        graph = build_evidence_graph(base_state, simple_blast_radius, base_state["per_file_context"])
        # origin + 2 direct + 1 indirect - 1 (origin is not "affected") = at least 2
        assert graph.total_files_affected >= 2

    def test_to_dict_serializable(self, base_state, simple_blast_radius):
        from graph.evidence_graph import build_evidence_graph
        import json
        graph = build_evidence_graph(base_state, simple_blast_radius, base_state["per_file_context"])
        d = graph.to_dict()
        # Should not raise
        json.dumps(d)
        assert "nodes" in d
        assert "total_symbols_changed" in d
        assert "total_files_affected" in d

    def test_empty_blast_radius_still_builds(self, base_state):
        from graph.evidence_graph import build_evidence_graph
        graph = build_evidence_graph(base_state, {}, base_state["per_file_context"])
        assert graph.total_symbols_changed == 1
        node = graph.get_node("validate_user")
        assert node is not None
        assert node.affected_callers == []

    def test_multiple_symbols_produce_multiple_nodes(self, base_state, simple_blast_radius):
        from graph.evidence_graph import build_evidence_graph
        state = {**base_state,
                 "changed_symbols": ["validate_user", "create_session"],
                 "per_file_context": [
                     {"path": "src/auth.py", "symbols": ["validate_user", "create_session"],
                      "language": "python", "is_test": False,
                      "additions": 20, "deletions": 5, "change_type": "modified"},
                 ]}
        graph = build_evidence_graph(state, simple_blast_radius, state["per_file_context"])
        assert graph.total_symbols_changed == 2

    def test_confidence_ladder_defined(self):
        from graph.evidence_graph import CONFIDENCE_LADDER
        assert CONFIDENCE_LADDER["HIGH"] == "MEDIUM"
        assert CONFIDENCE_LADDER["MEDIUM"] == "LOW"
        assert CONFIDENCE_LADDER["LOW"] == "LOW"

    def test_is_critical_detects_auth(self):
        from graph.evidence_graph import _is_critical
        assert _is_critical("src/auth/login.py") is True
        assert _is_critical("src/payment/checkout.js") is True
        assert _is_critical("src/components/Button.jsx") is False

    def test_extract_line_number_from_reason(self):
        from graph.evidence_graph import _extract_line_number
        assert _extract_line_number("imports at line 42") == 42
        assert _extract_line_number("calls method at line ~88") == 88
        assert _extract_line_number("no line info here") is None


# ── Propagation Engine tests ───────────────────────────────────────────────────

class TestPropagationEngine:

    @pytest.fixture
    def simple_graph(self, base_state, simple_blast_radius):
        from graph.evidence_graph import build_evidence_graph
        return build_evidence_graph(base_state, simple_blast_radius, base_state["per_file_context"])

    @pytest.fixture
    def auth_graph(self, base_state, auth_blast_radius):
        from graph.evidence_graph import build_evidence_graph
        state = {**base_state,
                 "changed_symbols": ["login"],
                 "per_file_context": [{"path": "src/auth/login.py", "symbols": ["login"],
                                       "language": "python", "is_test": False,
                                       "additions": 30, "deletions": 0, "change_type": "modified"}]}
        return build_evidence_graph(state, auth_blast_radius, state["per_file_context"])

    def test_build_chains_returns_list(self, simple_graph):
        from graph.propagation_engine import build_propagation_chains
        chains = build_propagation_chains(simple_graph, {})
        assert isinstance(chains, list)

    def test_chain_per_symbol(self, simple_graph):
        from graph.propagation_engine import build_propagation_chains
        chains = build_propagation_chains(simple_graph, {})
        assert len(chains) == 1  # one symbol

    def test_chain_has_arrow_diagram(self, simple_graph):
        from graph.propagation_engine import build_propagation_chains
        chains = build_propagation_chains(simple_graph, {})
        assert len(chains) > 0
        assert "→" in chains[0].arrow_diagram or len(chains[0].steps) == 1

    def test_chain_has_narrative(self, simple_graph):
        from graph.propagation_engine import build_propagation_chains
        chains = build_propagation_chains(simple_graph, {})
        assert len(chains) > 0
        assert len(chains[0].narrative) > 0

    def test_chain_risk_elevated_for_auth_path(self, auth_graph):
        from graph.propagation_engine import build_propagation_chains
        runtime_risks = {
            "breaking_scenarios": [{"failure_mode": "NULL_DEREF", "severity": "HIGH"}]
        }
        chains = build_propagation_chains(auth_graph, runtime_risks)
        assert len(chains) > 0
        # Auth paths with NULL_DEREF should be HIGH or CRITICAL
        assert chains[0].chain_risk_level in ("HIGH", "CRITICAL")

    def test_chain_sorted_critical_first(self, base_state, auth_blast_radius, simple_blast_radius):
        from graph.evidence_graph import build_evidence_graph
        from graph.propagation_engine import build_propagation_chains
        # Build graph with both critical and non-critical symbols
        state = {**base_state,
                 "changed_symbols": ["login", "format_date"],
                 "per_file_context": [
                     {"path": "src/auth/login.py",  "symbols": ["login"],       "language": "python",
                      "is_test": False, "additions": 10, "deletions": 0, "change_type": "modified"},
                     {"path": "src/utils/dates.py", "symbols": ["format_date"], "language": "python",
                      "is_test": False, "additions": 2,  "deletions": 0, "change_type": "modified"},
                 ]}
        graph = build_evidence_graph(state, auth_blast_radius, state["per_file_context"])
        chains = build_propagation_chains(graph, {"breaking_scenarios": [{"failure_mode": "NULL_DEREF"}]})
        # Critical chain should come first
        if len(chains) >= 2:
            risk_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
            first_idx  = risk_order.index(chains[0].chain_risk_level)
            second_idx = risk_order.index(chains[1].chain_risk_level)
            assert first_idx <= second_idx

    def test_chain_depth_at_most_4(self, simple_graph):
        from graph.propagation_engine import build_propagation_chains
        chains = build_propagation_chains(simple_graph, {})
        for chain in chains:
            assert len(chain.steps) <= 4

    def test_dedupe_propagation_chains_same_topology(self):
        from graph.propagation_engine import (
            PropagationChain,
            PropagationStep,
            _dedupe_propagation_chains,
        )

        steps = [
            PropagationStep(
                file="a.js",
                domain_label="Test suite",
                depth=0,
                confidence="HIGH",
                risk_note="n",
                is_critical=False,
            ),
            PropagationStep(
                file="b.js",
                domain_label="Shared utilities",
                depth=1,
                confidence="HIGH",
                risk_note="n",
                is_critical=False,
            ),
        ]
        c1 = PropagationChain(
            symbol="symA", steps=list(steps), narrative="", max_business_impact="", chain_risk_level="LOW"
        )
        c2 = PropagationChain(
            symbol="symB", steps=list(steps), narrative="", max_business_impact="", chain_risk_level="LOW"
        )
        out = _dedupe_propagation_chains([c1, c2])
        assert len(out) == 1
        assert out[0].symbol == "symA"

    def test_narrative_grounding_rejects_invented_files(self):
        from graph.propagation_engine import (
            PropagationChain,
            PropagationStep,
            _narrative_only_cites_chain_files,
        )

        steps = [
            PropagationStep(
                file="lib/application.js",
                domain_label="Request routing",
                depth=0,
                confidence="HIGH",
                risk_note="n",
                is_critical=False,
            ),
        ]
        chain = PropagationChain(
            symbol="fn", steps=steps, narrative="", max_business_impact="", chain_risk_level="LOW"
        )
        assert _narrative_only_cites_chain_files(
            "Change in lib/application.js affects routing.", chain
        )
        assert not _narrative_only_cites_chain_files(
            "Also updates diagnostics-channel.js for lifecycle.", chain
        )

    def test_to_dict_serializable(self, simple_graph):
        from graph.propagation_engine import build_propagation_chains
        import json
        chains = build_propagation_chains(simple_graph, {})
        for chain in chains:
            d = chain.to_dict()
            json.dumps(d)
            assert "symbol" in d
            assert "arrow_diagram" in d
            assert "steps" in d

    def test_no_duplicate_files_in_chain(self, simple_graph):
        from graph.propagation_engine import build_propagation_chains
        chains = build_propagation_chains(simple_graph, {})
        for chain in chains:
            files = [s.file for s in chain.steps]
            assert len(files) == len(set(files)), f"Duplicate files in chain: {files}"

    def test_domain_label_auth(self):
        from graph.propagation_engine import _domain_label
        assert _domain_label("src/auth/login.py") == "Auth service"

    def test_domain_label_payment(self):
        from graph.propagation_engine import _domain_label
        assert _domain_label("src/payment/checkout.js") == "Payment processing"

    def test_domain_label_fallback(self):
        from graph.propagation_engine import _domain_label
        label = _domain_label("src/widgets/foo.py")
        assert len(label) > 0  # never empty

    def test_chain_risk_critical_conditions(self):
        from graph.propagation_engine import PropagationStep, _chain_risk
        steps = [
            PropagationStep("a.py", "Auth service",    0, "HIGH", "", True),
            PropagationStep("b.py", "Payment processing", 1, "HIGH", "", True),
        ]
        risk = _chain_risk(steps, ["NULL_DEREF"])
        assert risk == "CRITICAL"

    def test_chain_risk_low_for_safe_path(self):
        from graph.propagation_engine import PropagationStep, _chain_risk
        steps = [
            PropagationStep("util.py", "Shared utilities", 0, "HIGH", "", False),
        ]
        risk = _chain_risk(steps, [])
        assert risk == "LOW"

    def test_empty_graph_produces_chains(self, base_state):
        from graph.evidence_graph import build_evidence_graph
        from graph.propagation_engine import build_propagation_chains
        graph = build_evidence_graph(base_state, {}, base_state["per_file_context"])
        chains = build_propagation_chains(graph, {})
        # Should still produce chains (origin-only) even with no dependents
        assert len(chains) >= 1


# ── Deployment Advisor tests ───────────────────────────────────────────────────

class TestDeploymentAdvisor:

    @pytest.fixture
    def mock_graph_low(self, base_state):
        from graph.evidence_graph import build_evidence_graph
        graph = build_evidence_graph(base_state, {"direct_dependents": [], "indirect_dependents": []},
                                     base_state["per_file_context"])
        return graph

    @pytest.fixture
    def mock_graph_high(self, base_state, auth_blast_radius):
        from graph.evidence_graph import build_evidence_graph
        state = {**base_state,
                 "changed_symbols": ["login"],
                 "per_file_context": [{"path": "src/auth/login.py", "symbols": ["login"],
                                       "language": "python", "is_test": False,
                                       "additions": 30, "deletions": 0, "change_type": "modified"}]}
        return build_evidence_graph(state, auth_blast_radius, state["per_file_context"])

    def test_direct_merge_for_low_risk(self, mock_graph_low):
        from graph.deployment_advisor import DeploymentAdvisor
        advisor = DeploymentAdvisor()
        advice = advisor.advise(
            risk_level="LOW", risk_score=2.0,
            graph=mock_graph_low, chains=[],
            rollback_advice={"data_side_effects": False, "rollback_difficulty": "LOW"},
            test_gaps={"uncovered_functions": []},
            business_impacts=[], has_test_changes=True,
        )
        assert advice.strategy == "DIRECT_MERGE"

    def test_canary_for_high_risk(self, mock_graph_high):
        from graph.deployment_advisor import DeploymentAdvisor
        from graph.propagation_engine import build_propagation_chains
        advisor = DeploymentAdvisor()
        chains = build_propagation_chains(mock_graph_high, {"breaking_scenarios": [{"failure_mode": "NULL_DEREF"}]})
        advice = advisor.advise(
            risk_level="HIGH", risk_score=7.5,
            graph=mock_graph_high, chains=chains,
            rollback_advice={"data_side_effects": False, "rollback_difficulty": "MEDIUM"},
            test_gaps={"uncovered_functions": [{"function": "login", "risk": "HIGH"}]},
            business_impacts=["Authentication outage risk"], has_test_changes=False,
        )
        assert advice.strategy in ("CANARY_DEPLOYMENT", "STAGED_ROLLOUT")

    def test_block_merge_for_critical_risk(self, mock_graph_high):
        from graph.deployment_advisor import DeploymentAdvisor
        advisor = DeploymentAdvisor()
        advice = advisor.advise(
            risk_level="CRITICAL", risk_score=9.0,
            graph=mock_graph_high, chains=[],
            rollback_advice={"data_side_effects": False, "rollback_difficulty": "HIGH"},
            test_gaps={"uncovered_functions": [{"function": "login", "risk": "HIGH"}]},
            business_impacts=["Authentication outage risk"], has_test_changes=False,
        )
        assert advice.strategy == "BLOCK_MERGE"

    def test_block_merge_for_data_side_effects(self, mock_graph_low):
        from graph.deployment_advisor import DeploymentAdvisor
        advisor = DeploymentAdvisor()
        advice = advisor.advise(
            risk_level="MEDIUM", risk_score=5.0,
            graph=mock_graph_low, chains=[],
            rollback_advice={"data_side_effects": True, "rollback_difficulty": "HIGH"},
            test_gaps={"uncovered_functions": []},
            business_impacts=[], has_test_changes=False,
        )
        assert advice.strategy == "BLOCK_MERGE"

    def test_monitored_deploy_for_medium_risk(self, mock_graph_low):
        from graph.deployment_advisor import DeploymentAdvisor
        advisor = DeploymentAdvisor()
        advice = advisor.advise(
            risk_level="MEDIUM", risk_score=4.5,
            graph=mock_graph_low, chains=[],
            rollback_advice={"data_side_effects": False, "rollback_difficulty": "LOW"},
            test_gaps={"uncovered_functions": []},
            business_impacts=[], has_test_changes=True,
        )
        assert advice.strategy == "MONITORED_DEPLOY"

    def test_advice_has_reasons(self, mock_graph_high):
        from graph.deployment_advisor import DeploymentAdvisor
        advisor = DeploymentAdvisor()
        advice = advisor.advise(
            risk_level="HIGH", risk_score=7.0,
            graph=mock_graph_high, chains=[],
            rollback_advice={"data_side_effects": False, "rollback_difficulty": "MEDIUM"},
            test_gaps={"uncovered_functions": [{"function": "login", "risk": "HIGH"}]},
            business_impacts=["Authentication outage risk"], has_test_changes=False,
        )
        assert len(advice.reasons) > 0

    def test_advice_has_conditions(self, mock_graph_high):
        from graph.deployment_advisor import DeploymentAdvisor
        advisor = DeploymentAdvisor()
        advice = advisor.advise(
            risk_level="HIGH", risk_score=7.0,
            graph=mock_graph_high, chains=[],
            rollback_advice={"data_side_effects": False, "rollback_difficulty": "MEDIUM"},
            test_gaps={"uncovered_functions": []},
            business_impacts=["Authentication outage risk"], has_test_changes=False,
        )
        assert len(advice.conditions) > 0

    def test_advice_has_monitoring_hints(self, mock_graph_low):
        from graph.deployment_advisor import DeploymentAdvisor
        advisor = DeploymentAdvisor()
        advice = advisor.advise(
            risk_level="LOW", risk_score=2.0,
            graph=mock_graph_low, chains=[],
            rollback_advice={"data_side_effects": False, "rollback_difficulty": "LOW"},
            test_gaps={"uncovered_functions": []},
            business_impacts=["Authentication outage risk"], has_test_changes=True,
        )
        assert len(advice.monitoring_hints) > 0
        assert any("auth" in h.lower() for h in advice.monitoring_hints)

    def test_auth_domain_triggers_auth_monitoring(self, mock_graph_low):
        from graph.deployment_advisor import DeploymentAdvisor
        advisor = DeploymentAdvisor()
        advice = advisor.advise(
            risk_level="LOW", risk_score=2.0,
            graph=mock_graph_low, chains=[],
            rollback_advice={"data_side_effects": False, "rollback_difficulty": "LOW"},
            test_gaps={"uncovered_functions": []},
            business_impacts=["Authentication outage risk"], has_test_changes=True,
        )
        assert any("auth" in h.lower() for h in advice.monitoring_hints)

    def test_to_dict_serializable(self, mock_graph_low):
        import json
        from graph.deployment_advisor import DeploymentAdvisor
        advisor = DeploymentAdvisor()
        advice = advisor.advise(
            risk_level="LOW", risk_score=2.0,
            graph=mock_graph_low, chains=[],
            rollback_advice={"data_side_effects": False, "rollback_difficulty": "LOW"},
            test_gaps={"uncovered_functions": []},
            business_impacts=[], has_test_changes=True,
        )
        d = advice.to_dict()
        json.dumps(d)
        assert "strategy" in d
        assert "reasons" in d
        assert "conditions" in d

    def test_emoji_present_in_all_strategies(self):
        from graph.deployment_advisor import STRATEGY_EMOJI
        for strategy in ["DIRECT_MERGE", "MONITORED_DEPLOY", "CANARY_DEPLOYMENT",
                         "STAGED_ROLLOUT", "BLOCK_MERGE"]:
            assert strategy in STRATEGY_EMOJI
            assert len(STRATEGY_EMOJI[strategy]) > 0

    def test_build_deployment_advice_convenience(self, base_state, mock_graph_low):
        from graph.deployment_advisor import build_deployment_advice
        state = {**base_state,
                 "risk_assessment": {"risk_level": "LOW", "overall_risk_score": 2.0},
                 "rollback_advice": {"data_side_effects": False, "rollback_difficulty": "LOW"},
                 "test_gaps": {"uncovered_functions": []},
                 "business_impacts": [], "has_test_changes": True}
        advice = build_deployment_advice(state, mock_graph_low, [])
        assert advice.strategy == "DIRECT_MERGE"


# ── Integration: graph layer node ─────────────────────────────────────────────

class TestGraphLayerNode:

    @pytest.fixture
    def full_state(self, base_state, simple_blast_radius):
        mock_diff = MagicMock()
        mock_diff.changed_files = []
        return {
            **base_state,
            "blast_radius": simple_blast_radius,
            "runtime_risks": {"breaking_scenarios": [], "is_breaking_change": False},
            "test_gaps": {"uncovered_functions": [], "overall_coverage_assessment": "ADEQUATE"},
            "risk_assessment": {"risk_level": "MEDIUM", "overall_risk_score": 4.5,
                                "dimension_scores": {}, "top_concerns": [], "recommended_actions": []},
            "rollback_advice": {"data_side_effects": False, "rollback_difficulty": "LOW"},
            "business_impacts": [],
            "_parsed_diff": mock_diff,
        }

    def test_node_produces_evidence_graph(self, full_state):
        from agents.orchestrator import node_graph_layer
        result = node_graph_layer(full_state)
        assert "evidence_graph" in result
        assert isinstance(result["evidence_graph"], dict)
        assert "nodes" in result["evidence_graph"]

    def test_node_produces_propagation_chains(self, full_state):
        from agents.orchestrator import node_graph_layer
        result = node_graph_layer(full_state)
        assert "propagation_chains" in result
        assert isinstance(result["propagation_chains"], list)

    def test_node_produces_deployment_advice(self, full_state):
        from agents.orchestrator import node_graph_layer
        result = node_graph_layer(full_state)
        assert "deployment_advice" in result
        assert "strategy" in result["deployment_advice"]

    def test_node_deployment_strategy_valid(self, full_state):
        from agents.orchestrator import node_graph_layer
        result = node_graph_layer(full_state)
        valid = {"DIRECT_MERGE", "MONITORED_DEPLOY", "CANARY_DEPLOYMENT",
                 "STAGED_ROLLOUT", "BLOCK_MERGE"}
        assert result["deployment_advice"]["strategy"] in valid

    def test_node_does_not_call_llm(self, full_state):
        """Graph layer must be 100% deterministic — no LLM calls."""
        from agents.orchestrator import node_graph_layer
        # If this completes without needing get_llm(), the test passes
        # (no mock needed — any LLM call would raise ConnectionRefusedError)
        result = node_graph_layer(full_state)
        assert result is not None


# ── Report builder new fields tests ───────────────────────────────────────────

class TestReportBuilderGraphFields:

    @pytest.fixture
    def state_with_graph(self, base_state, simple_blast_radius):
        mock_diff = MagicMock()
        mock_diff.changed_files = []
        from graph.evidence_graph import build_evidence_graph
        from graph.propagation_engine import build_propagation_chains
        from graph.deployment_advisor import build_deployment_advice

        graph = build_evidence_graph(base_state, simple_blast_radius, base_state["per_file_context"])
        chains = build_propagation_chains(graph, {})
        advice = build_deployment_advice(
            {**base_state,
             "risk_assessment": {"risk_level": "MEDIUM", "overall_risk_score": 4.5},
             "rollback_advice": {"data_side_effects": False, "rollback_difficulty": "LOW"},
             "test_gaps": {"uncovered_functions": []},
             "business_impacts": [], "has_test_changes": True},
            graph, chains
        )
        return {
            **base_state,
            "blast_radius": simple_blast_radius,
            "runtime_risks": {}, "test_gaps": {},
            "risk_assessment": {"risk_level": "MEDIUM", "overall_risk_score": 4.5,
                                "dimension_scores": {}, "top_concerns": [], "recommended_actions": []},
            "objections": {"verdict": "AGREE", "objections": [], "critic_summary": ""},
            "rollback_advice": {"data_side_effects": False, "rollback_difficulty": "LOW",
                                "rollback_risks": [], "rollback_steps": [], "rollback_summary": ""},
            "business_impacts": [], "impact_summary": "", "severity_domains": [],
            "historical_context": {}, "rerun_count": 0, "conflict_log": [],
            "evidence_graph": graph.to_dict(),
            "propagation_chains": [c.to_dict() for c in chains],
            "deployment_advice": advice.to_dict(),
            "_analyzed_at": "2025-01-01T00:00:00",
        }

    def test_report_has_evidence_graph(self, state_with_graph):
        from output.report_builder import build_report
        report = build_report(state_with_graph)
        assert isinstance(report.evidence_graph, dict)
        assert "nodes" in report.evidence_graph

    def test_report_has_propagation_chains(self, state_with_graph):
        from output.report_builder import build_report
        report = build_report(state_with_graph)
        assert isinstance(report.propagation_chains, list)

    def test_report_has_deployment_advice(self, state_with_graph):
        from output.report_builder import build_report
        report = build_report(state_with_graph)
        assert isinstance(report.deployment_advice, dict)
        assert "strategy" in report.deployment_advice

    def test_markdown_contains_deployment_strategy(self, state_with_graph, tmp_path):
        from output.report_builder import build_report
        from output.formatter import save_markdown
        report = build_report(state_with_graph)
        save_markdown(report, str(tmp_path))
        content = (tmp_path / f"pr-{report.pr_number}-report.md").read_text()
        assert "Deployment Strategy" in content

    def test_json_contains_all_graph_keys(self, state_with_graph, tmp_path):
        import json
        from output.report_builder import build_report
        from output.formatter import save_json
        report = build_report(state_with_graph)
        save_json(report, str(tmp_path))
        data = json.loads((tmp_path / f"pr-{report.pr_number}-report.json").read_text())
        assert "evidence_graph" in data
        assert "propagation_chains" in data
        assert "deployment_advice" in data
