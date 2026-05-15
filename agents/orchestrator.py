"""
Orchestrator — LangGraph StateGraph

This is the brain of the system. It defines:
  1. The shared state dict (the "whiteboard" all agents read/write)
  2. The graph nodes (one per agent)
  3. The edges (execution order + conditional loop for conflict resolution)

Graph flow:
  START
    → dependency_mapper
    → change_simulator
    → test_gap
    → risk_evaluator
    → critic
    → [conditional] if SIGNIFICANT_ISSUES and reruns < 2:
        → re-run flagged agents with Critic's objections
        → risk_evaluator (re-score)
        → critic (re-check)
    → END

The state dict is typed using TypedDict so LangGraph can manage it.
"""

import json
from typing import TypedDict, Annotated
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from tools.github_tools import make_github_tools
from config.settings import RISK_CONFLICT_THRESHOLD

import agents.dependency_mapper as dep_mapper
import agents.change_simulator as sim
import agents.test_gap as test_gap
from github.diff_static_risks import augment_test_gaps_with_diff
import agents.risk_evaluator as risk_eval
import agents.critic as critic_agent
import agents.rollback_advisor as rollback_adv
import agents.business_impact as biz_impact
from graph.evidence_graph import build_evidence_graph
from graph.propagation_engine import build_propagation_chains
from graph.deployment_advisor import build_deployment_advice

console = Console()


# ── State definition ───────────────────────────────────────────────────────────

class PRAnalysisState(TypedDict):
    # Input fields (set before graph runs)
    repo: str
    pr_number: int
    pr_title: str
    pr_url: str
    pr_author: str
    base_branch: str
    pr_html_url: str

    pr_description: str
    diff_summary: str
    total_additions: int
    total_deletions: int
    languages: list
    changed_files: list
    source_files: list
    test_files: list
    changed_symbols: list
    analysis_symbols: list
    has_test_changes: bool
    per_file_context: list
    analysis_per_file_context: list
    raw_diff: str
    repo_history: str
    review_comments: list
    review_states: list

    # Agent outputs (written by graph nodes)
    blast_radius: dict
    runtime_risks: dict
    test_gaps: dict
    risk_assessment: dict
    objections: dict
    rollback_advice: dict

    # Graph layer outputs (Priority 1-3)
    evidence_graph: dict
    propagation_chains: list
    deployment_advice: dict

    # Enhancement outputs (written post-graph)
    business_impacts: list
    impact_summary: str
    severity_domains: list
    historical_context: dict

    # Conflict resolution tracking
    rerun_count: int
    conflict_log: list      # list of objection dicts from each Critic run

    # Non-serializable (private — set in build_graph, not persisted)
    _github_client: object
    _memory_store: object
    _parsed_diff: object
    _analyzed_at: str


# ── Node functions ─────────────────────────────────────────────────────────────

def node_dependency_mapper(state: PRAnalysisState) -> dict:
    console.print("\n[bold cyan]▶ Agent 1/7:[/bold cyan] Dependency Mapper running...")
    client = state["_github_client"]
    tools_dict = make_github_tools(client)
    tools = [tools_dict["search_symbol"], tools_dict["file_tree"]]

    output = dep_mapper.run(state, tools)
    _log_agent_result("dependency_mapper", output)

    blast_data = output.data if output.success else {
        "error": output.error,
        "direct_dependents": [],
        "indirect_dependents": [],
        "blast_radius_summary": "Agent failed",
    }

    if output.success and blast_data:
        def _is_test_path(path: str) -> bool:
            p = path.lower()
            return any(seg in p for seg in [
                "test", "tests", "spec", "specs", "__tests__", "_test.", ".test.", ".spec."
            ])

        direct = [d for d in (blast_data.get("direct_dependents") or []) if not _is_test_path(d.get("file", ""))]
        indirect = [d for d in (blast_data.get("indirect_dependents") or []) if not _is_test_path(d.get("file", ""))]

        # Inject the PR's own changed source files as baseline dependents
        # The blast radius must always include the files actually changed in this PR
        source_files = state.get("source_files", [])
        per_file_ctx = state.get("analysis_per_file_context", state.get("per_file_context", []))
        existing_paths = {d.get("file", "") for d in direct}

        for src_file in source_files:
            if src_file not in existing_paths:
                # Find symbols for this file from per_file_context
                file_syms = []
                for fc in per_file_ctx:
                    if fc.get("path") == src_file:
                        file_syms = fc.get("symbols", [])
                        break
                sym_str = ", ".join(file_syms[:3]) if file_syms else "modified"
                direct.append({
                    "file": src_file,
                    "reason": f"Directly modified in PR ({sym_str})",
                    "confidence": "HIGH",
                    "_baseline": True,
                })
                existing_paths.add(src_file)

        blast_data["direct_dependents"] = direct
        blast_data["indirect_dependents"] = indirect
        blast_data["blast_radius_summary"] = f"{len(direct)} files directly affected, {len(indirect)} indirectly"

    return {"blast_radius": blast_data}


def node_change_simulator(state: PRAnalysisState) -> dict:
    console.print("\n[bold cyan]▶ Agent 2/7:[/bold cyan] Change Simulator running...")
    client = state["_github_client"]
    tools_dict = make_github_tools(client)
    tools = [tools_dict["fetch_file"]]

    output = sim.run(state, tools, blast_radius=state.get("blast_radius", {}))
    _log_agent_result("change_simulator", output)

    return {"runtime_risks": output.data if output.success else {"error": output.error, "breaking_scenarios": [], "is_breaking_change": False, "simulator_summary": "Agent failed"}}


def node_test_gap(state: PRAnalysisState) -> dict:
    console.print("\n[bold cyan]▶ Agent 3/7:[/bold cyan] Test Gap Agent running...")
    client = state["_github_client"]
    tools_dict = make_github_tools(client)
    tools = [tools_dict["find_test_files"], tools_dict["fetch_file"]]

    output = test_gap.run(state, tools)
    _log_agent_result("test_gap", output)

    if output.success and output.data:
        output.data = augment_test_gaps_with_diff(
            output.data,
            state.get("raw_diff", ""),
            per_file_context=state.get("analysis_per_file_context", state.get("per_file_context", [])),
        )

    return {"test_gaps": output.data if output.success else {"error": output.error, "uncovered_functions": [], "overall_coverage_assessment": "UNKNOWN", "test_gap_summary": "Agent failed"}}


def node_risk_evaluator(state: PRAnalysisState) -> dict:
    console.print("\n[bold cyan]▶ Agent 4/7:[/bold cyan] Risk Evaluator running...")
    output = risk_eval.run(state)
    _log_agent_result("risk_evaluator", output)

    return {"risk_assessment": output.data if output.success else {"error": output.error, "overall_risk_score": 5.0, "risk_level": "MEDIUM", "top_concerns": [], "recommended_actions": []}}


def node_rollback_advisor(state: PRAnalysisState) -> dict:
    console.print("\n[bold cyan]▶ Agent 5/7:[/bold cyan] Rollback Advisor running...")
    output = rollback_adv.run(state)
    _log_agent_result("rollback_advisor", output)

    return {"rollback_advice": output.data if output.success else {
        "rollback_difficulty": "MEDIUM",
        "rollback_risks": ["Unable to assess — manual review required"],
        "rollback_steps": [],
        "rollback_summary": "Agent failed",
        "confidence": 1,
    }}


def node_risk_evaluator_with_business(state: PRAnalysisState) -> dict:
    """Runs business impact classifier BEFORE risk evaluator so scores reflect business domains."""
    console.print("\n[bold cyan]▶ Agent 6/7:[/bold cyan] Business Impact + Risk Evaluator running...")

    # Run deterministic business impact classifier (no LLM call)
    biz = biz_impact.analyze(state)
    console.print(f"  [green]✓[/green] business_impact: {len(biz.get('business_impacts', []))} domains identified")

    # Inject business impacts into state for risk_evaluator to see
    enriched_state = {**state, **biz}
    output = risk_eval.run(enriched_state)
    _log_agent_result("risk_evaluator", output)

    risk_data = output.data if output.success else {
        "error": output.error, "overall_risk_score": 5.0, "risk_level": "MEDIUM",
        "top_concerns": [], "recommended_actions": [],
    }

    return {
        "risk_assessment": risk_data,
        "business_impacts": biz.get("business_impacts", []),
        "impact_summary": biz.get("impact_summary", ""),
        "severity_domains": biz.get("severity_domains", []),
    }


def node_critic(state: PRAnalysisState) -> dict:
    console.print("\n[bold cyan]▶ Agent 7/7:[/bold cyan] Critic Agent running...")
    output = critic_agent.run(state)
    _log_agent_result("critic", output)

    # Append to conflict log — include score snapshot for delta guard
    conflict_log = list(state.get("conflict_log", []))
    current_score = state.get("risk_assessment", {}).get("overall_risk_score", 0)
    entry = {**(output.data if output.data else {}), "_score_snapshot": current_score}
    conflict_log.append(entry)

    updates = {
        "objections": output.data if output.success else {"verdict": "AGREE", "objections": [], "critic_summary": "Critic failed"},
        "conflict_log": conflict_log,
    }

    # Apply critic score recommendations —
    # If the critic targets risk_evaluator and suggests a specific higher score, apply it.
    if output.success and output.data:
        import re
        risk_assessment = dict(state.get("risk_assessment", {}))
        current = float(risk_assessment.get("overall_risk_score", 0))
        applied = False

        for obj in (output.data.get("objections") or []):
            if not isinstance(obj, dict):
                continue
            target = obj.get("target_agent", "")
            if target != "risk_evaluator":
                continue

            # Extract numeric score from suggested_correction
            correction = obj.get("suggested_correction", "")
            scores = re.findall(r"(\d+(?:\.\d+)?)", correction)
            for s in scores:
                suggested = float(s)
                if 1.0 <= suggested <= 10.0 and suggested > current + 0.5:
                    console.print(
                        f"  [yellow]⚠ Applying critic correction: "
                        f"{current:.1f} → {suggested:.1f} ({correction[:60]})[/yellow]"
                    )
                    risk_assessment["overall_risk_score"] = suggested
                    risk_assessment["_critic_override"] = True
                    # Re-derive risk level
                    if suggested <= 3.0:
                        risk_assessment["risk_level"] = "LOW"
                    elif suggested <= 5.9:
                        risk_assessment["risk_level"] = "MEDIUM"
                    elif suggested <= 7.9:
                        risk_assessment["risk_level"] = "HIGH"
                    else:
                        risk_assessment["risk_level"] = "CRITICAL"
                    applied = True
                    break
            if applied:
                break

        if applied:
            updates["risk_assessment"] = risk_assessment

    return updates


def node_rerun_with_objections(state: PRAnalysisState) -> dict:
    """
    Re-runs flagged agents with the Critic's objections injected.
    Only called when verdict is SIGNIFICANT_ISSUES.
    """
    objections = state.get("objections", {})
    rerun_count = state.get("rerun_count", 0) + 1
    console.print(f"\n[bold yellow]⚠ Conflict resolution — re-run #{rerun_count}[/bold yellow]")
    console.print(f"  Critic verdict: {objections.get('verdict', '?')}")
    console.print(f"  Objections: {len(objections.get('objections', []))}")

    updates: dict = {"rerun_count": rerun_count}
    client = state["_github_client"]
    tools_dict = make_github_tools(client)

    for obj in objections.get("objections", [])[:2]:   # max 2 objections per round
        target = obj.get("target_agent", "")
        rerun_context = critic_agent.build_rerun_context(
            state.get(_agent_state_key(target), {}),
            obj,
        )

        console.print(f"  Re-running: [yellow]{target}[/yellow] — {obj.get('claim', '')[:60]}")

        if target == "dependency_mapper":
            augmented_state = dict(state)
            augmented_state["repo_history"] = rerun_context + "\n" + state.get("repo_history", "")
            tools = [tools_dict["search_symbol"], tools_dict["file_tree"]]
            output = dep_mapper.run(augmented_state, tools)
            if output.success:
                updates["blast_radius"] = output.data

        elif target == "change_simulator":
            augmented_state = dict(state)
            augmented_state["repo_history"] = rerun_context + "\n" + state.get("repo_history", "")
            tools = [tools_dict["fetch_file"]]
            output = sim.run(augmented_state, tools, blast_radius=state.get("blast_radius", {}))
            if output.success:
                updates["runtime_risks"] = output.data

        elif target == "test_gap":
            augmented_state = dict(state)
            augmented_state["repo_history"] = rerun_context + "\n" + state.get("repo_history", "")
            tools = [tools_dict["find_test_files"], tools_dict["fetch_file"]]
            output = test_gap.run(augmented_state, tools)
            if output.success:
                updates["test_gaps"] = output.data

        elif target == "risk_evaluator":
            # Re-run risk evaluator with updated state
            merged = {**state, **updates}
            output = risk_eval.run(merged)
            if output.success:
                updates["risk_assessment"] = output.data

    # Always re-run risk evaluator after any rerun to re-score
    merged_state = {**state, **updates}
    console.print("  Re-scoring risk after objection resolution...")
    risk_output = risk_eval.run(merged_state)
    if risk_output.success:
        updates["risk_assessment"] = risk_output.data

    return updates


# ── Conditional edge ───────────────────────────────────────────────────────────

def should_rerun(state: PRAnalysisState) -> str:
    """
    Decides whether to loop back for conflict resolution or proceed to END.

    Conditions for re-run:
      - Critic verdict is SIGNIFICANT_ISSUES
      - We haven't exceeded 2 re-run rounds already
      - Score delta from previous round is >= 0.5 (score delta guard)
    """
    from agents.orchestrator_patch import should_exit_early

    verdict = state.get("objections", {}).get("verdict", "AGREE")
    rerun_count = state.get("rerun_count", 0)

    if verdict == "SIGNIFICANT_ISSUES" and rerun_count < 2:
        # Score delta guard — don't loop if agents just repeated themselves
        conflict_log = state.get("conflict_log", [])
        if len(conflict_log) >= 2:
            prev_score = conflict_log[-2].get("_score_snapshot", 0)
            curr_score = state.get("risk_assessment", {}).get("overall_risk_score", 0)
            if should_exit_early(prev_score, curr_score, rerun_count):
                console.print("  [green]→ Score converged. Skipping further re-runs.[/green]")
                return "done"

        console.print(f"  [yellow]→ Routing to conflict resolution (round {rerun_count + 1}/2)[/yellow]")
        return "rerun"
    else:
        if rerun_count >= 2:
            console.print("  [dim]→ Max re-run rounds reached. Proceeding with current scores.[/dim]")
        else:
            console.print("  [green]→ Critic satisfied. Proceeding to final report.[/green]")
        return "done"


# ── Graph Layer Node ───────────────────────────────────────────────────────────

def node_graph_layer(state: PRAnalysisState) -> dict:
    """
    Runs the three graph engines in sequence after all agents complete.
    Entirely deterministic — no LLM calls.
    Builds: EvidenceGraph → PropagationChains → DeploymentAdvice
    """
    console.print("\n[bold cyan]▶ Graph Layer:[/bold cyan] Evidence graph + propagation + deployment advice...")

    blast_radius    = state.get("blast_radius", {})
    per_file_ctx    = state.get("analysis_per_file_context", state.get("per_file_context", []))
    runtime_risks   = state.get("runtime_risks", {})

    # Priority 1: Evidence Graph
    graph = build_evidence_graph(state, blast_radius, per_file_ctx)
    console.print(
        f"  [green]✓[/green] Evidence graph: {graph.total_symbols_changed} symbols, "
        f"{graph.total_files_affected} affected files, "
        f"depth {graph.max_propagation_depth}"
    )

    # Priority 2: Failure Propagation Chains
    chains = build_propagation_chains(graph, runtime_risks, state=state)
    if chains:
        console.print(f"  [green]✓[/green] Propagation chains: {len(chains)} chain(s)")
        for chain in chains[:2]:
            console.print(f"    [dim]{chain.arrow_diagram}[/dim]")

    # Priority 3: Deployment Advisor
    advice = build_deployment_advice(state, graph, chains)
    console.print(
        f"  [green]✓[/green] Deployment strategy: "
        f"[bold]{advice.strategy}[/bold] (confidence: {advice.confidence})"
    )

    return {
        "evidence_graph":    graph.to_dict(),
        "propagation_chains": [c.to_dict() for c in chains],
        "deployment_advice": advice.to_dict(),
    }


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph():
    """Build and compile the LangGraph StateGraph."""
    graph = StateGraph(PRAnalysisState)

    # Register nodes
    graph.add_node("dependency_mapper",            node_dependency_mapper)
    graph.add_node("change_simulator",             node_change_simulator)
    graph.add_node("test_gap",                     node_test_gap)
    graph.add_node("rollback_advisor",             node_rollback_advisor)
    graph.add_node("risk_evaluator_with_business", node_risk_evaluator_with_business)
    graph.add_node("critic",                       node_critic)
    graph.add_node("rerun_with_objections",        node_rerun_with_objections)
    graph.add_node("graph_layer",                  node_graph_layer)

    # Linear edges
    graph.add_edge(START,                          "dependency_mapper")
    graph.add_edge("dependency_mapper",            "change_simulator")
    graph.add_edge("change_simulator",             "test_gap")
    graph.add_edge("test_gap",                     "rollback_advisor")
    graph.add_edge("rollback_advisor",             "risk_evaluator_with_business")
    graph.add_edge("risk_evaluator_with_business", "critic")

    # Conditional edge after Critic
    graph.add_conditional_edges(
        "critic",
        should_rerun,
        {
            "rerun": "rerun_with_objections",
            "done":  "graph_layer",
        }
    )

    # After re-run: re-run critic to check if issues are resolved
    graph.add_edge("rerun_with_objections", "critic")
    graph.add_edge("graph_layer",           END)

    return graph.compile()


# ── Run function ───────────────────────────────────────────────────────────────

def run_analysis(context: dict) -> dict:
    """
    Main entry point. Takes the context dict built by Day 1's build_context()
    and runs the full multi-agent analysis.

    Returns the final state dict with all agent outputs.
    """
    console.print(Panel(
        f"[bold]PR:[/bold] {context.get('pr_title', '')}\n"
        f"[bold]Repo:[/bold] {context.get('repo', '')}\n"
        f"[bold]Diff:[/bold] {context.get('diff_summary', '')}",
        title="[bold green]Starting Multi-Agent Analysis[/bold green]",
        border_style="green",
    ))

    # Build initial state — merge context dict into PRAnalysisState shape
    initial_state: PRAnalysisState = {
        # PR identity
        "repo":         context["repo"],
        "pr_number":    context["pr_number"],
        "pr_title":     context["pr_title"],
        "pr_url":       context["pr_url"],
        "pr_author":    context["pr_author"],
        "base_branch":  context["base_branch"],
        "pr_html_url":  context["pr_html_url"],

        # PR body / description (contains CVE refs, design rationale)
        "pr_description": context.get("pr_description", ""),

        # Diff data
        "diff_summary":     context["diff_summary"],
        "total_additions":  context["total_additions"],
        "total_deletions":  context["total_deletions"],
        "languages":        context["languages"],
        "changed_files":    context["changed_files"],
        "source_files":     context["source_files"],
        "test_files":       context["test_files"],
        "changed_symbols":  context["changed_symbols"],
        "analysis_symbols": context.get("analysis_symbols", context["changed_symbols"]),
        "has_test_changes": context["has_test_changes"],
        "per_file_context": context["per_file_context"],
        "analysis_per_file_context": context.get("analysis_per_file_context", context["per_file_context"]),
        "raw_diff":         context["raw_diff"],
        "repo_history":     context["repo_history"],
        "review_comments":  context.get("review_comments", []),
        "review_states":    context.get("review_states", []),

        # Agent outputs — empty until agents run
        "blast_radius":    {},
        "runtime_risks":   {},
        "test_gaps":       {},
        "risk_assessment": {},
        "objections":      {},
        "rollback_advice": {},

        # Graph layer outputs
        "evidence_graph":    {},
        "propagation_chains": [],
        "deployment_advice": {},

        # Enhancement outputs
        "business_impacts":  [],
        "impact_summary":    "",
        "severity_domains":  [],
        "historical_context": {},

        # Conflict tracking
        "rerun_count":  0,
        "conflict_log": [],

        # Private — passed through but not serialized by LangGraph
        "_github_client": context["_github_client"],
        "_memory_store":  context["_memory_store"],
        "_parsed_diff":   context["_parsed_diff"],
        "_analyzed_at":   context["_analyzed_at"],
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    # ── Post-graph enrichment ──────────────────────────────────────────────────

    # Historical context from memory store
    memory = context.get("_memory_store")
    if memory:
        try:
            hist = memory.get_historical_context(
                context["repo"],
                context["changed_files"],
                context["changed_symbols"],
            )
            final_state["historical_context"] = hist
        except Exception:
            final_state["historical_context"] = {}

    # Adjudication summary for the final report
    from agents.orchestrator_patch import compute_adjudication_summary
    final_state["adjudication_summary"] = compute_adjudication_summary(
        final_state.get("conflict_log", []),
        final_state.get("risk_assessment", {}),
        final_state.get("rerun_count", 0),
        runtime_risks=final_state.get("runtime_risks"),
        risk_assessment=final_state.get("risk_assessment"),
        verdict=final_state.get("objections", {}).get("verdict", "AGREE"),
    )

    return final_state


# ── Helpers ────────────────────────────────────────────────────────────────────

def _agent_state_key(agent_name: str) -> str:
    mapping = {
        "dependency_mapper": "blast_radius",
        "change_simulator":  "runtime_risks",
        "test_gap":          "test_gaps",
        "risk_evaluator":    "risk_assessment",
    }
    return mapping.get(agent_name, agent_name)


def _log_agent_result(name: str, output):
    if output.success:
        console.print(f"  [green]✓[/green] {name} completed ({output.tool_calls_made} tool calls)")
    else:
        console.print(f"  [red]✗[/red] {name} failed: {output.error[:100]}")