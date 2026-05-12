"""
Counterfactual Deployment Advisor

Translates risk level + propagation depth + affected domains
into a concrete deployment strategy recommendation.

Keeps it simple as the GPT proposal correctly says:
  "Do NOT simulate infrastructure. Just infer safer rollout strategies."

Strategies (in order of conservatism):
  DIRECT_MERGE        — Low risk, merge and deploy normally
  MONITORED_DEPLOY    — Medium risk, deploy with enhanced monitoring + alerts
  CANARY_DEPLOYMENT   — High risk + limited blast radius, route 5% of traffic first
  STAGED_ROLLOUT      — High risk + wide blast radius, deploy to staging then prod
  BLOCK_MERGE         — Critical risk or data integrity concern, do not merge yet

Output:
  {
    "strategy": "CANARY_DEPLOYMENT",
    "confidence": "HIGH",
    "reasons": ["Auth module modified", "3 direct dependents"],
    "conditions": ["Add null-check tests before merge", "Set up alert on auth error rate"],
    "monitoring_hints": ["Watch auth_failure_rate metric", "Alert on 5xx spike"],
    "estimated_blast_radius": "3 files, 2 critical paths",
    "summary": "one-sentence recommendation"
  }

This is entirely deterministic — no LLM calls.
The decision tree is explicit and auditable.
"""

from dataclasses import dataclass, field
from graph.evidence_graph import EvidenceGraph
from graph.propagation_engine import PropagationChain


# ── Strategy definitions ────────────────────────────────────────────────────────

STRATEGY_DESCRIPTIONS = {
    "DIRECT_MERGE": "Low risk — merge and deploy normally with standard monitoring.",
    "MONITORED_DEPLOY": "Medium risk — deploy with enhanced monitoring and rollback plan ready.",
    "CANARY_DEPLOYMENT": "High risk — route 5-10% of traffic to new version first, monitor for 30 minutes before full rollout.",
    "STAGED_ROLLOUT": "High risk, wide blast radius — deploy to staging, soak-test, then phased production rollout.",
    "BLOCK_MERGE": "Critical risk or data integrity concern — do not merge until mitigations are in place.",
}

STRATEGY_EMOJI = {
    "DIRECT_MERGE":     "✅",
    "MONITORED_DEPLOY": "👁️",
    "CANARY_DEPLOYMENT":"🐤",
    "STAGED_ROLLOUT":   "🪜",
    "BLOCK_MERGE":      "🚫",
}


@dataclass
class DeploymentAdvice:
    """Full deployment recommendation for a PR."""
    strategy: str                        # one of the 5 strategies above
    confidence: str                      # HIGH / MEDIUM / LOW
    reasons: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)   # must-do before deploy
    monitoring_hints: list[str] = field(default_factory=list)
    estimated_blast_radius: str = ""
    summary: str = ""

    @property
    def emoji(self) -> str:
        return STRATEGY_EMOJI.get(self.strategy, "")

    @property
    def description(self) -> str:
        return STRATEGY_DESCRIPTIONS.get(self.strategy, "")

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "emoji": self.emoji,
            "confidence": self.confidence,
            "description": self.description,
            "reasons": self.reasons,
            "conditions": self.conditions,
            "monitoring_hints": self.monitoring_hints,
            "estimated_blast_radius": self.estimated_blast_radius,
            "summary": self.summary,
        }


class DeploymentAdvisor:
    """
    Decision tree that maps risk signals to a deployment strategy.
    Entirely deterministic — judges can audit every decision.
    """

    def advise(
        self,
        risk_level: str,
        risk_score: float,
        graph: EvidenceGraph,
        chains: list[PropagationChain],
        rollback_advice: dict,
        test_gaps: dict,
        business_impacts: list[str],
        has_test_changes: bool,
    ) -> DeploymentAdvice:

        reasons: list[str] = []
        conditions: list[str] = []
        monitoring: list[str] = []

        # ── Gather signals ─────────────────────────────────────────────────────
        total_affected   = graph.total_files_affected
        max_depth        = graph.max_propagation_depth
        critical_symbols = graph.critical_path_symbols
        has_data_risk    = rollback_advice.get("data_side_effects", False)
        rollback_diff    = rollback_advice.get("rollback_difficulty", "MEDIUM")
        uncovered        = test_gaps.get("uncovered_functions", [])
        chain_risk_max   = max(
            (c.chain_risk_level for c in chains),
            key=lambda r: {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}.get(r, 0),
            default="LOW",
        ) if chains else "LOW"

        critical_domains = {
            "Authentication outage risk",
            "Payment / checkout disruption",
            "Database integrity risk",
        }
        has_critical_domain = bool(set(business_impacts) & critical_domains)

        # ── Build reasons list ─────────────────────────────────────────────────
        if critical_symbols:
            reasons.append(f"Critical path symbols modified: {', '.join(critical_symbols[:3])}")
        if total_affected > 5:
            reasons.append(f"Wide blast radius: {total_affected} files affected")
        if max_depth >= 3:
            reasons.append(f"Deep propagation chain: {max_depth} hops from change origin")
        if has_data_risk:
            reasons.append("Data integrity side effects detected (possible migration)")
        if uncovered:
            fns = ", ".join(u.get("function", "?") for u in uncovered[:2])
            reasons.append(f"Uncovered changed functions: {fns}")
        if has_critical_domain:
            domains = list(set(business_impacts) & critical_domains)
            reasons.append(f"Affects critical business domain: {domains[0]}")
        if rollback_diff == "HIGH":
            reasons.append("High rollback difficulty — recovery is complex if deployment fails")
        if not has_test_changes and uncovered:
            reasons.append("No new tests in PR despite uncovered changed functions")

        # ── Build monitoring hints from affected domains ────────────────────────
        domain_monitors = {
            "Authentication outage risk":     "Monitor auth_failure_rate and 401 error rate",
            "Payment / checkout disruption":  "Monitor payment success rate and checkout completion",
            "API endpoint availability risk": "Monitor 5xx error rate and p99 latency",
            "Session management":             "Monitor session creation errors and token expiry anomalies",
            "Database integrity risk":        "Monitor slow query rate and database error logs",
            "Cache layer disruption":         "Monitor cache hit rate and cache error count",
        }
        for domain in business_impacts:
            hint = domain_monitors.get(domain)
            if hint and hint not in monitoring:
                monitoring.append(hint)

        if not monitoring:
            monitoring.append("Monitor application error rate (5xx) for 30 minutes post-deploy")
            monitoring.append("Set up rollback trigger if error rate increases > 5%")

        # ── Decision tree ──────────────────────────────────────────────────────
        strategy: str
        confidence: str

        if risk_level == "CRITICAL" or has_data_risk:
            strategy = "BLOCK_MERGE"
            confidence = "HIGH"
            conditions.append("Fix all uncovered HIGH-risk test gaps before merging")
            if has_data_risk:
                conditions.append("Review database migration for irreversibility")
                conditions.append("Prepare manual rollback script for data layer")
            conditions.append("Conduct senior engineering review of blast radius")

        elif risk_level == "HIGH" and (has_critical_domain or rollback_diff == "HIGH"):
            strategy = "STAGED_ROLLOUT"
            confidence = "HIGH"
            conditions.append("Deploy to staging environment first")
            conditions.append("Run integration test suite against staging")
            conditions.append("Hold in staging for minimum 1 hour")
            if uncovered:
                conditions.append(f"Add tests for: {', '.join(u.get('function','?') for u in uncovered[:2])}")

        elif risk_level == "HIGH" or (risk_level == "MEDIUM" and chain_risk_max == "HIGH"):
            strategy = "CANARY_DEPLOYMENT"
            confidence = "HIGH" if risk_level == "HIGH" else "MEDIUM"
            conditions.append("Route ≤10% of traffic to new version initially")
            conditions.append("Monitor for 30 minutes before increasing traffic")
            if critical_symbols:
                conditions.append(f"Set alert on error rate for paths using: {critical_symbols[0]}")

        elif risk_level == "MEDIUM" or (risk_level == "LOW" and chain_risk_max in ("MEDIUM", "HIGH")):
            strategy = "MONITORED_DEPLOY"
            confidence = "MEDIUM"
            conditions.append("Ensure rollback procedure is documented before deploying")
            if uncovered:
                conditions.append("Consider adding tests for uncovered functions in follow-up PR")

        else:
            strategy = "DIRECT_MERGE"
            confidence = "HIGH"
            if not reasons:
                reasons.append("Low risk score with no critical path impact")
                reasons.append("Test coverage adequate for changed functions")

        # ── Blast radius summary ───────────────────────────────────────────────
        blast_parts = [f"{total_affected} files affected"]
        if critical_symbols:
            blast_parts.append(f"{len(critical_symbols)} critical path symbol(s)")
        if max_depth > 1:
            blast_parts.append(f"propagation depth {max_depth}")
        blast_summary = ", ".join(blast_parts)

        # ── One-line summary ───────────────────────────────────────────────────
        summary = f"{STRATEGY_EMOJI.get(strategy, '')} {strategy.replace('_', ' ').title()}: {STRATEGY_DESCRIPTIONS[strategy].split('—')[1].strip()}"

        return DeploymentAdvice(
            strategy=strategy,
            confidence=confidence,
            reasons=reasons,
            conditions=conditions,
            monitoring_hints=monitoring,
            estimated_blast_radius=blast_summary,
            summary=summary,
        )


def build_deployment_advice(
    state: dict,
    graph: EvidenceGraph,
    chains: list[PropagationChain],
) -> DeploymentAdvice:
    """Public convenience function called from the orchestrator."""
    advisor = DeploymentAdvisor()
    advice = advisor.advise(
        risk_level=state.get("risk_assessment", {}).get("risk_level", "MEDIUM"),
        risk_score=state.get("risk_assessment", {}).get("overall_risk_score", 5.0),
        graph=graph,
        chains=chains,
        rollback_advice=state.get("rollback_advice", {}),
        test_gaps=state.get("test_gaps", {}),
        business_impacts=state.get("business_impacts", []),
        has_test_changes=state.get("has_test_changes", False),
    )

    # Enrich generic conditions with PR-specific LLM suggestions
    enriched = _llm_enrich_conditions(advice, state, graph, chains)
    if enriched:
        advice.conditions = enriched

    return advice


def _llm_enrich_conditions(
    advice: DeploymentAdvice,
    state: dict,
    graph: EvidenceGraph,
    chains: list[PropagationChain],
) -> list[str] | None:
    """
    Single LLM call to replace generic conditions with PR-specific ones.
    Returns enriched conditions list, or None to keep the deterministic ones.

    Strategy selection is NEVER touched — only the conditions text is improved.
    """
    from rich.console import Console
    console = Console()

    # Skip LLM for low-risk direct merges (not worth the call)
    if advice.strategy == "DIRECT_MERGE":
        return None

    try:
        from config.settings import get_llm
        from langchain_core.messages import SystemMessage, HumanMessage
        import json as _json

        system = """You are a deployment safety engineer. Given a PR's risk profile and a deployment strategy,
generate 3-5 SPECIFIC, ACTIONABLE pre-deploy conditions tailored to THIS PR.

Rules:
- Reference actual file names, function names, and symbols from the PR
- Suggest specific test cases, alerts, or metrics to watch
- Never say generic things like "review the code" or "test thoroughly"
- Each condition should be a concrete action an engineer can complete in <1 hour

Respond with ONLY a JSON array of condition strings. Example:
["Add unit test for `validate_user` returning null when token is expired",
 "Set up Datadog alert on auth.login.error_rate > 2% before deploying"]"""

        # Build a compact context for the LLM
        changed_symbols = state.get("changed_symbols", [])[:5]
        changed_files = [f.split("/")[-1] for f in state.get("changed_files", [])][:5]
        uncovered = [u.get("function", "?") for u in state.get("test_gaps", {}).get("uncovered_functions", [])][:3]
        critical = graph.critical_path_symbols[:3]

        chain_summary = ""
        for c in (chains or [])[:2]:
            chain_summary += f"  {c.arrow_diagram} ({c.chain_risk_level} risk)\n"

        human = f"""PR: {state.get('pr_title', '')}
Strategy: {advice.strategy}
Risk: {state.get('risk_assessment', {}).get('risk_level', '?')} ({state.get('risk_assessment', {}).get('overall_risk_score', '?')}/10)
Changed symbols: {', '.join(changed_symbols) or 'none'}
Changed files: {', '.join(changed_files) or 'none'}
Uncovered functions: {', '.join(uncovered) or 'none'}
Critical path symbols: {', '.join(critical) or 'none'}
Propagation chains:
{chain_summary or '  none'}
Business impacts: {', '.join(state.get('business_impacts', [])[:3]) or 'none'}
Rollback difficulty: {state.get('rollback_advice', {}).get('rollback_difficulty', '?')}

Current generic conditions (replace these with PR-specific ones):
{chr(10).join(f'  - {c}' for c in advice.conditions)}

Generate 3-5 specific, actionable conditions for this exact PR. JSON array only."""

        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=human),
        ])
        text = response.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        conditions = _json.loads(text)

        if isinstance(conditions, list) and 2 <= len(conditions) <= 8:
            console.print(f"  [green]✓[/green] LLM conditions: {len(conditions)} PR-specific conditions generated")
            return conditions
        else:
            console.print(f"  [yellow]⚠ LLM conditions: unexpected format — keeping deterministic conditions[/yellow]")
            return None

    except Exception as e:
        console.print(f"  [yellow]⚠ LLM conditions failed ({str(e)[:60]}) — keeping deterministic conditions[/yellow]")
        return None

