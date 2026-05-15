"""
Failure Propagation Engine 

Traverses the EvidenceGraph to produce human-readable failure chains.
Turns this:
  blast_radius: [session.py, checkout.py, payment.py]

Into this:
  auth.py → session.py → checkout.py → payment.py
  "Auth validation changed → Session creation affected → Checkout auth may fail → Payment retries may increase"

Design rules (from the GPT proposal — these are correct):
  - Graph traversal is deterministic, no LLM
  - Semantic labels come from path pattern matching (heuristics), no LLM
  - LLM only narrates the final chain (one call, cheap)
  - Max chain length: 4 hops (beyond that, confidence is too low to be useful)

The engine produces PropagationChain objects that the formatter renders
as the visual chain diagram in the report.
"""

from dataclasses import dataclass, field
from typing import Optional
from graph.evidence_graph import EvidenceGraph, SymbolNode, CRITICAL_PATH_FRAGMENTS


# ── Domain label heuristics ────────────────────────────────────────────────────
# Maps file path fragments → human-readable domain labels.
# These are the "semantic labels" the GPT proposal mentions.

DOMAIN_LABELS: list[tuple[list[str], str]] = [
    # Specific business domains first (highest priority)
    (["auth", "login", "oauth", "credential", "password"],  "Auth service"),
    (["session", "token", "jwt", "cookie"],                  "Session management"),
    (["payment", "billing", "stripe", "checkout", "charge"], "Payment processing"),
    (["order", "cart", "basket"],                            "Order management"),
    (["user", "profile", "account", "register"],             "User service"),
    # Infrastructure
    (["router", "route", "middleware", "handler"],           "Request routing"),
    (["database", "db", "migration", "schema", "model"],    "Data layer"),
    (["cache", "redis", "memcache"],                         "Cache layer"),
    (["queue", "worker", "job", "celery"],                   "Background jobs"),
    (["email", "notification", "mailer", "sms"],            "Notification service"),
    (["gateway", "proxy", "load_balancer", "nginx"],         "API gateway"),
    (["api", "controller", "endpoint"],                      "API layer"),
    # Frontend (before generic — 'component'/'view' are frontend-specific)
    (["render", "template", "component", "page", "widget"],  "Frontend rendering"),
    # Extended patterns — catches files heuristics used to miss
    (["core", "foundation", "kernel"],                       "Core module"),
    (["transform", "pipeline", "processor", "engine"],       "Data pipeline"),
    (["cron", "scheduler", "periodic"],                      "Scheduled tasks"),
    (["logger", "logging", "audit", "trace"],                "Observability"),
    (["storage", "s3", "blob", "upload", "file_store"],     "File storage"),
    (["webhook", "callback", "event", "listener"],           "Event system"),
    # Generic patterns last (lowest priority)
    (["config", "settings", "env"],                          "Configuration"),
    (["util", "helper", "common", "shared", "lib"],          "Shared utilities"),
    (["test", "spec", "fixture"],                            "Test suite"),
]

# Risk amplification — some domain transitions amplify risk
AMPLIFYING_TRANSITIONS: dict[tuple[str, str], str] = {
    ("Auth service", "Session management"):   "Session tokens may be corrupted",
    ("Session management", "Payment processing"): "Authenticated payment requests may fail",
    ("Auth service", "Payment processing"):   "Payment auth flow directly affected",
    ("Shared utilities", "Auth service"):     "Auth relies on changed utility",
    ("Data layer", "Auth service"):           "Auth data integrity at risk",
    ("Request routing", "Payment processing"): "Payment routes may be unreachable",
}


def _domain_label(filepath: str) -> str:
    """Assign a human-readable domain label to a file path."""
    lower = filepath.lower()
    for keywords, label in DOMAIN_LABELS:
        if any(kw in lower for kw in keywords):
            return label
    # Fallback: use the directory name
    parts = filepath.replace("\\", "/").split("/")
    if len(parts) >= 2:
        return parts[-2].replace("_", " ").replace("-", " ").title()
    return filepath.split("/")[-1].split(".")[0].title()


def _risk_verb(change_type: str, failure_modes: list[str]) -> str:
    """Generate a risk verb for the propagation narrative."""
    if "NULL_DEREF" in failure_modes:
        return "may cause null reference crash in"
    if "TYPE_ERROR" in failure_modes:
        return "introduces type mismatch in"
    if "MISSING_METHOD" in failure_modes:
        return "removes method relied upon by"
    if "SILENT_WRONG" in failure_modes:
        return "silently changes behavior seen by"
    if "SCHEMA_BREAK" in failure_modes:
        return "breaks API contract consumed by"
    if change_type == "removed":
        return "removes functionality used by"
    if change_type == "added":
        return "adds new dependency on"
    return "affects"


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class PropagationStep:
    """One hop in a failure propagation chain."""
    file: str
    domain_label: str
    depth: int                      # 0 = origin, 1 = direct, 2 = transitive
    confidence: str                 # HIGH / MEDIUM / LOW
    risk_note: str                  # e.g. "Session tokens may be corrupted"
    is_critical: bool


@dataclass
class PropagationChain:
    """
    A complete failure propagation path for one changed symbol.
    Rendered as a visual chain: A → B → C → D
    """
    symbol: str
    steps: list[PropagationStep] = field(default_factory=list)
    narrative: str = ""             # LLM-narrated one-paragraph summary
    max_business_impact: str = ""   # worst downstream business domain
    chain_risk_level: str = "LOW"   # LOW / MEDIUM / HIGH / CRITICAL

    @property
    def depth(self) -> int:
        return len(self.steps) - 1  # origin doesn't count

    @property
    def arrow_diagram(self) -> str:
        """Returns 'AuthService → Session → Checkout → Payment'"""
        return " → ".join(s.domain_label for s in self.steps)

    @property
    def file_chain(self) -> str:
        """Returns 'auth.py → session.py → checkout.py'"""
        return " → ".join(s.file.split("/")[-1] for s in self.steps)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "arrow_diagram": self.arrow_diagram,
            "file_chain": self.file_chain,
            "depth": self.depth,
            "chain_risk_level": self.chain_risk_level,
            "max_business_impact": self.max_business_impact,
            "narrative": self.narrative,
            "steps": [
                {
                    "file": s.file,
                    "domain_label": s.domain_label,
                    "depth": s.depth,
                    "confidence": s.confidence,
                    "risk_note": s.risk_note,
                    "is_critical": s.is_critical,
                }
                for s in self.steps
            ],
        }


# ── Engine ─────────────────────────────────────────────────────────────────────

class PropagationEngine:
    """
    Traverses the EvidenceGraph to produce PropagationChain objects.
    Entirely deterministic — no LLM calls in this class.
    """

    MAX_DEPTH = 4

    def build_chains(
        self,
        graph: EvidenceGraph,
        runtime_risks: dict,
    ) -> list[PropagationChain]:
        """
        Main entry point. Builds one PropagationChain per symbol node.
        Returns chains sorted by depth descending (longest/most impactful first).
        """
        chains: list[PropagationChain] = []

        # Extract failure modes from runtime risks for verb selection
        failure_modes = [
            s.get("failure_mode", "")
            for s in runtime_risks.get("breaking_scenarios", [])
        ]

        for node in graph.nodes:
            chain = self._build_chain_for_node(node, failure_modes)
            if chain:
                chains.append(chain)

        # Sort: critical paths first, then by depth
        chains.sort(key=lambda c: (
            0 if c.chain_risk_level == "CRITICAL" else
            1 if c.chain_risk_level == "HIGH" else
            2 if c.chain_risk_level == "MEDIUM" else 3,
            -c.depth,
        ))

        return chains

    def _build_chain_for_node(
        self,
        node: SymbolNode,
        failure_modes: list[str],
    ) -> Optional[PropagationChain]:
        """Build a single PropagationChain for one SymbolNode."""
        if not node.affected_callers and not node.transitive_dependencies:
            # Symbol has no callers — still show the origin-only chain
            origin_step = PropagationStep(
                file=node.changed_in,
                domain_label=_domain_label(node.changed_in),
                depth=0,
                confidence="HIGH",
                risk_note=f"{node.symbol} was {node.change_type}",
                is_critical=node.has_critical_path,
            )
            return PropagationChain(
                symbol=node.symbol,
                steps=[origin_step],
                chain_risk_level="LOW",
                max_business_impact=_domain_label(node.changed_in),
            )

        # Step 0: origin (the changed file itself)
        steps: list[PropagationStep] = [
            PropagationStep(
                file=node.changed_in,
                domain_label=_domain_label(node.changed_in),
                depth=0,
                confidence="HIGH",
                risk_note=f"{node.symbol} {node.change_type}",
                is_critical=_is_critical_path(node.changed_in),
            )
        ]

        # Step 1+: direct callers (pick most critical one to follow)
        direct_sorted = sorted(
            node.affected_callers,
            key=lambda c: (0 if c.is_critical_path else 1, c.confidence),
        )

        prev_domain = steps[0].domain_label
        for caller in direct_sorted[:self.MAX_DEPTH - 1]:
            domain = _domain_label(caller.file)
            transition_note = AMPLIFYING_TRANSITIONS.get(
                (prev_domain, domain),
                _risk_verb(node.change_type, failure_modes) + f" {domain}"
            )
            steps.append(PropagationStep(
                file=caller.file,
                domain_label=domain,
                depth=len(steps),
                confidence=caller.confidence,
                risk_note=transition_note,
                is_critical=caller.is_critical_path,
            ))
            prev_domain = domain
            if len(steps) >= self.MAX_DEPTH:
                break

        # Step 2+: transitive deps (pick deepest critical path)
        critical_transitive = sorted(
            [t for t in node.transitive_dependencies if _is_critical_path(t.file)],
            key=lambda t: -t.depth,
        )
        for trans in critical_transitive[:1]:
            # Avoid adding a file already in the chain
            existing_files = {s.file for s in steps}
            if trans.file not in existing_files and len(steps) < self.MAX_DEPTH:
                domain = _domain_label(trans.file)
                transition_note = AMPLIFYING_TRANSITIONS.get(
                    (prev_domain, domain),
                    f"propagates to {domain}",
                )
                steps.append(PropagationStep(
                    file=trans.file,
                    domain_label=domain,
                    depth=len(steps),
                    confidence=trans.confidence,
                    risk_note=transition_note,
                    is_critical=_is_critical_path(trans.file),
                ))

        # Compute chain risk level
        chain_risk = _chain_risk(steps, failure_modes)

        # Max business impact = most critical downstream domain
        downstream_domains = [s.domain_label for s in steps[1:]]
        max_impact = downstream_domains[0] if downstream_domains else _domain_label(node.changed_in)

        return PropagationChain(
            symbol=node.symbol,
            steps=steps,
            chain_risk_level=chain_risk,
            max_business_impact=max_impact,
        )


def _is_critical_path(filepath: str) -> bool:
    lower = filepath.lower()
    return any(frag in lower for frag in CRITICAL_PATH_FRAGMENTS)


def _chain_risk(steps: list[PropagationStep], failure_modes: list[str]) -> str:
    """Compute overall chain risk from its steps and failure modes."""
    critical_steps = sum(1 for s in steps if s.is_critical)
    depth = len(steps)

    has_critical_failure = any(
        m in failure_modes for m in ["NULL_DEREF", "TYPE_ERROR", "MISSING_METHOD", "SCHEMA_BREAK"]
    )

    if critical_steps >= 2 and has_critical_failure:
        return "CRITICAL"
    if critical_steps >= 1 and (has_critical_failure or depth >= 3):
        return "HIGH"
    if critical_steps >= 1 or depth >= 2:
        return "MEDIUM"
    return "LOW"


def _narrate_chain_template(chain: PropagationChain) -> str:
    """
    Generate a template-based narrative for a propagation chain.
    Used as the deterministic fallback when LLM narration is unavailable.
    """
    if len(chain.steps) <= 1:
        return f"Change to `{chain.symbol}` is isolated — no downstream propagation detected."

    origin = chain.steps[0]
    final  = chain.steps[-1]
    middle = chain.steps[1:-1]

    parts = [f"`{chain.symbol}` was modified in `{origin.file.split('/')[-1]}`."]

    if middle:
        via = ", ".join(f"`{s.file.split('/')[-1]}`" for s in middle)
        parts.append(f"This propagates through {via}")

    if final.is_critical:
        parts.append(
            f"and reaches `{final.file.split('/')[-1]}` ({final.domain_label}), "
            f"a critical production path. {final.risk_note}."
        )
    else:
        parts.append(f"ending at `{final.file.split('/')[-1]}` ({final.domain_label}).")

    return " ".join(parts)


def _allowed_file_tokens_from_chain(chain: PropagationChain) -> set[str]:
    """Basenames and full paths from chain steps (lowercase) for grounding checks."""
    tok: set[str] = set()
    for s in chain.steps:
        f = s.file.replace("\\", "/")
        tok.add(f.lower())
        tok.add(f.split("/")[-1].lower())
    return tok


def _narrative_only_cites_chain_files(narrative: str, chain: PropagationChain) -> bool:
    """Reject LLM prose that invents filenames not present in the chain steps."""
    import re

    cited = re.findall(
        r"[a-z0-9][a-z0-9_./-]*\.(?:js|mjs|cjs|ts|tsx|jsx|py|java|go|rb)",
        narrative.lower(),
    )
    if not cited:
        return True
    allowed = _allowed_file_tokens_from_chain(chain)
    for c in cited:
        base = c.split("/")[-1]
        if base in allowed or c in allowed:
            continue
        if any(a.endswith(base) or base in a for a in allowed):
            continue
        return False
    return True


def _llm_narrate_chains(
    chains: list[PropagationChain],
    state: dict,
) -> list[str]:
    """
    Batch-narrate all propagation chains in a single LLM call.
    Returns a list of narrative strings, one per chain.
    If the LLM call fails, returns empty list (caller falls back to templates).
    """
    from config.settings import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage
    from rich.console import Console
    import json as _json

    console = Console()

    if not chains:
        return []

    system = """You are a senior software engineer narrating failure propagation chains for a PR review.
For each chain, write ONE concise paragraph (2-3 sentences) explaining:
  1. What was changed and why it matters
  2. Which downstream systems are affected and how
  3. What the concrete risk is (crash, data corruption, silent wrong behavior, etc.)

Be specific and technical. Reference actual file names and function names that appear
in the chain Steps below only — never invent paths or files not listed in the steps.
Do NOT use generic phrases like "this could cause issues". State the specific failure mode.

Respond with ONLY a JSON array of strings, one narrative per chain. Example:
["Narrative for chain 1...", "Narrative for chain 2..."]"""

    chain_descriptions = []
    for i, chain in enumerate(chains):
        desc = (
            f"Chain {i+1}: {chain.symbol} ({chain.chain_risk_level} risk)\n"
            f"  Path: {chain.file_chain}\n"
            f"  Domains: {chain.arrow_diagram}\n"
            f"  Steps:\n"
        )
        for step in chain.steps:
            critical_tag = " [CRITICAL PATH]" if step.is_critical else ""
            desc += f"    {step.depth}. {step.file} ({step.domain_label}){critical_tag}: {step.risk_note}\n"
        chain_descriptions.append(desc)

    human = f"""PR: {state.get('pr_title', '')}
Diff: {state.get('diff_summary', '')}
Files changed: {len(state.get('changed_files', []))} (+{state.get('total_additions', 0)}/-{state.get('total_deletions', 0)})

Narrate these {len(chains)} failure propagation chain(s):

{''.join(chain_descriptions)}

Respond with ONLY a JSON array of {len(chains)} narrative string(s)."""

    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=human),
        ])
        text = response.content.strip()
        # Parse JSON array from response
        text = text.replace("```json", "").replace("```", "").strip()
        narratives = _json.loads(text)
        if isinstance(narratives, list) and len(narratives) == len(chains):
            grounded: list[str] = []
            for chain, nar in zip(chains, narratives):
                if isinstance(nar, str) and _narrative_only_cites_chain_files(nar, chain):
                    grounded.append(nar)
                else:
                    grounded.append(_narrate_chain_template(chain))
                    console.print(
                        "  [dim]Narration fell back to template (invented or missing file paths)[/dim]"
                    )
            console.print(f"  [green]✓[/green] LLM narration: {len(grounded)} chain(s) narrated")
            return grounded
        else:
            console.print(f"  [yellow]⚠ LLM returned {len(narratives) if isinstance(narratives, list) else 'non-list'}, expected {len(chains)} — using template fallback[/yellow]")
            return []
    except Exception as e:
        console.print(f"  [yellow]⚠ LLM narration failed ({str(e)[:60]}) — using template fallback[/yellow]")
        return []


def _dedupe_propagation_chains(chains: list[PropagationChain]) -> list[PropagationChain]:
    """
    Drop chains with identical (file, domain) hop topology. Different changed
    symbols can yield duplicate paths (e.g. same test → shared util chain).
    """
    seen: set[tuple[tuple[str, str], ...]] = set()
    out: list[PropagationChain] = []
    for c in chains:
        sig = tuple((s.file, s.domain_label) for s in c.steps)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
    return out


def build_propagation_chains(
    graph: EvidenceGraph,
    runtime_risks: dict,
    state: dict | None = None,
) -> list[PropagationChain]:
    """
    Public convenience function.
    Builds chains, then narrates them:
      1. Try LLM narration (single call, ~200 tokens) for PR-specific prose
      2. Fall back to template narratives if LLM fails or state is unavailable
    """
    engine = PropagationEngine()
    chains = engine.build_chains(graph, runtime_risks)
    chains = _dedupe_propagation_chains(chains)

    # Always attach template narratives first (free, instant)
    for chain in chains:
        chain.narrative = _narrate_chain_template(chain)

    # Attempt LLM narration if state context is available
    if state and chains:
        llm_narratives = _llm_narrate_chains(chains, state)
        if llm_narratives:
            for chain, narrative in zip(chains, llm_narratives):
                chain.narrative = narrative

    return chains
