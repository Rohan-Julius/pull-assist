"""
Evidence Graph

Transforms the flat blast radius list produced by the Dependency Mapper
into a structured, traversable graph keyed by symbol.

This is the backbone for:
  - Failure Propagation Engine (traverses paths)
  - Explainability (shows evidence per hop)
  - Confidence scoring (confidence degrades over hops)
  - Deployment Advisor (counts affected domains at each depth)

Structure per symbol:
  {
    "symbol": "validate_user",
    "changed_in": "auth.py",
    "change_type": "modified",
    "affected_callers": [
      {"file": "session.py", "line": 42, "reason": "imports validate_user", "confidence": "HIGH"}
    ],
    "transitive_dependencies": [
      {"file": "checkout.py", "path": ["checkout.py", "session.py", "auth.py"], "depth": 2}
    ]
  }

Design decisions:
  - Built deterministically from existing data — no new LLM calls
  - Confidence degrades per hop: HIGH→MEDIUM→LOW (each level of indirection)
  - Maximum depth is 3 hops (beyond that, noise dominates signal)
  - One EvidenceGraph object per PR analysis — lives in state["evidence_graph"]
"""

from dataclasses import dataclass, field
from typing import Optional


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class CallerEdge:
    """A direct caller of a changed symbol."""
    file: str
    line: Optional[int]
    reason: str
    confidence: str          # HIGH / MEDIUM / LOW
    is_critical_path: bool   # auth, payment, core, etc.


@dataclass
class TransitiveDep:
    """A file reachable through one or more hops from the changed symbol."""
    file: str
    path: list[str]          # full chain: [this_file, ..., changed_file]
    depth: int               # 1 = direct, 2 = via one intermediary, etc.
    confidence: str          # degrades per hop


@dataclass
class SymbolNode:
    """
    The core unit of the evidence graph.
    One SymbolNode per changed symbol.
    """
    symbol: str
    changed_in: str          # file where the symbol was changed
    change_type: str         # added / removed / modified
    affected_callers: list[CallerEdge] = field(default_factory=list)
    transitive_dependencies: list[TransitiveDep] = field(default_factory=list)

    @property
    def total_affected_files(self) -> int:
        direct = {c.file for c in self.affected_callers}
        transitive = {t.file for t in self.transitive_dependencies}
        return len(direct | transitive)

    @property
    def max_depth(self) -> int:
        if not self.transitive_dependencies:
            return 1 if self.affected_callers else 0
        return max(t.depth for t in self.transitive_dependencies)

    @property
    def has_critical_path(self) -> bool:
        return any(c.is_critical_path for c in self.affected_callers)


@dataclass
class EvidenceGraph:
    """
    Complete evidence graph for a PR.
    Contains one SymbolNode per changed symbol.
    """
    nodes: list[SymbolNode] = field(default_factory=list)
    repo: str = ""
    pr_number: int = 0

    @property
    def total_symbols_changed(self) -> int:
        return len(self.nodes)

    @property
    def total_files_affected(self) -> int:
        all_files: set[str] = set()
        for node in self.nodes:
            all_files.add(node.changed_in)
            all_files.update(c.file for c in node.affected_callers)
            all_files.update(t.file for t in node.transitive_dependencies)
        return len(all_files)

    @property
    def max_propagation_depth(self) -> int:
        if not self.nodes:
            return 0
        return max(n.max_depth for n in self.nodes)

    @property
    def critical_path_symbols(self) -> list[str]:
        return [n.symbol for n in self.nodes if n.has_critical_path]

    def get_node(self, symbol: str) -> Optional[SymbolNode]:
        return next((n for n in self.nodes if n.symbol == symbol), None)

    def all_affected_files(self) -> set[str]:
        files: set[str] = set()
        for node in self.nodes:
            files.update(c.file for c in node.affected_callers)
            files.update(t.file for t in node.transitive_dependencies)
        return files

    def to_dict(self) -> dict:
        """Serialise for state storage and JSON output."""
        return {
            "repo": self.repo,
            "pr_number": self.pr_number,
            "total_symbols_changed": self.total_symbols_changed,
            "total_files_affected": self.total_files_affected,
            "max_propagation_depth": self.max_propagation_depth,
            "critical_path_symbols": self.critical_path_symbols,
            "nodes": [
                {
                    "symbol": n.symbol,
                    "changed_in": n.changed_in,
                    "change_type": n.change_type,
                    "affected_callers": [
                        {
                            "file": c.file,
                            "line": c.line,
                            "reason": c.reason,
                            "confidence": c.confidence,
                            "is_critical_path": c.is_critical_path,
                        }
                        for c in n.affected_callers
                    ],
                    "transitive_dependencies": [
                        {
                            "file": t.file,
                            "path": t.path,
                            "depth": t.depth,
                            "confidence": t.confidence,
                        }
                        for t in n.transitive_dependencies
                    ],
                }
                for n in self.nodes
            ],
        }


# ── Builder ────────────────────────────────────────────────────────────────────

# Confidence degrades one step per hop
CONFIDENCE_LADDER = {"HIGH": "MEDIUM", "MEDIUM": "LOW", "LOW": "LOW"}

# Paths matching these strings are flagged as critical
CRITICAL_PATH_FRAGMENTS = {
    "auth", "login", "session", "token", "payment", "checkout",
    "billing", "order", "router", "middleware", "core", "base",
    "database", "migration", "security", "admin",
}


def _is_critical(filepath: str) -> bool:
    lower = filepath.lower()
    return any(frag in lower for frag in CRITICAL_PATH_FRAGMENTS)


def _extract_line_number(reason: str) -> Optional[int]:
    """Try to parse a line number from a reason string like 'calls foo at line ~87'."""
    import re
    m = re.search(r"line[^\d]*(\d+)", reason, re.IGNORECASE)
    return int(m.group(1)) if m else None


def build_evidence_graph(
    state: dict,
    blast_radius: dict,
    per_file_context: list,
) -> EvidenceGraph:
    """
    Builds an EvidenceGraph from the Dependency Mapper's blast_radius output
    and the parsed diff's per_file_context.

    This is the key integration point — no new LLM calls needed.
    The blast_radius already contains the caller data; we restructure it
    into a symbol-keyed graph with transitive dependency paths.

    Algorithm:
      1. For each changed symbol, find which direct_dependents reference it
      2. For each direct dependent, check if any indirect_dependent also lists
         that direct dependent as an intermediary (path reconstruction)
      3. Build SymbolNode with both layers
    """
    graph = EvidenceGraph(
        repo=state.get("repo", ""),
        pr_number=state.get("pr_number", 0),
    )

    direct_deps    = blast_radius.get("direct_dependents", [])
    indirect_deps  = blast_radius.get("indirect_dependents", [])
    changed_symbols = state.get("analysis_symbols", state.get("changed_symbols", []))

    # Build a lookup: direct file → its dependents (for transitive path reconstruction)
    # indirect reason strings often mention the direct file they flow through
    indirect_by_reason: dict[str, list[dict]] = {}
    for ind in indirect_deps:
        reason = ind.get("reason", "").lower()
        for direct in direct_deps:
            direct_file = direct.get("file", "")
            if direct_file.lower().split("/")[-1].split(".")[0] in reason:
                indirect_by_reason.setdefault(direct_file, []).append(ind)

    # Map each changed symbol to the files that reference it
    # We infer this from reason strings since the LLM annotates them
    for symbol in changed_symbols:
        # Find which changed file this symbol lives in
        changed_in = ""
        for file_ctx in per_file_context:
            if symbol in file_ctx.get("symbols", []):
                changed_in = file_ctx["path"]
                break

        if not changed_in and per_file_context:
            changed_in = per_file_context[0]["path"]

        # Find change type for this symbol
        change_type = "modified"
        parsed_diff = state.get("_parsed_diff")
        if parsed_diff:
            for f in parsed_diff.changed_files:
                for s in f.changed_symbols:
                    if s.name == symbol:
                        change_type = s.change_type
                        break

        # Build caller edges from direct_dependents that mention this symbol
        caller_edges: list[CallerEdge] = []
        for dep in direct_deps:
            reason = dep.get("reason", "")
            # Include if reason mentions the symbol or if it's a generic import
            if (symbol.lower() in reason.lower()
                    or "import" in reason.lower()
                    or "calls" in reason.lower()
                    or not changed_symbols  # if no symbols, all deps are relevant
            ):
                caller_edges.append(CallerEdge(
                    file=dep["file"],
                    line=_extract_line_number(reason),
                    reason=reason,
                    confidence=dep.get("confidence", "MEDIUM"),
                    is_critical_path=_is_critical(dep["file"]),
                ))

        # Build transitive dependencies with path reconstruction
        transitive: list[TransitiveDep] = []
        seen_transitive: set[str] = set()

        for caller in caller_edges:
            # Any indirect dep that flows through this caller
            for ind in indirect_by_reason.get(caller.file, []):
                ind_file = ind.get("file", "")
                if ind_file and ind_file not in seen_transitive:
                    seen_transitive.add(ind_file)
                    dep_conf = CONFIDENCE_LADDER.get(caller.confidence, "LOW")
                    transitive.append(TransitiveDep(
                        file=ind_file,
                        path=[ind_file, caller.file, changed_in],
                        depth=2,
                        confidence=dep_conf,
                    ))

        # Also check indirect_deps that directly mention the symbol
        for ind in indirect_deps:
            ind_file = ind.get("file", "")
            reason = ind.get("reason", "")
            if (ind_file and ind_file not in seen_transitive
                    and symbol.lower() in reason.lower()):
                seen_transitive.add(ind_file)
                transitive.append(TransitiveDep(
                    file=ind_file,
                    path=[ind_file, changed_in],
                    depth=2,
                    confidence="LOW",
                ))

        node = SymbolNode(
            symbol=symbol,
            changed_in=changed_in,
            change_type=change_type,
            affected_callers=caller_edges,
            transitive_dependencies=transitive,
        )
        graph.nodes.append(node)

    return graph
