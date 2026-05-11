"""
Business Impact Analyzer — Enhancement 3

This is NOT a full LangGraph agent (no tool calls, no state node).
It runs as a post-processing step after all agents complete.

Design decision: most business impact classification is deterministic
(file path patterns → domain labels). Only the summary sentence needs
an LLM call. This keeps it fast and cheap.

Output: { "business_impacts": [str], "impact_summary": str, "severity_domains": [str] }
"""

import re
from config.settings import BUSINESS_IMPACT_PATTERNS

# ── Deterministic classifier ───────────────────────────────────────────────────

def classify_from_paths(changed_files: list, blast_radius: dict) -> list[str]:
    """
    Map file paths → business impact labels using BUSINESS_IMPACT_PATTERNS.
    No LLM call — pure pattern matching.
    Returns deduplicated list of impact labels.
    """
    all_paths = list(changed_files)

    # Also include blast radius paths (they're affected even if not directly changed)
    for dep in blast_radius.get("direct_dependents", [])[:5]:
        all_paths.append(dep.get("file", ""))
    for dep in blast_radius.get("indirect_dependents", [])[:3]:
        all_paths.append(dep.get("file", ""))

    found_impacts: list[str] = []
    seen: set[str] = set()

    for path in all_paths:
        path_lower = path.lower()

        # Skip test/spec files — they don't cause business impact
        # Check directory segments AND filename patterns
        segments = set(path_lower.replace("\\", "/").split("/"))
        is_test = (
            segments & {"test", "tests", "spec", "specs", "__tests__", "fixtures", "mocks"}
            or any(p in path_lower for p in ("test_", "_test.", ".test.", ".spec."))
        )
        if is_test:
            continue

        for keywords, label in BUSINESS_IMPACT_PATTERNS:
            if label is None:
                continue   # skip test file patterns
            if any(kw in path_lower for kw in keywords):
                if label not in seen:
                    found_impacts.append(label)
                    seen.add(label)

    return found_impacts


def classify_from_runtime_risks(runtime_risks: dict) -> list[str]:
    """
    Extract additional business impacts from failure_mode descriptions.
    """
    impacts = []
    seen = set()

    for scenario in runtime_risks.get("breaking_scenarios", []):
        mode = scenario.get("failure_mode", "").lower()
        desc = scenario.get("failure_description", "").lower()
        combined = mode + " " + desc

        for keywords, label in BUSINESS_IMPACT_PATTERNS:
            if label is None:
                continue
            if any(kw in combined for kw in keywords):
                if label not in seen:
                    impacts.append(label)
                    seen.add(label)

    return impacts


def build_impact_summary(impacts: list[str], risk_level: str) -> str:
    """
    Build a human-readable impact summary sentence without LLM.
    Keeps things fast — judges see this in < 1 second.
    """
    if not impacts:
        return "No business-critical domains identified in affected files."

    if len(impacts) == 1:
        return f"{impacts[0]} — {risk_level} deployment risk."

    primary = impacts[0]
    others = impacts[1:3]
    others_str = " and ".join(others)
    return f"{primary} is the primary concern. Also affects: {others_str}. Overall risk: {risk_level}."


def analyze(state: dict) -> dict:
    """
    Main entry point. Called after all agents complete, before report build.

    Returns a dict to be merged into state:
      {
        "business_impacts": [str],   # list of domain impact strings
        "impact_summary": str,       # one-sentence summary
        "severity_domains": [str],   # high-severity domains only
      }
    """
    changed_files  = state.get("changed_files", [])
    blast_radius   = state.get("blast_radius", {})
    runtime_risks  = state.get("runtime_risks", {})
    risk_level     = state.get("risk_assessment", {}).get("risk_level", "MEDIUM")

    path_impacts    = classify_from_paths(changed_files, blast_radius)
    runtime_impacts = classify_from_runtime_risks(runtime_risks)

    # Merge, deduplicate, path-derived impacts first
    seen: set[str] = set()
    all_impacts: list[str] = []
    for impact in path_impacts + runtime_impacts:
        if impact not in seen:
            all_impacts.append(impact)
            seen.add(impact)

    # Severity domains = impacts that are in critical areas
    critical_domains = {
        "Authentication outage risk",
        "Payment / checkout disruption",
        "Database integrity risk",
        "API endpoint availability risk",
    }
    severity_domains = [i for i in all_impacts if i in critical_domains]

    summary = build_impact_summary(all_impacts, risk_level)

    return {
        "business_impacts":  all_impacts,
        "impact_summary":    summary,
        "severity_domains":  severity_domains,
    }
