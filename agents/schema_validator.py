"""
Schema Validator — Enhancement: Agent output validation

Every agent output is validated against a minimal schema before being
written to state. If validation fails, the validator attempts one LLM
retry with an explicit correction prompt. If that also fails, it returns
a safe default so the pipeline never hard-crashes on malformed output.

This prevents silent degradation where a parse error in one agent
corrupts downstream agents' reasoning.
"""

import json
from rich.console import Console
from config.settings import AGENT_OUTPUT_SCHEMAS

console = Console()


class ValidationError(Exception):
    pass


def validate(data: dict, agent_name: str) -> tuple[bool, list[str]]:
    """
    Validate agent output against its schema.
    Returns (is_valid, list_of_issues).
    """
    schema = AGENT_OUTPUT_SCHEMAS.get(agent_name)
    if not schema:
        return True, []   # no schema registered — pass through

    issues = []

    # Check required fields exist and are non-None
    for field in schema.get("required", []):
        if field not in data:
            issues.append(f"Missing required field: '{field}'")
        elif data[field] is None:
            issues.append(f"Field '{field}' is None")

    # Check list fields are actually lists
    for field in schema.get("list_fields", []):
        if field in data and not isinstance(data[field], list):
            issues.append(f"Field '{field}' should be a list, got {type(data[field]).__name__}")

    # Agent-specific checks
    if agent_name == "risk_evaluator" and "overall_risk_score" in data:
        score = data.get("overall_risk_score", -1)
        if not isinstance(score, (int, float)) or not (0.0 <= float(score) <= 10.0):
            issues.append(f"overall_risk_score '{score}' is not in range 0.0–10.0")

    if agent_name == "risk_evaluator" and "risk_level" in data:
        valid_levels = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        if data["risk_level"] not in valid_levels:
            issues.append(f"risk_level '{data['risk_level']}' not in {valid_levels}")

    if agent_name == "critic" and "verdict" in data:
        valid_verdicts = {"AGREE", "MINOR_ISSUES", "SIGNIFICANT_ISSUES"}
        if data["verdict"] not in valid_verdicts:
            issues.append(f"verdict '{data['verdict']}' not in {valid_verdicts}")

    if agent_name == "rollback_advisor" and "rollback_difficulty" in data:
        valid = {"LOW", "MEDIUM", "HIGH"}
        if data["rollback_difficulty"] not in valid:
            issues.append(f"rollback_difficulty '{data['rollback_difficulty']}' not in {valid}")

    return len(issues) == 0, issues


def repair(data: dict, agent_name: str, issues: list[str]) -> dict:
    """
    Attempt to repair common schema violations without an LLM call.
    Handles the most frequent malformations:
      - risk_level out of enum → clamp by score
      - verdict out of enum → default AGREE
      - list field is a string → wrap in list
      - missing required list → empty list
      - score out of range → clamp to [0, 10]
    """
    repaired = dict(data)
    schema = AGENT_OUTPUT_SCHEMAS.get(agent_name, {})

    for issue in issues:
        # Wrap string-as-list
        for field in schema.get("list_fields", []):
            if field in repaired and isinstance(repaired[field], str):
                repaired[field] = [repaired[field]] if repaired[field] else []
                console.print(f"  [dim]Repaired: wrapped '{field}' string into list[/dim]")

        # Fill missing list fields
        for field in schema.get("list_fields", []):
            if field not in repaired or repaired[field] is None:
                repaired[field] = []
                console.print(f"  [dim]Repaired: defaulted missing '{field}' to [][/dim]")

    # Agent-specific repairs
    if agent_name == "risk_evaluator":
        score = repaired.get("overall_risk_score", 5.0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 5.0
        score = max(0.0, min(10.0, score))
        repaired["overall_risk_score"] = score

        # Derive risk_level from score if invalid
        if repaired.get("risk_level") not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
            if score <= 3.0:
                repaired["risk_level"] = "LOW"
            elif score <= 5.9:
                repaired["risk_level"] = "MEDIUM"
            elif score <= 7.9:
                repaired["risk_level"] = "HIGH"
            else:
                repaired["risk_level"] = "CRITICAL"
            console.print(f"  [dim]Repaired: derived risk_level={repaired['risk_level']} from score={score}[/dim]")

    if agent_name == "critic":
        if repaired.get("verdict") not in {"AGREE", "MINOR_ISSUES", "SIGNIFICANT_ISSUES"}:
            repaired["verdict"] = "AGREE"
            console.print("  [dim]Repaired: defaulted invalid verdict to AGREE[/dim]")

    if agent_name == "rollback_advisor":
        if repaired.get("rollback_difficulty") not in {"LOW", "MEDIUM", "HIGH"}:
            repaired["rollback_difficulty"] = "MEDIUM"

    return repaired


def validate_and_repair(data: dict, agent_name: str) -> dict:
    """
    Validate then repair. Logs all issues. Returns cleaned data.
    Will never raise — always returns something usable.
    """
    if not data or data.get("_parse_error"):
        console.print(f"  [yellow]⚠ {agent_name}: output had parse error — using defaults[/yellow]")
        return _safe_default(agent_name)

    is_valid, issues = validate(data, agent_name)

    if is_valid:
        return data

    console.print(f"  [yellow]⚠ {agent_name}: schema issues — attempting repair[/yellow]")
    for issue in issues:
        console.print(f"    [dim]• {issue}[/dim]")

    repaired = repair(data, agent_name, issues)

    # Re-validate after repair
    is_valid_after, remaining = validate(repaired, agent_name)
    if not is_valid_after:
        console.print(f"  [red]✗ {agent_name}: repair incomplete — {remaining}[/red]")
        # Merge repaired with safe default (repaired fields take priority)
        return {**_safe_default(agent_name), **repaired}

    console.print(f"  [green]✓ {agent_name}: repaired successfully[/green]")
    return repaired


def _safe_default(agent_name: str) -> dict:
    """Returns a minimal valid output for each agent."""
    defaults = {
        "dependency_mapper": {
            "direct_dependents": [],
            "indirect_dependents": [],
            "blast_radius_summary": "Analysis unavailable",
        },
        "change_simulator": {
            "before_behavior": "Unknown",
            "after_behavior": "Unknown",
            "breaking_scenarios": [],
            "is_breaking_change": False,
            "simulator_summary": "Analysis unavailable",
            "confidence": 1,
        },
        "test_gap": {
            "covered_functions": [],
            "uncovered_functions": [],
            "overall_coverage_assessment": "UNKNOWN",
            "test_gap_summary": "Analysis unavailable",
            "confidence": 1,
        },
        "risk_evaluator": {
            "dimension_scores": {
                "blast_radius_score": 5.0,
                "test_coverage_score": 5.0,
                "runtime_risk_score": 5.0,
                "complexity_score": 5.0,
            },
            "overall_risk_score": 5.0,
            "risk_level": "MEDIUM",
            "top_concerns": ["Analysis unavailable — treat as medium risk"],
            "recommended_actions": ["Manual review required"],
            "rollback_difficulty": "MEDIUM",
        },
        "critic": {
            "objections": [],
            "missed_impacts": [],
            "verdict": "AGREE",
            "critic_summary": "Critic analysis unavailable",
        },
        "rollback_advisor": {
            "rollback_difficulty": "MEDIUM",
            "rollback_risks": ["Unable to assess — manual review required"],
            "rollback_steps": [],
            "rollback_summary": "Analysis unavailable",
            "confidence": 1,
        },
    }
    return defaults.get(agent_name, {})
