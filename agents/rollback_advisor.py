"""
Rollback Advisor Agent

Purpose: Estimate the difficulty and risks of rolling back this deployment
         if it causes an incident. This is different from risk_level:
         a LOW risk PR can have HIGH rollback difficulty (e.g. DB migrations).

Inputs:  diff, changed_files, runtime_risks, blast_radius
Outputs: rollback_difficulty, rollback_risks, rollback_steps, rollback_summary

Key heuristics (also given to LLM):
  DB migration present         → HIGH (irreversible schema changes)
  Config/env changes only      → LOW  (revert config, redeploy)
  API contract broken          → HIGH (consumers already using new schema)
  Pure addition, no deletions  → LOW  (just revert, nothing depended on it yet)
  Shared library changed       → MEDIUM-HIGH (all consumers need re-deploy too)
  Feature flag controlled      → LOW  (just flip the flag)
"""

from agents.base import run_without_tools, AgentOutput
from agents.schema_validator import validate_and_repair
from tools.context_budget import budget_history
import json

SYSTEM_PROMPT = """You are a Rollback Advisor agent. Your job is to assess rollback complexity.

You are answering one operational question:
  "If this PR is deployed and immediately causes an incident, how hard is it to roll back?"

This is NOT the same as risk probability — a low-risk PR can have HIGH rollback difficulty.

Rollback difficulty classification:
  LOW    — simple git revert + redeploy, < 15 minutes, no data side effects
  MEDIUM — requires coordinated action (notify consumers, staged rollback), 15–60 minutes
  HIGH   — data or schema changes make rollback dangerous or impossible without data loss

Heuristics to apply:
  DB migration files present           → HIGH   (schema changes are often irreversible)
  API response shape changed           → HIGH   (external consumers may have already adapted)
  Feature flag / config only           → LOW    (toggle flag, instant rollback)
  Pure addition (0 deletions)          → LOW    (remove addition, nothing depended on it yet)
  Shared core utility modified         → MEDIUM (all dependent services need coordinated rollback)
  Auth / session changes               → HIGH   (active sessions may be corrupted)
  Background job / queue changes       → MEDIUM (in-flight jobs may be in inconsistent state)
  Test-only changes                    → LOW    (no production impact)

For rollback_steps: be specific and operational, not generic.
  BAD:  "Revert the changes"
  GOOD: "1. git revert <commit>  2. Run: npm run migrate:down  3. Redeploy API service  4. Verify /health endpoint"

Respond with ONLY this JSON (no preamble):
{
  "rollback_difficulty": "LOW|MEDIUM|HIGH",
  "rollback_risks": [
    "specific risk 1 — e.g. active sessions using new token format will be invalidated",
    "specific risk 2"
  ],
  "rollback_steps": [
    "1. git revert <pr-commit>",
    "2. specific step",
    "3. specific verification"
  ],
  "feature_flag_possible": true,
  "data_side_effects": false,
  "rollback_summary": "one-line summary for the report",
  "confidence": 4
}"""


# ── Heuristic pre-classifier (no LLM needed for obvious cases) ────────────────

MIGRATION_PATTERNS = ["migration", "migrate", "schema", "alembic", "flyway", "liquibase", "knex"]
CONFIG_PATTERNS    = ["config", "env", ".yaml", ".yml", ".toml", ".ini", "settings", "feature_flag"]
SESSION_PATTERNS   = ["session", "auth", "token", "jwt", "cookie", "credential"]
QUEUE_PATTERNS     = ["queue", "worker", "celery", "sidekiq", "job", "task", "async"]


def _file_basenames_lower(changed_files: list) -> list[str]:
    return [f.replace("\\", "/").rsplit("/", 1)[-1].lower() for f in changed_files]


def _heuristic_difficulty(changed_files: list, total_deletions: int, runtime_risks: dict) -> str | None:
    """
    Fast heuristic classification before LLM call.
    Returns difficulty string if obvious, None if LLM judgment needed.
    """
    files_lower = [f.lower() for f in changed_files]
    basenames = _file_basenames_lower(changed_files)

    # DB migrations → always HIGH
    if any(any(pat in f for pat in MIGRATION_PATTERNS) for f in files_lower):
        return "HIGH"

    # Auth/session changes → HIGH (match filename only — avoid path substring false positives)
    if any(any(pat in bn for pat in SESSION_PATTERNS) for bn in basenames):
        return "HIGH"

    # Pure addition (nothing deleted) + no breaking change → LOW
    is_breaking = runtime_risks.get("is_breaking_change", True)
    if total_deletions == 0 and not is_breaking:
        return "LOW"

    # Config/env only → LOW
    all_config = all(
        any(pat in f for pat in CONFIG_PATTERNS)
        for f in files_lower
    )
    if all_config and changed_files:
        return "LOW"

    return None  # needs LLM judgment


def _sanitize_rollback_output(data: dict, state: dict) -> dict:
    """
    Clamp LLM hallucinations: PRs without migrations should not
    claim data_side_effects (those drive BLOCK_MERGE downstream).
    """
    if not isinstance(data, dict):
        return data
    changed = state.get("changed_files") or []
    paths_lower = [f.lower() for f in changed]
    basenames = _file_basenames_lower(changed)
    total_del = int(state.get("total_deletions", 0) or 0)

    migrationish = any(any(pat in p for pat in MIGRATION_PATTERNS) for p in paths_lower)
    sessionish = any(any(pat in bn for pat in SESSION_PATTERNS) for bn in basenames)

    # No migration/schema files → data_side_effects is always false
    # This is the #1 cause of false "Block Merge" recommendations
    if not migrationish:
        if data.get("data_side_effects"):
            data["_data_side_effects_cleared"] = True
        data["data_side_effects"] = False

    # Pure addition (0 deletions) + no migrations → low rollback difficulty
    if total_del == 0 and not migrationish:
        if data.get("rollback_difficulty") == "HIGH":
            data["rollback_difficulty"] = "LOW"
            data["_rollback_difficulty_reconciled"] = True

        risks = data.get("rollback_risks")
        if isinstance(risks, list) and not sessionish:
            bad = (
                "session", "authentication", "token format", "credential",
                "jwt", "cookie", "invalidated",
            )

            def _keep(r: object) -> bool:
                if not isinstance(r, str):
                    return True
                rl = r.lower()
                return not any(b in rl for b in bad)

            filtered = [r for r in risks if _keep(r)]
            if len(filtered) < len(risks):
                data["_rollback_risks_filtered"] = True
            if not filtered:
                filtered = [
                    "Revert removes added code paths; no schema or persistence changes "
                    "detected in changed file list.",
                ]
            data["rollback_risks"] = filtered

    return data


def run(state: dict) -> AgentOutput:
    changed_files  = state.get("changed_files", [])
    runtime_risks  = state.get("runtime_risks", {})
    blast_radius   = state.get("blast_radius", {})
    total_deletions = state.get("total_deletions", 0)
    history_text   = budget_history(state.get("repo_history", "No prior history."))

    # Try heuristic first (saves a LLM call for obvious cases)
    heuristic = _heuristic_difficulty(changed_files, total_deletions, runtime_risks)
    heuristic_note = ""
    if heuristic:
        heuristic_note = (
            f"\nHEURISTIC PRE-CLASSIFICATION: {heuristic}\n"
            f"Confirm or override this based on your full analysis.\n"
        )

    # Detect feature gates / feature flags in diff as rollback mechanisms
    raw_diff = state.get("raw_diff", "")
    pr_desc = state.get("pr_description", "")
    feature_gate_note = ""
    import re
    gate_patterns = [
        r"feature[_\-]?gate", r"feature[_\-]?flag", r"FeatureGate",
        r"featuregate\.\w+", r"feature_enabled", r"feature_toggle",
    ]
    combined_text = raw_diff[:5000] + " " + pr_desc
    gate_matches = []
    for pat in gate_patterns:
        gate_matches.extend(re.findall(pat, combined_text, re.IGNORECASE))
    if gate_matches:
        feature_gate_note = f"""
FEATURE GATE DETECTED: The diff contains feature gate/flag references ({', '.join(list(set(gate_matches))[:3])}).
- If a feature gate is introduced, DISABLING the gate may be a faster rollback than git revert.
- Set feature_flag_possible: true if the change can be disabled via a gate/flag.
- Include the gate disable step in rollback_steps.
"""

    human_message = f"""Assess rollback difficulty for this deployment.

PR: {state.get('pr_title', '')}
Description: {(pr_desc or '')[:300]}
Changed files: {json.dumps(changed_files)}
Additions: +{state.get('total_additions', 0)}  Deletions: -{total_deletions}
{heuristic_note}
{feature_gate_note}
=== RUNTIME RISKS ===
{json.dumps(runtime_risks, indent=2)}

=== BLAST RADIUS ===
{json.dumps(blast_radius, indent=2)}

=== REPO HISTORY ===
{history_text}

Apply the heuristics, provide specific rollback_steps (not generic advice),
and respond with ONLY the JSON object."""

    output = run_without_tools(
        system_prompt=SYSTEM_PROMPT,
        human_message=human_message,
        agent_name="rollback_advisor",
    )

    # If LLM failed but we have a heuristic, use it as fallback
    if not output.success and heuristic:
        from rich.console import Console
        Console().print("  [yellow]⚠ Rollback Advisor LLM failed — using heuristic[/yellow]")
        output.success = True
        output.data = {
            "rollback_difficulty": heuristic,
            "rollback_risks": ["Heuristic assessment only — manual review recommended"],
            "rollback_steps": ["git revert <commit>", "Redeploy", "Verify health endpoints"],
            "feature_flag_possible": False,
            "data_side_effects": heuristic == "HIGH",
            "rollback_summary": f"Heuristic: {heuristic} difficulty",
            "confidence": 2,
        }

    if output.success:
        output.data = validate_and_repair(output.data, "rollback_advisor")
        output.data = _sanitize_rollback_output(output.data, state)

    return output
