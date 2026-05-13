import os
from dotenv import load_dotenv

load_dotenv()
from langchain_openai import ChatOpenAI

def get_llm():
    """
    Returns a LangChain LLM pointed at your vLLM endpoint.
    vLLM exposes an OpenAI-compatible API so ChatOpenAI works out of the box.
    """
    import logging
    logger = logging.getLogger("pull-assist")
    logger.info(f"get_llm() → base_url={LLM_BASE_URL}  model={LLM_MODEL}  api_key={LLM_API_KEY[:8]}...")
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        request_timeout=90,          # prevent infinite hangs
    )
# ── GitHub ────────────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_API_BASE = "https://api.github.com"
GITHUB_SEARCH_MAX_RESULTS = 10      # max files returned by symbol search
GITHUB_FETCH_MAX_LINES = 200        # max lines returned from a single file fetch
GITHUB_CONTEXT_WINDOW_LINES = 100   # lines of context around a symbol (±50)
GITHUB_MAX_TOOL_CALLS_PER_AGENT = 3 # hard cap on API calls per agent per run

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-Coder-V2-Instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")
# Completion cap per request. Must leave room in vLLM's --max-model-len for
# the prompt, tool definitions, and agent scratchpad (often 2.5k–4k+ tokens).
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE = 0.1               # low temp for deterministic code analysis
# When false, use prompt-based tools (no --enable-auto-tool-choice on vLLM).
USE_NATIVE_TOOL_CALLING = os.getenv(
    "USE_NATIVE_TOOL_CALLING", "true"
).lower() in ("1", "true", "yes")

# ── Memory ────────────────────────────────────────────────────────────────────
MEMORY_DB_PATH = "memory/pr_history.db"
MEMORY_MAX_HISTORY_ENTRIES = 20     # max past PRs to inject as context

# ── Risk scoring ──────────────────────────────────────────────────────────────
RISK_WEIGHTS = {
    "blast_radius_files": 0.30,     # how many files are affected
    "test_gap_coverage": 0.30,      # uncovered changed paths
    "runtime_breakage": 0.25,       # severity of potential breakage
    "change_complexity": 0.15,      # size/complexity of the diff itself
}
RISK_CONFLICT_THRESHOLD = 2.0       # if Critic raises score by this much, re-run

# ── Diff parsing ──────────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rb": "ruby",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "csharp",
    ".php": "php",
    ".kt": "kotlin",
    ".swift": "swift",
}

# Symbol extraction regex patterns per language (used by diff_parser)
SYMBOL_PATTERNS = {
    "python":     [r"^def\s+(\w+)", r"^class\s+(\w+)"],
    "javascript": [r"function\s+(\w+)", r"const\s+(\w+)\s*=.*(?:=>|function)", r"class\s+(\w+)"],
    "typescript": [r"function\s+(\w+)", r"const\s+(\w+)\s*=.*(?:=>|function)", r"class\s+(\w+)", r"interface\s+(\w+)"],
    "java":       [r"(?:public|private|protected).*\s+(\w+)\s*\(", r"class\s+(\w+)"],
    "go":         [r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)", r"type\s+(\w+)\s+struct"],
    "ruby":       [r"def\s+(\w+)", r"class\s+(\w+)"],
    "rust":       [r"fn\s+(\w+)", r"struct\s+(\w+)", r"impl\s+(\w+)"],
}

# ── Agent prompts ─────────────────────────────────────────────────────────────
# All prompts live here so you can iterate without hunting through agent files.

DEPENDENCY_MAPPER_PROMPT = """You are a Dependency Mapper agent. Your job is to identify which parts
of the codebase are affected by the following code change.

CHANGED FILES AND SYMBOLS:
{changed_symbols}

SYMBOL SEARCH RESULTS FROM REPO:
{search_results}

REPO HISTORY CONTEXT:
{repo_history}

Based on the search results, identify:
1. Which files directly import or call the changed symbols (direct dependents)
2. Which files are likely transitively affected (indirect dependents)
3. Your confidence level for each (HIGH / MEDIUM / LOW)

Respond in this exact JSON format:
{{
  "direct_dependents": [
    {{"file": "path/to/file.py", "reason": "calls getUserById on line ~45", "confidence": "HIGH"}}
  ],
  "indirect_dependents": [
    {{"file": "path/to/file.py", "reason": "depends on orderService which calls getUserById", "confidence": "MEDIUM"}}
  ],
  "blast_radius_summary": "Brief summary of overall impact scope"
}}"""

CHANGE_SIMULATOR_PROMPT = """You are a Change Simulator agent. Your job is to reason about how
a code change will behave at runtime, and what could break.

CHANGED CODE (diff):
{diff_content}

CALLER FILE CONTEXT (from main branch):
{caller_context}

Think step by step:
1. What is the BEFORE behavior of the changed code?
2. What is the AFTER behavior?
3. Which callers make assumptions about the old behavior that would now break?
4. What is the most likely failure mode (TypeError, wrong return value, silent bug, etc.)?

Respond in this exact JSON format:
{{
  "before_behavior": "description of old behavior",
  "after_behavior": "description of new behavior",
  "breaking_scenarios": [
    {{
      "caller_file": "path/to/caller.py",
      "line_approx": 45,
      "failure_mode": "TypeError: Cannot read property X of null",
      "severity": "HIGH"
    }}
  ],
  "is_breaking_change": true,
  "simulator_summary": "Brief summary"
}}"""

TEST_GAP_AGENT_PROMPT = """You are a Test Gap Agent. Your job is to identify which changed code
paths have no corresponding test coverage.

CHANGED FUNCTIONS/METHODS:
{changed_symbols}

EXISTING TEST FILE CONTENT:
{test_file_content}

DIFF SUMMARY:
{diff_summary}

Analyze and identify:
1. Which changed functions have existing tests?
2. Which changed functions have NO tests?
3. Which NEW behaviors introduced by the diff have no test scenarios?

Respond in this exact JSON format:
{{
  "covered_functions": ["functionA", "functionB"],
  "uncovered_functions": [
    {{
      "function": "getUserById",
      "missing_scenario": "no test for null return case when user not found",
      "risk": "HIGH"
    }}
  ],
  "overall_coverage_assessment": "POOR / PARTIAL / ADEQUATE",
  "test_gap_summary": "Brief summary"
}}"""

RISK_EVALUATOR_PROMPT = """You are a Risk Evaluator agent. Given findings from other agents,
produce a final risk assessment for this PR.

BLAST RADIUS (from Dependency Mapper):
{blast_radius}

RUNTIME RISKS (from Change Simulator):
{runtime_risks}

TEST GAPS (from Test Gap Agent):
{test_gaps}

DIFF STATS:
{diff_stats}

Score this PR on a 0.0–10.0 risk scale across these dimensions:
- blast_radius_score (0–10): how many files/services are affected
- test_coverage_score (0–10): how much of the change is untested
- runtime_risk_score (0–10): severity of potential runtime failures
- complexity_score (0–10): size and complexity of the change

Respond in this exact JSON format:
{{
  "dimension_scores": {{
    "blast_radius_score": 7.0,
    "test_coverage_score": 8.5,
    "runtime_risk_score": 6.0,
    "complexity_score": 4.0
  }},
  "overall_risk_score": 7.2,
  "risk_level": "HIGH",
  "top_concerns": ["concern 1", "concern 2", "concern 3"],
  "recommended_actions": ["action 1", "action 2"],
  "rollback_difficulty": "EASY / MEDIUM / HARD"
}}"""

CRITIC_AGENT_PROMPT = """You are a Critic agent. Your job is to challenge the findings of other
agents, identify inconsistencies, and flag anything that seems understated or missed.

DEPENDENCY MAPPER OUTPUT:
{blast_radius}

CHANGE SIMULATOR OUTPUT:
{runtime_risks}

TEST GAP AGENT OUTPUT:
{test_gaps}

RISK EVALUATOR OUTPUT:
{risk_assessment}

ORIGINAL DIFF:
{diff_summary}

Look for:
1. Inconsistencies between agents (e.g. Dependency Mapper found 10 files but Risk Evaluator scored blast radius LOW)
2. Things that seem understated given the diff content
3. Important impacts none of the agents mentioned
4. Cases where a score seems too high or too low

Respond in this exact JSON format:
{{
  "objections": [
    {{
      "target_agent": "risk_evaluator",
      "claim": "blast_radius_score of 3.0 is too low",
      "reason": "Dependency Mapper found 8 direct dependents but score implies minimal impact",
      "suggested_correction": "blast_radius_score should be 7.0 or higher"
    }}
  ],
  "missed_impacts": ["anything important none of the agents mentioned"],
  "verdict": "AGREE / MINOR_ISSUES / SIGNIFICANT_ISSUES",
  "critic_summary": "Overall assessment of analysis quality"
}}"""


# ── Enhancement: Business impact path patterns ────────────────────────────────
# Maps file path fragments → business domain labels.
# Used by the business impact analyzer (no LLM needed for classification).
BUSINESS_IMPACT_PATTERNS = [
    (["auth", "login", "session", "token", "jwt", "oauth", "password", "credential"], "Authentication outage risk"),
    (["payment", "checkout", "billing", "stripe", "invoice", "charge", "order"],      "Payment / checkout disruption"),
    (["user", "profile", "account", "signup", "register"],                             "User account service impact"),
    (["admin", "dashboard", "backoffice", "management"],                               "Admin interface instability"),
    (["search", "index", "elastic", "solr", "query"],                                  "Search functionality degradation"),
    (["notification", "email", "sms", "push", "alert", "mailer"],                     "Notification delivery failure"),
    (["upload", "storage", "s3", "blob", "file", "media"],                             "File / media service impact"),
    (["cache", "redis", "memcache", "cdn"],                                            "Cache layer disruption"),
    (["database", "migration", "schema", "db", "sql", "mongo", "postgres"],           "Database integrity risk"),
    (["api", "route", "endpoint", "controller", "handler", "middleware"],             "API endpoint availability risk"),
    (["config", "env", "settings", "feature_flag", "toggle"],                        "Configuration instability"),
    (["worker", "queue", "job", "celery", "sidekiq", "async", "task"],               "Background job disruption"),
    (["report", "analytics", "metric", "track", "event"],                             "Analytics / reporting impact"),
    (["test", "spec", "fixture"],                                                      None),  # test files — no business impact
]

# ── Enhancement: Agent confidence threshold ───────────────────────────────────
# If an agent self-reports confidence < this, the Critic is automatically
# primed to scrutinise that agent's output more closely.
AGENT_CONFIDENCE_THRESHOLD = 3   # on a 1–5 scale

# ── Enhancement: Severity-weighted blast radius file patterns ─────────────────
# Files matching these patterns get their blast radius confidence upgraded.
HIGH_SEVERITY_PATH_PATTERNS = [
    "auth", "payment", "billing", "checkout", "security",
    "middleware", "router", "core", "base", "utils", "helpers",
    "config", "settings", "db", "database", "migration",
]

# ── Enhancement: Agent output schema definitions ──────────────────────────────
# Used by the schema validator to catch malformed agent JSON.
AGENT_OUTPUT_SCHEMAS = {
    "dependency_mapper": {
        "required": ["direct_dependents", "indirect_dependents", "blast_radius_summary"],
        "list_fields": ["direct_dependents", "indirect_dependents"],
    },
    "change_simulator": {
        "required": ["before_behavior", "after_behavior", "breaking_scenarios", "is_breaking_change", "simulator_summary"],
        "list_fields": ["breaking_scenarios"],
    },
    "test_gap": {
        "required": ["covered_functions", "uncovered_functions", "overall_coverage_assessment", "test_gap_summary"],
        "list_fields": ["uncovered_functions", "covered_functions"],
    },
    "risk_evaluator": {
        "required": ["dimension_scores", "overall_risk_score", "risk_level", "top_concerns", "recommended_actions"],
        "list_fields": ["top_concerns", "recommended_actions"],
    },
    "critic": {
        "required": ["objections", "verdict", "critic_summary"],
        "list_fields": ["objections", "missed_impacts"],
    },
    "rollback_advisor": {
        "required": ["rollback_difficulty", "rollback_risks", "rollback_steps", "rollback_summary"],
        "list_fields": ["rollback_risks", "rollback_steps"],
    },
}