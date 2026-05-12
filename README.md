# pull-assist

An AI-powered PR impact analysis system that uses multi-agent reasoning to assess risk, predict failures, and recommend deployment strategies.

**Status:** Analyzes GitHub PRs with LLM-driven agents, deterministic graph analysis, and conflict resolution loops.

## Features

- **Multi-Agent Analysis** — 7 specialized agents analyze different aspects of your code change
- **Blast Radius Mapping** — Identifies which files are affected by changed symbols (functions, classes, etc.)
- **Runtime Risk Prediction** — Simulates code changes to predict breaking scenarios and failure modes
- **Test Gap Detection** — Flags changed functions with missing test coverage
- **Business Impact Analysis** — Maps technical changes to business domains (auth, payment, etc.)
- **Rollback Advisor** — Assesses rollback complexity and provides recovery steps
- **Evidence Graph** — Structures dependencies into traversable graphs with confidence scoring
- **Failure Propagation** — Narrates how failures cascade through the codebase
- **Deployment Strategy** — Recommends canary, staged, or direct deployment based on risk
- **Conflict Resolution** — Re-runs agents when the Critic detects inconsistencies
- **Historical Context** — Learns from past PRs to detect risky patterns
- **Dual Output Formats** — JSON for machines, Markdown for humans

## Quick Start

### Installation

```bash
git clone https://github.com/Rohan-Julius/pull-assist.git
cd pull-assist
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file:

```env
GITHUB_TOKEN=ghp_...                    # GitHub personal access token
LLM_BASE_URL=http://localhost:8000/v1   # vLLM or OpenAI endpoint
LLM_MODEL=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
LLM_API_KEY=not-needed                  # For local vLLM
```

### Analyze a PR

```bash
# Analyze a GitHub PR by URL
python main.py --pr https://github.com/expressjs/express/pull/7171

# Analyze with verbose output
python main.py --pr https://github.com/expressjs/express/pull/7171 --verbose

# Analyze local diff + remote repo
python main.py --diff changes.patch --repo owner/repo

# Analyze local diff + local repo (fully offline)
python main.py --diff changes.patch --local-repo /path/to/repo

# Run only data pipeline (no agents)
python main.py --pr URL --day1-only
```

**Output:** Generated in `reports/`

- `pr-XXXX-report.md` — Formatted human-readable report
- `pr-XXXX-report.json` — Structured data for integration

## Architecture

### Agent Pipeline (in order)

1. **Dependency Mapper** — Finds which files import/call changed symbols
   - Tools: GitHub symbol search, file tree exploration
   - Output: Flat blast radius (direct & indirect dependents)

2. **Change Simulator** — Predicts runtime failures and breaking scenarios
   - Evidence-based; only reports breakage with proof
   - Output: Runtime risks, failure modes, severity

3. **Test Gap Agent** — Identifies test coverage gaps in changed code
   - Searches for existing tests, assesses coverage
   - Output: Covered/uncovered functions, overall assessment

4. **Rollback Advisor** — Assesses rollback difficulty and recovery complexity
   - Detects DB migrations, API contracts, config-only changes
   - Output: Rollback difficulty, steps, risks

5. **Business Impact + Risk Evaluator** — Classifies business domains and scores overall risk
   - Deterministic business domain classification
   - Weighted scoring: blast radius (30%) + test gaps (30%) + runtime risk (25%) + complexity (15%)
   - Output: Risk score (0–10), risk level (LOW/MEDIUM/HIGH/CRITICAL), business impacts

6. **Critic Agent** — Challenges all findings, detects inconsistencies
   - Finds understated risks and missed impacts
   - Returns: AGREE / MINOR_ISSUES / SIGNIFICANT_ISSUES
   - Triggers conflict resolution if needed

7. **Rerun with Objections** (conditional) — Re-runs flagged agents with Critic feedback
   - Max 2 rounds; stops if score converges
   - Re-scores risk after objection resolution

### Graph Layer

Three deterministic components transform raw dependency data into strategic recommendations:

**Evidence Graph** — Structures dependencies by symbol with full propagation paths

- Direct callers with line numbers and confidence scores
- Transitive dependencies with depth and confidence degradation
- Critical path detection (auth, payment, core modules)
- Max propagation depth: 3 hops

**Failure Propagation Engine** — Converts the graph into human-readable failure chains

- Domain labeling (Auth service, Payment processing, Data layer, etc.)
- Risk amplification for dangerous transitions
- Semantic risk narratives ("Session tokens may be corrupted", etc.)
- Example: `auth.py → session.py → checkout.py` becomes "Auth validation changed → Session creation affected → Checkout auth may fail"

**Deployment Advisor** — Recommends deployment strategy

- **DIRECT_MERGE** ✅ — Low risk, standard deploy
- **MONITORED_DEPLOY** 👁️ — Medium risk, enhanced monitoring
- **CANARY_DEPLOYMENT** 🐤 — High risk, route 5–10% traffic first
- **STAGED_ROLLOUT** 📋 — High risk + wide blast, soak-test first
- **BLOCK_MERGE** 🚫 — Critical risk, do not merge

## Report Contents

### Example Output

```json
{
  "pr_number": 7171,
  "pr_title": "Express Router API change",
  "pr_author": "user@github.com",
  "overall_risk_score": 7.8,
  "risk_level": "HIGH",
  "business_impacts": [
    "Request routing failures may cause API downtime",
    "Middleware execution order change affects all routes"
  ],
  "severity_domains": ["Request routing", "API layer"],
  "blast_radius": {
    "direct_dependents": [
      { "file": "src/router.js", "confidence": "HIGH" },
      { "file": "src/middleware.js", "confidence": "MEDIUM" }
    ],
    "indirect_dependents": [{ "file": "src/app.js", "confidence": "MEDIUM" }]
  },
  "test_gaps": {
    "uncovered_functions": [
      {
        "function": "Router.use",
        "missing_scenario": "nested middleware order"
      }
    ]
  },
  "propagation_chains": [
    {
      "origin": "router.js",
      "steps": [
        { "file": "middleware.js", "domain": "Request routing" },
        { "file": "app.js", "domain": "API layer" }
      ]
    }
  ],
  "deployment_strategy": "CANARY_DEPLOYMENT",
  "rollback_difficulty": "MEDIUM",
  "rollback_steps": [
    "Revert src/router.js to main branch",
    "Redeploy application",
    "Monitor error rates for 5 minutes"
  ]
}
```

### Markdown Report Sections

- **Risk Overview** — Score, level, business impacts
- **Blast Radius** — Direct & indirect affected files
- **Runtime Risks** — Predicted breaking scenarios
- **Test Coverage Gaps** — Missing tests with risk assessment
- **Business Impact** — Domain classification with severity
- **Rollback Advice** — Difficulty, steps, risks
- **Propagation Chains** — How failures cascade
- **Deployment Strategy** — Recommended approach + monitoring hints
- **Historical Context** — Past risks, recurring patterns
- **Conflict Log** — Re-run rounds and score evolution

## Configuration

Edit `config/settings.py`:

```python
# Risk scoring weights
RISK_WEIGHTS = {
    "blast_radius_files": 0.30,
    "test_gap_coverage": 0.30,
    "runtime_breakage": 0.25,
    "change_complexity": 0.15,
}

# LLM model and endpoint
LLM_MODEL = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
LLM_BASE_URL = "http://localhost:8000/v1"

# Memory store
MEMORY_DB_PATH = "memory/pr_history.db"
MEMORY_MAX_HISTORY_ENTRIES = 20

# GitHub API
GITHUB_SEARCH_MAX_RESULTS = 10
GITHUB_MAX_TOOL_CALLS_PER_AGENT = 3
```

## Development

### Run Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test suite
python -m pytest tests/test_agents.py -v
python -m pytest tests/test_diff_parser.py -v
python -m pytest tests/test_graph_layer.py -v

# Skip slow GitHub tests
python -m pytest tests/ -v -k "not invalid_token"
```

### Project Structure

```
pull-assist/
├── agents/                 # 7 agent implementations
│   ├── dependency_mapper.py
│   ├── change_simulator.py
│   ├── test_gap.py
│   ├── rollback_advisor.py
│   ├── risk_evaluator.py
│   ├── business_impact.py
│   ├── critic.py
│   └── orchestrator.py     # LangGraph StateGraph + pipeline
├── graph/                  # Graph layer
│   ├── evidence_graph.py
│   ├── propagation_engine.py
│   └── deployment_advisor.py
├── github/                 # GitHub API integration
│   ├── client.py
│   └── diff_parser.py
├── memory/                 # Historical context store
│   ├── store.py
│   └── schema.py
├── output/                 # Report generation
│   ├── report_builder.py
│   └── formatter.py
├── tools/                  # LLM tool definitions
│   ├── github_tools.py
│   └── context_budget.py
├── config/                 # Settings
│   └── settings.py
├── tests/                  # Test suite
├── main.py                 # Entry point
└── demo.py                 # Side-by-side PR comparison
```

### Key Design Decisions

- **SQLite for history** — Zero-config, no external DB needed. Swap to Postgres later if scaling.
- **Deterministic graph layer** — No LLM calls for propagation/strategy; keeps it fast and explainable.
- **LLM calls for narration** — Only agents call LLM; graph layer uses heuristics.
- **Conflict resolution loop** — Re-runs agents on SIGNIFICANT_ISSUES up to 2 rounds with score convergence guard.
- **Language-agnostic** — Symbol extraction supports Python, JavaScript, TypeScript, Java, Go, Ruby, Rust, more.

## Supported Languages

Symbol extraction for: Python, JavaScript, TypeScript, Java, Go, Ruby, Rust, C++, C, C#, PHP, Kotlin, Swift

## Performance

- **Typical PR analysis:** 30–60 seconds (4–7 LLM calls + GitHub searches)
- **Memory footprint:** ~200MB (SQLite + LLM context)
- **Max diff size:** 10k additions/deletions (budget constraints)

## Troubleshooting

**"ModuleNotFoundError: No module named 'langgraph'"**

```bash
pip install -r requirements.txt
```

**"Invalid GitHub token"**

- Verify `GITHUB_TOKEN` in `.env`
- Ensure token has `repo` scope

**"LLM connection refused"**

- Check `LLM_BASE_URL` (default: `http://localhost:8000/v1`)
- Ensure vLLM server is running (or set `--day1-only` to skip agents)

**"Could not resolve symbol in repo"**

- GitHub search has limits; some symbols may not be found
- Agent continues with LOW confidence

## Contributing

Contributions welcome! Focus areas:

- [ ] Additional symbol extraction patterns
- [ ] More business impact domains
- [ ] Risk scoring refinements
- [ ] Deployment strategy heuristics
- [ ] Test coverage improvements

## License

MIT

## Author

Rohan Julius

---

**Questions?** Check the [full documentation](docs/) or open an issue.
