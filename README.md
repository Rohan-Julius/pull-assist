<div align="center">

  <img src="assets/logo.png" alt="Pull Assist logo" width="560">

<br><br>

  <p>
    <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/LangGraph-Orchestrated-purple?logo=chainlink&logoColor=white" alt="LangGraph">
    <img src="https://img.shields.io/badge/vLLM-Inference-orange?logo=serverless&logoColor=white" alt="vLLM">
    <img src="https://img.shields.io/badge/AMD-ROCm-ED1C24?logo=amd&logoColor=white" alt="AMD ROCm">
    <img src="https://img.shields.io/badge/Model-DeepSeek--Coder--V2-4A90D9?logo=huggingface&logoColor=white" alt="DeepSeek">
    <img src="https://img.shields.io/badge/FastAPI-Proxy-009688?logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/Memory-SQLite-003B57?logo=sqlite&logoColor=white" alt="SQLite">
    <img src="https://img.shields.io/badge/License-MIT-green?logo=opensourceinitiative&logoColor=white" alt="License">
    <img src="https://img.shields.io/badge/CLI-pa_%2F_pullassist-black?logo=windowsterminal&logoColor=white" alt="CLI">
    <img src="https://img.shields.io/badge/PyPI-pull--assist-3775A9?logo=pypi&logoColor=white" alt="PyPI">
  </p>

  <p>
    <strong>Multi-agentic PR impact analysis</strong> — a LangGraph-orchestrated pipeline of specialist agents scores merge risk, maps blast radius, surfaces test gaps, simulates runtime breakage, and advises on rollback and deployment strategy.
  </p>

  <p>
    <em>Point it at a GitHub pull request or a local diff; get a structured report with risk score, top concerns, propagation chains, and next steps before you merge.</em>
  </p>

</div>

---

## Table of contents

- [How it works](#how-it-works)
- [Architecture](#architecture)
- [Agent pipeline](#agent-pipeline)
- [Graph layer](#graph-layer-deterministic)
- [Install via pip](#install-via-pip)
- [Custom CLI (`pa`)](#custom-cli-pa)
- [Backend: AMD GPU + vLLM](#backend-amd-gpu--vllm)
- [Quick start](#quick-start)
- [Input modes](#input-modes)
- [Configuration](#configuration)
- [Admin & GPU registry](#admin--gpu-registry)
- [Reports & memory](#reports--memory)
- [Development](#development)
- [Project structure](#project-structure)
- [License](#license)

---

## How it works

1. **Ingest** — Fetch a PR from GitHub (or read a local `.patch` / diff file).
2. **Parse** — Extract changed files, languages, symbols, and test coverage signals from the diff.
3. **Analyze** — Run a [LangGraph](https://github.com/langchain-ai/langgraph) orchestration of specialized LLM agents (with optional GitHub tool calls).
4. **Reason** — Build an evidence graph, failure propagation chains, and deployment advice (no extra LLM calls).
5. **Report** — Print a Rich terminal summary and save Markdown + JSON under `reports/`.
6. **Remember** — Persist results in a local SQLite memory store for repo history context on future runs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Your machine                                                           │
│  ┌──────────────┐    ┌─────────────┐    ┌────────────────────────────┐ │
│  │  pa /        │───▶│  GitHub API │    │  LangGraph orchestrator    │ │
│  │  pullassist  │    │  (PR, diff, │    │  + 7 specialized agents    │ │
│  │  main.py     │    │   search)   │    │  + graph layer             │ │
│  └──────┬───────┘    └─────────────┘    └─────────────┬──────────────┘ │
│         │                                              │                │
│         │         OpenAI-compatible API                │                │
│         └──────────────────────────────────────────────┘                │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  GPU server (AMD + ROCm)                                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │  Nginx       │───▶│  FastAPI     │───▶│  vLLM                    │  │
│  │  :443 / :80  │    │  Proxy :9000 │    │  (DeepSeek-Coder-V2)     │  │
│  │              │    │  auth, rate  │    │  :8000                   │  │
│  │              │    │  limits, SSE │    │  AMD ROCm / rocm/vllm    │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

| Layer                         | Role                                                                     |
| ----------------------------- | ------------------------------------------------------------------------ |
| **CLI** (`pa`, `pullassist`)  | User-facing commands, config, registry discovery                         |
| **Agents**                    | LLM reasoning + GitHub tools (symbol search, file fetch, test discovery) |
| **Graph layer**               | Deterministic evidence graph, propagation chains, deployment advice      |
| **Proxy** (`server/proxy.py`) | API keys, rate limiting, request queuing, SSE coalescing for LangChain   |
| **vLLM**                      | Model inference on AMD GPUs via ROCm; OpenAI-compatible `/v1` API        |

---

## Agent pipeline

Orchestration lives in `agents/orchestrator.py` as a **LangGraph `StateGraph`**. All agents share a typed state dict (`PRAnalysisState`) — a shared "whiteboard" each node reads and writes.

### Execution flow

```
START
  → Dependency Mapper      (GitHub tools: symbol search, file tree)
  → Change Simulator       (GitHub tools: fetch caller context)
  → Test Gap Agent         (GitHub tools: find tests, fetch test files)
  → Rollback Advisor       (LLM only)
  → Business Impact        (deterministic path patterns — no LLM)
  → Risk Evaluator         (LLM only — synthesizes all findings)
  → Critic                 (LLM only — challenges other agents)
  → [conditional] if SIGNIFICANT_ISSUES and reruns < 2:
        re-run flagged agents with Critic objections
        → Risk Evaluator (re-score)
        → Critic (re-check)
  → Graph Layer            (deterministic)
END
```

### Agents

| Agent                 | Tools                           | Purpose                                                                                 |
| --------------------- | ------------------------------- | --------------------------------------------------------------------------------------- |
| **Dependency Mapper** | `search_symbol`, `file_tree`    | Blast radius: direct/indirect dependents of changed symbols                             |
| **Change Simulator**  | `fetch_file`                    | Before/after runtime behavior; breaking scenarios per caller                            |
| **Test Gap Agent**    | `find_test_files`, `fetch_file` | Uncovered functions and missing test scenarios                                          |
| **Rollback Advisor**  | —                               | Rollback difficulty, risks, and step-by-step guidance                                   |
| **Business Impact**   | —                               | Classifies changed paths into business domains (auth, payments, etc.) via pattern rules |
| **Risk Evaluator**    | —                               | Weighted 0–10 risk score across blast radius, tests, runtime, complexity                |
| **Critic**            | —                               | Flags inconsistencies; can trigger re-runs and score corrections                        |

Tool-calling agents use LangChain's `AgentExecutor`. By default, **legacy prompt-based tools** work with plain vLLM. Set `USE_NATIVE_TOOL_CALLING=true` only when vLLM is started with `--enable-auto-tool-choice` and a matching `--tool-call-parser`.

After agents finish, outputs are validated against JSON schemas in `config/settings.py` (`AGENT_OUTPUT_SCHEMAS`).

---

## Graph layer (deterministic)

No additional LLM calls — built from agent outputs and diff metadata:

| Module                        | Output                                                                            |
| ----------------------------- | --------------------------------------------------------------------------------- |
| `graph/evidence_graph.py`     | Symbol-centric graph of callers and transitive deps (confidence degrades per hop) |
| `graph/propagation_engine.py` | Failure propagation chains with arrow diagrams                                    |
| `graph/deployment_advisor.py` | Deployment strategy recommendation (e.g. canary vs full rollout)                  |

Static diff analysis (`github/diff_static_risks.py`) augments test-gap findings from the raw patch.

---

## Install via pip

The package is published as **`pull-assist`** on PyPI-style metadata (`pyproject.toml`). After install, two console scripts are available: **`pa`** and **`pullassist`**.

### From PyPI

```bash
pip install pull-assist
```

Optional extras:

```bash
pip install "pull-assist[server]"   # FastAPI proxy (GPU host)
pip install "pull-assist[dev]"      # pytest, build, twine
```

### From source (GitHub)

```bash
git clone https://github.com/Rohan-Julius/pull-assist.git
cd pull-assist
pip install -e .
```

### Requirements

- **Python 3.10+**
- A **GitHub personal access token** (for PR URLs and remote repo tool calls)
- Access to an **LLM endpoint** (local vLLM or shared GPU server via the proxy)

Verify installation:

```bash
pa --version
pullassist --version
```

---

## Custom CLI (`pa`)

The CLI is built with **Click** and **Rich** (`cli/app.py`). It is the recommended interface for day-to-day use.

### Commands

| Command                                   | Description                                     |
| ----------------------------------------- | ----------------------------------------------- |
| `pa review <PR_URL>`                      | Analyze a GitHub pull request                   |
| `pa review --diff FILE --repo owner/repo` | Local patch + GitHub context for tools          |
| `pa review --diff FILE --local PATH`      | Fully offline (local git only)                  |
| `pa history [repo]`                       | Past analyses from the memory store             |
| `pa config show \| set \| reset`          | Manage `~/.pull-assist/config.json`             |
| `pa status`                               | Connectivity checks (GitHub, LLM, memory, deps) |
| `pa admin …`                              | GPU registry management (admin only)            |

### Examples

```bash
# Configure once
pa config set --token ghp_xxxxxxxx --key pa-your-api-key
pa config set --server http://your-gpu-host:9000/v1   # optional if using registry

# Verify setup
pa status

# Analyze a PR
pa review https://github.com/owner/repo/pull/123

# Local diff (offline)
git diff main..feature > changes.patch
pa review --diff changes.patch --local .

# Compact vs full report
pa review https://github.com/owner/repo/pull/123 --full

# Data pipeline only (no GPU)
pa review https://github.com/owner/repo/pull/123 --day1-only
```

### VS Code integration

When run inside VS Code (`TERM_PROGRAM=vscode`), `pa` can open a dedicated integrated terminal with a clean `pull-assist>` prompt (see `cli/app.py`).

### Legacy entry point

`main.py` remains available for scripting and direct `python main.py` usage with the same three input modes and `--day1-only` / `--verbose` flags.

---

## Backend: AMD GPU + vLLM

Inference runs on **AMD GPUs** using **ROCm** and **[vLLM](https://github.com/vllm-project/vllm)**. vLLM exposes an **OpenAI-compatible API** (`/v1/chat/completions`), which LangChain's `ChatOpenAI` uses via `config/settings.py`.

### Default model

`deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` (configurable via `LLM_MODEL` / `pa config set --model`)

### Two deployment options

**Option A — vLLM already running** (proxy only):

```bash
# On the GPU host: start vLLM on :8000 (see docker-compose.yml comments)
docker compose up -d
```

**Option B — Full stack** (vLLM + proxy + nginx):

```bash
docker compose -f docker-compose.full.yml up -d
```

For **AMD ROCm**, use the ROCm vLLM image in `docker-compose.full.yml`:

```yaml
image: rocm/vllm:latest
devices:
  - /dev/kfd
  - /dev/dri
```

Install ROCm drivers on the host first: [AMD ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/).

### Proxy layer

`server/proxy.py` is a **FastAPI** app that sits in front of vLLM:

- Per-user **API key** authentication
- **Rate limiting** and concurrency caps per key
- **Usage logging**
- **SSE coalescing** so LangChain clients that disable streaming still work through port `:9000`

```
CLI (:pa)  →  Proxy (:9000)  →  vLLM (:8000)
```

Start the proxy:

```bash
uvicorn server.proxy:app --host 0.0.0.0 --port 9000
```

Environment variables:

| Variable                 | Default                 | Description                     |
| ------------------------ | ----------------------- | ------------------------------- |
| `VLLM_BACKEND_URL`       | `http://localhost:8000` | Upstream vLLM base URL          |
| `PROXY_PORT`             | `9000`                  | Proxy listen port               |
| `RATE_LIMIT_PER_MINUTE`  | `30`                    | Requests per API key per minute |
| `MAX_CONCURRENT_PER_KEY` | `2`                     | Parallel requests per key       |
| `API_KEYS_FILE`          | `server/api_keys.json`  | Key store (gitignored)          |

### Running vLLM manually (AMD host)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
  --host 0.0.0.0 --port 8000 \
  --dtype float16 --max-model-len 16384 \
  --gpu-memory-utilization 0.85 --trust-remote-code
```

Set `LLM_MAX_TOKENS` (default `1024`) so prompt + completion fit within `--max-model-len`.

---

## Quick start

### End users (CLI + shared GPU)

```bash
pip install pull-assist
pa config set --token ghp_xxx --key pa-xxx
pa status
pa review https://github.com/owner/repo/pull/1
```

If your team uses the **GPU registry** (GitHub Gist), the active server URL is discovered automatically — you only need your API key and GitHub token.

### Developers (local)

```bash
python -m venv venv && source venv/bin/activate
pip install -e .
cp .env.example .env   # create and fill in tokens (see below)
python main.py --pr https://github.com/owner/repo/pull/1
```

Example `.env`:

```env
GITHUB_TOKEN=ghp_...
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=not-needed
LLM_MODEL=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
```

---

## Input modes

| Mode                     | Command                               | GitHub API                | GPU |
| ------------------------ | ------------------------------------- | ------------------------- | --- |
| GitHub PR URL            | `pa review <URL>`                     | Yes (PR + diff + reviews) | Yes |
| Local diff + remote repo | `pa review --diff f.patch --repo o/r` | Yes (tools only)          | Yes |
| Local diff + local repo  | `pa review --diff f.patch --local .`  | No                        | Yes |
| Context only             | `--day1-only`                         | As above                  | No  |

Supported languages for symbol extraction include Python, JavaScript/TypeScript, Java, Go, Ruby, Rust, and more (see `config/settings.py` → `SUPPORTED_LANGUAGES`).

---

## Configuration

Settings are layered:

1. **`~/.pull-assist/config.json`** (CLI — preferred for end users)
2. **Environment variables** (`PA_*` or `LLM_*`, `GITHUB_TOKEN`)
3. **`.env`** (local development)
4. **GPU registry Gist** (auto-discovers `LLM_BASE_URL` when active)

```bash
pa config show
pa config set --token ghp_... --server http://host:9000/v1 --key pa-abc --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
pa config reset
```

| Setting      | Env vars                          |
| ------------ | --------------------------------- |
| LLM server   | `PA_SERVER`, `LLM_BASE_URL`       |
| API key      | `PA_API_KEY`, `LLM_API_KEY`       |
| GitHub token | `PA_GITHUB_TOKEN`, `GITHUB_TOKEN` |
| Model        | `PA_MODEL`, `LLM_MODEL`           |

---

## Admin & GPU registry

For teams sharing one GPU server, a **GitHub Gist registry** (`cli/registry.py`) publishes the current proxy URL so users do not need the raw IP.

**Admin (one-time setup):**

```bash
pa admin init
git add cli/registry.py && git commit -m "Add registry gist ID"

# When GPU is up
pa admin set-gpu http://<GPU_IP>:9000/v1
pa admin deactivate    # mark offline
pa admin status
```

**Users:** `pa config set --token … --key …` — registry resolves the server when `active: true`.

---

## Reports & memory

| Output           | Location                                 |
| ---------------- | ---------------------------------------- |
| Terminal summary | Rich panels via `output/formatter.py`    |
| Markdown report  | `reports/<repo>-pr-<n>-<timestamp>.md`   |
| JSON report      | `reports/<repo>-pr-<n>-<timestamp>.json` |
| Analysis history | `memory/pr_history.db` (SQLite)          |

`pa history` and `pa history owner/repo` browse past runs. Prior analyses are injected into agent prompts as repo context.

---

## Development

```bash
# Install with dev deps
pip install -e ".[dev]"  # or: pip install -r requirements.txt

# Run tests
pytest

# Run without agents (diff parsing + context only)
python main.py --pr <URL> --day1-only
pa review <URL> --day1-only
```

Key test modules: `tests/test_agents.py`, `tests/test_graph_layer.py`, `tests/test_diff_parser.py`, `tests/test_orchestrator_patch.py`, `tests/test_proxy_sse.py`.

---

## Project structure

```
pull-assist/
├── agents/              # LangGraph nodes + specialist agents
│   ├── orchestrator.py  # StateGraph definition
│   ├── dependency_mapper.py
│   ├── change_simulator.py
│   ├── test_gap.py
│   ├── risk_evaluator.py
│   ├── critic.py
│   ├── rollback_advisor.py
│   └── business_impact.py
├── cli/                 # `pa` / `pullassist` Click CLI
├── config/settings.py   # LLM, GitHub, prompts, risk weights
├── github/              # PR client, diff parser, static risks
├── graph/               # Evidence graph, propagation, deployment
├── memory/              # SQLite PR history store
├── output/              # Report builder + Rich formatter
├── server/proxy.py      # FastAPI auth proxy for vLLM
├── tools/               # LangChain GitHub tools
├── main.py              # Script entry point
├── docker-compose.yml   # Proxy-only (vLLM external)
├── docker-compose.full.yml  # vLLM + proxy + nginx (AMD ROCm)
└── pyproject.toml       # Package metadata + console scripts
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Publishing to PyPI (maintainers)

1. Create accounts: [pypi.org](https://pypi.org/account/register/) and optionally [test.pypi.org](https://test.pypi.org/account/register/).
2. Enable **2FA** on PyPI (required for uploads).
3. Create an API token: Account → API tokens → scope **Entire account** (first upload) or project `pull-assist`.
4. Bump `version` in `pyproject.toml` and `cli/__init__.py` for each release.
5. Build and upload:

```bash
pip install build twine
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-AgENdHlwaS5vcmcC...   # your token — never commit this

# Dry run on TestPyPI first
./scripts/publish-pypi.sh test
pip install -i https://test.pypi.org/simple/ pull-assist

# Production
./scripts/publish-pypi.sh
```

The name **`pull-assist`** is not taken on PyPI yet (verified). Package builds pass `twine check`.

---

## Links

- Repository: [github.com/Rohan-Julius/pull-assist](https://github.com/Rohan-Julius/pull-assist)
- Issues & contributions: use GitHub Issues and Pull Requests on the repo above
