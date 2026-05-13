"""
FastAPI proxy layer for pull-assist.

Sits in front of your vLLM server and provides:
  - API key authentication (per-user keys)
  - Rate limiting (per-key, sliding window)
  - Request queuing / concurrency control
  - Usage logging
  - Version handshake for CLI compatibility

Deploy with:
  uvicorn server.proxy:app --host 0.0.0.0 --port 9000

Architecture:
  [CLI] → [This Proxy :9000] → [vLLM :8000]
"""

import os
import time
import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Any, Optional

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests as http_requests

# ── Configuration ─────────────────────────────────────────────────────────────

VLLM_BACKEND_URL = os.getenv("VLLM_BACKEND_URL", "http://localhost:8000")
PROXY_PORT = int(os.getenv("PROXY_PORT", "9000"))
PROXY_VERSION = "0.2.0"

# API key storage — in production, use a database
API_KEYS_FILE = Path(os.getenv("API_KEYS_FILE", "server/api_keys.json"))
USAGE_LOG_FILE = Path(os.getenv("USAGE_LOG_FILE", "server/usage.log"))

# Rate limiting: max requests per key per minute
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
MAX_CONCURRENT_PER_KEY = int(os.getenv("MAX_CONCURRENT_PER_KEY", "2"))

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pull-assist-proxy")

# ── In-memory state ──────────────────────────────────────────────────────────

# Rate limiter: key → list of timestamps
_rate_windows: dict[str, list[float]] = defaultdict(list)

# Concurrency tracker: key → count of active requests
_active_requests: dict[str, int] = defaultdict(int)


# ── API key management ────────────────────────────────────────────────────────

def _load_api_keys() -> dict:
    """Load API keys from disk. Format: {key_string: {user, created_at, active}}"""
    if not API_KEYS_FILE.exists():
        return {}
    try:
        return json.loads(API_KEYS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_api_keys(keys: dict):
    """Persist API keys to disk."""
    API_KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    API_KEYS_FILE.write_text(json.dumps(keys, indent=2) + "\n")


def _validate_api_key(key: str) -> Optional[dict]:
    """Validate an API key and return user info, or None if invalid."""
    keys = _load_api_keys()
    info = keys.get(key)
    if info and info.get("active", True):
        return info
    return None


# ── Rate limiting ─────────────────────────────────────────────────────────────

def _check_rate_limit(key: str) -> bool:
    """Returns True if the request is within rate limits."""
    now = time.time()
    window = _rate_windows[key]

    # Remove timestamps older than 60 seconds
    _rate_windows[key] = [t for t in window if now - t < 60]

    if len(_rate_windows[key]) >= RATE_LIMIT_PER_MINUTE:
        return False

    _rate_windows[key].append(now)
    return True


def _check_concurrency(key: str) -> bool:
    """Returns True if under concurrent request limit."""
    return _active_requests[key] < MAX_CONCURRENT_PER_KEY


# ── Usage logging ─────────────────────────────────────────────────────────────

def _log_usage(key: str, user: str, endpoint: str, status: int, duration: float):
    """Append a usage record to the log file."""
    USAGE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user": user,
        "key_prefix": key[:8] + "...",
        "endpoint": endpoint,
        "status": status,
        "duration_ms": round(duration * 1000),
    }
    with open(USAGE_LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def _coalesce_openai_chat_sse(sse_text: str) -> Optional[dict[str, Any]]:
    """
    Turn an OpenAI-style SSE chat completion body into a single non-streaming
    chat.completion JSON object. Used when vLLM streams despite stream=false
    or when an older proxy forwarded stream=true.
    """
    if not sse_text or "data:" not in sse_text:
        return None

    content_parts: list[str] = []
    # index -> {id, name, arg_parts}
    tool_acc: dict[int, dict[str, Any]] = {}
    finish_reason: Optional[str] = None
    model: Optional[str] = None
    cid: Optional[str] = None

    def _append_delta_text(delta: dict[str, Any]) -> None:
        """Accumulate text from delta (content, reasoning, multimodal lists)."""
        c = delta.get("content")
        if isinstance(c, str) and c:
            content_parts.append(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text")
                    if isinstance(t, str) and t:
                        content_parts.append(t)
        rc = delta.get("reasoning_content")
        if isinstance(rc, str) and rc:
            content_parts.append(rc)

    for raw_line in sse_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        if not payload.startswith("{"):
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        cid = chunk.get("id") or cid
        model = chunk.get("model") or model
        for ch in chunk.get("choices") or []:
            if not isinstance(ch, dict):
                continue
            delta = ch.get("delta")
            if not isinstance(delta, dict):
                delta = {}
            if ch.get("finish_reason"):
                finish_reason = ch["finish_reason"]
            _append_delta_text(delta)
            for tc in delta.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                idx = int(tc.get("index", 0))
                slot = tool_acc.setdefault(
                    idx, {"id": "", "name": "", "arg_parts": []}
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if isinstance(fn, dict):
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arg_parts"].append(fn["arguments"])

    tool_calls: list[dict[str, Any]] = []
    for idx in sorted(tool_acc.keys()):
        t = tool_acc[idx]
        if t["id"] or t["name"] or t["arg_parts"]:
            tool_calls.append(
                {
                    "id": t["id"] or f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "arguments": "".join(t["arg_parts"]),
                    },
                }
            )

    text = "".join(content_parts)
    if not text and not tool_calls:
        return None

    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": cid or "chatcmpl-coalesced",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or "",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop"),
            }
        ],
    }


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="pull-assist Proxy",
    description="Auth + rate-limiting proxy for vLLM backend",
    version=PROXY_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth dependency ───────────────────────────────────────────────────────────

async def verify_api_key(request: Request) -> dict:
    """FastAPI dependency that validates the API key from the Authorization header."""
    auth = request.headers.get("Authorization", "")

    # Accept "Bearer <key>" or just "<key>"
    if auth.startswith("Bearer "):
        key = auth[7:]
    else:
        key = auth

    if not key or key == "not-needed":
        # Allow unauthenticated access if no keys are configured
        keys = _load_api_keys()
        if keys:
            raise HTTPException(status_code=401, detail="API key required")
        return {"user": "anonymous", "key": "none"}

    user_info = _validate_api_key(key)
    if not user_info:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not _check_rate_limit(key):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({RATE_LIMIT_PER_MINUTE}/min). Try again shortly."
        )

    if not _check_concurrency(key):
        raise HTTPException(
            status_code=429,
            detail=f"Too many concurrent requests (max {MAX_CONCURRENT_PER_KEY}). Wait for current analysis to finish."
        )

    return {"user": user_info.get("user", "unknown"), "key": key}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — does not require auth."""
    try:
        resp = http_requests.get(f"{VLLM_BACKEND_URL}/v1/models", timeout=5)
        backend_ok = resp.status_code == 200
    except Exception:
        backend_ok = False

    return {
        "status": "healthy" if backend_ok else "degraded",
        "proxy_version": PROXY_VERSION,
        "backend_reachable": backend_ok,
        "backend_url": VLLM_BACKEND_URL,
    }


@app.get("/version")
async def version():
    """Version handshake endpoint for CLI compatibility checking."""
    return {
        "proxy_version": PROXY_VERSION,
        "min_cli_version": "0.1.0",
        "backend_url": VLLM_BACKEND_URL,
    }


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    """Proxy the /v1/models endpoint from vLLM."""
    try:
        resp = http_requests.get(f"{VLLM_BACKEND_URL}/v1/models", timeout=10)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Backend unreachable: {e}")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, auth: dict = Depends(verify_api_key)):
    """Proxy chat completions to vLLM with auth + rate limiting."""
    key = auth["key"]
    user = auth["user"]

    _active_requests[key] += 1
    start = time.time()

    try:
        raw = await request.body()
        if not raw or not raw.strip():
            raise HTTPException(
                status_code=400,
                detail="Empty request body; expected a JSON chat completions payload.",
            )
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON body: {e}",
            ) from e

        # Buffered JSON only: force non-streaming toward vLLM. Some stacks still
        # return SSE; we coalesce that below if resp.json() fails.
        if isinstance(body, dict):
            body = dict(body)
            body["stream"] = False
            body.pop("stream_options", None)

        resp = http_requests.post(
            f"{VLLM_BACKEND_URL}/v1/chat/completions",
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )

        duration = time.time() - start
        _log_usage(key, user, "/v1/chat/completions", resp.status_code, duration)

        # Safely parse response — vLLM may return SSE even when stream=false
        try:
            resp_data = resp.json()
        except Exception as parse_err:
            logger.debug("vLLM response not JSON (%s); trying SSE coalesce", parse_err)
            try:
                coalesced = _coalesce_openai_chat_sse(resp.text or "")
            except Exception as coal_err:
                logger.warning("SSE coalesce failed: %s", coal_err)
                coalesced = None
            if coalesced is None and resp.text and "data:" in resp.text:
                logger.warning(
                    "SSE present but coalesce returned None; prefix=%r",
                    (resp.text or "")[:500],
                )
            if coalesced is not None:
                logger.info(
                    "Coalesced SSE chat completion from vLLM (first line: %s)",
                    (resp.text or "").splitlines()[0][:120] if resp.text else "",
                )
                return JSONResponse(content=coalesced, status_code=200)

            resp_data = {
                "error": {
                    "message": f"Backend returned non-JSON response (HTTP {resp.status_code}): {(resp.text or '')[:300]}",
                    "type": "backend_error",
                    "code": resp.status_code,
                }
            }
            if resp.status_code == 200:
                # If status was 200 but body isn't JSON, that's a 502
                return JSONResponse(content=resp_data, status_code=502)

        return JSONResponse(content=resp_data, status_code=resp.status_code)

    except http_requests.exceptions.Timeout:
        duration = time.time() - start
        _log_usage(key, user, "/v1/chat/completions", 504, duration)
        raise HTTPException(status_code=504, detail="Backend timeout (>120s)")

    except http_requests.exceptions.ConnectionError:
        duration = time.time() - start
        _log_usage(key, user, "/v1/chat/completions", 502, duration)
        raise HTTPException(status_code=502, detail="Cannot reach vLLM backend")

    except Exception as e:
        duration = time.time() - start
        _log_usage(key, user, "/v1/chat/completions", 500, duration)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        _active_requests[key] = max(0, _active_requests[key] - 1)


@app.post("/v1/completions")
async def completions(request: Request, auth: dict = Depends(verify_api_key)):
    """Proxy legacy completions to vLLM."""
    key = auth["key"]
    user = auth["user"]

    _active_requests[key] += 1
    start = time.time()

    try:
        body = await request.json()
        if isinstance(body, dict):
            body["stream"] = False
        resp = http_requests.post(
            f"{VLLM_BACKEND_URL}/v1/completions",
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        duration = time.time() - start
        _log_usage(key, user, "/v1/completions", resp.status_code, duration)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    except Exception as e:
        duration = time.time() - start
        _log_usage(key, user, "/v1/completions", 500, duration)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        _active_requests[key] = max(0, _active_requests[key] - 1)


# ── Admin endpoints ───────────────────────────────────────────────────────────

class CreateKeyRequest(BaseModel):
    user: str
    note: str = ""


@app.post("/admin/keys")
async def create_api_key(req: CreateKeyRequest, request: Request):
    """Create a new API key. Protected by ADMIN_SECRET env var."""
    admin_secret = os.getenv("ADMIN_SECRET", "")
    provided = request.headers.get("X-Admin-Secret", "")

    if admin_secret and provided != admin_secret:
        raise HTTPException(status_code=403, detail="Invalid admin secret")

    new_key = f"pa-{uuid.uuid4().hex[:24]}"
    keys = _load_api_keys()
    keys[new_key] = {
        "user": req.user,
        "note": req.note,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active": True,
    }
    _save_api_keys(keys)

    logger.info(f"Created API key for user '{req.user}': {new_key[:12]}...")
    return {"key": new_key, "user": req.user}


@app.get("/admin/keys")
async def list_api_keys(request: Request):
    """List all API keys (masked)."""
    admin_secret = os.getenv("ADMIN_SECRET", "")
    provided = request.headers.get("X-Admin-Secret", "")

    if admin_secret and provided != admin_secret:
        raise HTTPException(status_code=403, detail="Invalid admin secret")

    keys = _load_api_keys()
    return [
        {
            "key_prefix": k[:12] + "...",
            "user": v.get("user", "?"),
            "created_at": v.get("created_at", "?"),
            "active": v.get("active", True),
        }
        for k, v in keys.items()
    ]


@app.get("/admin/usage")
async def get_usage(request: Request, limit: int = 50):
    """Get recent usage logs."""
    admin_secret = os.getenv("ADMIN_SECRET", "")
    provided = request.headers.get("X-Admin-Secret", "")

    if admin_secret and provided != admin_secret:
        raise HTTPException(status_code=403, detail="Invalid admin secret")

    if not USAGE_LOG_FILE.exists():
        return []

    lines = USAGE_LOG_FILE.read_text().strip().split("\n")
    records = []
    for line in lines[-limit:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records
