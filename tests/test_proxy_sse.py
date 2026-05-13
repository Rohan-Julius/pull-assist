"""Unit tests for server/proxy.py SSE coalescing (no server required)."""
import json


def test_coalesce_simple_content():
    from server.proxy import _coalesce_openai_chat_sse

    sse = """data: {"id":"x","choices":[{"delta":{"content":"Hello"}}]}
data: {"id":"x","choices":[{"delta":{"content":" world"}}]}
data: [DONE]
"""
    out = _coalesce_openai_chat_sse(sse)
    assert out is not None
    assert out["choices"][0]["message"]["content"] == "Hello world"
    assert out["choices"][0]["finish_reason"] == "stop"


def test_coalesce_empty_returns_none():
    from server.proxy import _coalesce_openai_chat_sse

    assert _coalesce_openai_chat_sse("") is None
    assert _coalesce_openai_chat_sse("not sse") is None


def test_coalesce_tool_arguments():
    from server.proxy import _coalesce_openai_chat_sse

    c1 = json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "search", "arguments": '{"q":'},
                            }
                        ]
                    }
                }
            ]
        }
    )
    c2 = json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '"x"}'}}
                        ]
                    }
                }
            ]
        }
    )
    sse = f"data: {c1}\ndata: {c2}\ndata: [DONE]\n"
    out = _coalesce_openai_chat_sse(sse)
    assert out is not None
    tc = out["choices"][0]["message"]["tool_calls"]
    assert len(tc) == 1
    assert tc[0]["function"]["name"] == "search"
    assert tc[0]["function"]["arguments"] == '{"q":"x"}'


def test_coalesce_skips_null_choice_objects():
    from server.proxy import _coalesce_openai_chat_sse

    inner = json.dumps({"choices": [None, {"delta": {"content": "ok"}}]})
    sse = f"data: {inner}\ndata: [DONE]\n"
    out = _coalesce_openai_chat_sse(sse)
    assert out is not None
    assert out["choices"][0]["message"]["content"] == "ok"


def test_coalesce_reasoning_content():
    from server.proxy import _coalesce_openai_chat_sse

    inner = json.dumps({"choices": [{"delta": {"reasoning_content": "think"}}]})
    sse = f"data: {inner}\ndata: [DONE]\n"
    out = _coalesce_openai_chat_sse(sse)
    assert out is not None
    assert "think" in (out["choices"][0]["message"]["content"] or "")
