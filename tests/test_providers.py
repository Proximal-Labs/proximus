"""Adapters against the real vendor SDKs, with HTTP mocked at the transport.

Each test sends a request through the SDK exactly as a run would, checks the
wire body, and parses a canned provider response, so request serialization,
response parsing and history replay are covered without network access.
"""

from __future__ import annotations

import json
import logging
from importlib.resources import files
from typing import Any

import anthropic
import httpx
import httpx2
import openai
import pytest
from harbor.llms.base import OutputLengthExceededError

from harbor_proximus.config import ProximusConfig
from harbor_proximus.llm import LLM
from harbor_proximus.providers.anthropic import AnthropicProvider
from harbor_proximus.providers.base import ModelCall, ParsedResponse
from harbor_proximus.providers.openai import (
    OpenAIChatProvider,
    OpenAIResponsesProvider,
)

SYSTEM = {"role": "system", "content": "You are Proximus."}
TASK = {"role": "user", "content": "Fix the bug."}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


def make_llm(model: str, monkeypatch, **config: Any) -> LLM:
    return LLM(
        model,
        ProximusConfig(**config),
        remaining_sec=lambda: None,
        logger=logging.getLogger("test"),
    )


def call_for(llm: LLM, messages: list[dict], tools=TOOLS) -> ModelCall:
    config = llm._config
    return ModelCall(
        model=llm.wire_model,
        messages=messages,
        tools=tools,
        timeout=60.0,
        reasoning_effort=config.reasoning_effort,
        temperature=config.temperature,
        max_thinking_tokens=config.max_thinking_tokens,
        session_id=llm.session_id,
        kwargs=dict(config.llm_call_kwargs),
    )


class Recorder:
    """A transport handler that records request bodies and replies in turn."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[Any] = []

    def bodies(self) -> list[dict]:
        return [json.loads(request.content) for request in self.requests]

    def __call__(self, request):
        self.requests.append(request)
        return self.replies.pop(0)(request)


# ----- Anthropic -----


def anthropic_sse(message: dict[str, Any]) -> str:
    """A Messages API stream that accumulates to ``message``."""
    usage = message.get("usage", {})
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                **{k: v for k, v in message.items() if k != "content"},
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {**usage, "output_tokens": 1},
            },
        }
    ]
    for index, block in enumerate(message["content"]):
        kind = block["type"]
        start = {"text": {"type": "text", "text": ""}}.get(kind)
        deltas: list[dict[str, Any]] = []
        if kind == "text":
            deltas = [{"type": "text_delta", "text": block["text"]}]
        elif kind == "thinking":
            start = {"type": "thinking", "thinking": "", "signature": ""}
            deltas = [
                {"type": "thinking_delta", "thinking": block["thinking"]},
                {"type": "signature_delta", "signature": block["signature"]},
            ]
        elif kind == "tool_use":
            start = {**block, "input": {}}
            deltas = [
                {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
            ]
        else:
            start = block
        events.append(
            {"type": "content_block_start", "index": index, "content_block": start}
        )
        events += [
            {"type": "content_block_delta", "index": index, "delta": delta}
            for delta in deltas
        ]
        events.append({"type": "content_block_stop", "index": index})
    events.append(
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": message["stop_reason"],
                "stop_sequence": None,
                "stop_details": message.get("stop_details"),
            },
            "usage": usage,
        }
    )
    events.append({"type": "message_stop"})
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def anthropic_reply(message: dict[str, Any]):
    def reply(request):
        return httpx2.Response(
            200,
            text=anthropic_sse(message),
            headers={"content-type": "text/event-stream"},
        )

    return reply


def use_anthropic(llm: LLM, recorder: Recorder) -> None:
    llm.provider.client = anthropic.AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(
            transport=httpx2.MockTransport(recorder)
        ),
    )


TOOL_TURN = {
    "model": "claude-opus-5-5",
    "content": [
        {"type": "thinking", "thinking": "Look first.", "signature": "SIG1"},
        {"type": "text", "text": "Checking."},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "bash",
            "input": {"command": "ls"},
        },
    ],
    "stop_reason": "tool_use",
    "usage": {
        "input_tokens": 100,
        "cache_read_input_tokens": 1000,
        "cache_creation_input_tokens": 50,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 0,
            "ephemeral_1h_input_tokens": 50,
        },
        "output_tokens": 20,
    },
}


async def test_anthropic_request_and_turn_replay(monkeypatch):
    llm = make_llm("anthropic/claude-opus-5-5", monkeypatch, reasoning_effort="high")
    recorder = Recorder(anthropic_reply(TOOL_TURN), anthropic_reply(TOOL_TURN))
    use_anthropic(llm, recorder)

    parsed = await llm.provider.complete(call_for(llm, [SYSTEM, TASK]))

    body = recorder.bodies()[0]
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == 128000  # the table's output cap
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}
    assert body["system"] == [
        {
            "type": "text",
            "text": "You are Proximus.",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]
    assert body["tools"][0]["name"] == "bash"
    assert body["tools"][0]["input_schema"]["required"] == ["command"]
    assert "tool_choice" not in body
    assert body["stream"] is True

    assert parsed.content == "Checking."
    assert parsed.reasoning_content == "Look first."
    assert [(c.call_id, c.name) for c in parsed.tool_calls] == [("toolu_1", "bash")]
    assert json.loads(parsed.tool_calls[0].raw_arguments) == {"command": "ls"}
    assert (parsed.prompt_tokens, parsed.cached_tokens) == (1150, 1000)
    # Table rates: input $4/M, cache read $0.20/M, 1h write $8/M, output $20/M.
    expected = 100 * 4e-06 + 1000 * 2e-07 + 50 * 8e-06 + 20 * 2e-05
    assert parsed.cost_usd == pytest.approx(expected)
    assert parsed.request_thinking == {"type": "adaptive", "display": "summarized"}

    # The turn replays exactly as the API returned it, signature included.
    history = [
        SYSTEM,
        TASK,
        llm.assistant_message(parsed),
        {"role": "tool", "tool_call_id": "toolu_1", "content": "a.py"},
    ]
    await llm.provider.complete(call_for(llm, history))
    second = recorder.bodies()[1]
    assert second["messages"][1] == {
        "role": "assistant",
        "content": TOOL_TURN["content"],
    }
    assert second["messages"][2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": "a.py",
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }


def test_anthropic_thinking_controls(monkeypatch):
    def body(**config) -> dict:
        llm = make_llm("anthropic/claude-test", monkeypatch, **config)
        return llm.provider.build_request(call_for(llm, [TASK]))

    budget = body(max_thinking_tokens=2048)
    assert budget["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert budget["max_tokens"] == 32000  # no table cap: the API needs one
    effort = body(reasoning_effort="medium")
    assert effort["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert effort["output_config"] == {"effort": "medium"}
    assert "thinking" not in body()


def test_anthropic_llm_call_kwargs_win(monkeypatch):
    llm = make_llm(
        "anthropic/claude-opus-5-5",
        monkeypatch,
        temperature=0.2,
        llm_call_kwargs={"max_tokens": 1000, "temperature": 0.5, "top_k": 3, "x": 1},
    )
    request = llm.provider.build_request(call_for(llm, [TASK]))
    assert request["max_tokens"] == 1000
    # Unknown to the SDK (it dropped temperature with the API): sent in the
    # body, never as a keyword the SDK would reject.
    assert request["extra_body"] == {"temperature": 0.5, "top_k": 3, "x": 1}
    assert "temperature" not in request


async def test_anthropic_server_side_fallback(monkeypatch):
    """A mid-output refusal fallback: only text survives from before the
    marker, the rescue model's blocks follow, and routing is reported.

    The turn replays verbatim with the marker in place (the server validates
    the thinking runs on each side of it); only the declined model's tool
    call, which never ran, is dropped."""
    llm = make_llm(
        "anthropic/claude-opus-5-5",
        monkeypatch,
        reasoning_effort="max",
        llm_call_kwargs={"anthropic_fallbacks": "default"},
    )
    turn = {
        "model": "claude-opus-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "DECLINED"},
            {"type": "text", "text": "Partial "},
            {"type": "tool_use", "id": "toolu_x", "name": "bash", "input": {}},
            {
                "type": "fallback",
                "from": {"model": "claude-opus-5-5"},
                "to": {"model": "claude-opus-5"},
            },
            {"type": "thinking", "thinking": "", "signature": "RESCUE"},
            {"type": "text", "text": "answer."},
        ],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 4,
            "output_tokens_details": {"thinking_tokens": 17},
            "iterations": [
                {"type": "message", "model": "claude-opus-5-5", "input_tokens": 10},
                {
                    "type": "fallback_message",
                    "model": "claude-opus-5",
                    "output_tokens": 4,
                },
            ],
        },
    }
    recorder = Recorder(anthropic_reply(turn))
    use_anthropic(llm, recorder)

    parsed = await llm.provider.complete(call_for(llm, [SYSTEM, TASK]))

    request = recorder.requests[0]
    assert "server-side-fallback-2026-07-01" in request.headers["anthropic-beta"]
    assert json.loads(request.content)["fallbacks"] == "default"
    assert parsed.native is not None
    assert [b["type"] for b in parsed.native["content"]] == [
        "thinking",
        "text",
        "fallback",
        "thinking",
        "text",
    ]
    assert parsed.native["content"][2]["to"] == {"model": "claude-opus-5"}
    assert parsed.tool_calls == []
    assert parsed.content == "Partial answer."
    assert parsed.response_model == "claude-opus-5"
    assert parsed.fallback_attempted and parsed.fallback_used
    assert parsed.fallback_iterations[-1]["type"] == "fallback_message"
    assert parsed.reasoning_tokens == 17
    assert parsed.request_fallbacks == "default"
    assert parsed.request_output_config == {"effort": "max"}


def test_anthropic_fallbacks_require_an_anthropic_api(monkeypatch):
    with pytest.raises(ValueError, match="anthropic_fallbacks"):
        make_llm(
            "openai/gpt-test",
            monkeypatch,
            llm_call_kwargs={"anthropic_fallbacks": "default"},
        )


async def test_anthropic_truncation_raises(monkeypatch):
    llm = make_llm("anthropic/claude-opus-5-5", monkeypatch)
    turn = {**TOOL_TURN, "content": [{"type": "text", "text": "cut"}]}
    turn["stop_reason"] = "max_tokens"
    use_anthropic(llm, Recorder(anthropic_reply(turn)))
    with pytest.raises(OutputLengthExceededError):
        await llm.provider.complete(call_for(llm, [TASK]))


def test_anthropic_thinking_only_turn_replays_as_text(monkeypatch):
    """A turn ending in thinking is rejected for the whole conversation."""
    llm = make_llm("anthropic/claude-opus-5-5", monkeypatch)
    bare = ParsedResponse(
        native={
            "format": "anthropic",
            "content": [{"type": "thinking", "thinking": "", "signature": "S"}],
        },
        opaque_reasoning=True,
    )
    assert not bare.is_empty
    message = llm.assistant_message(bare)
    request = llm.provider.build_request(
        call_for(llm, [TASK, message, {"role": "user", "content": "Continue."}])
    )
    assert request["messages"][1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "(no output this turn)"}],
    }


# ----- OpenAI -----


def use_openai(llm: LLM, recorder: Recorder) -> None:
    llm.provider.client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="https://api.example.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(recorder)),
    )


def chat_reply(message: dict[str, Any], finish_reason: str = "tool_calls"):
    def reply(request):
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-v4-pro",
                "choices": [
                    {"index": 0, "message": message, "finish_reason": finish_reason}
                ],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 50,
                    "total_tokens": 1050,
                    "prompt_tokens_details": {"cached_tokens": 800},
                    "completion_tokens_details": {"reasoning_tokens": 30},
                },
            },
        )

    return reply


async def test_chat_completions_for_compatible_endpoints(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    llm = make_llm(
        "deepseek/deepseek-v4-pro",
        monkeypatch,
        reasoning_effort="high",
        model_info={"max_input_tokens": 1000, "input_cost_per_token": 1e-06},
    )
    assert isinstance(llm.provider, OpenAIChatProvider)
    assert str(llm.provider.client.base_url).rstrip("/") == "https://api.deepseek.com"
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
    }
    recorder = Recorder(
        chat_reply({"role": "assistant", "content": None, "tool_calls": [call]}),
        chat_reply({"role": "assistant", "content": "x"}, "length"),
    )
    use_openai(llm, recorder)

    parsed = await llm.provider.complete(call_for(llm, [SYSTEM, TASK]))
    assert [(c.call_id, c.name) for c in parsed.tool_calls] == [("call_1", "bash")]
    assert (parsed.prompt_tokens, parsed.cached_tokens) == (1000, 800)
    assert parsed.cost_usd == pytest.approx(200 * 1e-06 + 800 * 1e-06)

    history = [SYSTEM, TASK, llm.assistant_message(parsed)]
    with pytest.raises(OutputLengthExceededError):
        await llm.provider.complete(call_for(llm, history))
    body = recorder.bodies()[1]
    assert body["reasoning_effort"] == "high"
    assert body["tools"] == TOOLS
    assert body["messages"][2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [call],
    }


def responses_reply(output: list[dict[str, Any]], status: str = "completed", **extra):
    def reply(request):
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "model": "gpt-5.4",
                "status": status,
                "output": output,
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": {
                    "input_tokens": 500,
                    "input_tokens_details": {"cached_tokens": 400},
                    "output_tokens": 60,
                    "output_tokens_details": {"reasoning_tokens": 40},
                    "total_tokens": 560,
                },
                **extra,
            },
        )

    return reply


REASONING_A = {
    "type": "reasoning",
    "id": "rs_A",
    "encrypted_content": "ENC_A",
    "summary": [{"type": "summary_text", "text": "thought A"}],
}
REASONING_B = {
    "type": "reasoning",
    "id": "rs_B",
    "encrypted_content": "ENC_B",
    "summary": [],
}
MESSAGE = {
    "type": "message",
    "id": "msg_1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "Diagnosing.", "annotations": []}],
}
FUNCTION_CALL = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_1",
    "name": "bash",
    "arguments": '{"command": "ls"}',
    "status": "completed",
}


async def test_responses_request_and_interleaved_replay(monkeypatch):
    """The output items replay verbatim and in order (encrypted chain of
    thought, message phase), so the model keeps its own reasoning."""
    llm = make_llm("openai/gpt-test", monkeypatch, reasoning_effort="xhigh")
    llm.session_id = "session-1"
    message = {**MESSAGE, "phase": "commentary"}
    output = [REASONING_A, message, REASONING_B, FUNCTION_CALL]
    recorder = Recorder(responses_reply(output), responses_reply([MESSAGE]))
    use_openai(llm, recorder)

    parsed = await llm.provider.complete(call_for(llm, [SYSTEM, TASK]))

    body = recorder.bodies()[0]
    assert body["store"] is False
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["reasoning"] == {"effort": "xhigh", "summary": "auto"}
    assert body["prompt_cache_key"] == "session-1"
    assert body["tools"][0] == {
        "type": "function",
        "name": "bash",
        "description": "Run a command.",
        "parameters": TOOLS[0]["function"]["parameters"],
        "strict": False,
    }
    assert body["input"][0] == {
        "type": "message",
        "role": "system",
        "content": [{"type": "input_text", "text": "You are Proximus."}],
    }

    assert parsed.content == "Diagnosing."
    assert parsed.reasoning_content == "thought A"
    assert parsed.tool_calls[0].call_id == "call_1"
    assert (parsed.prompt_tokens, parsed.cached_tokens) == (500, 400)

    history = [
        SYSTEM,
        TASK,
        llm.assistant_message(parsed),
        {"role": "tool", "tool_call_id": "call_1", "content": "a.py"},
    ]
    await llm.provider.complete(call_for(llm, history))
    replay = recorder.bodies()[1]["input"]
    assert replay[2:6] == output
    assert replay[6] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "a.py",
    }


async def test_responses_reasoning_only_turn_keeps_its_cot(monkeypatch):
    """A reasoning-only turn's encrypted CoT is replayed, and a placeholder
    message follows (a reasoning item needs a following item)."""
    llm = make_llm("openai/gpt-test", monkeypatch, reasoning_effort="high")
    recorder = Recorder(responses_reply([REASONING_B]), responses_reply([MESSAGE]))
    use_openai(llm, recorder)

    parsed = await llm.provider.complete(call_for(llm, [TASK]))
    assert not parsed.is_empty
    message = llm.assistant_message(parsed)
    assert message["content"] == "(no output this turn)"

    cont = {"role": "user", "content": "Continue."}
    await llm.provider.complete(call_for(llm, [TASK, message, cont]))
    replay = recorder.bodies()[1]["input"]
    assert replay[1]["encrypted_content"] == "ENC_B"
    assert replay[2] == {
        "type": "message",
        "role": "assistant",
        "content": "(no output this turn)",
    }


async def test_responses_empty_turn_and_truncation(monkeypatch):
    llm = make_llm("openai/gpt-test", monkeypatch, reasoning_effort="high")
    truncated = responses_reply(
        [MESSAGE],
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
    )
    use_openai(llm, Recorder(responses_reply([]), truncated))

    empty = await llm.provider.complete(call_for(llm, [TASK]))
    assert empty.is_empty
    with pytest.raises(OutputLengthExceededError):
        await llm.provider.complete(call_for(llm, [TASK]))


def test_routing(monkeypatch):
    assert isinstance(
        make_llm("anthropic/claude-test", monkeypatch).provider, AnthropicProvider
    )
    assert isinstance(
        make_llm("openai/gpt-test", monkeypatch).provider, OpenAIResponsesProvider
    )
    with pytest.raises(ValueError, match="prefix"):
        make_llm("gpt-test", monkeypatch)
    with pytest.raises(ValueError, match="anthropic_fallbacks"):
        make_llm(
            "openai/gpt-test",
            monkeypatch,
            llm_call_kwargs={"anthropic_fallbacks": "default"},
        )


def test_other_routes_need_their_own_endpoint_and_key(monkeypatch):
    """The OpenAI SDK must never send OPENAI_API_KEY to another provider, or
    another provider's key to api.openai.com."""
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("ZAI_API_KEY", "zai-key")
    info = {"model_info": {"max_input_tokens": 1000}}
    with pytest.raises(ValueError, match="api_base"):
        make_llm("openrouter/anthropic/claude-test", monkeypatch, **info)
    with pytest.raises(ValueError, match="api_base"):
        make_llm("zai/glm-test", monkeypatch, **info)
    llm = make_llm("zai/glm-test", monkeypatch, api_base="https://api.z.ai/v4", **info)
    assert str(llm.provider.client.base_url).rstrip("/") == "https://api.z.ai/v4"
    assert llm.provider.client.api_key == "zai-key"
    assert llm.wire_model == "glm-test"


def test_builtin_table_is_the_eval_models():
    from harbor_proximus.model_info import builtin_table, completion_cost

    # The packaged file (conftest adds test-only models to the loaded table).
    source = files("harbor_proximus").joinpath("model_info.json")
    table = json.loads(source.read_text())
    assert table == {k: v for k, v in builtin_table().items() if k in table}
    assert set(table) == {
        "anthropic/claude-opus-5-5",
        "anthropic/claude-haiku-5-5",
        "openai/gpt-5.6-sol",
        "openai/gpt-6-astra",
    }
    for entry in table.values():
        assert entry["max_input_tokens"] > 0 and entry["supports_vision"] is True
    haiku = table["anthropic/claude-haiku-5-5"]
    # Over 100k prompt tokens (cache reads and writes count), every rate steps up.
    small = completion_cost(haiku, input_tokens=1000, output_tokens=1000)
    assert small == pytest.approx(1000 * 1e-07 + 1000 * 5e-07)
    large = completion_cost(
        haiku,
        input_tokens=1000,
        output_tokens=1000,
        cache_read_tokens=100_000,
        cache_write_1h_tokens=10,
    )
    assert large == pytest.approx(
        1000 * 5e-07 + 1000 * 2.5e-06 + 100_000 * 5e-08 + 10 * 1e-06
    )


def test_empty_text_blocks_are_not_replayed():
    from harbor_proximus.providers.anthropic import replayable_blocks

    blocks = replayable_blocks(
        [
            {"type": "thinking", "thinking": "", "signature": "S"},
            {"type": "text", "text": ""},
            {"type": "tool_use", "id": "t", "name": "bash", "input": {}},
        ]
    )
    assert [b["type"] for b in blocks] == ["thinking", "tool_use"]
