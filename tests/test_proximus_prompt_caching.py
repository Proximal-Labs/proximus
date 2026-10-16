"""The second cache breakpoint must land on the newest message, every turn.

Everything after that breakpoint is re-read at full input price on the next
call, so a breakpoint that cannot advance is a pure cost leak -- and it fails
silently, since the run still succeeds and only the bill moves.

The shape that broke it: an assistant turn that is a pure tool call carries
``content: None`` (payload in ``tool_calls``), so it cannot host a
``cache_control`` block. Tool messages were skipped outright. In a
conversation that is only assistant-tool_call -> tool-result, that left
nothing taggable after the opening instruction, pinning the breakpoint to
``messages[1]`` for the whole run: a tool-heavy run then has almost no cache
hits and costs several times as much as a well-anchored one.

The fix tags tool messages at the top level -- the key litellm reads in
``convert_to_anthropic_tool_result`` -- so no tool_result content
transformation is disturbed.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import httpx
import pytest
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from harbor_proximus.config import ProximusConfig
from harbor_proximus.llm import LLM, _AnthropicFallbackHTTPHandler


def _llm() -> LLM:
    return LLM(
        model_name="anthropic/claude-test-model",
        config=ProximusConfig(),
        remaining_sec=lambda: None,
        logger=logging.getLogger("test"),
    )


def _breakpoints(cached: list[dict]) -> list[int]:
    """Indices of messages carrying a cache breakpoint, however it's hosted."""
    marked = []
    for i, message in enumerate(cached):
        if message.get("cache_control") is not None:
            marked.append(i)
            continue
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(item, dict) and item.get("cache_control") is not None
            for item in content
        ):
            marked.append(i)
    return marked


def _bash_turns(n: int) -> list[dict]:
    """A pure tool-calling conversation: no assistant text, no later user turn."""
    messages: list[dict] = [
        {"role": "system", "content": "You are Proximus."},
        {"role": "user", "content": "Optimize the codegen backend."},
    ]
    for i in range(n):
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"cmd":"make"}'},
                    }
                ],
            }
        )
        messages.append(
            {"role": "tool", "tool_call_id": f"call_{i}", "content": f"output {i}"}
        )
    return messages


def test_breakpoint_advances_through_pure_tool_calls():
    """The regression: the anchor must reach the newest message, not messages[1]."""
    messages = _bash_turns(40)
    cached = _llm()._apply_prompt_caching(messages)

    marks = _breakpoints(cached)
    assert marks[0] == 0, "system prompt must always be tagged"
    assert marks[-1] == len(cached) - 1, (
        f"second breakpoint stalled at index {marks[-1]} of {len(cached) - 1}; "
        "everything after it is re-read at full price every turn"
    )
    assert len(marks) == 2, f"expected exactly 2 breakpoints, got {len(marks)}"


def test_tool_message_is_tagged_top_level_not_in_content():
    """litellm reads cache_control off the tool message itself.

    Rewriting a tool message's ``content`` into provider blocks is what the
    old code was avoiding by skipping tool messages; the top-level key gets
    the breakpoint without touching the tool_result transformation.
    """
    cached = _llm()._apply_prompt_caching(_bash_turns(3))
    last = cached[-1]

    assert last["role"] == "tool"
    assert last["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert last["content"] == "output 2", "tool content must be left untouched"


def test_assistant_text_still_preferred_when_present():
    """An assistant turn with real content is a valid anchor and still wins."""
    messages = _bash_turns(2)
    messages.append({"role": "assistant", "content": "Here is the plan."})
    cached = _llm()._apply_prompt_caching(messages)

    marks = _breakpoints(cached)
    assert marks[-1] == len(cached) - 1
    assert cached[-1]["content"][0]["cache_control"] is not None


def test_multimodal_breakpoint_tags_text_not_image():
    """Z.ai accepts cache_control on an image but silently blanks the image."""
    messages = [
        {"role": "system", "content": "You are Proximus."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        },
    ]
    cached = _llm()._apply_prompt_caching(messages)
    text, image = cached[-1]["content"]

    assert text["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "cache_control" not in image


def test_non_anthropic_is_untouched():
    """gpt uses server-side prefix caching; this path must stay a no-op."""
    llm = LLM(
        model_name="openai/gpt-5.4",
        config=ProximusConfig(),
        remaining_sec=lambda: None,
        logger=logging.getLogger("test"),
    )
    messages = _bash_turns(3)
    assert llm._apply_prompt_caching(messages) is messages


def test_anthropic_fallback_routing_metadata():
    response = httpx.Response(
        200,
        json={
            "model": "claude-opus-5",
            "stop_reason": "end_turn",
            "stop_details": None,
            "usage": {
                "output_tokens_details": {"thinking_tokens": 17},
                "iterations": [
                    {"type": "message", "input_tokens": 10, "output_tokens": 0},
                    {
                        "type": "fallback_message",
                        "model": "claude-opus-5",
                        "input_tokens": 10,
                        "output_tokens": 4,
                    },
                ],
            },
        },
    )

    routing = _AnthropicFallbackHTTPHandler.routing_from_response(response)

    assert routing["response_model"] == "claude-opus-5"
    assert routing["fallback_attempted"] is True
    assert routing["fallback_used"] is True
    assert routing["thinking_tokens"] == 17
    assert routing["fallback_iterations"][-1]["type"] == "fallback_message"


@pytest.mark.asyncio
async def test_anthropic_fallback_handler_injects_body_and_beta(monkeypatch):
    handler = _AnthropicFallbackHTTPHandler("default")
    response = httpx.Response(
        200,
        json={
            "model": "claude-test-model",
            "stop_reason": "end_turn",
            "usage": {"iterations": [{"type": "message"}]},
        },
    )
    post = AsyncMock(return_value=response)
    monkeypatch.setattr(AsyncHTTPHandler, "post", post)

    await handler.post(
        "https://api.anthropic.com/v1/messages",
        json={
            "model": "claude-test-model",
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "max"},
        },
        headers={"anthropic-beta": "interleaved-thinking-2025-05-14"},
    )

    sent = post.await_args.kwargs
    assert sent["json"]["fallbacks"] == "default"
    beta = sent["headers"]["anthropic-beta"].split(",")
    assert "interleaved-thinking-2025-05-14" in beta
    assert "server-side-fallback-2026-07-01" in beta
    assert handler.last_request["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
    }
    assert handler.last_request["output_config"] == {"effort": "max"}
    assert handler.last_routing["request_fallbacks"] == "default"
