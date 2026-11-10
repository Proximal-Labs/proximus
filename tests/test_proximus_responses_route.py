"""use_responses_api: forcing the Responses API route."""

import logging

from harbor_proximus import llm
from harbor_proximus.config import ProximusConfig


def _routed(model: str, **options) -> llm.LLM:
    config = ProximusConfig(model_info={"max_input_tokens": 1_000_000}, **options)
    return llm.LLM(model, config, lambda: None, logging.getLogger("test"))


def test_use_responses_api_forces_the_route_without_reasoning_effort():
    tools = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]
    default = _routed("openai/openai/unknown-reasoner")
    assert default.wire_model_name() == "openai/openai/unknown-reasoner"
    assert default.uses_responses_bridge(tools) is False

    forced = _routed("openai/openai/unknown-reasoner", use_responses_api=True)
    assert forced.wire_model_name() == "openai/responses/openai/unknown-reasoner"
    assert forced.uses_responses_bridge(tools) is True
    assert forced.resolve_context_limit() == 1_000_000


def test_use_responses_api_wire_names():
    assert (
        _routed("unknown-reasoner", use_responses_api=True).wire_model_name()
        == "openai/responses/unknown-reasoner"
    )
    already = _routed("openai/responses/unknown-reasoner", use_responses_api=True)
    assert already.wire_model_name() == "openai/responses/unknown-reasoner"
    claude = _routed("anthropic/claude-test", use_responses_api=True)
    assert claude.wire_model_name() == "anthropic/claude-test"
    assert claude.uses_responses_bridge(None) is False


def test_replayed_reasoning_items_drop_the_summary_and_keep_the_encrypted_content():
    item = {
        "type": "reasoning",
        "id": "rs_1",
        "encrypted_content": "gAAA",
        "summary": [{"type": "summary_text", "text": "Planning the work."}],
    }
    parsed = llm.ParsedResponse(
        reasoning_content="Planning the work.",
        reasoning_items=[item],
        tool_calls=[
            llm.NormalizedToolCall(call_id="call_1", name="bash", raw_arguments="{}")
        ],
    )
    message = _routed(
        "openai/openai/unknown-reasoner", use_responses_api=True
    ).assistant_message(parsed)
    assert message["reasoning_items"] == [
        {"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAA", "summary": []}
    ]
    assert item["summary"], "the parsed response is left as it was"


def test_gateway_invalid_request_is_recognised_only_by_its_wording():
    import litellm

    def bad(message: str) -> Exception:
        return litellm.exceptions.BadRequestError(
            message=message, model="m", llm_provider="openai"
        )

    gateway = bad(
        '{"error":{"message":"The upstream model rejected the request as invalid. Check the request parameters and content."}}'
    )
    assert llm._is_gateway_invalid_request(gateway)
    assert not llm._is_retryable(gateway), (
        "only the bounded retry in query() retries it"
    )
    assert not llm._is_gateway_invalid_request(bad("Unknown parameter: foo"))


def _gateway_error() -> Exception:
    import litellm

    return litellm.exceptions.BadRequestError(
        message='{"error":{"message":"The upstream model rejected the request as invalid."}}',
        model="m",
        llm_provider="openai",
    )


async def test_gateway_invalid_request_is_retried_a_bounded_number_of_times(
    monkeypatch,
):
    forced = _routed("openai/openai/unknown-reasoner", use_responses_api=True)
    calls = {"n": 0}

    async def flaky(messages, tools):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _gateway_error()
        return object()

    monkeypatch.setattr(forced, "_acompletion", flaky)
    monkeypatch.setattr(
        forced, "_parse_response", lambda response: llm.ParsedResponse(content="ok")
    )
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)
    parsed = await forced.query(messages=[], tools=None)
    assert parsed.content == "ok" and calls["n"] == 3 and parsed.extras["attempts"] == 3

    async def always(messages, tools):
        calls["n"] += 1
        raise _gateway_error()

    calls["n"] = 0
    monkeypatch.setattr(forced, "_acompletion", always)
    import pytest

    with pytest.raises(Exception, match="rejected the request as invalid"):
        await forced.query(messages=[], tools=None)
    assert calls["n"] == llm.GATEWAY_INVALID_REQUEST_RETRIES + 1


async def test_gateway_invalid_request_stays_fatal_on_chat_completions(monkeypatch):
    chat = _routed("openai/openai/unknown-reasoner")
    calls = {"n": 0}

    async def always(messages, tools):
        calls["n"] += 1
        raise _gateway_error()

    monkeypatch.setattr(chat, "_acompletion", always)
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)
    import pytest

    with pytest.raises(Exception, match="rejected the request as invalid"):
        await chat.query(messages=[], tools=None)
    assert calls["n"] == 1
