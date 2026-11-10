"""Provider-side facts kept per turn in ParsedResponse.extras and the step record."""

import logging

from litellm.types.utils import Choices, Message, ModelResponse, Usage

from harbor_proximus import llm
from harbor_proximus.config import ProximusConfig


def _llm() -> llm.LLM:
    config = ProximusConfig(model_info={"max_input_tokens": 1_000_000})
    return llm.LLM("openai/test-model", config, lambda: None, logging.getLogger("test"))


def _response(prompt_details: dict | None, **hidden) -> ModelResponse:
    usage = {"prompt_tokens": 400_000, "completion_tokens": 50, "total_tokens": 400_050}
    if prompt_details is not None:
        usage["prompt_tokens_details"] = prompt_details
    response = ModelResponse(
        id="chatcmpl-abc123",
        model="test-model",
        choices=[
            Choices(
                message=Message(role="assistant", content="ok"), finish_reason="stop"
            )
        ],
        usage=Usage(**usage),
    )
    response._hidden_params = {"response_cost": 0.5, **hidden}
    return response


def test_extras_keep_usage_as_reported_including_cache_writes():
    written = _llm()._parse_response(_response({"cache_write_tokens": 399_997}))
    assert written.cached_tokens == 0
    assert (
        written.extras["usage"]["prompt_tokens_details"]["cache_write_tokens"]
        == 399_997
    )
    assert written.extras["id"] == "chatcmpl-abc123"
    assert "choices" not in written.extras


def test_reported_zero_and_absent_count_stay_distinguishable():
    zero = _llm()._parse_response(_response({"cached_tokens": 0}))
    absent = _llm()._parse_response(_response(None))
    assert zero.extras["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert not (absent.extras["usage"].get("prompt_tokens_details") or {}).get(
        "cached_tokens"
    )


def test_headers_latency_and_request_settings_are_kept():
    parsed = _llm()._parse_response(
        _response(
            {"cached_tokens": 10},
            additional_headers={
                "x-request-id": "req_1",
                "openai-processing-ms": "812",
                "set-cookie": "secret",
            },
            proximus_latency_sec=1.25,
            proximus_request={
                "model": "openai/test-model",
                "extra_body": {"reasoning_effort": "xhigh"},
            },
        )
    )
    assert parsed.extras["headers"] == {
        "x-request-id": "req_1",
        "openai-processing-ms": "812",
    }
    assert parsed.extras["latency_sec"] == 1.25
    assert parsed.extras["request"]["extra_body"] == {"reasoning_effort": "xhigh"}


def test_request_settings_leave_out_messages_tools_and_credentials():
    settings = llm._request_settings(
        {
            "model": "openai/test-model",
            "messages": [{"role": "user", "content": "secret task"}],
            "tools": [{"type": "function"}],
            "api_key": "sk-secret",
            "api_base": "https://gateway.example/api",
            "reasoning_effort": "high",
        }
    )
    assert settings == {
        "model": "openai/test-model",
        "api_base": "https://gateway.example/api",
        "reasoning_effort": "high",
    }
