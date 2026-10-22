"""Retry classification: transient provider failures are retried within the
run budget; requests the provider rejected as invalid are not.

OpenAI answers some large agentic contexts with ``invalid_request_error /
code: invalid_prompt``, a 400. That flag is a nondeterministic classifier
fire, not a malformed request: replaying the payload verbatim usually
returns 200 OK. So the policy error is retried, while genuine 400s -- a
malformed request retried is a malformed request -- are not.
"""

from __future__ import annotations

import asyncio
import json
import logging

import anthropic
import httpx2
import pytest

from harbor_proximus.config import ProximusConfig
from harbor_proximus.llm import (
    LLM,
    LLM_RETRY_MAX_WAIT_SEC,
    InvalidImageError,
    ParsedResponse,
    ProviderRetryBudgetError,
    RequestTooLargeError,
    _is_retryable,
    _ProviderHealth,
    _WaitCappedByBudget,
    is_image_limit_error,
    is_invalid_image_error,
    is_request_too_large_error,
    is_transient_image_account_error,
)
from tests.test_proximus_loop import api_error, timeout_error


def _llm(**config) -> LLM:
    return LLM(
        model_name="openai/gpt-5.4",
        config=ProximusConfig(**config),
        remaining_sec=lambda: None,
        logger=logging.getLogger("test"),
    )


def _policy_error():
    return api_error(
        400,
        "Invalid prompt: your prompt was flagged as potentially violating our "
        "usage policy. (code: invalid_prompt)",
    )


def _moonshot_image_suspension():
    return api_error(
        400,
        "Invalid request: prepare image failed error, status code: 403, "
        "message: Your account is not active, current state: suspended",
    )


def _anthropic_error(status: int, message: str) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIStatusError(
        message, response=httpx2.Response(status, request=request), body=None
    )


def _sequence(llm, monkeypatch, outcomes):
    """Make each attempt raise or return the next outcome; count attempts."""
    calls = {"n": 0}

    async def send(call):
        calls["n"] += 1
        outcome = outcomes[min(calls["n"], len(outcomes)) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(llm, "_send", send)
    # Zero out tenacity's exponential wait so the tests do not sleep.
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)
    return calls


async def test_content_policy_flag_is_retried(monkeypatch):
    """Two flags then a success: the run survives on the third attempt."""
    llm = _llm()
    ok = ParsedResponse(content="sentinel-response")
    calls = _sequence(llm, monkeypatch, [_policy_error(), _policy_error(), ok])

    result = await llm.query(messages=[], tools=None)

    assert calls["n"] == 3, "policy flag should have been retried twice"
    assert result.content == "sentinel-response"


async def test_moonshot_image_suspension_is_retried(monkeypatch):
    """Moonshot's temporary image-service suspension is a replayable outage."""
    llm = _llm()
    ok = ParsedResponse(content="recovered")
    calls = _sequence(llm, monkeypatch, [_moonshot_image_suspension(), ok])

    result = await llm.query(messages=[], tools=None)

    assert calls["n"] == 2
    assert result.content == "recovered"


@pytest.mark.parametrize(
    "error",
    [
        api_error(400, "malformed request"),
        api_error(401, "bad key"),
        api_error(404, "no such model"),
        _anthropic_error(400, "messages: roles must alternate"),
    ],
    ids=["openai-400", "openai-401", "openai-404", "anthropic-400"],
)
async def test_rejected_requests_fail_on_the_first_attempt(monkeypatch, error):
    llm = _llm()
    calls = _sequence(llm, monkeypatch, [error])

    with pytest.raises(type(error)):
        await llm.query(messages=[], tools=None)
    assert calls["n"] == 1


@pytest.mark.parametrize(
    "error",
    [
        api_error(429, "rate limited"),
        api_error(500, "server error"),
        timeout_error(),
        _anthropic_error(529, "overloaded"),
    ],
    ids=["openai-429", "openai-500", "timeout", "anthropic-529"],
)
def test_transient_failures_are_retried(error):
    assert _is_retryable(error) is True


def test_quota_errors_are_retried_whatever_their_status():
    """A gateway can relay Google's RESOURCE_EXHAUSTED quota error as a 400;
    quota pressure is what the jittered retry is for."""
    quota = api_error(
        400,
        '{"error": {"code": 429, "message": "Quota exceeded for metric: '
        'generate_content_paid_tier_3_input_token_count", '
        '"status": "RESOURCE_EXHAUSTED"}}',
    )
    assert _is_retryable(quota) is True


def test_missing_keys_fail_up_front(monkeypatch):
    """A missing key fails at construction, never inside the retry loop."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        LLM(
            "openrouter/vendor/model",
            ProximusConfig(model_info={"max_input_tokens": 1000}),
            remaining_sec=lambda: None,
            logger=logging.getLogger("t"),
        )


def test_api_key_env_indirection(monkeypatch):
    """api_key_env names the variable; the key reaches only the SDK client."""
    monkeypatch.setenv("META_API_KEY", "sk-meta-test")
    llm = LLM(
        "openai/test-model",
        ProximusConfig(api_key_env="META_API_KEY"),
        remaining_sec=lambda: None,
        logger=logging.getLogger("t"),
    )
    assert llm.provider.client.api_key == "sk-meta-test"


def test_retry_wait_is_capped_by_max_and_remaining_budget():
    state = object()
    assert (
        _WaitCappedByBudget(lambda _: 9999, lambda: None)(state)
        == LLM_RETRY_MAX_WAIT_SEC
    )
    assert _WaitCappedByBudget(lambda _: 9999, lambda: 17.5)(state) == 17.5
    assert _WaitCappedByBudget(lambda _: 9999, lambda: 0)(state) == 0


def test_provider_health_is_atomic_structured_and_redacts_keys(tmp_path):
    path = tmp_path / "provider_health.json"
    health = _ProviderHealth(path, "moonshot/kimi-k3", logging.getLogger("t"))
    initial = json.loads(path.read_text())
    assert initial["state"] == "healthy"
    assert initial["provider"] == "moonshot"

    health.retrying(RuntimeError("balance exhausted for ak-secret_key"), 9999)
    retrying = json.loads(path.read_text())
    assert retrying["state"] == "retrying"
    assert retrying["consecutive_errors"] == 1
    assert retrying["next_retry_sec"] == 9999
    assert "ak-secret_key" not in retrying["error_message"]

    health.success()
    recovered = json.loads(path.read_text())
    assert recovered["state"] == "healthy"
    assert recovered["consecutive_errors"] == 0
    assert recovered["last_success_at"] is not None
    assert recovered["last_error_at"] is not None


def test_api_key_env_missing_fails_at_construction(monkeypatch):
    monkeypatch.delenv("NOPE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NOPE_API_KEY"):
        LLM(
            "openai/x",
            ProximusConfig(api_key_env="NOPE_API_KEY"),
            remaining_sec=lambda: None,
            logger=logging.getLogger("t"),
        )


def _always_timing_out(monkeypatch, remaining_sec, **config):
    llm = LLM(
        "openai/gpt-5.4",
        ProximusConfig(**config),
        remaining_sec=lambda: remaining_sec(calls["n"]),
        logger=logging.getLogger("t"),
    )
    calls = _sequence(llm, monkeypatch, [timeout_error()])
    return llm, calls


async def test_retry_loop_stops_when_budget_spent(monkeypatch):
    """The retry loop must not outlive the budget: with remaining at zero,
    a retryable failure gets NO second attempt (measured live before the
    fix: a 30-min-budget run sat 100+ minutes inside one turn's retries)."""
    llm, calls = _always_timing_out(monkeypatch, lambda n: 0.0, time_budget_sec=600)
    with pytest.raises(Exception, match="timed out"):
        await llm.query([], tools=None)
    assert calls["n"] == 1


@pytest.mark.parametrize("drained_after", [20, 35])
async def test_retry_loop_budget_is_the_only_stop_for_budgeted_runs(
    monkeypatch, drained_after
):
    """Budgeted runs retry until the BUDGET is spent — no attempt cap. A
    provider outage of any length pauses the run instead of killing it
    (a fixed 12-attempt schedule dies on ~1h outages)."""
    llm, calls = _always_timing_out(
        monkeypatch,
        lambda n: 0.0 if n >= drained_after else 500.0,
        time_budget_sec=600,
    )
    with pytest.raises(Exception, match="timed out"):
        await llm.query([], tools=None)
    assert calls["n"] == drained_after


async def test_retry_loop_unbudgeted_runs_keep_full_schedule(monkeypatch):
    llm, calls = _always_timing_out(monkeypatch, lambda n: None)
    with pytest.raises(Exception, match="timed out"):
        await llm.query([], tools=None)
    assert calls["n"] == 12


def test_request_too_large_detection_and_no_retry_classification():
    """Body-cap rejections arrive under many guises (a gateway 413 can be a
    bare error string) — detection is message/status based, and even an
    unconverted raw error is never retried."""
    raw = RuntimeError(
        '{"code":"RequestEntityTooLarge","message":"Request '
        'body size exceeds maximum allowed size: 10485760 bytes"}'
    )
    assert is_request_too_large_error(raw) is True
    assert _is_retryable(raw) is False
    assert is_request_too_large_error(api_error(413, "too big")) is True
    assert is_request_too_large_error(_anthropic_error(413, "too big")) is True
    assert is_request_too_large_error(RuntimeError("some other error")) is False
    image_cap = RuntimeError(
        "request contains 51 images, exceeding the maximum of 50 allowed per request"
    )
    assert is_image_limit_error(image_cap) is True
    assert is_image_limit_error(RuntimeError("some other error")) is False
    assert _is_retryable(RequestTooLargeError("x")) is False
    assert _is_retryable(ProviderRetryBudgetError("budget gone")) is False


def test_invalid_image_detection_and_no_retry_classification():
    dimensions = RuntimeError(
        '{"code":"invalid_image","error":"Image dimensions 19x2 are too '
        'small. Both width and height must be at least 8 pixels."}'
    )
    pixels = RuntimeError(
        '{"code":"invalid_image","error":"Image has 464 total pixels '
        '(29x16), which is below the minimum of 512 pixels."}'
    )
    assert is_invalid_image_error(dimensions) is True
    assert is_invalid_image_error(pixels) is True
    assert is_invalid_image_error(RuntimeError("image download timed out")) is False
    assert _is_retryable(InvalidImageError("x")) is False


async def test_invalid_image_bad_request_is_typed_on_first_attempt(monkeypatch):
    llm = _llm()
    error = api_error(
        400, '{"code":"invalid_image","error":"Image dimensions 19x2 are too small."}'
    )
    calls = _sequence(llm, monkeypatch, [error])

    with pytest.raises(InvalidImageError, match="19x2"):
        await llm.query(messages=[], tools=None)
    assert calls["n"] == 1


async def test_raw_gateway_413_is_typed_and_not_retried(monkeypatch):
    """Through the real _acompletion conversion: a raw provider 413 surfaces
    as RequestTooLargeError on the FIRST attempt (the agent loop's eviction
    recovery depends on the type; retrying burns budget on a deterministic
    failure)."""
    llm = _llm()
    error = RuntimeError("Request body size exceeds maximum allowed size: 10485760")
    calls = _sequence(llm, monkeypatch, [error])

    with pytest.raises(RequestTooLargeError):
        await llm.query(messages=[], tools=None)
    assert calls["n"] == 1


@pytest.mark.parametrize(
    "message",
    [
        "This model's maximum context length is 128000 tokens.",
        "prompt is too long: 250000 tokens > 200000 maximum",
    ],
)
async def test_context_overflow_is_typed_for_compaction(monkeypatch, message):
    from harbor.llms.base import ContextLengthExceededError

    llm = _llm()
    calls = _sequence(llm, monkeypatch, [api_error(400, message)])

    with pytest.raises(ContextLengthExceededError):
        await llm.query(messages=[], tools=None)
    assert calls["n"] == 1


def test_moonshot_image_suspension_match_is_narrow():
    suspended = _moonshot_image_suspension()
    assert is_transient_image_account_error(suspended) is True
    assert _is_retryable(suspended) is True

    other_image_403 = api_error(
        400, "prepare image failed error, status code: 403, forbidden URL"
    )
    assert is_transient_image_account_error(other_image_403) is False
    assert _is_retryable(other_image_403) is False


def test_cancellation_is_never_retried():
    """CancelledError/GeneratorExit are teardown control flow, not provider
    failures — retrying them swallows the stop: a stopped run would die
    inside a tenacity retry sleep, skipping auto-submission and the restore."""
    assert _is_retryable(asyncio.CancelledError()) is False
    assert _is_retryable(GeneratorExit()) is False
    assert _is_retryable(KeyboardInterrupt()) is False
