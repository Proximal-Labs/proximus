"""The content-policy retry: a flagged prompt must not kill the run.

OpenAI answers some large agentic contexts with
``invalid_request_error / code: invalid_prompt``, which litellm surfaces as
ContentPolicyViolationError -- a *subclass of BadRequestError*, and
BadRequestError is on ``query``'s do-not-retry list. That combination made a
single flag fatal.

It should not be. The flag is a nondeterministic classifier fire, not a
malformed request: replaying a flagged payload verbatim usually returns
200 OK. So the policy error is retried, while genuine
BadRequests -- a malformed request retried is a malformed request -- are not.
"""

from __future__ import annotations

import json
import logging

import pytest
from litellm.exceptions import (
    AuthenticationError as LiteLLMAuthenticationError,
)
from litellm.exceptions import (
    BadRequestError as LiteLLMBadRequestError,
)
from litellm.exceptions import (
    ContentPolicyViolationError as LiteLLMContentPolicyViolationError,
)

from harbor_proximus.config import ProximusConfig
from harbor_proximus.llm import (
    LLM,
    LLM_RETRY_MAX_WAIT_SEC,
    ParsedResponse,
    _ProviderHealth,
    _WaitCappedByBudget,
)


def _llm() -> LLM:
    return LLM(
        model_name="openai/gpt-5.4",
        config=ProximusConfig(),
        remaining_sec=lambda: None,
        logger=logging.getLogger("test"),
    )


def _policy_error() -> LiteLLMContentPolicyViolationError:
    return LiteLLMContentPolicyViolationError(
        message=(
            "Invalid prompt: your prompt was flagged as potentially violating "
            "our usage policy."
        ),
        model="gpt-5.4",
        llm_provider="openai",
    )


def _moonshot_image_suspension() -> LiteLLMBadRequestError:
    return LiteLLMBadRequestError(
        message=(
            "MoonshotException - Invalid request: prepare image failed error, "
            "status code: 403, message: Your account is not active, current "
            "state: suspended"
        ),
        model="kimi-k3",
        llm_provider="moonshot",
    )


@pytest.mark.asyncio
async def test_content_policy_flag_is_retried(monkeypatch):
    """Two flags then a success: the run survives on the third attempt."""
    llm = _llm()
    calls = {"n": 0}

    async def fake_acompletion(messages, tools):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _policy_error()
        return "sentinel-response"

    monkeypatch.setattr(llm, "_acompletion", fake_acompletion)
    monkeypatch.setattr(
        llm, "_parse_response", lambda r: ParsedResponse(content=str(r), tool_calls=[])
    )
    # Zero out tenacity's exponential wait so the test does not sleep.
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)

    result = await llm.query(messages=[], tools=None)

    assert calls["n"] == 3, "policy flag should have been retried twice"
    assert result.content == "sentinel-response"


@pytest.mark.asyncio
async def test_moonshot_image_suspension_is_retried(monkeypatch):
    """Moonshot's temporary image-service suspension is a replayable outage."""
    llm = _llm()
    calls = {"n": 0}

    async def fake_acompletion(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _moonshot_image_suspension()
        return "recovered"

    monkeypatch.setattr(llm, "_acompletion", fake_acompletion)
    monkeypatch.setattr(
        llm, "_parse_response", lambda r: ParsedResponse(content=str(r), tool_calls=[])
    )
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)

    result = await llm.query(messages=[], tools=None)

    assert calls["n"] == 2
    assert result.content == "recovered"


@pytest.mark.asyncio
async def test_plain_bad_request_still_fatal(monkeypatch):
    """The exclusion must survive: a real BadRequest is not worth retrying."""
    llm = _llm()
    calls = {"n": 0}

    async def fake_acompletion(messages, tools):
        calls["n"] += 1
        raise LiteLLMBadRequestError(
            message="malformed request", model="gpt-5.4", llm_provider="openai"
        )

    monkeypatch.setattr(llm, "_acompletion", fake_acompletion)
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)

    with pytest.raises(LiteLLMBadRequestError):
        await llm.query(messages=[], tools=None)
    assert calls["n"] == 1, "a plain BadRequest must fail on the first attempt"


@pytest.mark.asyncio
async def test_auth_error_still_fatal(monkeypatch):
    """Retrying a bad key just burns the budget."""
    llm = _llm()
    calls = {"n": 0}

    async def fake_acompletion(messages, tools):
        calls["n"] += 1
        raise LiteLLMAuthenticationError(
            message="bad key", model="gpt-5.4", llm_provider="openai"
        )

    monkeypatch.setattr(llm, "_acompletion", fake_acompletion)
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)

    with pytest.raises(LiteLLMAuthenticationError):
        await llm.query(messages=[], tools=None)
    assert calls["n"] == 1


async def test_api_key_env_indirection(monkeypatch, tmp_path):
    """api_key_env names the variable; the resolved key rides ONLY in the call
    kwargs (litellm's provider-prefix convention would otherwise force every
    OpenAI-compatible endpoint onto the single OPENAI_API_KEY)."""
    import litellm

    from harbor_proximus.config import ProximusConfig
    from harbor_proximus.llm import LLM

    monkeypatch.setenv("META_API_KEY", "sk-meta-test")
    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        raise litellm.exceptions.AuthenticationError("stop here", "openai", "m")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    llm = LLM(
        "openai/test-model",
        ProximusConfig(api_key_env="META_API_KEY"),
        remaining_sec=lambda: None,
        logger=logging.getLogger("t"),
    )
    with pytest.raises(litellm.exceptions.AuthenticationError):
        await llm.query([], tools=None)
    assert captured["api_key"] == "sk-meta-test"


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
    from harbor_proximus.config import ProximusConfig
    from harbor_proximus.llm import LLM

    monkeypatch.delenv("NOPE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NOPE_API_KEY"):
        LLM(
            "openai/x",
            ProximusConfig(api_key_env="NOPE_API_KEY"),
            remaining_sec=lambda: None,
            logger=logging.getLogger("t"),
        )


async def test_retry_loop_stops_when_budget_exhausted(monkeypatch):
    """Per-attempt timeouts are budget-aware but the retry LOOP was not: a
    persistently failing call could hold a turn open for hours past the
    budget. With the budget spent, the retryer must give up immediately."""
    import litellm as _litellm

    calls = {"n": 0}

    async def always_timeout(**kwargs):
        calls["n"] += 1
        raise _litellm.exceptions.Timeout("hang", "openai", "m")

    monkeypatch.setattr(_litellm, "acompletion", always_timeout)
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)

    llm = LLM(
        "openai/gpt-5.4",
        ProximusConfig(time_budget_sec=600),
        remaining_sec=lambda: 0.0,  # budget already spent
        logger=logging.getLogger("t"),
    )
    with pytest.raises(_litellm.exceptions.Timeout):
        await llm.query([], tools=None)
    assert calls["n"] == 1  # no second attempt once the budget is gone


async def test_retry_loop_still_retries_with_budget_left(monkeypatch):
    """Budgeted runs retry until the BUDGET is spent — no attempt cap. A
    provider outage of any length pauses the run instead of killing it
    (a fixed 12-attempt schedule dies on ~1h outages)."""
    import litellm as _litellm

    calls = {"n": 0}

    async def always_timeout(**kwargs):
        calls["n"] += 1
        raise _litellm.exceptions.Timeout("hang", "openai", "m")

    monkeypatch.setattr(_litellm, "acompletion", always_timeout)
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)

    llm = LLM(
        "openai/gpt-5.4",
        ProximusConfig(time_budget_sec=600),
        # The budget drains as attempts burn; spent after the 20th attempt —
        # well past the old 12-attempt cap.
        remaining_sec=lambda: 0.0 if calls["n"] >= 20 else 500.0,
        logger=logging.getLogger("t"),
    )
    with pytest.raises(_litellm.exceptions.Timeout):
        await llm.query([], tools=None)
    assert calls["n"] == 20  # beyond the old cap; stopped by budget, not count


async def test_retry_loop_stops_when_budget_spent(monkeypatch):
    """The retry loop must not outlive the budget: with remaining at zero,
    a retryable failure gets NO second attempt (measured live before the
    fix: a 30-min-budget run sat 100+ minutes inside one turn's retries)."""
    import litellm as _litellm

    calls = {"n": 0}

    async def always_timeout(**kwargs):
        calls["n"] += 1
        raise _litellm.exceptions.Timeout("hang", "openai", "m")

    monkeypatch.setattr(_litellm, "acompletion", always_timeout)
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)

    llm = LLM(
        "openai/gpt-5.4",
        ProximusConfig(time_budget_sec=600),
        remaining_sec=lambda: 0.0,
        logger=logging.getLogger("t"),
    )
    with pytest.raises(_litellm.exceptions.Timeout):
        await llm.query([], tools=None)
    assert calls["n"] == 1


async def test_retry_loop_budget_is_the_only_stop_for_budgeted_runs(monkeypatch):
    """Same contract at a different depth: 35 attempts before the budget
    drains, every one taken."""
    import litellm as _litellm

    calls = {"n": 0}

    async def always_timeout(**kwargs):
        calls["n"] += 1
        raise _litellm.exceptions.Timeout("hang", "openai", "m")

    monkeypatch.setattr(_litellm, "acompletion", always_timeout)
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)

    llm = LLM(
        "openai/gpt-5.4",
        ProximusConfig(time_budget_sec=600),
        remaining_sec=lambda: 0.0 if calls["n"] >= 35 else 100.0,
        logger=logging.getLogger("t"),
    )
    with pytest.raises(_litellm.exceptions.Timeout):
        await llm.query([], tools=None)
    assert calls["n"] == 35


async def test_retry_loop_unbudgeted_runs_keep_full_schedule(monkeypatch):
    import litellm as _litellm

    calls = {"n": 0}

    async def always_timeout(**kwargs):
        calls["n"] += 1
        raise _litellm.exceptions.Timeout("hang", "openai", "m")

    monkeypatch.setattr(_litellm, "acompletion", always_timeout)
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)

    llm = LLM(
        "openai/gpt-5.4",
        ProximusConfig(),  # no budget -> remaining is None -> never budget-stop
        remaining_sec=lambda: None,
        logger=logging.getLogger("t"),
    )
    with pytest.raises(_litellm.exceptions.Timeout):
        await llm.query([], tools=None)
    assert calls["n"] == 12


def test_request_too_large_detection_and_no_retry_classification():
    """Body-cap rejections arrive under many guises (a gateway 413 can be a
    RAW AnthropicError, not a litellm class) — detection is message/status
    based, and even an un-converted raw error is never retried."""
    from harbor_proximus.llm import (
        RequestTooLargeError,
        _is_retryable,
        is_image_limit_error,
        is_request_too_large_error,
    )

    raw = RuntimeError(
        'AnthropicError - {"code":"RequestEntityTooLarge","message":"Request '
        'body size exceeds maximum allowed size: 10485760 bytes"}'
    )
    assert is_request_too_large_error(raw) is True
    assert _is_retryable(raw) is False

    class Status413(Exception):
        status_code = 413

    assert is_request_too_large_error(Status413("nope")) is True
    assert is_request_too_large_error(RuntimeError("some other error")) is False
    image_cap = RuntimeError(
        "request contains 51 images, exceeding the maximum of 50 allowed per request"
    )
    assert is_image_limit_error(image_cap) is True
    assert is_image_limit_error(RuntimeError("some other error")) is False
    assert _is_retryable(RequestTooLargeError("x")) is False


def test_invalid_image_detection_and_no_retry_classification():
    from harbor_proximus.llm import (
        InvalidImageError,
        _is_retryable,
        is_invalid_image_error,
    )

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


def test_responses_unknown_items_is_not_retried():
    """xAI reasoning-only Responses output becomes LiteLLM
    ``APIConnectionError: Unknown items in responses API response``.
    That payload is deterministic — retrying it sleeps the budget."""
    from litellm.exceptions import APIConnectionError as LiteLLMAPIConnectionError

    from harbor_proximus.llm import (
        ProviderRetryBudgetError,
        _is_retryable,
        is_responses_unknown_item_error,
    )

    unknown = LiteLLMAPIConnectionError(
        "Unknown items in responses API response: [ResponseReasoningItem("
        "id='rs_1', summary=[], type='reasoning', encrypted_content='ENC')]",
        "xai",
        "grok-test-model",
    )
    assert is_responses_unknown_item_error(unknown) is True
    assert _is_retryable(unknown) is False
    assert _is_retryable(ProviderRetryBudgetError("budget gone")) is False
    timeout = LiteLLMAPIConnectionError("connection reset", "xai", "m")
    assert is_responses_unknown_item_error(timeout) is False
    assert _is_retryable(timeout) is True


@pytest.mark.asyncio
async def test_responses_unknown_items_fails_on_first_attempt(monkeypatch):
    from litellm.exceptions import APIConnectionError as LiteLLMAPIConnectionError

    llm = _llm()
    calls = {"n": 0}

    async def fake_acompletion_api(**kwargs):
        calls["n"] += 1
        raise LiteLLMAPIConnectionError(
            "Unknown items in responses API response: [ResponseReasoningItem]",
            "xai",
            "grok-test-model",
        )

    monkeypatch.setattr("litellm.acompletion", fake_acompletion_api)
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)

    with pytest.raises(LiteLLMAPIConnectionError, match="Unknown items"):
        await llm.query(messages=[], tools=None)
    assert calls["n"] == 1


def test_reasoning_only_responses_output_keeps_encrypted_cot():
    """LiteLLM's bridge used to emit zero choices for a reasoning-only
    Responses turn, which dropped encrypted_content. The patch turns that
    output into a chat choice so the next request can replay CoT."""
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )
    from openai.types.responses import ResponseReasoningItem

    import harbor_proximus.llm as llm_mod

    llm_mod._install_litellm_reasoning_only_patch()

    assert llm_mod._is_reasoning_output_item(
        ResponseReasoningItem(
            id="rs_1", summary=[], type="reasoning", encrypted_content="ENC"
        )
    )
    item = ResponseReasoningItem(
        id="rs_1",
        summary=[],
        type="reasoning",
        encrypted_content="ENC",
    )
    choices = LiteLLMResponsesTransformationHandler._convert_response_output_to_choices(
        [item]
    )
    assert len(choices) == 1
    stored = choices[0].message.reasoning_items
    assert stored and stored[0]["id"] == "rs_1"
    assert stored[0]["encrypted_content"] == "ENC"

    dict_choices = (
        LiteLLMResponsesTransformationHandler._convert_response_output_to_choices(
            [{"type": "reasoning", "id": "rs_2", "encrypted_content": "ENC2"}]
        )
    )
    assert dict_choices[0].message.reasoning_items[0]["encrypted_content"] == "ENC2"

    # Unrelated unknown items must still produce no choice so LiteLLM can
    # raise rather than invent an assistant turn.
    assert (
        LiteLLMResponsesTransformationHandler._convert_response_output_to_choices(
            [object()]
        )
        == []
    )

    empty = LiteLLMResponsesTransformationHandler._convert_response_output_to_choices(
        []
    )
    assert len(empty) == 1
    assert empty[0].message.content == ""
    assert empty[0].message.role == "assistant"


@pytest.mark.asyncio
async def test_invalid_image_bad_request_is_typed_on_first_attempt(monkeypatch):
    from harbor_proximus.llm import InvalidImageError

    llm = _llm()
    calls = {"n": 0}

    async def fake_acompletion_api(**kwargs):
        calls["n"] += 1
        raise LiteLLMBadRequestError(
            message=(
                'XaiException - {"code":"invalid_image","error":"Image '
                'dimensions 19x2 are too small."}'
            ),
            model="grok-4",
            llm_provider="xai",
        )

    monkeypatch.setattr("litellm.acompletion", fake_acompletion_api)
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)

    with pytest.raises(InvalidImageError, match="19x2"):
        await llm.query(messages=[], tools=None)
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_raw_gateway_413_is_typed_and_not_retried(monkeypatch):
    """Through the real _acompletion conversion: a raw provider 413 surfaces
    as RequestTooLargeError on the FIRST attempt (the agent loop's eviction
    recovery depends on the type; retrying burns budget on a deterministic
    failure)."""
    from harbor_proximus.llm import RequestTooLargeError

    llm = _llm()
    calls = {"n": 0}

    async def fake_acompletion_api(**kwargs):
        calls["n"] += 1
        raise RuntimeError(
            "Request body size exceeds maximum allowed size: 10485760 bytes"
        )

    monkeypatch.setattr("litellm.acompletion", fake_acompletion_api)
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)

    with pytest.raises(RequestTooLargeError):
        await llm.query(messages=[], tools=None)
    assert calls["n"] == 1


def test_moonshot_image_suspension_match_is_narrow():
    from harbor_proximus.llm import (
        _is_retryable,
        is_transient_image_account_error,
    )

    suspended = _moonshot_image_suspension()
    assert is_transient_image_account_error(suspended) is True
    assert _is_retryable(suspended) is True

    other_image_403 = LiteLLMBadRequestError(
        message="prepare image failed error, status code: 403, forbidden URL",
        model="kimi-k3",
        llm_provider="moonshot",
    )
    assert is_transient_image_account_error(other_image_403) is False
    assert _is_retryable(other_image_403) is False


def test_mismapped_google_quota_429_is_retried():
    """litellm 1.97 raises Google 429 RESOURCE_EXHAUSTED as BadRequestError
    (the retryDelay string "14.403979925s" contains "403" and its generic
    403 branch fires first). Quota pressure is what the jittered retry is
    FOR — these must not fail fast like real BadRequests."""
    from harbor_proximus.llm import _is_retryable

    quota = LiteLLMBadRequestError(
        message=(
            'GeminiException BadRequestError - {"error": {"code": 429, '
            '"message": "You exceeded your current quota... Quota exceeded '
            "for metric: generativelanguage.googleapis.com/"
            "generate_content_paid_tier_3_input_token_count, limit: 20000000"
            '... Please retry in 14.403979925s.", '
            '"status": "RESOURCE_EXHAUSTED"}}'
        ),
        model="gemini-3.7-flash",
        llm_provider="gemini",
    )
    assert _is_retryable(quota) is True
    plain = LiteLLMBadRequestError(
        message="malformed request", model="gemini-3.7-flash", llm_provider="gemini"
    )
    assert _is_retryable(plain) is False  # real BadRequests still fail fast


def test_cancellation_is_never_retried():
    """CancelledError/GeneratorExit are teardown control flow, not provider
    failures — retrying them swallows the stop: a stopped run would die
    inside a tenacity retry sleep, skipping auto-submission and the restore."""
    import asyncio

    from harbor_proximus.llm import _is_retryable

    assert _is_retryable(asyncio.CancelledError()) is False
    assert _is_retryable(GeneratorExit()) is False
    assert _is_retryable(KeyboardInterrupt()) is False
