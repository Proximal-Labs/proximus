"""LiteLLM configuration and transport."""

import copy
import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast, override

import litellm
import tenacity
from harbor.agents.model_connection import (
    PROVIDERS,
    ModelConnectionSpec,
    resolve_model_connection,
)
from harbor.llms.base import ContextLengthExceededError, OutputLengthExceededError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config
from litellm.types.utils import ModelResponse

from harbor_proximus import gemini_interactions
from harbor_proximus.config import BUDGET_MARGIN_SEC, ProximusConfig
from harbor_proximus.prompts import NO_OUTPUT_PLACEHOLDER

_MODEL_REGISTRATION_LOCK = threading.Lock()


def _register_model_info(model_info: dict[str, dict[str, Any]]) -> None:
    """Register model metadata without LiteLLM's swallowed lookup noise.

    Providerless entries are required to price provider responses that echo a
    bare model name. LiteLLM probes those names before registration, prints its
    provider-list hint, then catches the resulting lookup error internally.
    """
    if not model_info:
        return
    with _MODEL_REGISTRATION_LOCK:
        suppress_debug_info = litellm.suppress_debug_info
        try:
            litellm.suppress_debug_info = True
            litellm.register_model(model_info)
        finally:
            litellm.suppress_debug_info = suppress_debug_info


# Providers whose credential in Harbor's PROVIDERS table is not a bearer API key.
_NON_BEARER_PROVIDERS = frozenset({"amazon-bedrock", "sagemaker", "vertex_ai"})
assert _NON_BEARER_PROVIDERS <= set(PROVIDERS)


def _litellm_knows(model: str) -> bool:
    try:
        litellm.get_model_info(model)
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class Credentials:
    api_key: str | None
    api_base: str | None
    consumed_env_keys: frozenset[str]
    """Agent-env variables used for the model, to hide from the sandbox."""


def resolve_credentials(
    model_name: str, api_key_env: str | None, extra_env: dict[str, str] | None
) -> Credentials:
    """The model's API key and base URL, as Harbor's own agents resolve them.

    Harbor's agent env (`--ae`, AgentConfig.env) wins over the host, with the
    provider's usual variable names; host variables stay with LiteLLM's own
    lookup. ``api_key_env`` names the key variable explicitly (agent env, then
    host) and must be set. Bedrock/SageMaker/Vertex "keys" in Harbor's table
    are not bearer tokens, so those providers keep LiteLLM's lookup.
    """
    agent_env = extra_env or {}
    used: list[str] = []

    def from_agent_env(*names: str) -> tuple[str, str] | None:
        for name in names:
            if name in agent_env:
                used.append(name)
                return name, agent_env[name]
        return None

    connection = resolve_model_connection(
        model_name, ModelConnectionSpec(), from_agent_env
    )
    cloud_auth = connection.provider in _NON_BEARER_PROVIDERS
    consumed = {
        name for name in used if not cloud_auth or name.endswith(("_BASE", "_URL"))
    }
    api_key = None if cloud_auth else connection.api_key or None
    if api_key_env:
        api_key = agent_env.get(api_key_env, os.environ.get(api_key_env)) or None
        if api_key is None:
            raise ValueError(
                f"api_key_env={api_key_env!r} is not set in the "
                "environment — the run cannot authenticate this model."
            )
        if api_key_env in agent_env:
            consumed.add(api_key_env)
    return Credentials(api_key, connection.configured_base_url, frozenset(consumed))


# Provider error strings that indicate context-window overflow
# Adapted from Harbor's LiteLLM context-length check
_CONTEXT_LENGTH_ERROR_PHRASES = (
    "context length exceeded",
    "context_length_exceeded",
    "maximum context length",
    "`inputs` tokens + `max_new_tokens`",
    "model's context length",  # vllm 0.16.0 error message
    "prompt is too long",  # Anthropic via OpenAI-compatible proxies
    "input is too long for requested model",  # Bedrock via proxy
)


def is_context_length_error(error: Exception) -> bool:
    """Check provider error payloads for context-length overflow signals."""
    parts = [
        str(error),
        str(getattr(error, "body", "")),
        str(getattr(error, "message", "")),
        str(getattr(error, "error", "")),
    ]
    combined = " ".join(part.lower() for part in parts if part)
    return any(phrase in combined for phrase in _CONTEXT_LENGTH_ERROR_PHRASES)


# Match provider error text to classify request-size and image-limit errors.
_REQUEST_TOO_LARGE_PHRASES = (
    "request body size exceeds",  # Alibaba's 413 error code
    "exceeded limit on max bytes to request body",
    "badrequest.toolarge",
    "requestentitytoolarge",
    "request entity too large",
    "payload too large",
    "request_too_large",  # OpenAI's 413 error code
)
_IMAGE_LIMIT_PHRASES = (
    "images, exceeding the maximum of",  # a provider: 51 images, max 50
    "images exceeding the maximum of",
    "too many images",
)
_INVALID_IMAGE_CODE = "invalid_image"


class RequestTooLargeError(Exception):
    """The serialized request exceeded a provider/gateway body-size cap.

    Deterministic for a given history — never retried; the agent loop
    recovers by evicting attached images and informing the model."""


class ImageLimitExceededError(RequestTooLargeError):
    """The request exceeded a provider's hard image-count limit.

    Unlike a byte cap, retaining nearly the maximum number of images invites
    another immediate failure. The agent compacts the image-bearing history.
    """


class InvalidImageError(Exception):
    """A provider rejected an attached image as malformed or out of bounds.

    The unchanged request is deterministic, so the agent loop removes the
    rejected image payload and relays the provider error to the model. The
    model can then repair the source file and call ``view_image`` again.
    """


class ProviderRetryBudgetError(RuntimeError):
    """The run budget ended while still retrying a provider error.

    That is a failed trial, not a clean completion: the model never
    finished the turn, so Harbor must record an exception rather than
    pin a partial workspace as ``done``.
    """


def _error_text(error: BaseException) -> str:
    parts = [
        str(error),
        str(getattr(error, "body", "")),
        str(getattr(error, "message", "")),
        str(getattr(error, "error", "")),
    ]
    return " ".join(part.lower() for part in parts if part)


def is_image_limit_error(error: BaseException) -> bool:
    """Check provider payloads for hard image-count limit rejections."""
    combined = _error_text(error)
    return any(phrase in combined for phrase in _IMAGE_LIMIT_PHRASES)


def is_invalid_image_error(error: BaseException) -> bool:
    """Check provider payloads for an explicit invalid-image error code."""
    return _INVALID_IMAGE_CODE in _error_text(error)


def is_request_too_large_error(error: BaseException) -> bool:
    if getattr(error, "status_code", None) == 413:
        return True
    combined = _error_text(error)
    return any(phrase in combined for phrase in _REQUEST_TOO_LARGE_PHRASES)


def is_transient_image_account_error(error: BaseException) -> bool:
    """Detect Moonshot's temporary image-service account suspension."""
    combined = _error_text(error)
    return all(
        marker in combined
        for marker in (
            "prepare image failed",
            "status code: 403",
            "current state: suspended",
        )
    )


def is_responses_unknown_item_error(error: BaseException) -> bool:
    """LiteLLM raises this when a Responses turn has no message/tool choice.

    xAI sometimes returns only ``ResponseReasoningItem`` (encrypted CoT, no
    message). LiteLLM then emits zero choices and maps the ValueError to
    ``APIConnectionError``, which the retry loop treated as a transient
    outage. Replaying the same response never grows a message, so retries
    just sleep the remaining budget in backoff.
    """
    return "unknown items in responses api response" in _error_text(error)


def _is_retryable(error: BaseException) -> bool:
    """ContentPolicyViolationError is retryable despite subclassing
    BadRequestError: OpenAI's `invalid_prompt` fires nondeterministically
    and a verbatim replay usually passes."""
    if not isinstance(error, Exception):
        # BaseException-only classes (CancelledError, GeneratorExit,
        # KeyboardInterrupt): cancellation/teardown control flow, never a
        # provider failure. tenacity's retry predicate sees BaseExceptions,
        # and retrying a CancelledError swallows the STOP: a stopped run would
        # sit in a tenacity sleep ("Retrying ... as it raised
        # CancelledError") until the SIGKILL, skipping auto-submission and
        # the end-of-run restore.
        return False
    if isinstance(error, litellm.exceptions.ContentPolicyViolationError):
        return True
    if isinstance(
        error, litellm.exceptions.BadRequestError
    ) and is_transient_image_account_error(error):
        # Moonshot's image-preparation service can temporarily suspend an
        # otherwise-working account and then recover; litellm maps its
        # nested 403 to a fatal BadRequestError.
        return True
    if isinstance(error, litellm.exceptions.BadRequestError) and any(
        marker in str(error) for marker in ("RESOURCE_EXHAUSTED", "Quota exceeded")
    ):
        # litellm 1.97 mismaps Google 429 quota errors to BadRequestError:
        # the response's retryDelay (e.g. "14.403979925s") contains "403" and
        # litellm's `elif "403" in error_str` fires before its quota branch.
        # These are rate limits — exactly what the jittered retry exists for.
        return True
    if is_request_too_large_error(error):
        # Deterministic for a given history (verbatim replays carry the same
        # oversized body) — the agent loop recovers by evicting images. The
        # message check also covers raw provider exceptions that litellm
        # re-raises unmapped (the token-plan 413 arrived as AnthropicError).
        return False
    if isinstance(error, ProviderRetryBudgetError):
        return False
    if is_responses_unknown_item_error(error):
        # Same response, same missing message/tool
        return False
    return not isinstance(
        error,
        (
            ContextLengthExceededError,
            OutputLengthExceededError,
            RequestTooLargeError,
            InvalidImageError,
            litellm.exceptions.AuthenticationError,
            litellm.exceptions.BadRequestError,
            litellm.exceptions.NotFoundError,
        ),
    )


def _is_reasoning_output_item(item: Any) -> bool:
    if isinstance(item, dict):
        return item.get("type") == "reasoning"
    return getattr(item, "type", None) == "reasoning" or type(item).__name__ == (
        "ResponseReasoningItem"
    )


def _reasoning_item_dict(item: Any) -> dict[str, Any]:
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        _build_reasoning_item,
    )

    if isinstance(item, dict):
        return _build_reasoning_item(
            item_id=str(item.get("id") or ""),
            encrypted_content=item.get("encrypted_content"),
            summary_raw=item.get("summary"),
        )
    return _build_reasoning_item(
        item_id=str(getattr(item, "id", "") or ""),
        encrypted_content=getattr(item, "encrypted_content", None),
        summary_raw=getattr(item, "summary", None),
    )


def _choices_from_reasoning_only_output(output_items: list[Any]) -> list[Any]:
    """A Responses turn that is only reasoning still has a CoT to replay.

    LiteLLM's chat bridge holds ``ResponseReasoningItem`` until a following
    message or tool call, then raises if neither arrives. xAI does emit
    reasoning-only turns; dropping them loses encrypted_content and the
    next request starts without the model's own chain of thought.
    """
    from litellm.types.llms.openai import ChatCompletionReasoningItem
    from litellm.types.utils import Choices, Message

    reasoning_items = [_reasoning_item_dict(item) for item in output_items]
    summary_text = " ".join(
        str(part.get("text") or "")
        for item in reasoning_items
        for part in (item.get("summary") or [])
        if isinstance(part, dict) and part.get("text")
    )
    message = Message(
        role="assistant",
        content="",
        reasoning_content=summary_text or None,
        reasoning_items=cast(list[ChatCompletionReasoningItem] | None, reasoning_items),
    )
    return [Choices(message=message, finish_reason="stop", index=0)]


def _install_litellm_reasoning_only_patch() -> None:
    """Accept reasoning-only Responses output so encrypted CoT is not lost."""
    try:
        from litellm.completion_extras.litellm_responses_transformation.transformation import (
            LiteLLMResponsesTransformationHandler,
        )
    except ImportError:
        return
    orig = LiteLLMResponsesTransformationHandler._convert_response_output_to_choices
    if getattr(orig, "_proximus_reasoning_only", False):
        return

    def wrapped(
        output_items: list[Any],
        handle_raw_dict_callback: Callable | None = None,
    ) -> list[Any]:
        choices = orig(output_items, handle_raw_dict_callback=handle_raw_dict_callback)
        if choices:
            return choices
        items = list(output_items or [])
        if items and all(_is_reasoning_output_item(item) for item in items):
            return _choices_from_reasoning_only_output(items)
        if not items:
            # xAI also emits a Responses turn with zero output items.
            # LiteLLM then raises ``Unknown items in responses API response: []``
            # as APIConnectionError. Treat it as an empty assistant turn so the
            # loop can stall/continue instead of failing a multi-hour trial
            # after auto-submit.
            from litellm.types.utils import Choices, Message

            return [
                Choices(
                    message=Message(role="assistant", content=""),
                    finish_reason="stop",
                    index=0,
                )
            ]
        return choices

    wrapped._proximus_reasoning_only = True  # type: ignore[attr-defined]
    LiteLLMResponsesTransformationHandler._convert_response_output_to_choices = (
        staticmethod(wrapped)
    )


# 1h TTL: tool calls regularly outlive Anthropic's default 5m cache.
CACHE_CONTROL: dict[str, Any] = {"type": "ephemeral", "ttl": "1h"}
# Twelve attempts for unbudgeted runs. Budgeted runs keep retrying until their
# wall-clock budget is spent, but every individual sleep is bounded: without an
# explicit max, Tenacity's exponential schedule can reach hours during a long
# provider outage, leaving runs falsely looking healthy.
LLM_RETRIES = 12
LLM_RETRY_MAX_WAIT_SEC = 300.0
# Per request timeout: 30 mins
LLM_TIMEOUT_SEC = 1800.0
_SECRET_TOKEN_RE = re.compile(r"\b(?:ak|sk)-[A-Za-z0-9_-]+\b")


class _WaitCappedByBudget(tenacity.wait.wait_base):
    """Bound exponential backoff and never sleep past the run budget."""

    def __init__(self, wait: Any, remaining_sec: Callable[[], float | None]):
        self._wait = wait
        self._remaining_sec = remaining_sec

    @override
    def __call__(self, retry_state: tenacity.RetryCallState) -> float:
        delay = self._wait(retry_state) if callable(self._wait) else self._wait
        remaining = self._remaining_sec()
        if remaining is not None:
            delay = min(float(delay), max(0.0, float(remaining)))
        return max(0.0, min(float(delay), LLM_RETRY_MAX_WAIT_SEC))


class _ProviderHealth:
    """Atomic, machine-readable transport health for external monitoring."""

    def __init__(self, path: Path | None, model: str, logger: logging.Logger):
        self._path = path
        self._logger = logger
        self._data: dict[str, Any] = {
            "schema_version": 1,
            "model": model,
            "provider": model.split("/", 1)[0],
            "state": "healthy",
            "updated_at": time.time(),
            "last_success_at": None,
            "last_error_at": None,
            "retry_started_at": None,
            "consecutive_errors": 0,
            "next_retry_sec": None,
            "error_type": None,
            "error_message": None,
        }
        self._write()

    @staticmethod
    def _safe_message(error: BaseException) -> str:
        return _SECRET_TOKEN_RE.sub(lambda m: m.group(0)[:3] + "***", str(error))[:1000]

    def _write(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._data, sort_keys=True))
            tmp.replace(self._path)
        except OSError as error:
            self._logger.debug("Could not write provider health: %s", error)

    def retrying(self, error: BaseException, next_retry_sec: float) -> None:
        now = time.time()
        if self._data["state"] != "retrying":
            self._data["retry_started_at"] = now
        self._data.update(
            state="retrying",
            updated_at=now,
            last_error_at=now,
            consecutive_errors=int(self._data["consecutive_errors"]) + 1,
            next_retry_sec=round(max(0.0, next_retry_sec), 3),
            error_type=type(error).__name__,
            error_message=self._safe_message(error),
        )
        self._write()

    def success(self) -> None:
        now = time.time()
        self._data.update(
            state="healthy",
            updated_at=now,
            last_success_at=now,
            retry_started_at=None,
            consecutive_errors=0,
            next_retry_sec=None,
            error_type=None,
            error_message=None,
        )
        self._write()

    def failed(self, error: BaseException) -> None:
        now = time.time()
        self._data.update(
            state="failed",
            updated_at=now,
            last_error_at=now,
            next_retry_sec=None,
            error_type=type(error).__name__,
            error_message=self._safe_message(error),
        )
        self._write()


@dataclass
class NormalizedToolCall:
    call_id: str
    name: str
    raw_arguments: str


@dataclass
class ParsedResponse:
    content: str = ""
    reasoning_content: str | None = None
    thinking_blocks: Any | None = None
    # OpenAI Responses-API reasoning items (opaque, encrypted); replayed on
    # the next turn so the model keeps its own chain of thought.
    reasoning_items: Any | None = None
    # Provider-owned context required on later turns. Gemini uses this field
    # for thought signatures, including signatures attached to plain text.
    provider_specific_fields: dict[str, Any] | None = None
    tool_calls: list[NormalizedToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Thinking tokens inside completion_tokens, when the provider itemizes
    # them (Gemini thoughtsTokenCount, OpenAI reasoning_tokens). 0 otherwise.
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    # The concrete model that served this turn. This differs from the
    # requested handle when Anthropic's server-side refusal fallback runs.
    response_model: str | None = None
    fallback_attempted: bool = False
    fallback_used: bool = False
    fallback_iterations: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None
    stop_details: dict[str, Any] | None = None
    request_thinking: dict[str, Any] | None = None
    request_output_config: dict[str, Any] | None = None
    request_fallbacks: str | None = None

    @property
    def is_empty(self) -> bool:
        """No text, no reasoning, no tool calls."""
        return not (
            self.tool_calls
            or self.content.strip()
            or (self.reasoning_content or "").strip()
            or self.thinking_blocks
            or self.reasoning_items
        )


type Message = dict[str, Any]


class _AnthropicFallbackHTTPHandler(AsyncHTTPHandler):
    """Inject Anthropic's beta fallback field and retain routing provenance.

    LiteLLM 1.97 reserves the top-level ``fallbacks`` kwarg for its own
    client-side list and serializes ``extra_body`` literally on Anthropic,
    so neither route can express the API's string-valued ``fallbacks:
    "default"``. Inject at the provider boundary until LiteLLM supports it.
    """

    def __init__(self, fallbacks: str):
        super().__init__()
        self.fallbacks = fallbacks
        self.last_routing: dict[str, Any] = {}
        self.last_request: dict[str, Any] = {}

    async def post(self, url: str, **kwargs: Any):
        body = kwargs.get("json")
        if isinstance(body, dict):
            body = copy.deepcopy(body)
            body["fallbacks"] = self.fallbacks
            kwargs["json"] = body
            self.last_request = {
                "model": body.get("model"),
                "thinking": body.get("thinking"),
                "output_config": body.get("output_config"),
                "fallbacks": body.get("fallbacks"),
                "max_tokens": body.get("max_tokens"),
            }
        headers = dict(kwargs.get("headers") or {})
        beta = {
            value.strip()
            for value in headers.get("anthropic-beta", "").split(",")
            if value.strip()
        }
        beta.add("server-side-fallback-2026-07-01")
        headers["anthropic-beta"] = ",".join(sorted(beta))
        kwargs["headers"] = headers
        response = await super().post(url, **kwargs)
        routing = self.routing_from_response(response)
        routing.update(
            {
                "request_thinking": self.last_request.get("thinking"),
                "request_output_config": self.last_request.get("output_config"),
                "request_fallbacks": self.last_request.get("fallbacks"),
            }
        )
        self.last_routing = routing
        return response

    @staticmethod
    def routing_from_response(response: Any) -> dict[str, Any]:
        try:
            body = response.json()
        except Exception:  # noqa: BLE001 - telemetry must not fail a turn
            return {}
        usage = body.get("usage") if isinstance(body, dict) else None
        raw_iterations = usage.get("iterations") if isinstance(usage, dict) else None
        output_details = (
            usage.get("output_tokens_details") if isinstance(usage, dict) else None
        )
        iterations = [
            {
                "type": item.get("type"),
                "model": item.get("model"),
                "input_tokens": item.get("input_tokens"),
                "output_tokens": item.get("output_tokens"),
            }
            for item in (raw_iterations or [])
            if isinstance(item, dict)
        ]
        stop_reason = body.get("stop_reason")
        fallback_attempted = any(
            item.get("type") == "fallback_message" for item in iterations
        )
        return {
            "response_model": body.get("model"),
            "fallback_attempted": fallback_attempted,
            "fallback_used": (stop_reason != "refusal" and fallback_attempted),
            "fallback_iterations": iterations,
            "thinking_tokens": (
                int(output_details.get("thinking_tokens") or 0)
                if isinstance(output_details, dict)
                else 0
            ),
            "stop_reason": stop_reason,
            "stop_details": body.get("stop_details"),
        }


def drop_orphan_tool_results(messages: list[Message]) -> list[Message]:
    """A tool result is valid only directly after the assistant turn whose
    tool_calls produced it"""
    open_ids: set[str] = set()
    kept: list[Message] = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            if message.get("tool_call_id") not in open_ids:
                continue
        else:
            calls = message.get("tool_calls") if role == "assistant" else None
            open_ids = {call["id"] for call in calls or []}
        kept.append(message)
    return kept


class _StopWhenBudgetSpent(tenacity.stop.stop_base):
    """Stop retrying once the run's remaining budget is spent (None = no
    budget = never stops on this condition)."""

    def __init__(self, remaining_sec: Callable[[], float | None]):
        self._remaining_sec = remaining_sec

    @override
    def __call__(self, retry_state: tenacity.RetryCallState) -> bool:
        remaining = self._remaining_sec()
        return remaining is not None and remaining <= 0


class LLM:
    """One model endpoint. ``remaining_sec`` (None = no budget) caps the
    per-call timeout so a hung request plus retry can't swallow the run."""

    def __init__(
        self,
        model_name: str,
        config: ProximusConfig,
        remaining_sec: Callable[[], float | None],
        logger: logging.Logger,
        provider_health_path: Path | None = None,
        *,
        extra_env: dict[str, str] | None = None,
    ):
        if config.model_api == "gemini_interactions" and config.temperature is None:
            raise ValueError("Gemini Interactions requires temperature")
        _install_litellm_reasoning_only_patch()
        self.model_name = model_name
        self._config = config
        self._remaining_sec = remaining_sec
        self._logger = logger
        self._provider_health = _ProviderHealth(
            provider_health_path, model_name, logger
        )
        if config.litellm_model_info:
            _register_model_info(config.litellm_model_info)
        if config.model_info:
            # Providerless too (responses echo the bare name, which prices
            # them), unless LiteLLM already knows that name for another route.
            names = {model_name}
            bare = model_name.split("/", 1)[-1]
            if bare != model_name and not _litellm_knows(bare):
                names.add(bare)
            _register_model_info({name: config.model_info for name in names})
        credentials = resolve_credentials(model_name, config.api_key_env, extra_env)
        self._api_key: str | None = credentials.api_key
        self._api_base: str | None = credentials.api_base
        # Proximus.run blanks these in the sandbox: Harbor overlays the agent
        # env onto sandbox commands.
        self.consumed_env_keys: frozenset[str] = credentials.consumed_env_keys
        # Set by the agent once the run's session id exists; scopes the
        # provider-side prompt cache to this run.
        self.session_id: str = ""
        self._interactions: gemini_interactions.InteractionsClient | None = None
        if config.model_api == "gemini_interactions":
            if not model_name.startswith("gemini/"):
                raise ValueError(
                    "model_api='gemini_interactions' needs a gemini/ model, "
                    f"not {model_name!r}"
                )
            self._interactions = gemini_interactions.InteractionsClient(
                model_name, self._api_key, config.api_base or self._api_base
            )

    def resolve_context_limit(self) -> int:
        error = f"No valid max_input_tokens for {self.model_name!r}."
        try:
            info = litellm.get_model_info(self.model_name)
        except Exception as e:
            raise ValueError(error) from e
        limit = info.get("max_input_tokens")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError(error)
        return limit

    def supports_vision(self) -> bool:
        try:
            return litellm.supports_vision(model=self.model_name)
        except Exception as e:
            self._logger.debug(f"Could not determine vision support: {e}")
            return False

    def is_anthropic(self) -> bool:
        return any(x in self.model_name.lower() for x in ("anthropic", "claude"))

    def is_xai(self) -> bool:
        if "xai" in self.model_name.lower():
            return True
        try:
            return (
                litellm.get_model_info(self.model_name).get("litellm_provider") == "xai"
            )
        except Exception as e:
            self._logger.debug(f"Could not determine provider: {e}")
            return False

    def uses_responses_bridge(self, tools: list[dict[str, Any]] | None) -> bool:
        """Whether litellm will route this request to a Responses API.

        Registry-declared ``mode=responses`` models always bridge, including
        non-openai namespaces. Otherwise the gpt-5.4+ OpenAI heuristic still
        applies. Responses-only params
        400 on /v1/chat/completions, so this must match litellm's own gate.
        """
        if self.is_anthropic() or not self._config.reasoning_effort:
            return False
        try:
            if litellm.get_model_info(self.model_name).get("mode") == "responses":
                return True
        except Exception as e:
            self._logger.debug(f"Could not determine model mode: {e}")
        model = self.model_name.lower()
        if "openai" not in model and "/" in model:
            return False  # some other provider's namespace
        if tools is None:
            return False
        return bool(OpenAIGPT5Config.is_model_gpt_5_4_plus_model(model))

    async def query(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
    ) -> ParsedResponse:
        stops: list[tenacity.stop.stop_base] = [
            _StopWhenBudgetSpent(self._remaining_sec)
        ]
        if self._remaining_sec() is None:
            stops.append(tenacity.stop_after_attempt(LLM_RETRIES))
        log_before_sleep = tenacity.before_sleep_log(self._logger, logging.WARNING)

        def before_sleep(retry_state: tenacity.RetryCallState) -> None:
            error = retry_state.outcome.exception() if retry_state.outcome else None
            sleep = retry_state.next_action.sleep if retry_state.next_action else 0.0
            if error is not None:
                self._provider_health.retrying(error, sleep)
            log_before_sleep(retry_state)

        retryer = tenacity.AsyncRetrying(
            stop=tenacity.stop_any(*stops),
            wait=_WaitCappedByBudget(
                tenacity.wait_random_exponential(
                    multiplier=2, min=4, max=LLM_RETRY_MAX_WAIT_SEC
                ),
                self._remaining_sec,
            ),
            retry=tenacity.retry_if_exception(_is_retryable),
            # The warning remains useful locally; provider_health.json in the
            # agent logs carries the same state for external monitoring.
            before_sleep=before_sleep,
            reraise=True,
        )
        try:
            response = await retryer(self._acompletion, messages, tools)
        except BaseException as error:
            if isinstance(error, Exception):
                self._provider_health.failed(error)
            raise
        self._provider_health.success()
        return self._parse_response(response)

    async def _acompletion(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
    ) -> ModelResponse:
        timeout = LLM_TIMEOUT_SEC
        remaining = self._remaining_sec()
        if remaining is not None:
            timeout = min(timeout, max(float(BUDGET_MARGIN_SEC), remaining * 0.5))
        bridged = self.uses_responses_bridge(tools)
        messages_for_call = self._apply_prompt_caching(messages)
        if bridged and self.is_xai():
            # LiteLLM's chat→Responses bridge lifts string system messages
            # into `instructions`, and XAIResponsesAPIConfig then drops that
            # field. List content stays an input item (xAI accepts role=
            # system on /v1/responses).
            messages_for_call = self._xai_system_as_input_items(messages_for_call)
        completion_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages_for_call,
            "drop_params": True,
            "timeout": timeout,
        }
        if bridged:
            # Must ride in extra_body: litellm silently drops unknown
            # top-level params. store=False + include encrypted reasoning
            # makes client-managed replay possible on OpenAI and xAI;
            # "auto" summaries cannot 400; prompt_cache_key is the
            # Responses sticky-routing field (xAI: same role as
            # x-grok-conv-id on Chat Completions).
            extra_body: dict[str, Any] = {
                "include": ["reasoning.encrypted_content"],
                "store": False,
                "reasoning_summary": "auto",
            }
            if self.session_id:
                extra_body["prompt_cache_key"] = self.session_id
            completion_kwargs["extra_body"] = extra_body
        # Deliberately no tool_choice, ever: Anthropic's hierarchical cache
        # invalidation would flush the messages cache if it changed between
        # a normal turn and the compaction summary call.
        if tools is not None:
            completion_kwargs["tools"] = tools
        if self._config.reasoning_effort is not None:
            completion_kwargs["reasoning_effort"] = self._config.reasoning_effort
        if self._api_key is not None:
            completion_kwargs["api_key"] = self._api_key
        if self._api_base is not None:
            completion_kwargs["api_base"] = self._api_base
        if self._config.api_base is not None:
            completion_kwargs["api_base"] = self._config.api_base
        if self._config.temperature is not None:
            completion_kwargs["temperature"] = self._config.temperature
        if self._config.max_thinking_tokens is not None and self.is_anthropic():
            completion_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._config.max_thinking_tokens,
            }
        completion_kwargs.update(self._config.llm_call_kwargs)
        fallback_handler: _AnthropicFallbackHTTPHandler | None = None
        server_fallback = completion_kwargs.pop("anthropic_fallbacks", None)
        if server_fallback is not None:
            if not self.is_anthropic():
                raise ValueError("anthropic_fallbacks requires an Anthropic model")
            fallback_handler = _AnthropicFallbackHTTPHandler(str(server_fallback))
            completion_kwargs["client"] = fallback_handler

        try:
            if self._interactions is not None:
                request = gemini_interactions.build_request(
                    self.model_name,
                    messages,
                    tools,
                    reasoning_effort=self._config.reasoning_effort,
                    temperature=self._config.temperature,
                    call_kwargs={"timeout": timeout, **self._config.llm_call_kwargs},
                )
                response = await self._interactions.create(request)
            else:
                raw_response = await litellm.acompletion(**completion_kwargs)
                if completion_kwargs.get("stream"):
                    # Streaming keeps long reasoning turns active at gateways which
                    # otherwise buffer the full Anthropic response and time it out.
                    # Rebuild the ordinary ModelResponse so all downstream parsing
                    # (reasoning blocks, partial tool JSON, usage/cache telemetry,
                    # finish reasons, and cost) follows the same path as before.
                    chunks = [chunk async for chunk in raw_response]
                    built = litellm.stream_chunk_builder(chunks, messages=messages)
                    if built is None:
                        raise RuntimeError(
                            f"Model {self.model_name} returned an empty stream."
                        )
                    response = cast(ModelResponse, built)
                else:
                    response = cast(ModelResponse, raw_response)
        except litellm.exceptions.ContextWindowExceededError as e:
            raise ContextLengthExceededError from e
        except litellm.exceptions.BadRequestError as e:
            if is_context_length_error(e):
                raise ContextLengthExceededError from e
            if is_image_limit_error(e):
                raise ImageLimitExceededError(str(e)) from e
            if is_invalid_image_error(e):
                raise InvalidImageError(str(e)) from e
            if is_request_too_large_error(e):
                raise RequestTooLargeError(str(e)) from e
            raise
        except Exception as e:  # noqa: BLE001 — typed, then re-raised
            # Size/count rejections arrive under many guises (the token-plan
            # 413 surfaced as a raw AnthropicError, not a litellm class) —
            # normalize them before the agent loop chooses its recovery path.
            if is_image_limit_error(e):
                raise ImageLimitExceededError(str(e)) from e
            if is_invalid_image_error(e):
                raise InvalidImageError(str(e)) from e
            if is_request_too_large_error(e):
                raise RequestTooLargeError(str(e)) from e
            raise
        finally:
            if fallback_handler is not None:
                await fallback_handler.close()

        if fallback_handler is not None:
            hidden = getattr(response, "_hidden_params", None)
            if not isinstance(hidden, dict):
                hidden = {}
                response._hidden_params = hidden
            hidden["anthropic_fallback"] = fallback_handler.last_routing

        choice = response.choices[0]
        truncated = choice.finish_reason == "length" or self._hit_output_limit(response)
        if truncated:
            content = choice.message.content or ""
            raise OutputLengthExceededError(
                f"Model {self.model_name} hit the max output tokens limit.",
                truncated_response=content,
            )
        return response

    @staticmethod
    def _hit_output_limit(response: ModelResponse) -> bool:
        """Whether a Responses-API turn was cut off at max_output_tokens: the
        bridge never reports finish_reason ``length``; the signal lives in
        ``incomplete_details`` instead."""
        hidden = getattr(response, "_hidden_params", None)
        if not isinstance(hidden, dict):
            return False
        for candidate in (hidden, hidden.get("additional_headers")):
            if not isinstance(candidate, dict):
                continue
            details = candidate.get("incomplete_details")
            reason = (
                details.get("reason")
                if isinstance(details, dict)
                else getattr(details, "reason", None)
            )
            if reason == "max_output_tokens":
                return True
        return False

    def _parse_response(self, response: ModelResponse) -> ParsedResponse:
        # Merge every choice: the Responses bridge splits a turn's text and
        # tool calls across choices. Harmless for single-choice providers.
        choices = list(response.choices)
        if not choices:
            return ParsedResponse(cost_usd=self._extract_cost(response))
        content_parts: list[str] = []
        tool_calls: list[NormalizedToolCall] = []
        # Accumulated across every choice; replaying only the first would
        # drop part of the chain of thought.
        reasoning_items: list[Any] = []
        reasoning_parts: list[str] = []
        thinking_blocks: Any | None = None
        provider_specific_fields: dict[str, Any] = {}
        for choice in choices:
            choice_message = choice.message
            if choice_message.content:
                content_parts.append(str(choice_message.content))
            for index, raw_call in enumerate(choice_message.tool_calls or []):
                # litellm 1.97+ types tool_calls as a union with OpenAI
                # "custom tool" calls, which carry no .function. We never
                # register custom tools, so one arriving is a provider bug —
                # skip it rather than die mid-turn.
                function = getattr(raw_call, "function", None)
                if function is None:
                    self._logger.warning(
                        f"Dropping non-function tool call: {raw_call!r}"
                    )
                    continue
                tool_calls.append(
                    NormalizedToolCall(
                        call_id=raw_call.id or f"call_{uuid.uuid4().hex[:8]}_{index}",
                        name=function.name or "",
                        raw_arguments=function.arguments or "{}",
                    )
                )
            # getattr is load-bearing: litellm deletes these attributes when
            # the provider omits them, so direct access raises.
            items = getattr(choice_message, "reasoning_items", None)
            if items:
                reasoning_items.extend(items)
            summary = getattr(choice_message, "reasoning_content", None)
            if summary:
                reasoning_parts.append(str(summary))
            blocks = getattr(choice_message, "thinking_blocks", None)
            if blocks and thinking_blocks is None:
                thinking_blocks = blocks
            fields = getattr(choice_message, "provider_specific_fields", None)
            if isinstance(fields, dict):
                for key, value in fields.items():
                    current = provider_specific_fields.get(key)
                    if isinstance(current, list) and isinstance(value, list):
                        current.extend(copy.deepcopy(value))
                    elif key not in provider_specific_fields:
                        provider_specific_fields[key] = copy.deepcopy(value)

        # litellm sets ``usage`` only when the provider reports it.
        usage = getattr(response, "usage", None)
        prompt_tokens = completion_tokens = cached_tokens = reasoning_tokens = 0
        if usage is not None:
            prompt_tokens = int(usage.prompt_tokens or 0)
            completion_tokens = int(usage.completion_tokens or 0)
            details = usage.prompt_tokens_details
            cached_tokens = int(details.cached_tokens or 0) if details else 0
            completion_details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = int(
                getattr(completion_details, "reasoning_tokens", None) or 0
            )

        parsed = ParsedResponse(
            content="\n".join(content_parts),
            reasoning_content="\n".join(reasoning_parts) or None,
            thinking_blocks=thinking_blocks,
            reasoning_items=reasoning_items or None,
            provider_specific_fields=provider_specific_fields or None,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens,
            cost_usd=self._extract_cost(response),
            response_model=str(getattr(response, "model", "") or "") or None,
        )
        hidden = getattr(response, "_hidden_params", None)
        routing = hidden.get("anthropic_fallback") if isinstance(hidden, dict) else None
        if isinstance(routing, dict):
            parsed.response_model = (
                routing.get("response_model") or parsed.response_model
            )
            parsed.fallback_attempted = bool(routing.get("fallback_attempted"))
            parsed.fallback_used = bool(routing.get("fallback_used"))
            parsed.reasoning_tokens = max(
                parsed.reasoning_tokens,
                int(routing.get("thinking_tokens") or 0),
            )
            iterations = routing.get("fallback_iterations")
            if isinstance(iterations, list):
                parsed.fallback_iterations = copy.deepcopy(iterations)
            parsed.stop_reason = routing.get("stop_reason")
            details = routing.get("stop_details")
            if isinstance(details, dict):
                parsed.stop_details = copy.deepcopy(details)
            thinking = routing.get("request_thinking")
            if isinstance(thinking, dict):
                parsed.request_thinking = copy.deepcopy(thinking)
            output_config = routing.get("request_output_config")
            if isinstance(output_config, dict):
                parsed.request_output_config = copy.deepcopy(output_config)
            fallbacks = routing.get("request_fallbacks")
            if fallbacks is not None:
                parsed.request_fallbacks = str(fallbacks)
        if parsed.is_empty:
            # A 200 with nothing in it — neither refusal nor truncation
            # (both raise). Log the wire facts for diagnosis.
            self._logger.warning(
                "Empty model response: finish_reasons="
                f"{[c.finish_reason for c in choices]} "
                f"completion_tokens={completion_tokens}"
            )
        return parsed

    @staticmethod
    def _extract_cost(response: ModelResponse) -> float:
        # litellm prices every acompletion response into
        # _hidden_params["response_cost"] (None for unpriced models). Stream
        # aggregation rebuilds usage but does not retain that hidden field, so
        # price the rebuilt response from its registered model metadata.
        hidden_params = getattr(response, "_hidden_params", None)
        if isinstance(hidden_params, dict):
            hidden_cost = hidden_params.get("response_cost")
            if hidden_cost is not None:
                return float(hidden_cost)
        try:
            return float(litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:  # noqa: BLE001 — telemetry must not fail a model turn
            return 0.0

    def assistant_messages(self, parsed: ParsedResponse) -> list[Message]:
        """The turn as one or two messages, ready to append to history.
        Two when a bridged OpenAI turn has both prose and tool calls: the
        Responses transformation drops ``content`` on the tool_calls path,
        so the prose would vanish from history on replay."""
        message = self.assistant_message(parsed)
        if (
            parsed.content
            and parsed.tool_calls
            and parsed.reasoning_items
            and not self.is_anthropic()
        ):
            prose: Message = {"role": "assistant", "content": parsed.content}
            message["content"] = None
            return [prose, message]
        return [message]

    def assistant_message(self, parsed: ParsedResponse) -> Message:
        message: Message = {
            "role": "assistant",
            "content": parsed.content or None,
        }
        if parsed.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.raw_arguments,
                    },
                }
                for call in parsed.tool_calls
            ]
        # On a bare turn thinking becomes the message's final block, which
        # Anthropic 400s — permanently, for the whole conversation.
        if parsed.thinking_blocks and (parsed.tool_calls or parsed.content):
            message["thinking_blocks"] = parsed.thinking_blocks
        if parsed.reasoning_items:
            # Responses API uses opaque items instead of its readable summary.
            message["reasoning_items"] = parsed.reasoning_items
        elif parsed.reasoning_content:
            message["reasoning_content"] = parsed.reasoning_content
        if parsed.provider_specific_fields:
            # Gemini thought signatures preserve the model's reasoning state.
            # LiteLLM reattaches them to the correct wire parts on replay.
            message["provider_specific_fields"] = parsed.provider_specific_fields
        # An assistant message with neither content nor tool_calls is invalid
        # at several providers
        if not message.get("tool_calls") and not message.get("content"):
            message["content"] = NO_OUTPUT_PLACEHOLDER
        return message

    @staticmethod
    def _xai_system_as_input_items(messages: list[Message]) -> list[Message]:
        """Keep xAI system prompts as Responses input items.

        LiteLLM lifts a string ``role=system`` message into ``instructions``
        and the xAI Responses adapter drops that parameter. Non-string
        content takes the input-item branch instead.
        """
        rewritten: list[Message] = []
        for message in messages:
            content = message.get("content")
            if message.get("role") == "system" and isinstance(content, str):
                rewritten.append(
                    {**message, "content": [{"type": "text", "text": content}]}
                )
            else:
                rewritten.append(message)
        return rewritten

    def _apply_prompt_caching(self, messages: list[Message]) -> list[Message]:
        """Place Anthropic cache breakpoints: the system prompt plus the
        latest message that can carry one — everything after the second
        breakpoint is re-read at full input price every turn.

        Tool messages are tagged at the TOP LEVEL, not inside ``content``:
        that is the key litellm copies onto the emitted ``tool_result``
        block. Without it, a run driven purely through bash (assistant
        ``content: None`` -> tool result) has nothing taggable after the
        opening instruction and the breakpoint never advances."""
        if not self.is_anthropic():
            return messages

        # Wire-only copy: the tagging rewrites content into provider shapes.
        cached = copy.deepcopy(messages)

        def tag(message: Message) -> bool:
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": dict(CACHE_CONTROL),
                    }
                ]
                return True
            if isinstance(content, list):
                for item in reversed(content):
                    # Never attach a cache breakpoint directly to an image.
                    # Z.ai's Anthropic-compatible vision route accepts that
                    # payload but silently presents a blank image to the model.
                    # Prefer the nearest text block; the image remains inside
                    # the cached message prefix without carrying cache_control.
                    if isinstance(item, dict) and item.get("type") == "text":
                        cast(dict[str, Any], item)["cache_control"] = dict(
                            CACHE_CONTROL
                        )
                        return True
            return False

        if cached:
            tag(cached[0])
        for message in reversed(cached[1:]):
            role = message.get("role")
            if role in ("user", "assistant") and tag(message):
                break
            if role == "tool":
                message["cache_control"] = dict(CACHE_CONTROL)
                break
        return cached
