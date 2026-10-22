"""Model routing, credentials and the retrying transport."""

import json
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import pydantic
import tenacity
from harbor.agents.model_connection import (
    PROVIDERS,
    ModelConnectionSpec,
    resolve_model_connection,
)
from harbor.llms.base import ContextLengthExceededError, OutputLengthExceededError

from harbor_proximus.config import BUDGET_MARGIN_SEC, ProximusConfig
from harbor_proximus.model_info import ModelInfo, ModelRegistry
from harbor_proximus.providers.anthropic import AnthropicProvider
from harbor_proximus.providers.base import (
    PLACEHOLDER,
    Message,
    ModelCall,
    NormalizedToolCall,
    ParsedResponse,
    Provider,
)
from harbor_proximus.providers.openai import (
    OpenAIChatProvider,
    OpenAIResponsesProvider,
)

__all__ = [
    "LLM",
    "Message",
    "NormalizedToolCall",
    "ParsedResponse",
]


@dataclass(frozen=True)
class Credentials:
    api_key: str | None
    api_base: str | None
    """A base URL from the environment (never the provider default)."""
    consumed_env_keys: frozenset[str]
    """Agent-env variables used for the model, to hide from the sandbox."""


def model_route(model_name: str) -> str:
    """The provider part of ``provider/model``."""
    route, _, model = model_name.partition("/")
    if not route or not model:
        raise ValueError(
            f"Model {model_name!r} needs a provider prefix: anthropic/<model>, "
            "openai/<model>, or <provider>/<model> for an OpenAI-compatible API."
        )
    return route.lower()


def resolve_credentials(
    model_name: str, api_key_env: str | None, extra_env: dict[str, str] | None
) -> Credentials:
    """The model's API key and base URL, as Harbor's own agents resolve them.

    Harbor's agent env (`--ae`, AgentConfig.env) wins over the host, with the
    provider's usual variable names. ``api_key_env`` names the key variable
    explicitly (agent env, then host) and must be set.
    """
    model_route(model_name)
    agent_env = extra_env or {}
    used: list[str] = []

    def lookup(*names: str) -> tuple[str, str] | None:
        for name in names:
            if name in agent_env:
                used.append(name)
                return name, agent_env[name]
        for name in names:
            if os.environ.get(name):
                return name, os.environ[name]
        return None

    connection = resolve_model_connection(model_name, ModelConnectionSpec(), lookup)
    api_key = connection.api_key or None
    if api_key_env:
        api_key = agent_env.get(api_key_env, os.environ.get(api_key_env)) or None
        if api_key is None:
            raise ValueError(
                f"api_key_env={api_key_env!r} is not set in the "
                "environment — the run cannot authenticate this model."
            )
        if api_key_env in agent_env:
            used.append(api_key_env)
    return Credentials(api_key, connection.configured_base_url, frozenset(used))


# Provider error strings that indicate context-window overflow
# Adapted from Harbor's LiteLLM context-length check
_CONTEXT_LENGTH_ERROR_PHRASES = (
    "context length exceeded",
    "context_length_exceeded",
    "maximum context length",
    "`inputs` tokens + `max_new_tokens`",
    "model's context length",  # vllm 0.16.0 error message
    "prompt is too long",  # Anthropic, also via OpenAI-compatible proxies
    "input is too long for requested model",  # Bedrock via proxy
)


# Requests the provider rejected as invalid, unauthenticated, forbidden or
# unknown: replaying them verbatim cannot succeed.
_REJECTED_STATUSES = frozenset({400, 401, 403, 404, 422})


def _status(error: BaseException) -> int | None:
    """HTTP status of an OpenAI or Anthropic SDK error."""
    return getattr(error, "status_code", None)


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
# OpenAI's nondeterministic `invalid_prompt` flag and other safety-system
# rejections: a verbatim replay usually passes.
_CONTENT_POLICY_PHRASES = (
    "content_policy_violation",
    "invalid_prompt",
    "violating our usage policy",
    "rejected as a result of the safety system",
)


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
    if _status(error) == 413:
        return True
    combined = _error_text(error)
    return any(phrase in combined for phrase in _REQUEST_TOO_LARGE_PHRASES)


def is_content_policy_error(error: BaseException) -> bool:
    combined = _error_text(error)
    return any(phrase in combined for phrase in _CONTENT_POLICY_PHRASES)


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


def _is_retryable(error: BaseException) -> bool:
    """Retry transport failures, rate limits and 5xx; never a request the
    provider rejected (400/401/403/404/422) or a broken configuration."""
    if not isinstance(error, Exception):
        # BaseException-only classes (CancelledError, GeneratorExit,
        # KeyboardInterrupt): cancellation/teardown control flow, never a
        # provider failure. tenacity's retry predicate sees BaseExceptions,
        # and retrying a CancelledError swallows the STOP: a stopped run would
        # sit in a tenacity sleep ("Retrying ... as it raised
        # CancelledError") until the SIGKILL, skipping auto-submission and
        # the end-of-run restore.
        return False
    if is_transient_image_account_error(error):
        # Moonshot's image-preparation service can temporarily suspend an
        # otherwise-working account and then recover; it answers 400.
        return True
    if is_content_policy_error(error):
        return True
    text = str(error).lower()
    if "resource_exhausted" in text or "quota exceeded" in text:
        # Google quota errors are rate limits, whatever status they carry.
        return True
    if is_request_too_large_error(error):
        # Deterministic for a given history (verbatim replays carry the same
        # oversized body) — the agent loop recovers by evicting images.
        return False
    if isinstance(
        error,
        (
            ContextLengthExceededError,
            OutputLengthExceededError,
            RequestTooLargeError,
            InvalidImageError,
            ProviderRetryBudgetError,
            # Local and deterministic: a bad option value or an adapter bug
            # fails identically on every attempt (and each one is paid).
            pydantic.ValidationError,
            KeyError,
            AttributeError,
            TypeError,
        ),
    ):
        return False
    return _status(error) not in _REJECTED_STATUSES


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
        self.model_name = model_name
        self._config = config
        self._remaining_sec = remaining_sec
        self._logger = logger
        self._provider_health = _ProviderHealth(
            provider_health_path, model_name, logger
        )
        overrides = dict(config.extra_model_info)
        if config.model_info:
            overrides[model_name] = config.model_info
        self.registry = ModelRegistry(overrides)
        # The provider's own model id: the name without its route.
        self.wire_model = model_name.split("/", 1)[-1]
        credentials = resolve_credentials(model_name, config.api_key_env, extra_env)
        # Proximus.run blanks these in the sandbox: Harbor overlays the agent
        # env onto sandbox commands.
        self.consumed_env_keys: frozenset[str] = credentials.consumed_env_keys
        # Set by the agent once the run's session id exists; scopes the
        # provider-side prompt cache to this run.
        self.session_id: str = ""
        self.provider = self._make_provider(credentials)
        if "anthropic_fallbacks" in config.llm_call_kwargs and not isinstance(
            self.provider, AnthropicProvider
        ):
            raise ValueError("anthropic_fallbacks requires an Anthropic model")

    def _make_provider(self, credentials: Credentials) -> Provider:
        """anthropic/ uses the Messages API, openai/ the Responses API, and
        any other route Chat Completions at that provider's endpoint."""
        route = model_route(self.model_name)
        common: dict[str, Any] = {
            "model_name": self.model_name,
            "api_key": credentials.api_key,
            "base_url": self._config.api_base or credentials.api_base,
            "registry": self.registry,
        }
        if route == "anthropic":
            return AnthropicProvider(**common)
        if route == "openai":
            return OpenAIResponsesProvider(**common)
        # Never let the OpenAI SDK fall back to api.openai.com or read
        # OPENAI_API_KEY for another provider's model.
        access = PROVIDERS.get(route)
        common["base_url"] = common["base_url"] or (access and access.base_url)
        if not common["base_url"] or not common["api_key"]:
            raise ValueError(
                f"{self.model_name!r} needs api_base and an API key "
                "(api_key_env or the provider's key variable)."
            )
        return OpenAIChatProvider(**common)

    def model_info(self) -> ModelInfo | None:
        return self.registry.lookup(self.model_name)

    def resolve_context_limit(self) -> int:
        info = self.model_info() or {}
        limit = info.get("max_input_tokens")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError(f"No valid max_input_tokens for {self.model_name!r}.")
        return limit

    def supports_vision(self) -> bool:
        return (self.model_info() or {}).get("supports_vision") is True

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
            parsed = await retryer(self._acompletion, messages, tools)
        except BaseException as error:
            if isinstance(error, Exception):
                self._provider_health.failed(error)
            raise
        self._provider_health.success()
        if parsed.is_empty:
            # A 200 with nothing in it — neither refusal nor truncation
            # (both raise). Log the wire facts for diagnosis.
            self._logger.warning(
                f"Empty model response: stop_reason={parsed.stop_reason} "
                f"completion_tokens={parsed.completion_tokens}"
            )
        return parsed

    async def _acompletion(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
    ) -> ParsedResponse:
        """One attempt, with provider errors normalized for the agent loop."""
        timeout = LLM_TIMEOUT_SEC
        remaining = self._remaining_sec()
        if remaining is not None:
            timeout = min(timeout, max(float(BUDGET_MARGIN_SEC), remaining * 0.5))
        call = ModelCall(
            model=self.wire_model,
            messages=messages,
            tools=tools,
            timeout=timeout,
            reasoning_effort=self._config.reasoning_effort,
            temperature=self._config.temperature,
            max_thinking_tokens=self._config.max_thinking_tokens,
            session_id=self.session_id,
            kwargs=dict(self._config.llm_call_kwargs),
        )
        try:
            return await self._send(call)
        except (ContextLengthExceededError, OutputLengthExceededError):
            raise
        except Exception as e:  # noqa: BLE001 — typed, then re-raised
            # Size/count rejections arrive under many guises (status codes,
            # gateway wrappers) — normalize them before the agent loop
            # chooses its recovery path.
            # Also mid-stream errors, which carry no 4xx status.
            if is_context_length_error(e):
                raise ContextLengthExceededError from e
            if is_image_limit_error(e):
                raise ImageLimitExceededError(str(e)) from e
            if is_invalid_image_error(e):
                raise InvalidImageError(str(e)) from e
            if is_request_too_large_error(e):
                raise RequestTooLargeError(str(e)) from e
            raise

    async def _send(self, call: ModelCall) -> ParsedResponse:
        return await self.provider.complete(call)

    def assistant_messages(self, parsed: ParsedResponse) -> list[Message]:
        """The turn as messages, ready to append to history."""
        return [self.assistant_message(parsed)]

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
        if parsed.native:
            # Replayed verbatim by the adapter that produced it (thinking
            # signatures, encrypted reasoning).
            message["native"] = parsed.native
        # An assistant message with neither content nor tool_calls is invalid
        # at several providers
        if not message.get("tool_calls") and not message.get("content"):
            message["content"] = PLACEHOLDER
        return message
