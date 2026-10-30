"""Gemini through the Interactions API, using Google's ``google-genai`` SDK.

Selected by ``model_api: gemini_interactions``. As Google recommends,
interactions are stored (``store=True``) and each turn continues the previous
one with ``previous_interaction_id``, sending only the new steps. Each assistant
turn records its interaction id and a digest of the history it extends; when
compaction or image eviction rewrites that history, the next request sends the
whole history again and starts a new chain. With ``store=False`` every request
sends the whole history. The model's own steps (thoughts with their signatures,
function calls, output) are kept on the assistant message and replayed
verbatim, as the API requires for stateless requests.

Replies come back as a LiteLLM ``ModelResponse`` and errors as LiteLLM
exceptions, so parsing, cost, retries and context-overflow handling are shared
with the chat path. Each request attempt goes through LiteLLM's logging, so
registered logging callbacks receive the same success and failure events, with
a standard logging payload, as for a chat completion.
"""

import asyncio
import datetime
import hashlib
import json
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

import litellm
from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
from litellm.litellm_core_utils.llm_response_utils.response_metadata import (
    update_response_metadata,
)
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.litellm_core_utils.rules import Rules
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    CompletionTokensDetailsWrapper,
    Function,
    ModelResponse,
    PromptTokensDetailsWrapper,
    Usage,
)
from litellm.types.utils import (
    Message as LiteLLMMessage,
)
from litellm.utils import function_setup

from harbor_proximus.prompts import NO_OUTPUT_PLACEHOLDER

PROVIDER = "gemini"
# Keys on the assistant message: the turn's raw Interaction steps, and the
# stored interaction they belong to ({"id": ..., "history": digest}).
STEPS_FIELD = "interaction_steps"
LINK_FIELD = "interaction_link"
# Steps that come from our side of the conversation, never replayed as model output.
_INPUT_STEP_TYPES = frozenset({"user_input", "function_result"})
# Gemini's wording for an over-long prompt (not covered by llm.py's phrases).
_CONTEXT_PHRASES = (
    "exceeds the maximum number of tokens",
    "input token count",
    "context length",
    "context window",
)
# Keys of llm_call_kwargs that belong in generation_config, by their Interactions name.
_GENERATION_KEYS = {
    "max_tokens": "max_output_tokens",
    "max_output_tokens": "max_output_tokens",
    "temperature": "temperature",
    "top_p": "top_p",
    "seed": "seed",
    "stop": "stop_sequences",
    "stop_sequences": "stop_sequences",
}
_USAGE_KEYS = (
    "total_input_tokens",
    "total_output_tokens",
    "total_thought_tokens",
    "total_cached_tokens",
)

type Message = dict[str, Any]


def _text_content(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _user_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [_text_content(content)] if content else []
    parts: list[dict[str, Any]] = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and part.get("text"):
            parts.append(_text_content(str(part["text"])))
        elif part.get("type") == "image_url":
            image = part.get("image_url")
            url = str(image.get("url") if isinstance(image, dict) else image or "")
            if url.startswith("data:") and ";base64," in url:
                mime, data = url[5:].split(";base64,", 1)
                parts.append({"type": "image", "mime_type": mime, "data": data})
            elif url:
                parts.append({"type": "image", "uri": url})
    return parts


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        str(part.get("text") or "")
        for part in content or []
        if isinstance(part, dict) and part.get("type") == "text"
    )


def _assistant_steps(message: Message) -> list[dict[str, Any]]:
    fields = message.get("provider_specific_fields")
    stored = fields.get(STEPS_FIELD) if isinstance(fields, dict) else None
    if isinstance(stored, list) and stored:
        return [dict(step) for step in stored]
    # No recorded steps (a turn not produced by this API): rebuild what we can.
    steps: list[dict[str, Any]] = []
    text = _text_of(message.get("content"))
    if text and text != NO_OUTPUT_PLACEHOLDER:
        steps.append({"type": "model_output", "content": [_text_content(text)]})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {"raw": function.get("arguments")}
        steps.append(
            {
                "type": "function_call",
                "id": call.get("id") or "",
                "name": function.get("name") or "",
                "arguments": arguments
                if isinstance(arguments, dict)
                else {"value": arguments},
            }
        )
    return steps


def to_input(messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
    """Chat messages as (system_instruction, Interaction input steps)."""
    system, steps, _ = _convert(messages)
    return system, steps


def _convert(
    messages: list[Message],
) -> tuple[str | None, list[dict[str, Any]], tuple[dict[str, Any], int] | None]:
    """to_input, plus the last stored interaction link and the number of steps
    up to and including that assistant turn."""
    system: list[str] = []
    steps: list[dict[str, Any]] = []
    call_names: dict[str, str] = {}
    link: tuple[dict[str, Any], int] | None = None
    for message in messages:
        role = message.get("role")
        if role == "system":
            text = _text_of(message.get("content"))
            if text:
                system.append(text)
        elif role == "user":
            content = _user_content(message.get("content"))
            if content:
                steps.append({"type": "user_input", "content": content})
        elif role == "assistant":
            for step in _assistant_steps(message):
                if step.get("type") == "function_call":
                    call_names[str(step.get("id"))] = str(step.get("name") or "")
                steps.append(step)
            fields = message.get("provider_specific_fields")
            if isinstance(fields, dict) and isinstance(fields.get(LINK_FIELD), dict):
                link = (fields[LINK_FIELD], len(steps))
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            result: dict[str, Any] = {
                "type": "function_result",
                "call_id": call_id,
                "result": [_text_content(_text_of(message.get("content")))],
            }
            if call_names.get(call_id):
                result["name"] = call_names[call_id]
            steps.append(result)
    return ("\n\n".join(system) or None), steps, link


def history_digest(system: str | None, steps: list[dict[str, Any]]) -> str:
    """Identifies a history, so a stored interaction is only continued while
    the local history still matches what the server holds."""
    encoded = json.dumps([system, steps], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class Request:
    body: dict[str, Any]
    """The create() body."""
    options: dict[str, Any]
    """SDK call options (timeout, extra_body, ...)."""
    system: str | None
    history: list[dict[str, Any]]
    """The whole history this request extends, sent or not."""
    messages: list[Message] = field(default_factory=list)
    """The chat messages it was built from, as LiteLLM logs them."""


def to_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """OpenAI-style function tools as Interaction function declarations."""
    if not tools:
        return None
    declared: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") if tool.get("type") == "function" else None
        if not isinstance(function, dict):
            continue
        entry: dict[str, Any] = {"type": "function", "name": function["name"]}
        if function.get("description"):
            entry["description"] = function["description"]
        if function.get("parameters"):
            entry["parameters"] = function["parameters"]
        declared.append(entry)
    return declared or None


def build_request(
    model_name: str,
    messages: list[Message],
    tools: list[dict[str, Any]] | None,
    *,
    reasoning_effort: str | None,
    temperature: float | None,
    call_kwargs: dict[str, Any],
) -> Request:
    system, steps, link = _convert(messages)
    generation: dict[str, Any] = {}
    if reasoning_effort is not None:
        generation["thinking_level"] = reasoning_effort
        generation["thinking_summaries"] = "auto"
    if temperature is not None:
        generation["temperature"] = temperature
    body: dict[str, Any] = {
        "model": model_name.removeprefix("gemini/"),
        "input": steps,
        "store": True,
    }
    if system:
        body["system_instruction"] = system
    declared = to_tools(tools)
    if declared:
        body["tools"] = declared
    options: dict[str, Any] = {}
    for key, value in call_kwargs.items():
        if key in _GENERATION_KEYS:
            if key in ("stop", "stop_sequences") and isinstance(value, str):
                value = [value]
            generation[_GENERATION_KEYS[key]] = value
        elif key == "generation_config" and isinstance(value, dict):
            generation.update(value)
        elif key in ("timeout", "extra_body", "extra_headers", "api_version"):
            options[key] = value
        else:
            body[key] = value
    if generation:
        body["generation_config"] = generation
    if body["store"] and link is not None:
        stored, end = link
        if stored.get("history") == history_digest(system, steps[:end]):
            body["previous_interaction_id"] = stored["id"]
            body["input"] = steps[end:]
    return Request(body, options, system, steps, messages)


def _dump(step: Any) -> dict[str, Any]:
    if getattr(step, "is_unknown", False):
        raw = getattr(step, "raw", None)
        return dict(raw) if isinstance(raw, dict) else {"type": "UNKNOWN", "raw": raw}
    return step.model_dump(mode="json", by_alias=True, exclude_none=True)


def _usage(totals: dict[str, int]) -> Usage:
    prompt = totals.get("total_input_tokens", 0)
    thought = totals.get("total_thought_tokens", 0)
    # Output excludes thoughts; count them as completion tokens, as LiteLLM
    # does for generateContent, so cost matches the chat path.
    completion = totals.get("total_output_tokens", 0) + thought
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=totals.get("total_cached_tokens", 0)
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(
            reasoning_tokens=thought
        ),
    )


def to_model_response(
    model_name: str,
    steps: list[dict[str, Any]],
    status: str,
    usage: dict[str, int],
    link: dict[str, Any] | None = None,
) -> ModelResponse:
    """Model steps as one chat choice, with the steps (and the stored
    interaction's link) kept for the next request."""
    text: list[str] = []
    thoughts: list[str] = []
    calls: list[ChatCompletionMessageToolCall] = []
    for step in steps:
        kind = step.get("type")
        if kind == "model_output":
            text.extend(
                str(part.get("text") or "")
                for part in step.get("content") or []
                if isinstance(part, dict) and part.get("type") == "text"
            )
        elif kind == "thought":
            thoughts.extend(
                str(part.get("text") or "")
                for part in step.get("summary") or []
                if isinstance(part, dict) and part.get("type") == "text"
            )
        elif kind == "function_call":
            calls.append(
                ChatCompletionMessageToolCall(
                    id=step.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=Function(
                        name=step.get("name") or "",
                        arguments=json.dumps(step.get("arguments") or {}),
                    ),
                )
            )
    message = LiteLLMMessage(
        role="assistant",
        content="".join(text) or None,
        tool_calls=calls or None,
        reasoning_content="\n".join(t for t in thoughts if t) or None,
        provider_specific_fields={STEPS_FIELD: steps, LINK_FIELD: link}
        if link
        else ({STEPS_FIELD: steps} if steps else None),
    )
    if status in ("incomplete", "budget_exceeded"):
        finish = "length"
    else:
        finish = "tool_calls" if calls else "stop"
    choice = Choices(
        index=0,
        message=message,
        finish_reason=finish,
        provider_specific_fields={"interaction_status": status},
    )
    return ModelResponse(
        id=f"interaction-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=model_name,
        choices=[choice],
        usage=_usage(usage),
    )


def _litellm_error(error: Exception, model: str) -> Exception:
    """SDK error as the LiteLLM exception the chat path would have raised."""
    status = getattr(error, "status_code", None)
    message = f"GeminiInteractionsException - {error}"
    kw: dict[str, Any] = {"message": message, "model": model, "llm_provider": PROVIDER}
    # The SDK's error classes are private, so match them by name.
    names = {cls.__name__ for cls in type(error).__mro__}
    exc = litellm.exceptions
    if status == 400 and any(phrase in message.lower() for phrase in _CONTEXT_PHRASES):
        return exc.ContextWindowExceededError(**kw)
    if status == 401:
        return exc.AuthenticationError(**kw)
    if status == 404:
        return exc.NotFoundError(**kw)
    if status == 408 or "APITimeoutError" in names or isinstance(error, TimeoutError):
        return exc.Timeout(**kw)
    if status == 413:
        return exc.BadRequestError(
            **{**kw, "message": f"request entity too large: {message}"}
        )
    if status == 429:
        return exc.RateLimitError(**kw)
    if status == 503:
        return exc.ServiceUnavailableError(**kw)
    if isinstance(status, int) and status >= 500:
        return exc.InternalServerError(**kw)
    if "APIConnectionError" in names:
        return exc.APIConnectionError(**kw)
    # Other 4xx (403 too, as LiteLLM maps Gemini's) and local SDK errors such
    # as an unknown request field fail the same way on every retry.
    return exc.BadRequestError(**kw)


def _interaction_errors(
    interaction: Any, steps: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The errors recorded on the interaction and on its model_output steps,
    as the API sent them."""
    errors = [_dump(error) for error in getattr(interaction, "errors", None) or []]
    errors += [
        step["error"]
        for step in steps
        if step.get("type") == "model_output" and step.get("error")
    ]
    return errors


class _CallLog:
    """One request attempt, logged as LiteLLM logs a chat completion so its
    logging callbacks receive the same events and standard payload."""

    def __init__(self, model_name: str, request: Request, api_base: str):
        self.start = datetime.datetime.now()
        call_id = str(uuid.uuid4())
        # acompletion's own setup: registers litellm.callbacks and creates
        # the call's logging object.
        self._logging, _ = function_setup(
            "acompletion",
            Rules(),
            self.start,
            model=model_name,
            messages=request.messages,
            litellm_call_id=call_id,
        )
        self._model_name = model_name
        self._api_base = api_base
        params = {
            key: value
            for key, value in request.body.items()
            if key not in ("model", "input", "system_instruction")
        }
        self._logging.update_environment_variables(
            model=model_name.removeprefix("gemini/"),
            user=None,
            optional_params=params,
            litellm_params=get_litellm_params(
                custom_llm_provider=PROVIDER,
                api_base=api_base,
                litellm_call_id=call_id,
            ),
            custom_llm_provider=PROVIDER,
        )
        self._logging.pre_call(
            input=request.messages,
            api_key="",
            additional_args={"complete_input_dict": request.body, "api_base": api_base},
        )

    def success(self, response: ModelResponse) -> None:
        end = datetime.datetime.now()
        # Sets the response's cost and api_base hidden params, as acompletion does.
        update_response_metadata(
            result=response,
            logging_obj=self._logging,
            model=self._model_name,
            kwargs={"api_base": self._api_base},
            start_time=self.start,
            end_time=end,
        )
        GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(
            async_coroutine=self._logging.async_success_handler(
                result=response, start_time=self.start, end_time=end
            )
        )
        self._logging.handle_sync_success_callbacks_for_async_calls(
            result=response, start_time=self.start, end_time=end
        )

    async def failure(self, error: Exception) -> None:
        """Call from the except block, for the traceback."""
        end = datetime.datetime.now()
        trace = traceback.format_exc()
        self._logging.failure_handler(error, trace, self.start, end)
        await self._logging.async_failure_handler(error, trace, self.start, end)


class InteractionsClient:
    """One async google-genai client per LLM."""

    def __init__(self, model_name: str, api_key: str | None, api_base: str | None):
        try:
            from google import genai
            from google.genai._gaos.utils import RetryConfig
        except ImportError as e:  # pragma: no cover - depends on the extra
            raise ImportError(
                "model_api='gemini_interactions' needs google-genai: "
                "install harbor-proximus[gemini]"
            ) from e
        self.model_name = model_name
        http_options = genai.types.HttpOptions(base_url=api_base)
        # Raises here, at agent construction, when no API key is found.
        self._client = genai.Client(api_key=api_key, http_options=http_options)
        sdk = self._client.aio.interactions.sdk_configuration
        # The agent loop owns retries and logs every attempt. The SDK's
        # Interactions layer retries at least once whatever retry_options
        # say, which would hide a failed request, so turn it off. This is
        # private SDK API: fail loudly rather than retry unlogged if it moves.
        if not hasattr(sdk, "retry_config"):
            raise RuntimeError(
                "this google-genai version has no Interactions retry_config; "
                "model_api='gemini_interactions' cannot turn off its retries"
            )
        sdk.retry_config = RetryConfig("none", None, False)
        self._server_url = sdk.get_server_details()[0]
        self._api_version = sdk.globals.api_version

    def endpoint(self, api_version: str | None = None) -> str:
        """The URL create() posts to."""
        return f"{self._server_url}/{api_version or self._api_version}/interactions"

    async def create(self, request: Request) -> ModelResponse:
        log = _CallLog(
            self.model_name, request, self.endpoint(request.options.get("api_version"))
        )
        try:
            response = await self._create(request)
        except Exception as error:  # noqa: BLE001 - logged, then re-raised
            await log.failure(error)
            raise
        log.success(response)
        return response

    async def _create(self, request: Request) -> ModelResponse:
        try:
            # httpx applies the SDK's timeout to each connect, read and write;
            # this bounds the whole call.
            async with asyncio.timeout(request.options.get("timeout")):
                interaction = await self._client.aio.interactions.create(
                    **request.body, **request.options
                )
        except Exception as error:  # noqa: BLE001 - mapped, then raised
            raise _litellm_error(error, self.model_name) from error
        status = str(interaction.status or "")
        steps = [_dump(step) for step in interaction.steps or []]
        steps = [step for step in steps if step.get("type") not in _INPUT_STEP_TYPES]
        errors = _interaction_errors(interaction, steps)
        if status == "failed" or errors:
            # The schema has no refusal or safety field, and error codes are
            # free-form, so a block cannot be told from a fault here: the raw
            # status and errors go into the message instead.
            raise litellm.exceptions.InternalServerError(
                message=(
                    f"GeminiInteractionsException - interaction status {status!r}, "
                    f"errors: {json.dumps(errors)}"
                ),
                model=self.model_name,
                llm_provider=PROVIDER,
            )
        if status == "cancelled":
            raise litellm.exceptions.APIConnectionError(
                message="GeminiInteractionsException - interaction cancelled",
                model=self.model_name,
                llm_provider=PROVIDER,
            )
        totals = {
            key: int(getattr(interaction.usage, key, None) or 0) for key in _USAGE_KEYS
        }
        link = None
        if request.body.get("store") and interaction.id:
            history = history_digest(request.system, request.history + steps)
            link = {"id": interaction.id, "history": history}
        return to_model_response(self.model_name, steps, status, totals, link)
