"""Request and response shapes shared by the API adapters.

History is kept in OpenAI chat format. Each adapter translates it to its wire
format and stores its own assistant turn under ``native`` so later requests
replay it verbatim (thinking signatures, encrypted reasoning).
"""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from harbor_proximus.model_info import ModelInfo, ModelRegistry

type Message = dict[str, Any]

PLACEHOLDER = "(no output this turn)"


@dataclass
class NormalizedToolCall:
    call_id: str
    name: str
    raw_arguments: str


@dataclass
class ParsedResponse:
    content: str = ""
    # Readable reasoning (summaries or thinking text), for trajectories.
    reasoning_content: str | None = None
    # The provider's own assistant turn, replayed verbatim on later requests:
    # {"format": <api>, ...}. Opaque outside its adapter.
    native: dict[str, Any] | None = None
    # Reasoning returned only in encrypted or omitted form. Such a turn is
    # not empty, although it has no readable output.
    opaque_reasoning: bool = False
    tool_calls: list[NormalizedToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Thinking tokens inside completion_tokens, when the provider itemizes them.
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
            or self.opaque_reasoning
        )


@dataclass
class ModelCall:
    """One request, before translation to a provider's wire format."""

    model: str
    """The provider's model id (the run's model name without its route)."""
    messages: list[Message]
    tools: list[dict[str, Any]] | None
    timeout: float
    reasoning_effort: str | None = None
    temperature: float | None = None
    max_thinking_tokens: int | None = None
    session_id: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)
    """``llm_call_kwargs``; they override every other request setting."""


class Provider(ABC):
    """One API format, with its SDK client."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None,
        base_url: str | None,
        registry: ModelRegistry,
    ):
        self.model_name = model_name
        self.registry = registry
        self.client = self.make_client(api_key, base_url)

    @abstractmethod
    def make_client(self, api_key: str | None, base_url: str | None) -> Any: ...

    @abstractmethod
    async def complete(self, call: ModelCall) -> ParsedResponse: ...

    def model_info(self, response_model: str | None = None) -> ModelInfo | None:
        """Metadata for the model that served a response (falls back to the
        requested model)."""
        route = self.model_name.split("/", 1)[0]
        if response_model:
            served = self.registry.lookup(f"{route}/{response_model}")
            if served:
                return served
        return self.registry.lookup(self.model_name)


def merge_kwargs(request: dict[str, Any], method: Callable[..., Any], kwargs: dict):
    """Apply ``llm_call_kwargs`` last, then move every key the SDK method does
    not take into ``extra_body`` (the SDKs reject unknown keyword arguments).
    An explicit ``extra_body`` wins over moved keys."""
    request.update(kwargs)
    known = set(inspect.signature(method).parameters)
    moved = {key: request.pop(key) for key in list(request) if key not in known}
    if moved:
        request["extra_body"] = {**moved, **(request.get("extra_body") or {})}


def function_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """The ``function`` objects of OpenAI chat tool schemas."""
    return [tool["function"] for tool in tools or [] if tool.get("function")]


def native_of(message: Message, api: str) -> dict[str, Any] | None:
    native = message.get("native")
    if isinstance(native, dict) and native.get("format") == api:
        return native
    return None


def text_of(content: Any) -> str:
    """The text of chat-format content (a string or a list of parts)."""
    if isinstance(content, str):
        return content
    return "\n".join(
        part.get("text", "")
        for part in content or []
        if isinstance(part, dict) and part.get("type") == "text"
    )
