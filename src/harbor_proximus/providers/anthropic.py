"""Anthropic Messages API (Claude)."""

import copy
import json
from typing import Any

import anthropic
from harbor.llms.base import ContextLengthExceededError, OutputLengthExceededError

from harbor_proximus.model_info import completion_cost
from harbor_proximus.providers.base import (
    PLACEHOLDER,
    Message,
    ModelCall,
    NormalizedToolCall,
    ParsedResponse,
    Provider,
    function_tools,
    merge_kwargs,
    native_of,
)

API = "anthropic"
# 1h TTL: tool calls regularly outlive Anthropic's default 5m cache.
CACHE_CONTROL: dict[str, Any] = {"type": "ephemeral", "ttl": "1h"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Output cap when the model has no table entry (the API requires one).
DEFAULT_MAX_TOKENS = 32000
# Response blocks in their request shape; other block types replay as-is.
_REQUEST_KEYS = {
    "text": ("type", "text"),
    "thinking": ("type", "thinking", "signature"),
    "redacted_thinking": ("type", "data"),
    "tool_use": ("type", "id", "name", "input"),
}


class AnthropicProvider(Provider):
    def make_client(
        self, api_key: str | None, base_url: str | None
    ) -> anthropic.AsyncAnthropic:
        # Retries belong to the agent's budget-aware loop, not the SDK.
        return anthropic.AsyncAnthropic(
            api_key=api_key, base_url=base_url, max_retries=0
        )

    async def complete(self, call: ModelCall) -> ParsedResponse:
        request = self.build_request(call)
        fallbacks = request.pop("fallbacks", None)
        # Always streamed: long thinking turns outlast non-streaming timeouts.
        if fallbacks is None:
            stream = self.client.messages.stream(**request)
        else:
            betas = [*request.pop("betas", []), FALLBACK_BETA]
            stream = self.client.beta.messages.stream(
                **request, betas=betas, fallbacks=fallbacks
            )
        async with stream as events:
            message = await events.get_final_message()
        return self.parse(
            message.model_dump(mode="json", exclude_none=True), request, fallbacks
        )

    def build_request(self, call: ModelCall) -> dict[str, Any]:
        """The ``messages.stream`` arguments (``fallbacks`` popped by the caller)."""
        info = self.model_info() or {}
        system, messages = to_wire(call.messages)
        apply_cache_control(system, messages)
        request: dict[str, Any] = {
            "model": call.model,
            "max_tokens": int(info.get("max_output_tokens") or DEFAULT_MAX_TOKENS),
            "messages": messages,
            "timeout": call.timeout,
        }
        if system:
            request["system"] = system
        if call.tools is not None:
            request["tools"] = [
                {
                    "name": function["name"],
                    "description": function.get("description", ""),
                    "input_schema": function.get("parameters")
                    or {"type": "object", "properties": {}},
                }
                for function in function_tools(call.tools)
            ]
        if call.max_thinking_tokens is not None:
            request["thinking"] = {
                "type": "enabled",
                "budget_tokens": call.max_thinking_tokens,
            }
        elif call.reasoning_effort is not None:
            # Without display the API omits the thinking text.
            request["thinking"] = {"type": "adaptive", "display": "summarized"}
            request["output_config"] = {"effort": call.reasoning_effort}
        if call.temperature is not None:
            request["temperature"] = call.temperature
        kwargs = dict(call.kwargs)
        kwargs.pop("stream", None)
        fallbacks = kwargs.pop("anthropic_fallbacks", None)
        method = (
            self.client.messages.stream
            if fallbacks is None
            else self.client.beta.messages.stream
        )
        merge_kwargs(request, method, kwargs)
        if fallbacks is not None:
            request["fallbacks"] = fallbacks
        return request

    def parse(
        self, body: dict[str, Any], request: dict[str, Any], fallbacks: Any
    ) -> ParsedResponse:
        blocks = replayable_blocks(body.get("content") or [])
        text = "".join(b["text"] for b in blocks if b["type"] == "text")
        stop_reason = body.get("stop_reason")
        if stop_reason == "max_tokens":
            raise OutputLengthExceededError(
                f"Model {self.model_name} hit the max output tokens limit.",
                truncated_response=text,
            )
        if stop_reason == "model_context_window_exceeded":
            raise ContextLengthExceededError(
                f"Model {self.model_name} filled its context window."
            )
        thinking = [b["thinking"] for b in blocks if b["type"] == "thinking"]
        usage = body.get("usage") or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        read = int(usage.get("cache_read_input_tokens") or 0)
        written = int(usage.get("cache_creation_input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        details = usage.get("output_tokens_details") or {}
        iterations = [
            {
                key: item.get(key)
                for key in ("type", "model", "input_tokens", "output_tokens")
            }
            for item in usage.get("iterations") or []
        ]
        attempted = any(item["type"] == "fallback_message" for item in iterations)
        response_model = body.get("model") or None
        return ParsedResponse(
            content=text,
            reasoning_content="\n".join(t for t in thinking if t) or None,
            native={"format": API, "content": blocks},
            opaque_reasoning=any(
                b["type"] in ("thinking", "redacted_thinking") for b in blocks
            ),
            tool_calls=[
                NormalizedToolCall(b["id"], b["name"], json.dumps(b["input"]))
                for b in blocks
                if b["type"] == "tool_use"
            ],
            prompt_tokens=input_tokens + read + written,
            completion_tokens=output_tokens,
            reasoning_tokens=int(details.get("thinking_tokens") or 0),
            cached_tokens=read,
            # Every breakpoint Proximus sets has the 1h TTL.
            cost_usd=completion_cost(
                self.model_info(response_model),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=read,
                cache_write_1h_tokens=written,
            ),
            response_model=response_model,
            fallback_attempted=attempted,
            fallback_used=attempted and stop_reason != "refusal",
            fallback_iterations=iterations,
            stop_reason=stop_reason,
            stop_details=body.get("stop_details"),
            request_thinking=copy.deepcopy(request.get("thinking")),
            request_output_config=copy.deepcopy(request.get("output_config")),
            request_fallbacks=None if fallbacks is None else str(fallbacks),
        )


def replayable_blocks(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Response blocks in their request shape, ready to replay verbatim.

    After a mid-output refusal fallback, ``fallback`` markers stay in place:
    the server validates the thinking runs on each side of one against it.
    Only the declined model's tool calls before the last marker are dropped;
    they never ran. Empty text blocks are rejected on replay.
    """
    boundary = max(
        (i for i, block in enumerate(content) if block["type"] == "fallback"),
        default=-1,
    )
    kept = []
    for index, block in enumerate(content):
        kind = block["type"]
        if (kind == "tool_use" and index < boundary) or (
            kind == "text" and not block["text"]
        ):
            continue
        keys = _REQUEST_KEYS.get(kind)
        kept.append({k: block[k] for k in keys} if keys else block)
    return kept


def content_blocks(content: Any) -> list[dict[str, Any]]:
    """Chat-format content as Anthropic blocks (text and data-URL images)."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[dict[str, Any]] = []
    for part in content or []:
        if part.get("type") == "text" and part.get("text"):
            blocks.append({"type": "text", "text": part["text"]})
        elif part.get("type") == "image_url":
            header, data = part["image_url"]["url"].split(";base64,", 1)
            source = {"type": "base64", "media_type": header[5:], "data": data}
            blocks.append({"type": "image", "source": source})
    return blocks


def assistant_blocks(message: Message) -> list[dict[str, Any]]:
    native = native_of(message, API)
    if native:
        blocks = copy.deepcopy(native["content"])
        # A turn ending in thinking is rejected; such turns replay as text.
        if any(b["type"] in ("text", "tool_use") for b in blocks):
            return blocks
    blocks = content_blocks(message.get("content"))
    for call in message.get("tool_calls") or []:
        try:
            arguments = json.loads(call["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            arguments = {}  # the tool already reported the error
        blocks.append(
            {
                "type": "tool_use",
                "id": call["id"],
                "name": call["function"]["name"],
                "input": arguments if isinstance(arguments, dict) else {},
            }
        )
    return blocks or [{"type": "text", "text": PLACEHOLDER}]


def to_wire(
    messages: list[Message],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Chat history as (system blocks, Messages API turns).

    Consecutive tool results and the user message after them (attached
    images, continue prompts) form one user turn, tool results first.
    """
    system: list[dict[str, Any]] = []
    wire: list[dict[str, Any]] = []

    def append(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if wire and wire[-1]["role"] == role:
            wire[-1]["content"].extend(blocks)
        else:
            wire.append({"role": role, "content": blocks})

    for message in messages:
        role = message["role"]
        content = message.get("content")
        if role == "system" and not wire:
            system.extend(content_blocks(content))
        elif role in ("system", "user"):
            append("user", content_blocks(content))
        elif role == "tool":
            result = content if isinstance(content, str) else content_blocks(content)
            append(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": result,
                    }
                ],
            )
        elif role == "assistant":
            append("assistant", assistant_blocks(message))
    return system, wire


def apply_cache_control(
    system: list[dict[str, Any]], messages: list[dict[str, Any]]
) -> None:
    """Cache breakpoints on the system prompt and the latest block that can
    carry one (not an image or thinking block)."""
    if system:
        system[-1]["cache_control"] = dict(CACHE_CONTROL)
    for message in reversed(messages):
        for block in reversed(message["content"]):
            if block["type"] in ("text", "tool_result", "tool_use"):
                block["cache_control"] = dict(CACHE_CONTROL)
                return
