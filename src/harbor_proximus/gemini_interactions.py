"""Stateless Gemini Interactions through LiteLLM."""

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

import litellm
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    ModelResponse,
)
from litellm.types.utils import Message as LiteLLMMessage

from harbor_proximus.prompts import NO_OUTPUT_PLACEHOLDER

STEPS_FIELD = "interaction_steps"
type Message = dict[str, Any]
_GENERATION_KEYS = {
    "max_tokens": "max_output_tokens",
    "max_output_tokens": "max_output_tokens",
    "temperature": "temperature",
    "top_p": "top_p",
    "seed": "seed",
    "stop": "stop_sequences",
    "stop_sequences": "stop_sequences",
}


@dataclass(frozen=True)
class Request:
    body: dict[str, Any]
    options: dict[str, Any]


def _text_content(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _user_content(content: Any) -> list[dict[str, Any]]:
    """Convert chat text and image parts to Interaction content parts."""
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


def to_input(
    messages: list[Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert chat messages into system instructions and Interaction steps."""
    system: list[str] = []
    steps: list[dict[str, Any]] = []
    call_names: dict[str, str] = {}
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
    return ("\n\n".join(system) or None), steps


def to_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """OpenAI-style function tools as Interaction function declarations."""
    declared: list[dict[str, Any]] = []
    for tool in tools or []:
        function = tool.get("function") if tool.get("type") == "function" else None
        if not isinstance(function, dict):
            continue
        entry: dict[str, Any] = {"type": "function", "name": function["name"]}
        if function.get("description"):
            entry["description"] = function["description"]
        if function.get("parameters"):
            entry["parameters"] = function["parameters"]
        declared.append(entry)
    return declared


def build_request(
    model_name: str,
    messages: list[Message],
    tools: list[dict[str, Any]] | None,
    *,
    reasoning_effort: str | None,
    temperature: float | None,
    call_kwargs: dict[str, Any],
) -> Request:
    system, steps = to_input(messages)
    if reasoning_effort is None:
        raise ValueError("Gemini Interactions requires reasoning_effort")
    generation: dict[str, Any] = {
        "thinking_level": reasoning_effort,
        "thinking_summaries": "auto",
    }
    if temperature is None:
        raise ValueError("Gemini Interactions requires temperature")
    generation["temperature"] = temperature
    body: dict[str, Any] = {
        "model": model_name,
        "input": steps,
        "store": False,
    }
    if system:
        body["system_instruction"] = system
    declared = to_tools(tools)
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
    if isinstance(options.get("extra_body"), dict):
        options["extra_body"] = {
            key: value
            for key, value in options["extra_body"].items()
            if key not in {"store", "previous_interaction_id"}
        }
    body.pop("previous_interaction_id", None)
    body["store"] = False
    return Request(body, options)


def to_model_response(
    model_name: str,
    steps: list[dict[str, Any]],
    status: str | None,
    hidden_params: dict[str, Any],
) -> ModelResponse:
    """Convert model steps to a chat choice, retaining raw steps for replay."""
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
        provider_specific_fields={STEPS_FIELD: steps},
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
    response = ModelResponse(
        id=f"interaction-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=model_name,
        choices=[choice],
    )

    response._hidden_params = dict(hidden_params)
    return response


class InteractionsClient:
    """Use LiteLLM for transport, logging, cost and exception mapping."""

    def __init__(self, model_name: str, api_key: str | None, api_base: str | None):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base

    async def create(self, request: Request) -> ModelResponse:
        interaction: Any = await litellm.interactions.acreate(
            **request.body,
            **request.options,
            api_key=self.api_key,
            api_base=self.api_base,
        )
        return to_model_response(
            self.model_name,
            interaction.steps or [],
            interaction.status,
            interaction._hidden_params,
        )
