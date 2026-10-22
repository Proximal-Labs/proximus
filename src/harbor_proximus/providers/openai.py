"""OpenAI Responses API (GPT models) and Chat Completions (other
OpenAI-compatible endpoints)."""

import copy
from typing import Any

import openai
from harbor.llms.base import OutputLengthExceededError
from openai.types.chat import ChatCompletion
from openai.types.responses import Response

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
    text_of,
)

RESPONSES_API = "openai-responses"


class _OpenAIProvider(Provider):
    def make_client(
        self, api_key: str | None, base_url: str | None
    ) -> openai.AsyncOpenAI:
        # Retries belong to the agent's budget-aware loop, not the SDK.
        return openai.AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    def _cost(self, model: str | None, prompt: int, cached: int, out: int) -> float:
        return completion_cost(
            self.model_info(model),
            input_tokens=prompt - cached,
            output_tokens=out,
            cache_read_tokens=cached,
        )


class OpenAIResponsesProvider(_OpenAIProvider):
    async def complete(self, call: ModelCall) -> ParsedResponse:
        response = await self.client.responses.create(**self.build_request(call))
        return self.parse(response)

    def build_request(self, call: ModelCall) -> dict[str, Any]:
        # store=False with encrypted reasoning: the run replays its own chain
        # of thought instead of relying on server-side state.
        request: dict[str, Any] = {
            "model": call.model,
            "input": to_responses_wire(call.messages),
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "timeout": call.timeout,
        }
        if call.tools is not None:
            request["tools"] = [
                {
                    "type": "function",
                    "name": function["name"],
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters") or {},
                    # Strict schemas reject Proximus's optional arguments.
                    "strict": False,
                }
                for function in function_tools(call.tools)
            ]
        if call.reasoning_effort is not None:
            request["reasoning"] = {"effort": call.reasoning_effort, "summary": "auto"}
        if call.session_id:
            request["prompt_cache_key"] = call.session_id
        if call.temperature is not None:
            request["temperature"] = call.temperature
        merge_kwargs(request, self.client.responses.create, call.kwargs)
        return request

    def parse(self, response: Response) -> ParsedResponse:
        details = response.incomplete_details
        if details is not None and details.reason == "max_output_tokens":
            raise OutputLengthExceededError(
                f"Model {self.model_name} hit the max output tokens limit.",
                truncated_response=response.output_text,
            )
        output = [
            item.model_dump(mode="json", exclude_none=True) for item in response.output
        ]
        usage = response.usage
        prompt = usage.input_tokens if usage else 0
        cached = usage.input_tokens_details.cached_tokens if usage else 0
        completion = usage.output_tokens if usage else 0
        summaries = [
            part["text"]
            for item in output
            if item["type"] == "reasoning"
            for part in item.get("summary", [])
        ]
        return ParsedResponse(
            content=response.output_text,
            reasoning_content="\n".join(summaries) or None,
            native={"format": RESPONSES_API, "output": output} if output else None,
            opaque_reasoning=any(item["type"] == "reasoning" for item in output),
            tool_calls=[
                NormalizedToolCall(item["call_id"], item["name"], item["arguments"])
                for item in output
                if item["type"] == "function_call"
            ],
            prompt_tokens=prompt,
            completion_tokens=completion,
            reasoning_tokens=usage.output_tokens_details.reasoning_tokens
            if usage
            else 0,
            cached_tokens=cached,
            cost_usd=self._cost(response.model, prompt, cached, completion),
            response_model=response.model,
        )


def to_responses_wire(messages: list[Message]) -> list[dict[str, Any]]:
    """History as Responses input items; the model's own turns replay as
    the output items it returned."""
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "assistant":
            native = native_of(message, RESPONSES_API)
            if native:
                items.extend(copy.deepcopy(native["output"]))
                # A reasoning item must be followed by a message or call.
                if all(item["type"] == "reasoning" for item in native["output"]):
                    items.append(
                        {"type": "message", "role": "assistant", "content": PLACEHOLDER}
                    )
                continue
            calls = message.get("tool_calls") or []
            text = text_of(message.get("content"))
            if text or not calls:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": text or PLACEHOLDER,
                    }
                )
            items += [
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"] or "{}",
                }
                for call in calls
            ]
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": text_of(message.get("content")),
                }
            )
        else:
            items.append(
                {"type": "message", "role": role, "content": _input_parts(message)}
            )
    return items


def _input_parts(message: Message) -> list[dict[str, Any]]:
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    parts = []
    for part in content or []:
        if part.get("type") == "text":
            parts.append({"type": "input_text", "text": part["text"]})
        elif part.get("type") == "image_url":
            image = part["image_url"]
            parts.append(
                {
                    "type": "input_image",
                    "image_url": image["url"],
                    "detail": image.get("detail") or "auto",
                }
            )
    return parts


class OpenAIChatProvider(_OpenAIProvider):
    """Plain Chat Completions, for OpenAI-compatible endpoints."""

    async def complete(self, call: ModelCall) -> ParsedResponse:
        request: dict[str, Any] = {
            "model": call.model,
            # History is already in chat format; drop Proximus's own keys.
            "messages": [
                {key: value for key, value in message.items() if key != "native"}
                for message in call.messages
            ],
            "timeout": call.timeout,
        }
        if call.tools is not None:
            request["tools"] = call.tools
        if call.reasoning_effort is not None:
            request["reasoning_effort"] = call.reasoning_effort
        if call.temperature is not None:
            request["temperature"] = call.temperature
        merge_kwargs(request, self.client.chat.completions.create, call.kwargs)
        return self.parse(await self.client.chat.completions.create(**request))

    def parse(self, response: ChatCompletion) -> ParsedResponse:
        choice = response.choices[0]
        message = choice.message
        if choice.finish_reason == "length":
            raise OutputLengthExceededError(
                f"Model {self.model_name} hit the max output tokens limit.",
                truncated_response=message.content or "",
            )
        usage = response.usage
        prompt = usage.prompt_tokens if usage else 0
        details = usage.prompt_tokens_details if usage else None
        cached = (details.cached_tokens or 0) if details else 0
        completion = usage.completion_tokens if usage else 0
        return ParsedResponse(
            content=message.content or "",
            tool_calls=[
                NormalizedToolCall(
                    call.id, call.function.name, call.function.arguments or "{}"
                )
                for call in message.tool_calls or []
                if call.type == "function"
            ],
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_tokens=cached,
            cost_usd=self._cost(response.model, prompt, cached, completion),
            response_model=response.model,
        )
