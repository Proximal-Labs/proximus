"""Gemini Interactions API path: the real google-genai SDK over a fake transport."""

import asyncio
import json
import subprocess
import sys
import time

import httpx
import litellm
import pytest
from google import genai
from harbor.llms.base import ContextLengthExceededError, OutputLengthExceededError
from harbor.models.agent.context import AgentContext
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

from harbor_proximus import Proximus
from harbor_proximus import gemini_interactions as gi
from harbor_proximus.llm import _is_retryable
from tests.test_proximus_loop import FakeEnv, read_messages

pytestmark = pytest.mark.unit

SIGNED_THOUGHT = {
    "type": "thought",
    "signature": "opaque-sig",
    "summary": [{"type": "text", "text": "Check the workspace."}],
}


def call_step(call_id: str, name: str, **arguments) -> dict:
    return {
        "type": "function_call",
        "id": call_id,
        "name": name,
        "arguments": arguments,
    }


def interaction(steps, status="requires_action", usage=None, id="i-1") -> dict:
    return {
        "id": id,
        "status": status,
        "steps": steps,
        "usage": usage or {"total_input_tokens": 100, "total_output_tokens": 10},
    }


def api_error(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": status, "message": message}})


class FakeServer:
    """The Interactions endpoint behind a real google-genai client.

    Each HTTP request takes the next reply; the last one repeats. A callable
    reply is awaited with the request. ``requests`` holds every JSON body sent
    on the wire, ``urls`` the URL each was posted to."""

    def __init__(self):
        self.replies: list = []
        self.requests: list[dict] = []
        self.urls: list[str] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        self.urls.append(str(request.url))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if callable(reply):
            reply = await reply(request)
        return (
            reply
            if isinstance(reply, httpx.Response)
            else httpx.Response(200, json=reply)
        )


@pytest.fixture
def server(monkeypatch):
    server = FakeServer()
    real_client = genai.Client

    def client(*, api_key=None, http_options=None):
        http_options.httpx_async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(server.handle)
        )
        return real_client(api_key=api_key, http_options=http_options)

    monkeypatch.setattr(genai, "Client", client)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return server


def make_agent(tmp_path, **kwargs) -> Proximus:
    return Proximus(
        logs_dir=tmp_path / "logs",
        model_name="gemini/test-model",
        model_api="gemini_interactions",
        reasoning_effort="high",
        **kwargs,
    )


def test_messages_become_steps():
    messages = [
        {"role": "system", "content": "Be careful."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look:"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        },
        {
            "role": "assistant",
            "content": "Running.",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
    ]
    system, steps = gi.to_input(messages)
    assert system == "Be careful."
    assert steps == [
        {
            "type": "user_input",
            "content": [
                {"type": "text", "text": "Look:"},
                {"type": "image", "mime_type": "image/png", "data": "AAAA"},
            ],
        },
        {"type": "model_output", "content": [{"type": "text", "text": "Running."}]},
        call_step("c1", "bash", command="ls"),
        {
            "type": "function_result",
            "call_id": "c1",
            "result": [{"type": "text", "text": "a.txt"}],
            "name": "bash",
        },
    ]


def test_recorded_steps_replay_verbatim():
    recorded = [SIGNED_THOUGHT, call_step("c1", "bash", command="ls")]
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [],
        "provider_specific_fields": {gi.STEPS_FIELD: recorded},
    }
    assert gi.to_input([message])[1] == recorded


def test_response_maps_to_a_chat_choice():
    steps = [
        SIGNED_THOUGHT,
        {"type": "model_output", "content": [{"type": "text", "text": "Listing."}]},
        call_step("c9", "bash", command="ls"),
    ]
    usage = {
        "total_input_tokens": 1000,
        "total_output_tokens": 50,
        "total_thought_tokens": 200,
        "total_cached_tokens": 600,
        "total_tokens": 1250,
    }
    response = gi.to_model_response(
        "gemini/test-model", steps, "requires_action", usage
    )
    message = response.choices[0].message
    assert message.content == "Listing."
    assert message.reasoning_content == "Check the workspace."
    assert message.tool_calls[0].id == "c9"
    assert json.loads(message.tool_calls[0].function.arguments) == {"command": "ls"}
    assert message.provider_specific_fields == {gi.STEPS_FIELD: steps}
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.usage.completion_tokens == 250  # output + thoughts
    assert response.usage.completion_tokens_details.reasoning_tokens == 200
    assert response.usage.prompt_tokens_details.cached_tokens == 600


def test_request_settings():
    request = gi.build_request(
        "gemini/test-model",
        [{"role": "user", "content": "hi"}],
        [
            {
                "type": "function",
                "function": {"name": "bash", "parameters": {"type": "object"}},
            }
        ],
        reasoning_effort="high",
        temperature=None,
        call_kwargs={"timeout": 60, "max_tokens": 4096, "service_tier": "priority"},
    )
    body = request.body
    assert body["model"] == "test-model"
    assert body["store"] is True
    assert "previous_interaction_id" not in body
    assert body["generation_config"] == {
        "thinking_level": "high",
        "thinking_summaries": "auto",
        "max_output_tokens": 4096,
    }
    assert body["tools"] == [
        {"type": "function", "name": "bash", "parameters": {"type": "object"}}
    ]
    assert body["service_tier"] == "priority"
    assert request.options == {"timeout": 60}


def build(messages, **call_kwargs) -> gi.Request:
    return gi.build_request(
        "gemini/test-model",
        messages,
        None,
        reasoning_effort=None,
        temperature=None,
        call_kwargs=call_kwargs,
    )


def linked_turn(previous: gi.Request, steps: list, interaction_id: str) -> dict:
    """An assistant turn as the client records it after a stored interaction."""
    digest = gi.history_digest(previous.system, previous.history + steps)
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "ls"}'},
            }
        ],
        "provider_specific_fields": {
            gi.STEPS_FIELD: steps,
            gi.LINK_FIELD: {"id": interaction_id, "history": digest},
        },
    }


def test_stored_interaction_is_continued_while_history_matches():
    first = [{"role": "user", "content": "List the files."}]
    turn = linked_turn(build(first), [call_step("c1", "bash", command="ls")], "i-1")
    result = {"role": "tool", "tool_call_id": "c1", "content": "a.txt"}

    continued = build([*first, turn, result]).body
    assert continued["previous_interaction_id"] == "i-1"
    assert [step["type"] for step in continued["input"]] == ["function_result"]

    # Compaction or image eviction rewrote the history the server holds.
    rewritten = [{"role": "user", "content": "List the files, briefly."}]
    resent = build([*rewritten, turn, result]).body
    assert "previous_interaction_id" not in resent
    assert [step["type"] for step in resent["input"]] == [
        "user_input",
        "function_call",
        "function_result",
    ]

    stateless = build([*first, turn, result], store=False).body
    assert stateless["store"] is False
    assert "previous_interaction_id" not in stateless
    assert len(stateless["input"]) == 3


async def test_run_replays_thought_signatures_and_results(tmp_path, server):
    server.replies = [
        interaction([SIGNED_THOUGHT, call_step("c1", "bash", command="echo hi")]),
        interaction([call_step("c2", "submit")], id="i-2"),
        interaction([call_step("c3", "submit")], id="i-3"),
    ]
    agent = make_agent(tmp_path)
    env = FakeEnv()
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    assert env.exec_calls == ["echo hi"]
    first, second, third = server.requests
    assert first["store"] is True
    assert "previous_interaction_id" not in first
    assert first["generation_config"]["thinking_level"] == "high"
    assert first["input"][0]["type"] == "user_input"
    # Later turns continue the stored interaction and send only the new steps;
    # tools and generation_config are interaction-scoped, so they are resent.
    assert second["previous_interaction_id"] == "i-1"
    assert second["input"][0] == {
        "type": "function_result",
        "call_id": "c1",
        "name": "bash",
        "result": [{"type": "text", "text": "exit code: 0\nok"}],
    }
    assert SIGNED_THOUGHT not in second["input"]
    assert second["tools"] == first["tools"]
    assert third["previous_interaction_id"] == "i-2"
    assistant = [m for m in read_messages(agent) if m.get("role") == "assistant"]
    fields = assistant[0]["provider_specific_fields"]
    assert fields[gi.STEPS_FIELD][0] == SIGNED_THOUGHT
    assert fields[gi.LINK_FIELD]["id"] == "i-1"


@pytest.mark.parametrize(
    ("status", "message", "retryable"),
    [
        (429, "RESOURCE_EXHAUSTED", True),
        (500, "internal", True),
        (503, "unavailable", True),
        (400, "invalid argument", False),
        (401, "API key not valid", False),
        (403, "permission denied", False),
        (404, "model not found", False),
    ],
)
async def test_api_errors_keep_chat_retry_rules(server, status, message, retryable):
    server.replies = [api_error(status, message)]
    client = gi.InteractionsClient("gemini/test-model", None, None)
    with pytest.raises(Exception) as caught:
        await client.create(
            gi.Request({"model": "test-model", "input": "hi"}, {}, None, [])
        )
    assert f"Error code: {status}" in str(caught.value)
    assert _is_retryable(caught.value) is retryable


async def test_unknown_request_field_is_not_retried(server):
    server.replies = [interaction([])]
    client = gi.InteractionsClient("gemini/test-model", None, None)
    request = build([{"role": "user", "content": "hi"}], max_completion_tokens=100)
    with pytest.raises(litellm.exceptions.BadRequestError) as caught:
        await client.create(request)
    assert not _is_retryable(caught.value)
    assert server.requests == []


async def test_request_timeout_bounds_the_whole_call(server):
    async def slow(request):
        await asyncio.sleep(3)
        return httpx.Response(200, json=interaction([]))

    server.replies = [slow]
    client = gi.InteractionsClient("gemini/test-model", None, None)
    started = time.monotonic()
    with pytest.raises(litellm.exceptions.Timeout) as caught:
        await client.create(
            gi.Request(
                {"model": "test-model", "input": "hi"}, {"timeout": 1.0}, None, []
            )
        )
    assert time.monotonic() - started < 1.5
    assert _is_retryable(caught.value)


def test_missing_api_key_fails_at_construction(tmp_path, server, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        make_agent(tmp_path)


async def test_context_overflow_is_reported(tmp_path, server):
    server.replies = [
        api_error(
            400,
            "The input token count (600000) exceeds the maximum number of tokens allowed (524288).",
        )
    ]
    agent = make_agent(tmp_path)
    with pytest.raises(ContextLengthExceededError):
        await agent._llm.query([{"role": "user", "content": "hi"}], tools=None)


async def test_incomplete_interaction_is_an_output_limit(tmp_path, server):
    server.replies = [
        interaction(
            [{"type": "model_output", "content": [{"type": "text", "text": "cut"}]}],
            status="incomplete",
        )
    ]
    agent = make_agent(tmp_path)
    with pytest.raises(OutputLengthExceededError):
        await agent._llm.query([{"role": "user", "content": "hi"}], tools=None)


def test_requires_a_gemini_model(tmp_path, server):
    with pytest.raises(ValueError, match="gemini/"):
        Proximus(
            logs_dir=tmp_path / "logs",
            model_name="openai/gpt-test",
            model_api="gemini_interactions",
        )


def test_chat_path_works_without_the_gemini_extra(tmp_path):
    # None in sys.modules makes `import google.genai` fail, as without the extra.
    script = (
        "import sys; sys.modules['google.genai'] = None\n"
        "from pathlib import Path\nfrom harbor_proximus import Proximus\n"
        f"Proximus(logs_dir=Path({str(tmp_path)!r}), model_name='openai/gpt-5')\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


class PayloadLogger(CustomLogger):
    """Collects the standard logging payload of every logged attempt."""

    def __init__(self):
        super().__init__()
        self.payloads: list[dict] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.payloads.append(kwargs["standard_logging_object"])

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.payloads.append(kwargs["standard_logging_object"])

    async def logged(self) -> list[dict]:
        await GLOBAL_LOGGING_WORKER.flush()
        return sorted(self.payloads, key=lambda payload: payload["startTime"])


@pytest.fixture
async def payloads(monkeypatch):
    # Earlier tests' success events may still be queued, on an earlier event
    # loop: move the worker to this one and deliver them first.
    GLOBAL_LOGGING_WORKER.start()
    await GLOBAL_LOGGING_WORKER.flush()
    logger = PayloadLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    # LiteLLM copies litellm.callbacks into these when a call is set up.
    for name in (
        "input_callback",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        monkeypatch.setattr(litellm, name, [])
    litellm.register_model(
        {
            "gemini/test-model": {
                "max_input_tokens": 200_000,
                "litellm_provider": "gemini",
                "mode": "chat",
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 4e-6,
                "cache_read_input_token_cost": 1e-7,
            }
        }
    )
    return logger


async def test_callbacks_log_each_stored_call(tmp_path, server, payloads):
    usage = {
        "total_input_tokens": 1000,
        "total_output_tokens": 50,
        "total_thought_tokens": 200,
        "total_cached_tokens": 600,
    }
    server.replies = [
        interaction([SIGNED_THOUGHT, call_step("c1", "bash", command="ls")]),
        interaction(
            [{"type": "model_output", "content": [{"type": "text", "text": "Done."}]}],
            status="completed",
            usage=usage,
            id="i-2",
        ),
    ]
    agent = make_agent(tmp_path)
    first = [{"role": "user", "content": "List the files."}]
    turn = await agent._llm.query(first, tools=None)
    second = [
        *first,
        agent._llm.assistant_message(turn),
        {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
    ]
    parsed = await agent._llm.query(second, tools=None)

    logged = await payloads.logged()
    assert [payload["status"] for payload in logged] == ["success", "success"]
    assert server.requests[1]["previous_interaction_id"] == "i-1"
    chained = logged[1]
    assert chained["messages"] == second
    assert chained["model"] == "test-model"
    assert chained["custom_llm_provider"] == "gemini"
    assert chained["call_type"] == "acompletion"
    assert chained["api_base"] == server.urls[1]
    assert (
        server.urls[1]
        == "https://generativelanguage.googleapis.com/v1beta/interactions"
    )
    assert chained["startTime"] <= chained["endTime"]
    assert chained["response_time"] == pytest.approx(
        chained["endTime"] - chained["startTime"], abs=0.01
    )
    assert chained["prompt_tokens"] == 1000
    assert chained["completion_tokens"] == 250  # output + thoughts
    tokens = chained["metadata"]["usage_object"]
    assert tokens["prompt_tokens_details"]["cached_tokens"] == 600
    assert tokens["completion_tokens_details"]["reasoning_tokens"] == 200
    assert chained["response_cost"] == pytest.approx(parsed.cost_usd)
    assert chained["response_cost"] == pytest.approx(
        400 * 1e-6 + 600 * 1e-7 + 250 * 4e-6
    )
    choice = chained["response"]["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "Done."
    assert choice["provider_specific_fields"] == {"interaction_status": "completed"}
    assert logged[0]["response"]["choices"][0]["finish_reason"] == "tool_calls"


async def test_each_failed_attempt_is_logged(tmp_path, server, payloads, monkeypatch):
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)
    server.replies = [
        api_error(503, "The model is overloaded."),
        api_error(503, "The model is overloaded."),
        interaction([call_step("c1", "submit")]),
    ]
    agent = make_agent(tmp_path)
    parsed = await agent._llm.query([{"role": "user", "content": "hi"}], tools=None)

    assert parsed.tool_calls[0].name == "submit"
    # One HTTP request per attempt: the SDK retries nothing on its own.
    assert len(server.requests) == 3
    logged = await payloads.logged()
    assert [payload["status"] for payload in logged] == [
        "failure",
        "failure",
        "success",
    ]
    for failure in logged[:2]:
        error = failure["error_information"]
        assert error["error_code"] == "503"
        assert error["error_class"] == "ServiceUnavailableError"
        assert "The model is overloaded." in error["error_message"]
        assert failure["api_base"] == server.urls[0]
        assert failure["model"] == "test-model"
        assert failure["custom_llm_provider"] == "gemini"
        assert failure["messages"] == [{"role": "user", "content": "hi"}]
        assert failure["startTime"] <= failure["endTime"]
    assert logged[0]["id"] != logged[1]["id"]


@pytest.mark.parametrize(
    ("status", "steps", "errors"),
    [
        ("failed", [], [{"code": "safety/blocked", "message": "Blocked for safety."}]),
        (
            "completed",
            [{"type": "model_output", "error": {"code": 3, "message": "No output."}}],
            None,
        ),
    ],
)
async def test_interaction_errors_keep_the_raw_details(
    server, payloads, status, steps, errors
):
    reply = interaction(steps, status=status)
    if errors:
        reply["errors"] = errors
    server.replies = [reply]
    client = gi.InteractionsClient("gemini/test-model", None, None)
    with pytest.raises(litellm.exceptions.InternalServerError) as caught:
        await client.create(build([{"role": "user", "content": "hi"}]))
    message = str(caught.value)
    assert f"interaction status {status!r}" in message
    for error in errors or [steps[0]["error"]]:
        assert json.dumps(error)[1:-1] in message
    assert _is_retryable(caught.value)
    (logged,) = await payloads.logged()
    assert logged["status"] == "failure"
    assert logged["error_information"]["error_message"] == message
