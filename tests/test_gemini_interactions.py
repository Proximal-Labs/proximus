import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
from litellm import exceptions

from harbor_proximus import gemini_interactions as gi

_REAL_CONNECT = socket.socket.connect

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
    message: Any = response.choices[0].message
    assert message.content == "Listing."
    assert message.reasoning_content == "Check the workspace."
    assert message.tool_calls[0].id == "c9"
    assert json.loads(message.tool_calls[0].function.arguments) == {"command": "ls"}
    assert message.provider_specific_fields == {gi.STEPS_FIELD: steps}
    assert response.choices[0].finish_reason == "tool_calls"
    assert response._hidden_params == usage


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
        temperature=0.0,
        call_kwargs={"timeout": 60, "max_tokens": 4096, "service_tier": "priority"},
    )
    body = request.body
    assert body["model"] == "gemini/test-model"
    assert body["store"] is False
    assert "previous_interaction_id" not in body
    assert body["generation_config"] == {
        "thinking_level": "high",
        "thinking_summaries": "auto",
        "max_output_tokens": 4096,
        "temperature": 0.0,
    }
    assert body["tools"] == [
        {"type": "function", "name": "bash", "parameters": {"type": "object"}}
    ]
    assert body["service_tier"] == "priority"
    assert request.options == {"timeout": 60}


@pytest.mark.asyncio
async def test_litellm_transport_and_replay(monkeypatch):
    real_connect = _REAL_CONNECT

    def local_connect(sock, address):
        assert address[0] == "127.0.0.1"
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_connect)
    requests = []
    reply = interaction([SIGNED_THOUGHT, call_step("c1", "bash", command="ls")])
    code = 200

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(reply).encode())

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = gi.InteractionsClient(
        "gemini/gemini-3-flash-preview",
        "test-key",
        f"http://127.0.0.1:{server.server_port}",
    )

    def request(messages):
        return gi.build_request(
            client.model_name,
            messages,
            [],
            reasoning_effort="high",
            temperature=0.0,
            call_kwargs={},
        )

    try:
        response = await client.create(request([{"role": "user", "content": "hi"}]))
        assert len(requests) == 1
        assert requests[0]["store"] is False
        assert response._hidden_params.get("response_cost") is not None
        message: Any = response.choices[0].message
        steps = message.provider_specific_fields[gi.STEPS_FIELD]
        assert steps == reply["steps"]
        await client.create(
            request(
                [
                    {
                        "role": "assistant",
                        "provider_specific_fields": {gi.STEPS_FIELD: steps},
                    }
                ]
            )
        )
        assert len(requests) == 2
        assert requests[-1]["input"] == steps
        assert requests[-1]["store"] is False
        for status, message, error in [
            (
                400,
                "The input token count exceeds the maximum number of tokens allowed",
                exceptions.ContextWindowExceededError,
            ),
            (429, "rate limited", exceptions.RateLimitError),
            (503, "unavailable", exceptions.ServiceUnavailableError),
        ]:
            code = status
            reply = {"error": {"code": status, "message": message}}
            before = len(requests)
            with pytest.raises(error):
                await client.create(request([{"role": "user", "content": "hi"}]))
            assert len(requests) == before + 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_reasoning_required():
    with pytest.raises(ValueError, match="reasoning_effort"):
        gi.build_request(
            "gemini/test",
            [],
            [],
            reasoning_effort=None,
            temperature=0.0,
            call_kwargs={},
        )


def test_temperature_required():
    with pytest.raises(ValueError, match="Gemini Interactions requires temperature"):
        gi.build_request(
            "gemini/test-model",
            [],
            [],
            reasoning_effort="high",
            temperature=None,
            call_kwargs={},
        )


def test_config_default_temperature_error_reaches_caller(monkeypatch):
    import logging

    from harbor_proximus.config import ProximusConfig
    from harbor_proximus.llm import LLM

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    config = ProximusConfig(model_api="gemini_interactions", reasoning_effort="high")
    assert config.temperature is None
    with pytest.raises(ValueError, match="Gemini Interactions requires temperature"):
        LLM("gemini/test-model", config, lambda: None, logging.getLogger(__name__))


def test_config_default_reasoning_effort_error_reaches_caller(monkeypatch):
    import logging

    from harbor_proximus.config import ProximusConfig
    from harbor_proximus.llm import LLM

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    config = ProximusConfig(model_api="gemini_interactions", temperature=0.0)
    assert config.reasoning_effort is None
    with pytest.raises(
        ValueError, match="Gemini Interactions requires reasoning_effort"
    ):
        LLM("gemini/test-model", config, lambda: None, logging.getLogger(__name__))
