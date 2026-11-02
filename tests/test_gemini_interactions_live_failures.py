"""Interactions failure modes seen live on gemini/carbon-g, against a fake server.

In 272 one-hour FrontierSWE trials on the Interactions API, 11 got a model_output
holding only an empty text block with citations (after which every request was
rejected) and 1 got a recitation 400; neither may end a trial.
"""

import asyncio
import json
import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import litellm
import pytest

from harbor_proximus import gemini_interactions as gi
from harbor_proximus.config import ProximusConfig
from harbor_proximus.llm import LLM, _is_retryable

_REAL_CONNECT = socket.socket.connect
SIGNED_THOUGHT = {
    "type": "thought",
    "signature": "opaque-sig",
    "summary": [{"type": "text", "text": "Check the workspace."}],
}
# Seen live: a model_output whose only text block is empty and carries citations.
CITATION_ONLY_OUTPUT = {
    "type": "model_output",
    "content": [
        {
            "type": "text",
            "text": "",
            "annotations": [{"type": "url_citation", "url": "https://example.com"}],
        }
    ],
}
USAGE = {"total_input_tokens": 100, "total_output_tokens": 10}


def call_step(call_id: str, name: str, **arguments) -> dict:
    return {
        "type": "function_call",
        "id": call_id,
        "name": name,
        "arguments": arguments,
    }


def interaction(steps, status="requires_action") -> dict:
    return {"id": "i-1", "status": status, "steps": steps, "usage": USAGE}


class FakeServer:
    """Replies in order: a dict (200 JSON), (status, dict), ("hang", seconds) or
    ("trickle", seconds) which sends headers then one body byte per 0.2 s."""

    def __init__(self):
        self.replies: list[Any] = []
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                outer.requests.append(
                    json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                )
                reply = outer.replies.pop(0) if outer.replies else interaction([])
                if isinstance(reply, tuple) and reply[0] == "hang":
                    time.sleep(reply[1])
                    reply = interaction([])
                if isinstance(reply, tuple) and reply[0] == "trickle":
                    body = json.dumps(interaction([])).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    deadline = time.time() + reply[1]
                    for i, byte in enumerate(body):
                        if time.time() < deadline:
                            time.sleep(0.2)
                        try:
                            self.wfile.write(bytes([byte]))
                            self.wfile.flush()
                        except OSError:
                            return
                    return
                code, payload = reply if isinstance(reply, tuple) else (200, reply)
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_port}"


@pytest.fixture
def server(monkeypatch):
    def local_connect(sock, address):
        assert address[0] == "127.0.0.1"
        return _REAL_CONNECT(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_connect)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    fake = FakeServer()
    fake.thread.start()
    yield fake
    fake.httpd.shutdown()
    fake.httpd.server_close()


def make_llm(server: FakeServer, **call_kwargs) -> LLM:
    config = ProximusConfig(
        model_api="gemini_interactions",
        reasoning_effort="high",
        temperature=1.0,
        api_base=server.base,
        llm_call_kwargs=call_kwargs,
        model_info={"max_input_tokens": 1_000_000},
    )
    return LLM("gemini/test-model", config, lambda: None, logging.getLogger(__name__))


def tool_turn(llm: LLM, history: list, parsed, output: str = "a.txt") -> list:
    return [
        *history,
        *llm.assistant_messages(parsed),
        *(
            {"role": "tool", "tool_call_id": call.call_id, "content": output}
            for call in parsed.tool_calls
        ),
    ]


def has_empty_text(steps: list[dict]) -> bool:
    return any(
        part.get("type") == "text" and not part.get("text")
        for step in steps
        for part in (step.get("content") or step.get("result") or [])
        if isinstance(part, dict)
    )


def test_citation_only_output_is_not_replayed(server):
    """Live (runner bec7781): the API 400s every later request that contains this block
    ("Missing text in content of type text."), and 400s are not retried."""
    server.replies = [
        interaction(
            [
                SIGNED_THOUGHT,
                call_step("c1", "bash", command="ls"),
                CITATION_ONLY_OUTPUT,
            ]
        ),
        interaction([call_step("c2", "submit")]),
    ]
    llm = make_llm(server)
    first = [{"role": "user", "content": "List the files."}]

    async def turns():
        parsed = await llm.query(first, tools=None)
        return await llm.query(tool_turn(llm, first, parsed), tools=None)

    assert asyncio.run(turns()).tool_calls[0].name == "submit"
    replayed = server.requests[1]["input"]
    assert not has_empty_text(replayed)
    assert [step["type"] for step in replayed] == [
        "user_input",
        "thought",
        "function_call",
        "function_result",
    ]
    assert replayed[1] == SIGNED_THOUGHT


def test_recitation_block_is_not_fatal(server):
    """Live (runner bec7781): output blocked as recitation arrives as a 400 with code
    "recitation". The chat path gets an empty content_filter turn for the same event; here it
    should not end the trial."""
    server.replies = [
        (
            400,
            {
                "error": {
                    "code": 400,
                    "status": "INVALID_ARGUMENT",
                    "message": "Request blocked due to copyright/recitation content. "
                    "Please modify your input and retry.",
                }
            },
        )
    ]
    llm = make_llm(server)
    parsed = asyncio.run(
        llm.query([{"role": "user", "content": "Read lib.c."}], tools=None)
    )
    assert parsed.is_empty and not parsed.tool_calls
    assert len(server.requests) == 1  # not retried


def test_empty_tool_output_is_not_sent_as_empty_text(server):
    """The API rejects empty text blocks. Proximus's bash tool already says "(no output)",
    but any other tool message with empty content reaches the wire as text ''."""
    server.replies = [
        interaction([call_step("c1", "bash", command="true")]),
        interaction([]),
    ]
    llm = make_llm(server)
    first = [{"role": "user", "content": "Run true."}]

    async def turns():
        parsed = await llm.query(first, tools=None)
        return await llm.query(tool_turn(llm, first, parsed, output=""), tools=None)

    asyncio.run(turns())
    assert server.requests[1]["input"][-1]["result"] == [
        {"type": "text", "text": "(no output)"}
    ]


@pytest.mark.parametrize("mode", ["hang", "trickle"])
def test_call_is_bounded_by_timeout(server, mode):
    """A silent or slowly trickling reply must end near the configured timeout (the runner
    bounds the whole call with asyncio.timeout because HTTP timeouts are per read)."""
    server.replies = [(mode, 6.0)]
    llm = make_llm(server)
    request = gi.build_request(
        "gemini/test-model",
        [{"role": "user", "content": "hi"}],
        None,
        reasoning_effort="high",
        temperature=1.0,
        call_kwargs={"timeout": 1.0},
    )
    client = llm._interactions
    assert client is not None
    started = time.monotonic()
    with pytest.raises(litellm.exceptions.Timeout) as timed_out:
        asyncio.run(client.create(request))
    assert time.monotonic() - started < 3.0
    assert _is_retryable(timed_out.value)


def test_generatecontent_overlay_kwargs_are_rejected_at_config():
    """The generateContent overlay for carbon-g carries api_base and thinkingConfig in
    llm_call_kwargs. Forwarded, api_base collides with the client's own and raises a
    TypeError that the retry policy treats as transient; reject them when the config loads."""
    with pytest.raises(ValueError, match="api_base"):
        ProximusConfig(
            model_api="gemini_interactions",
            reasoning_effort="high",
            temperature=1.0,
            llm_call_kwargs={
                "api_base": "https://generativelanguage.googleapis.com/v1alpha",
                "thinkingConfig": {"thinkingLevel": "high", "includeThoughts": True},
            },
        )
    # The chat path still takes them.
    ProximusConfig(
        llm_call_kwargs={"api_base": "https://example.com", "thinkingConfig": {}}
    )
