"""The documented execution order also applies to multi-call model responses."""

import asyncio

from harbor.models.agent.context import AgentContext

from tests.test_proximus_loop import (
    FakeCompletion,
    FakeEnv,
    FakeMessage,
    FakeResponse,
    FakeToolCall,
    make_agent,
    read_messages,
)
from tests.test_submission_mode import done_response


async def test_multiple_bash_calls_finish_in_order(tmp_path, monkeypatch):
    events = []

    class OrderedEnv(FakeEnv):
        async def exec(self, command, **kwargs):
            if command not in {"first", "second"}:
                return await super().exec(command, **kwargs)
            events.append((command, "start"))
            await asyncio.sleep(0)
            result = await super().exec(command, **kwargs)
            events.append((command, "end"))
            return result

    fake = FakeCompletion(
        [
            FakeResponse(
                FakeMessage(
                    tool_calls=[
                        FakeToolCall("first", "bash", '{"command":"first"}'),
                        FakeToolCall("second", "bash", '{"command":"second"}'),
                    ]
                )
            ),
            done_response(),
        ]
    )
    fake.install(monkeypatch)
    agent = make_agent(tmp_path, submission_mode="none")
    await agent.run("Work in order", OrderedEnv(), AgentContext())
    assert events == [
        ("first", "start"),
        ("first", "end"),
        ("second", "start"),
        ("second", "end"),
    ]
    assert [m["tool_call_id"] for m in read_messages(agent) if m["role"] == "tool"] == [
        "first",
        "second",
        "finish",
    ]
