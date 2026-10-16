"""Unit tests for the Proximus agent loop (LLM calls mocked)."""

import json
import logging
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import litellm
import pytest
from harbor.agents.factory import AgentFactory
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import NetworkMode, NetworkPolicy
from harbor.models.trial.config import AgentConfig

from harbor_proximus import Proximus
from harbor_proximus.prompts import (
    CONTINUE_MESSAGE,
    EMPTY_RESPONSE_MESSAGE,
    OUTPUT_LENGTH_MESSAGE,
    PROGRESS_FILE,
    SUBMIT_GUIDANCE,
    TASK_TIMER_GUIDANCE,
    fmt_duration,
)
from harbor_proximus.timer import DEADLINE_PATH, write_deadline
from harbor_proximus.tools import EnvironmentGoneError

pytestmark = pytest.mark.unit


class FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.type = "function"
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(
        self,
        content=None,
        tool_calls=None,
        reasoning_content=None,
        provider_specific_fields=None,
    ):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content
        self.thinking_blocks = None
        self.provider_specific_fields = provider_specific_fields


class FakeChoice:
    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason


class FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_tokens_details = None


class FakeResponse:
    def __init__(self, message, prompt_tokens=100, completion_tokens=10):
        self.choices = [FakeChoice(message)]
        self.usage = FakeUsage(prompt_tokens, completion_tokens)
        self._hidden_params = {"response_cost": 0.01}


def bash_response(command: str, prompt_tokens: int = 100) -> FakeResponse:
    return FakeResponse(
        FakeMessage(
            content="Running a command.",
            tool_calls=[
                FakeToolCall("call_1", "bash", json.dumps({"command": command}))
            ],
        ),
        prompt_tokens=prompt_tokens,
    )


def submit_response(**arguments) -> FakeResponse:
    """A bare submit (defaults to a workspace snapshot; extra kwargs simulate a model passing
    stray arguments, which are ignored)."""
    return FakeResponse(
        FakeMessage(
            content="Done.",
            tool_calls=[FakeToolCall("call_f", "submit", json.dumps(arguments))],
        )
    )


def finish_responses() -> list[FakeResponse]:
    """Two consecutive submits: the second confirms the first as final and
    ends the run (the only way to end via submit)."""
    return [submit_response(), submit_response()]


def fake_hash(index: int) -> str:
    return f"{index:040x}"


def text_response(text: str) -> FakeResponse:
    return FakeResponse(FakeMessage(content=text))


class FakeEnv:
    """Environment double. ``has_timer`` emulates a task image that ships
    ``/app/timer.sh`` (the task-owned countdown); with it off, the re-anchor's
    probe reports the script missing, like a timer-less task. The
    harness-owned deadline file is written either way."""

    def __init__(self, has_timer: bool = True):
        self.exec_calls: list[str] = []
        self.all_commands: list[str] = []
        # Exec env vars, index-aligned with all_commands (the harness passes
        # script inputs like wd/commit/idx/winner out-of-band).
        self.all_envs: list[dict] = []
        self.all_users: list = []
        self.has_timer = has_timer
        self.task_env_config = SimpleNamespace(workdir="/app")
        self.network_policy = NetworkPolicy()

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.all_commands.append(command)
        self.all_envs.append(dict(env or {}))
        # Harness bookkeeping (workspace pre-create, repository setup,
        # submission recording, the end-of-run restore, the deadline file)
        # is kept out of exec_calls so tests assert on task commands only.
        if (
            not command.startswith("mkdir -p /tmp/proximus")
            and "is-inside-work-tree" not in command
            and "rev-parse -q --verify" not in command
            and 'echo "matched"' not in command
            and "refs/proximus/end-state" not in command
            and "/tmp/proximus/PROGRESS.md" not in command
            and DEADLINE_PATH not in command
            and "/app/.timer" not in command
        ):
            self.exec_calls.append(command)
        if "/app/.timer" in command and "timer.sh" in command:
            # The re-anchor echoes the fresh countdown, or the missing marker.
            stdout = "600" if self.has_timer else "missing"
            return ExecResult(stdout=stdout, stderr=None, return_code=0)
        if 'echo "matched"' in command:
            # The confirm path's winner match: a confirm without new work
            # in between matches the winner and reuses it.
            return ExecResult(stdout="matched", stderr=None, return_code=0)
        if "rev-parse -q --verify" in command:
            # The submit-record snippet's parsed output, one distinct commit
            # hash per record (the model commits new work between submits).
            n = sum(
                "rev-parse -q --verify" in c and 'echo "matched"' not in c
                for c in self.all_commands
            )
            return ExecResult(
                stdout=f"subject=wip solution\nhash={fake_hash(n)}",
                stderr=None,
                return_code=0,
            )
        return ExecResult(stdout="ok", stderr=None, return_code=0)

    async def download_file(self, source_path, target_path):
        raise FileNotFoundError(source_path)


class FakeCompletion:
    """Replaces litellm.acompletion; pops queued responses and records calls.
    A queued Exception instance is raised instead of returned."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("FakeCompletion ran out of queued responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_bridged_agent(tmp_path, **kwargs) -> Proximus:
    """An agent whose model actually satisfies litellm's chat->Responses
    bridge gate (OpenAI, gpt-5.4+, reasoning_effort, tools)."""
    kwargs.setdefault("reasoning_effort", "xhigh")
    return Proximus(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-5.4",
        **kwargs,
    )


def make_agent(tmp_path, **kwargs) -> Proximus:
    return Proximus(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-test",
        **kwargs,
    )


def small_context(monkeypatch, limit: int = 1000) -> None:
    """Pin the model's context limit (resolved in __init__ from litellm's
    registry) so a 900-token turn crosses the fixed 0.85 compaction
    threshold."""
    monkeypatch.setattr(
        litellm, "get_model_info", lambda model: {"max_input_tokens": limit}
    )


def read_trajectory(agent: Proximus) -> dict:
    return json.loads((agent.logs_dir / "trajectory.json").read_text())


def read_messages(agent: Proximus) -> list[dict]:
    return json.loads((agent.logs_dir / "messages.json").read_text())


def test_proximus_loads_through_custom_agent_factory():
    config = AgentConfig(import_path="harbor_proximus:Proximus")
    assert AgentFactory.get_agent_class_from_config(config) is Proximus


def test_requires_model_name(tmp_path):
    with pytest.raises(ValueError, match="model_name"):
        Proximus(logs_dir=tmp_path)


@pytest.mark.parametrize(
    ("secs", "expected"),
    [
        (22_200, "6h 10m"),
        (510, "8m 30s"),
        (25, "25s"),
        (3600, "1h"),
        (-1, "0s"),
    ],
)
def test_fmt_duration(secs, expected):
    assert fmt_duration(secs) == expected


async def test_basic_run_tool_call_then_submit(tmp_path, monkeypatch):
    fake = FakeCompletion([bash_response("echo hi"), *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    env = FakeEnv()
    context = AgentContext()
    await agent.run("Say hi", env, context)  # type: ignore[arg-type]

    assert env.exec_calls == ["echo hi"]
    assert len(fake.calls) == 3
    # Native FC: tools are passed on every call.
    assert all("tools" in call for call in fake.calls)

    assert context.n_input_tokens == 300
    assert context.n_output_tokens == 30
    assert context.cost_usd == pytest.approx(0.03)
    assert context.metadata is not None
    assert context.metadata["n_turns"] == 3
    # The confirm named the same commit, so it REUSED the armed record
    # instead of minting a second one (the empty-confirm-pin trap).
    assert context.metadata["n_submissions"] == 1
    submissions = context.metadata["submissions"]
    assert [s["index"] for s in submissions] == [1]
    assert submissions[0]["hash"] == fake_hash(1)
    assert submissions[0]["ok"] is True
    assert "record_sec" in submissions[0]
    assert context.metadata["submitted_commit"] == fake_hash(1)

    trajectory = read_trajectory(agent)
    sources = [step["source"] for step in trajectory["steps"]]
    assert sources == ["system", "user", "agent", "agent", "agent"]
    bash_step = trajectory["steps"][2]
    assert bash_step["tool_calls"][0]["function_name"] == "bash"
    assert bash_step["observation"]["results"][0]["source_call_id"] == "call_1"
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 300


async def test_submit_reports_snapshot_time_left_and_confirm_flow(
    tmp_path, monkeypatch
):
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path, time_budget_sec=7200)
    context = AgentContext()
    await agent.run("Do a task", FakeEnv(), context)  # type: ignore[arg-type]

    # First submit: recorded-commit note + time left + continue-or-confirm;
    # the consecutive submit confirms and ends the run.
    assert len(fake.calls) == 2
    messages = read_messages(agent)
    tool_contents = [m["content"] for m in messages if m.get("role") == "tool"]
    first, second = tool_contents
    assert f"recorded from commit {fake_hash(1)[:12]}" in first
    assert '"wip solution"' in first
    assert "time budget remains (100% left)" in first
    assert "confirm this as your final submission" in first
    assert "Final submission confirmed" in second
    user_contents = [m["content"] for m in messages if m.get("role") == "user"]
    # Budget is surfaced in the initial task message.
    assert "time budget of 2h" in user_contents[0]
    # No harness-injected status updates: the initial task message stays the
    # only user message. The model checks the time itself via the in-sandbox
    # deadline file.
    assert len(user_contents) == 1


async def test_end_gate_rearms_after_intervening_work(tmp_path, monkeypatch):
    # submit -> guidance -> bash (resets the gate) -> submit -> guidance
    # again -> consecutive submit -> confirmed.
    fake = FakeCompletion(
        [
            submit_response(),
            bash_response("make test"),
            *finish_responses(),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path, time_budget_sec=7200)
    context = AgentContext()
    await agent.run("Do a task", FakeEnv(), context)  # type: ignore[arg-type]

    assert len(fake.calls) == 4
    messages = read_messages(agent)
    tool_contents = [m["content"] for m in messages if m.get("role") == "tool"]
    assert sum("confirm this as your final submission" in c for c in tool_contents) == 2
    assert sum("Final submission confirmed" in c for c in tool_contents) == 1
    assert context.metadata is not None
    assert context.metadata["n_turns"] == 4
    # Two recordings; the final confirm reused the second one.
    assert context.metadata["n_submissions"] == 2


async def test_same_turn_double_submit_does_not_confirm(tmp_path, monkeypatch):
    # Two submits emitted in ONE turn must not self-confirm; the
    # confirmation has to come on a later turn.
    double_submit = FakeResponse(
        FakeMessage(
            content="Done.",
            tool_calls=[
                FakeToolCall("call_f1", "submit", "{}"),
                FakeToolCall("call_f2", "submit", "{}"),
            ],
        )
    )
    fake = FakeCompletion([double_submit, submit_response()])
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path, time_budget_sec=7200)
    await agent.run("Do a task", FakeEnv(), AgentContext())  # type: ignore[arg-type]

    assert len(fake.calls) == 2
    messages = read_messages(agent)
    tool_contents = [m["content"] for m in messages if m.get("role") == "tool"]
    assert sum("confirm this as your final submission" in c for c in tool_contents) == 2
    assert sum("Final submission confirmed" in c for c in tool_contents) == 1


async def test_confirm_flow_without_time_budget(tmp_path, monkeypatch):
    # Without a budget the flow is identical, minus the time-left sentence.
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    await agent.run("Do a task", FakeEnv(), AgentContext())  # type: ignore[arg-type]
    assert len(fake.calls) == 2
    messages = read_messages(agent)
    tool_contents = [m["content"] for m in messages if m.get("role") == "tool"]
    assert "confirm this as your final submission" in tool_contents[0]
    assert "time budget remains" not in tool_contents[0]
    assert "Final submission confirmed" in tool_contents[1]


async def test_submit_empty_arguments_string_is_valid(tmp_path, monkeypatch):
    # Providers commonly send "" instead of "{}" for zero-parameter tools.
    empty_args = FakeResponse(
        FakeMessage(
            content="Done.",
            tool_calls=[FakeToolCall("call_f", "submit", "")],
        )
    )
    fake = FakeCompletion([empty_args, submit_response()])
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    context = AgentContext()
    await agent.run("Do a task", FakeEnv(), context)  # type: ignore[arg-type]

    assert context.metadata is not None
    # The empty-args submit recorded; the second confirmed by reuse.
    assert context.metadata["n_submissions"] == 1


async def test_plain_submit_records_and_continues(tmp_path, monkeypatch):
    fake = FakeCompletion(
        [
            submit_response(),  # records; bash next turn resets the gate
            bash_response("echo more"),
            *finish_responses(),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)  # no budget: same confirm flow
    env = FakeEnv()
    context = AgentContext()
    await agent.run("Do a task", env, context)  # type: ignore[arg-type]

    assert len(fake.calls) == 4
    messages = read_messages(agent)
    tool_contents = [m["content"] for m in messages if m.get("role") == "tool"]
    assert any("Submission #1 recorded" in c for c in tool_contents)
    assert any("Submission #2 recorded" in c for c in tool_contents)
    assert any("Final submission confirmed" in c for c in tool_contents)
    # Two record calls (the confirm resolves the same commit and reuses,
    # recording nothing), then exactly one command after the loop: the
    # end-of-run restore checks out the winning (newest pinned) commit so
    # the verifier and the snapshot score the submitted tree, keeping the
    # pre-restore HEAD reachable via refs/proximus/end-state.
    records = [
        c
        for c in env.all_commands
        if "rev-parse -q --verify" in c and 'echo "matched"' not in c
    ]
    assert len(records) == 2
    restore = env.all_commands[-1]
    assert "refs/proximus/end-state" in restore
    assert env.all_envs[-1]["winner"] == fake_hash(2)
    assert context.metadata is not None
    assert context.metadata["n_submissions"] == 2
    assert context.metadata["submitted_commit"] == fake_hash(2)


async def test_submit_ignores_stray_arguments(tmp_path, monkeypatch):
    # Models trained on the old schema still pass summary/end_run; a
    # submission must never fail on harmless extras — and end_run has no
    # effect (the confirm flow is the only way to end).
    fake = FakeCompletion(
        [
            submit_response(summary="all done", end_run=True, reason="done"),
            bash_response("echo more"),
            *finish_responses(),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    context = AgentContext()
    await agent.run("Do a task", FakeEnv(), context)  # type: ignore[arg-type]

    # The stray end_run=True did not end the run: the bash turn still ran.
    assert len(fake.calls) == 4
    messages = read_messages(agent)
    tool_contents = [m["content"] for m in messages if m.get("role") == "tool"]
    assert "Submission #1 recorded" in tool_contents[0]
    assert not any("Invalid arguments" in c for c in tool_contents)
    assert context.metadata is not None
    # Two recordings; the final confirm reused the second one.
    assert context.metadata["n_submissions"] == 2


async def test_failed_record_falls_back_to_final_state(tmp_path, monkeypatch):
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    class BrokenGitEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            self.record_calls: list[str] = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
            if "rev-parse -q --verify" in command:
                self.all_commands.append(command)
                self.record_calls.append(command)
                return ExecResult(stdout=None, stderr="disk full", return_code=1)
            return await super().exec(
                command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user
            )

    env = BrokenGitEnv()
    context = AgentContext()
    agent = make_agent(tmp_path)
    await agent.run("Do a task", env, context)  # type: ignore[arg-type]

    messages = read_messages(agent)
    tool_contents = [m["content"] for m in messages if m.get("role") == "tool"]
    assert any("could not be recorded" in c for c in tool_contents)
    assert any("final workspace state" in c for c in tool_contents)
    # The consecutive submit still ends the session despite failed recordings
    # (a broken-git sandbox must not trap the run).
    assert any("Final submission confirmed" in c for c in tool_contents)
    assert context.metadata is not None
    assert context.metadata["n_submissions"] == 0
    # Failed model submit calls remain recorded for telemetry. A confirmed
    # ending never adds a hidden auto-submission or changes the selected state.
    assert [s["ok"] for s in context.metadata["submissions"]] == [False, False]
    assert all(not s.get("auto") for s in context.metadata["submissions"])
    assert context.metadata["submitted_commit"] is None
    assert len(env.record_calls) == 2


async def test_proactive_compaction(tmp_path, monkeypatch):
    # Turn 1 reports prompt usage over threshold (0.85 * 1000); before turn 2
    # the agent compacts (summary call), then continues and finishes.
    fake = FakeCompletion(
        [
            bash_response("ls", prompt_tokens=900),
            text_response("handoff summary of the work"),
            *finish_responses(),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    small_context(monkeypatch)
    agent = make_agent(tmp_path)
    await agent.run("Big task", FakeEnv(), AgentContext())  # type: ignore[arg-type]

    # The summary request must be shaped EXACTLY like a normal agent turn —
    # same tools, no tool_choice. Anthropic invalidates the whole messages
    # cache when tool_choice changes, so the old tool_choice="none" made
    # every compaction re-read the full trail at the uncached input rate.
    assert "tool_choice" not in fake.calls[1]
    assert fake.calls[1].get("tools") == fake.calls[0].get("tools")

    trajectory = read_trajectory(agent)
    compaction_steps = [
        step
        for step in trajectory["steps"]
        if step.get("extra", {}).get("context_management", {}).get("type")
        == "compaction"
    ]
    assert len(compaction_steps) == 1

    messages = read_messages(agent)
    # History was replaced: system prompt + handoff + verbatim recent tail.
    assert messages[0]["role"] == "system"
    assert "[Context compacted]" in messages[1]["content"]
    assert "handoff summary of the work" in messages[1]["content"]
    # The most recent pre-compaction turn survives verbatim after the handoff.
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"][0]["function"]["name"] == "bash"
    assert messages[3]["role"] == "tool"


async def test_summary_retries_once_when_model_calls_tools(tmp_path, monkeypatch):
    # The summary prompt says "Do not call any tools."; without tool_choice
    # enforcement a model may call one anyway (no usable text) — the summary
    # call is retried once instead of silently losing the handoff.
    tool_call_only = FakeResponse(
        FakeMessage(
            content="",
            tool_calls=[FakeToolCall("call_x", "bash", json.dumps({"command": "ls"}))],
        )
    )
    fake = FakeCompletion(
        [
            bash_response("ls", prompt_tokens=900),
            tool_call_only,  # summary attempt 1: ignored the instruction
            text_response("recovered handoff summary"),  # attempt 2
            *finish_responses(),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    small_context(monkeypatch)
    agent = make_agent(tmp_path)
    env = FakeEnv()
    await agent.run("Big task", env, AgentContext())  # type: ignore[arg-type]

    assert len(fake.calls) == 5
    # The stray tool call from the summary attempt is never executed.
    assert env.exec_calls == ["ls"]
    messages = read_messages(agent)
    assert "recovered handoff summary" in messages[1]["content"]


async def test_time_budget_exhausted_ends_run_without_querying(tmp_path, monkeypatch):
    # The budget check sits at the top of the turn: an exhausted budget ends
    # the run before any LLM call is made (FakeCompletion would raise if hit).
    fake = FakeCompletion([])
    monkeypatch.setattr(litellm, "acompletion", fake)
    monkeypatch.setattr(Proximus, "_elapsed", lambda self: 4000.0)

    agent = make_agent(tmp_path, time_budget_sec=3600)
    context = AgentContext()
    await agent.run("Task", FakeEnv(), context)  # type: ignore[arg-type]

    assert fake.calls == []
    assert context.metadata is not None
    assert context.metadata["n_turns"] == 1


async def test_context_overflow_compacts_and_retries_the_turn(tmp_path, monkeypatch):
    # Turn 1's query overflows the context window: the loop compacts
    # (summary call) and then RETRIES the turn instead of discarding it.
    overflow = litellm.ContextWindowExceededError(
        "prompt too long", model="gpt-test", llm_provider="openai"
    )
    fake = FakeCompletion(
        [
            overflow,
            text_response("handoff summary after overflow"),
            bash_response("echo hi"),
            *finish_responses(),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    env = FakeEnv()
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    # The overflowed call, the summary call, then the three normal turns.
    assert len(fake.calls) == 5
    # The retried turn's work happened; nothing was silently dropped.
    assert env.exec_calls == ["echo hi"]
    messages = read_messages(agent)
    assert "handoff summary after overflow" in messages[1]["content"]


async def test_output_length_exceeded_asks_for_smaller_output(tmp_path, monkeypatch):
    # finish_reason="length" surfaces as OutputLengthExceededError: the turn
    # is discarded and the model is asked to re-issue in smaller pieces.
    truncated = text_response("way too long, cut off mid-")
    truncated.choices[0].finish_reason = "length"
    fake = FakeCompletion([truncated, bash_response("echo hi"), *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    env = FakeEnv()
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    assert len(fake.calls) == 4
    assert env.exec_calls == ["echo hi"]
    messages = read_messages(agent)
    nudges = [
        m
        for m in messages
        if m.get("role") == "user" and m.get("content") == OUTPUT_LENGTH_MESSAGE
    ]
    assert len(nudges) == 1
    # The truncated assistant turn itself is not kept in the conversation.
    assert not any("cut off mid-" in str(m.get("content")) for m in messages)


async def test_text_only_turns_are_neither_nagged_nor_terminal(tmp_path, monkeypatch):
    # Prose is legitimate work: a text-only turn is recorded and the loop
    # continues, with no injected "you did not call any tool" nudge: such a
    # nudge suppresses natural-language output entirely, pushing reasoning
    # into comment-only scratch files.
    # Each prose turn IS answered with a neutral user message, though:
    # Anthropic rejects requests whose conversation ends assistant-side.
    responses = [text_response(f"thinking {n}...") for n in range(7)]
    responses.append(submit_response())
    responses.append(submit_response())
    fake = FakeCompletion(responses)
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    context = AgentContext()
    await agent.run("Task", FakeEnv(), context)  # type: ignore[arg-type]

    # All seven prose turns ran (the old cap ended the run at five), then the
    # submit pair confirmed and ended it.
    assert len(fake.calls) == 9
    assert context.metadata is not None
    assert context.metadata["stalled"] is False
    messages = read_messages(agent)
    assert not [
        m
        for m in messages
        if m.get("role") == "user" and "did not call any tool" in str(m.get("content"))
    ]
    continuations = [
        m
        for m in messages
        if m.get("role") == "user" and m.get("content") == CONTINUE_MESSAGE
    ]
    assert len(continuations) == 7
    # No request may end with an assistant message.
    for call in fake.calls:
        assert call["messages"][-1]["role"] in ("user", "tool")


# ----- empty-turn stall breaker -----
# Disjoint from the prose-turn behavior above: a turn with text (or
# reasoning) is work; a turn with NOTHING is a stuck model. A model that
# treats its own wrap-up as the end of the session otherwise spins empty
# completions until the budget or the orchestrator kill (hundreds of turns on
# a run without a budget).


def empty_response() -> FakeResponse:
    return FakeResponse(FakeMessage(content=None))


async def test_consecutive_empty_turns_end_the_run(tmp_path, monkeypatch):
    from harbor_proximus.proximus import EMPTY_TURN_STALL_LIMIT

    # Exactly the limit is queued: FakeCompletion would raise on one more.
    fake = FakeCompletion([empty_response() for _ in range(EMPTY_TURN_STALL_LIMIT)])
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    context = AgentContext()
    await agent.run("Task", FakeEnv(), context)  # type: ignore[arg-type]

    assert len(fake.calls) == EMPTY_TURN_STALL_LIMIT
    assert context.metadata is not None
    assert context.metadata["stalled"] is True
    assert context.metadata["n_turns"] == EMPTY_TURN_STALL_LIMIT


async def test_any_output_resets_the_stall_counter(tmp_path, monkeypatch):
    # Two empties, then prose; two empties, then a thinking-only turn: each
    # resets the count, so only the final unbroken run of empties ends it.
    thinking_only = FakeResponse(
        FakeMessage(content=None, reasoning_content="reflecting...")
    )
    fake = FakeCompletion(
        [
            empty_response(),
            empty_response(),
            text_response("let me reconsider the approach"),
            empty_response(),
            empty_response(),
            thinking_only,
            *[empty_response() for _ in range(3)],
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    context = AgentContext()
    await agent.run("Task", FakeEnv(), context)  # type: ignore[arg-type]

    assert len(fake.calls) == 9
    assert context.metadata is not None
    assert context.metadata["stalled"] is True


async def test_empty_turn_gets_the_directive_closer(tmp_path, monkeypatch):
    # A bare "Continue." does not break a mute loop (a model can keep
    # returning 200s with nothing in them until the stall guard ends the run),
    # so an empty turn is answered with the directive closer instead. Prose turns keep the neutral one.
    fake = FakeCompletion(
        [
            empty_response(),
            text_response("still working through the approach"),
            *[empty_response() for _ in range(3)],
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    context = AgentContext()
    await agent.run("Task", FakeEnv(), context)  # type: ignore[arg-type]

    assert context.metadata is not None
    assert context.metadata["stalled"] is True
    closers = [
        m.get("content") for m in read_messages(agent) if m.get("role") == "user"
    ]
    # The prose turn keeps "Continue."; every empty one is nagged.
    assert closers.count(CONTINUE_MESSAGE) == 1
    assert closers.count(EMPTY_RESPONSE_MESSAGE) == 3


async def test_opaque_thinking_blocks_do_not_count_as_empty(tmp_path, monkeypatch):
    # Anthropic can return thinking blocks with no readable text (redacted
    # thinking): a model still thinking is not provably stuck.
    blocks_only = FakeResponse(FakeMessage(content=None))
    blocks_only.choices[0].message.thinking_blocks = [
        {"type": "redacted_thinking", "data": "opaque"}
    ]
    fake = FakeCompletion(
        [
            empty_response(),
            empty_response(),
            blocks_only,
            *[empty_response() for _ in range(3)],
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    context = AgentContext()
    await agent.run("Task", FakeEnv(), context)  # type: ignore[arg-type]

    assert len(fake.calls) == 6
    assert context.metadata is not None
    assert context.metadata["stalled"] is True


async def test_invalid_tool_arguments_reported(tmp_path, monkeypatch):
    bad_call = FakeResponse(
        FakeMessage(
            content="",
            tool_calls=[FakeToolCall("call_x", "bash", "not valid json")],
        )
    )
    fake = FakeCompletion([bad_call, *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    env = FakeEnv()
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    assert env.exec_calls == []  # invalid args never reach the executor
    messages = read_messages(agent)
    tool_contents = [m["content"] for m in messages if m.get("role") == "tool"]
    assert any("Invalid arguments for 'bash'" in c for c in tool_contents)


def test_system_prompt_teaches_commit_then_submit_and_renders_placeholders(tmp_path):
    agent = make_agent(tmp_path)
    assert "multiple times" not in agent._system_prompt
    # The two-submit ending and the revisability of submissions are discovered AT
    # submit time (continue-or-confirm nudge, disarm note), never announced up front.
    assert "submit` again" not in agent._system_prompt
    assert "twice" not in agent._system_prompt
    assert "confirm" not in agent._system_prompt
    assert "record it with `submit`" in agent._system_prompt
    assert "record it with `submit`" in agent._system_prompt
    assert str(PROGRESS_FILE) in agent._system_prompt
    assert "{{SUBMIT_GUIDANCE}}" not in agent._system_prompt
    assert "{{TIMER_GUIDANCE}}" not in agent._system_prompt
    assert "{{PROGRESS_FILE}}" not in agent._system_prompt


def test_default_tool_surface_for_non_vision_models(tmp_path, monkeypatch):
    # Work happens through bash for tool-surface neutrality (matches
    # mini-swe-agent / terminus); list_submissions is read-only bookkeeping
    # and compaction is automatic (no model-facing tool).
    monkeypatch.setattr(litellm, "supports_vision", lambda model: False)
    agent = make_agent(tmp_path)
    names = [s["function"]["name"] for s in agent._tool_schemas]
    assert names == ["bash", "list_submissions", "submit"]


def test_vision_models_get_view_image(tmp_path, monkeypatch):
    monkeypatch.setattr(litellm, "supports_vision", lambda model: True)
    agent = make_agent(tmp_path)
    names = [s["function"]["name"] for s in agent._tool_schemas]
    assert names == ["bash", "view_image", "list_submissions", "submit"]


def test_non_vision_models_lack_view_image(tmp_path, monkeypatch):
    monkeypatch.setattr(litellm, "supports_vision", lambda model: False)
    agent = make_agent(tmp_path)
    names = [s["function"]["name"] for s in agent._tool_schemas]
    assert "view_image" not in names


def test_vision_probe_failure_degrades_to_no_view_image(tmp_path, monkeypatch):
    # A registry hiccup must not kill agent construction; the model is
    # treated as blind and the run proceeds on bash+submit.
    def supports_vision(model):
        raise ValueError(f"model not in registry: {model}")

    monkeypatch.setattr(litellm, "supports_vision", supports_vision)
    agent = make_agent(tmp_path)
    names = [s["function"]["name"] for s in agent._tool_schemas]
    assert names == ["bash", "list_submissions", "submit"]


async def test_view_image_attaches_user_image_message(tmp_path, monkeypatch):
    import base64 as b64mod

    view_call = FakeResponse(
        FakeMessage(
            content="Looking at the plot.",
            tool_calls=[
                FakeToolCall(
                    "call_v", "view_image", json.dumps({"path": "/app/plot.png"})
                )
            ],
        )
    )
    fake = FakeCompletion([view_call, *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)
    monkeypatch.setattr(litellm, "supports_vision", lambda model: True)

    png = make_tiny_png()

    class ImageEnv(FakeEnv):
        async def download_file(self, source_path, target_path):
            from pathlib import Path

            Path(target_path).write_bytes(png)

    agent = make_agent(tmp_path)
    await agent.run("Check the plot", ImageEnv(), AgentContext())  # type: ignore[arg-type]

    messages = read_messages(agent)
    image_messages = [
        m
        for m in messages
        if m.get("role") == "user"
        and isinstance(m.get("content"), list)
        and m["content"][0].get("type") == "image_url"
    ]
    assert len(image_messages) == 1
    url = image_messages[0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert b64mod.b64decode(url.split(",", 1)[1]) == png

    # The trajectory records only the text acknowledgment, not the payload.
    trajectory = read_trajectory(agent)
    observations = [
        r["content"]
        for s in trajectory["steps"]
        for r in ((s.get("observation") or {}).get("results") or [])
    ]
    assert any("Attached image /app/plot.png" in c for c in observations)
    assert not any("base64," in c for c in observations)


async def test_image_does_not_split_tool_result_block(tmp_path, monkeypatch):
    # Anthropic requires all tool_results contiguously after the assistant
    # turn; a view_image in a multi-call batch must attach its image AFTER
    # the full tool-result block (a live trial 400'd on this).
    batch = FakeResponse(
        FakeMessage(
            content="Look and list.",
            tool_calls=[
                FakeToolCall(
                    "call_v", "view_image", json.dumps({"path": "/app/a.png"})
                ),
                FakeToolCall("call_b", "bash", json.dumps({"command": "ls"})),
            ],
        )
    )
    fake = FakeCompletion([batch, *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)
    monkeypatch.setattr(litellm, "supports_vision", lambda model: True)

    png = make_tiny_png()

    class ImageEnv(FakeEnv):
        async def download_file(self, source_path, target_path):
            from pathlib import Path

            Path(target_path).write_bytes(png)

    agent = make_agent(tmp_path)
    await agent.run("Task", ImageEnv(), AgentContext())  # type: ignore[arg-type]

    messages = read_messages(agent)
    # Find the batch assistant turn; the two tool results must be adjacent,
    # then the image user message.
    idx = next(
        i
        for i, m in enumerate(messages)
        if m.get("role") == "assistant" and (m.get("tool_calls") or [])
    )
    assert messages[idx + 1]["role"] == "tool"
    assert messages[idx + 2]["role"] == "tool"
    assert messages[idx + 3]["role"] == "user"
    assert isinstance(messages[idx + 3]["content"], list)
    assert messages[idx + 3]["content"][0]["type"] == "image_url"


def make_tiny_png() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (0, 120, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


async def test_disabled_tool_call_returns_error(tmp_path, monkeypatch):
    # view_image exists as a handler but is not enabled for non-vision
    # models; calling it must yield the not-available error, not execution.
    view_call = FakeResponse(
        FakeMessage(
            content="",
            tool_calls=[
                FakeToolCall("call_v", "view_image", json.dumps({"path": "/a.png"}))
            ],
        )
    )
    fake = FakeCompletion([view_call, *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)
    monkeypatch.setattr(litellm, "supports_vision", lambda model: False)

    agent = make_agent(tmp_path)
    env = FakeEnv()
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    assert env.exec_calls == []
    messages = read_messages(agent)
    tool_contents = [m["content"] for m in messages if m.get("role") == "tool"]
    assert any("not available in this session" in c for c in tool_contents)


async def test_no_network_note_in_initial_message(tmp_path, monkeypatch):
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    env = FakeEnv()
    env.network_policy = NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
    agent = make_agent(tmp_path)
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    messages = read_messages(agent)
    assert "no internet access" in messages[1]["content"]


async def test_public_network_gets_no_note(tmp_path, monkeypatch):
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)  # FakeEnv's NetworkPolicy defaults to public
    await agent.run("Task", FakeEnv(), AgentContext())  # type: ignore[arg-type]

    messages = read_messages(agent)
    assert "internet" not in messages[1]["content"].lower()


async def test_time_budget_ends_run_mid_stream(tmp_path, monkeypatch):
    # There is no turn cap: a run that never submits ends only when the
    # per-turn budget check trips. Two turns fit inside the budget; the
    # third does not, so exactly two LLM calls are made.
    fake = FakeCompletion([bash_response("echo 1"), bash_response("echo 2")])
    monkeypatch.setattr(litellm, "acompletion", fake)
    monkeypatch.setattr(Proximus, "_elapsed", lambda self: self._run.n_turns * 1500.0)

    agent = make_agent(tmp_path, time_budget_sec=3600)
    context = AgentContext()
    await agent.run("Task", FakeEnv(), context)  # type: ignore[arg-type]

    assert len(fake.calls) == 2
    assert context.metadata is not None
    assert context.metadata["n_turns"] == 3  # the third turn stopped at the gate


# ----- in-sandbox task timer -----


def timer_commands(env: FakeEnv) -> list[str]:
    return [c for c in env.all_commands if "/app/.timer" in c]


async def test_task_timer_reanchored_to_budget(tmp_path, monkeypatch):
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path, time_budget_sec=600)
    env = FakeEnv()
    context = AgentContext()
    await agent.run("Task", env, context)  # type: ignore[arg-type]

    relaunch = timer_commands(env)[0]
    # Seeded from the REMAINING budget (measured at arm time, ~the full 600s
    # here), not the run's hard timeout; the sandbox-timer anchor rewrite is
    # tried first, and the legacy /app/timer.sh restart is what changes the
    # .timer file set.
    assert "remaining=" in relaunch
    assert 'TASK_BUDGET_SECS="$remaining"' in relaunch
    assert "sandbox-timer" in relaunch
    assert "setsid /app/timer.sh" in relaunch


async def test_task_timer_never_stopped(tmp_path, monkeypatch):
    # The daemon is left running: it dies with the sandbox at teardown, and
    # ensure_git keeps .timer out of the repository so it cannot leak into a
    # submission. The single /app/.timer command is the re-anchor itself.
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path, time_budget_sec=600)
    env = FakeEnv()
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    assert len(timer_commands(env)) == 1
    assert "timer.sh" in timer_commands(env)[0]


async def test_task_timer_untouched_without_budget(tmp_path, monkeypatch):
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    env = FakeEnv()
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    # No budget means no truth to re-seed with; leave the task's clock alone.
    assert timer_commands(env) == []


async def test_task_timer_missing_script_is_not_an_error(tmp_path, monkeypatch):
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path, time_budget_sec=600)
    env = FakeEnv(has_timer=False)
    context = AgentContext()
    await agent.run("Task", env, context)  # type: ignore[arg-type]

    # Timer-less tasks are normal; the run proceeds and nothing is stopped.
    assert context.metadata is not None
    assert context.metadata["n_turns"] == 2
    assert len(timer_commands(env)) == 1


async def test_task_timer_reanchor_failure_does_not_fail_the_run(tmp_path, monkeypatch):
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    class BrokenTimerEnv(FakeEnv):
        async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
            if "/app/.timer" in command:
                raise RuntimeError("exec channel died")
            return await super().exec(command, cwd, env, timeout_sec, user)

    agent = make_agent(tmp_path, time_budget_sec=600)
    context = AgentContext()
    await agent.run("Task", BrokenTimerEnv(), context)  # type: ignore[arg-type]

    assert context.metadata is not None
    assert context.metadata["n_turns"] == 2


# ----- harness-owned deadline file -----
# Written once in EVERY sandbox with a budget, so a task image that ships no
# timer of its own still gives the model a clock it can read on demand. A
# static file has nothing to keep fresh and nothing to stop at run end.


def deadline_commands(env: FakeEnv) -> list[str]:
    return [c for c in env.all_commands if DEADLINE_PATH in c]


async def test_deadline_written_from_the_budget(tmp_path, monkeypatch):
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path, time_budget_sec=600)
    env = FakeEnv()
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    writes = deadline_commands(env)
    assert len(writes) == 1
    # Anchored to the sandbox's own clock — the model compares against
    # in-sandbox `date`, so the harness clock must not be the anchor.
    match = re.search(r"date \+%s\) \+ (\d+) \)", writes[0])
    assert match is not None
    # Seeded from the harness budget less the setup already spent, so it never
    # overstates the time left the way a boot- or timeout-anchored clock does.
    assert 590 <= int(match.group(1)) <= 600


async def test_write_deadline_seeds_the_exact_remaining_budget():
    env = FakeEnv()
    await write_deadline(env, 36000, logging.getLogger("test"))  # type: ignore[arg-type]
    # The budget the harness paces on, never the larger agent timeout that the
    # runner sets (36000 vs 37800 would gift the model half an hour).
    assert "+ 36000" in env.all_commands[0]
    assert "37800" not in env.all_commands[0]


@pytest.mark.parametrize("remaining_sec", [0, -5])
async def test_deadline_skipped_when_the_budget_is_already_spent(remaining_sec):
    env = FakeEnv()
    await write_deadline(env, remaining_sec, logging.getLogger("test"))  # type: ignore[arg-type]
    # Nothing written rather than a deadline already in the past.
    assert env.all_commands == []


async def test_deadline_written_when_task_ships_no_timer(tmp_path, monkeypatch):
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path, time_budget_sec=600)
    env = FakeEnv(has_timer=False)
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    # The whole point: the re-anchor is a no-op here, the deadline is not.
    assert len(deadline_commands(env)) == 1


async def test_deadline_skipped_without_a_budget(tmp_path, monkeypatch):
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    env = FakeEnv()
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    # No budget means there is no deadline to write.
    assert deadline_commands(env) == []


async def test_deadline_write_failure_does_not_fail_the_run(tmp_path, monkeypatch):
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    class BrokenDeadlineEnv(FakeEnv):
        async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
            if DEADLINE_PATH in command:
                raise RuntimeError("exec channel died")
            return await super().exec(command, cwd, env, timeout_sec, user)

    agent = make_agent(tmp_path, time_budget_sec=600)
    context = AgentContext()
    await agent.run("Task", BrokenDeadlineEnv(), context)  # type: ignore[arg-type]

    assert context.metadata is not None
    assert context.metadata["n_turns"] == 2
    assert context.metadata["submitted_commit"] is not None


def test_system_prompt_advertises_deadline_outside_submit_guidance(tmp_path):
    prompt = make_agent(tmp_path)._system_prompt

    lines = [line for line in prompt.splitlines() if DEADLINE_PATH in line]
    assert len(lines) == 1
    assert "seconds" in lines[0]
    assert "submit" not in lines[0].lower()
    assert DEADLINE_PATH not in SUBMIT_GUIDANCE


async def test_only_the_harness_deadline_is_advertised(tmp_path, monkeypatch):
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path, time_budget_sec=600)
    env = FakeEnv()
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    # One canonical path: the prompt and the setup write cannot drift.
    assert DEADLINE_PATH in agent._system_prompt
    assert DEADLINE_PATH in deadline_commands(env)[0]
    # The task's own timer is deliberately left unadvertised here — half the
    # images ship none, and the prompt is identical for all of them. Where it
    # does exist the re-anchor keeps it agreeing, and the task instruction
    # names it; the harness never touches /app/.timer's file set.
    assert "/app/.timer" not in agent._system_prompt


# ----- task-provided timer -----
# Tasks ship their own timer and teach it in the instruction.
# task_provides_timer=True makes that clock the ONLY model-facing one: the
# harness writes no deadline file and the system prompt points at the task's
# timer — one timer story, not two. The RE-ANCHOR still runs: untouched, the
# task's countdown measures the sandbox's hard timeout (budget + verifier +
# pad), and the instruction's clock must mean "seconds until this run ends".


async def test_task_provided_timer_skips_deadline_but_reanchors(tmp_path, monkeypatch):
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path, time_budget_sec=600, task_provides_timer=True)
    env = FakeEnv()
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    # No harness clock for the model — but the task's own clock is reseeded
    # to the budget so the instruction's timer tells the truth.
    assert deadline_commands(env) == []
    assert timer_commands(env) != []


def test_task_provided_timer_swaps_the_prompt_line(tmp_path):
    prompt = make_agent(tmp_path, task_provides_timer=True)._system_prompt

    assert DEADLINE_PATH not in prompt
    assert TASK_TIMER_GUIDANCE.lstrip("- ") in prompt
    # The default prompt has the deadline line and no task-timer pointer.
    default_prompt = make_agent(tmp_path)._system_prompt
    assert DEADLINE_PATH in default_prompt
    assert TASK_TIMER_GUIDANCE.lstrip("- ") not in default_prompt


async def test_task_provided_timer_keeps_budget_pacing(tmp_path, monkeypatch):
    # The flag scopes to in-sandbox clocks only: the budget still opens the
    # initial message and still times the submit guidance.
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path, time_budget_sec=7200, task_provides_timer=True)
    await agent.run("Do a task", FakeEnv(), AgentContext())  # type: ignore[arg-type]

    messages = read_messages(agent)
    user_contents = [m["content"] for m in messages if m.get("role") == "user"]
    assert "time budget of 2h" in user_contents[0]
    tool_contents = [m["content"] for m in messages if m.get("role") == "tool"]
    assert "time budget remains (100% left)" in tool_contents[0]


# ---------------------------------------------------------------------------
# PROGRESS.md re-injection
# ---------------------------------------------------------------------------

_PROGRESS_NOTES = "NOTES: wire format decoded; startup packet still failing."


class ProgressEnv(FakeEnv):
    """Env whose PROGRESS.md read returns known notes."""

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        result = await super().exec(
            command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user
        )
        if "PROGRESS.md" in command:
            return ExecResult(stdout=_PROGRESS_NOTES, stderr=None, return_code=0)
        return result


async def test_progress_file_reinjected_into_handoff(tmp_path, monkeypatch):
    fake = FakeCompletion(
        [
            bash_response("ls", prompt_tokens=900),
            text_response("handoff summary of the work"),
            *finish_responses(),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    small_context(monkeypatch)
    agent = make_agent(tmp_path)
    env = ProgressEnv()
    await agent.run("Big task", env, AgentContext())  # type: ignore[arg-type]

    handoff = read_messages(agent)[1]["content"]
    assert "[Context compacted]" in handoff
    assert _PROGRESS_NOTES in handoff
    assert any("PROGRESS.md" in c for c in env.all_commands)


async def test_dead_sandbox_ends_the_run(tmp_path, monkeypatch):
    """A sandbox that dies MID-RUN must end the run, not be retried forever.

    A dead sandbox can arrive as ordinary error tool results, leaving the loop
    no reason to stop: a run could spend thousands of turns of tokens against
    a sandbox that died in its first minute. Environments classify their
    SDK's sandbox-gone errors as EnvironmentGoneError, ToolExecutor lets it
    propagate, and the loop must fail the trial on the FIRST one after
    preserving the terminal trajectory step. A sandbox already dead at setup
    also propagates; see test_dead_sandbox_at_setup_fails_the_trial.
    """
    fake = FakeCompletion([bash_response(f"echo {n}") for n in range(40)])
    monkeypatch.setattr(litellm, "acompletion", fake)

    class DiesOnFirstToolEnv(FakeEnv):
        """Setup succeeds; the model's first bash call finds the sandbox gone."""

        async def exec(self, command, **kw):
            if command.startswith("echo "):
                raise EnvironmentGoneError(
                    "Modal sandbox is gone: NotFoundError: Modal Sandbox with "
                    "container ID ta-xyz not found. This means this Sandbox "
                    "has already shut down."
                )
            return await super().exec(command, **kw)

    agent = make_agent(tmp_path)
    context = AgentContext()
    with pytest.raises(EnvironmentGoneError):
        await agent.run("Task", DiesOnFirstToolEnv(), context)  # type: ignore[arg-type]

    assert context.metadata is not None
    # Stopped on the first typed error rather than running out the 40 queued
    # responses.
    assert context.metadata["n_turns"] == 1, context.metadata["n_turns"]
    # And it is distinguishable from a voluntary early finish.
    assert context.metadata["env_lost"] is True


async def test_dead_sandbox_at_setup_fails_the_trial(tmp_path, monkeypatch):
    """Dead before the loop ever starts: setup is fail-fast, so the typed
    error propagates out of run() and the trial errors immediately with the
    real cause — no budget burned toward a run that can never submit."""
    fake = FakeCompletion([bash_response("echo 0")])
    monkeypatch.setattr(litellm, "acompletion", fake)

    class DeadEnv(FakeEnv):
        async def exec(self, command, **kw):
            raise EnvironmentGoneError("Modal sandbox is gone")

    agent = make_agent(tmp_path)
    with pytest.raises(EnvironmentGoneError):
        await agent.run("Task", DeadEnv(), AgentContext())  # type: ignore[arg-type]


async def test_transient_tool_errors_do_not_end_the_run(tmp_path, monkeypatch):
    """The breaker must fire on a DEAD SANDBOX, not on a failing command.

    A command that exits nonzero, or a one-off exec error, is normal and the
    agent should keep working; only the unrecoverable-sandbox signature counts,
    and the counter resets on any non-matching result.
    """
    fake = FakeCompletion(
        [bash_response("false"), *([bash_response("echo ok")] * 3), *finish_responses()]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    class FlakyEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            self.n = 0

        async def exec(self, command, **kw):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("transient: connection reset")
            return await super().exec(command, **kw)

    agent = make_agent(tmp_path)
    context = AgentContext()
    await agent.run("Task", FlakyEnv(), context)  # type: ignore[arg-type]

    assert context.metadata is not None
    assert context.metadata["env_lost"] is False
    assert context.metadata["n_turns"] >= 4


def test_cache_breakpoints_use_the_1h_ttl(tmp_path):
    """Breakpoints carry ttl=1h, on the system prompt and the latest message.

    ttl="1h" bills cache writes at a premium over the 5-minute default and
    needs no beta header. The premium
    is paid on the small per-turn increment while an expiry re-reads the
    whole prefix at full price, so one avoided miss covers ~70 turns of it.

    The second breakpoint lands on the trailing *tool* message here, not the
    assistant turn before it -- a later anchor caches strictly more. A tool
    message hosts it as a top-level ``cache_control`` key (the one litellm
    reads) rather than inside ``content``, so this counts both shapes.
    """
    agent = Proximus(logs_dir=tmp_path / "logs", model_name="anthropic/claude-opus-5")
    wire = agent._llm._apply_prompt_caching(  # type: ignore[attr-defined]
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "working"},
            {"role": "tool", "tool_call_id": "c1", "content": "out"},
        ]
    )
    tagged = [
        blk["cache_control"]
        for m in wire
        if isinstance(m.get("content"), list)
        for blk in m["content"]
        if isinstance(blk, dict) and "cache_control" in blk
    ] + [m["cache_control"] for m in wire if m.get("cache_control") is not None]
    assert len(tagged) == 2, "expected system + latest-message breakpoints"
    for cache_control in tagged:
        assert cache_control == {"type": "ephemeral", "ttl": "1h"}
    assert wire[-1]["role"] == "tool", "the anchor must be the newest message"
    assert wire[-1]["cache_control"] is not None


def test_non_anthropic_models_are_not_tagged(tmp_path):
    """OpenAI caches automatically and exposes no TTL control -- litellm lists
    no cache_creation_input_token_cost_above_1hr for it, and its automatic
    caching already reaches high hit rates untouched."""
    agent = Proximus(logs_dir=tmp_path / "logs", model_name="openai/gpt-test")
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    assert agent._llm._apply_prompt_caching(msgs) is msgs  # type: ignore[attr-defined]


async def test_openai_reasoning_items_round_trip(tmp_path, monkeypatch):
    """The OpenAI equivalent of Anthropic thinking-block replay.

    litellm bridges gpt-5.4+ tool calls to the Responses API and hands back
    opaque reasoning items; unless they are replayed the model loses its own
    chain of thought every turn.
    """
    items = [{"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC1"}]
    first = bash_response("echo hi")
    first.choices[0].message.reasoning_items = items
    fake = FakeCompletion([first, *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_bridged_agent(tmp_path)
    await agent.run("Task", FakeEnv(), AgentContext())  # type: ignore[arg-type]

    # Stored on the assistant turn...
    replayed = [
        m
        for m in read_messages(agent)
        if m.get("role") == "assistant" and m.get("reasoning_items")
    ]
    assert replayed and replayed[0]["reasoning_items"] == items
    # ...and sent back on the following request.
    assert any(
        any(m.get("reasoning_items") == items for m in call["messages"])
        for call in fake.calls[1:]
    )
    # The request asks for the encrypted blobs; store=False is what makes
    # OpenAI attach them (full param set asserted in
    # test_bridged_request_carries_responses_params).
    body = fake.calls[0]["extra_body"]
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["store"] is False


async def test_reasoning_only_turn_replays_encrypted_cot(tmp_path, monkeypatch):
    """A Responses turn with encrypted CoT and no message/tool must still
    replay the reasoning items; dropping them loses the model's CoT after a
    reasoning-only output."""
    items = [{"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC1"}]
    first = FakeResponse(FakeMessage(content=None))
    first.choices[0].message.reasoning_items = items
    fake = FakeCompletion([first, *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_bridged_agent(tmp_path)
    await agent.run("Task", FakeEnv(), AgentContext())  # type: ignore[arg-type]

    replayed = [
        m
        for m in read_messages(agent)
        if m.get("role") == "assistant" and m.get("reasoning_items")
    ]
    assert replayed and replayed[0]["reasoning_items"] == items
    assert replayed[0]["content"] == "(no output this turn)"
    assert any(
        any(m.get("reasoning_items") == items for m in call["messages"])
        for call in fake.calls[1:]
    )


@pytest.mark.parametrize(
    "model_name",
    ["deepseek/deepseek-v4-pro", "moonshot/kimi-k3"],
)
async def test_chat_reasoning_content_round_trip(tmp_path, monkeypatch, model_name):
    """Thinking chat models receive their complete prior assistant turn."""
    thought = "I should inspect the workspace before deciding what to change."
    first = bash_response("echo hi")
    first.choices[0].message.reasoning_content = thought
    fake = FakeCompletion([first, *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = Proximus(
        logs_dir=tmp_path / "logs",
        model_name=model_name,
        api_key_env=None,
    )
    await agent.run("Task", FakeEnv(), AgentContext())  # type: ignore[arg-type]

    replayed = [
        message
        for message in read_messages(agent)
        if message.get("role") == "assistant"
        and message.get("reasoning_content") == thought
    ]
    assert replayed
    assert any(
        message.get("reasoning_content") == thought
        for message in fake.calls[1]["messages"]
    )


async def test_gemini_thought_signature_round_trip(tmp_path, monkeypatch):
    """Gemini needs its opaque thought signature on the next model turn."""
    fields = {"thought_signatures": ["opaque-signature"]}
    first = bash_response("echo hi")
    first.choices[0].message.reasoning_content = "I should inspect the workspace."
    first.choices[0].message.provider_specific_fields = fields
    fake = FakeCompletion([first, *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = Proximus(
        logs_dir=tmp_path / "logs",
        model_name="gemini/test-model",
        api_key_env=None,
    )
    await agent.run("Task", FakeEnv(), AgentContext())  # type: ignore[arg-type]

    assert any(
        message.get("provider_specific_fields") == fields
        for message in fake.calls[1]["messages"]
    )


async def test_anthropic_gets_no_reasoning_items_or_extra_body(tmp_path, monkeypatch):
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = Proximus(
        logs_dir=tmp_path / "logs",
        model_name="anthropic/claude-opus-test",
        reasoning_effort="high",
    )
    await agent.run("Task", FakeEnv(), AgentContext())  # type: ignore[arg-type]

    assert "extra_body" not in fake.calls[0]
    assert not [
        m
        for m in read_messages(agent)
        if m.get("role") == "assistant" and m.get("reasoning_items")
    ]


async def test_split_choices_are_merged(tmp_path, monkeypatch):
    """litellm's Responses bridge splits a text+tool_calls turn across two
    choices; taking choices[0] would silently drop the tool call."""
    split = FakeResponse(FakeMessage(content="Diagnosing the failure."))
    tool_choice = FakeChoice(
        FakeMessage(
            content=None,
            tool_calls=[
                FakeToolCall("call_x", "bash", json.dumps({"command": "echo hi"}))
            ],
        ),
        finish_reason="tool_calls",
    )
    split.choices.append(tool_choice)
    fake = FakeCompletion([split, *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    env = FakeEnv()
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    # The tool call from the SECOND choice ran, and the prose from the first
    # is preserved on the assistant turn.
    assert env.exec_calls == ["echo hi"]
    assistant = [m for m in read_messages(agent) if m.get("role") == "assistant"]
    assert "Diagnosing the failure." in str(assistant[0].get("content"))
    assert assistant[0]["tool_calls"][0]["function"]["name"] == "bash"


def test_context_limit_is_longest_window_uncapped_by_price_tiers(tmp_path, monkeypatch):
    # No price-tier capping: the limit is the longest window the registry
    # reports, tier pricing keys notwithstanding.
    monkeypatch.setattr(
        litellm,
        "get_model_info",
        lambda model: {
            "max_input_tokens": 1_050_000,
            "max_tokens": 128_000,
            "input_cost_per_token_above_272k_tokens": 1e-05,
        },
    )
    agent = make_agent(tmp_path)
    assert agent._context_limit == 1_050_000


async def test_bridged_request_carries_responses_params(tmp_path, monkeypatch):
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_bridged_agent(tmp_path)
    await agent.run("Task", FakeEnv(), AgentContext())  # type: ignore[arg-type]

    body = fake.calls[0]["extra_body"]
    assert body["store"] is False
    assert body["include"] == ["reasoning.encrypted_content"]
    # The summary is what puts a human-readable trace in the trajectory.
    assert body["reasoning_summary"] == "auto"
    # prompt_cache_key must ride in extra_body: a top-level kwarg is silently
    # filtered out before any provider sees it.
    assert body["prompt_cache_key"] == agent._run.session_id
    assert "prompt_cache_key" not in fake.calls[0]


async def test_xai_responses_request_keeps_system(tmp_path, monkeypatch):
    """xAI Responses needs the shared Responses extra_body, plus list-content
    system messages so LiteLLM does not drop them as unsupported
    ``instructions``."""
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)
    monkeypatch.setattr(
        litellm,
        "get_model_info",
        lambda model: {"mode": "responses", "litellm_provider": "xai"},
    )

    agent = Proximus(
        logs_dir=tmp_path / "logs",
        model_name="xai/grok-test-model",
        reasoning_effort="high",
    )
    await agent.run("Task", FakeEnv(), AgentContext())  # type: ignore[arg-type]

    call = fake.calls[0]
    assert call["model"] == "xai/grok-test-model"
    body = call["extra_body"]
    assert body["store"] is False
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["reasoning_summary"] == "auto"
    assert body["prompt_cache_key"] == agent._run.session_id
    systems = [m for m in call["messages"] if m.get("role") == "system"]
    assert systems
    assert isinstance(systems[0]["content"], list)
    assert systems[0]["content"][0]["type"] == "text"
    assert systems[0]["content"][0]["text"]


async def test_no_responses_params_without_reasoning_effort(tmp_path, monkeypatch):
    # Not bridged -> these params would 400 against /chat/completions.
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    await agent.run("Task", FakeEnv(), AgentContext())  # type: ignore[arg-type]

    assert "extra_body" not in fake.calls[0]


async def test_reasoning_merged_across_choices(tmp_path, monkeypatch):
    # A turn can interleave [reasoning_A, message, reasoning_B, function_call];
    # replaying only the first drops half the chain of thought.
    split = FakeResponse(FakeMessage(content="thinking out loud"))
    split.choices[0].message.reasoning_items = [{"id": "rs_A"}]
    split.choices[0].message.reasoning_content = "thought A"
    second = FakeChoice(
        FakeMessage(
            content=None,
            tool_calls=[
                FakeToolCall("call_y", "bash", json.dumps({"command": "echo hi"}))
            ],
        ),
        finish_reason="tool_calls",
    )
    second.message.reasoning_items = [{"id": "rs_B"}]
    second.message.reasoning_content = "thought B"
    split.choices.append(second)
    fake = FakeCompletion([split, *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_bridged_agent(tmp_path)
    await agent.run("Task", FakeEnv(), AgentContext())  # type: ignore[arg-type]

    msgs = read_messages(agent)
    items = [
        i
        for m in msgs
        if m.get("role") == "assistant"
        for i in (m.get("reasoning_items") or [])
    ]
    assert [i["id"] for i in items] == ["rs_A", "rs_B"]
    # Prose is split into its own assistant message so the replay transform
    # cannot drop it, and both summaries reach the trajectory.
    prose = [m for m in msgs if m.get("role") == "assistant" and m.get("content")]
    assert any("thinking out loud" in str(m["content"]) for m in prose)
    steps = read_trajectory(agent)["steps"]
    assert any(
        "thought A" in str(s.get("reasoning_content") or "")
        and "thought B" in str(s.get("reasoning_content") or "")
        for s in steps
    )


async def test_bare_thinking_turn_is_not_replayed(tmp_path, monkeypatch):
    """Anthropic rejects an assistant message whose final block is thinking.

    Replaying thinking blocks on a turn with neither tool calls nor text
    produces exactly that, and the API then rejects the whole conversation
    mid-run. Only reachable because text-only turns are not capped.
    """
    bare = FakeResponse(FakeMessage(content=None))
    bare.choices[0].message.thinking_blocks = [{"type": "thinking", "thinking": "hm"}]
    working = bash_response("echo hi")
    working.choices[0].message.thinking_blocks = [
        {"type": "thinking", "thinking": "go"}
    ]
    fake = FakeCompletion([bare, working, *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = Proximus(
        logs_dir=tmp_path / "logs",
        model_name="anthropic/claude-opus-test",
        reasoning_effort="high",
    )
    await agent.run("Task", FakeEnv(), AgentContext())  # type: ignore[arg-type]

    msgs = [m for m in read_messages(agent) if m.get("role") == "assistant"]
    # Bare turns get explicit placeholder text: content=None with no tool_calls
    # is a 400 at some providers ("content or tool_calls must be set").
    bare_turns = [
        m
        for m in msgs
        if not m.get("tool_calls") and m.get("content") == "(no output this turn)"
    ]
    assert bare_turns, "the prose/thinking-only turn should still be recorded"
    assert all("thinking_blocks" not in m for m in bare_turns), (
        "a turn with no tool calls and no text must not carry thinking blocks"
    )
    # ...while a turn that DOES call a tool still replays them (Anthropic
    # requires that for tool-use continuity).
    tool_turns = [m for m in msgs if m.get("tool_calls")]
    assert any(m.get("thinking_blocks") for m in tool_turns)


# ----- environment-gone handling -----


class DyingEnv(FakeEnv):
    """FakeEnv whose sandbox "dies": exec on the given command raises the
    typed EnvironmentGoneError, as a mapped provider (Modal, Daytona) does.
    Harness bookkeeping commands keep succeeding so setup stays clean."""

    def __init__(self, dead_command: str, **kwargs):
        super().__init__(**kwargs)
        self.dead_command = dead_command

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if command == self.dead_command:
            raise EnvironmentGoneError(
                "Modal sandbox is gone: SandboxTerminatedError: killed"
            )
        return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec)


class DisconnectedEnv(FakeEnv):
    """Modal exec control channel is disconnected, but setup stays clean."""

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if command == "boom":
            raise ConnectionError("[Errno -2] Name or service not known")
        return await super().exec(
            command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user
        )


async def test_env_gone_ends_run_on_first_typed_error(tmp_path, monkeypatch):
    # One typed EnvironmentGoneError ends the run: no second LLM call, no
    # marker counting, and the loss is recorded in the metadata.
    fake = FakeCompletion([bash_response("boom")])
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    context = AgentContext()
    with pytest.raises(EnvironmentGoneError):
        await agent.run("Do a task", DyingEnv("boom"), context)  # type: ignore[arg-type]

    assert len(fake.calls) == 1
    assert context.metadata is not None
    assert context.metadata["env_lost"] is True
    assert context.metadata["n_submissions"] == 0

    # The failed call still gets a tool result, so the recorded conversation
    # stays provider-valid.
    messages = read_messages(agent)
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["tool_call_id"] == "call_1"
    assert "sandbox is gone" in messages[-1]["content"]

    # Unlike the legacy marker path, the typed path drains the batch, so the
    # turn is still recorded as a trajectory step.
    trajectory = read_trajectory(agent)
    last_step = trajectory["steps"][-1]
    assert last_step["tool_calls"][0]["function_name"] == "bash"
    assert "sandbox is gone" in last_step["observation"]["results"][0]["content"]


async def test_exec_connection_loss_fails_after_one_model_turn(tmp_path, monkeypatch):
    fake = FakeCompletion([bash_response("boom")])
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    context = AgentContext()
    with pytest.raises(EnvironmentGoneError, match="Name or service not known"):
        await agent.run("Do a task", DisconnectedEnv(), context)  # type: ignore[arg-type]

    assert len(fake.calls) == 1
    assert context.metadata is not None
    assert context.metadata["env_lost"] is True
    assert context.metadata["ended_by_submit"] is False
    trajectory = read_trajectory(agent)
    assert trajectory["steps"][-1]["tool_calls"][0]["function_name"] == "bash"
    assert (
        "Name or service not known"
        in (trajectory["steps"][-1]["observation"]["results"][0]["content"])
    )


async def test_env_gone_mid_batch_fills_remaining_results(tmp_path, monkeypatch):
    # First call of the batch hits the dead sandbox; the second must not
    # execute but still needs a tool result for protocol validity.
    double_bash = FakeResponse(
        FakeMessage(
            content="Two commands.",
            tool_calls=[
                FakeToolCall("call_1", "bash", json.dumps({"command": "boom"})),
                FakeToolCall("call_2", "bash", json.dumps({"command": "echo after"})),
            ],
        )
    )
    fake = FakeCompletion([double_bash])
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    env = DyingEnv("boom")
    context = AgentContext()
    with pytest.raises(EnvironmentGoneError):
        await agent.run("Do a task", env, context)  # type: ignore[arg-type]

    assert context.metadata is not None
    assert context.metadata["env_lost"] is True
    assert "echo after" not in env.exec_calls

    messages = read_messages(agent)
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_1", "call_2"]
    assert "sandbox is gone" in tool_messages[0]["content"]
    assert "Not executed" in tool_messages[1]["content"]


def test_drop_orphan_tool_results_enforces_pairing():
    from harbor_proximus.llm import drop_orphan_tool_results

    asst = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "a",
                "type": "function",
                "function": {"name": "bash", "arguments": "{}"},
            }
        ],
    }
    result = {"role": "tool", "tool_call_id": "a", "content": "ok"}
    closer = {"role": "user", "content": "Continue."}
    kept = drop_orphan_tool_results(
        [
            {"role": "tool", "tool_call_id": "stale", "content": "owner was cut"},
            asst,
            result,
            closer,
            {"role": "tool", "tool_call_id": "a", "content": "after a user turn"},
        ]
    )
    assert kept == [asst, result, closer]


async def test_disarm_is_announced_on_the_cancelling_call(tmp_path, monkeypatch):
    """submit (arms) -> bash (disarms): the bash RESULT must state the confirmation
    was cancelled. A silent lapse leaves models re-submitting until the budget
    runs out ("I submitted twice — why hasn't the run ended?")."""
    fake = FakeCompletion(
        [
            submit_response(),
            bash_response("echo hi"),
            submit_response(call_id="again"),
            submit_response(call_id="confirm"),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    env = FakeEnv()
    context = AgentContext()
    await agent.run("Task", env, context)  # type: ignore[arg-type]

    tools_out = [m["content"] for m in read_messages(agent) if m.get("role") == "tool"]
    noted = [
        c for c in tools_out if "cancelled your pending submission confirmation" in c
    ]
    assert len(noted) == 1  # exactly the disarming bash call carries the note
    assert "Submission #" not in noted[0]  # ... and it is not a submit result
    # The lapse costs nothing durable: the next submit re-records, its confirm ends the run.
    assert any("Final submission confirmed" in c for c in tools_out)
    assert context.metadata is not None and context.metadata["n_submissions"] == 2


async def test_viewed_image_saved_to_disk_and_referenced_in_step(tmp_path, monkeypatch):
    """The conversation's base64 blob is not an archive: every viewed image is
    also persisted under <logs_dir>/images/ and referenced from the step's
    extra, so external tooling can carry it."""
    view_call = FakeResponse(
        FakeMessage(
            content="Looking.",
            tool_calls=[
                FakeToolCall(
                    "call_v", "view_image", json.dumps({"path": "/app/plot.png"})
                )
            ],
        )
    )
    fake = FakeCompletion([view_call, *finish_responses()])
    monkeypatch.setattr(litellm, "acompletion", fake)
    monkeypatch.setattr(litellm, "supports_vision", lambda model: True)
    png = make_tiny_png()

    class ImageEnv(FakeEnv):
        async def download_file(self, source_path, target_path):
            from pathlib import Path

            Path(target_path).write_bytes(png)

    agent = make_agent(tmp_path)
    await agent.run("Check the plot", ImageEnv(), AgentContext())  # type: ignore[arg-type]

    saved = list((agent.logs_dir / "images").glob("turn*.png"))
    assert len(saved) == 1
    assert saved[0].read_bytes() == png

    trajectory = read_trajectory(agent)
    refs = [
        (s.get("extra") or {}).get("images")
        for s in trajectory["steps"]
        if (s.get("extra") or {}).get("images")
    ]
    assert refs == [[f"images/{saved[0].name}"]]


async def test_request_too_large_evicts_newest_image_and_retries(tmp_path, monkeypatch):
    # A gateway body-cap rejection (413 on the WHOLE request) evicts the
    # NEWEST attached image — least prompt cache busted, and it is the image
    # the model can re-view compressed — relays the provider error verbatim,
    # and the run continues. The older image survives untouched.
    from harbor_proximus.prompts import REQUEST_TOO_LARGE_MESSAGE

    view1 = FakeResponse(
        FakeMessage(
            content="look 1",
            tool_calls=[
                FakeToolCall("c1", "view_image", json.dumps({"path": "/app/a.png"}))
            ],
        )
    )
    view2 = FakeResponse(
        FakeMessage(
            content="look 2",
            tool_calls=[
                FakeToolCall("c2", "view_image", json.dumps({"path": "/app/b.png"}))
            ],
        )
    )
    gateway_413 = RuntimeError(
        '{"code":"RequestEntityTooLarge","message":"Request body size exceeds '
        'maximum allowed size: 10485760 bytes"}'
    )
    fake = FakeCompletion(
        [
            view1,
            view2,
            gateway_413,
            bash_response("echo recovered"),
            *finish_responses(),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)
    monkeypatch.setattr(litellm, "supports_vision", lambda model: True)

    png = make_tiny_png()

    class ImageEnv(FakeEnv):
        async def download_file(self, source_path, target_path):
            from pathlib import Path

            Path(target_path).write_bytes(png)

    env = ImageEnv()
    agent = make_agent(tmp_path)
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    # The failing call was not retried; the turn was re-run after eviction.
    assert env.exec_calls == ["echo recovered"]
    messages = read_messages(agent)
    image_parts = [
        (i, p)
        for i, m in enumerate(messages)
        if isinstance(m.get("content"), list)
        for p in m["content"]
        if isinstance(p, dict) and p.get("type") == "image_url"
    ]
    placeholders = [
        i
        for i, m in enumerate(messages)
        if isinstance(m.get("content"), list)
        for p in m["content"]
        if isinstance(p, dict) and "image removed" in p.get("text", "")
    ]
    assert len(image_parts) == 1 and len(placeholders) == 1
    assert image_parts[0][0] < placeholders[0]  # the OLDER image survived
    notes = [
        m["content"]
        for m in messages
        if m.get("role") == "user"
        and str(m.get("content", "")).startswith("Request failed: ")
    ]
    assert len(notes) == 1
    assert "10485760" in notes[0]  # the provider's error, verbatim
    assert notes[0] == REQUEST_TOO_LARGE_MESSAGE.format(error=gateway_413)


async def test_invalid_image_is_removed_and_error_returned_to_model(
    tmp_path, monkeypatch
):
    from harbor_proximus.prompts import INVALID_IMAGE_MESSAGE

    view = FakeResponse(
        FakeMessage(
            content="inspect crop",
            tool_calls=[
                FakeToolCall("c1", "view_image", json.dumps({"path": "/app/tiny.png"}))
            ],
        )
    )
    invalid_image = RuntimeError(
        '{"code":"invalid_image","error":"Image dimensions 19x2 are too '
        'small. Both width and height must be at least 8 pixels."}'
    )
    fake = FakeCompletion(
        [
            view,
            invalid_image,
            bash_response("python resize_and_pad.py /app/tiny.png"),
            *finish_responses(),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)
    monkeypatch.setattr(litellm, "supports_vision", lambda model: True)

    png = make_tiny_png()

    class ImageEnv(FakeEnv):
        async def download_file(self, source_path, target_path):
            from pathlib import Path

            Path(target_path).write_bytes(png)

    env = ImageEnv()
    agent = make_agent(tmp_path)
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    assert env.exec_calls == ["python resize_and_pad.py /app/tiny.png"]
    messages = read_messages(agent)
    assert not any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for message in messages
        for part in (
            message.get("content") if isinstance(message.get("content"), list) else []
        )
    )
    assert any(
        isinstance(part, dict) and "provider rejected it" in part.get("text", "")
        for message in messages
        for part in (
            message.get("content") if isinstance(message.get("content"), list) else []
        )
    )
    notes = [
        m["content"]
        for m in messages
        if m.get("role") == "user"
        and str(m.get("content", "")).startswith("Request failed: ")
    ]
    expected_error = str(invalid_image)
    assert notes == [INVALID_IMAGE_MESSAGE.format(error=expected_error)]
    assert "19x2" in notes[0]
    assert "call `view_image` again" in notes[0]


async def test_image_count_limit_compacts_instead_of_repeated_eviction(
    tmp_path, monkeypatch
):
    # a provider rejects the entire history above 50 images. A count cap is
    # unlike a byte cap: keeping 50 images would leave the run one view away
    # from failing again, while replacing even one already busts the cache.
    # Remove image payloads, summarize once, and continue from a compact handoff.
    views = [
        FakeResponse(
            FakeMessage(
                content=f"look {n}",
                tool_calls=[
                    FakeToolCall(
                        f"c{n}", "view_image", json.dumps({"path": f"/app/{n}.png"})
                    )
                ],
            )
        )
        for n in range(51)
    ]
    image_limit = RuntimeError(
        "request contains 51 images, exceeding the maximum of 50 allowed per request"
    )
    fake = FakeCompletion(
        [
            *views,
            image_limit,
            FakeResponse(FakeMessage(content="Kept the important visual findings.")),
            bash_response("echo recovered"),
            *finish_responses(),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)
    monkeypatch.setattr(litellm, "supports_vision", lambda model: True)

    png = make_tiny_png()

    class ImageEnv(FakeEnv):
        async def download_file(self, source_path, target_path):
            from pathlib import Path

            Path(target_path).write_bytes(png)

    env = ImageEnv()
    agent = make_agent(tmp_path)
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    assert env.exec_calls == ["echo recovered"]
    messages = read_messages(agent)
    assert not any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for message in messages
        for part in (
            message.get("content") if isinstance(message.get("content"), list) else []
        )
    )
    assert any(
        "Kept the important visual findings." in str(message.get("content"))
        for message in messages
    )
    trajectory = read_trajectory(agent)
    assert any(
        (step.get("extra") or {}).get("context_management", {}).get("type")
        == "compaction"
        for step in trajectory["steps"]
    )


async def test_request_too_large_without_images_is_fatal(tmp_path, monkeypatch):
    # Nothing evictable means nothing recoverable — the typed error
    # propagates instead of looping forever.
    from harbor_proximus import llm as llm_mod

    fake = FakeCompletion(
        [RuntimeError("Request body size exceeds maximum allowed size: 10485760 bytes")]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    with pytest.raises(llm_mod.RequestTooLargeError):
        await agent.run("Task", FakeEnv(), AgentContext())  # type: ignore[arg-type]
    assert len(fake.calls) == 1  # deterministic: never retried


async def test_provider_outage_past_budget_fails_the_run(tmp_path, monkeypatch):
    # A provider down longer than the run has left: retries stop on budget,
    # and the trial FAILS rather than looking like a clean completion.
    import asyncio

    from harbor_proximus import llm as llm_mod

    calls = {"n": 0}

    async def down(**kwargs):
        calls["n"] += 1
        await asyncio.sleep(0.15)  # burns past the tiny budget on attempt 1
        raise litellm.exceptions.Timeout("provider down", "openai", "m")

    monkeypatch.setattr(litellm, "acompletion", down)
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)

    agent = make_agent(tmp_path, time_budget_sec=0.1)
    with pytest.raises(llm_mod.ProviderRetryBudgetError, match="Budget expired"):
        await agent.run("Task", FakeEnv(), AgentContext())  # type: ignore[arg-type]
    assert calls["n"] >= 1


async def test_env_gone_skips_restore(tmp_path, monkeypatch):
    # A dead sandbox ends the run outright: no end-of-run winner restore —
    # env_lost in the metadata is the whole report. A submission is recorded
    # FIRST so a winner exists: without one, restore_winner() returns early
    # and the skip would be untestable.
    fake = FakeCompletion([submit_response(), bash_response("boom")])
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    env = DyingEnv("boom")
    context = AgentContext()
    with pytest.raises(EnvironmentGoneError):
        await agent.run("Do a task", env, context)  # type: ignore[arg-type]

    assert context.metadata is not None
    assert context.metadata["env_lost"] is True
    assert context.metadata["n_submissions"] == 1  # the winner exists...
    # ...but nothing touches the dead sandbox after the loop ends: no
    # pre-restore end-state pin, no checkout.
    assert not any("end-state" in c or "checkout" in c for c in env.all_commands), (
        "restore must not run against a dead sandbox"
    )


async def test_finalize_only_on_confirmed_submit(tmp_path, monkeypatch):
    from harbor_proximus.submissions import Submissions

    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)
    restore = AsyncMock(return_value=True)
    monkeypatch.setattr(Submissions, "restore_winner", restore)
    agent = make_agent(
        tmp_path, auto_submit_on_exit=False, finalize_only_on_submit=True
    )
    context = AgentContext()
    await agent.run("Task", FakeEnv(), context)  # type: ignore[arg-type]

    restore.assert_awaited_once()
    assert context.metadata is not None
    assert context.metadata["ended_by_submit"] is True


async def test_finalize_only_skips_budget_exit(tmp_path, monkeypatch):
    from harbor_proximus.submissions import Submissions

    fake = FakeCompletion([])
    monkeypatch.setattr(litellm, "acompletion", fake)
    monkeypatch.setattr(Proximus, "_elapsed", lambda self: 4000.0)
    restore = AsyncMock(return_value=True)
    monkeypatch.setattr(Submissions, "restore_winner", restore)
    agent = make_agent(
        tmp_path,
        time_budget_sec=3600,
        auto_submit_on_exit=False,
        finalize_only_on_submit=True,
    )
    context = AgentContext()
    await agent.run("Task", FakeEnv(), context)  # type: ignore[arg-type]

    restore.assert_not_awaited()
    assert context.metadata is not None
    assert context.metadata["ended_by_submit"] is False
    assert context.metadata["n_submissions"] == 0


async def test_confirmed_submit_never_auto_pins_a_new_winner(tmp_path, monkeypatch):
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path, auto_submit_on_exit=True, finalize_only_on_submit=True)
    auto_submit = AsyncMock()
    monkeypatch.setattr(agent, "_auto_final_submission", auto_submit)
    context = AgentContext()
    await agent.run("Task", FakeEnv(), context)  # type: ignore[arg-type]

    auto_submit.assert_not_awaited()
    assert context.metadata is not None
    assert context.metadata["ended_by_submit"] is True
    assert context.metadata["submitted_commit"] == fake_hash(1)
