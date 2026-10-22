"""Git-free runs must preserve files and finish without submission machinery."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext

from harbor_proximus import hard_timeout_sec
from harbor_proximus.config import ProximusConfig
from harbor_proximus.task_config import FINALIZE_MARGIN_SEC
from tests.test_proximus_loop import (
    FakeCompletion,
    FakeEnv,
    FakeMessage,
    FakeResponse,
    FakeToolCall,
    bash_response,
    make_agent,
    read_messages,
    read_trajectory,
    small_context,
    text_response,
)


def done_response(*extra):
    return FakeResponse(
        FakeMessage(tool_calls=[FakeToolCall("finish", "done", "{}"), *extra])
    )


class NoGitEnv(FakeEnv):
    async def exec(self, command, **kwargs):
        assert "git " not in command and "refs/proximus" not in command
        return await super().exec(command, **kwargs)


def assert_no_submissions(agent, context, reason):
    assert context.metadata["end_reason"] == reason
    assert context.metadata["submission_mode"] == "none"
    assert not context.metadata["ended_by_submit"]
    assert context.metadata["submitted_commit"] is None
    assert context.metadata["n_submissions"] == 0
    assert not (agent.logs_dir / "submissions.jsonl").exists()
    assert not hasattr(agent, "_submissions")


def test_mode_validation_and_tools(tmp_path):
    with pytest.raises(ValueError):
        ProximusConfig(submission_mode="files")
    agent = make_agent(tmp_path, submission_mode="none")
    assert set(agent._tools) == {"bash", "done"}
    assert "call `done`" in agent._system_prompt.lower()
    assert "check the remaining time and budget" in agent._system_prompt
    assert "continue working" in agent._system_prompt
    assert "Call `done` only after" in agent._system_prompt
    assert "continue working instead" in agent._tools["done"].description
    assert "grading" not in agent._system_prompt.lower()
    assert "submit" not in agent._system_prompt.lower()
    with pytest.raises(ValueError, match="cannot remove done"):
        make_agent(tmp_path, submission_mode="none", disallowed_tools=["done"])
    assert hard_timeout_sec({"submission_mode": "none", "time_budget_sec": 1200}) == (
        1200 + FINALIZE_MARGIN_SEC
    )
    assert hard_timeout_sec({"submission_mode": "none"}) is None


async def test_done_ends_once_and_skips_later_calls(tmp_path, monkeypatch):
    fake = FakeCompletion(
        [
            done_response(FakeToolCall("late", "bash", '{"command":"touch bad"}')),
        ]
    )
    fake.install(monkeypatch)
    agent = make_agent(tmp_path, submission_mode="none", time_budget_sec=1200)
    context = AgentContext()
    env = NoGitEnv()
    await agent.run("Write the requested file.", env, context)
    assert len(fake.calls) == 1
    assert env.exec_calls == []
    assert_no_submissions(agent, context, "done")
    assert "earlier tool call" in read_messages(agent)[-1]["content"]
    assert (
        read_trajectory(agent)["steps"][-1]["tool_calls"][0]["function_name"] == "done"
    )


@pytest.mark.parametrize(
    "ending", ["time_budget", "max_turns", "stalled", "error", "cancelled"]
)
async def test_other_exits_never_run_git(tmp_path, monkeypatch, ending):
    responses = {
        "time_budget": [],
        "max_turns": [text_response("Working.")],
        "stalled": [text_response("")] * 3,
        "error": [RuntimeError("provider failed")],
        "cancelled": [asyncio.CancelledError()],
    }[ending]
    fake = FakeCompletion(responses)
    fake.install(monkeypatch)
    agent = make_agent(
        tmp_path,
        submission_mode="none",
        time_budget_sec=1200,
        max_turns=1 if ending == "max_turns" else None,
    )
    if ending == "time_budget":
        monkeypatch.setattr(agent, "_elapsed", lambda: 1201)
    context = AgentContext()
    if ending in {"error", "cancelled"}:
        error = RuntimeError if ending == "error" else asyncio.CancelledError
        monkeypatch.setattr(agent._llm, "query", AsyncMock(side_effect=error()))
        with pytest.raises(error):
            await agent.run("Write the requested file.", NoGitEnv(), context)
    else:
        await agent.run("Write the requested file.", NoGitEnv(), context)
    assert_no_submissions(agent, context, "error" if ending == "cancelled" else ending)
    if ending == "stalled":
        assert "call `done`" in read_messages(agent)[-2]["content"]
        assert "submit" not in json.dumps(read_messages(agent)).lower()


async def test_compaction_keeps_done_and_has_no_git_advice(tmp_path, monkeypatch):
    small_context(monkeypatch)
    fake = FakeCompletion(
        [
            bash_response("echo progress", prompt_tokens=900),
            text_response("Work in progress."),
            done_response(),
        ]
    )
    fake.install(monkeypatch)
    agent = make_agent(tmp_path, submission_mode="none")
    context = AgentContext()
    await agent.run("Write the requested file.", NoGitEnv(), context)
    assert "git log" not in json.dumps(read_messages(agent))
    assert "submit" not in json.dumps(read_messages(agent)).lower()
    assert all(
        {t["function"]["name"] for t in c["tools"]} == {"bash", "done"}
        for c in fake.calls
    )
    assert_no_submissions(agent, context, "done")


@pytest.mark.parametrize("finish", [True, False])
async def test_real_workspace_files_survive_done_or_turn_limit(
    tmp_path, monkeypatch, finish
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact = workspace / "result.json"
    command = "printf '%s' '{\"value\":42}' > result.json"

    class FileEnv(NoGitEnv):
        async def exec(self, command, cwd=None, **kwargs):
            if cwd == str(workspace):
                process = await asyncio.create_subprocess_exec(
                    "bash",
                    "-c",
                    command,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await process.communicate()
                return ExecResult(
                    stdout=stdout.decode(),
                    stderr=stderr.decode(),
                    return_code=process.returncode,
                )
            return await super().exec(command, cwd=cwd, **kwargs)

    responses = [bash_response(command)] + ([done_response()] if finish else [])
    fake = FakeCompletion(responses)
    fake.install(monkeypatch)
    agent = make_agent(tmp_path, submission_mode="none", max_turns=2 if finish else 1)
    env = FileEnv()
    env.task_env_config.workdir = str(workspace)
    context = AgentContext()
    await agent.run("Write result.json.", env, context)
    assert artifact.read_bytes() == b'{"value":42}'
    assert sorted(p.name for p in workspace.iterdir()) == ["result.json"]
    assert_no_submissions(agent, context, "done" if finish else "max_turns")
