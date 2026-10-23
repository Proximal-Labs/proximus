"""Terminus-style options and subclass hooks; unset, each keeps the default behavior."""

import json

import pytest
from harbor.llms.base import ContextLengthExceededError
from harbor.models.agent.context import AgentContext
from pydantic import BaseModel

from harbor_proximus import Proximus, prompts, tools
from tests.test_proximus_loop import (
    FakeCompletion,
    FakeEnv,
    FakeMessage,
    FakeResponse,
    FakeToolCall,
    api_error,
    bash_response,
    finish_responses,
    make_agent,
    read_messages,
    read_trajectory,
    small_context,
    text_response,
)


def tool_names(agent: Proximus) -> list[str]:
    return [schema["function"]["name"] for schema in agent._tool_schemas]


def test_defaults_are_unchanged(tmp_path):
    agent = make_agent(tmp_path)
    assert agent._system_prompt == prompts.system_prompt(agent.config)
    assert agent.summary_request() == prompts.SUMMARY_REQUEST
    assert tool_names(agent) == ["bash", "list_submissions", "submit"]


async def test_max_turns_ends_the_run_with_the_exit_submit(tmp_path, monkeypatch):
    fake = FakeCompletion([bash_response("one"), bash_response("two")])
    fake.install(monkeypatch)
    agent = make_agent(tmp_path, max_turns=2)
    env = FakeEnv()
    context = AgentContext()
    await agent.run("Task", env, context)  # type: ignore[arg-type]

    assert len(fake.calls) == 2
    assert context.metadata["n_turns"] == 2
    assert context.metadata["n_submissions"] == 1  # the exit auto-submit


async def test_compaction_threshold_and_enable_summarize(tmp_path, monkeypatch):
    small_context(monkeypatch)
    for kwargs in ({"compaction_threshold": 0.95}, {"enable_summarize": False}):
        fake = FakeCompletion(
            [bash_response("ls", prompt_tokens=900), *finish_responses()]
        )
        fake.install(monkeypatch)
        agent = make_agent(tmp_path, **kwargs)
        await agent.run("Task", FakeEnv(), AgentContext())  # type: ignore[arg-type]
        assert len(fake.calls) == 3  # no summary call


async def test_overflow_without_summarize_ends_the_run(tmp_path, monkeypatch):
    overflow = api_error(400, "prompt is too long: 250000 tokens > 200000 maximum")
    FakeCompletion([overflow]).install(monkeypatch)
    agent = make_agent(tmp_path, enable_summarize=False)
    with pytest.raises(ContextLengthExceededError):
        await agent.run("Task", FakeEnv(), AgentContext())  # type: ignore[arg-type]


def test_disallowed_tools(tmp_path):
    agent = make_agent(tmp_path, disallowed_tools=["list_submissions", "view_image"])
    assert tool_names(agent) == ["bash", "submit"]
    for name in ("bash", "submit"):
        with pytest.raises(ValueError, match="cannot remove"):
            make_agent(tmp_path, disallowed_tools=[name])
    with pytest.raises(ValueError, match="unknown tools"):
        make_agent(tmp_path, disallowed_tools=["edit"])


async def test_prompt_template_wraps_the_instruction(tmp_path, monkeypatch):
    template = tmp_path / "prompt.j2"
    template.write_text("Solve carefully.\n\n{{ instruction }}")
    FakeCompletion(finish_responses()).install(monkeypatch)
    agent = make_agent(tmp_path, prompt_template_path=str(template))
    await agent.run("Fix the bug", FakeEnv(), AgentContext())  # type: ignore[arg-type]
    assert "Solve carefully.\n\nFix the bug" in read_messages(agent)[1]["content"]
    with pytest.raises(ValueError, match="not found"):
        make_agent(tmp_path, prompt_template_path=str(tmp_path / "missing.j2"))


async def test_llm_shorthands_reach_every_request(tmp_path, monkeypatch):
    # How each API spells them, and llm_call_kwargs winning, is covered per
    # adapter in tests/test_providers.py.
    fake = FakeCompletion(finish_responses())
    fake.install(monkeypatch)
    agent = Proximus(
        logs_dir=tmp_path / "logs",
        model_name="anthropic/claude-test",
        temperature=0.2,
        api_base="https://option",
        max_thinking_tokens=2048,
        llm_call_kwargs={"max_tokens": 1000},
        model_info={"max_input_tokens": 54321},
    )
    assert agent._context_limit == 54321
    assert str(agent._llm.provider.client.base_url).rstrip("/") == "https://option"
    await agent.run("Task", FakeEnv(), AgentContext())  # type: ignore[arg-type]
    for call in fake.calls:
        assert call["temperature"] == 0.2
        assert call["max_thinking_tokens"] == 2048
        assert call["kwargs"] == {"max_tokens": 1000}


class CountArgs(BaseModel):
    path: str


class CustomProximus(Proximus):
    def system_prompt(self) -> str:
        return super().system_prompt() + "\n\nPrefer small commits."

    def initial_message(self, instruction, workdir, environment) -> str:
        return f"[custom] {instruction}"

    def summary_request(self) -> str:
        return "Summarize briefly."

    def build_tools(self) -> dict[str, tools.Tool]:
        count = tools.Tool(
            "count_lines", "Count lines in a file.", CountArgs, self._count_lines
        )
        return {**super().build_tools(), count.name: count}

    async def _count_lines(self, args: CountArgs) -> tools.ToolResult:
        result = await self.environment.exec(f"wc -l {args.path}", timeout_sec=30)
        return tools.ToolResult(str(result.stdout))


async def test_subclass_hooks(tmp_path, monkeypatch):
    small_context(monkeypatch)
    count = FakeResponse(
        FakeMessage(
            content="Counting.",
            tool_calls=[
                FakeToolCall("call_c", "count_lines", json.dumps({"path": "a.py"}))
            ],
        ),
        prompt_tokens=900,
    )
    fake = FakeCompletion([count, text_response("summary"), *finish_responses()])
    fake.install(monkeypatch)
    agent = CustomProximus(logs_dir=tmp_path / "logs", model_name="openai/gpt-test")
    env = FakeEnv()
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    steps = read_trajectory(agent)["steps"]
    assert steps[0]["message"].endswith("Prefer small commits.")
    assert steps[1]["message"] == "[custom] Task"
    assert "count_lines" in tool_names(agent)
    assert "wc -l a.py" in env.exec_calls
    assert fake.calls[1]["messages"][-1]["content"] == "Summarize briefly."


class HandoffProximus(Proximus):
    def handoff_message(self, summary, progress) -> str:
        return f"[handoff] {summary}"


async def test_sandbox_paths_and_handoff_hook(tmp_path, monkeypatch):
    small_context(monkeypatch)
    turns = [bash_response("ls", prompt_tokens=900), text_response("summary")]
    fake = FakeCompletion([*turns, *finish_responses()])
    fake.install(monkeypatch)
    agent = HandoffProximus(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-test",
        progress_file="/tmp/notes.md",
        bash_output_file_prefix="/tmp/output-",
    )
    env = FakeEnv()
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    assert "/tmp/notes.md" in agent._system_prompt
    assert "mkdir -p /tmp" in env.exec_calls
    assert any("/tmp/notes.md" in command for command in env.exec_calls)
    assert agent._executor._bash_output_file_prefix == "/tmp/output-"
    assert "/tmp/notes.md" in Proximus.handoff_message(agent, "summary", "notes")
    after = fake.calls[2]["messages"]
    assert any(m.get("content") == "[handoff] summary" for m in after)


def test_sandbox_paths_must_be_absolute(tmp_path):
    for option in ("progress_file", "bash_output_file_prefix"):
        with pytest.raises(ValueError, match="absolute sandbox path"):
            make_agent(tmp_path, **{option: "notes.md"})
