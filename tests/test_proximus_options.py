"""Terminus-style options and subclass hooks; unset, each keeps the default behavior."""

import json

import litellm
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
    assert agent._system_prompt == prompts.system_prompt(False)
    assert agent.summary_request() == prompts.SUMMARY_REQUEST
    assert tool_names(agent) == ["bash", "list_submissions", "submit"]


async def test_max_turns_ends_the_run_with_the_exit_submit(tmp_path, monkeypatch):
    fake = FakeCompletion([bash_response("one"), bash_response("two")])
    monkeypatch.setattr(litellm, "acompletion", fake)
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
        monkeypatch.setattr(litellm, "acompletion", fake)
        agent = make_agent(tmp_path, **kwargs)
        await agent.run("Task", FakeEnv(), AgentContext())  # type: ignore[arg-type]
        assert len(fake.calls) == 3  # no summary call


async def test_overflow_without_summarize_ends_the_run(tmp_path, monkeypatch):
    overflow = litellm.ContextWindowExceededError(
        "prompt too long", model="gpt-test", llm_provider="openai"
    )
    monkeypatch.setattr(litellm, "acompletion", FakeCompletion([overflow]))
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
    monkeypatch.setattr(litellm, "acompletion", FakeCompletion(finish_responses()))
    agent = make_agent(tmp_path, prompt_template_path=str(template))
    await agent.run("Fix the bug", FakeEnv(), AgentContext())  # type: ignore[arg-type]
    assert "Solve carefully.\n\nFix the bug" in read_messages(agent)[1]["content"]
    with pytest.raises(ValueError, match="not found"):
        make_agent(tmp_path, prompt_template_path=str(tmp_path / "missing.j2"))


async def test_llm_shorthands_and_llm_call_kwargs_win(tmp_path, monkeypatch):
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)
    agent = Proximus(
        logs_dir=tmp_path / "logs",
        model_name="anthropic/claude-test",
        temperature=0.2,
        api_base="https://option",
        max_thinking_tokens=2048,
        llm_call_kwargs={"api_base": "https://call-kwargs"},
        model_info={"max_input_tokens": 54321, "litellm_provider": "anthropic"},
    )
    assert agent._context_limit == 54321
    await agent.run("Task", FakeEnv(), AgentContext())  # type: ignore[arg-type]
    call = fake.calls[0]
    assert call["temperature"] == 0.2
    assert call["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert call["api_base"] == "https://call-kwargs"


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
    monkeypatch.setattr(litellm, "acompletion", fake)
    agent = CustomProximus(logs_dir=tmp_path / "logs", model_name="openai/gpt-test")
    env = FakeEnv()
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    steps = read_trajectory(agent)["steps"]
    assert steps[0]["message"].endswith("Prefer small commits.")
    assert steps[1]["message"] == "[custom] Task"
    assert "count_lines" in tool_names(agent)
    assert "wc -l a.py" in env.exec_calls
    assert fake.calls[1]["messages"][-1]["content"] == "Summarize briefly."
