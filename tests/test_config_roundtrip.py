"""Configuration round trips: every ``[agent.proximus]`` key round-trips through
``task_agent_kwargs`` and Harbor's agent factory, bad config fails before any
sandbox exists, and ``--ak`` values parse the way ``harbor run`` parses them."""

import json
import os
import tomllib
from pathlib import Path
from typing import Any

import pytest
from harbor.agents.factory import AgentFactory
from harbor.cli.utils import parse_kwargs
from harbor.models.trial.config import AgentConfig

from harbor_proximus import Proximus, task_agent_kwargs
from harbor_proximus.config import ProximusConfig
from harbor_proximus.llm import LLM

MODEL = "openai/gpt-test"
# Optional: the tasks/ directory of a FrontierSWE v2 checkout.
FRONTIER_SWE_TASKS = os.environ.get("FRONTIER_SWE_TASKS")


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(
            f"{json.dumps(k)} = {toml_value(v)}" for k, v in value.items()
        )
        return "{ " + items + " }"
    raise TypeError(f"cannot TOML-serialize {value!r}")


def write_task(task_dir: Path, table: dict[str, Any], header: str = "") -> Path:
    lines = [header, "[agent.proximus]"]
    lines += [f"{key} = {toml_value(value)}" for key, value in table.items()]
    text = "\n".join(lines) + "\n"
    (task_dir / "task.toml").write_text(text)
    assert tomllib.loads(text)["agent"]["proximus"] == table
    return task_dir


def make_agent(kwargs: dict[str, Any], logs_dir: Path):
    return AgentFactory.create_agent_from_config(
        AgentConfig(
            import_path="harbor_proximus:Proximus", model_name=MODEL, kwargs=kwargs
        ),
        logs_dir=logs_dir,
    )


# One valid, non-default value per ProximusConfig field. A new field without an
# entry here fails test_values_cover_every_field.
FIELD_VALUES: dict[str, Any] = {
    "time_budget_sec": 1234.5,
    "submission_mode": "none",
    "submission_timeout_sec": 42,
    "auto_submit_on_exit": False,
    "restore_on_exit": False,
    "finalize_only_on_submit": True,
    "task_provides_timer": True,
    "reasoning_effort": "high",
    "api_key_env": "PROXIMUS_TIER0_KEY",
    "workspace_git_excludes": ["node_modules", "target"],
    "extra_model_info": {
        MODEL: {"max_input_tokens": 50000, "supports_vision": True},
        "gpt-test": {"max_input_tokens": 50000, "input_cost_per_token": 1e-06},
    },
    "llm_call_kwargs": {"timeout": 5, "extra_body": {"a": 1}},
    "max_turns": 7,
    "enable_summarize": False,
    "enable_view_image": True,
    "compaction_threshold": 0.5,
    "disallowed_tools": ["list_submissions"],
    "prompt_template_path": "prompt.j2",
    "progress_file": "/tmp/notes.md",
    "bash_output_file_prefix": "/tmp/output-",
    "temperature": 0.3,
    "api_base": "http://127.0.0.1:9/v1",
    "max_thinking_tokens": 2048,
    "model_info": {"max_input_tokens": 77777, "supports_vision": True},
}


def test_values_cover_every_field():
    assert set(FIELD_VALUES) == set(ProximusConfig.model_fields)
    for field, value in FIELD_VALUES.items():
        info = ProximusConfig.model_fields[field]
        default = info.get_default(call_default_factory=True)
        assert value != default, f"{field}: test value equals the default"


@pytest.mark.parametrize("field", sorted(FIELD_VALUES))
def test_every_field_round_trips_through_task_toml_and_factory(
    tmp_path, field, monkeypatch
):
    value = FIELD_VALUES[field]
    if field == "enable_view_image":
        monkeypatch.setattr(LLM, "supports_vision", lambda self: True)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    expected = value
    if field == "prompt_template_path":
        (task_dir / value).write_text("Do this: {{ instruction }}")
        expected = str((task_dir / value).resolve())
    if field == "api_key_env":
        monkeypatch.setenv(value, "roundtrip-key")

    kwargs = task_agent_kwargs(write_task(task_dir, {field: value}), MODEL)
    assert kwargs[field] == expected
    # No [agent].timeout_sec and no /app artifact: nothing else is defaulted.
    assert set(kwargs) == {field}

    agent = make_agent(kwargs, tmp_path / "logs")
    assert getattr(agent.config, field) == expected

    if field == "disallowed_tools":
        assert "list_submissions" not in agent._tools
        assert {"bash", "submit"} <= set(agent._tools)
    if field == "model_info":
        assert agent._context_limit == 77777
    if field == "extra_model_info":
        assert agent._context_limit == 50000
        assert "view_image" not in agent._tools
    if field == "enable_view_image":
        assert "view_image" in agent._tools
    if field == "api_key_env":
        assert agent._llm.provider.client.api_key == "roundtrip-key"


def test_all_fields_together(tmp_path, monkeypatch):
    monkeypatch.setenv(FIELD_VALUES["api_key_env"], "roundtrip-key")
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "prompt.j2").write_text("{{ instruction }}")
    # max_thinking_tokens and reasoning_effort are mutually exclusive.
    values = {k: v for k, v in FIELD_VALUES.items() if k != "max_thinking_tokens"}
    kwargs = task_agent_kwargs(write_task(task_dir, values), MODEL)
    agent = make_agent(kwargs, tmp_path / "logs")
    for field, value in values.items():
        if field == "prompt_template_path":
            value = str((task_dir / value).resolve())
        assert getattr(agent.config, field) == value, field


# --- Bad config fails before any sandbox -------------------------------------


@pytest.mark.parametrize(
    "toml",
    [
        pytest.param("[agent.proximus]\nmax_turn = 3\n", id="typo-key"),
        pytest.param('[agent.proximus]\nmax_turns = "three"\n', id="wrong-type-int"),
        pytest.param(
            '[agent.proximus]\nenable_summarize = "nah"\n', id="wrong-type-bool"
        ),
        pytest.param("[agent.proximus]\nllm_call_kwargs = 5\n", id="wrong-type-dict"),
        pytest.param(
            '[agent.proximus]\ndisallowed_tools = "bash"\n', id="wrong-type-list"
        ),
        pytest.param("[agent.proximus]\nmax_turns = 0\n", id="max-turns-below-1"),
        pytest.param(
            '[agent.proximus]\nenable_view_image = true\ndisallowed_tools = ["view_image"]\n',
            id="contradictory-image-options",
        ),
        pytest.param(
            '[agent.proximus]\nenable_view_image = "nah"\n', id="bad-image-option"
        ),
        pytest.param(
            "[agent.proximus]\ncompaction_threshold = 1.5\n", id="threshold-gt-1"
        ),
        pytest.param(
            "[agent.proximus]\nmax_thinking_tokens = 512\n", id="thinking-lt-1024"
        ),
        pytest.param('[agent.proximus]\nmodels = "x"\n', id="models-string"),
        pytest.param("[agent.proximus]\nmodels = [1]\n", id="models-array"),
        pytest.param("[agent.proximus]\nmodels = []\n", id="models-empty-array"),
        pytest.param('[agent.proximus]\nmodels = ""\n', id="models-empty-string"),
        pytest.param(
            f'[agent.proximus]\nmodels = {{ "{MODEL}" = 3 }}\n',
            id="model-entry-not-table",
        ),
        pytest.param(
            f'[agent.proximus.models."{MODEL}"]\nreasoning_efort = "max"\n',
            id="typo-in-model-table",
        ),
    ],
)
def test_bad_task_config_raises_value_error(tmp_path, toml):
    (tmp_path / "task.toml").write_text(toml)
    with pytest.raises(ValueError):
        task_agent_kwargs(tmp_path, MODEL)


def test_bash_or_submit_in_disallowed_tools_fails_at_construction(tmp_path):
    for tool in ("bash", "submit"):
        write_task(tmp_path, {"disallowed_tools": [tool]})
        kwargs = task_agent_kwargs(tmp_path, MODEL)
        with pytest.raises(ValueError, match="cannot remove bash or submit"):
            make_agent(kwargs, tmp_path / "logs")


def test_unknown_disallowed_tool_fails_at_construction(tmp_path):
    write_task(tmp_path, {"disallowed_tools": ["rm_rf"]})
    kwargs = task_agent_kwargs(tmp_path, MODEL)
    with pytest.raises(ValueError, match="unknown tools"):
        make_agent(kwargs, tmp_path / "logs")


def test_missing_template_fails_before_any_sandbox(tmp_path):
    write_task(tmp_path, {"prompt_template_path": "missing.j2"})
    with pytest.raises(ValueError, match="prompt_template_path not found"):
        task_agent_kwargs(tmp_path, MODEL)
    with pytest.raises(ValueError, match="prompt_template_path not found"):
        make_agent(
            {"prompt_template_path": str(tmp_path / "missing.j2")}, tmp_path / "logs"
        )


def test_template_without_instruction_fails_at_preflight(tmp_path):
    (tmp_path / "prompt.j2").write_text("no placeholder here")
    with pytest.raises(ValueError):
        Proximus.parse_options({"prompt_template_path": str(tmp_path / "prompt.j2")})


def test_thinking_controls_are_exclusive(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        make_agent(
            {"max_thinking_tokens": 2048, "reasoning_effort": "high"}, tmp_path / "logs"
        )


@pytest.mark.parametrize(
    "kwargs",
    [{"max_turn": 3}, {"max_turns": "three"}, {"compaction_threshold": 0}],
)
def test_bad_kwargs_fail_at_construction(tmp_path, kwargs):
    with pytest.raises(ValueError):
        make_agent(kwargs, tmp_path / "logs")


# --- --ak parsing parity with harbor run -------------------------------------


def test_parse_kwargs_types():
    parsed = parse_kwargs(["a=1", "b=true", 'c={"x":1}', "d=hello", "e=1.5"])
    assert parsed == {"a": 1, "b": True, "c": {"x": 1}, "d": "hello", "e": 1.5}
    assert type(parsed["a"]) is int
    assert type(parsed["b"]) is bool
    assert type(parsed["c"]) is dict
    assert type(parsed["d"]) is str
    assert type(parsed["e"]) is float


def test_ak_values_validate_with_right_types(tmp_path):
    parsed = parse_kwargs(
        [
            "max_turns=3",
            "enable_summarize=false",
            'llm_call_kwargs={"timeout":5}',
            "reasoning_effort=high",
            "temperature=0.2",
            "time_budget_sec=600",
            'workspace_git_excludes=["node_modules"]',
            "api_base=None",
        ]
    )
    config = ProximusConfig.model_validate(parsed)
    assert config.max_turns == 3 and type(config.max_turns) is int
    assert config.enable_summarize is False
    assert config.llm_call_kwargs == {"timeout": 5}
    assert config.reasoning_effort == "high"
    assert config.temperature == 0.2
    assert config.time_budget_sec == 600.0
    assert config.workspace_git_excludes == ["node_modules"]
    assert config.api_base is None

    # --ak lands in AgentConfig.kwargs as task_agent_kwargs overrides.
    (tmp_path / "task.toml").write_text("[agent.proximus]\nmax_turns = 9\n")
    kwargs = task_agent_kwargs(tmp_path, MODEL, overrides=parsed)
    agent = make_agent(kwargs, tmp_path / "logs")
    assert agent.config.max_turns == 3
    assert agent.config.enable_summarize is False


# --- workspace_git_excludes defaulting on real FrontierSWE v2 tasks ----------


def frontier_swe_tasks() -> list[Path]:
    root = Path(FRONTIER_SWE_TASKS) if FRONTIER_SWE_TASKS else None
    if root is None or not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if (p / "task.toml").is_file())


@pytest.mark.skipif(not frontier_swe_tasks(), reason="set FRONTIER_SWE_TASKS to run")
@pytest.mark.parametrize("task_dir", frontier_swe_tasks(), ids=lambda p: p.name)
def test_frontier_swe_workspace_git_excludes_are_git_wildmatch(task_dir):
    kwargs = task_agent_kwargs(task_dir, MODEL)
    for pattern in kwargs.get("workspace_git_excludes", []):
        assert pattern and not pattern.startswith("./"), pattern
        assert not pattern.endswith("/**"), pattern
    ProximusConfig.model_validate(kwargs)

    override = task_agent_kwargs(
        task_dir, MODEL, overrides={"workspace_git_excludes": ["only-this"]}
    )
    assert override["workspace_git_excludes"] == ["only-this"]
