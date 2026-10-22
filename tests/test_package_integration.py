"""Exercise the external-agent contract against unmodified upstream Harbor."""

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import PurePosixPath

import pytest
from harbor.agents.factory import AgentFactory
from harbor.models.agent.context import AgentContext
from harbor.models.trial.config import AgentConfig

from harbor_proximus import Proximus, utils
from tests.test_proximus_loop import FakeCompletion, FakeEnv, finish_responses


@pytest.mark.parametrize(
    "selector",
    [
        {"import_path": "harbor_proximus:Proximus"},
        {"name": "harbor_proximus:Proximus"},
    ],
)
async def test_factory_run_and_artifacts(tmp_path, monkeypatch, selector):
    assert importlib.metadata.version("harbor") == "0.24.0"
    fake = FakeCompletion(finish_responses())
    fake.install(monkeypatch)
    config = AgentConfig(
        **selector,
        model_name="openai/test-model",
        override_setup_timeout_sec=30,
        kwargs={"time_budget_sec": 600, "task_provides_timer": True},
    )
    agent = AgentFactory.create_agent_from_config(
        config,
        logs_dir=tmp_path / "logs",
        environment_logs_dir=PurePosixPath("/logs/custom-agent"),
    )
    assert isinstance(agent, Proximus)
    assert agent.config.time_budget_sec == 600
    assert agent.environment_logs_dir == PurePosixPath("/logs/custom-agent")
    assert agent.options is agent.config
    env = FakeEnv()
    context = AgentContext()
    await agent.setup(env)
    await agent.run("Implement the task.", env, context)

    assert len(fake.calls) == 2
    records = [
        json.loads(line)
        for line in (agent.logs_dir / "submissions.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["ok"] is True
    assert context.metadata["ended_by_submit"] is True
    assert context.metadata["submitted_commit"] == records[0]["hash"]
    assert any(values.get("winner") == records[0]["hash"] for values in env.all_envs)
    assert (agent.logs_dir / "messages.json").is_file()
    trajectory = json.loads((agent.logs_dir / "trajectory.json").read_text())
    assert trajectory["agent"]["version"] == importlib.metadata.version(
        "harbor-proximus"
    )
    assert agent.capabilities.atif


def test_preflight_validates_agent_options():
    Proximus.preflight({"time_budget_sec": 600})
    with pytest.raises(ValueError, match="Unknown option 'unsupported'"):
        Proximus.preflight({"unsupported": True})
    schema = Proximus.options_schema()
    assert "time_budget_sec" in schema["properties"]


def test_all_shell_and_prompt_resources_present():
    assert utils.read_template("system.md").strip()
    for name in [
        "ensure_git.sh",
        "record_commit.sh",
        "matches_winner.sh",
        "restore_winner.sh",
        "reanchor_timer.sh",
    ]:
        assert utils.read_script(name).strip()


def test_user_registry_controls_context_and_vision(tmp_path):
    model = "openai/proximus-test-vision"
    entry = {
        "max_input_tokens": 123456,
        "max_output_tokens": 4096,
        "supports_vision": True,
        "supports_function_calling": True,
        "litellm_provider": "openai",
        "mode": "chat",
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000002,
    }
    agent = Proximus(
        logs_dir=tmp_path,
        model_name=model,
        # The pre-1.0 option name, still accepted for existing job configs.
        litellm_model_info={model: entry},
        reasoning_effort="high",
        enable_view_image=True,
    )
    assert agent._context_limit == 123456
    assert "view_image" in agent._tools
    assert agent.config.reasoning_effort == "high"


def test_host_credentials_from_harbor_agent_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PROXIMUS_TEST_KEY", "host-key")
    agent = AgentFactory.create_agent_from_config(
        AgentConfig(
            import_path="harbor_proximus:Proximus",
            model_name="openai/test-model",
            env={"PROXIMUS_TEST_KEY": "configured-key"},
            kwargs={"api_key_env": "PROXIMUS_TEST_KEY"},
        ),
        logs_dir=tmp_path,
    )
    assert agent._llm.provider.client.api_key == "configured-key"
    assert "configured-key" not in agent.config.model_dump_json()


def test_provider_credentials_follow_harbor_agent_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "host-key")

    def make(env):
        return AgentFactory.create_agent_from_config(
            AgentConfig(
                import_path="harbor_proximus:Proximus",
                model_name="anthropic/test-model",
                env=env,
            ),
            logs_dir=tmp_path,
        )

    # Host credentials are used but not blanked: the sandbox never had them.
    host = make({})._llm
    assert host.provider.client.api_key == "host-key"
    assert host.consumed_env_keys == frozenset()
    # Anthropic's SDK knows its endpoint; Harbor's default would double /v1.
    assert str(host.provider.client.base_url).rstrip("/") == "https://api.anthropic.com"
    configured = make(
        {"ANTHROPIC_API_KEY": "agent-key", "ANTHROPIC_BASE_URL": "https://proxy"}
    )
    assert configured._llm.provider.client.api_key == "agent-key"
    assert str(configured._llm.provider.client.base_url).rstrip("/") == "https://proxy"


def test_options_are_described_in_the_harbor_schema():
    fields = Proximus.options_model.model_json_schema()["properties"]
    assert all(field.get("description") for field in fields.values())


def test_import_and_construction_are_offline_and_never_load_litellm(tmp_path):
    # Harbor still declares litellm; Proximus must never import it, even
    # where it is installed.
    script = """
import socket, sys
from pathlib import Path
def blocked(*args, **kwargs):
    raise AssertionError('Unexpected network access')
socket.socket.connect = blocked
socket.getaddrinfo = blocked
from harbor_proximus import Proximus
for model in ('anthropic/claude-opus-5-5', 'anthropic/claude-haiku-5-5', 'openai/gpt-5.6-sol', 'openai/gpt-6-astra'):
    Proximus(logs_dir=Path(sys.argv[1]), model_name=model, reasoning_effort='high')
assert 'litellm' not in sys.modules, 'litellm was imported'
assert Proximus.name() == 'proximus'
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "logs")],
        cwd=tmp_path,
        env=dict(os.environ),
        check=True,
        timeout=30,
        capture_output=True,
        text=True,
    )


def _agent(tmp_path, model, env, **kwargs):
    return AgentFactory.create_agent_from_config(
        AgentConfig(
            import_path="harbor_proximus:Proximus",
            model_name=model,
            env=env,
            kwargs=kwargs,
        ),
        logs_dir=tmp_path,
    )


def test_consumed_credentials_are_reported_for_blanking(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    agent = _agent(
        tmp_path,
        "openai/test-model",
        {"OPENAI_API_KEY": "k", "OPENAI_BASE_URL": "https://gw", "HF_TOKEN": "keep"},
    )
    assert agent._llm.consumed_env_keys == {"OPENAI_API_KEY", "OPENAI_BASE_URL"}
    named = _agent(
        tmp_path,
        "openai/test-model",
        {"MY_KEY": "k2", "OPENAI_BASE_URL": "https://gw"},
        api_key_env="MY_KEY",
    )
    # api_key_env no longer disables agent-env base URLs.
    assert named._llm.provider.client.api_key == "k2"
    assert str(named._llm.provider.client.base_url).rstrip("/") == "https://gw"
    assert named._llm.consumed_env_keys == {"MY_KEY", "OPENAI_BASE_URL"}


def test_routes_without_an_endpoint_are_rejected_up_front(tmp_path):
    # Bedrock/Vertex/SageMaker/Azure need cloud SDK auth, not an API key.
    with pytest.raises(ValueError, match="api_base"):
        _agent(tmp_path, "bedrock/test-model", {"AWS_ACCESS_KEY_ID": "AKIATEST"})


def test_bare_model_names_are_rejected(tmp_path):
    # Routing never guesses a provider (Harbor would ask litellm).
    with pytest.raises(ValueError, match="provider prefix"):
        _agent(tmp_path, "gpt-4o", {})


async def test_run_blanks_consumed_credentials_in_the_sandbox(tmp_path, monkeypatch):
    import contextlib

    fake = FakeCompletion(finish_responses())
    fake.install(monkeypatch)
    scopes = []

    class ScopedEnv(FakeEnv):
        @contextlib.contextmanager
        def scoped_exec_env(self, env):
            scopes.append(dict(env))
            yield

    agent = _agent(tmp_path, "openai/test-model", {"OPENAI_API_KEY": "sk-secret"})
    await agent.run("Task", ScopedEnv(), AgentContext())  # type: ignore[arg-type]
    assert scopes == [{"OPENAI_API_KEY": ""}]


def test_model_info_stays_with_its_agent(tmp_path):
    # Overrides are per agent: another run of the bare name keeps the table.
    custom = _agent(
        tmp_path,
        "openrouter/openai/gpt-5.6-sol",
        {"OPENROUTER_API_KEY": "k"},
        model_info={"max_input_tokens": 1234},
    )
    assert custom._context_limit == 1234
    plain = _agent(tmp_path, "openai/gpt-5.6-sol", {"OPENAI_API_KEY": "k"})
    assert plain._context_limit == 922_000


def test_resolve_credentials_unit():
    from harbor_proximus.llm import resolve_credentials

    env = {"OPENAI_API_KEY": "k", "OPENAI_BASE_URL": "https://gw"}
    plain = resolve_credentials("openai/m", None, env)
    assert (plain.api_key, plain.api_base) == ("k", "https://gw")
    assert plain.consumed_env_keys == {"OPENAI_API_KEY", "OPENAI_BASE_URL"}
    named = resolve_credentials("openai/m", "MY_KEY", {**env, "MY_KEY": "k2"})
    assert named.api_key == "k2"
    # The unused default key is hidden too: blanking more is safer.
    assert named.consumed_env_keys == {"MY_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"}
    with pytest.raises(ValueError, match="MY_MISSING"):
        resolve_credentials("openai/m", "MY_MISSING", {})
