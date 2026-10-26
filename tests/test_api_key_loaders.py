"""``api_key_loader``: a loader is named by a built-in name or ``module:Class``,
built from the agent env before any sandbox starts, and asked for a key before
every model request, retries included."""

import logging

import litellm
import pytest

from harbor_proximus.auth import (
    ApiKeyLoader,
    CredentialError,
    Env,
    loader_class,
    resolve_credentials,
)
from harbor_proximus.auth.anthropic_wif import AnthropicWIF
from harbor_proximus.config import ProximusConfig
from harbor_proximus.llm import LLM


class CountingLoader(ApiKeyLoader):
    """A custom loader: a new key for every request."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.loads = 0

    @classmethod
    def from_env(cls, model_name: str, env: Env) -> "CountingLoader":
        return cls(env.require("TEST_KEY_PREFIX"))

    async def load(self) -> str:
        self.loads += 1
        return f"{self.prefix}-{self.loads}"


class FlakyLoader(ApiKeyLoader):
    def __init__(self, retryable: bool):
        self.retryable = retryable
        self.loads = 0

    @classmethod
    def from_env(cls, model_name: str, env: Env) -> "FlakyLoader":
        return cls(retryable=True)

    async def load(self) -> str:
        self.loads += 1
        if self.loads == 1:
            raise CredentialError("identity service down", retryable=self.retryable)
        return "recovered-key"


SPEC = f"{__name__}:CountingLoader"
MODEL = "openai/gpt-test"


def make_llm(loader_spec: str = SPEC) -> LLM:
    return LLM(
        MODEL,
        ProximusConfig(api_key_loader=loader_spec),
        remaining_sec=lambda: None,
        logger=logging.getLogger("t"),
        extra_env={"TEST_KEY_PREFIX": "k", "OPENAI_API_KEY": "sk-unused"},
    )


def test_builtin_names_and_import_paths_resolve():
    assert loader_class("anthropic_wif") is AnthropicWIF
    assert loader_class(SPEC) is CountingLoader


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("nope", "neither a built-in loader"),
        ("os:path", "not an ApiKeyLoader subclass"),
        ("no_such_module_for_proximus:Loader", "cannot be imported"),
        ("harbor_proximus.auth:Missing", "cannot be imported"),
    ],
)
def test_bad_loader_fails_in_config(spec, message):
    with pytest.raises(ValueError, match=message):
        ProximusConfig(api_key_loader=spec)


def test_api_key_env_and_loader_are_exclusive(monkeypatch):
    with pytest.raises(ValueError, match="not both"):
        ProximusConfig(api_key_env="MY_KEY", api_key_loader=SPEC)


def test_loader_settings_are_hidden_and_the_static_key_is_unused():
    credentials = resolve_credentials(
        MODEL,
        None,
        {"TEST_KEY_PREFIX": "k", "OPENAI_API_KEY": "sk-unused"},
        api_key_loader=SPEC,
    )
    assert credentials.api_key is None
    assert isinstance(credentials.api_key_loader, CountingLoader)
    assert credentials.api_key_loader.prefix == "k"
    assert credentials.consumed_env_keys == {"TEST_KEY_PREFIX", "OPENAI_API_KEY"}


def test_env_prefers_the_agent_env_and_treats_empty_as_unset(monkeypatch):
    monkeypatch.setenv("SETTING", "host")
    monkeypatch.setenv("BLANK", "")
    env = Env({"SETTING": "agent"})
    assert env.get("SETTING") == "agent"
    assert env.get("BLANK") is None
    assert env.consumed == {"SETTING"}
    with pytest.raises(ValueError, match="BLANK is not set"):
        env.require("BLANK")


async def test_a_key_is_loaded_for_every_attempt(monkeypatch):
    original = litellm.acompletion
    keys: list[str] = []

    async def fake_acompletion(**kwargs):
        keys.append(kwargs["api_key"])
        if len(keys) == 2:
            raise litellm.exceptions.APIConnectionError("blip", "openai", "gpt-test")
        return await original(**kwargs, mock_response="ok")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)
    llm = make_llm()

    await llm.query([{"role": "user", "content": "hi"}], tools=None)
    await llm.query([{"role": "user", "content": "hi"}], tools=None)

    # The second query's first attempt failed; its retry loaded a new key.
    assert keys == ["k-1", "k-2", "k-3"]


@pytest.mark.parametrize(("retryable", "loads"), [(True, 2), (False, 1)])
async def test_loader_errors_follow_their_retryable_flag(monkeypatch, retryable, loads):
    async def fake_acompletion(**kwargs):
        return await original(**kwargs, mock_response="ok")

    original = litellm.acompletion
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr("tenacity.wait_random_exponential", lambda **_: 0)
    llm = make_llm()
    loader = FlakyLoader(retryable)
    llm._api_key_loader = loader

    if retryable:
        parsed = await llm.query([{"role": "user", "content": "hi"}], tools=None)
        assert parsed.content == "ok"
    else:
        with pytest.raises(CredentialError):
            await llm.query([{"role": "user", "content": "hi"}], tools=None)
    assert loader.loads == loads
