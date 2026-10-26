"""A model's API key and base URL, as Harbor's own agents resolve them."""

import os
from dataclasses import dataclass

from harbor.agents.model_connection import (
    PROVIDERS,
    ModelConnectionSpec,
    resolve_model_connection,
)

from harbor_proximus.auth.loader import ApiKeyLoader, Env, loader_class

# Providers whose credential in Harbor's PROVIDERS table is not a bearer API key.
_NON_BEARER_PROVIDERS = frozenset({"amazon-bedrock", "sagemaker", "vertex_ai"})
assert _NON_BEARER_PROVIDERS <= set(PROVIDERS)


@dataclass(frozen=True)
class Credentials:
    api_key: str | None
    api_base: str | None
    consumed_env_keys: frozenset[str]
    """Agent-env variables used for the model, to hide from the sandbox."""
    api_key_loader: ApiKeyLoader | None = None
    """Loads the key for each request; ``api_key`` is then None."""


def resolve_credentials(
    model_name: str,
    api_key_env: str | None,
    extra_env: dict[str, str] | None,
    *,
    api_key_loader: str | None = None,
) -> Credentials:
    """The model's API key and base URL, as Harbor's own agents resolve them.

    Harbor's agent env (`--ae`, AgentConfig.env) wins over the host, with the
    provider's usual variable names; host variables stay with LiteLLM's own
    lookup. ``api_key_env`` names the key variable explicitly (agent env, then
    host) and must be set. Bedrock/SageMaker/Vertex "keys" in Harbor's table
    are not bearer tokens, so those providers keep LiteLLM's lookup.

    ``api_key_loader`` names an :class:`ApiKeyLoader` that loads the key for
    each request instead. A key in the agent env is then still hidden from the
    sandbox, but never used.
    """
    agent_env = extra_env or {}
    used: list[str] = []

    def from_agent_env(*names: str) -> tuple[str, str] | None:
        for name in names:
            if name in agent_env:
                used.append(name)
                return name, agent_env[name]
        return None

    connection = resolve_model_connection(
        model_name, ModelConnectionSpec(), from_agent_env
    )
    cloud_auth = connection.provider in _NON_BEARER_PROVIDERS
    consumed = {
        name for name in used if not cloud_auth or name.endswith(("_BASE", "_URL"))
    }
    if api_key_loader:
        env = Env(agent_env)
        loader = loader_class(api_key_loader).from_env(model_name, env)
        return Credentials(
            None,
            connection.configured_base_url,
            frozenset(consumed | env.consumed),
            loader,
        )
    api_key = None if cloud_auth else connection.api_key or None
    if api_key_env:
        api_key = agent_env.get(api_key_env, os.environ.get(api_key_env)) or None
        if api_key is None:
            raise ValueError(
                f"api_key_env={api_key_env!r} is not set in the "
                "environment — the run cannot authenticate this model."
            )
        if api_key_env in agent_env:
            consumed.add(api_key_env)
    return Credentials(api_key, connection.configured_base_url, frozenset(consumed))
