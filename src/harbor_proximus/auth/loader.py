"""API key loaders, for credentials that are fetched or renewed during a run."""

import importlib
import os
from abc import ABC, abstractmethod

# Built-in loaders by name. Any other ``api_key_loader`` value is a
# ``module:Class`` import path.
BUILTIN_LOADERS = {
    "anthropic_wif": "harbor_proximus.auth.anthropic_wif:AnthropicWIF",
}


class CredentialError(RuntimeError):
    """A loader could not produce a key.

    ``retryable`` is False when a retry would only repeat the failure (a
    rejected identity, a missing permission), so the run fails at once instead
    of retrying until its budget is spent.
    """

    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


class Env:
    """Harbor's agent env (``--ae``) first, then the host, like ``api_key_env``.

    Agent-env names that are read are recorded in ``consumed`` and blanked in
    the sandbox, so the model never sees them.
    """

    def __init__(self, agent_env: dict[str, str] | None = None):
        self._agent_env = agent_env or {}
        self.consumed: set[str] = set()

    def get(self, name: str) -> str | None:
        if name in self._agent_env:
            self.consumed.add(name)
            return self._agent_env[name] or None
        return os.environ.get(name) or None

    def require(self, name: str) -> str:
        value = self.get(name)
        if value is None:
            raise ValueError(f"{name} is not set in the agent env or the host")
        return value


class ApiKeyLoader(ABC):
    """Loads a model's API key when each request is made.

    Select one with the ``api_key_loader`` option. It is built once when the
    agent is constructed, before any sandbox starts, and :meth:`load` is
    awaited before every model request, retries included, so a loader can
    renew a key that is about to expire.
    """

    @classmethod
    @abstractmethod
    def from_env(cls, model_name: str, env: Env) -> "ApiKeyLoader":
        """The loader for ``model_name``, configured through ``env``. Raise
        ValueError for a model or setup it cannot serve."""

    @abstractmethod
    async def load(self) -> str:
        """A key valid for the next request. Raise CredentialError if none
        can be loaded."""


def loader_class(spec: str) -> type[ApiKeyLoader]:
    """The loader named by ``spec``: a built-in name or ``module:Class``."""
    path = BUILTIN_LOADERS.get(spec, spec)
    module_name, _, attr = path.partition(":")
    if not module_name or not attr:
        builtin = ", ".join(sorted(BUILTIN_LOADERS))
        raise ValueError(
            f"api_key_loader={spec!r} is neither a built-in loader ({builtin}) "
            "nor a module:Class path"
        )
    try:
        cls = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"api_key_loader={spec!r} cannot be imported: {e}") from e
    if not (isinstance(cls, type) and issubclass(cls, ApiKeyLoader)):
        raise ValueError(f"api_key_loader={spec!r} is not an ApiKeyLoader subclass")
    return cls
