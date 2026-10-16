"""Proximus, a long-horizon coding agent for Harbor."""

from importlib import import_module
from typing import Any

# Exported lazily, so importing the package stays cheap and offline.
_EXPORTS = {
    "Proximus": "proximus",
    "task_agent_kwargs": "task_config",
    "hard_timeout_sec": "task_config",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
