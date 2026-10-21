"""Proximus settings declared by a task, under ``[agent.proximus]`` in task.toml.

Upstream Harbor ignores unknown keys in ``[agent]`` and does not give a custom
agent its task directory, so the runner reads this table and passes the result
as ``AgentConfig.kwargs``::

    [agent]
    timeout_sec = 72000.0

    [agent.proximus]
    task_provides_timer = true

    [agent.proximus.models."anthropic/claude-opus-5"]
    reasoning_effort = "max"
    api_key_env = "ANTHROPIC_API_KEY"

Layers, later wins: the caller's defaults, ``[agent.proximus]``, the table for
the run's model under ``models``, then the caller's overrides. Nested tables such as
``llm_call_kwargs`` merge per key, so overriding one ``extra_body`` field keeps
the rest. ``time_budget_sec`` defaults to ``[agent].timeout_sec``, a relative
``prompt_template_path`` is resolved against the task directory, and
``workspace_git_excludes`` defaults to the excludes of the task's ``/app``
artifact, so submissions never commit what grading would drop anyway.
"""

import tomllib
from pathlib import Path
from typing import Any

from harbor_proximus.config import ProximusConfig
from harbor_proximus.submissions import (
    DEFAULT_SUBMISSION_TIMEOUT_SEC,
    RESTORE_TIMEOUT_SEC,
)

# Cleanup slack, also used when there is no Git finalization.
FINALIZE_MARGIN_SEC = 300

TABLE = "proximus"


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _normalize_exclude(pattern: str) -> str:
    """Git-wildmatch form of an artifact exclude; "" for whole-workspace forms.
    A leading ``./`` never matches in git, and ``dir/**`` is ``dir``."""
    while pattern.startswith("./"):
        pattern = pattern[2:]
    while pattern.endswith("/**"):
        pattern = pattern[: -len("/**")]
    pattern = pattern.rstrip("/")
    return "" if pattern in ("", ".", "*", "**") else pattern


def _app_artifact_excludes(data: dict[str, Any]) -> list[str]:
    for entry in data.get("artifacts") or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("source", "")).rstrip("/") != "/app":
            continue
        excludes: list[str] = []
        patterns = entry.get("exclude") or []
        for pattern in patterns if isinstance(patterns, list) else []:
            normalized = _normalize_exclude(str(pattern))
            if normalized and normalized not in excludes:
                excludes.append(normalized)
        return excludes
    return []


def task_agent_kwargs(
    task_dir: Path,
    model_name: str,
    overrides: dict[str, Any] | None = None,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Proximus kwargs for running ``model_name`` on the task in ``task_dir``.

    Only explicitly set options are returned; everything else keeps the
    ProximusConfig default. Raises ValueError on an unknown or invalid option.
    """
    data = tomllib.loads((Path(task_dir) / "task.toml").read_text())
    agent = data.get("agent") or {}
    if not isinstance(agent.get(TABLE, {}), dict):
        raise ValueError(f"[agent.{TABLE}] must be a table")
    table = dict(agent.get(TABLE) or {})
    models = table.pop("models", {})
    if not isinstance(models, dict):
        raise ValueError(f"[agent.{TABLE}.models] must be a table of model names")
    model_table = models.get(model_name, {})
    if not isinstance(model_table, dict):
        raise ValueError(f'[agent.{TABLE}.models."{model_name}"] must be a table')

    # Relative template paths in the task's own tables are task-relative.
    for layer in (table, model_table):
        template = layer.get("prompt_template_path")
        if isinstance(template, str) and not Path(template).is_absolute():
            layer["prompt_template_path"] = str((Path(task_dir) / template).resolve())
    kwargs = _merge(defaults or {}, table)
    kwargs = _merge(kwargs, dict(model_table))
    kwargs = _merge(kwargs, overrides or {})
    if "time_budget_sec" not in kwargs and agent.get("timeout_sec") is not None:
        kwargs["time_budget_sec"] = float(agent["timeout_sec"])
    if "workspace_git_excludes" not in kwargs:
        excludes = _app_artifact_excludes(data)
        if excludes:
            kwargs["workspace_git_excludes"] = excludes
    try:
        ProximusConfig.model_validate(kwargs)
    except Exception as e:
        raise ValueError(f"invalid [agent.{TABLE}] in {task_dir}: {e}") from e
    return kwargs


def hard_timeout_sec(kwargs: dict[str, Any]) -> float | None:
    """Harbor's hard agent timeout (``AgentConfig.override_timeout_sec``) for
    these kwargs: the work budget plus room for the exit auto-submit and
    restore, so Harbor's kill never cuts finalization short. Git-free mode
    needs only the cleanup margin. None without a budget."""
    budget = kwargs.get("time_budget_sec")
    if budget is None:
        return None
    if kwargs.get("submission_mode", "git") == "none":
        return float(budget) + FINALIZE_MARGIN_SEC
    submission = kwargs.get("submission_timeout_sec", DEFAULT_SUBMISSION_TIMEOUT_SEC)
    return float(budget) + float(submission) + RESTORE_TIMEOUT_SEC + FINALIZE_MARGIN_SEC
