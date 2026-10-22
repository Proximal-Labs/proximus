"""Model metadata: context limits, vision support and prices.

The built-in table (``model_info.json``) lists the models Proximus is
evaluated with, keyed ``route/model``, with values from each vendor's model
and pricing pages. Config entries (``model_info``, ``extra_model_info``) use
the same keys and win over it; any other model needs ``model_info``.
"""

import json
import re
from functools import cache
from importlib.resources import files
from typing import Any

type ModelInfo = dict[str, Any]


@cache
def builtin_table() -> dict[str, ModelInfo]:
    text = files("harbor_proximus").joinpath("model_info.json").read_text()
    return json.loads(text)


class ModelRegistry:
    """Look up model metadata by the run's provider-prefixed model name."""

    def __init__(self, overrides: dict[str, ModelInfo] | None = None):
        self._overrides = dict(overrides or {})

    def lookup(self, model_name: str) -> ModelInfo | None:
        """Overrides by full name, then the table, then overrides by the bare
        name (responses echo the model id without its route)."""
        if model_name in self._overrides:
            return self._overrides[model_name]
        if model_name in builtin_table():
            return builtin_table()[model_name]
        return self._overrides.get(model_name.split("/", 1)[-1])


def _rate(info: ModelInfo, key: str, prompt_tokens: int) -> float | None:
    """The per-token rate for ``key`` at this prompt size: the highest
    ``<key>_above_<N>k_tokens`` tier the prompt exceeds, else the base rate."""
    tiers = sorted(
        (int(match.group(1)) * 1000, value)
        for name, value in info.items()
        if (match := re.fullmatch(re.escape(key) + r"_above_(\d+)k_tokens", name))
    )
    rate = info.get(key)
    for threshold, value in tiers:
        if prompt_tokens > threshold:
            rate = value
    return None if rate is None else float(rate)


def completion_cost(
    info: ModelInfo | None,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_1h_tokens: int = 0,
) -> float:
    """USD cost of one response. ``input_tokens`` excludes cache reads and
    writes; unpriced models cost 0."""
    if not info:
        return 0.0
    prompt = input_tokens + cache_read_tokens + cache_write_1h_tokens
    input_rate = _rate(info, "input_cost_per_token", prompt) or 0.0
    read_rate = _rate(info, "cache_read_input_token_cost", prompt)
    write_rate = _rate(info, "cache_creation_input_token_cost_above_1hr", prompt)
    return (
        input_tokens * input_rate
        + output_tokens * (_rate(info, "output_cost_per_token", prompt) or 0.0)
        + cache_read_tokens * (input_rate if read_rate is None else read_rate)
        + cache_write_1h_tokens * (input_rate if write_rate is None else write_rate)
    )
