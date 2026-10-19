"""Context capacity has one source: the registry's max_input_tokens."""

import litellm
import pytest

from harbor_proximus import Proximus


@pytest.mark.parametrize("output_limit", [1024, 500_000])
def test_context_uses_only_input_limit(tmp_path, monkeypatch, output_limit):
    monkeypatch.setattr(
        litellm,
        "get_model_info",
        lambda model: {"max_input_tokens": 32000, "max_tokens": output_limit},
    )
    agent = Proximus(logs_dir=tmp_path, model_name="openai/context-test")
    assert agent._context_limit == 32000


@pytest.mark.parametrize("limit", [None, 0, -1, True, False, 1.5, "32000"])
def test_invalid_input_limit_fails_at_construction(tmp_path, monkeypatch, limit):
    monkeypatch.setattr(
        litellm,
        "get_model_info",
        lambda model: {"max_input_tokens": limit, "max_tokens": 200_000},
    )
    with pytest.raises(
        ValueError, match=r"No valid max_input_tokens for 'openai/context-test'\."
    ):
        Proximus(logs_dir=tmp_path, model_name="openai/context-test")


def test_missing_input_limit_does_not_use_max_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(
        litellm, "get_model_info", lambda model: {"max_tokens": 200_000}
    )
    with pytest.raises(
        ValueError, match=r"No valid max_input_tokens for 'openai/context-test'\."
    ):
        Proximus(logs_dir=tmp_path, model_name="openai/context-test")


def test_registry_error_identifies_model_and_preserves_cause(tmp_path, monkeypatch):
    error = RuntimeError("registry lookup failed")

    def failed_lookup(model):
        raise error

    monkeypatch.setattr(litellm, "get_model_info", failed_lookup)
    with pytest.raises(
        ValueError, match=r"No valid max_input_tokens for 'openai/context-test'\."
    ) as exc:
        Proximus(logs_dir=tmp_path, model_name="openai/context-test")
    assert exc.value.__cause__ is error


def test_unknown_model_requires_explicit_input_limit(tmp_path):
    model = "openai/proximus-unregistered-context-test"
    with pytest.raises(ValueError, match="No valid max_input_tokens") as exc:
        Proximus(logs_dir=tmp_path, model_name=model)
    assert model in str(exc.value)

    agent = Proximus(
        logs_dir=tmp_path,
        model_name=model,
        model_info={"max_input_tokens": 64000},
    )
    assert agent._context_limit == 64000
