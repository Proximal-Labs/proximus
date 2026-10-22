"""Context capacity has one source: the model table's max_input_tokens."""

import pytest

from harbor_proximus import Proximus


@pytest.mark.parametrize("output_limit", [1024, 500_000])
def test_context_uses_only_input_limit(tmp_path, output_limit):
    agent = Proximus(
        logs_dir=tmp_path,
        model_name="openai/context-test",
        model_info={"max_input_tokens": 32000, "max_output_tokens": output_limit},
    )
    assert agent._context_limit == 32000


@pytest.mark.parametrize("limit", [None, 0, -1, True, False, 1.5, "32000"])
def test_invalid_input_limit_fails_at_construction(tmp_path, limit):
    with pytest.raises(
        ValueError, match=r"No valid max_input_tokens for 'openai/context-test'\."
    ):
        Proximus(
            logs_dir=tmp_path,
            model_name="openai/context-test",
            model_info={"max_input_tokens": limit, "max_output_tokens": 200_000},
        )


def test_missing_input_limit_does_not_use_output_limit(tmp_path):
    with pytest.raises(
        ValueError, match=r"No valid max_input_tokens for 'openai/context-test'\."
    ):
        Proximus(
            logs_dir=tmp_path,
            model_name="openai/context-test",
            model_info={"max_output_tokens": 200_000},
        )


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


def test_builtin_table_knows_current_models(tmp_path):
    agent = Proximus(logs_dir=tmp_path, model_name="anthropic/claude-opus-5-5")
    assert agent._context_limit == 1_000_000
