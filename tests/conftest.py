"""Keep the compatibility suite offline and independent of local credentials."""

import os
import socket

import pytest

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


@pytest.fixture(autouse=True)
def fake_model_limits(monkeypatch):
    """Give test-only models explicit limits without changing production lookup."""
    import copy

    import litellm

    monkeypatch.setattr(litellm, "model_cost", copy.deepcopy(litellm.model_cost))
    models = [
        "openai/gpt-test",
        "openai/test-model",
        "openai/claude-custom",
        "openai/Claude-custom",
        "anthropic/claude-test",
        "anthropic/claude-opus-test",
        "anthropic/test-model",
        "bedrock/test-model",
        "gemini/test-model",
    ]
    litellm.register_model(
        {
            model: {
                "max_input_tokens": 200_000,
                "litellm_provider": model.split("/", 1)[0],
            }
            for model in models
        }
    )


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network access is forbidden in the offline test suite")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
