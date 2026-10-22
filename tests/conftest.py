"""Keep the compatibility suite offline and independent of local credentials."""

import socket

import pytest

from harbor_proximus import model_info

TEST_MODELS = [
    "openai/gpt-test",
    "openai/test-model",
    "openai/claude-custom",
    "openai/Claude-custom",
    "anthropic/claude-test",
    "anthropic/claude-opus-test",
    "anthropic/test-model",
]


@pytest.fixture(autouse=True)
def fake_model_limits(monkeypatch):
    """Give test-only models explicit limits without changing production lookup."""
    table = {
        **model_info.builtin_table(),
        **{name: {"max_input_tokens": 200_000} for name in TEST_MODELS},
    }
    monkeypatch.setattr(model_info, "builtin_table", lambda: table)


@pytest.fixture(autouse=True)
def fake_api_keys(monkeypatch):
    """SDK clients are built with the agent; the network is blocked anyway."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network access is forbidden in the offline test suite")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
