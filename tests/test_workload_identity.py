"""Anthropic Workload Identity Federation: the access token is exchanged from
a fresh STS identity token, renewed before it expires, shared by concurrent
callers, and sent as ``Authorization: Bearer`` with no ``x-api-key``.

STS is a fake minter, the token endpoint an ``httpx.MockTransport``, and the
Messages API a fake upstream behind ``httpx.AsyncClient.send``.
"""

import asyncio
import json
import logging
import threading
import time

import httpx
import pytest

from harbor_proximus import workload_identity
from harbor_proximus.config import ProximusConfig
from harbor_proximus.llm import LLM, _is_retryable, resolve_credentials
from harbor_proximus.workload_identity import (
    AUDIENCE,
    REFRESH_MARGIN_SEC,
    TOKEN_URL,
    AccessTokenSource,
    StsIdentityToken,
    WorkloadIdentityConfig,
    WorkloadIdentityError,
)

MODEL = "anthropic/claude-test"
HOST_KEY = "sk-ant-api03-host-key-must-not-leak"
AGENT_KEY = "sk-ant-api03-agent-key-must-not-leak"
WIF_ENV = {
    "ANTHROPIC_FEDERATION_RULE_ID": "fdrl_test",
    "ANTHROPIC_ORGANIZATION_ID": "org-test",
    "ANTHROPIC_SERVICE_ACCOUNT_ID": "svac_test",
    "ANTHROPIC_WORKSPACE_ID": "wrkspc_test",
}
CONFIG = WorkloadIdentityConfig(
    federation_rule_id="fdrl_test",
    organization_id="org-test",
    service_account_id="svac_test",
    workspace_id="wrkspc_test",
    sts_region="us-west-2",
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeSts:
    """Mints a distinct single-use identity token per call, like STS (jti)."""

    def __init__(self) -> None:
        self.minted: list[str] = []
        self._lock = threading.Lock()

    def __call__(self) -> str:
        with self._lock:
            jwt = f"header.jti-{len(self.minted)}.signature"
            self.minted.append(jwt)
            return jwt


class FakeTokenEndpoint:
    """Anthropic's /v1/oauth/token: rejects a reused assertion, as the API
    does for a replayed jti."""

    def __init__(self, expires_in: int = 600, delay_sec: float = 0.0) -> None:
        self.expires_in = expires_in
        self.delay_sec = delay_sec
        self.requests: list[httpx.Request] = []
        self.status = 200

    @property
    def assertions(self) -> list[str]:
        return [json.loads(r.content)["assertion"] for r in self.requests]

    def handler(self, request: httpx.Request) -> httpx.Response:
        time.sleep(self.delay_sec)
        assertion = json.loads(request.content)["assertion"]
        reused = assertion in self.assertions
        self.requests.append(request)
        if self.status != 200:
            error = {"type": "authentication_error", "message": "Authentication failed"}
            return httpx.Response(self.status, json={"type": "error", "error": error})
        if reused:
            return httpx.Response(401, json={"error": {"message": "jti reused"}})
        return httpx.Response(
            200,
            json={
                "access_token": f"sk-ant-oat01-test-{len(self.requests)}",
                "expires_in": self.expires_in,
                "token_type": "bearer",
            },
        )


def make_source(
    endpoint: FakeTokenEndpoint, sts: FakeSts, clock: FakeClock | None = None
) -> AccessTokenSource:
    return AccessTokenSource(
        CONFIG,
        identity_token=sts,
        transport=httpx.MockTransport(endpoint.handler),
        clock=clock or FakeClock(),
    )


async def test_exchange_sends_a_fresh_sts_token_and_the_ids():
    endpoint, sts = FakeTokenEndpoint(), FakeSts()
    token = await make_source(endpoint, sts).token()

    assert token == "sk-ant-oat01-test-1"
    (request,) = endpoint.requests
    assert str(request.url) == TOKEN_URL
    assert json.loads(request.content) == {
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "assertion": sts.minted[0],
        "federation_rule_id": "fdrl_test",
        "organization_id": "org-test",
        "service_account_id": "svac_test",
        "workspace_id": "wrkspc_test",
    }
    assert "x-api-key" not in request.headers
    assert "authorization" not in request.headers


async def test_token_is_reused_until_close_to_expiry_then_refreshed():
    endpoint, sts, clock = FakeTokenEndpoint(expires_in=600), FakeSts(), FakeClock()
    source = make_source(endpoint, sts, clock)

    first = await source.token()
    clock.now += 600 - REFRESH_MARGIN_SEC - 1  # 121 s left: still fresh
    assert await source.token() == first
    assert len(endpoint.requests) == 1

    clock.now += 2  # 119 s left: renew before it lapses
    second = await source.token()
    assert second != first
    assert len(endpoint.requests) == 2


async def test_every_exchange_uses_a_newly_minted_identity_token():
    endpoint, sts, clock = FakeTokenEndpoint(expires_in=600), FakeSts(), FakeClock()
    source = make_source(endpoint, sts, clock)

    tokens = []
    for _ in range(5):
        tokens.append(await source.token())
        clock.now += 600  # expired
    assert len(set(tokens)) == 5
    assert endpoint.assertions == sts.minted
    assert len(set(sts.minted)) == 5


async def test_concurrent_callers_share_one_refresh():
    endpoint, sts = FakeTokenEndpoint(delay_sec=0.2), FakeSts()
    source = make_source(endpoint, sts)

    tokens = await asyncio.gather(*(source.token() for _ in range(16)))

    assert set(tokens) == {"sk-ant-oat01-test-1"}
    assert len(endpoint.requests) == 1
    assert len(sts.minted) == 1


async def test_a_rejected_exchange_fails_fast_without_leaking_the_assertion():
    endpoint, sts = FakeTokenEndpoint(), FakeSts()
    endpoint.status = 401
    with pytest.raises(WorkloadIdentityError) as raised:
        await make_source(endpoint, sts).token()
    assert not raised.value.retryable
    assert not _is_retryable(raised.value)
    assert "Authentication failed" in str(raised.value)
    assert sts.minted[0] not in str(raised.value)


async def test_a_server_error_during_exchange_is_retryable():
    endpoint, sts = FakeTokenEndpoint(), FakeSts()
    endpoint.status = 503
    with pytest.raises(WorkloadIdentityError) as raised:
        await make_source(endpoint, sts).token()
    assert _is_retryable(raised.value)


class FakeStsClient:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self.error = error

    def get_web_identity_token(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {"WebIdentityToken": f"jwt-{len(self.calls)}"}


def test_sts_minter_uses_a_regional_client(monkeypatch):
    from boto3.session import Session

    client = FakeStsClient()
    created: list[dict] = []

    def fake_client(self, service, **kwargs):
        created.append({"service": service, **kwargs})
        return client

    monkeypatch.setattr(Session, "client", fake_client)
    mint = StsIdentityToken("us-west-2")

    assert [mint(), mint()] == ["jwt-1", "jwt-2"]
    assert created == [{"service": "sts", "region_name": "us-west-2"}]
    assert client.calls[0] == {
        "Audience": [AUDIENCE],
        "SigningAlgorithm": "RS256",
        "DurationSeconds": 900,
    }


def test_sts_access_denied_fails_fast(monkeypatch):
    from boto3.session import Session
    from botocore.exceptions import ClientError

    denied = ClientError(
        {
            "Error": {"Code": "AccessDenied", "Message": "not authorized"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        },
        "GetWebIdentityToken",
    )
    client = FakeStsClient(error=denied)
    monkeypatch.setattr(Session, "client", lambda *a, **k: client)

    with pytest.raises(WorkloadIdentityError) as raised:
        StsIdentityToken("us-west-2")()
    assert not _is_retryable(raised.value)


# Credential resolution


@pytest.fixture
def aws_region(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")


def test_ids_come_from_agent_env_then_host_and_api_keys_are_unused(
    monkeypatch, aws_region
):
    for name in WIF_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", HOST_KEY)
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_host")
    agent_env = {
        "ANTHROPIC_API_KEY": AGENT_KEY,
        "ANTHROPIC_FEDERATION_RULE_ID": "fdrl_agent",
        "ANTHROPIC_ORGANIZATION_ID": "org-agent",
    }

    credentials = resolve_credentials(
        MODEL, None, agent_env, anthropic_workload_identity=True
    )

    assert credentials.api_key is None
    assert credentials.federation == WorkloadIdentityConfig(
        federation_rule_id="fdrl_agent",
        organization_id="org-agent",
        service_account_id=None,
        workspace_id="wrkspc_host",
        sts_region="us-west-2",
    )
    # The unused agent-env key is still hidden from the sandbox.
    assert credentials.consumed_env_keys == {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_FEDERATION_RULE_ID",
        "ANTHROPIC_ORGANIZATION_ID",
    }


@pytest.mark.parametrize(
    ("model", "kwargs", "env", "message"),
    [
        (MODEL, {}, {}, "ANTHROPIC_FEDERATION_RULE_ID"),
        ("openai/gpt-test", {}, WIF_ENV, "anthropic/ model"),
        (MODEL, {"api_key_env": "MY_KEY"}, WIF_ENV, "not both"),
    ],
)
def test_invalid_setup_fails_at_construction(
    monkeypatch, aws_region, model, kwargs, env, message
):
    for name in WIF_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MY_KEY", HOST_KEY)
    with pytest.raises(ValueError, match=message):
        LLM(
            model,
            ProximusConfig(anthropic_workload_identity=True, **kwargs),
            remaining_sec=lambda: None,
            logger=logging.getLogger("t"),
            extra_env=env,
        )


def test_missing_aws_region_fails_at_construction(monkeypatch, tmp_path):
    for name in ("AWS_DEFAULT_REGION", "AWS_REGION", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="AWS_DEFAULT_REGION"):
        resolve_credentials(MODEL, None, WIF_ENV, anthropic_workload_identity=True)


# The request on the wire


def anthropic_response(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    message = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [],
        "stop_reason": None,
        "usage": {"input_tokens": 3, "output_tokens": 0},
    }
    if not body.get("stream"):
        message.update(
            content=[{"type": "text", "text": "ok"}],
            stop_reason="end_turn",
            usage={"input_tokens": 3, "output_tokens": 1},
        )
        return httpx.Response(200, json=message, request=request)
    events = [
        {"type": "message_start", "message": message},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "ok"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]
    text = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    return httpx.Response(
        200,
        content=text.encode(),
        headers={"content-type": "text/event-stream"},
        request=request,
    )


@pytest.fixture
def upstream(monkeypatch):
    """Every request LiteLLM sends, answered like the Messages API."""
    sent: list[httpx.Request] = []

    async def send(self, request, **kwargs):
        sent.append(request)
        return anthropic_response(request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    return sent


def wif_llm(monkeypatch, source: AccessTokenSource, **options) -> LLM:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    monkeypatch.setenv("ANTHROPIC_API_KEY", HOST_KEY)
    monkeypatch.setattr(workload_identity, "shared_source", lambda config: source)
    return LLM(
        MODEL,
        ProximusConfig(anthropic_workload_identity=True, **options),
        remaining_sec=lambda: None,
        logger=logging.getLogger("t"),
        extra_env={**WIF_ENV, "ANTHROPIC_API_KEY": AGENT_KEY},
    )


MESSAGES = [{"role": "user", "content": "Reply with ok."}]


@pytest.mark.parametrize(
    "llm_call_kwargs",
    [{}, {"stream": True}, {"anthropic_fallbacks": "default"}],
    ids=["plain", "stream", "server-fallback-client"],
)
async def test_access_token_is_sent_as_bearer_without_x_api_key(
    monkeypatch, upstream, llm_call_kwargs
):
    source = make_source(FakeTokenEndpoint(), FakeSts())
    llm = wif_llm(monkeypatch, source, llm_call_kwargs=llm_call_kwargs)

    parsed = await llm.query(MESSAGES, tools=None)

    assert parsed.content == "ok"
    (request,) = upstream
    assert request.url.path == "/v1/messages"
    assert request.headers["authorization"] == "Bearer sk-ant-oat01-test-1"
    assert "x-api-key" not in request.headers
    wire = str(request.headers) + request.content.decode()
    assert HOST_KEY not in wire and AGENT_KEY not in wire


async def test_each_request_renews_an_expiring_token(monkeypatch, upstream):
    endpoint, sts, clock = FakeTokenEndpoint(expires_in=600), FakeSts(), FakeClock()
    llm = wif_llm(monkeypatch, make_source(endpoint, sts, clock))

    await llm.query(MESSAGES, tools=None)
    clock.now += 300
    await llm.query(MESSAGES, tools=None)
    clock.now += 200  # 100 s left
    await llm.query(MESSAGES, tools=None)

    assert [r.headers["authorization"] for r in upstream] == [
        "Bearer sk-ant-oat01-test-1",
        "Bearer sk-ant-oat01-test-1",
        "Bearer sk-ant-oat01-test-2",
    ]
    assert (
        endpoint.assertions
        == sts.minted
        == [
            "header.jti-0.signature",
            "header.jti-1.signature",
        ]
    )


async def test_api_key_mode_is_unchanged(monkeypatch, upstream):
    """Off by default: WIF variables in the environment change nothing."""
    for name, value in WIF_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("ANTHROPIC_API_KEY", HOST_KEY)

    def no_source(config):
        raise AssertionError("WIF is opt-in")

    monkeypatch.setattr(workload_identity, "shared_source", no_source)
    llm = LLM(
        MODEL,
        ProximusConfig(),
        remaining_sec=lambda: None,
        logger=logging.getLogger("t"),
        extra_env={"ANTHROPIC_API_KEY": AGENT_KEY},
    )

    await llm.query(MESSAGES, tools=None)

    (request,) = upstream
    assert request.headers["x-api-key"] == AGENT_KEY
    assert "authorization" not in request.headers
    assert llm.consumed_env_keys == {"ANTHROPIC_API_KEY"}


def test_shared_source_is_one_per_configuration():
    other = WorkloadIdentityConfig(
        federation_rule_id="fdrl_other",
        organization_id="org-test",
        service_account_id=None,
        workspace_id=None,
        sts_region="us-west-2",
    )
    source = workload_identity.shared_source(CONFIG)
    assert workload_identity.shared_source(CONFIG) is source
    assert workload_identity.shared_source(other) is not source
