"""``api_key_loader = "anthropic_wif"``: Anthropic Workload Identity Federation
with AWS STS identity tokens.

The host's AWS role mints a short-lived STS web identity token for the
``https://api.anthropic.com`` audience, and Anthropic exchanges it at
``/v1/oauth/token`` for an access token (``sk-ant-oat…``) that lives about ten
minutes. There is no refresh token: every renewal mints a new STS token, since
each one carries a ``jti`` and can be exchanged only once. LiteLLM sends
``sk-ant-oat`` keys as ``Authorization: Bearer`` and never as ``x-api-key``,
which is how the API accepts them.

Settings, from Harbor's agent env first, then the host:

- ``ANTHROPIC_FEDERATION_RULE_ID`` and ``ANTHROPIC_ORGANIZATION_ID``
- ``ANTHROPIC_SERVICE_ACCOUNT_ID`` and ``ANTHROPIC_WORKSPACE_ID``, when the
  federation rule needs them

STS has no global endpoint for this call, so it uses boto3's default region
(``AWS_DEFAULT_REGION``, ``~/.aws/config``), else ``AWS_REGION``.
"""

import asyncio
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import httpx
import litellm

from harbor_proximus.auth.loader import ApiKeyLoader, CredentialError, Env

AUDIENCE = "https://api.anthropic.com"
TOKEN_URL = "https://api.anthropic.com/v1/oauth/token"
GRANT_TYPE = "urn:ietf:params:oauth:grant-type:jwt-bearer"
# The beta flags Anthropic's own SDK sends on a jwt-bearer exchange.
EXCHANGE_BETA = "oauth-2025-04-20,oidc-federation-2026-04-01"
# Renew this long before expiry, so a request never starts with a token about
# to lapse. Matches the Anthropic SDK's advisory refresh.
REFRESH_MARGIN_SEC = 120.0
EXCHANGE_TIMEOUT_SEC = 30.0
STS_TOKEN_DURATION_SEC = 900

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FederationConfig:
    federation_rule_id: str
    organization_id: str
    service_account_id: str | None
    workspace_id: str | None
    sts_region: str


class StsIdentityToken:
    """Mints a new AWS STS web identity token for Anthropic on every call."""

    def __init__(self, region: str):
        self._region = region
        self._client: Any = None

    def __call__(self) -> str:
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

        try:
            if self._client is None:
                from boto3.session import Session

                self._client = Session().client("sts", region_name=self._region)
            response = self._client.get_web_identity_token(
                Audience=[AUDIENCE],
                SigningAlgorithm="RS256",
                DurationSeconds=STS_TOKEN_DURATION_SEC,
            )
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            status = int(
                e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            )
            retryable = status >= 500 or status == 429 or "Throttl" in code
            raise CredentialError(
                f"AWS STS GetWebIdentityToken failed: {e}", retryable=retryable
            ) from e
        except BotoCoreError as e:
            retryable = not isinstance(e, NoCredentialsError)
            raise CredentialError(
                f"AWS STS GetWebIdentityToken failed: {e}", retryable=retryable
            ) from e
        return str(response["WebIdentityToken"])


@dataclass(frozen=True)
class _AccessToken:
    token: str
    expires_at: float


class AnthropicWIF(ApiKeyLoader):
    """The current access token, exchanged again when it is close to expiry.

    One loader per configuration per process, so concurrent trials share a
    token. It is safe across threads and event loops: callers that find the
    token stale wait for one shared refresh.
    """

    _shared: ClassVar[dict[FederationConfig, "AnthropicWIF"]] = {}
    _shared_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        config: FederationConfig,
        *,
        identity_token: Callable[[], str] | None = None,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._config = config
        self._identity_token = identity_token or StsIdentityToken(config.sts_region)
        self._transport = transport
        self._clock = clock
        self._lock = threading.Lock()
        self._current: _AccessToken | None = None

    @classmethod
    def from_env(cls, model_name: str, env: Env) -> "AnthropicWIF":
        try:
            provider = litellm.get_llm_provider(model_name)[1]
        except Exception:  # noqa: BLE001 - an unroutable model is a config error
            provider = None
        if provider != "anthropic":
            raise ValueError(
                f"api_key_loader 'anthropic_wif' needs an anthropic/ model, "
                f"not {model_name!r}"
            )
        region = _default_aws_region()
        if not region:
            raise ValueError(
                "api_key_loader 'anthropic_wif' needs an AWS region for STS: "
                "set AWS_DEFAULT_REGION"
            )
        config = FederationConfig(
            federation_rule_id=env.require("ANTHROPIC_FEDERATION_RULE_ID"),
            organization_id=env.require("ANTHROPIC_ORGANIZATION_ID"),
            service_account_id=env.get("ANTHROPIC_SERVICE_ACCOUNT_ID"),
            workspace_id=env.get("ANTHROPIC_WORKSPACE_ID"),
            sts_region=region,
        )
        with cls._shared_lock:
            loader = cls._shared.get(config)
            if loader is None:
                loader = cls._shared[config] = cls(config)
            return loader

    async def load(self) -> str:
        current = self._current
        if current is not None and self._is_fresh(current):
            return current.token
        # STS and the exchange are blocking calls; keep them off the loop.
        return await asyncio.to_thread(self._refresh_if_stale)

    def _is_fresh(self, current: _AccessToken) -> bool:
        return current.expires_at - self._clock() > REFRESH_MARGIN_SEC

    def _refresh_if_stale(self) -> str:
        with self._lock:
            current = self._current
            if current is None or not self._is_fresh(current):
                # Never reuse an identity token: each one is single-use.
                current = self._exchange(self._identity_token())
                self._current = current
            return current.token

    def _exchange(self, assertion: str) -> _AccessToken:
        config = self._config
        body = {
            "grant_type": GRANT_TYPE,
            "assertion": assertion,
            "federation_rule_id": config.federation_rule_id,
            "organization_id": config.organization_id,
        }
        if config.service_account_id:
            body["service_account_id"] = config.service_account_id
        if config.workspace_id:
            body["workspace_id"] = config.workspace_id
        started = self._clock()
        try:
            with httpx.Client(
                transport=self._transport, timeout=EXCHANGE_TIMEOUT_SEC
            ) as client:
                response = client.post(
                    TOKEN_URL, json=body, headers={"anthropic-beta": EXCHANGE_BETA}
                )
        except httpx.HTTPError as e:
            raise CredentialError(
                f"Anthropic token exchange failed: {e}", retryable=True
            ) from e
        if response.status_code != 200:
            status = response.status_code
            raise CredentialError(
                f"Anthropic token exchange returned {status}: "
                f"{_error_message(response)}",
                retryable=status in (408, 429) or status >= 500,
            )
        try:
            data = response.json()
            token = data["access_token"]
            expires_in = float(data["expires_in"])
        except (ValueError, KeyError, TypeError) as e:
            raise CredentialError(
                "Anthropic token exchange response lacks access_token/expires_in",
                retryable=True,
            ) from e
        if not isinstance(token, str) or not token or expires_in <= 0:
            raise CredentialError(
                "Anthropic token exchange returned an unusable token", retryable=True
            )
        _logger.debug(
            "Exchanged an AWS identity token for an Anthropic access token "
            "(expires in %ds)",
            expires_in,
        )
        # Expiry counts from before the request, never past the real one.
        return _AccessToken(token, started + expires_in)


def _default_aws_region() -> str | None:
    from boto3.session import Session

    return Session().region_name or os.environ.get("AWS_REGION")


def _error_message(response: httpx.Response) -> str:
    """The API's error message, never the raw body: token endpoints can echo
    the assertion."""
    try:
        error = response.json().get("error")
        message = error.get("message") if isinstance(error, dict) else None
    except (ValueError, AttributeError):
        message = None
    return str(message or response.reason_phrase)[:300]
