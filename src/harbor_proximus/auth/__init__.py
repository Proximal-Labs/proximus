"""Where a model's API key comes from.

By default it is a static key, resolved the way Harbor's own agents do
(:func:`resolve_credentials`). With the ``api_key_loader`` option, an
:class:`ApiKeyLoader` loads the key for each request instead, for credentials
that expire. ``anthropic_wif`` (Anthropic Workload Identity Federation) is
built in. A custom loader is a ``module:Class`` path to a subclass.
"""

from harbor_proximus.auth.loader import (
    BUILTIN_LOADERS,
    ApiKeyLoader,
    CredentialError,
    Env,
    loader_class,
)
from harbor_proximus.auth.resolve import Credentials, resolve_credentials

__all__ = [
    "BUILTIN_LOADERS",
    "ApiKeyLoader",
    "CredentialError",
    "Credentials",
    "Env",
    "loader_class",
    "resolve_credentials",
]
