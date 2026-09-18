"""Token caching in AscentClient.

These used to patch ``msal.ConfidentialClientApplication`` and assert a new
bearer was acquired from Azure. There is no directory to acquire one from any
more -- ServicePrincipalTokenProvider hands back a fixed placeholder -- but the
caching logic around it is unchanged and still worth pinning: the client caches
on truthiness and recomputes an expiry, and both of those have bitten before
(an empty placeholder re-acquired on every request; a missing ``expires_in``
raised KeyError inside every medical-coder call).
"""

import time

from ascent_platform.azure.token import ServicePrincipalTokenProvider
from ascent_platform.external.ascent_client import AscentClient


def _client() -> AscentClient:
    return AscentClient(
        base_url="https://example.com",
        azure_client_id="client_id",
        azure_client_secret="client_secret",
        azure_tenant_id="tenant_id",
    )


def test_get_headers_uses_the_cache_while_it_is_valid():
    client = _client()
    client.token_cache = "cached_token"
    client.token_expiry = time.time() + 600

    assert client.get_headers()["Authorization"] == "Bearer cached_token"


def test_get_headers_replaces_an_expired_token():
    client = _client()
    client.token_cache = "old_token"
    client.token_expiry = time.time() - 1

    headers = client.get_headers()

    assert headers["Authorization"] != "Bearer old_token"
    assert headers["Authorization"].startswith("Bearer ")
    assert client.token_expiry > time.time()


def test_the_placeholder_bearer_is_never_empty():
    """Callers cache on truthiness, so "" would re-acquire on every request."""
    assert ServicePrincipalTokenProvider().token


def test_acquire_reports_an_expiry():
    """Its absence raised KeyError('expires_in') from every coder call."""
    assert ServicePrincipalTokenProvider().acquire()["expires_in"] > 0
