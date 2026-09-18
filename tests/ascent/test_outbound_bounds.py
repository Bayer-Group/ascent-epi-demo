"""Bounds on the two outbound calls that sit on every MCP data-tool path.

Both were missed by an earlier pass, which bounded one instance of each class of bug and
left a sibling: the Snowflake OBO exchange (MSAL, untimed, default thread pool)
and the backend's own medical-coder client (1000s socket timeouts, total=None,
plus a synchronous token refresh called from a coroutine).
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

# --- Snowflake OBO exchange ------------------------------------------------


# --- Medical-coder client --------------------------------------------------


def test_coder_http_bounds_are_sane():
    from ascent_platform.http import timeouts as constants

    assert constants.MEDICAL_CODER_TOTAL_TIMEOUT <= 600, "no effective ceiling on a coder call"
    assert constants.MEDICAL_CODER_CONNECT_TIMEOUT <= 30
    assert constants.MEDICAL_CODER_READ_TIMEOUT <= 300


def test_coder_request_sets_a_total_timeout():
    """sock_* bounds alone let a slow-drip response run forever.

    The timeout is built by ``medical_coder_timeout()`` rather than inline, so
    this checks the object the request actually receives instead of reading the
    source for a ClientTimeout literal that is no longer there.
    """
    from ascent_platform.http import medical_coder_timeout

    t = medical_coder_timeout()
    assert t.total is not None, "total=None — the response body is unbounded"
    assert t.total > 0
    assert t.sock_connect is not None and t.sock_read is not None


def test_async_path_does_not_call_the_blocking_token_refresh():
    """get_headers() sleeps; calling it from a coroutine blocks the loop."""
    import inspect

    from ascent_platform.external import ascent_client

    source = inspect.getsource(ascent_client.AscentClient.query_ascent_api)
    assert "self.get_headers()" not in source, "blocking get_headers() called from async code"
    assert "await self.aget_headers()" in source


@pytest.mark.asyncio
async def test_aget_headers_backs_off_without_blocking_the_loop():
    """A token refresh that retries must not stall unrelated coroutines."""
    from ascent_platform.external import ascent_client

    client = object.__new__(ascent_client.AscentClient)
    client.token_cache = None
    client.token_expiry = 0
    client.tokens = MagicMock()
    client.tokens.client_id = "cid"
    calls = {"n": 0}

    def flaky(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("transient")
        return {"access_token": "tok", "expires_in": 3600}

    client.tokens.acquire = flaky

    ticks = 0

    async def other_work():
        nonlocal ticks
        for _ in range(5):
            await asyncio.sleep(0.01)
            ticks += 1

    # Bind the real sleep before patching, or the stub recurses into itself.
    real_sleep = asyncio.sleep
    with patch.object(ascent_client.asyncio, "sleep", new=lambda _s: real_sleep(0.02)):
        headers, _ = await asyncio.gather(client.aget_headers(), other_work())

    assert headers == {"Authorization": "Bearer tok"}
    assert calls["n"] == 2
    # The loop kept running during the backoff.
    assert ticks == 5


@pytest.mark.asyncio
async def test_aget_headers_uses_the_cached_token():
    from ascent_platform.external import ascent_client

    client = object.__new__(ascent_client.AscentClient)
    client.token_cache = "cached"
    client.token_expiry = time.time() + 600
    client.tokens = MagicMock()
    client.tokens.acquire = MagicMock(side_effect=AssertionError("should not refetch"))

    assert await client.aget_headers() == {"Authorization": "Bearer cached"}
