"""MCP authentication: off by default, one shared token when you need it.

A local install has no directory to authenticate against and one user, so the
scheme is deliberately small:

* **No ``API_AUTH_TOKEN`` set** -- the MCP endpoints are open. Correct for a
  stack bound to localhost serving its own synthetic data, and the default so
  that ``docker compose up`` works with no ceremony.
* **``API_AUTH_TOKEN`` set** -- callers must present it as a bearer. Enough to
  put the stack somewhere reachable without handing over the tools.

A shared token is not an identity system: it cannot tell two callers apart,
cannot be revoked individually, and everything still runs as one local user.
It exists so that "expose this" has an answer better than "don't", not to
imply the authorisation model came back.

What is actually at stake if it is open, which is worth knowing before
deciding: the warehouse is attached READ_ONLY, so no query can damage the
data, and the data is synthetic anyway. The real exposure is the cohort tools,
which write to Postgres, and the LLM key -- an open MCP endpoint lets anyone
spend it.
"""

import logging

import redis.asyncio as redis

from ascent_platform.config.runtime import get_runtime_settings

logger = logging.getLogger(__name__)

_settings = get_runtime_settings()

# Async client: /mcp/health awaits ping(), and the MCP event store is async.
# A synchronous redis.Redis returns a bool from ping(), so awaiting it raises
# and the health check reports Redis unreachable while it is perfectly fine.
#
# Bounded deliberately. An unbounded client waits forever on a Redis that is
# reachable but wedged, and this one sits on the MCP event-store and health
# paths -- so the whole surface hangs rather than degrading. The default
# client sets no socket timeout and no connection cap at all.
redis_client = redis.Redis(
    host=_settings.REDIS_HOST,
    port=_settings.REDIS_PORT,
    decode_responses=True,
    socket_timeout=_settings.CACHE_REDIS_SOCKET_TIMEOUT,
    socket_connect_timeout=_settings.CACHE_REDIS_SOCKET_TIMEOUT,
    max_connections=_settings.CACHE_REDIS_MAX_CONNECTIONS,
)


def _build_auth():
    token = (_settings.API_AUTH_TOKEN or "").strip()
    if not token:
        logger.warning(
            "=" * 72 + "\nMCP AUTHENTICATION IS DISABLED -- no API_AUTH_TOKEN is set.\n"
            "Anyone who can reach this port can call every tool, and every tool "
            "call spends the configured LLM key. There is no per-caller quota.\n"
            "This is the intended default for a loopback-bound demo. Set "
            "API_AUTH_TOKEN before exposing the port to anything else.\n" + "=" * 72
        )
        return None

    from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

    logger.info("MCP authentication enabled via API_AUTH_TOKEN")
    return StaticTokenVerifier(
        tokens={
            token: {
                "client_id": "local",
                "sub": "local-user",
                "scopes": [],
            }
        }
    )


combined_auth_provider = _build_auth()
