"""Which databases hold synthetic data.

Lived in ``ascent_mcp/_synthetic.py``. It is not an MCP concern -- it queries
Snowflake for a property of the data -- and ``ascent_domain.database_access``
needed it to decide what a reviewer-key request may list. That import had to be
function-local, with the comment "ascent_mcp depends on ascent, not vice versa",
which is a cycle being dodged rather than avoided.

The MCP-facing half (the warning text, the tool decorator, the annotation
helper) stays in ascent_mcp and re-exports these.
"""

import logging

from ascent_platform.constants import ExpirationTTL

logger = logging.getLogger(__name__)


SYNTHETIC_CACHE_KEY = "synthetic_databases"


async def _fetch_synthetic_databases() -> set[str]:
    """Which databases hold synthetic data. Here, all of them.

    Every shipped database is generated, so this returns all of them rather
    than an empty set: callers use it to decide what a restricted caller may
    list, and empty would hide every database instead of allowing all.
    """
    from ascent_platform.warehouse.bootstrap import DATABASES

    return {db.name for db in DATABASES}


async def get_synthetic_databases() -> set[str]:
    """Return the cached set of synthetic database names.

    Only caches on a successful fetch.  Transient Snowflake errors propagate
    so callers can decide how to handle them (the decorator falls back to
    returning the original result without a warning).
    """
    from ascent_platform.cache.aiocache_backend import cache

    cached_value = await cache.get(SYNTHETIC_CACHE_KEY)
    if cached_value is not None:
        return cached_value

    synthetic = await _fetch_synthetic_databases()
    await cache.set(SYNTHETIC_CACHE_KEY, synthetic, ttl=ExpirationTTL.HOUR * 3)
    return synthetic


async def is_synthetic_database(database_name: str) -> bool:
    """Check whether a database is flagged as synthetic."""
    synthetic = await get_synthetic_databases()
    return database_name in synthetic
