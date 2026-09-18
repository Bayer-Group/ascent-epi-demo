"""Synthetic data detection and warning utilities for MCP tools.

Checks the IS_SYNTHETIC flag on ASCENT_OMOP_METADATA / ASCENT_NON_OMOP_METADATA.
The ``inject_synthetic_warning`` decorator injects the warning directly into the
dict returned by each tool function, so it works for both sync and background
(task=True) execution paths.

Heavy dependencies (warehouse session, cache) are imported inside the functions
that need them so this module can be imported without side effects.
"""

import functools
import logging

from ascent_domain.synthetic import (  # noqa: F401  (re-exported for the tools)
    SYNTHETIC_CACHE_KEY,
    get_synthetic_databases,
    is_synthetic_database,
)

logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)

SYNTHETIC_DATA_WARNING = (
    "IMPORTANT: Please note that this answer is based on synthetic data "
    "({database_name}), and thus is not reliable. "
    "Do not use these results for clinical decisions, literature comparisons, "
    "or any downstream analysis."
)


async def annotate_databases_with_synthetic_flag(database_pairs) -> list[dict]:
    """Return database dicts annotated with ``is_synthetic``.

    Shared by list-returning tools (``get_omop_databases``,
    ``get_non_omop_databases``). Fail-safe: if the synthetic lookup raises,
    every entry defaults to ``is_synthetic: False`` so the listing still
    returns.
    """
    try:
        synthetic_dbs = await get_synthetic_databases()
    except Exception:
        logger.warning("Could not check synthetic databases", exc_info=True)
        synthetic_dbs = set()
    return [
        {
            "database_name": pair.database_name,
            "database_schemas": pair.database_schemas,
            "is_synthetic": pair.database_name in synthetic_dbs,
        }
        for pair in database_pairs
    ]


def inject_synthetic_warning(func):
    """Decorator that injects a synthetic-data warning into dict results.

    Wraps an async tool function. If the ``database_name`` keyword argument
    points to a synthetic database, a ``synthetic_warning`` key is added to
    the returned dict. Non-dict results are returned unchanged.

    Must be placed *below* ``@mcp.tool(...)`` so it wraps the function before
    FastMCP registers it.  This ensures the warning is present in both sync
    and background (task=True) execution paths.
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        database_name = kwargs.get("database_name")
        if not database_name or not isinstance(result, dict):
            return result
        try:
            if await is_synthetic_database(database_name):
                result["synthetic_warning"] = SYNTHETIC_DATA_WARNING.format(
                    database_name=database_name,
                )
        except Exception:
            # Intentionally broad: warehouse, cache, and network errors should
            # never prevent a tool from returning its primary result.
            # Safe for cancellation: asyncio.CancelledError is a BaseException
            # (Python 3.9+) and is not caught here.
            logger.warning("Failed to check synthetic status for %s", database_name, exc_info=True)
        return result

    return wrapper
