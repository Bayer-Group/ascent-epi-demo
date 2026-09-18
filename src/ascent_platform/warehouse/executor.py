"""Dedicated executor for blocking warehouse calls.

DuckDB is synchronous and holds the GIL only for short stretches, so queries
run in a thread. Running them on Python's default executor — sized
``min(32, cpu + 4)`` and shared with every other blocking call in the process —
is what produced observed stalls: a handful of slow queries exhausted
the pool and unrelated work queued behind them with no timeout and no error.

One executor is shared process-wide rather than one per package — two pools of
20 would defeat the point.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable

from ascent_platform.config.runtime import get_runtime_settings

logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(
    max_workers=get_runtime_settings().WAREHOUSE_EXECUTOR_WORKERS,
    thread_name_prefix="warehouse",
)


def get_executor() -> ThreadPoolExecutor:
    return _EXECUTOR


_shutdown_done = False


def shutdown_executor() -> None:
    """Idempotent: both this module's atexit hook and the backend's own
    registration reach it, and a double shutdown would log twice and block a
    second time on an already-drained pool.
    """
    global _shutdown_done
    if _shutdown_done:
        return
    _shutdown_done = True
    logger.info("Shutting down warehouse executor...")
    _EXECUTOR.shutdown(wait=True)


atexit.register(shutdown_executor)


async def run_db(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run blocking warehouse work on the dedicated executor.

    Accepts keyword arguments because the driver needs them: ``executemany`` is
    called as ``executemany(query, params, _exec_async=True, _no_results=True)``.
    """
    loop = asyncio.get_running_loop()
    if args or kwargs:
        return await loop.run_in_executor(_EXECUTOR, partial(fn, *args, **kwargs))
    return await loop.run_in_executor(_EXECUTOR, fn)
