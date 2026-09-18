"""Admission control for heavy MCP pipeline tools.

Bounds how many heavy pipeline runs execute concurrently per worker process.
Beyond the cap, callers wait briefly for a slot and then receive a fast,
structured "busy" error instead of queueing invisibly until the client's
timeout — under load, invisible queueing was
indistinguishable from a hang.

The semaphore is per event loop (one loop per uvicorn worker in production;
isolated loops in tests). Fleet capacity = limit x number of workers.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict

logger = logging.getLogger(__name__)

_HEAVY_TOOL_CONCURRENCY = int(os.environ.get("MCP_HEAVY_TOOL_CONCURRENCY", "8"))
_ACQUIRE_TIMEOUT_SECONDS = float(os.environ.get("MCP_HEAVY_TOOL_ACQUIRE_TIMEOUT", "10"))

_semaphores: Dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}


class ServerBusy(Exception):
    """All heavy-tool slots are occupied; the caller should retry shortly."""

    def __init__(self, tool_name: str):
        super().__init__(
            f"Server is at capacity ({_HEAVY_TOOL_CONCURRENCY} concurrent heavy runs); '{tool_name}' was not started. Please retry in a moment."
        )
        self.tool_name = tool_name


def _get_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    semaphore = _semaphores.get(loop)
    if semaphore is None:
        semaphore = asyncio.Semaphore(max(1, _HEAVY_TOOL_CONCURRENCY))
        _semaphores[loop] = semaphore
    return semaphore


@asynccontextmanager
async def heavy_tool_slot(tool_name: str) -> AsyncIterator[None]:
    """Acquire a heavy-tool execution slot or fail fast with :class:`ServerBusy`."""
    semaphore = _get_semaphore()
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=_ACQUIRE_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        logger.warning("Heavy-tool capacity exhausted; rejecting '%s'", tool_name)
        raise ServerBusy(tool_name) from exc
    try:
        yield
    finally:
        semaphore.release()
