"""SSE keep-alive heartbeat for long-running MCP tool steps.

External MCP clients communicate over an SSE stream. Proxies and load balancers
(e.g. AWS ALB) enforce an idle-connection timeout (commonly 60–120 s). If no
SSE event is sent during a long-running step the proxy closes the connection
before the tool completes, causing ``ClosedResourceError`` on the server and
``RemoteProtocolError`` on the client.

This module provides a context manager that sends periodic MCP progress
notifications during long-running awaits so the SSE stream is never idle.

This works for FOREGROUND stateless requests: it has been observed
holding a 32-minute call open with a progress notification every 30s.

Important caveat for ``task=True`` tools: once FastMCP promotes a tool to a
Docket background task, ``Context.report_progress`` no longer emits a progress
notification. It writes the update into Redis instead (see
``fastmcp/server/context.py``), where it is visible via ``tasks/get`` but sends
nothing over the wire. The heartbeat below therefore does *not* keep the
connection alive for background tasks. fastmcp's ``PingMiddleware`` is not an
alternative here: it is incompatible with ``stateless_http=True``, because its
``send_ping`` awaits a client response on a per-request stream that has already
closed, which crashes the session.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager, suppress
from typing import AsyncIterator, Optional

from fastmcp import Context

logger = logging.getLogger(__name__)

_KEEPALIVE_INTERVAL_SECONDS = 30


class StepTimeout(Exception):
    """A pipeline step exceeded its time budget.

    Raised (instead of hanging) so the tool can return a structured error,
    the client sees a fast visible failure, and — for ``task=True`` tools —
    the Docket worker slot is released instead of being held for hours.
    """

    def __init__(self, message: str, budget_seconds: float):
        super().__init__(
            f"Step '{message}' exceeded its {budget_seconds:.0f}s budget. The step was cancelled to keep the server responsive; please retry."
        )
        self.step_message = message
        self.budget_seconds = budget_seconds


def step_budget(name: str, default_seconds: float) -> float:
    """Per-step budget, overridable via ``MCP_STEP_BUDGET_<NAME>`` (seconds)."""
    raw = os.environ.get(f"MCP_STEP_BUDGET_{name.upper()}")
    try:
        return float(raw) if raw else default_seconds
    except ValueError:
        logger.warning("Invalid MCP_STEP_BUDGET_%s=%r; using default %ss", name.upper(), raw, default_seconds)
        return default_seconds


@asynccontextmanager
async def progress_keepalive(
    ctx: Context,
    current: int,
    total: int,
    message: str,
    interval: int = _KEEPALIVE_INTERVAL_SECONDS,
    budget: Optional[float] = None,
) -> AsyncIterator[None]:
    """Keep the SSE connection alive during a long-running step and, when
    ``budget`` is set, bound the step's duration.

    Heartbeat: spawns a background task that re-sends the current progress
    notification every *interval* seconds, resetting the proxy idle timer
    without changing visible progress state.

    Budget: when set, the wrapped body is cancelled once it exceeds *budget*
    seconds and :class:`StepTimeout` is raised. This bounds each *step* rather
    than the whole tool — a legitimately long run that keeps advancing through
    its steps is never cut off, while a wedged step fails fast and frees its
    Docket slot. (Cancellation propagates into genuinely-async work; sync work
    offloaded to threads must additionally carry its own library timeout.)

    Args:
        ctx: FastMCP Context used to send progress notifications.
        current: Current step index (same value used in the surrounding
            ``ctx.report_progress`` call).
        total: Total number of steps.
        message: Human-readable status message shown to the client.
        interval: Seconds between keep-alive pings (default: 30 s).
        budget: Optional per-step time budget in seconds (default: unbounded,
            preserving previous behavior for existing call sites).

    Example::

        await ctx.report_progress(7, total, "Executing queries on database")
        async with progress_keepalive(ctx, 7, total, "Executing queries on database",
                                      budget=step_budget("EXECUTE_QUERIES", 300)):
            results = await clues_api.execute_queries(...)
    """

    async def _heartbeat() -> None:
        while True:
            await asyncio.sleep(interval)
            try:
                await ctx.report_progress(current, total, message)
                logger.info("SSE keep-alive sent: step %d/%d – %s", current, total, message)
            except Exception as exc:
                # If the stream is already closed, stop quietly.
                logger.info("SSE keep-alive stopped: %s", exc)
                break

    task = asyncio.create_task(_heartbeat())
    started = time.perf_counter()
    try:
        if budget is None:
            yield
        else:
            try:
                async with asyncio.timeout(budget):
                    yield
            except TimeoutError as exc:
                elapsed = time.perf_counter() - started
                logger.error(
                    "Step budget exceeded: step %d/%d – %s (budget=%ss, elapsed=%.1fs)",
                    current,
                    total,
                    message,
                    budget,
                    elapsed,
                    extra={
                        "mcp_step": message,
                        "mcp_step_index": current,
                        "mcp_step_budget_seconds": budget,
                        "duration_ms": elapsed * 1000.0,
                        "mcp_event": "step_timeout",
                    },
                )
                raise StepTimeout(message, budget) from exc
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

        # Always record how long the step took. Without this there is no way to
        # tell a budget that is genuinely too tight from a step that is wedged —
        # the distinction that made the 300s SQL_TEMPLATE ceiling look like an
        # infrastructure deadline rather than a mis-sized budget.
        elapsed = time.perf_counter() - started
        near_budget = budget is not None and elapsed > budget * 0.5
        (logger.warning if near_budget else logger.info)(
            "Step finished: %d/%d – %s in %.1fs (budget=%s)",
            current,
            total,
            message,
            elapsed,
            f"{budget:.0f}s" if budget else "none",
            extra={
                "mcp_step": message,
                "mcp_step_index": current,
                "mcp_step_budget_seconds": budget,
                "duration_ms": elapsed * 1000.0,
                "mcp_event": "step_complete",
            },
        )
