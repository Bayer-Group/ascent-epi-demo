"""Targeted tests for the MCP resilience fixes (step budgets, admission
control, bulkheads). See mcp-tool-hang-issue analysis: heavy `task=True`
tools hung for 10-20 min under load because downstream calls were unbounded
and nothing enforced per-step liveness.
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ascent_mcp.tools import _capacity
from ascent_mcp.tools._capacity import ServerBusy, heavy_tool_slot
from ascent_mcp.tools._keepalive import StepTimeout, progress_keepalive, step_budget

SRC = Path(__file__).resolve().parents[2] / "src"


def _ctx() -> AsyncMock:
    ctx = AsyncMock()
    ctx.report_progress = AsyncMock()
    return ctx


# ---------------------------------------------------------------- step guard

async def test_step_guard_cancels_over_budget_step():
    started = time.monotonic()
    with pytest.raises(StepTimeout) as excinfo:
        async with progress_keepalive(_ctx(), 1, 4, "stuck step", budget=0.2):
            await asyncio.sleep(30)  # simulated wedged downstream call
    elapsed = time.monotonic() - started
    assert elapsed < 5, "budget must fire promptly, not wait for the body"
    assert "stuck step" in str(excinfo.value)
    assert excinfo.value.budget_seconds == 0.2


async def test_step_guard_lets_under_budget_step_complete():
    result = []
    async with progress_keepalive(_ctx(), 1, 4, "healthy step", budget=5):
        await asyncio.sleep(0.01)
        result.append("done")
    assert result == ["done"]


async def test_step_guard_unbounded_by_default():
    # No budget -> behaves exactly like the legacy keepalive (no deadline).
    async with progress_keepalive(_ctx(), 1, 4, "legacy call site"):
        await asyncio.sleep(0.01)


async def test_step_guard_stops_heartbeat_after_timeout():
    ctx = _ctx()
    with pytest.raises(StepTimeout):
        async with progress_keepalive(ctx, 1, 4, "stuck", interval=1, budget=0.05):
            await asyncio.sleep(30)
    # Give any zombie heartbeat a chance to fire, then confirm silence.
    calls_after_exit = ctx.report_progress.await_count
    await asyncio.sleep(1.2)
    assert ctx.report_progress.await_count == calls_after_exit


def test_step_budget_env_override(monkeypatch):
    monkeypatch.setenv("MCP_STEP_BUDGET_ENTITIES", "42.5")
    assert step_budget("ENTITIES", 240) == 42.5
    monkeypatch.setenv("MCP_STEP_BUDGET_ENTITIES", "not-a-number")
    assert step_budget("ENTITIES", 240) == 240
    monkeypatch.delenv("MCP_STEP_BUDGET_ENTITIES")
    assert step_budget("ENTITIES", 240) == 240


# ---------------------------------------------------------- admission control

async def test_capacity_rejects_when_slots_exhausted(monkeypatch):
    monkeypatch.setattr(_capacity, "_HEAVY_TOOL_CONCURRENCY", 1)
    monkeypatch.setattr(_capacity, "_ACQUIRE_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(_capacity, "_semaphores", {})

    release = asyncio.Event()

    async def occupant():
        async with heavy_tool_slot("occupant"):
            await release.wait()

    holder = asyncio.create_task(occupant())
    await asyncio.sleep(0.05)  # let it take the only slot

    started = time.monotonic()
    with pytest.raises(ServerBusy):
        async with heavy_tool_slot("rejected"):
            pass  # pragma: no cover
    assert time.monotonic() - started < 2, "busy rejection must be fast"

    release.set()
    await holder


async def test_capacity_releases_slot_after_use(monkeypatch):
    monkeypatch.setattr(_capacity, "_HEAVY_TOOL_CONCURRENCY", 1)
    monkeypatch.setattr(_capacity, "_ACQUIRE_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(_capacity, "_semaphores", {})

    async with heavy_tool_slot("first"):
        pass
    async with heavy_tool_slot("second"):  # would raise if the slot leaked
        pass


async def test_capacity_releases_slot_on_error(monkeypatch):
    monkeypatch.setattr(_capacity, "_HEAVY_TOOL_CONCURRENCY", 1)
    monkeypatch.setattr(_capacity, "_ACQUIRE_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(_capacity, "_semaphores", {})

    with pytest.raises(RuntimeError):
        async with heavy_tool_slot("boom"):
            raise RuntimeError("step failed")
    async with heavy_tool_slot("after-error"):
        pass


# ----------------------------------------------------- structured error frame

def test_error_result_frames():
    from ascent_mcp.tools_v1_clues import _error_result

    timeout_frame = _error_result(StepTimeout("Generating SQL template", 300))
    assert timeout_frame["error"] == "timeout"
    assert timeout_frame["retryable"] is True
    assert "300" in timeout_frame["message"]

    busy_frame = _error_result(ServerBusy("generate_epidemiological_sql_non_omop"))
    assert busy_frame["error"] == "server_busy"
    assert busy_frame["retryable"] is True


# ------------------------------------------------------------------ bulkheads

def test_warehouse_hot_path_uses_dedicated_executor():
    """The hot query paths must not touch the shared default thread pool.

    Snowflake was replaced by the DuckDB warehouse, but the bulkhead reason is
    unchanged: blocking driver calls on Python's default pool starve every
    other coroutine in the process.
    """
    source = (SRC / "ascent_platform" / "warehouse" / "session.py").read_text()
    assert "asyncio.to_thread" not in source
    assert "run_in_executor(None" not in source


def test_telemetry_offloads_use_dedicated_executor():
    source = (SRC / "ascent_mcp" / "telemetry.py").read_text()
    assert "run_in_executor(\n                        None" not in source
    assert "run_in_executor(None" not in source
    assert "_TELEMETRY_EXECUTOR" in source


def test_mcp_redis_client_is_bounded():
    source = (SRC / "ascent_mcp" / "authentication.py").read_text()
    assert "socket_timeout" in source
    assert "socket_connect_timeout" in source
    assert "max_connections" in source
