import asyncio
import threading
from contextvars import ContextVar

import pytest

from ascent_platform.llm import executor
from ascent_platform.llm.clients.assistants import is_transient_llm_error


@pytest.fixture
def pool(monkeypatch):
    instance = executor.BlockingLLMExecutor(max_workers=1)
    monkeypatch.setattr(executor, "get_executor", lambda: instance)
    yield instance
    instance.shutdown(wait=True)


async def _wait_until_set(event):
    assert await asyncio.wait_for(asyncio.to_thread(event.wait, 2), timeout=3)


async def test_excess_work_is_rejected_not_queued(pool):
    started = threading.Event()
    release = threading.Event()

    def work():
        started.set()
        assert release.wait(3)
        return "done"

    task = asyncio.create_task(executor.run_blocking_llm_call(work))
    try:
        await _wait_until_set(started)
        for _ in range(30):
            with pytest.raises(executor.LLMCapacityError):
                await executor.run_blocking_llm_call(lambda: pytest.fail("must not be submitted"))
        assert pool._work_queue.qsize() == 0
    finally:
        release.set()
        assert await task == "done"
    assert await executor.run_blocking_llm_call(lambda: "next") == "next"


async def test_cancelled_caller_does_not_release_running_worker(pool):
    started = threading.Event()
    release = threading.Event()

    def work():
        started.set()
        assert release.wait(3)
        raise ValueError("late failure after caller cancellation")

    task = asyncio.create_task(executor.run_blocking_llm_call(work))
    try:
        await _wait_until_set(started)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(executor.LLMCapacityError):
            await executor.run_blocking_llm_call(lambda: None)
    finally:
        release.set()

    async def next_call():
        while True:
            try:
                return await executor.run_blocking_llm_call(lambda: "recovered")
            except executor.LLMCapacityError:
                await asyncio.sleep(0.001)

    assert await asyncio.wait_for(next_call(), timeout=3) == "recovered"


async def test_worker_errors_release_capacity_and_context_is_preserved(pool):
    trace = ContextVar("trace", default=None)
    trace.set("request-id")
    assert await executor.run_blocking_llm_call(trace.get) == "request-id"

    def fail():
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        await executor.run_blocking_llm_call(fail)
    assert await executor.run_blocking_llm_call(lambda: "ok") == "ok"


def test_capacity_errors_are_not_automatically_retried():
    assert not is_transient_llm_error(executor.LLMCapacityError("busy"))


def test_shutdown_is_idempotent_and_does_not_create_a_pool(monkeypatch):
    monkeypatch.setattr(executor, "_executor", None)
    executor.shutdown_executor()
    assert executor._executor is None
    pool = executor.get_executor()
    executor.shutdown_executor()
    assert pool._shutdown
    assert executor._executor is None
    executor.shutdown_executor()


async def test_lifespan_drains_executor_even_when_body_fails(monkeypatch):
    from types import SimpleNamespace

    from ascent_http import lifespan

    events = []
    monkeypatch.setattr(
        lifespan,
        "scheduler",
        SimpleNamespace(
            start=lambda: events.append("start"),
            add_job=lambda *args, **kwargs: None,
            shutdown=lambda: events.append("stop scheduler"),
        ),
    )
    monkeypatch.setattr(lifespan, "shutdown_llm_executor", lambda: events.append("drain executor"))
    with pytest.raises(RuntimeError, match="body failed"):
        async with lifespan.lifespan(None):
            raise RuntimeError("body failed")
    assert events == ["start", "stop scheduler", "drain executor"]


def test_submit_failure_returns_the_slot():
    pool = executor.BlockingLLMExecutor(max_workers=1)
    pool.shutdown()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="shutdown"):
            pool.submit(lambda: None)
