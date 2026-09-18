"""Bound synchronous SDK work without queuing an unbounded request backlog.

There is deliberately no application queue: once every worker is occupied,
reject new work with a non-retryable capacity error. A cancelled caller does
not free its slot until the underlying SDK call actually finishes.
"""

from __future__ import annotations

import asyncio
import atexit
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from functools import partial
from threading import BoundedSemaphore, Lock
from typing import Callable, TypeVar

from ascent_platform.config.runtime import get_runtime_settings

T = TypeVar("T")


class LLMCapacityError(RuntimeError):
    """All blocking provider workers are occupied; callers should retry later."""


class BlockingLLMExecutor(ThreadPoolExecutor):
    def __init__(self, max_workers: int):
        super().__init__(max_workers=max_workers, thread_name_prefix="blocking-llm")
        self._slots = BoundedSemaphore(max_workers)

    def submit(self, fn, /, *args, **kwargs):
        if not self._slots.acquire(blocking=False):
            raise LLMCapacityError("Blocking LLM capacity exhausted; retry after outstanding requests finish")
        try:
            future = super().submit(fn, *args, **kwargs)
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        return future


_executor: BlockingLLMExecutor | None = None
_closing = False
_lock = Lock()


def get_executor() -> BlockingLLMExecutor:
    global _executor
    with _lock:
        if _closing:
            raise LLMCapacityError("Blocking LLM executor is shutting down")
        if _executor is None:
            _executor = BlockingLLMExecutor(get_runtime_settings().LLM_BLOCKING_EXECUTOR_WORKERS)
        return _executor


def shutdown_executor() -> None:
    """Drain SDK work before allowing a subsequent lifespan to create a pool."""
    global _executor, _closing
    with _lock:
        if _executor is None:
            return
        pool, _executor = _executor, None
        _closing = True
    try:
        pool.shutdown(wait=True, cancel_futures=True)
    finally:
        with _lock:
            _closing = False


def _observe_exception(future: asyncio.Future) -> None:
    # A caller can disappear while a shielded SDK call is still running. Consume
    # its eventual exception to avoid an unhandled-future warning in that case.
    if not future.cancelled():
        future.exception()


async def run_blocking_llm_call(func: Callable[..., T], *args, **kwargs) -> T:
    context = copy_context()
    future: Future[T] = get_executor().submit(context.run, partial(func, *args, **kwargs))
    wrapped = asyncio.wrap_future(future)
    wrapped.add_done_callback(_observe_exception)
    return await asyncio.shield(wrapped)


atexit.register(shutdown_executor)
