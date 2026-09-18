"""A retry waits its backoff once.

``async_retry`` delegates to ``handle_retry`` when the instance defines one,
and the default implementation sleeps before returning True. Falling through to
the decorator's own sleep as well doubles every delay, which on the slowest
path turns a bounded retry into twice the wait for no benefit.
"""

from __future__ import annotations

import asyncio

import pytest

from ascent_platform.llm.clients import assistants
from ascent_platform.llm.clients.assistants import async_retry, async_retry_fn


class _Client:
    max_retries = 3
    base_delay = 1.0
    backoff_factor = 2.0

    def calculate_sleep_time(self, attempt: int, base_delay: float, backoff_factor: float) -> float:
        return base_delay * (backoff_factor**attempt)

    async def handle_retry(self, err, attempt, max_retries, base_delay, backoff_factor) -> bool:
        if attempt < max_retries - 1:
            await asyncio.sleep(self.calculate_sleep_time(attempt, base_delay, backoff_factor))
            return True
        return False

    @async_retry(retry_on=(ConnectionError,))
    async def always_fails(self):
        raise ConnectionError("boom")


@pytest.fixture
def recorded_sleeps(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(delay, *args, **kwargs):
        slept.append(delay)

    monkeypatch.setattr(assistants.asyncio, "sleep", fake_sleep)
    return slept


async def test_each_backoff_is_waited_once(recorded_sleeps):
    with pytest.raises(ConnectionError):
        await _Client().always_fails()

    assert recorded_sleeps == [1.0, 2.0]


async def test_a_client_without_handle_retry_still_backs_off(recorded_sleeps):
    class Plain:
        max_retries = 3
        base_delay = 1.0
        backoff_factor = 2.0

        @async_retry(retry_on=(ConnectionError,))
        async def always_fails(self):
            raise ConnectionError("boom")

    with pytest.raises(ConnectionError):
        await Plain().always_fails()

    assert recorded_sleeps == [1.0, 2.0]


async def test_zero_delay_override_is_preserved(recorded_sleeps):
    class Client(_Client):
        @async_retry(max_retries=3, base_delay=0, backoff_factor=0, retry_on=ConnectionError)
        async def call(self):
            raise ConnectionError("temporary")

    with pytest.raises(ConnectionError):
        await Client().call()
    assert recorded_sleeps == [0, 0]


async def test_function_adapter_uses_same_backoff(recorded_sleeps):
    @async_retry_fn(max_retries=3, base_delay=2, backoff_factor=3, retry_on=ConnectionError)
    async def call():
        raise ConnectionError("temporary")

    with pytest.raises(ConnectionError):
        await call()
    assert recorded_sleeps == [2, 6]


async def test_custom_handler_can_stop_retrying(recorded_sleeps):
    class Client(_Client):
        attempts = 0

        async def handle_retry(self, *args):
            return False

        @async_retry(retry_on=ConnectionError)
        async def call(self):
            self.attempts += 1
            raise ConnectionError("temporary")

    client = Client()
    with pytest.raises(ConnectionError):
        await client.call()
    assert client.attempts == 1
    assert recorded_sleeps == []


async def test_cancellation_is_never_retried(recorded_sleeps):
    calls = 0

    @async_retry_fn()
    async def call():
        nonlocal calls
        calls += 1
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await call()
    assert calls == 1
    assert recorded_sleeps == []


async def test_zero_attempts_is_rejected_before_calling(recorded_sleeps):
    @async_retry_fn(max_retries=0)
    async def call():
        pytest.fail("invalid retry settings must be rejected first")

    with pytest.raises(ValueError, match="positive"):
        await call()
