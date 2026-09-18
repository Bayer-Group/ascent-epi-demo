"""Nothing on the LLM path may be synchronous.

It used to be. ``BaseLLMClient.send`` was sync, so ``LLMRouter.send_message``
-- which is async -- reached the provider through
``loop.run_in_executor(None, ...)``. That does not block the event loop, but it
holds a worker in the DEFAULT thread pool for the entire call, including
``time.sleep`` backoff of up to 15s across three retries. The default pool is
shared with every other blocking call in the process; that is the starvation
mode which the Snowflake layer has a dedicated bulkhead to avoid and
which the LLM path was simply exempted from.

The clients are natively async now (``genai.Client.aio``, ``AsyncAzureOpenAI``),
so there is no executor hop and no thread. These tests keep it that way.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
LLM = SRC / "ascent_platform/llm"


def _client_classes():
    from ascent_platform.llm.clients.azure_chat import (
        AzureOpenAIClient,
        GPT4oMiniClient,
        OmodelsClient,
    )
    from ascent_platform.llm.clients.base import BaseLLMClient
    from ascent_platform.llm.clients.gemini import GeminiClient

    return [BaseLLMClient, AzureOpenAIClient, GPT4oMiniClient, OmodelsClient, GeminiClient]


@pytest.mark.parametrize("cls", _client_classes(), ids=lambda c: c.__name__)
def test_send_and_underscore_send_are_coroutines(cls):
    for name in ("send", "_send"):
        fn = getattr(cls, name, None)
        if fn is None:
            continue
        assert inspect.iscoroutinefunction(fn), (
            f"{cls.__name__}.{name} is synchronous; the router awaits it directly, "
            f"so a sync implementation would either break or need an executor back."
        )


def test_the_router_does_not_reach_for_a_thread_pool():
    source = (LLM / "router.py").read_text()
    tree = ast.parse(source)
    offenders = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) in {"run_in_executor", "to_thread"}]
    assert not offenders, (
        f"router.py:{offenders} hands an LLM call to a thread pool again. The clients are async; awaiting them directly is the point."
    )


def test_no_blocking_sleep_anywhere_on_the_llm_path():
    """time.sleep in a coroutine blocks the loop; in a worker it wastes a thread."""
    offenders = []
    for path in LLM.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "sleep" and getattr(node.func.value, "id", None) == "time":
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, f"blocking time.sleep on the LLM path: {offenders}"


def test_the_assistant_family_is_async_too():
    """The twelve provider clients that came down from ascent_domain.omop."""
    from ascent_platform.llm.factory import available_assistant_types

    types_ = available_assistant_types()
    assert types_, "assistant registry is empty -- this test would assert nothing"
    for name, cls in sorted(types_.items()):
        assert inspect.iscoroutinefunction(cls.get_response), f"assistant {name!r} ({cls.__name__}) has a synchronous get_response"


def test_the_scan_would_notice_a_sync_send(tmp_path):
    """All-async and a broken detector look identical."""
    probe = tmp_path / "probe.py"
    probe.write_text("import time\ndef f():\n    time.sleep(1)\n")
    found = [
        n
        for n in ast.walk(ast.parse(probe.read_text()))
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "sleep" and getattr(n.func.value, "id", None) == "time"
    ]
    assert found, "detector missed a time.sleep"
