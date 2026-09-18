"""No medical-coder client may be unbounded.

An unbounded request hangs a caller indefinitely. There is one AscentClient,
and it takes its bound from ``medical_coder_timeout()``.

Which shapes what this file checks. Asserting "every ClientTimeout in this file
sets total" passes vacuously once the client stops constructing one inline, so
the client is checked for *using the helper* and the helper is checked for
setting the bound. The vacuity is the point: a substring version of this check
passes on prose in a comment that merely mentions total=None.

Checked by AST rather than grep, for the same reason.
"""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"

CLIENTS = [
    "ascent_platform/external/ascent_client.py",
]


def test_there_is_exactly_one_ascent_client():
    """A second copy drifts: one of them keeps unbounded timeouts long after
    the other is fixed."""
    found = sorted(str(p.relative_to(SRC)) for p in SRC.rglob("ascent_client.py") if "__pycache__" not in str(p))
    assert found == CLIENTS, f"AscentClient is defined in {found}"


@pytest.mark.parametrize("relpath", CLIENTS)
def test_client_takes_its_bound_from_the_shared_helper(relpath):
    """Either call medical_coder_timeout(), or build a ClientTimeout with a
    total. Anything else means the request is unbounded."""
    tree = ast.parse((SRC / relpath).read_text())
    uses_helper = any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "medical_coder_timeout" for n in ast.walk(tree))
    inline = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "ClientTimeout"]
    assert uses_helper or inline, f"{relpath} sets no aiohttp timeout at all"
    for call in inline:
        total = next((kw.value for kw in call.keywords if kw.arg == "total"), None)
        assert total is not None, f"{relpath}:{call.lineno} ClientTimeout has no total"
        assert not (isinstance(total, ast.Constant) and total.value is None), (
            f"{relpath}:{call.lineno} total=None — a slow-drip response never times out"
        )


def test_the_shared_helper_actually_bounds_the_request():
    """medical_coder_timeout() is the single definition of the bound, so the
    assertion above is only worth anything if this holds."""
    from ascent_platform.http import medical_coder_timeout

    t = medical_coder_timeout()
    assert t.total is not None and t.total > 0
    assert t.sock_connect is not None and t.sock_read is not None


@pytest.mark.parametrize("relpath", CLIENTS)
def test_async_path_does_not_call_the_blocking_token_refresh(relpath):
    """get_headers() sleeps between retries; calling it from a coroutine blocks
    the event loop for the whole backoff plus an untimed MSAL call.
    """
    tree = ast.parse((SRC / relpath).read_text())
    assert any(isinstance(n, ast.AsyncFunctionDef) and n.name == "query_ascent_api" for n in ast.walk(tree)), (
        f"{relpath} has no async query_ascent_api — this test would assert nothing"
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "query_ascent_api":
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == "get_headers":
                    pytest.fail(f"{relpath}:{inner.lineno} blocking get_headers() called from async code")


def test_shared_bounds_are_above_the_measured_tail():
    """sock_read must exceed the service's longest silent think time.

    The coder is silent while it runs LLM filtering, so a tight read timeout
    severs slow but healthy calls.
    """
    from ascent_platform.http import (
        MEDICAL_CODER_READ_TIMEOUT,
        MEDICAL_CODER_TOTAL_TIMEOUT,
    )

    assert MEDICAL_CODER_READ_TIMEOUT >= 250
    assert MEDICAL_CODER_TOTAL_TIMEOUT >= MEDICAL_CODER_READ_TIMEOUT
    assert MEDICAL_CODER_TOTAL_TIMEOUT <= 900, "no effective ceiling"
