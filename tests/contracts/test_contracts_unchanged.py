"""The frontend and the agents depend on these. Refactors must not move them.

MCP tool descriptions are part of the contract, not documentation: the model
reads them to choose a tool. A reworded docstring changes agent behaviour even
though no schema field moved, so descriptions are compared too.

If a diff here is intentional, regenerate with
    PYTHONPATH=./src python tests/contracts/capture_contracts.py
and let the fixture diff be the thing reviewed.
"""

import json
import pathlib

import pytest

from tests.contracts.capture_contracts import mcp_contract

FIX = pathlib.Path(__file__).parent / "fixtures"


def _diff(expected: dict, actual: dict):
    added = sorted(set(actual) - set(expected))
    removed = sorted(set(expected) - set(actual))
    changed = {k: {"was": expected[k], "now": actual[k]}
               for k in sorted(set(expected) & set(actual)) if expected[k] != actual[k]}
    return added, removed, changed


def test_mcp_tool_contract_is_unchanged():
    expected = json.loads((FIX / "mcp_tools.json").read_text())
    added, removed, changed = _diff(expected, mcp_contract())
    assert not removed, f"MCP tools disappeared (breaking for agents): {removed}"
    assert not changed, f"MCP tool signatures/descriptions changed: {json.dumps(changed, indent=2)[:2000]}"
    assert not added, f"new MCP tools -- intentional? regenerate the fixture: {added}"


def test_rest_contract_covers_every_route_the_app_serves():
    """The fixture must list exactly what the app exposes over HTTP.

    This used to assert `len(data) > 100` as a truncation guard, back when the
    REST API served 129 operations for the frontend. Those routers are gone --
    the distribution is MCP-only -- so a count threshold now says nothing.
    Naming the routes is a stronger guard anyway: it fails both when a capture
    silently produces nothing AND when a route quietly appears or disappears,
    which a count of "more than 100" would happily wave through.
    """
    data = json.loads((FIX / "rest_api.json").read_text())

    assert set(data) == {
        # Liveness. Deliberately dependency-free, unlike /mcp/health.
        "GET /api/public/health",
        # Rollup across the MCP servers and their dependencies.
        "GET /mcp/health",
        # Required by the MCP authorization spec for resource discovery.
        "GET /.well-known/oauth-protected-resource/mcp",
        "GET /.well-known/oauth-protected-resource/mcp/ascent-mcp-v1",
        "GET /.well-known/oauth-protected-resource/mcp/ascent-experimental",
    }, "the set of HTTP routes changed -- intentional? regenerate the fixture"

    health = data["GET /api/public/health"]["responses"]["200"]
    assert health.get("const") == "ok" or health.get("type") == "string"


def test_rest_contract_is_unchanged():
    """Builds the OpenAPI document in-process and diffs it against the fixture.

    No HTTP and no deployed app: FastAPI can produce the schema from the app
    object, which is what the frontend ultimately consumes. This is the gate for
    the remaining DTO work -- moving MetaDataResponse or changing what a route
    returns fails here with a readable diff.
    """
    from tests.contracts.capture_contracts import rest_contract

    try:
        from src.main import app
    except Exception as exc:  # pragma: no cover - config-dependent
        pytest.skip(f"app not importable in this environment: {type(exc).__name__}")

    actual = rest_contract(app.openapi())
    expected = json.loads((FIX / "rest_api.json").read_text())
    added, removed, changed = _diff(expected, actual)
    assert not removed, f"REST operations disappeared (breaking for the frontend): {removed}"
    assert not changed, (
        "REST request/response schemas changed:\n"
        + json.dumps(changed, indent=2)[:3000]
    )
    assert not added, f"new REST operations -- intentional? regenerate the fixture: {added}"
