"""Unit tests for synthetic-data detection.

The data half (which databases are synthetic) is ascent_domain.synthetic; the
presentation half (the warning, the tool decorator, the annotation helper) is
ascent_mcp._synthetic.

Covers the ``inject_synthetic_warning`` decorator, the Snowflake fetch
helpers, the Redis-cached ``get_synthetic_databases`` lookup, and the
``annotate_databases_with_synthetic_flag`` list helper.
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ascent_domain.synthetic import _fetch_synthetic_databases, get_synthetic_databases
from ascent_mcp._synthetic import (
    SYNTHETIC_DATA_WARNING,
    annotate_databases_with_synthetic_flag,
    inject_synthetic_warning,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(return_value):
    """Create a mock async tool function decorated with inject_synthetic_warning."""

    @inject_synthetic_warning
    async def tool_func(*, database_name: str = "", **kwargs):
        return return_value

    return tool_func


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("ascent_mcp._synthetic.is_synthetic_database", new_callable=AsyncMock, return_value=True)
async def test_synthetic_database_injects_warning(mock_is_synthetic):
    """synthetic_warning key is injected for synthetic databases."""
    tool = _make_tool({"data": "test"})
    result = await tool(database_name="SYNTHETIC_EHR_OMOP")

    assert "synthetic_warning" in result
    assert "SYNTHETIC_EHR_OMOP" in result["synthetic_warning"]
    assert "synthetic data" in result["synthetic_warning"]
    assert result["data"] == "test"
    mock_is_synthetic.assert_awaited_once_with("SYNTHETIC_EHR_OMOP")


@pytest.mark.asyncio
@patch("ascent_mcp._synthetic.is_synthetic_database", new_callable=AsyncMock, return_value=False)
async def test_non_synthetic_database_passes_through(mock_is_synthetic):
    """Dict is returned unchanged for non-synthetic databases."""
    original = {"answer": 42}
    tool = _make_tool(original)
    result = await tool(database_name="REAL_DB")

    assert result is original
    assert "synthetic_warning" not in result
    mock_is_synthetic.assert_awaited_once_with("REAL_DB")


@pytest.mark.asyncio
@patch("ascent_mcp._synthetic.is_synthetic_database", new_callable=AsyncMock)
async def test_no_database_name_passes_through(mock_is_synthetic):
    """Dict is returned unchanged when database_name kwarg is absent."""

    @inject_synthetic_warning
    async def tool_func(**kwargs):
        return {"answer": 42}

    result = await tool_func()

    assert result == {"answer": 42}
    assert "synthetic_warning" not in result
    mock_is_synthetic.assert_not_awaited()


@pytest.mark.asyncio
@patch("ascent_mcp._synthetic.is_synthetic_database", new_callable=AsyncMock, side_effect=RuntimeError("Snowflake down"))
async def test_synthetic_check_failure_passes_through(mock_is_synthetic):
    """If is_synthetic_database raises, dict is returned unchanged (no crash)."""
    original = {"answer": 42}
    tool = _make_tool(original)
    result = await tool(database_name="SOME_DB")

    assert result is original
    assert "synthetic_warning" not in result


@pytest.mark.asyncio
@patch("ascent_mcp._synthetic.is_synthetic_database", new_callable=AsyncMock, return_value=True)
async def test_non_dict_return_value_unchanged(mock_is_synthetic):
    """Non-dict return values are returned unchanged."""

    @inject_synthetic_warning
    async def tool_func(*, database_name: str = ""):
        return ["item1", "item2"]

    result = await tool_func(database_name="SYNTH_DB")

    assert result == ["item1", "item2"]
    mock_is_synthetic.assert_not_awaited()


@pytest.mark.asyncio
@patch("ascent_mcp._synthetic.is_synthetic_database", new_callable=AsyncMock, return_value=True)
async def test_warning_text_contains_database_name(mock_is_synthetic):
    """The warning text matches the template with the actual database name."""
    tool = _make_tool({"answer": 42})
    result = await tool(database_name="MY_SYNTH_DB")

    expected_warning = SYNTHETIC_DATA_WARNING.format(database_name="MY_SYNTH_DB")
    assert result["synthetic_warning"] == expected_warning


@pytest.mark.asyncio
@patch("ascent_mcp._synthetic.is_synthetic_database", new_callable=AsyncMock, return_value=True)
async def test_original_keys_preserved(mock_is_synthetic):
    """Original dict keys are preserved alongside the warning."""
    tool = _make_tool({"main_answer": "42 patients", "query_filled": "SELECT 1"})
    result = await tool(database_name="SYNTH_DB")

    assert result["main_answer"] == "42 patients"
    assert result["query_filled"] == "SELECT 1"
    assert "synthetic_warning" in result


@pytest.mark.asyncio
@patch("ascent_mcp._synthetic.is_synthetic_database", new_callable=AsyncMock)
async def test_empty_database_name_passes_through(mock_is_synthetic):
    """Empty string database_name is treated as absent."""
    tool = _make_tool({"answer": 42})
    result = await tool(database_name="")

    assert "synthetic_warning" not in result
    mock_is_synthetic.assert_not_awaited()


@pytest.mark.asyncio
@patch("ascent_mcp._synthetic.is_synthetic_database", new_callable=AsyncMock, side_effect=asyncio.CancelledError)
async def test_cancelled_error_propagates(mock_is_synthetic):
    """asyncio.CancelledError is not swallowed — it must propagate for proper task cancellation."""
    tool = _make_tool({"answer": 42})

    with pytest.raises(asyncio.CancelledError):
        await tool(database_name="SOME_DB")


# ---------------------------------------------------------------------------
# _fetch_synthetic_databases tests
#
# _fetch_synthetic_databases uses lazy imports (from ascent_platform.snowflake.session
# import ...) so we inject a mock module into sys.modules for the duration of
# each test via a fixture.
# ---------------------------------------------------------------------------


@pytest.fixture()
def snowflake_mocks(monkeypatch):
    """Provide mock CONNECTION_POOL, execute_query, and ProgrammingError.

    Injects a mock ``ascent_platform.snowflake.session`` module so the lazy import
    inside ``_fetch_synthetic_databases`` resolves without side effects.
    """
    cursor = MagicMock()
    cursor.close = MagicMock()
    connection = MagicMock()
    connection.cursor.return_value = cursor

    pool = AsyncMock()
    pool.get_connection.return_value = connection
    pool.release_connection = MagicMock()

    mock_execute = AsyncMock()

    mock_module = MagicMock()
    mock_module.CONNECTION_POOL = pool
    mock_module.execute_query = mock_execute

    monkeypatch.setitem(sys.modules, "ascent_platform.snowflake.session", mock_module)

    return SimpleNamespace(
        pool=pool,
        connection=connection,
        cursor=cursor,
        execute_query=mock_execute,
    )


async def test_fetch_returns_every_shipped_database():
    """All shipped data is generated, so every database is synthetic.

    These five tests used to mock two Snowflake metadata tables and assert the
    IS_SYNTHETIC rows came back merged. There is no such query any more: the
    open-source build ships nothing but generated data, so the answer is the
    shipped database list. Mocking a query that is never made passed
    vacuously against whatever the real function returned.
    """
    from ascent_platform.warehouse.bootstrap import DATABASES

    result = await _fetch_synthetic_databases()

    assert result == {db.name for db in DATABASES}


async def test_fetch_never_reports_an_empty_set():
    """An empty set would hide every database from a restricted caller.

    Callers treat this as an allowlist, so "nothing is synthetic" and "nothing
    is visible" are the same answer -- the failure mode this guards.
    """
    assert await _fetch_synthetic_databases()


# ---------------------------------------------------------------------------
# get_synthetic_databases (caching) tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("ascent_domain.synthetic._fetch_synthetic_databases", new_callable=AsyncMock)
@patch("ascent_platform.cache.aiocache_backend.cache")
async def test_get_returns_cached_value_on_hit(mock_cache, mock_fetch):
    """Cached non-None value is returned without querying Snowflake."""
    mock_cache.get = AsyncMock(return_value={"CACHED_DB"})

    result = await get_synthetic_databases()

    assert result == {"CACHED_DB"}
    mock_fetch.assert_not_awaited()


@pytest.mark.asyncio
@patch("ascent_domain.synthetic._fetch_synthetic_databases", new_callable=AsyncMock)
@patch("ascent_platform.cache.aiocache_backend.cache")
async def test_get_honors_cached_empty_set(mock_cache, mock_fetch):
    """An empty set in cache is a valid hit — Snowflake is not re-queried."""
    mock_cache.get = AsyncMock(return_value=set())

    result = await get_synthetic_databases()

    assert result == set()
    mock_fetch.assert_not_awaited()


@pytest.mark.asyncio
@patch("ascent_domain.synthetic._fetch_synthetic_databases", new_callable=AsyncMock, return_value={"FRESH_DB"})
@patch("ascent_platform.cache.aiocache_backend.cache")
async def test_get_fetches_and_caches_on_miss(mock_cache, mock_fetch):
    """Cache miss (None) triggers Snowflake fetch and stores result."""
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.set = AsyncMock()

    result = await get_synthetic_databases()

    assert result == {"FRESH_DB"}
    mock_fetch.assert_awaited_once()
    mock_cache.set.assert_awaited_once()
    # Verify the cached value and that a TTL was provided
    call_args = mock_cache.set.call_args
    assert call_args[0][1] == {"FRESH_DB"}
    assert "ttl" in call_args[1]


@pytest.mark.asyncio
@patch("ascent_domain.synthetic._fetch_synthetic_databases", new_callable=AsyncMock, side_effect=RuntimeError("Snowflake down"))
@patch("ascent_platform.cache.aiocache_backend.cache")
async def test_get_propagates_fetch_error(mock_cache, mock_fetch):
    """Transient Snowflake errors propagate so callers can handle them."""
    mock_cache.get = AsyncMock(return_value=None)

    with pytest.raises(RuntimeError, match="Snowflake down"):
        await get_synthetic_databases()


# ---------------------------------------------------------------------------
# annotate_databases_with_synthetic_flag tests
# ---------------------------------------------------------------------------


def _make_pair(name: str, schemas: list[str] | None = None):
    """Build a DatabasePair-like object for the annotator."""
    return SimpleNamespace(
        database_name=name,
        database_schemas=schemas if schemas is not None else ["PUBLIC"],
    )


@pytest.mark.asyncio
@patch("ascent_mcp._synthetic.get_synthetic_databases", new_callable=AsyncMock, return_value={"SYNTH_DB_A"})
async def test_annotate_mixed_pairs(mock_get):
    """Entries are flagged correctly based on membership in the synthetic set."""
    pairs = [_make_pair("SYNTH_DB_A"), _make_pair("REAL_DB_B")]

    result = await annotate_databases_with_synthetic_flag(pairs)

    assert result == [
        {"database_name": "SYNTH_DB_A", "database_schemas": ["PUBLIC"], "is_synthetic": True},
        {"database_name": "REAL_DB_B", "database_schemas": ["PUBLIC"], "is_synthetic": False},
    ]


@pytest.mark.asyncio
@patch("ascent_mcp._synthetic.get_synthetic_databases", new_callable=AsyncMock, return_value={"A", "B"})
async def test_annotate_all_synthetic(mock_get):
    """When every pair is in the synthetic set, all entries are flagged True."""
    pairs = [_make_pair("A"), _make_pair("B")]

    result = await annotate_databases_with_synthetic_flag(pairs)

    assert all(entry["is_synthetic"] is True for entry in result)


@pytest.mark.asyncio
@patch("ascent_mcp._synthetic.get_synthetic_databases", new_callable=AsyncMock, return_value=set())
async def test_annotate_none_synthetic(mock_get):
    """Empty synthetic set produces all-False flags."""
    pairs = [_make_pair("A"), _make_pair("B")]

    result = await annotate_databases_with_synthetic_flag(pairs)

    assert all(entry["is_synthetic"] is False for entry in result)


@pytest.mark.asyncio
@patch("ascent_mcp._synthetic.get_synthetic_databases", new_callable=AsyncMock, side_effect=RuntimeError("Snowflake down"))
async def test_annotate_fail_safe_on_lookup_error(mock_get):
    """Listing still returns when the synthetic lookup raises; all flags default to False."""
    pairs = [_make_pair("A"), _make_pair("B")]

    result = await annotate_databases_with_synthetic_flag(pairs)

    assert len(result) == 2
    assert all(entry["is_synthetic"] is False for entry in result)


@pytest.mark.asyncio
@patch("ascent_mcp._synthetic.get_synthetic_databases", new_callable=AsyncMock, return_value=set())
async def test_annotate_empty_input(mock_get):
    """Empty input yields an empty list."""
    assert await annotate_databases_with_synthetic_flag([]) == []


@pytest.mark.asyncio
@patch("ascent_mcp._synthetic.get_synthetic_databases", new_callable=AsyncMock, return_value={"A"})
async def test_annotate_preserves_schemas(mock_get):
    """database_schemas are passed through unchanged."""
    pairs = [_make_pair("A", ["S1", "S2"])]

    result = await annotate_databases_with_synthetic_flag(pairs)

    assert result[0]["database_schemas"] == ["S1", "S2"]
