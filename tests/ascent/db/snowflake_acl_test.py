from typing import Set
from unittest.mock import AsyncMock, patch

from ascent_domain.database_access import (
    ACL_CACHE_KEY,
    _get_allowlist,
    _get_allowlist_cached,
    _show_databases_for_user,
    get_user_databases,
)
from ascent_platform.constants import ExpirationTTL

# ---------------------------------------------------------------------------
# _get_allowlist
# ---------------------------------------------------------------------------


async def test_get_allowlist():
    """The allowlist is the shipped database set, not a metadata query."""
    from ascent_platform.warehouse.bootstrap import DATABASES

    assert await _get_allowlist() == {db.name for db in DATABASES}


# ---------------------------------------------------------------------------
# _get_allowlist_cached
# ---------------------------------------------------------------------------


@patch("ascent_domain.database_access.cache", new_callable=AsyncMock)
async def test_get_allowlist_cached_hit(mock_cache):
    expected: Set[str] = {"DB_A_OMOP", "DB_B_OMOP"}
    mock_cache.get.return_value = expected

    result = await _get_allowlist_cached()

    assert result == expected
    mock_cache.get.assert_called_once_with(ACL_CACHE_KEY)
    mock_cache.set.assert_not_called()


@patch("ascent_domain.database_access.cache", new_callable=AsyncMock)
@patch("ascent_domain.database_access._get_allowlist", new_callable=AsyncMock)
async def test_get_allowlist_cached_miss(mock_get_allowlist, mock_cache):
    """A cache miss computes the allowlist once and stores it with a TTL."""
    expected: Set[str] = {"DB_A_OMOP", "DB_B_OMOP"}
    mock_cache.get.return_value = None
    mock_get_allowlist.return_value = expected

    result = await _get_allowlist_cached()

    assert result == expected
    mock_get_allowlist.assert_awaited_once()
    mock_cache.set.assert_called_once_with(ACL_CACHE_KEY, expected, ttl=ExpirationTTL.HOUR * 3)


# ---------------------------------------------------------------------------
# _show_databases_for_user
# ---------------------------------------------------------------------------


async def test_show_databases_for_user():
    """Everything shipped is visible.

    There is one identity and no grant model, so there is not even a user
    connection to open.
    """
    from ascent_platform.warehouse.bootstrap import DATABASES

    result = await _show_databases_for_user("fake-oauth-token")

    assert result >= {db.name for db in DATABASES}


# ---------------------------------------------------------------------------
# get_user_databases
# ---------------------------------------------------------------------------


@patch("ascent_domain.database_access._get_allowlist_cached", new_callable=AsyncMock)
@patch("ascent_domain.database_access._show_databases_for_user", new_callable=AsyncMock)
async def test_get_user_databases_intersection(mock_show_dbs, mock_allowlist):
    mock_allowlist.return_value = {"DB_A_OMOP", "DB_B_OMOP", "DB_C_OMOP"}
    mock_show_dbs.return_value = {"DB_A_OMOP", "DB_C_OMOP", "SNOWFLAKE", "SOME_OTHER_DB"}

    result = await get_user_databases("fake-token")

    # Only databases in BOTH allowlist AND user's Snowflake role are returned
    assert result == {"DB_A_OMOP", "DB_C_OMOP"}
    mock_allowlist.assert_called_once()
    mock_show_dbs.assert_called_once_with("fake-token")


@patch("ascent_domain.database_access._get_allowlist_cached", new_callable=AsyncMock)
@patch("ascent_domain.database_access._show_databases_for_user", new_callable=AsyncMock)
async def test_get_user_databases_no_overlap(mock_show_dbs, mock_allowlist):
    mock_allowlist.return_value = {"DB_A_OMOP", "DB_B_OMOP"}
    mock_show_dbs.return_value = {"SNOWFLAKE", "SOME_INTERNAL_DB"}

    result = await get_user_databases("fake-token")

    assert result == set()


@patch("ascent_domain.database_access._get_allowlist_cached", new_callable=AsyncMock)
@patch("ascent_domain.database_access._show_databases_for_user", new_callable=AsyncMock)
async def test_get_user_databases_full_access(mock_show_dbs, mock_allowlist):
    allowlist = {"DB_A_OMOP", "DB_B_OMOP", "DB_C_OMOP"}
    mock_allowlist.return_value = allowlist
    mock_show_dbs.return_value = allowlist | {"SNOWFLAKE"}

    result = await get_user_databases("fake-token")

    assert result == allowlist
