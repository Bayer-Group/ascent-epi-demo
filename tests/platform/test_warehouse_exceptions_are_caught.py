"""The handlers that recover from a bad query must catch what the warehouse raises.

DuckDB exports its own ``ProgrammingError`` and ``DatabaseError`` under the
same names the Snowflake connector uses, so an ``except`` clause importing them
from the wrong package still reads correctly and never matches. The SQL
self-healing loop and the missing-cohort-table branch both depended on it.
"""

from __future__ import annotations

import duckdb
import pytest


def _raise(sql: str) -> BaseException:
    con = duckdb.connect(":memory:")
    try:
        con.execute(sql)
    except BaseException as exc:  # noqa: BLE001 - the point is what type comes out
        return exc
    finally:
        con.close()
    raise AssertionError(f"{sql!r} did not raise")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM no_such_table",
        "SELECT no_such_column FROM (SELECT 1 AS a)",
        "SELECT 1 +",
    ],
)
def test_warehouse_errors_are_duckdb_errors(sql):
    from duckdb import Error as WarehouseError

    assert isinstance(_raise(sql), WarehouseError)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM no_such_table",
        "SELECT no_such_column FROM (SELECT 1 AS a)",
    ],
)
def test_snowflake_exceptions_never_match(sql):
    """The shape of the original defect, pinned so it cannot come back."""
    from snowflake.connector import DatabaseError, ProgrammingError

    assert not isinstance(_raise(sql), (ProgrammingError, DatabaseError))


def test_the_missing_table_branch_recognises_duckdbs_wording():
    from ascent_domain.data_queries import cohort  # noqa: F401  (import guard)

    assert "does not exist" in str(_raise("SELECT * FROM no_such_table"))


def test_the_self_healing_loop_can_read_the_message():
    """handle_db_exception feeds ``db_ex.args[0]`` to the repair prompt."""
    exc = _raise("SELECT 1 +")
    assert exc.args and isinstance(exc.args[0], str)
