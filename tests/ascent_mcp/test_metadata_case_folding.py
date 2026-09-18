"""INFORMATION_SCHEMA lookups must not care about identifier case.

The catalog does not agree with itself. Snowflake folds unquoted identifiers to
upper, so a table is ``PERSON``; the DuckDB warehouse this repo ships preserves
creation case, so the same table is ``person`` while its schema is still
``CDM``. An exact ``=`` matched the schema and missed the table, so
``describe_table("PERSON")`` -- the conventional OMOP spelling, and the one a
model reaches for -- returned ``{"columns": []}`` with no error. That is
indistinguishable from a table that genuinely has no columns, which is why it
went unnoticed.
"""

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from ascent_mcp import _tools_shared


class _CapturingExecutor:
    """Records the SQL it is handed and answers as a case-folding catalog would."""

    def __init__(self, rows: pd.DataFrame):
        self.rows = rows
        self.queries: list[str] = []

    async def execute_sql_query(self, query: str):
        self.queries.append(query)
        # Mimic a catalog that stores the table lower-cased and the schema
        # upper-cased: only a folded comparison can match both at once.
        if "UPPER(" not in query:
            return pd.DataFrame(columns=self.rows.columns)
        return self.rows


@pytest.fixture
def columns_frame():
    return pd.DataFrame(
        [
            {"COLUMN_NAME": "person_id", "DATA_TYPE": "BIGINT", "IS_NULLABLE": "NO"},
            {"COLUMN_NAME": "year_of_birth", "DATA_TYPE": "BIGINT", "IS_NULLABLE": "YES"},
        ]
    )


@pytest.mark.parametrize("table_name", ["PERSON", "person", "PeRsOn"])
async def test_describe_table_is_case_insensitive(columns_frame, table_name):
    executor = _CapturingExecutor(columns_frame)
    with (
        patch.object(_tools_shared, "authorize_user_db", AsyncMock()),
        patch.object(_tools_shared, "user_sql_executor", return_value=executor),
    ):
        result = await _tools_shared._describe_table_handler("DB", "CDM", table_name)

    assert [c["name"] for c in result["columns"]] == ["person_id", "year_of_birth"], (
        f"describe_table({table_name!r}) lost its columns to identifier case"
    )
    assert "UPPER(TABLE_NAME)" in executor.queries[0]
    assert "UPPER(TABLE_SCHEMA)" in executor.queries[0]


@pytest.mark.parametrize("schema", ["CDM", "cdm"])
async def test_list_tables_is_case_insensitive(schema):
    frame = pd.DataFrame([{"TABLE_NAME": "person"}, {"TABLE_NAME": "death"}])
    executor = _CapturingExecutor(frame)
    with (
        patch.object(_tools_shared, "authorize_user_db", AsyncMock()),
        patch.object(_tools_shared, "user_sql_executor", return_value=executor),
    ):
        result = await _tools_shared._list_tables_handler("DB", schema)

    assert result == {"tables": ["person", "death"], "count": 2}
    assert "UPPER(TABLE_SCHEMA)" in executor.queries[0]


def test_identifier_matching_still_rejects_injection():
    """Case folding must not become a hole in identifier validation."""
    for hostile in ("person'; DROP TABLE person; --", "a b", "x'", '"y"'):
        with pytest.raises(ValueError):
            _tools_shared._identifier_matches("TABLE_NAME", hostile, "table_name")
