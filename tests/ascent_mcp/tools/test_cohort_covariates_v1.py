"""Unit tests for the v1 cohort covariate tooling.

Covers the pure validation/column helpers and the early-return guards of the
``cohort_add_covariates`` / ``share_cohort`` tools (no Snowflake/Postgres
needed). The full persistence path is covered by the data-layer tests in
``tests/ascent/data_queries/cohort_crud_test.py`` and the local MCP e2e.
"""

from unittest.mock import AsyncMock

import pytest

# Force-load data_definitions first to resolve the import cycle pulled in by
# tools_v1_cohort -> snowflake_session (see test_resolve_placeholders_v1).
import ascent_domain.models.data_definitions  # noqa: F401, E402
from ascent_domain.models.data_definitions import ColumnInfo, TableInfo  # noqa: E402
from ascent_mcp.tools_v1_cohort import (  # noqa: E402
    _assert_read_only_select,
    _covariate_columns,
    _covariates_from_table_info,
    _find_id_column,
    cohort_add_covariates,
    share_cohort,
)


def _ctx():
    ctx = AsyncMock()
    ctx.report_progress = AsyncMock()
    return ctx


# ─── read-only SQL guard ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT person_id, age FROM patients",
        "  select person_id from t ;",
        "WITH c AS (SELECT 1) SELECT * FROM c",
    ],
)
def test_assert_read_only_select_accepts(sql):
    _assert_read_only_select(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM patients",
        "SELECT 1; DROP TABLE patients",
        "UPDATE patients SET age = 1",
        "INSERT INTO t VALUES (1)",
        "CREATE TABLE x (a int)",
        "EXPLAIN SELECT 1",
    ],
)
def test_assert_read_only_select_rejects(sql):
    with pytest.raises(ValueError):
        _assert_read_only_select(sql)


# ─── column helpers ───────────────────────────────────────────────────────────


def test_find_id_column():
    assert _find_id_column(["person_id", "age"]) == "person_id"
    assert _find_id_column(["subject_id", "age"]) == "subject_id"
    assert _find_id_column(["age", "sex"]) is None


def test_covariate_columns_excludes_base():
    assert _covariate_columns(["person_id", "index_date", "age", "sex"]) == ["age", "sex"]


def test_covariates_from_table_info_excludes_base_columns():
    table_info = TableInfo(
        name="T",
        size=1,
        size_bytes=1,
        columns=[
            ColumnInfo(name="SUBJECT_ID", type="NUMBER"),
            ColumnInfo(name="INDEX_DATE", type="DATE"),
            ColumnInfo(name="AGE", type="NUMBER"),
        ],
    )
    covs = _covariates_from_table_info(table_info, source="inline", definition=None)
    assert [c.name for c in covs] == ["AGE"]
    assert covs[0].source == "inline"


# ─── tool early-return guards ─────────────────────────────────────────────────


async def test_cohort_add_covariates_invalid_uuid():
    result = await cohort_add_covariates("not-a-uuid", "SELECT person_id, age FROM t", _ctx())
    assert "error" in result


async def test_cohort_add_covariates_rejects_non_select():
    result = await cohort_add_covariates("123e4567-e89b-12d3-a456-426614174000", "DELETE FROM patients", _ctx())
    assert "error" in result


async def test_share_cohort_invalid_uuid():
    result = await share_cohort("not-a-uuid", "user@example.com")
    assert "error" in result
