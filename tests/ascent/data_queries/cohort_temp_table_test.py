"""Unit tests for cohort temp-table materialization and cohort-ref SQL rewriting.

Cohort-scoped question SQL references ASCENT.ASCENT_COHORTS.<table>, which only
the machine user can read. These helpers copy the cohort into a session-scoped
TEMPORARY table on the user's RWD connection and rewrite the SQL to point at it.
"""

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import duckdb
import pandas as pd
import pytest

from ascent_domain.data_queries.cohort import (
    TEMP_COHORT_TABLE_PREFIX,
    CohortTableMissingError,
    _temp_table_spec,
    cohort_temp_table,
    healed_away_cohort_access,
    restore_cohort_table_ref,
    rewrite_cohort_table_ref,
)
from ascent_domain.models.data_definitions import ColumnInfo, UserCohortDescriptor
from ascent_http.error_mapping import status_for


def _descriptor(**overrides) -> UserCohortDescriptor:
    defaults = dict(
        database="SYNTHETIC_CLAIMS",
        database_schema="CDM",
        table="DEV_USR_ABC123",
        size=2,
        attributes=[
            ColumnInfo(name="SUBJECT_ID", type="NUMBER"),
            ColumnInfo(name="INDEX_DATE", type="DATE"),
        ],
        owner_id="test@example.com",
    )
    defaults.update(overrides)
    return UserCohortDescriptor(**defaults)


# ---------------------------------------------------------------------------
# rewrite / restore
# ---------------------------------------------------------------------------


def test_rewrite_replaces_fully_qualified_ref():
    sql = "WITH c AS (SELECT * FROM ASCENT.ASCENT_COHORTS.DEV_USR_ABC123) SELECT COUNT(*) FROM c"
    out = rewrite_cohort_table_ref(sql, "DEV_USR_ABC123", "TMP_COHORT_X")
    assert "ASCENT" not in out
    assert "FROM TMP_COHORT_X" in out


def test_rewrite_replaces_schema_qualified_ref_and_is_case_insensitive():
    sql = "SELECT * FROM ascent_cohorts.dev_usr_abc123 t"
    out = rewrite_cohort_table_ref(sql, "DEV_USR_ABC123", "TMP_COHORT_X")
    assert out == "SELECT * FROM TMP_COHORT_X t"


def test_rewrite_does_not_touch_other_tables():
    sql = "SELECT * FROM ASCENT.ASCENT_COHORTS.DEV_USR_ABC1234"  # longer suffix, different cohort
    assert rewrite_cohort_table_ref(sql, "DEV_USR_ABC123", "TMP") == sql


def test_rewrite_rejects_malicious_table_name():
    with pytest.raises(ValueError):
        rewrite_cohort_table_ref("SELECT 1", "BAD;DROP TABLE X", "TMP")


def test_restore_roundtrips():
    original = "SELECT COUNT(*) FROM ASCENT.ASCENT_COHORTS.DEV_USR_ABC123 c JOIN diag d ON d.patid = c.SUBJECT_ID"
    rewritten = rewrite_cohort_table_ref(original, "DEV_USR_ABC123", "TMP_COHORT_X")
    assert restore_cohort_table_ref(rewritten, "DEV_USR_ABC123", "TMP_COHORT_X") == original


# ---------------------------------------------------------------------------
# healed_away_cohort_access
# ---------------------------------------------------------------------------


def test_healed_away_cohort_access_detects_ascent_auth_error():
    result = SimpleNamespace(
        self_healing_attempted=True,
        original_error_message="002003 (02000): SQL compilation error:\nDatabase 'ASCENT' does not exist or not authorized.",
    )
    assert healed_away_cohort_access(result) is True


@pytest.mark.parametrize(
    "healing, message",
    [
        (False, "Database 'ASCENT' does not exist or not authorized."),
        (True, None),
        (True, "invalid identifier 'FOO'"),
    ],
)
def test_healed_away_cohort_access_negative_cases(healing, message):
    result = SimpleNamespace(self_healing_attempted=healing, original_error_message=message)
    assert healed_away_cohort_access(result) is False


# ---------------------------------------------------------------------------
# _temp_table_spec
# ---------------------------------------------------------------------------


def test_temp_table_spec_maps_lowercase_df_columns():
    df = pd.DataFrame({"subject_id": [1], "index_date": [date(2020, 1, 1)]})
    spec = _temp_table_spec(df, _descriptor())
    assert spec == [("subject_id", "SUBJECT_ID", "NUMBER"), ("index_date", "INDEX_DATE", "DATE")]


def test_temp_table_spec_missing_column_raises():
    df = pd.DataFrame({"subject_id": [1]})
    with pytest.raises(ValueError, match="missing expected column"):
        _temp_table_spec(df, _descriptor())


def test_temp_table_spec_rejects_unsafe_type():
    df = pd.DataFrame({"subject_id": [1], "index_date": [date(2020, 1, 1)]})
    descriptor = _descriptor(
        attributes=[
            ColumnInfo(name="SUBJECT_ID", type="NUMBER); DROP TABLE X"),
            ColumnInfo(name="INDEX_DATE", type="DATE"),
        ]
    )
    with pytest.raises(ValueError, match="Unsupported column type"):
        _temp_table_spec(df, descriptor)


# ---------------------------------------------------------------------------
# cohort_temp_table
# ---------------------------------------------------------------------------


def _mock_ascent_db(mock_get_db):
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=MagicMock())
    cm.__aexit__ = AsyncMock(return_value=False)
    mock_get_db.return_value = cm


@pytest.mark.asyncio
@patch("ascent_domain.data_queries.cohort.execute_query", new_callable=AsyncMock)
@patch("ascent_domain.data_queries.cohort.fetch_cohort_dataframe", new_callable=AsyncMock)
@patch("ascent_domain.data_queries.cohort.get_db")
async def test_cohort_temp_table_materializes_inserts_and_drops(mock_get_db, mock_fetch, mock_execute):
    _mock_ascent_db(mock_get_db)
    mock_fetch.return_value = pd.DataFrame({"subject_id": [1, 2], "index_date": [date(2020, 1, 1), None]})
    descriptor = _descriptor()
    connection = MagicMock()

    async with cohort_temp_table(connection, descriptor) as temp_table:
        assert temp_table == f"{TEMP_COHORT_TABLE_PREFIX}{descriptor.id.hex.upper()}"
        # Cohort rows are read from ASCENT via the machine user.
        mock_get_db.assert_called_once_with("ASCENT")
        assert "FROM ASCENT_COHORTS.DEV_USR_ABC123" in mock_fetch.await_args.args[1]

    statements = [call.args[1] for call in mock_execute.await_args_list]
    assert f"CREATE OR REPLACE TEMPORARY TABLE {temp_table} (SUBJECT_ID NUMBER, INDEX_DATE DATE);" in statements[0]

    insert_call = mock_execute.await_args_list[1]
    assert f"INSERT INTO {temp_table} (SUBJECT_ID, INDEX_DATE) VALUES (?, ?);" in insert_call.args[1]
    assert insert_call.kwargs["bulk"] is True
    rows = [tuple(r) for r in insert_call.kwargs["params"]]
    assert rows == [(1, "2020-01-01"), (2, None)]  # dates normalized, NaN/NaT as None

    assert f"DROP TABLE IF EXISTS {temp_table};" in statements[-1]
    # All temp-table statements run on the caller's session, not the ASCENT cursor.
    for call in mock_execute.await_args_list:
        assert call.args[0] is connection.cursor.return_value


@pytest.mark.asyncio
@patch("ascent_domain.data_queries.cohort.execute_query", new_callable=AsyncMock)
@patch("ascent_domain.data_queries.cohort.fetch_cohort_dataframe", new_callable=AsyncMock)
@patch("ascent_domain.data_queries.cohort.get_db")
async def test_cohort_temp_table_empty_cohort_skips_insert(mock_get_db, mock_fetch, mock_execute):
    _mock_ascent_db(mock_get_db)
    mock_fetch.return_value = pd.DataFrame({"subject_id": [], "index_date": []})

    async with cohort_temp_table(MagicMock(), _descriptor()) as temp_table:
        pass

    statements = [call.args[1] for call in mock_execute.await_args_list]
    assert len(statements) == 2  # CREATE + DROP, no INSERT
    assert "CREATE OR REPLACE TEMPORARY TABLE" in statements[0]
    assert f"DROP TABLE IF EXISTS {temp_table};" in statements[1]


@pytest.mark.asyncio
@patch("ascent_domain.data_queries.cohort.execute_query", new_callable=AsyncMock)
@patch("ascent_domain.data_queries.cohort.fetch_cohort_dataframe", new_callable=AsyncMock)
@patch("ascent_domain.data_queries.cohort.get_db")
async def test_cohort_temp_table_drops_even_when_body_raises(mock_get_db, mock_fetch, mock_execute):
    _mock_ascent_db(mock_get_db)
    mock_fetch.return_value = pd.DataFrame({"subject_id": [1], "index_date": [date(2020, 1, 1)]})

    with pytest.raises(RuntimeError, match="query blew up"):
        async with cohort_temp_table(MagicMock(), _descriptor()) as temp_table:
            raise RuntimeError("query blew up")

    assert f"DROP TABLE IF EXISTS {temp_table};" in mock_execute.await_args_list[-1].args[1]


@pytest.mark.asyncio
@patch("ascent_domain.data_queries.cohort.execute_query", new_callable=AsyncMock)
@patch("ascent_domain.data_queries.cohort.fetch_cohort_dataframe", new_callable=AsyncMock)
@patch("ascent_domain.data_queries.cohort.get_db")
async def test_cohort_temp_table_rejects_malicious_source_table(mock_get_db, mock_fetch, mock_execute):
    with pytest.raises(ValueError):
        async with cohort_temp_table(MagicMock(), _descriptor(table="EVIL;DROP")):
            pass

    mock_get_db.assert_not_called()
    mock_execute.assert_not_called()


@pytest.mark.asyncio
@patch("ascent_domain.data_queries.cohort.execute_query", new_callable=AsyncMock)
@patch("ascent_domain.data_queries.cohort.fetch_cohort_dataframe", new_callable=AsyncMock)
@patch("ascent_domain.data_queries.cohort.get_db")
async def test_cohort_temp_table_missing_source_raises_cohort_table_missing(mock_get_db, mock_fetch, mock_execute):
    """A garbage-collected/dropped cohort table surfaces as a clean 409, not a raw warehouse error.

    The exception is the one DuckDB actually raises for a missing table. A
    Snowflake error stands in for nothing here: it cannot reach this handler,
    so simulating one tests the ``except`` clause against an exception the
    warehouse never produces.
    """
    _mock_ascent_db(mock_get_db)
    mock_fetch.side_effect = duckdb.CatalogException("Catalog Error: Table with name DEV_USR_ABC123 does not exist!")

    with pytest.raises(CohortTableMissingError) as excinfo:
        async with cohort_temp_table(MagicMock(), _descriptor()):
            pass

    assert status_for(excinfo.value) == 409
    assert "DEV_USR_ABC123" in str(excinfo.value.detail)
    mock_execute.assert_not_called()  # no temp table DDL was attempted
