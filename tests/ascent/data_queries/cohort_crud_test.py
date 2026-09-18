import uuid
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from pydantic import HttpUrl
from sqlalchemy import delete, update

from ascent_domain.data_queries.cohort import (
    OWNED_COHORT_TABLE_PREFIX,
    QUERY_USER_COHORT_TABLES,
    QUERY_USER_COHORT_TABLES_OWNED,
    CohortAccessDeniedError,
    CohortDemographicsUnsupportedError,
    CohortNotFoundError,
    CohortNotQueryableError,
    _get_cohort_table_info,
    _infer_sf_type,
    _sanitize_identifier,
    authorize_cohort,
    default_cohort_name,
    delete_abandoned_cohort_descriptors,
    delete_cohort_descriptors,
    delete_orphaned_cohort_tables,
    ensure_cohort_demographics_supported,
    ensure_cohort_queryable,
    flag_orphaned_cohort_descriptors,
    get_cohort_descriptor,
    get_cohort_descriptors,
    get_studies,
    grant_cohort_access,
    replace_cohort_columns,
    resolve_cohort_role,
    revoke_cohort_access,
    store_cohort,
    store_cohort_descriptor,
    update_cohort_descriptor,
    update_cohort_descriptor_funnel,
)
from ascent_domain.models.data_definitions import (
    ColumnInfo,
    Criteria,
    DBCohortDescriptor,
    DBFunnelCriterion,
    FunnelCriterion,
    TableInfo,
    UserCohortDescriptor,
    UserCohortDescriptorPatch,
    role_at_least,
)
from ascent_http.error_mapping import status_for
from ascent_http.settings import settings
from tests.ascent.data_queries.data_queries_test import assert_query_equal
from tests.ascent.utils.concept_list_mother import default_medical_concept


@pytest.fixture
def mock_funnel():
    return [dict(id=uuid.uuid4(), index=0, description="Some criterion", query_template="query template", query="query", type="include", size=1337)]


@pytest.fixture
def mock_study_cohort_id():
    return uuid.uuid4()


@pytest.fixture
def mock_study_cohort(mock_study_cohort_id, mock_current_user):
    cohort = MagicMock()
    cohort.id = mock_study_cohort_id
    cohort.name = "Test Study Cohort"
    cohort.created = datetime.now()
    cohort.state = "pending"
    cohort.description = "Test description"
    cohort.origin = "study"
    cohort.owner_id = mock_current_user.email
    cohort.database = "TEST"
    cohort.database_schema = "TEST_SCHEMA"
    cohort.table = "STUDY_COHORT"
    cohort.size = 42
    cohort.attributes = [dict(name="SUBJECT_ID", type="NUMBER")]
    cohort.study_id = uuid.uuid4()
    cohort.funnel = None
    cohort.user_concepts = None
    cohort.access = []
    return cohort


@pytest.fixture
def mock_user_cohort_id():
    return uuid.uuid4()


@pytest.fixture
def mock_user_cohort(mock_user_cohort_id, mock_funnel, mock_current_user):
    cohort = MagicMock()
    cohort.id = mock_user_cohort_id
    cohort.name = "Test User Cohort"
    cohort.created = datetime.now()
    cohort.state = "active"
    cohort.description = "Test description"
    cohort.origin = "user"
    cohort.database = "TEST"
    cohort.database_schema = "TEST_SCHEMA"
    cohort.table = "USER_COHORT"
    cohort.size = 42
    cohort.attributes = [dict(name="SUBJECT_ID", type="NUMBER")]
    cohort.owner_id = mock_current_user.email
    cohort.coding_system = "Source"
    cohort.criteria = {"include": ["Some inclusion criteria"], "exclude": [], "index_date": None}
    cohort.query_template = "test query template"
    cohort.query = "test query"
    cohort.funnel = mock_funnel
    cohort.user_concepts = [default_medical_concept().model_dump()]
    cohort.explanation = "This is a test cohort explanation."
    cohort.covariates = None
    cohort.access = []
    return cohort


@pytest.fixture
def mock_study_id():
    return uuid.uuid4()


@pytest.fixture
def mock_study(mock_study_id, mock_study_cohort):
    study = MagicMock()
    study.id = mock_study_id
    study.ext_id = "1337"
    study.name = "TEST_STUDY"
    study.description = "Test description"
    study.protocol_link = HttpUrl("https://example.com/")
    study.cohorts = [mock_study_cohort]
    return study


@pytest.fixture
def mock_app_db_with_cohort(mock_app_db, mock_study_cohort, mock_user_cohort):
    mock_app_db.scalars.return_value.one.return_value = mock_study_cohort
    mock_app_db.scalars.return_value.one_or_none.return_value = mock_study_cohort
    mock_app_db.scalars.return_value.all.return_value = [mock_study_cohort, mock_user_cohort]
    return mock_app_db


@pytest.fixture
def mock_app_db_with_study(mock_app_db, mock_study):
    mock_app_db.scalars.return_value.one.return_value = mock_study
    mock_app_db.scalars.return_value.all.return_value = [mock_study]
    return mock_app_db


@patch("ascent_domain.data_queries.cohort.execute_query", new_callable=AsyncMock)
async def test_get_cohort_table_info(mock_execute_query: AsyncMock, mock_db):
    mock_execute_query.side_effect = [([(13, 37)], None), ([("SUBJECT_ID", "NUMBER")], None)]

    table_info = await _get_cohort_table_info(mock_db, "TABLE_NAME")

    assert table_info.size == 13
    assert table_info.size_bytes == 37
    assert table_info.columns == [ColumnInfo(name="SUBJECT_ID", type="NUMBER")]

    # ROW_COUNT/BYTES are Snowflake-only columns on information_schema.tables.
    # The standard view has neither, so the row count is a COUNT(*) and the
    # byte size is reported as 0 rather than invented.
    mock_execute_query.assert_any_call(
        mock_db,
        r"""SELECT COUNT(*) AS ROW_COUNT, 0 AS BYTES
                       FROM ASCENT.ASCENT_COHORTS.TABLE_NAME;""",
    )
    mock_execute_query.assert_any_call(
        mock_db,
        r"""SELECT COLUMN_NAME, DATA_TYPE
                         FROM ASCENT.information_schema.columns
                         WHERE UPPER(TABLE_CATALOG) = 'ASCENT'
                           AND UPPER(TABLE_SCHEMA) = 'ASCENT_COHORTS'
                           AND UPPER(TABLE_NAME) = UPPER('TABLE_NAME')
                         ORDER BY COLUMN_NAME;""",
    )


async def test_get_cohort_descriptor(mock_study_cohort_id, mock_study_cohort, mock_app_db_with_cohort, mock_current_user):
    descriptor = await get_cohort_descriptor(mock_app_db_with_cohort, mock_current_user, mock_study_cohort_id)

    for key, value in descriptor.model_dump().items():
        assert getattr(mock_study_cohort, key) == value

    mock_app_db_with_cohort.scalars.assert_called_once()


async def test_get_cohort_descriptor_not_found(mock_study_cohort_id, mock_app_db_with_cohort, mock_current_user):
    mock_app_db_with_cohort.scalars.return_value.one_or_none.return_value = None

    with pytest.raises(CohortNotFoundError):
        await get_cohort_descriptor(mock_app_db_with_cohort, mock_current_user, mock_study_cohort_id)

    mock_app_db_with_cohort.scalars.assert_called_once()


@pytest.mark.parametrize(
    "restrict_origin,include_pending,clauses",
    [
        (None, False, 2),
        (None, True, 2),
        ("user", False, 3),
        ("user", True, 3),
        ("study", False, 3),
        ("study", True, 3),
    ],
)
async def test_get_cohort_descriptors(
    mock_study_cohort, mock_user_cohort, mock_app_db_with_cohort, mock_current_user, restrict_origin, include_pending, clauses
):
    descriptors = await get_cohort_descriptors(mock_app_db_with_cohort, mock_current_user, restrict_origin, include_pending)

    for ref_descriptor, descriptor in zip((mock_study_cohort, mock_user_cohort), descriptors):
        for key, value in descriptor.model_dump().items():
            assert getattr(ref_descriptor, key) == value

    mock_app_db_with_cohort.scalars.assert_called_once()
    query = mock_app_db_with_cohort.scalars.call_args[0][0]
    assert len(query.whereclause.clauses) == clauses


async def test_update_cohort_descriptor(mock_study_cohort_id, mock_study_cohort, mock_app_db_with_cohort, mock_current_user):
    patch = UserCohortDescriptorPatch(name="New Name", description="New Description", state="orphaned")

    descriptor = await update_cohort_descriptor(mock_app_db_with_cohort, mock_current_user, mock_study_cohort_id, patch)

    descriptor_dump = descriptor.model_dump()
    descriptor_dump["name"] = "New Name"
    descriptor_dump["description"] = "New Description"
    descriptor_dump["state"] = "orphaned"
    for key, value in descriptor_dump.items():
        assert getattr(mock_study_cohort, key) == value

    mock_app_db_with_cohort.scalars.assert_called_once()
    mock_app_db_with_cohort.commit.assert_called_once()


async def test_update_cohort_descriptor_not_found(mock_study_cohort_id, mock_study_cohort, mock_app_db_with_cohort, mock_current_user):
    mock_app_db_with_cohort.scalars.return_value.one_or_none.return_value = None
    patch = UserCohortDescriptorPatch(name="New Name", description="New Description", state="orphaned")

    with pytest.raises(CohortNotFoundError):
        await update_cohort_descriptor(mock_app_db_with_cohort, mock_current_user, mock_study_cohort_id, patch)

    mock_app_db_with_cohort.scalars.assert_called_once()
    mock_app_db_with_cohort.commit.assert_not_called()


async def test_update_cohort_descriptor_funnel(mock_app_db, mock_current_user, mock_user_cohort_id, mock_user_cohort, mock_funnel):
    mock_user_cohort.funnel = []
    funnel = list(map(FunnelCriterion.model_validate, mock_funnel))
    mock_app_db.execute.return_value = funnel
    mock_app_db.scalars.return_value.one_or_none.return_value = mock_user_cohort

    query_delete_funnel_expected = delete(DBFunnelCriterion).where(DBFunnelCriterion.cohort_id == mock_user_cohort_id)

    # act
    updated_descriptor = await update_cohort_descriptor_funnel(mock_app_db, mock_current_user, mock_user_cohort_id, funnel)

    # assert
    (query_delete_funnel,) = mock_app_db.execute.call_args[0]
    assert_query_equal(query_delete_funnel_expected, query_delete_funnel)
    assert len(mock_user_cohort.funnel) == 1
    assert updated_descriptor == UserCohortDescriptor.model_validate(mock_user_cohort)


async def test_update_cohort_descriptor_funnel_not_found(mock_app_db, mock_current_user, mock_user_cohort_id, mock_user_cohort, mock_funnel):
    mock_user_cohort.funnel = []
    funnel = list(map(FunnelCriterion.model_validate, mock_funnel))
    mock_app_db.scalars.return_value.one_or_none.return_value = None

    # act — authorization fails before any funnel mutation
    with pytest.raises(CohortNotFoundError):
        await update_cohort_descriptor_funnel(mock_app_db, mock_current_user, mock_user_cohort_id, funnel)

    # assert
    mock_app_db.execute.assert_not_called()
    assert len(mock_user_cohort.funnel) == 0


async def test_delete_cohort_descriptors(mock_user_cohort, mock_app_db_with_cohort, mock_current_user):
    # owner-scoped: load returns an owned cohort, so deletion is allowed
    mock_app_db_with_cohort.scalars.return_value.one_or_none.return_value = mock_user_cohort

    await delete_cohort_descriptors(mock_app_db_with_cohort, mock_current_user, {mock_user_cohort.id})

    mock_app_db_with_cohort.delete.assert_any_call(mock_user_cohort)
    mock_app_db_with_cohort.commit.assert_called_once()


async def test_delete_cohort_descriptors_not_found(mock_app_db_with_cohort, mock_current_user):
    mock_app_db_with_cohort.scalars.return_value.one_or_none.return_value = None

    with pytest.raises(CohortNotFoundError):
        await delete_cohort_descriptors(mock_app_db_with_cohort, mock_current_user, {uuid.uuid4()})

    mock_app_db_with_cohort.delete.assert_not_called()
    mock_app_db_with_cohort.commit.assert_not_called()


async def test_get_studies(mock_study, mock_app_db_with_study):
    (study,) = await get_studies(mock_app_db_with_study)

    study_dump = study.model_dump()
    study_dump.pop("cohorts")

    for key, value in study_dump.items():
        assert getattr(mock_study, key) == value

    mock_app_db_with_study.scalars.assert_called_once()


@patch("ascent_domain.data_queries.cohort.execute_query", new_callable=AsyncMock)
@patch("ascent_domain.data_queries.cohort._get_cohort_table_info", new_callable=AsyncMock)
async def test_store_cohort(mock_get_cohort_table_info, mock_execute_query, mock_db, mock_app_db, mock_current_user):
    # arrange
    cohort_id = uuid.uuid4()
    cohort_name = f"{OWNED_COHORT_TABLE_PREFIX}{cohort_id.hex.upper()}"
    attributes = [ColumnInfo(name="SUBJECT_ID", type="NUMBER")]
    expected_table_info = TableInfo(name=cohort_name, size=10, size_bytes=42, columns=attributes)
    mock_get_cohort_table_info.return_value = expected_table_info
    cohort = pd.DataFrame(
        columns=["person_id", "index_date", "surprise"],
        data=[[1, "2024-02-01", 1], [2, datetime(2024, 2, 2), 2], [3, date(2024, 2, 3), 3], [4, None, 4], [5, "3004-03-21", 5]],
    )
    cohort_raw = [(1, "2024-02-01"), (2, "2024-02-02"), (3, "2024-02-03"), (4, None), (5, None)]

    # act
    table_info = await store_cohort(cohort_id, mock_db, cohort)

    # assert
    # test RWD DB queries
    mock_execute_query.assert_any_call(mock_db, rf"""CREATE TABLE ASCENT_COHORTS.{table_info.name} (SUBJECT_ID NUMBER, INDEX_DATE DATE);""")
    mock_execute_query.assert_any_call(
        mock_db, rf"""INSERT INTO ASCENT_COHORTS.{table_info.name} (SUBJECT_ID, INDEX_DATE) VALUES (?, ?);""", params=cohort_raw, bulk=True
    )
    # test descriptor
    assert table_info.name.startswith(f"{settings.RELEASE_VERSION.upper()}_USR_")
    assert table_info.size == 10
    assert table_info.columns == attributes


@pytest.mark.parametrize(
    "series, expected",
    [
        (pd.Series([1, 2, 3]), "NUMBER"),
        (pd.Series([1.5, 2.0]), "FLOAT"),
        (pd.Series([True, False]), "BOOLEAN"),
        (pd.Series(["F", "M"]), "VARCHAR"),
        (pd.Series(pd.to_datetime(["2024-01-01", "2024-02-02"])), "DATE"),
        (pd.Series(["7", "8"]), "NUMBER"),
        (pd.Series(["7.2", "8.1"]), "FLOAT"),
        (pd.Series([None, None]), "VARCHAR"),
    ],
)
def test_infer_sf_type(series, expected):
    assert _infer_sf_type(series) == expected


@pytest.mark.parametrize("name", ["age", "Baseline_A1C", "_x1"])
def test_sanitize_identifier_ok(name):
    assert _sanitize_identifier(name) == name.strip().upper()


@pytest.mark.parametrize("name", ["bad name", "weird-col", "1col", "drop;", "a b"])
def test_sanitize_identifier_rejects(name):
    with pytest.raises(ValueError):
        _sanitize_identifier(name)


@patch("ascent_domain.data_queries.cohort.execute_query", new_callable=AsyncMock)
@patch("ascent_domain.data_queries.cohort._get_cohort_table_info", new_callable=AsyncMock)
async def test_store_cohort_with_covariates(mock_get_cohort_table_info, mock_execute_query, mock_db):
    cohort_id = uuid.uuid4()
    cohort_name = f"{OWNED_COHORT_TABLE_PREFIX}{cohort_id.hex.upper()}"
    columns = [ColumnInfo(name="SUBJECT_ID", type="NUMBER"), ColumnInfo(name="AGE", type="NUMBER"), ColumnInfo(name="SEX", type="TEXT")]
    mock_get_cohort_table_info.return_value = TableInfo(name=cohort_name, size=2, size_bytes=42, columns=columns)
    cohort = pd.DataFrame(
        columns=["person_id", "index_date", "age", "sex"],
        data=[[1, "2024-01-01", 64, "F"], [2, "2024-02-02", 58, "M"]],
    )

    table_info = await store_cohort(cohort_id, mock_db, cohort, covariate_columns=["age", "sex"])

    mock_execute_query.assert_any_call(
        mock_db, rf"""CREATE TABLE ASCENT_COHORTS.{table_info.name} (SUBJECT_ID NUMBER, INDEX_DATE DATE, AGE NUMBER, SEX VARCHAR);"""
    )
    mock_execute_query.assert_any_call(
        mock_db,
        rf"""INSERT INTO ASCENT_COHORTS.{table_info.name} (SUBJECT_ID, INDEX_DATE, AGE, SEX) VALUES (?, ?, ?, ?);""",
        params=[(1, "2024-01-01", 64, "F"), (2, "2024-02-02", 58, "M")],
        bulk=True,
    )


@patch("ascent_domain.data_queries.cohort.execute_query", new_callable=AsyncMock)
@patch("ascent_domain.data_queries.cohort._get_cohort_table_info", new_callable=AsyncMock)
async def test_store_cohort_covariate_name_collision(mock_get_cohort_table_info, mock_execute_query, mock_db):
    cohort_id = uuid.uuid4()
    cohort = pd.DataFrame(columns=["person_id", "Subject_Id"], data=[[1, 9]])
    with pytest.raises(ValueError):
        await store_cohort(cohort_id, mock_db, cohort, covariate_columns=["Subject_Id"])


@patch("ascent_domain.data_queries.cohort.execute_query", new_callable=AsyncMock)
@patch("ascent_domain.data_queries.cohort._get_cohort_table_info", new_callable=AsyncMock)
async def test_replace_cohort_columns(mock_get_cohort_table_info, mock_execute_query, mock_db):
    cohort_id = uuid.uuid4()
    cohort_name = f"{OWNED_COHORT_TABLE_PREFIX}{cohort_id.hex.upper()}"
    columns = [ColumnInfo(name="SUBJECT_ID", type="NUMBER"), ColumnInfo(name="A1C", type="FLOAT")]
    mock_get_cohort_table_info.return_value = TableInfo(name=cohort_name, size=2, size_bytes=42, columns=columns)
    merged = pd.DataFrame(columns=["SUBJECT_ID", "A1C"], data=[[1, 7.2], [2, 8.1]])

    table_info = await replace_cohort_columns(cohort_id, mock_db, merged)

    # Built in a staging table and swapped in only once the insert succeeded.
    # Writing in place lost the cohort whenever the insert failed: the table
    # was already replaced, so the cohort went to zero rows while its
    # descriptor still advertised the old size.
    staging = f"{table_info.name}_STAGING"
    mock_execute_query.assert_any_call(
        mock_db, rf"""CREATE OR REPLACE TABLE ASCENT_COHORTS.{staging} (SUBJECT_ID NUMBER, A1C FLOAT);"""
    )
    mock_execute_query.assert_any_call(
        mock_db,
        rf"""INSERT INTO ASCENT_COHORTS.{staging} (SUBJECT_ID, A1C) VALUES (?, ?);""",
        params=[(1, 7.2), (2, 8.1)],
        bulk=True,
    )
    mock_execute_query.assert_any_call(
        mock_db, rf"""ALTER TABLE ASCENT_COHORTS.{staging} RENAME TO {table_info.name};"""
    )

    # The swap must come after the insert, or it is not a swap.
    statements = [call.args[1] for call in mock_execute_query.call_args_list]
    assert statements.index(
        rf"""INSERT INTO ASCENT_COHORTS.{staging} (SUBJECT_ID, A1C) VALUES (?, ?);"""
    ) < statements.index(rf"""ALTER TABLE ASCENT_COHORTS.{staging} RENAME TO {table_info.name};""")


@patch("ascent_domain.data_queries.cohort.execute_query", new_callable=AsyncMock)
@patch("ascent_domain.data_queries.cohort._get_cohort_table_info", new_callable=AsyncMock)
async def test_replace_cohort_columns_keeps_the_cohort_when_the_swap_fails(
    mock_get_cohort_table_info, mock_execute_query, mock_db
):
    """A failed swap must not leave the staging table behind."""
    cohort_id = uuid.uuid4()
    cohort_name = f"{OWNED_COHORT_TABLE_PREFIX}{cohort_id.hex.upper()}"
    staging = f"{cohort_name}_STAGING"
    mock_get_cohort_table_info.return_value = TableInfo(
        name=cohort_name, size=0, size_bytes=0, columns=[ColumnInfo(name="SUBJECT_ID", type="NUMBER")]
    )

    async def fail_on_rename(db, sql, *args, **kwargs):
        if sql.startswith("ALTER TABLE"):
            raise RuntimeError("swap failed")

    mock_execute_query.side_effect = fail_on_rename
    merged = pd.DataFrame(columns=["SUBJECT_ID"], data=[[1]])

    with pytest.raises(RuntimeError, match="swap failed"):
        await replace_cohort_columns(cohort_id, mock_db, merged)

    mock_execute_query.assert_any_call(mock_db, rf"""DROP TABLE IF EXISTS ASCENT_COHORTS.{staging};""")


async def test_store_cohort_descriptor(mock_app_db):
    attributes = [ColumnInfo(name="SUBJECT_ID", type="NUMBER")]
    descriptor = UserCohortDescriptor(
        database="TEST_DB",
        table="TABLE_NAME",
        size=42,
        attributes=attributes,
        owner_id="test@example.com",
        criteria=Criteria(include=["Some inclusion criteria"]),
        query_template="query_template",
        query="query",
    )

    await store_cohort_descriptor(mock_app_db, descriptor)

    mock_app_db.add.assert_called_once()
    mock_app_db.commit.assert_called_once()

    db_descriptor = mock_app_db.add.call_args[0][0]
    for key, value in descriptor.model_dump().items():
        assert getattr(db_descriptor, key) == value


@patch("ascent_domain.data_queries.cohort.datetime")
async def test_delete_abandoned_cohort_descriptors(mock_datetime, mock_app_db):
    # arrange
    mock_datetime.now.return_value = datetime(year=2025, month=1, day=10)

    result = MagicMock()
    result.rowcount = 42
    mock_app_db.execute.return_value = result

    expected_query = delete(DBCohortDescriptor).where(
        DBCohortDescriptor.origin == "user",
        DBCohortDescriptor.state == "pending",
        DBCohortDescriptor.created < datetime(year=2025, month=1, day=9),
    )

    # act
    await delete_abandoned_cohort_descriptors(mock_app_db)

    # assert
    (query,) = mock_app_db.execute.call_args[0]

    assert_query_equal(query, expected_query)
    mock_app_db.commit.assert_called_once()


@patch("ascent_domain.data_queries.cohort.datetime")
@patch("ascent_domain.data_queries.cohort.execute_query", new_callable=AsyncMock)
async def test_flag_orphaned_cohort_descriptors(mock_execute_query, mock_datetime, mock_db, mock_app_db):
    # arrange
    mock_datetime.now.return_value = datetime(year=2025, month=1, day=10)
    user_cohort_tables = [
        "OWNED_USER_COHORT",
        "OWNED_USER_COHORT_ORPHANED",
        "OTHER_USER_COHORT",
    ]

    mock_execute_query.return_value = ([(t,) for t in user_cohort_tables], ...)

    expected_query = (
        update(DBCohortDescriptor)
        .where(
            DBCohortDescriptor.origin == "user",
            DBCohortDescriptor.table.notin_(user_cohort_tables),
            # Fresh descriptors are exempt from orphan-flagging (GC_MIN_COHORT_AGE)
            DBCohortDescriptor.created < datetime(year=2025, month=1, day=9),
        )
        .values(state="orphaned")
    )

    # act
    await flag_orphaned_cohort_descriptors(mock_db, mock_app_db)

    # assert
    (query,) = mock_app_db.execute.call_args[0]

    assert_query_equal(query, expected_query)
    mock_execute_query.assert_called_once_with(mock_db, QUERY_USER_COHORT_TABLES)


def test_gc_owned_tables_query_has_age_guard():
    """The deletion sweep must never see tables younger than GC_MIN_COHORT_AGE —
    a fresh table's descriptor may not be committed yet. The flagging snapshot,
    in contrast, must include ALL existing tables (excluding fresh ones there
    would cause false orphan flags)."""
    assert "AND CREATED < DATEADD('hour', -24, CURRENT_TIMESTAMP())" in QUERY_USER_COHORT_TABLES_OWNED
    assert "DATEADD" not in QUERY_USER_COHORT_TABLES


@pytest.mark.parametrize("state", ["active", "pending"])
def test_ensure_cohort_queryable_allows_live_states(state):
    ensure_cohort_queryable(SimpleNamespace(id=uuid.uuid4(), state=state))


def test_ensure_cohort_queryable_rejects_orphaned():
    with pytest.raises(CohortNotQueryableError) as excinfo:
        ensure_cohort_queryable(SimpleNamespace(id=uuid.uuid4(), state="orphaned"))

    assert status_for(excinfo.value) == 409


# Lower-case variant included deliberately: the OMOP check keys off the
# "_OMOP" suffix, and it must not be case-sensitive. "synthetic_claims"
# used to sit here, but it is the non-OMOP database in this build -- it
# belongs in the rejection test below, not this one.
@pytest.mark.parametrize("database", ["SYNTHETIC_EHR_OMOP", "synthetic_ehr_omop"])
def test_ensure_cohort_demographics_supported_allows_omop(database):
    ensure_cohort_demographics_supported(SimpleNamespace(id=uuid.uuid4(), database=database))


def test_ensure_cohort_demographics_supported_rejects_non_omop():
    """Table One SQL references OMOP-CDM tables (PERSON, CONCEPT, ...), which do
    not exist in non-OMOP databases — reject with a clean 422 instead of letting
    Snowflake raise 'Object PERSON does not exist or not authorized'."""
    with pytest.raises(CohortDemographicsUnsupportedError) as excinfo:
        ensure_cohort_demographics_supported(SimpleNamespace(id=uuid.uuid4(), database="SOME_CLAIMS_DB"))

    assert status_for(excinfo.value) == 422
    assert "SOME_CLAIMS_DB" in str(excinfo.value.detail)


@patch("ascent_domain.data_queries.cohort.execute_query", new_callable=AsyncMock)
async def test_delete_orphaned_cohort_tables(mock_execute_query: AsyncMock, mock_db, mock_app_db, mock_user_cohort):
    # arrange
    user_cohort_tables_owned = [
        "OWNED_USER_COHORT",
        # Only table that may be deleted
        "OWNED_USER_COHORT_ORPHANED",
    ]
    user_cohort_tables_referenced = [
        "OWNED_USER_COHORT",
        # Tables with other prefixes should not be referenced in the first place, but we don't bother either
        "OTHER_USER_COHORT",
    ]

    mock_execute_query.side_effect = [
        ([(t,) for t in user_cohort_tables_owned], ...),
        ...,
    ]
    mock_app_db.execute.return_value = [(t,) for t in user_cohort_tables_referenced]

    # act
    await delete_orphaned_cohort_tables(mock_db, mock_app_db)

    # assert
    mock_execute_query.assert_any_call(mock_db, QUERY_USER_COHORT_TABLES_OWNED)
    mock_execute_query.assert_any_call(mock_db, r"DROP TABLE ASCENT.ASCENT_COHORTS.OWNED_USER_COHORT_ORPHANED;")
    assert len(mock_execute_query.call_args_list) == 2  # make sure no other calls happened


# ─── access control (owner / editor / viewer) ─────────────────────────────────


def _access_desc(owner_id="owner@x.com", origin="user", access=()):
    return SimpleNamespace(owner_id=owner_id, origin=origin, access=list(access))


def _grant(email, role):
    return SimpleNamespace(user_email=email, role=role)


def test_resolve_cohort_role_owner():
    assert resolve_cohort_role(_access_desc(owner_id="a@x.com"), "a@x.com") == "owner"


def test_resolve_cohort_role_from_grant():
    desc = _access_desc(owner_id="a@x.com", access=[_grant("b@x.com", "editor")])
    assert resolve_cohort_role(desc, "b@x.com") == "editor"


def test_resolve_cohort_role_study_and_public_are_viewer():
    assert resolve_cohort_role(_access_desc(owner_id=None, origin="study"), "z@x.com") == "viewer"
    assert resolve_cohort_role(_access_desc(owner_id=None, origin="user"), "z@x.com") == "viewer"


def test_resolve_cohort_role_none_without_access():
    assert resolve_cohort_role(_access_desc(owner_id="a@x.com", origin="user"), "z@x.com") is None


@pytest.mark.parametrize(
    "role,required,ok",
    [
        ("owner", "editor", True),
        ("editor", "editor", True),
        ("viewer", "editor", False),
        ("viewer", "viewer", True),
        ("owner", "owner", True),
        ("editor", "owner", False),
        (None, "viewer", False),
    ],
)
def test_role_at_least(role, required, ok):
    assert role_at_least(role, required) is ok


def test_default_cohort_name_from_criteria():
    assert default_cohort_name("SYNTHETIC_EHR_OMOP", ["Type 2 diabetes"]).startswith("Type 2 diabetes — ")


def test_default_cohort_name_from_database():
    assert default_cohort_name("SYNTHETIC_EHR_OMOP").startswith("Synthetic Ehr cohort — ")


async def test_authorize_cohort_owner_ok(mock_user_cohort, mock_app_db, mock_current_user):
    mock_app_db.scalars.return_value.one_or_none.return_value = mock_user_cohort
    descriptor, role = await authorize_cohort(mock_app_db, mock_current_user, mock_user_cohort.id, "editor")
    assert role == "owner"
    assert descriptor.id == mock_user_cohort.id


async def test_authorize_cohort_viewer_denied_editor(mock_app_db, mock_current_user):
    cohort = MagicMock()
    cohort.owner_id = "someone@else.com"
    cohort.origin = "user"
    cohort.access = [_grant(mock_current_user.email, "viewer")]
    mock_app_db.scalars.return_value.one_or_none.return_value = cohort
    with pytest.raises(CohortAccessDeniedError):
        await authorize_cohort(mock_app_db, mock_current_user, uuid.uuid4(), "editor")


async def test_authorize_cohort_no_access_not_found(mock_app_db, mock_current_user):
    cohort = MagicMock()
    cohort.owner_id = "someone@else.com"
    cohort.origin = "user"
    cohort.access = []
    mock_app_db.scalars.return_value.one_or_none.return_value = cohort
    with pytest.raises(CohortNotFoundError):
        await authorize_cohort(mock_app_db, mock_current_user, uuid.uuid4(), "viewer")


async def test_grant_cohort_access_owner_adds(mock_user_cohort, mock_app_db, mock_current_user):
    mock_app_db.scalars.return_value.one_or_none.side_effect = [mock_user_cohort, None]
    await grant_cohort_access(mock_app_db, mock_current_user, mock_user_cohort.id, "b@x.com", "viewer")
    mock_app_db.add.assert_called_once()
    mock_app_db.commit.assert_called_once()


async def test_grant_cohort_access_denied_for_non_owner(mock_app_db, mock_current_user):
    # a viewer (has access but not owner) cannot share -> 403
    cohort = MagicMock()
    cohort.owner_id = "someone@else.com"
    cohort.origin = "user"
    cohort.access = [_grant(mock_current_user.email, "viewer")]
    mock_app_db.scalars.return_value.one_or_none.return_value = cohort
    with pytest.raises(CohortAccessDeniedError):
        await grant_cohort_access(mock_app_db, mock_current_user, uuid.uuid4(), "b@x.com", "viewer")


async def test_grant_cohort_access_rejects_owner_role(mock_app_db, mock_current_user):
    with pytest.raises(ValueError):
        await grant_cohort_access(mock_app_db, mock_current_user, uuid.uuid4(), "b@x.com", "owner")


async def test_revoke_cohort_access_owner(mock_user_cohort, mock_app_db, mock_current_user):
    mock_app_db.scalars.return_value.one_or_none.return_value = mock_user_cohort
    await revoke_cohort_access(mock_app_db, mock_current_user, mock_user_cohort.id, "b@x.com")
    mock_app_db.execute.assert_called_once()
    mock_app_db.commit.assert_called_once()
