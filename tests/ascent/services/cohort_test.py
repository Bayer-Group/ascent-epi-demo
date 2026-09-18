import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pandas as pd
import pytest

from ascent_domain.cohort import (
    CannotShareStudyCohortsException,
    create_cohort,
    create_cohort_funnel,
    prepare_criteria,
    preview_cohort_query,
    share_cohort,
)
from ascent_domain.models.data_definitions import (
    ColumnInfo,
    Criteria,
    MedicalConcept,
    StudyCohortDescriptor,
    TableInfo,
    User,
    UserCohortDescriptor,
)
from ascent_domain.omop.models.inference.rag import RAGProcessor
from ascent_http.services.ai_services import CriteriaService


@pytest.fixture
def mock_criteria_rag() -> RAGProcessor:
    return AsyncMock()


@pytest.fixture
def mock_criteria_service(mock_criteria_rag) -> CriteriaService:
    service = AsyncMock()
    service.rag = mock_criteria_rag
    return service


@pytest.fixture
def mock_patients():
    return pd.DataFrame(columns=["person_id", "index_date"], data=[[i, datetime(year=2000, month=1, day=i + 1)] for i in range(10)])


@pytest.fixture
def mock_user_cohort_id():
    return uuid.uuid4()


@pytest.fixture
def mock_user_cohort_table_info():
    return TableInfo(name="USER_COHORT", size=10, size_bytes=42, columns=[ColumnInfo(name="SUBJECT_ID", type="NUMBER")])


@pytest.fixture
def mock_user_cohort(mock_user_cohort_id, mock_user_cohort_table_info):
    return UserCohortDescriptor(
        id=mock_user_cohort_id,
        name="User Cohort",
        database="TEST_DB",
        table=mock_user_cohort_table_info.name,
        size=mock_user_cohort_table_info.size,
        attributes=mock_user_cohort_table_info.columns,
        owner_id="test@example.com",
        criteria=Criteria(include=["Some inclusion criteria"]),
        query_template="query_template",
        query="query",
        explanation="This is a test cohort explanation.",
    )


@pytest.mark.parametrize(
    "service_response,exclude_expected,index_date_expected",
    [
        ({"include": ["incl"]}, [], None),
        ({"include": ["incl"], "exclude": ["excl"]}, ["excl"], None),
        ({"include": ["incl"], "index_date": "idx"}, [], "idx"),
        ({"include": ["incl"], "index_date": ["idx"]}, [], "idx"),
    ],
)
async def test_prepare_criteria(mock_criteria_service, service_response, exclude_expected, index_date_expected):
    # arrange
    user_input = "Some user query"
    mock_criteria_service.text_to_criteria.return_value = service_response

    # act
    criteria = await prepare_criteria(user_input=user_input, service=mock_criteria_service)

    # assert
    mock_criteria_service.text_to_criteria.assert_called_with(user_input)
    assert criteria.include == ["incl"]
    assert criteria.exclude == exclude_expected
    assert criteria.index_date == index_date_expected


async def test_preview_cohort_query(mock_criteria_rag, mock_criteria_service):
    # arrange
    query_template = "Some query template"
    query = "Some query"
    concepts_raw = [{"name": "test", "category": "condition", "value": []}]

    mock_criteria_service.criteria_to_data.return_value = ..., query_template, query, ...
    mock_criteria_service.find_medical_concepts = MagicMock(return_value=concepts_raw)

    criteria = Criteria()

    # act
    preview = await preview_cohort_query(
        database="SOME_DB", database_schema="SOME_SCHEMA", coding_system="Source", criteria=criteria, service=mock_criteria_service
    )

    # assert
    mock_criteria_service.criteria_to_data.assert_called_with(
        criteria_dict=criteria.model_dump(),
        preferred_coding_system="Source",
        snowflake_database="SOME_DB",
        snowflake_database_schema="SOME_SCHEMA",
        main_path=None,
        log_folder=None,
        querylib_file=None,
        verify=False,
        run_query=False,
        custom_code_mappings={},
    )
    mock_criteria_service.find_medical_concepts.assert_called_with(query_template=query_template, database="SOME_DB")
    assert preview == {
        "query_template": query_template,
        "query": query,
        "concepts": list(map(MedicalConcept.from_medical_coder, concepts_raw)),
    }


@patch("ascent_domain.cohort.uuid")
@patch("ascent_domain.cohort.get_db")
@patch("ascent_domain.cohort.get_db_as_user")
@patch("ascent_domain.cohort.fetch_cohort_dataframe", new_callable=AsyncMock)
@patch("ascent_domain.cohort.store_cohort", new_callable=AsyncMock)
@patch("ascent_domain.cohort.store_cohort_descriptor", new_callable=AsyncMock)
async def test_create_cohort(
    mock_store_cohort_descriptor,
    mock_store_cohort,
    mock_fetch_cohort_dataframe,
    mock_get_db_as_user,
    mock_get_db,
    mock_uuid,
    mock_app_db,
    mock_user,
    mock_criteria_rag,
    mock_criteria_service,
):
    # arrange
    criteria = Criteria()
    criteria_dict = criteria.model_dump()
    query_template = "Some query template"
    query = "Some query"
    cohort_id = uuid4()
    cohort = pd.DataFrame(data=None, columns=["person_id", "index_date"])
    table_info = TableInfo(name="SOME_TABLE_NAME", columns=[ColumnInfo(name="person_id", type="NUMBER")], size=42, size_bytes=1337)
    custom_code_mappings = None

    mock_rwd_cursor = MagicMock(name="rwd_cursor")
    mock_ascent_cursor = MagicMock(name="ascent_cursor")
    mock_get_db_as_user.return_value.__aenter__.return_value = mock_rwd_cursor
    mock_get_db_as_user.return_value.__aexit__.return_value = None
    mock_get_db.return_value.__aenter__.return_value = mock_ascent_cursor
    mock_get_db.return_value.__aexit__.return_value = None

    mock_criteria_service.criteria_to_data.return_value = ..., query_template, query, cohort
    mock_fetch_cohort_dataframe.return_value = cohort
    mock_store_cohort.return_value = table_info
    mock_uuid.uuid4 = lambda: cohort_id
    mock_criteria_service.get_query_explanation.return_value = "This is a test explanation."

    token_fetcher = AsyncMock(return_value="sf-token")

    # act
    descriptor = await create_cohort(
        database="SOME_DB",
        database_schema="SOME_SCHEMA",
        coding_system="Source",
        criteria=criteria,
        user_concepts=None,
        token_fetcher=token_fetcher,
        app_db=mock_app_db,
        user=mock_user,
        service=mock_criteria_service,
    )

    # assert
    mock_criteria_service.criteria_to_data.assert_called_with(
        criteria_dict=criteria_dict,
        preferred_coding_system="Source",
        snowflake_database="SOME_DB",
        snowflake_database_schema="SOME_SCHEMA",
        main_path=None,
        log_folder=None,
        querylib_file=None,
        custom_code_mappings=custom_code_mappings,
        verify=False,
        run_query=False,
    )
    # SELECT runs through SSO RWD cursor; write runs through machine-user ASCENT cursor.
    mock_get_db_as_user.assert_called_once_with("SOME_DB", "sf-token", "SOME_SCHEMA")
    mock_get_db.assert_called_once_with("ASCENT")
    mock_fetch_cohort_dataframe.assert_called_once_with(mock_rwd_cursor, query)
    mock_store_cohort.assert_called_once_with(cohort_id=cohort_id, ascent_db=mock_ascent_cursor, cohort=cohort)
    mock_store_cohort_descriptor.assert_called_once_with(app_db=mock_app_db, descriptor=descriptor)


@patch("ascent_domain.cohort.update_cohort_descriptor_funnel")
async def test_create_cohort_funnel(
    mock_update_cohort_descriptor_funnel, mock_app_db, mock_user, mock_criteria_service, mock_user_cohort, mock_patients
):
    cohorts = [mock_patients] * 3
    types = ["include", "include", "exclude"]
    descriptions = ["Criterion 1", "Criterion 2", "Criterion 3"]
    sizes = [1, 2, 3]

    mock_criteria_service.split_query_by_criteria.return_value = (
        mock_user_cohort.query_template,
        mock_user_cohort.query,
        cohorts,
        types,
        descriptions,
    )
    mock_criteria_service.get_patient_funnel_from_dfs = MagicMock(return_value=(sizes, None))
    mock_update_cohort_descriptor_funnel.return_value = None

    # act
    funnel = await create_cohort_funnel(descriptor=mock_user_cohort, app_db=mock_app_db, user=mock_user, service=mock_criteria_service)

    # assert
    mock_criteria_service.split_query_by_criteria.assert_called_once_with(
        preferred_coding_system=mock_user_cohort.coding_system,
        snowflake_database=mock_user_cohort.database,
        snowflake_database_schema=mock_user_cohort.database_schema,
        query_template_pred=mock_user_cohort.query_template,
        criteria_dict=mock_user_cohort.criteria.model_dump(),
        custom_code_mappings=None,
    )
    mock_update_cohort_descriptor_funnel.assert_called_once_with(mock_app_db, mock_user, mock_user_cohort.id, funnel)


@patch("ascent_domain.cohort.grant_cohort_access", new_callable=AsyncMock)
@patch("ascent_domain.cohort.authorize_cohort", new_callable=AsyncMock)
async def test_share_cohort(mock_authorize_cohort, mock_grant_cohort_access):
    """Sharing grants the recipient access without changing the cohort's owner."""
    cohort_id = uuid4()
    recipient_email = "other.user@example.com"
    app_db = AsyncMock()
    cohort_descriptor = UserCohortDescriptor(
        id=cohort_id,
        database="dummy",
        database_schema="dummy",
        table="dummy",
        size=5,
        attributes=[ColumnInfo(name="dumy_name", type="dummy_type")],
        owner_id="cohort.owner@example.com",
        criteria=Criteria(include=["incl"]),
        query_template="query_template",
        query="query",
    )
    current_user = User()
    mock_authorize_cohort.return_value = (cohort_descriptor, "owner")

    # act
    await share_cohort(cohort_id, recipient_email, app_db, current_user, role="editor")

    # assert: owner authorized, recipient granted access; no copy/ownership change
    mock_authorize_cohort.assert_called_once_with(app_db, current_user, cohort_id, "owner")
    mock_grant_cohort_access.assert_called_once_with(app_db, current_user, cohort_id, recipient_email, "editor")
    app_db.add.assert_not_called()


@patch("ascent_domain.cohort.grant_cohort_access", new_callable=AsyncMock)
@patch("ascent_domain.cohort.authorize_cohort", new_callable=AsyncMock)
async def test_cannot_share_study_cohorts(mock_authorize_cohort, mock_grant_cohort_access):
    """Sharing cohorts belonging to studies should raise an exception"""
    # arrange
    cohort_id = uuid4()
    app_db = AsyncMock()
    current_user = User()
    cohort_descriptor = StudyCohortDescriptor(
        id=cohort_id,
        origin="study",
        database="dummy",
        database_schema="dummy",
        table="dummy",
        size=5,
        attributes=[ColumnInfo(name="dumy_name", type="dummy_type")],
        study_id=uuid4(),
    )
    mock_authorize_cohort.return_value = (cohort_descriptor, "owner")

    # act
    with pytest.raises(CannotShareStudyCohortsException):
        await share_cohort(cohort_id, "other.user@example.com", app_db, current_user)

    mock_grant_cohort_access.assert_not_called()
