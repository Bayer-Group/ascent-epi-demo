import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from sqlalchemy import String, cast, func, select
from sqlalchemy.exc import NoResultFound

from ascent_domain.data_queries.data_queries import (
    add_error_message,
    add_medical_concepts,
    add_or_update_executed_queries,
    add_or_update_query,
    add_or_update_query_result,
    get_query_history,
    get_vocabulary_ids,
)
from ascent_domain.models.data_definitions import (
    ApplicationError,
    ExecutedQuery,
    MedicalConcepts,
    PaginatedResponse,
    Query,
    QueryResult,
    UserSearch,
)


@pytest.fixture
def mock_cohort_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def mock_queries(mock_current_user, mock_cohort_id) -> List[Query]:
    return [
        Query(
            id=i,
            session_id="asdf",
            user_email=mock_current_user.email,
            user_input=f"Some user input {i}",
            answer=f"Some gpt answer {i}",
            query_template=f"Some sql template {i}",
            query_filled_preview=f"Some whatever {i}",
            query_filled_search=f"Some edited query {i}",
            database_name=f"Db name {i}",
            created_date=datetime(year=2021, month=1, day=i + 1),
            cohort_id=mock_cohort_id if i % 2 else None,
        )
        for i in range(5)
    ]


@pytest.fixture
def mock_searches(mock_queries) -> List[UserSearch]:
    return [UserSearch(id=q.id, created=q.created_date, query=q.user_input, cohort_id=q.cohort_id) for q in mock_queries]


@pytest.fixture
def mock_searches_page(mock_searches) -> PaginatedResponse[UserSearch]:
    return PaginatedResponse(items=mock_searches, page=13, page_size=37, total=1337)


def assert_query_equal(expected_query, query):
    assert str(expected_query.compile()) == str(query.compile())


@pytest.mark.parametrize("vocab_ids", [(["vocab1", "vocab2", "vocab3"]), (["vocab4", "vocab5"]), ([])])
@patch("ascent_domain.data_queries.data_queries.execute_query", new_callable=AsyncMock)
async def test_get_vocabulary_ids(mock_execute_query, mock_db, vocab_ids):
    mock_execute_query.return_value = ([(id,) for id in vocab_ids], ...)

    result = list(await get_vocabulary_ids(mock_db))

    assert result == vocab_ids
    mock_execute_query.assert_called_once_with(mock_db, r"SELECT DISTINCT c.VOCABULARY_ID FROM concept c")


def _make_async_cm(yield_value):
    @asynccontextmanager
    async def _cm():
        yield yield_value

    return _cm()


@pytest.mark.asyncio
@patch("ascent_domain.data_queries.data_queries.AsyncSessionLocal")
async def test_add_or_update_query__updates_existing(mock_sessionmaker):
    # Arrange: session + async context managers
    session = MagicMock()
    mock_sessionmaker.return_value = _make_async_cm(session)
    session.begin.return_value = _make_async_cm(None)

    data = {"session_id": "123", "user_email": "user_email"}
    existing = Query(session_id="123", user_email="old_value")

    # 1st await: UPDATE result with rowcount > 0
    update_result = MagicMock()
    type(update_result).rowcount = PropertyMock(return_value=1)

    # 2nd await: SELECT result -> Result.scalars() -> ScalarResult.all() == [existing]
    select_result = MagicMock()
    scalars_result = MagicMock()
    scalars_result.all.return_value = [existing]
    # IMPORTANT: scalars is a SYNC method, not async
    select_result.scalars.return_value = scalars_result

    # session.execute is awaited twice; return update_result then select_result
    session.execute = AsyncMock(side_effect=[update_result, select_result])

    # Act
    result = await add_or_update_query(data)

    # Assert
    # Simulate that UPDATE changed the row; since we returned the same instance, update it here
    existing.user_email = "user_email"

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].user_email == "user_email"

    assert session.add.call_count == 0
    assert session.execute.await_count == 2
    session.commit.assert_not_called()  # commit handled by session.begin()
    session.begin.assert_called_once()


@pytest.mark.asyncio
@patch("ascent_domain.data_queries.data_queries.AsyncSessionLocal")
async def test_add_or_update_query_new_query(mock_sessionmaker):
    session = MagicMock()
    mock_sessionmaker.return_value = _make_async_cm(session)
    session.begin.return_value = _make_async_cm(None)

    data = {"session_id": "123", "user_email": "user_email"}

    # 1st await: UPDATE returns rowcount == 0 -> triggers insert
    update_result = MagicMock()
    type(update_result).rowcount = PropertyMock(return_value=0)

    # 2nd await: SELECT should return the newly inserted-like row
    inserted_like = Query(**data)
    select_result = MagicMock()
    scalars_result = MagicMock()
    scalars_result.all.return_value = [inserted_like]  # return list with one row
    select_result.scalars.return_value = scalars_result

    # session.execute awaited twice: UPDATE then SELECT
    session.execute = AsyncMock(side_effect=[update_result, select_result])

    result = await add_or_update_query(data)

    # Assert
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].session_id == "123"
    assert result[0].user_email == "user_email"

    session.add.assert_called_once()  # inserted new row
    assert session.execute.await_count == 2  # update + select
    session.commit.assert_not_called()  # commit handled by session.begin()
    session.begin.assert_called_once()


@patch("ascent_domain.data_queries.data_queries.AsyncSessionLocal")
async def test_add_or_update_query_result_existing(mock_async_session_local):
    mock_session = AsyncMock()
    mock_async_session_local.return_value.__aenter__.return_value = mock_session

    # Create a mock QueryResult instance
    mock_query_result = MagicMock(spec=QueryResult)

    # Create a mock for the result of session.execute()
    mock_execute_result = MagicMock()
    mock_execute_result.scalar_one = MagicMock(return_value=mock_query_result)

    # Set up execute to return the mock_execute_result
    mock_session.execute.return_value = mock_execute_result

    query_id = 1
    data = {"database_name": "test_db", "database_data": {"result": "test_result"}}

    result = await add_or_update_query_result(query_id, data)

    assert isinstance(result, QueryResult)
    mock_session.execute.assert_awaited_once()
    mock_session.commit.assert_awaited_once()

    # Additional assertions to verify that setattr was called correctly
    assert mock_query_result.database_name == "test_db"
    assert mock_query_result.database_data == {"result": "test_result"}


@patch("ascent_domain.data_queries.data_queries.AsyncSessionLocal")
async def test_add_or_update_query_result_new(mock_async_session_local, mock_session):
    mock_async_session_local.return_value.__aenter__.return_value = mock_session

    # Create a mock result that mimics the behavior of an SQLAlchemy result
    mock_result = AsyncMock()
    mock_result.scalar_one = MagicMock(side_effect=NoResultFound())

    # Set up the mock session to return our mock result
    mock_session.execute.return_value = mock_result

    query_id = 1
    data = {"database_name": "test_db", "database_data": {"result": "test_result"}}

    result = await add_or_update_query_result(query_id, data)

    assert isinstance(result, QueryResult)
    mock_session.add.assert_called_once()
    mock_session.commit.assert_awaited_once()

    # Check that the correct QueryResult was created
    called_with = mock_session.add.call_args[0][0]
    assert isinstance(called_with, QueryResult)
    assert called_with.query_id == query_id
    assert called_with.database_name == data["database_name"]
    assert called_with.database_data == data["database_data"]


@patch("ascent_domain.data_queries.data_queries.AsyncSessionLocal")
async def test_add_medical_concepts(mock_async_session_local, mock_session):
    mock_async_session_local.return_value.__aenter__.return_value = mock_session

    query_id = 1
    data = {"query": "test_query", "database": "test_database"}

    result = await add_medical_concepts(query_id, data)

    assert isinstance(result, MedicalConcepts)
    assert result.concept_name == "test_query"
    assert result.database == "test_database"
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()


@patch("ascent_domain.data_queries.data_queries.AsyncSessionLocal")
async def test_add_or_update_executed_queries_existing(mock_async_session_local, mock_session):
    # Create a mock for ExecutedQuery
    executed_query_mock = MagicMock(spec=ExecutedQuery)
    executed_query_mock.database_name = "test_db"
    executed_query_mock.query = "SELECT * FROM table"

    # Setup the return value for scalar_one to return the mocked ExecutedQuery
    mock_session.execute.return_value.scalar_one = MagicMock(return_value=executed_query_mock)

    mock_async_session_local.return_value.__aenter__.return_value = mock_session

    query_id = 1
    data = {"database_name": "test_db", "query": "SELECT * FROM table"}

    result = await add_or_update_executed_queries(query_id, data)

    assert isinstance(result, ExecutedQuery)
    mock_session.execute.assert_called_once()
    mock_session.commit.assert_called_once()


@patch("ascent_domain.data_queries.data_queries.AsyncSessionLocal")
async def test_add_or_update_query_result_new_2(mock_async_session_local, mock_session):
    mock_async_session_local.return_value.__aenter__.return_value = mock_session

    # Create a mock result that mimics the behavior of an SQLAlchemy result
    mock_result = AsyncMock()
    mock_result.scalar_one = MagicMock(side_effect=NoResultFound())

    # Set up the mock session to return our mock result
    mock_session.execute.return_value = mock_result

    query_id = 1
    data = {"database_name": "test_db", "database_data": {"result": "test_result"}}

    result = await add_or_update_query_result(query_id, data)

    # Assert that a new QueryResult was added
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()
    assert isinstance(result, QueryResult)


@patch("ascent_domain.data_queries.data_queries.AsyncSessionLocal")
async def test_add_error_message(mock_async_session_local, mock_session):
    mock_async_session_local.return_value.__aenter__.return_value = mock_session

    message = "Test error message"
    user_id = 1

    result = await add_error_message(message, user_id)

    assert isinstance(result, ApplicationError)
    assert result.message == message
    assert result.user_id == user_id
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()


@patch("ascent_domain.data_queries.data_queries.paginate_query", new_callable=AsyncMock)
async def test_get_query_history(mock_paginate_query, mock_app_db, mock_queries, mock_searches_page):
    # arrange
    mock_paginate_query.return_value = mock_searches_page
    mock_app_db.scalars.return_value.all.return_value = mock_queries

    partition_key = func.coalesce(cast(Query.flow_session_id, String), cast(Query.id, String))
    base = select(
        Query,
        func.row_number()
        .over(
            partition_by=partition_key,
            order_by=Query.created_date.desc(),
        )
        .label("rn"),
    ).where(Query.flow == "OMOP")
    cte = base.cte("latest_queries")
    expected_query = select(Query).join(cte, Query.id == cte.c.id).where(cte.c.rn == 1)

    # act
    history = await get_query_history(mock_app_db, "OMOP", None, None, 13, 37)
    actual_app_db, actual_query = mock_paginate_query.call_args[0]

    # assert
    assert history == mock_searches_page
    assert actual_app_db == mock_app_db
    assert_query_equal(expected_query, actual_query)


@patch("ascent_domain.data_queries.data_queries.paginate_query", new_callable=AsyncMock)
async def test_get_query_history_conditions(mock_paginate_query, mock_app_db, mock_current_user, mock_cohort_id, mock_queries, mock_searches_page):
    # arrange
    mock_paginate_query.return_value = mock_searches_page
    mock_app_db.scalars.return_value.all.return_value = mock_queries

    partition_key = func.coalesce(cast(Query.flow_session_id, String), cast(Query.id, String))
    base = (
        select(
            Query,
            func.row_number()
            .over(
                partition_by=partition_key,
                order_by=Query.created_date.desc(),
            )
            .label("rn"),
        )
        .where(Query.flow == "OMOP")
        .where(Query.user_email == mock_current_user.email)
        .where(Query.cohort_id == mock_cohort_id)
    )
    cte = base.cte("latest_queries")
    expected_base_query = select(Query).join(cte, Query.id == cte.c.id).where(cte.c.rn == 1)

    # act
    history = await get_query_history(mock_app_db, "OMOP", mock_current_user, mock_cohort_id, 13, 37)
    actual_app_db, actual_base_query = mock_paginate_query.call_args[0]

    # assert
    assert history == mock_searches_page
    assert actual_app_db == mock_app_db
    assert_query_equal(expected_base_query, actual_base_query)
