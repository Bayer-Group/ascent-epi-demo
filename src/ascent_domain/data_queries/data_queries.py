import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import String, cast, func, select, text, update
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import NoResultFound

from ascent_domain.data_queries.pagination import paginate_query
from ascent_domain.models.data_definitions import (
    FLOW,
    ApplicationError,
    ExecutedQuery,
    MedicalConcepts,
    PaginatedResponse,
    Query,
    QueryResult,
    User,
    UserInfo,
    UserQuestionsResponse,
    UserSearch,
)
from ascent_platform.db.postgresql_session import AsyncSessionLocal
from ascent_platform.warehouse.session import execute_query

logger = logging.getLogger(__name__)


async def get_vocabulary_ids(db):
    """
    This method is used to return list of vocabularies that is used inside concept search window
    :param db:
    :return: Multiple number of string type vocabularies
    :rtype: List
    """
    results, _ = await execute_query(db, r"SELECT DISTINCT c.VOCABULARY_ID FROM concept c")
    return [result[0] for result in results]


async def add_or_update_query(data: dict):
    session_id = data.get("session_id")
    flow_session_id = data.get("flow_session_id")
    database_name = data.get("database_name")
    interpretation = data.get("interpretation")
    interpretation_index = data.get("interpretation_index")

    if not flow_session_id and not session_id:
        raise ValueError("Either 'flow_session_id' or 'session_id' must be provided.")

    # Build match condition
    conds = []

    if flow_session_id:
        conds.append(Query.flow_session_id == flow_session_id)
    else:
        conds.append(Query.session_id == session_id)

    if database_name is not None:
        conds.append(Query.database_name == database_name)

    if interpretation is not None:
        conds.append(Query.interpretation == interpretation)

    if interpretation_index is not None:
        conds.append(Query.interpretation_index == interpretation_index)

    async with AsyncSessionLocal() as session:  # type: AsyncSession
        async with session.begin():
            # 1) Update all matches
            upd = update(Query).where(*conds).values(**data).execution_options(synchronize_session=False)
            result = await session.execute(upd)

            # 2) If nothing matched, insert a new row
            if not result.rowcount:
                obj = Query(**data)
                session.add(obj)

        # 3) Return all rows that match the same condition
        res = await session.execute(select(Query).where(*conds))
        return res.scalars().all()


async def add_or_update_query_result(query_id, data):
    async with AsyncSessionLocal() as session:
        try:
            # Check if a QueryResult with the given query_id and database_name exists
            stmt = select(QueryResult).filter_by(query_id=query_id, database_name=data["database_name"])
            result = await session.execute(stmt)
            query_result_obj = result.scalar_one()
            # Update existing QueryResult with new data
            for key, value in data.items():
                setattr(query_result_obj, key, value)
        except NoResultFound:
            # Create a new QueryResult if one doesn't exist
            query_result_obj = QueryResult(query_id=query_id, **data)
            session.add(query_result_obj)
        except Exception:
            return None  # Do nothing

        await session.commit()  # Commit the transaction
        return query_result_obj  # Return the Query result object


async def add_medical_concepts(query_id, data):
    async with AsyncSessionLocal() as session:
        # Change the key 'query' to 'concept_name'
        data["concept_name"] = data.pop("query")
        # Create a new ConceptCounts if one doesn't exist
        medical_concepts_obj = MedicalConcepts(query_id=query_id, **data)
        session.add(medical_concepts_obj)

        await session.commit()  # Commit the transaction
        return medical_concepts_obj  # Return the medical_concepts_obj


async def add_or_update_executed_queries(query_id, data):
    async with AsyncSessionLocal() as session:
        try:
            # Check if a ExecutedQuery with the given query_id and database_name exists
            stmt = select(ExecutedQuery).filter_by(query_id=query_id, database_name=data["database_name"])
            result = await session.execute(stmt)
            executed_query_obj = result.scalar_one()
            # Update existing ExecutedQuery with new data
            for key, value in data.items():
                setattr(executed_query_obj, key, value)
        except NoResultFound:
            # Create a new QueryResult if one doesn't exist
            executed_query_obj = ExecutedQuery(query_id=query_id, **data)
            session.add(executed_query_obj)

        await session.commit()  # Commit the transaction
        return executed_query_obj  # Return the Executed Query object


async def add_error_message(message, user_id):
    # To save application errors in to the database
    async with AsyncSessionLocal() as session:
        # Create a new ApplicationError
        error_obj = ApplicationError(message=message, user_id=user_id)
        session.add(error_obj)

        await session.commit()  # Commit the transaction
        return error_obj  # Return the error object


async def get_unique_users_last_week(db):
    # Get the current UTC time
    current_utc_time = datetime.now(timezone.utc)
    # Calculate the date one week ago
    one_week_ago = current_utc_time - timedelta(weeks=1)

    # Create a select statement to retrieve unique users from last week until now
    stmt = select(UserInfo.user_id).filter(UserInfo.created_date >= one_week_ago).distinct()

    compiled_stmt = stmt.compile(dialect=postgresql.dialect())
    logger.debug(compiled_stmt)

    # Execute the query and retrieve the result
    result = await db.execute(stmt)

    # Extract user_ids from the result
    unique_user_ids = list(result.scalars().all())

    return unique_user_ids


async def get_user_questions(db, page: int, per_page: int, user: Optional[str]) -> List[UserQuestionsResponse]:
    user_condition = "AND user_id LIKE :user_id" if user is not None else ""

    total_rows_subquery = f"""(SELECT COUNT(*) FROM user_analytics
                               WHERE data ->> 'search_text' IS NOT NULL
                               {user_condition})"""

    query = f"""SELECT id, user_id, created_date, data ->> 'search_text' as search_text,
                {total_rows_subquery} as total_rows
                FROM user_analytics
                WHERE data ->> 'search_text' IS NOT NULL
                {user_condition}
                ORDER BY created_date DESC OFFSET :offset LIMIT :limit"""

    stmt = text(query)

    params = {"offset": (int(per_page) * (int(page) - 1)), "limit": int(per_page)}

    if user is not None:
        params["user_id"] = f"%{user}%"

    result = await db.execute(stmt, params)
    rows = result.fetchall()

    response = [UserQuestionsResponse(id=row[0], user_id=row[1], created_date=str(row[2]), search_text=row[3], total_rows=row[4]) for row in rows]

    return response


def _query_to_search(q: Query) -> UserSearch:
    return UserSearch(id=q.id, created=q.created_date, query=q.user_input, cohort_id=q.cohort_id)


async def get_query_history(
    db: AsyncSessionLocal, flow: FLOW, user: User | None = None, cohort_id: uuid.UUID | None = None, page: int = 0, page_size: int = 10
) -> PaginatedResponse[UserSearch]:
    """
    Return paginated query history, keeping only one row per flow_session_id.
    If flow_session_id is NULL those rows are treated individually.
    """
    partition_key = func.coalesce(cast(Query.flow_session_id, String), cast(Query.id, String))

    base = select(
        Query,
        func.row_number()
        .over(
            partition_by=partition_key,
            order_by=Query.created_date.desc(),  # latest per flow_session_id
        )
        .label("rn"),
    ).where(Query.flow == flow)

    if user:
        base = base.where(Query.user_email == user.email)

    if cohort_id:
        base = base.where(Query.cohort_id == cohort_id)

    cte = base.cte("latest_queries")
    uniq_query = select(Query).join(cte, Query.id == cte.c.id).where(cte.c.rn == 1)

    result = await paginate_query(
        db,
        uniq_query,
        on_response=_query_to_search,
        order_by=Query.created_date.desc(),
        page=page,
        page_size=page_size,
    )
    return result
