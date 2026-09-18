"""The user-scoped query API, with a single local identity.

On Snowflake these functions existed to carry the caller's SSO session: each
one took a ``token_fetcher``, exchanged it for a Snowflake-scoped token and
opened a per-user connection. There is one identity here, so that half is
gone and ``token_fetcher`` is accepted-and-ignored, which keeps ~25 call
sites unchanged.

What is NOT merely auth, and is preserved: ``user_sql_session`` and
``user_cursor_session`` hold **one** connection open for the whole block so
that session state survives between statements. TEMPORARY tables are the
reason -- the non-OMOP pipeline materialises one and then queries it, and
DuckDB scopes temp tables to a connection exactly as Snowflake does. Handing
out a fresh cursor per statement would make the temp table vanish between the
CREATE and the SELECT, which fails as a confusing "table does not exist"
rather than anything that names the cause.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import pandas as pd

from ascent_platform.warehouse.executor import run_db
from ascent_platform.warehouse.session import (
    WarehouseCursor,
    get_db,
    get_db_schema,
    root_connection,
)
from ascent_platform.warehouse.sql_executor import SqlExecutor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def get_db_as_user(
    database_name: str,
    oauth_token: Optional[str] = None,
    database_schema: Optional[str] = None,
) -> AsyncIterator[WarehouseCursor]:
    """``get_db`` under the caller's identity. There is only one here."""
    async with get_db(database_name, database_schema) as cursor:
        yield cursor


def user_sql_executor(
    database: str,
    schema: Optional[str],
    token_fetcher: Any = None,
    role: Optional[str] = None,
) -> SqlExecutor:
    """An executor for ``database.schema``. Roles are a Snowflake concept."""
    return SqlExecutor(database, schema or "")


class _HeldCursor:
    """One cursor, shared for the duration of a session block.

    A lock is kept even though there is no network round trip: callers fan out
    with ``asyncio.gather``, and a single DuckDB cursor cannot run concurrent
    statements any more than a Snowflake one could.
    """

    def __init__(self, cursor: WarehouseCursor) -> None:
        self.cursor = cursor
        self.lock = asyncio.Lock()

    @asynccontextmanager
    async def __call__(self, *_args: Any, **_kwargs: Any) -> AsyncIterator[WarehouseCursor]:
        async with self.lock:
            yield self.cursor


@asynccontextmanager
async def user_sql_session(
    database: str,
    schema: Optional[str],
    token_fetcher: Any = None,
    role: Optional[str] = None,
) -> AsyncIterator[tuple[SqlExecutor, Any]]:
    """Yield ``(executor, connection)`` sharing one session.

    Every statement the executor runs -- including SQL self-healing retries --
    lands on the same connection, so a temp table created early is still there
    later.
    """
    resolved = schema or await get_db_schema(database)
    cursor = WarehouseCursor(root_connection(), database, resolved)
    try:
        yield SqlExecutor(database, resolved, cursor=cursor), cursor
    finally:
        cursor.close()


@asynccontextmanager
async def user_cursor_session(
    database: str,
    schema: Optional[str],
    token_fetcher: Any = None,
) -> AsyncIterator[tuple[Any, Any]]:
    """Yield ``(cursor_provider, connection)`` pinned to one session."""
    resolved = schema or await get_db_schema(database)
    cursor = WarehouseCursor(root_connection(), database, resolved)
    held = _HeldCursor(cursor)
    try:
        yield held, cursor
    finally:
        cursor.close()


def make_user_cursor_provider(token_fetcher: Any = None):
    """A provider that opens a cursor per request, as the standard routing did."""

    @asynccontextmanager
    async def provider(database_name: str, database_schema: Optional[str] = None) -> AsyncIterator[WarehouseCursor]:
        async with get_db(database_name, database_schema) as cursor:
            yield cursor

    return provider


async def execute_user_sql_to_df(
    database: str,
    schema: Optional[str],
    token_fetcher: Any,
    query: str,
    role: Optional[str] = None,
) -> pd.DataFrame:
    """Run one statement and return a DataFrame, DDL/DML included.

    ``SqlExecutor`` always fetches as a DataFrame, which on Snowflake meant it
    could not handle statements returning a status row rather than an Arrow
    result. The same split exists here, so writes keep this separate path.
    """
    resolved = schema or await get_db_schema(database)

    def _run() -> pd.DataFrame:
        cursor = WarehouseCursor(root_connection(), database, resolved)
        try:
            cursor.execute(query)
            try:
                return cursor.fetch_pandas_all()
            except Exception:  # noqa: BLE001 - DDL/DML has no tabular result
                rows = cursor.fetchall()
                columns = [meta[0] for meta in cursor.description or ()]
                return pd.DataFrame(rows, columns=columns)
        finally:
            cursor.close()

    return await run_db(_run)
