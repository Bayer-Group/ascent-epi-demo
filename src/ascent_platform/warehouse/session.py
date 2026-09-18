"""Query access to the local warehouse.

Replaces the Snowflake session layer. The signatures here are deliberately the
ones callers already use -- ``get_db`` yields a cursor, ``execute_query``
returns ``(rows, lowercased_columns)`` -- because 56 call sites across four
packages depend on that shape. Swapping the engine is a platform concern; it
should not be a domain-wide refactor.

Three things differ from the Snowflake original, all of them simplifications:

* **No connection pool.** DuckDB opens a local file; there is no login round
  trip to amortise. The pool existed because every Snowflake connection cost
  an authentication, which is also why its release-key bug was expensive.
* **No async polling.** Snowflake's ``execute_async`` + ``get_query_status``
  loop existed because queries outlive an HTTP request. DuckDB answers
  in-process, so work is handed to the existing thread pool instead.
* **No OAuth/OBO.** There is one local identity, so the user-scoped variants
  collapse onto the same path.

SQL arrives in Snowflake dialect -- from the prompts, the query library and
the model -- and is transpiled per statement. That happens on the cursor
rather than in ``execute_query`` because callers also execute directly
against the cursor, and a transpiler that only covers one of the two paths is
worse than none.
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Optional

from ascent_platform.warehouse.bootstrap import (
    COHORT_DATABASE,
    COHORT_SCHEMA,
    DATABASES,
    bootstrap,
    cohort_attach_sql,
)
from ascent_platform.warehouse.executor import run_db

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_root: Any = None
_schemas: dict[str, str] = {db.name: db.schema for db in DATABASES}
# Resolvable like any other database, so get_db("ASCENT") works.
_schemas[COHORT_DATABASE] = COHORT_SCHEMA


def _connect(data_dir: Path, work_dir: Path, scale: str):
    """Open one DuckDB root connection with every logical database attached."""
    import duckdb

    built = bootstrap(data_dir, work_dir, scale)
    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": False})
    for name, path in built.items():
        con.execute(f"ATTACH '{path}' AS \"{name}\" (READ_ONLY);")
        logger.info("attached %s", name)

    # Cohorts are the one thing this process writes, and they go to Postgres
    # rather than a DuckDB file so that several workers can write at once.
    # Attached through DuckDB so a cohort can still be joined against the
    # patient tables in a single query. Best-effort: everything except cohort
    # persistence works without it.
    try:
        con.execute("INSTALL postgres; LOAD postgres;")
        con.execute(cohort_attach_sql())
        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{COHORT_DATABASE}"."{COHORT_SCHEMA}";')
        logger.info("attached %s -> postgres (writable)", COHORT_DATABASE)
    except Exception as exc:  # noqa: BLE001 - only cohort persistence needs it
        logger.warning("cohort store unavailable (%s); persistence will fail", exc)

    # Everything this process legitimately touches is attached by now, so drop
    # the ability to reach anything else. DuckDB leaves external access ON by
    # default, which makes the filesystem readable through plain SQL:
    # `SELECT content FROM read_text('.env')` returns the LLM key, and COPY ...
    # TO writes files. Since execute_sql forwards caller SQL essentially
    # unchanged, that turned the tool into arbitrary host file read/write.
    #
    # Ordering is the whole trick: ATTACH and INSTALL both need external
    # access, so the lockdown has to come last. Already-attached databases keep
    # working -- including the Postgres cohort store, whose reads, writes and
    # DDL are unaffected -- and the flag cannot be turned back on for the life
    # of the connection, so no later statement can undo it.
    con.execute("SET enable_external_access=false;")
    logger.info("external access disabled on the warehouse connection")
    return con


def root_connection():
    """The process-wide connection, opened on first use."""
    global _root
    if _root is None:
        with _lock:
            if _root is None:
                from ascent_platform.config.runtime import get_runtime_settings

                settings = get_runtime_settings()
                _root = _connect(
                    Path(settings.WAREHOUSE_DATA_DIR),
                    Path(settings.WAREHOUSE_WORK_DIR),
                    settings.WAREHOUSE_SCALE,
                )
    return _root


def transpile(query: str) -> str:
    """Snowflake SQL in, DuckDB SQL out.

    A failure here is not fatal: most generated SQL is plain enough to run
    unchanged, and refusing to execute would turn a dialect nuance into an
    outage. The original is returned and DuckDB gets to judge it.
    """
    import sqlglot

    try:
        statements = sqlglot.transpile(query, read="snowflake", write="duckdb")
    except Exception as exc:  # noqa: BLE001 - any parse failure falls back
        logger.debug("transpile failed, running as written: %s", exc)
        return query
    return statements[0] if len(statements) == 1 else ";\n".join(statements)


class WarehouseCursor:
    """A DBAPI-shaped cursor that transpiles on the way in.

    Only the surface the callers use: ``execute``, the fetch family,
    ``description`` and ``close``.
    """

    def __init__(self, connection: Any, database: str, schema: str) -> None:
        self._cursor = connection.cursor()
        self.database = database
        self.schema = schema
        self._cursor.execute(f'USE "{database}"."{schema}";')

    def execute(self, query: str, params: Optional[Iterable[Any]] = None):
        self._cursor.execute(transpile(query), list(params) if params else None)
        return self

    def executemany(self, query: str, params: Iterable[Any]):
        self._cursor.executemany(transpile(query), list(params))
        return self

    def fetchall(self):
        return self._cursor.fetchall()

    def fetchone(self):
        return self._cursor.fetchone()

    def fetch_pandas_all(self):
        """A DataFrame with Snowflake-cased columns.

        Snowflake upper-cases unquoted identifiers; DuckDB lower-cases them.
        Callers index results by the Snowflake spelling -- ``df["TABLE_NAME"]``
        -- and pandas has no case-insensitive lookup, so without this they
        silently take an empty branch instead of raising. list_tables returned
        {"tables": [], "count": 0} for a populated schema exactly this way.

        (``execute_query`` lower-cases its column list; that is a separate,
        also-deliberate contract for the tuple path.)
        """
        frame = self._cursor.df()
        frame.columns = [str(c).upper() for c in frame.columns]
        return frame

    @property
    def connection(self):
        """The cursor's own connection, Snowflake-style.

        Callers reach through the cursor for the database name --
        ``db_session.connection.database`` in sql_post_processor. A
        SnowflakeCursor exposes its parent connection; this cursor already
        carries the database and schema it is scoped to, so it stands in for
        its own connection rather than leaking the shared DuckDB handle, which
        knows nothing about which database a caller meant.
        """
        return self

    @property
    def description(self):
        return self._cursor.description

    def close(self) -> None:
        try:
            self._cursor.close()
        except Exception:  # noqa: BLE001 - closing must not raise
            logger.debug("cursor close failed", exc_info=True)


@asynccontextmanager
async def get_db(database_name: str, database_schema: Optional[str] = None) -> AsyncIterator[WarehouseCursor]:
    """Yield a cursor scoped to *database_name* and a schema.

    No implicit default database. The Snowflake original defaulted to a
    specific customer's data, so a caller that forgot the argument silently
    queried it; that was tightened before and stays tightened here.
    """
    if not database_name:
        raise ValueError("database_name is required; no default database fallback allowed")

    schema = database_schema or await get_db_schema(database_name)
    cursor = WarehouseCursor(root_connection(), database_name, schema)
    try:
        yield cursor
    finally:
        cursor.close()


async def get_db_schema(database: str = "") -> str:
    """The schema to use for *database*."""
    if not database:
        raise ValueError("database is required")
    try:
        return _schemas[database.upper()]
    except KeyError:
        raise ValueError(f"Unknown database {database!r}. Available: {', '.join(sorted(_schemas))}") from None


async def get_all_db_schemas(database: str = "") -> list[str]:
    """Every schema for *database*. The local warehouse ships one apiece."""
    return [await get_db_schema(database)]


async def execute_query(
    cursor: WarehouseCursor,
    query: str,
    params: Optional[list[Any]] = None,
    bulk: bool = False,
):
    """Run *query* and return ``(rows, lowercased_column_names)``.

    The column lowercasing is contract: callers index results by lowercase
    name, and Snowflake handed back uppercase identifiers.
    """

    def _run():
        if params and bulk:
            cursor.executemany(query, params)
        else:
            cursor.execute(query, params)
        rows = cursor.fetchall()
        columns = [meta[0].lower() for meta in cursor.description or ()]
        return rows, columns

    try:
        return await run_db(_run)
    except Exception as exc:  # noqa: BLE001 - logged then re-raised, as before
        logger.error("Warehouse error: %s", exc)
        logger.error("Query: %s", query)
        raise


async def get_available_non_omop_databases() -> set[str]:
    """The non-OMOP databases this deployment can see."""
    from ascent_platform.warehouse.db_type import is_omop_database

    return {name for name in _schemas if not is_omop_database(name)}


async def initialize_connection_pool() -> None:
    """Build and attach the warehouse ahead of the first request.

    Kept under the old name so application startup does not have to care that
    there is no longer a pool to fill.
    """
    await run_db(root_connection)
