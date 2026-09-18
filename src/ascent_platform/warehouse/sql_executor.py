"""Execute a statement against one database/schema and return a DataFrame.

The non-OMOP pipeline's main entry to the warehouse. Two behaviours are
contract rather than accident and are preserved deliberately:

* **Failure returns ``{"error": str(e)}`` instead of raising.** Callers branch
  on that dict, and the SQL-repair loop feeds the message back to the model.
  Raising here would look tidier and would break the repair path.
* **Success returns a pandas DataFrame**, because downstream code calls
  ``.empty``, ``.to_markdown()`` and friends on it.

There is one local identity, so there is one connection path.
``connection_provider`` is accepted so call sites that pass it keep working,
and is ignored.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from ascent_platform.warehouse.executor import run_db
from ascent_platform.warehouse.session import WarehouseCursor, root_connection

logger = logging.getLogger(__name__)


class SqlExecutor:
    """Runs SQL against a single database and schema."""

    def __init__(
        self,
        database: str,
        schema: str,
        connection_provider: Optional[Any] = None,
        cursor: Optional[WarehouseCursor] = None,
    ) -> None:
        """
        Args:
            database: The warehouse database to query.
            schema: The schema within it.
            connection_provider: Accepted for call-site compatibility and
                ignored -- the local warehouse has a single identity.
            cursor: Run every statement on this cursor instead of opening one
                per query. Session-scoped state -- TEMPORARY tables above all --
                only survives between statements when they share a cursor.
        """
        self.database = database
        self.schema = schema
        self._held = cursor
        if connection_provider is not None:
            logger.debug("connection_provider ignored: the local warehouse has one identity")
        logger.info("Initialized SQL executor for database: %s, schema: %s", database, schema)

    async def execute_sql_query(self, sql_query: str) -> Dict[str, Any] | pd.DataFrame:
        """Execute *sql_query*; return a DataFrame, or an error dict on failure."""
        sql_start_time = time.time()
        try:
            sql_result: Dict[str, Any] | pd.DataFrame = await run_db(self._run, sql_query)
            logger.info("SQL query executed successfully")
        except Exception as e:  # noqa: BLE001 - reported to the caller, by contract
            logger.error("SQL execution error: %s", e)
            sql_result = {"error": str(e)}

        sql_duration = round(time.time() - sql_start_time, 1)
        sql_execution_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            "Query execution completed in %s seconds at %s",
            sql_duration,
            sql_execution_datetime,
        )
        return sql_result

    def _run(self, sql_query: str) -> pd.DataFrame:
        if self._held is not None:
            self._held.execute(sql_query)
            return self._held.fetch_pandas_all()
        cursor = WarehouseCursor(root_connection(), self.database, self.schema)
        try:
            cursor.execute(sql_query)
            return cursor.fetch_pandas_all()
        finally:
            cursor.close()
