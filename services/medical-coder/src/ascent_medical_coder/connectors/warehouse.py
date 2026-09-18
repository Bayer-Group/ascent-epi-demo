"""Vocabulary access over a local DuckDB file.

Two things are worth knowing:

* **Queries are transpiled.** The coder's SQL is written in the Snowflake
  dialect, so it is translated per statement; a transpile failure falls back
  to running the original, which DuckDB accepts unchanged in most cases.
* **``database`` and ``schema`` are advisory.** The vocabulary is one local
  file. Callers pass ``("ASCENT","PUBLIC")``, ``("SYNTHETIC_EHR_OMOP", ...)``
  and an LTS database interchangeably, all meaning "the concept tables", so
  the arguments are accepted and the same file is opened.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT = Path(os.environ.get("VOCABULARY_DB", "/data/vocabulary.duckdb"))


def _transpile(query: str) -> str:
    try:
        import sqlglot

        return sqlglot.transpile(query, read="snowflake", write="duckdb")[0]
    except Exception as exc:
        logger.debug("transpile failed, running as written: %s", exc)
        return query


class WarehouseConnector:
    """Reads the concept tables from the local vocabulary database."""

    def __init__(self, database: str = "", schema: str = "", path: Path | None = None) -> None:
        self.database = database
        self.schema = schema
        self._path = Path(path or _DEFAULT)
        self._conn: Any = None

    def connect(self) -> Any:
        import duckdb

        if not self._path.exists():
            raise FileNotFoundError(
                f"No vocabulary database at {self._path}. Build it with "
                f"scripts/build_vocabulary.py, or set VOCABULARY_DB."
            )
        return duckdb.connect(str(self._path), read_only=True)

    def _ensure_conn(self) -> Any:
        if self._conn is None:
            self._conn = self.connect()
        return self._conn

    @property
    def conn(self) -> Any:
        return self._ensure_conn()

    def _fetch_sync(self, query: str, params: Any = None) -> pd.DataFrame:
        cursor = self._ensure_conn().cursor()
        try:
            if params:
                cursor.execute(_transpile(query), params)
            else:
                cursor.execute(_transpile(query))
            frame = cursor.df()
            # The pipeline indexes results by the Snowflake spelling --
            # df["VOCABULARY_ID"] -- and pandas raises KeyError rather than
            # matching case-insensitively, so normalise to upper case.
            frame.columns = [str(c).upper() for c in frame.columns]
            return frame
        finally:
            cursor.close()

    async def fetch_data_with_pandas(self, query: str) -> pd.DataFrame:
        return await asyncio.to_thread(self._fetch_sync, query)

    async def fetch_data_with_cursor(self, sql_query: str, params: Any = None) -> pd.DataFrame:
        return await asyncio.to_thread(self._fetch_sync, sql_query, params)


# The name the pipeline imports.
SnowflakeConnector = WarehouseConnector
