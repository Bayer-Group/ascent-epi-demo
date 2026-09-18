"""Database and schema listings, as the API and the MCP tools present them.

These three lived in ``ascent/db/snowflake_session.py`` -- a database package
inside the FastAPI app -- alongside twenty-six re-exports of
``ascent_platform.snowflake.session``. Callers reached into the app package to
get platform functions, which is the layering inverted and most of why
ascent_mcp had two dozen imports of ascent_http.

They are domain logic: they query Snowflake and return DatabasePairOption and
MetaDataResponse. The re-exports are gone; callers import the platform directly.
"""

import asyncio
import logging
from pathlib import Path
from typing import List

from ascent_domain.models.database import DatabasePairOption, MetaDataResponse
from ascent_platform.warehouse import catalog
from ascent_platform.warehouse.executor import run_db
from ascent_platform.warehouse.session import get_all_db_schemas

logger = logging.getLogger(__name__)


def _catalog_dir() -> Path:
    from ascent_platform.config.runtime import get_runtime_settings

    return Path(get_runtime_settings().WAREHOUSE_WORK_DIR) / "catalog"


__all__ = ["get_db_pairs", "get_db_pair", "get_omop_metadata"]


async def get_db_pairs(database_names: List[str]) -> List[DatabasePairOption]:
    """
    Get a list of database pairs from the given database names.
    """
    tasks = [get_db_pair(db) for db in sorted(database_names)]

    pairs = await asyncio.gather(*tasks)

    # Filter out databases that have no schemas
    pairs = [pair for pair in pairs if len(pair.database_schemas) > 0]
    return pairs


async def get_db_pair(database_name: str) -> DatabasePairOption:
    schemas = await get_all_db_schemas(database_name)

    return DatabasePairOption(database_name=database_name, database_schemas=schemas)


async def get_omop_metadata(database_name: str, database_schema: str, non_omop: bool = False) -> MetaDataResponse:
    """
    Get avaialable metadata for OMOP databases.
    """

    # The catalog is profiled from the warehouse itself, so the statistics
    # cannot drift from what is stored. The ``non_omop`` flag has no effect.
    try:
        document = await run_db(catalog.load, database_name, database_schema, _catalog_dir())
        return MetaDataResponse.model_validate(document)
    except Exception as e:
        logger.exception(f"Error while reading the catalog: {e}")
        raise
