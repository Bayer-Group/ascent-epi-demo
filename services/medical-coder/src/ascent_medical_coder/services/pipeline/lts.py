"""Learn-to-search (LTS) cache.

Records, per query string, whether each candidate concept was accepted by the
LLM filter, so future runs of the same query can skip re-filtering.

Inert as configured here: the warehouse connector opens the local vocabulary
database read-only and that database has no ``ASCENT_LTS`` table, so both the
read and the write path raise. Callers in ``filtering.py`` log the failure and
continue without the cache.
"""

from __future__ import annotations

import asyncio
import logging

from ascent_medical_coder.connectors.warehouse import SnowflakeConnector

logger = logging.getLogger(__name__)

LTS_DATABASE = "ASCENT"
LTS_SCHEMA = "PUBLIC"
LTS_TABLE_FQN = f"{LTS_DATABASE}.{LTS_SCHEMA}.ASCENT_LTS"


async def set_lts(query: str, concept_ids: list[int], accepted: bool) -> None:
    """
    Inserts or updates a list of query-concept_id pairs in the ASCENT_LTS table.

    Args:
        query: The query string.
        concept_ids: A list of concept IDs.
        accepted: A boolean indicating if the concepts were accepted by the LLM.
    """

    concept_ids = [int(i) for i in concept_ids]  # cast all to int

    await asyncio.to_thread(_set_lts_sync, query, concept_ids, accepted)


async def get_lts(query: str, concept_ids: list[int]) -> dict[int, bool]:
    """
    Gets a list of concept_id pairs in the ASCENT_LTS table.

    Args:
        query: The query string.
        concept_ids: A list of concept IDs.

    Returns:
        A dict of the concept id - bool if accepted.
    """

    concept_ids = list(concept_ids)

    if not concept_ids:
        return {}

    return await asyncio.to_thread(_get_lts_sync, query, concept_ids)


async def apply_lts(query: str, concept_ids: list[int]) -> tuple[bool, list[bool]]:
    """
    Applies the LTS  to concept IDs.

    Args:
        query: The query string.
        concept_ids: A list of concept IDs.

    Returns:
        A tuple consisting of:
            A bool if all of the LTS hit.
            A list of bools representing the LTS.
    """

    concept_ids = list(concept_ids)

    lts = await get_lts(query, concept_ids)

    try:
        return True, [lts[concept_id] for concept_id in concept_ids]
    except KeyError:
        return False, []  # lts miss


async def update_lts(query: str, concept_ids_accepted: list[int], concept_ids_all: list[int]) -> None:
    """
    Inserts or updates the ASCENT_LTS table.

    Args:
        query: The query string.
        concept_ids_accepted: A list of concept IDs accepted.
        concept_ids_all: A list of all the concept IDs considered.
    """

    concept_ids_not_accepted = set(concept_ids_all) - set(concept_ids_accepted)

    concept_ids_accepted = list(concept_ids_accepted)
    concept_ids_not_accepted = list(concept_ids_not_accepted)

    await set_lts(query, concept_ids_accepted, True)
    await set_lts(query, concept_ids_not_accepted, False)


def _set_lts_sync(query: str, concept_ids: list[int], accepted: bool) -> None:
    connector = SnowflakeConnector(database=LTS_DATABASE, schema=LTS_SCHEMA)
    conn = connector.conn

    try:
        with conn.cursor() as cur:
            values_to_merge = [(query, str(cid), accepted) for cid in concept_ids]

            if not values_to_merge:
                logging.info("No data to merge.")
                return

            value_placeholders = ", ".join(["(%s, %s, %s)"] * len(values_to_merge))

            merge_sql = f"""
                MERGE INTO {LTS_TABLE_FQN} AS target
                USING (
                    VALUES {value_placeholders}
                ) AS source (QUERY, CONCEPT_ID, ACCEPTED)
                ON target.QUERY = source.QUERY AND target.CONCEPT_ID = source.CONCEPT_ID
                WHEN MATCHED THEN
                    UPDATE SET target.ACCEPTED = source.ACCEPTED
                WHEN NOT MATCHED THEN
                    INSERT (QUERY, CONCEPT_ID, ACCEPTED)
                    VALUES (source.QUERY, source.CONCEPT_ID, source.ACCEPTED);
            """

            flat_values = [item for tpl in values_to_merge for item in tpl]

            cur.execute(merge_sql, flat_values)

            logging.info(f"Successfully merged {cur.rowcount} rows.")
    finally:
        conn.close()


def _get_lts_sync(query: str, concept_ids: list[int]) -> dict[int, bool]:
    connector = SnowflakeConnector(database=LTS_DATABASE, schema=LTS_SCHEMA)
    conn = connector.conn

    result_dict: dict[int, bool] = {}

    try:
        with conn.cursor() as cur:
            placeholders = ", ".join(["%s"] * len(concept_ids))
            sql_query = f"""
                SELECT CONCEPT_ID, ACCEPTED
                FROM {LTS_TABLE_FQN}
                WHERE QUERY = %s
                AND CONCEPT_ID IN ({placeholders})
            """

            params = [query] + [str(cid) for cid in concept_ids]

            cur.execute(sql_query, params)

            for concept_id, accepted in cur.fetchall():
                result_dict[int(concept_id)] = bool(accepted)
    finally:
        conn.close()

    return result_dict
