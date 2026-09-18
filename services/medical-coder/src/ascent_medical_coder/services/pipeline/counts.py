"""Patient count calculation.

The OMOP path looks up pre-aggregated counts from the ``*_standard_counts`` /
``*_source_counts`` tables, selected by ``standard_concept``. The non-OMOP
path is not part of this distribution and raises on use.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import namedtuple

from ascent_medical_coder.connectors.warehouse import SnowflakeConnector
from ascent_medical_coder.services.pipeline.tables import table_details

logger = logging.getLogger(__name__)

Result = namedtuple("Result", ["CONCEPT_ID", "CONCEPT_CODE", "VOCABULARY_ID"])


def get_non_omop_patient_counts_cls():
    """Resolve the non-OMOP ``PatientCounts`` implementation.

    Not part of this distribution: OMOP patient counts and every part of
    coding are unaffected.
    """
    raise NotImplementedError(
        "Non-OMOP patient counts are not part of this distribution. OMOP patient counts are unaffected."
    )


async def is_non_omop_database(database: str):
    """Always raises: non-OMOP patient counts are not part of this distribution."""
    return await get_non_omop_patient_counts_cls().is_non_omop_database(database)


__all__ = [
    "Result",
    "calculate_patient_counts",
    "get_non_omop_patient_counts_cls",
    "get_snowflake_connector",
    "get_table_details",
    "is_non_omop_database",
]


async def calculate_patient_counts(
    concepts: dict[str, list[dict]],
    database: str,
    schema: str | None,
    domain_id: str,
    standard_concept: str,
    batch_size: int = 100,
) -> dict[str, list[dict]]:
    start_time = time.time()

    table_name = get_table_details(domain_id)["table_name"]
    suffix = "standard_counts" if standard_concept else "source_counts"
    if schema:
        temp_table_name = f"{database.lower()}_{schema.lower()}_{table_name}_{suffix}"
    else:
        temp_table_name = f"{database.lower()}_{table_name}_{suffix}"

    snowflake = get_snowflake_connector("ASCENT", "PUBLIC")

    try:
        concept_details = []
        for concept_group in concepts.values():
            for concept in concept_group:
                concept_data = concept["CONCEPT_DATA"]
                concept_details.append(
                    (concept_data["CONCEPT_ID"], concept_data["CONCEPT_CODE"], concept_data["VOCABULARY_ID"])
                )

        def execute_batch_query(details_batch):
            concept_ids = [str(detail[0]) for detail in details_batch]
            concept_codes = [f"'{detail[1]}'" for detail in details_batch]
            vocabulary_ids = [f"'{detail[2]}'" for detail in details_batch]
            tuples_sql = ", ".join(
                f"({id}, {code}, {vocab})"
                for id, code, vocab in zip(concept_ids, concept_codes, vocabulary_ids, strict=False)
            )
            sql_query = f"""
                SELECT CONCEPT_ID, CONCEPT_CODE, VOCABULARY_ID, PATIENTS_COUNT
                FROM {temp_table_name}
                WHERE (CONCEPT_ID, CONCEPT_CODE, VOCABULARY_ID) IN (
                    {tuples_sql}
                )
            """
            with snowflake.conn.cursor() as cur:
                cur.execute(sql_query)
                return cur.fetchall()

        result_dict = {}
        for i in range(0, len(concept_details), batch_size):
            batch = concept_details[i : i + batch_size]
            try:
                results = await asyncio.to_thread(execute_batch_query, batch)
            except Exception as e:
                logger.warning(
                    "Patient count lookup failed for %s; returning concepts without counts: %s",
                    temp_table_name,
                    e,
                )
                # Leave all PATIENT_COUNT as None for this request.
                break
            for row in results:
                key = Result(row["CONCEPT_ID"], row["CONCEPT_CODE"], row["VOCABULARY_ID"])
                result_dict[key] = row["PATIENTS_COUNT"]

        for concept_group in concepts.values():
            for concept in concept_group:
                concept_data = concept["CONCEPT_DATA"]
                key = Result(concept_data["CONCEPT_ID"], concept_data["CONCEPT_CODE"], concept_data["VOCABULARY_ID"])
                concept_data["PATIENT_COUNT"] = result_dict.get(key)

        end_time = time.time()
        print(f"____Execution time calculate_patient_counts____: {end_time - start_time} seconds")

        return concepts
    finally:
        snowflake.conn.close()


def get_table_details(domain: str) -> dict[str, str]:
    for k, v in table_details.items():
        if isinstance(k, tuple):
            if domain in k:
                return v
        elif domain == k:
            return v
    raise ValueError(f"No table details for given domain {domain} found.")


def get_snowflake_connector(database, schema):
    return SnowflakeConnector(database=database, schema=schema)
