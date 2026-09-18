"""Bulk concept search over the OMOP ``concept`` table.

Covers the ``execute_concept_search`` path and its helpers: SQL, similarity
scoring, ordering, and the optional patient-count enrichment.
"""

from __future__ import annotations

import logging

import pandas as pd

from ascent_medical_coder.connectors.warehouse import SnowflakeConnector
from ascent_medical_coder.schemas.concept_search import (
    BulkConceptSearchRequest,
    Concept,
    SearchType,
)
from ascent_medical_coder.services.pipeline.counts import get_table_details
from ascent_medical_coder.services.pipeline.tables import vocabulary_domain_mapping

logger = logging.getLogger(__name__)


async def execute_concept_search(request: BulkConceptSearchRequest) -> list[Concept]:
    sql_query = _build_concept_search_query(request.search_type)

    db = get_snowflake_connector()
    result = await db.fetch_data_with_cursor(sql_query, {"query": request.query})
    concepts = _parse_result(result)

    if request.database:
        return await calculate_patient_counts(concepts, request.database)

    return concepts


def _build_concept_search_query(search_type: SearchType) -> str:
    search_condition = {
        SearchType.CONCEPT_ID: "c.concept_id::STRING = st.term",
        SearchType.CONCEPT_CODE: "c.concept_code = st.term",
        SearchType.CONCEPT_NAME: "CONTAINS(UPPER(c.concept_name), UPPER(st.term))",
    }[search_type]

    similarity_score = {
        SearchType.CONCEPT_ID: "100",
        SearchType.CONCEPT_CODE: "90",
        SearchType.CONCEPT_NAME: (
            "CASE WHEN UPPER(c.concept_name) = UPPER(st.term) THEN 80 "
            "ELSE EDITDISTANCE(UPPER(c.concept_name), UPPER(st.term)) END"
        ),
    }[search_type]

    return f"""
        WITH search_terms AS (
            SELECT TRIM(value) AS term
            FROM TABLE(SPLIT_TO_TABLE(%(query)s, ','))
        ),
        concept_search AS (
            SELECT
                c.concept_id,
                c.concept_name,
                c.concept_code,
                c.vocabulary_id,
                c.domain_id,
                c.standard_concept,
                c.invalid_reason,
                {similarity_score} AS similarity_score
            FROM
                concept c
            JOIN
                search_terms st
            ON
                {search_condition}
        )
        SELECT
            cs.concept_id,
            cs.concept_name,
            cs.concept_code,
            cs.vocabulary_id,
            cs.invalid_reason,
            cs.similarity_score
        FROM
            concept_search cs
        ORDER BY
            cs.similarity_score ASC,
            cs.concept_name
    """


def _parse_result(result: pd.DataFrame) -> list[Concept]:
    concepts: list[Concept] = []
    for _, row in result.iterrows():
        data = row.to_dict()
        data["IS_VALID"] = bool(pd.isna(data.get("INVALID_REASON")))
        concepts.append(Concept(**data))
    return concepts


async def calculate_patient_counts(concepts: list[Concept], database: str) -> list[Concept]:
    # Group concepts by their vocabulary
    concept_groups: dict[str, list[Concept]] = {}
    for concept in concepts:
        concept_groups.setdefault(concept.VOCABULARY_ID, []).append(concept)

    # Process each group separately
    for vocabulary_id, group_concepts in concept_groups.items():
        # Determine the domain and table name based on the vocabulary
        _domain, table_info = get_domain_and_table(vocabulary_id)
        if not table_info:
            continue  # Skip if we can't determine the table

        table_name = table_info["table_name"]
        suffix = "standard_counts"
        temp_table_name = f"{database.lower()}_{table_name}_{suffix}"

        concept_tuples = [(str(c.CONCEPT_ID), f"'{c.CONCEPT_CODE}'", f"'{c.VOCABULARY_ID}'") for c in group_concepts]

        sql_query = f"""
            SELECT CONCEPT_ID, CONCEPT_CODE, VOCABULARY_ID, PATIENTS_COUNT
            FROM {temp_table_name}
            WHERE (CONCEPT_ID, CONCEPT_CODE, VOCABULARY_ID) IN (
                {", ".join(f"({cid}, {code}, {vocab})" for cid, code, vocab in concept_tuples)}
            )
        """

        db = get_snowflake_connector("ASCENT", "PUBLIC")
        result_df: pd.DataFrame = await db.fetch_data_with_cursor(sql_query)

        result_dict = {
            (row["CONCEPT_ID"], row["CONCEPT_CODE"], row["VOCABULARY_ID"]): row["PATIENTS_COUNT"]
            for _, row in result_df.iterrows()
        }

        for concept in group_concepts:
            key = (concept.CONCEPT_ID, concept.CONCEPT_CODE, concept.VOCABULARY_ID)
            concept.PATIENT_COUNT = result_dict.get(key, 0)

    return concepts


def get_domain_and_table(vocabulary_id: str) -> tuple[str | None, dict[str, str] | None]:
    domain = vocabulary_domain_mapping.get(vocabulary_id)
    if domain:
        return domain, get_table_details(domain)
    return None, None


def get_snowflake_connector(database: str = "SYNTHETIC_EHR_OMOP", schema: str = "CDM") -> SnowflakeConnector:
    return SnowflakeConnector(database=database, schema=schema)
