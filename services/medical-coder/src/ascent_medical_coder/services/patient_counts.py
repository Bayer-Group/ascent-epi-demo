"""Patient-count lookups for medical codes (OMOP + non-OMOP routing).

Ported faithfully from the legacy ``api/api/patient_counts.py``. OMOP codes are
grouped by vocabulary and looked up in the pre-aggregated ``*_standard_counts`` /
``*_source_counts`` tables; non-OMOP databases route to the LLM-driven
:class:`PatientCounts` workflow. SQL, batching, ordering, and response fields are
preserved verbatim.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from fastapi import HTTPException
from starlette import status

from ascent_medical_coder.connectors.warehouse import SnowflakeConnector
from ascent_medical_coder.core.settings import get_settings
from ascent_medical_coder.schemas.patient_counts import (
    CodePatientCount,
    PatientCountRequest,
    PatientCountResponse,
)
from ascent_medical_coder.services.pipeline.counts import (
    get_non_omop_patient_counts_cls,
    get_snowflake_connector,
    get_table_details,
    is_non_omop_database,
)
from ascent_medical_coder.services.pipeline.tables import vocabulary_domain_mapping
from ascent_medical_coder.services.pipeline.utils import parse_db_name_and_schema

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


def _build_count_table_name(database: str, schema: str | None, table_name: str, standard_concept: str | None) -> str:
    suffix = "standard_counts" if standard_concept else "source_counts"
    if schema:
        return f"{database.lower()}_{schema.lower()}_{table_name}_{suffix}"
    return f"{database.lower()}_{table_name}_{suffix}"


def _query_batch(snowflake: SnowflakeConnector, count_table: str, vocabulary_id: str, codes: list[str]) -> list[dict]:
    escaped_codes = ", ".join(f"'{c}'" for c in codes)
    sql = f"""
        SELECT CONCEPT_ID, CONCEPT_CODE, VOCABULARY_ID, PATIENTS_COUNT
        FROM {count_table}
        WHERE VOCABULARY_ID = '{vocabulary_id}'
          AND CONCEPT_CODE IN ({escaped_codes})
    """
    with snowflake.conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


async def _fetch_counts_for_vocabulary(
    db: str,
    schema: str | None,
    vocabulary_id: str,
    codes: list[str],
    standard_concept: str | None,
) -> dict[str, tuple[int | None, int | None]]:
    """Query patient counts for a single vocabulary group. Returns {code: (concept_id, count)}."""
    domain = vocabulary_domain_mapping.get(vocabulary_id)
    if not domain:
        logger.warning("Vocabulary '%s' has no domain mapping — skipping %d codes", vocabulary_id, len(codes))
        return {}

    try:
        table_name = get_table_details(domain)["table_name"]
    except ValueError:
        logger.warning("No table details for domain '%s' (vocabulary '%s') — skipping", domain, vocabulary_id)
        return {}

    count_table = _build_count_table_name(db, schema, table_name, standard_concept)
    snowflake = get_snowflake_connector("ASCENT", "PUBLIC")

    try:
        result_dict: dict[str, tuple[int | None, int | None]] = {}
        for i in range(0, len(codes), BATCH_SIZE):
            batch = codes[i : i + BATCH_SIZE]
            rows = await asyncio.to_thread(_query_batch, snowflake, count_table, vocabulary_id, batch)
            for row in rows:
                result_dict[row["CONCEPT_CODE"]] = (
                    row["CONCEPT_ID"],
                    row["PATIENTS_COUNT"],
                )
        return result_dict
    finally:
        snowflake.conn.close()


async def get_patient_counts(request: PatientCountRequest) -> PatientCountResponse:
    if not request.codes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="codes list must not be empty.",
        )

    db, schema = parse_db_name_and_schema(request.database, request.schema)

    non_omop, non_omop_schema = await is_non_omop_database(db)
    if non_omop:
        return await _handle_non_omop_counts(request, db, schema or non_omop_schema)

    # Group codes by vocabulary so each group hits the right count table
    vocab_groups: dict[str, list[str]] = defaultdict(list)
    for pair in request.codes:
        vocab_groups[pair.vocabulary_id].append(pair.code)

    unsupported = [v for v in vocab_groups if v not in vocabulary_domain_mapping]
    if unsupported:
        supported = ", ".join(sorted(vocabulary_domain_mapping.keys()))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported vocabularies: {unsupported}. Supported: {supported}",
        )

    tasks = {
        vocab: _fetch_counts_for_vocabulary(db, schema, vocab, codes, request.standard_concept)
        for vocab, codes in vocab_groups.items()
    }
    fetched: dict[str, dict[str, tuple[int | None, int | None]]] = {}
    for vocab, coro in tasks.items():
        fetched[vocab] = await coro

    # Build response preserving original pair order
    results: list[CodePatientCount] = []
    for pair in request.codes:
        lookup = fetched.get(pair.vocabulary_id, {})
        match = lookup.get(pair.code)
        if match:
            concept_id, patient_count = match
            results.append(
                CodePatientCount(
                    concept_code=pair.code,
                    concept_id=concept_id,
                    vocabulary_id=pair.vocabulary_id,
                    patient_count=patient_count,
                    found=True,
                )
            )
        else:
            results.append(
                CodePatientCount(
                    concept_code=pair.code,
                    vocabulary_id=pair.vocabulary_id,
                    found=False,
                )
            )

    return PatientCountResponse(
        results=results,
        database=request.database,
        codes_submitted=len(request.codes),
        codes_found=sum(1 for r in results if r.found),
    )


async def _handle_non_omop_counts(
    request: PatientCountRequest,
    db: str,
    schema: str,
) -> PatientCountResponse:
    """Handle patient counts for non-OMOP databases via LLM-generated SQL."""
    service = get_non_omop_patient_counts_cls()(model_name=get_settings().llm.non_omop_model)
    codes = [(pair.code, pair.vocabulary_id) for pair in request.codes]
    response = await service.get_patient_counts(db, schema, codes)

    results: list[CodePatientCount] = [
        CodePatientCount(
            concept_code=r.code,
            concept_id=None,
            vocabulary_id=r.vocabulary_id,
            patient_count=r.patient_count,
            found=r.found,
        )
        for r in response.results
    ]

    return PatientCountResponse(
        results=results,
        database=request.database,
        codes_submitted=response.codes_submitted,
        codes_found=response.codes_found,
        generated_sql=response.generated_sql,
    )
