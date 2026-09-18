"""Validate medical codes against the ASCENT OMOP vocabulary.

Ported faithfully from the legacy ``api/api/validate_codes.py``. Input pairs are
deduplicated (order preserved), batch-looked-up in
``CONCEPT``, and each input pair is mapped back to a
result. SQL, batching, dedup, ``is_valid`` semantics, and response fields are
preserved verbatim.
"""

from __future__ import annotations

import asyncio
import logging

import pandas as pd
from fastapi import HTTPException
from starlette import status

from ascent_medical_coder.connectors.warehouse import SnowflakeConnector
from ascent_medical_coder.schemas.validate_codes import (
    ValidateCodesRequest,
    ValidateCodesResponse,
    ValidatedCode,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = 500


def _escape(value: str) -> str:
    """Escape single quotes for safe SQL string interpolation."""
    return value.replace("'", "''")


def _query_batch(
    connector: SnowflakeConnector,
    pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], dict]:
    """Look up a batch of (code, vocabulary_id) pairs in CONCEPT."""
    in_clause = ", ".join(f"('{_escape(code)}', '{_escape(vocab)}')" for code, vocab in pairs)
    sql = f"""
        SELECT CONCEPT_ID, CONCEPT_NAME, CONCEPT_CODE, VOCABULARY_ID,
               DOMAIN_ID, STANDARD_CONCEPT, INVALID_REASON
        FROM CONCEPT
        WHERE (CONCEPT_CODE, VOCABULARY_ID) IN ({in_clause})
    """
    try:
        cursor = connector.conn.cursor()
        try:
            df = cursor.execute(sql).df()
        finally:
            cursor.close()
        df.columns = [str(c).upper() for c in df.columns]
    except Exception:
        logger.exception("concept validation query failed")
        return {}

    mapping: dict[tuple[str, str], dict] = {}
    for _, row in df.iterrows():
        key = (str(row["CONCEPT_CODE"]), str(row["VOCABULARY_ID"]))
        mapping[key] = {
            "concept_id": int(row["CONCEPT_ID"]),
            "concept_name": str(row["CONCEPT_NAME"]),
            "domain_id": str(row.get("DOMAIN_ID") or ""),
            "standard_concept": row.get("STANDARD_CONCEPT") or None,
            "is_valid": bool(pd.isna(row.get("INVALID_REASON"))),
        }
    return mapping


async def validate_codes(request: ValidateCodesRequest) -> ValidateCodesResponse:
    """
    Validate a list of medical codes against the ASCENT OMOP vocabulary.

    Each code is checked for existence in CONCEPT.
    Returns the official concept name, concept ID, domain, and standard concept
    status for each valid code, and marks invalid codes accordingly.
    """
    if not request.codes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="codes list must not be empty.",
        )

    logger.info("/validate-codes invoked with %d codes", len(request.codes))

    # Deduplicate input pairs while preserving order
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in request.codes:
        key = (item.code, item.vocabulary_id)
        if key not in seen:
            seen.add(key)
            pairs.append(key)

    # Batch lookup against Snowflake
    connector = SnowflakeConnector(database="ASCENT", schema="PUBLIC")
    try:
        all_validated: dict[tuple[str, str], dict] = {}
        for i in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[i : i + BATCH_SIZE]
            batch_result = await asyncio.to_thread(_query_batch, connector, batch)
            all_validated.update(batch_result)
    finally:
        connector.conn.close()

    # Build response preserving original order
    results: list[ValidatedCode] = []
    for item in request.codes:
        hit = all_validated.get((item.code, item.vocabulary_id))
        if hit:
            results.append(
                ValidatedCode(
                    code=item.code,
                    vocabulary_id=item.vocabulary_id,
                    valid=True,
                    concept_id=hit["concept_id"],
                    concept_name=hit["concept_name"],
                    domain_id=hit["domain_id"],
                    standard_concept=hit["standard_concept"],
                    is_valid=hit["is_valid"],
                )
            )
        else:
            results.append(
                ValidatedCode(
                    code=item.code,
                    vocabulary_id=item.vocabulary_id,
                    valid=False,
                    invalid_reason=f"Code '{item.code}' not found in vocabulary '{item.vocabulary_id}'",
                )
            )

    codes_valid = sum(1 for r in results if r.valid)
    return ValidateCodesResponse(
        results=results,
        codes_submitted=len(request.codes),
        codes_valid=codes_valid,
        codes_invalid=len(request.codes) - codes_valid,
    )
