"""Web-search-grounded medical-code search with OMOP validation.

Fully LLM-agnostic: the provider is chosen per request via the payload's
``provider`` enum (falling back to ``settings.llm.grounded_search_provider``),
and the transport goes through the grounded-search factory — no vendor is named
here. Supported-vocabulary rows are batch-validated against
``CONCEPT`` and unsupported rows are returned unvalidated.
Validation flow (validated / not_found / not_supported), SQL, and response
fields are preserved from the legacy service.
"""

from __future__ import annotations

import asyncio
import logging

import pandas as pd

from ascent_medical_coder.connectors.warehouse import SnowflakeConnector
from ascent_medical_coder.core.settings import get_settings
from ascent_medical_coder.schemas.grounded_search import (
    GroundedSearchConcept,
    GroundedSearchRequest,
    GroundedSearchResponse,
    ValidationStatus,
)
from ascent_medical_coder.services.pipeline.fallback import LLMSearchFallback
from ascent_medical_coder.services.pipeline.tables import vocabulary_domain_mapping

logger = logging.getLogger(__name__)

_DRUG_DOMAINS = {"drug", "drug_class"}

_SUPPORTED_VOCABULARIES = set(vocabulary_domain_mapping.keys())

# One fallback handler per provider key, to avoid re-wrapping per request.
_fallbacks: dict[str | None, LLMSearchFallback] = {}


def _get_fallback(provider: str | None) -> LLMSearchFallback:
    if provider not in _fallbacks:
        _fallbacks[provider] = LLMSearchFallback(provider=provider)
    return _fallbacks[provider]


async def grounded_search(request: GroundedSearchRequest) -> GroundedSearchResponse:
    """
    Search for medical codes using a web-search-grounded LLM.

    Returns validated results when the code's vocabulary is supported in internal
    OMOP ontologies (CONCEPT). When a vocabulary is not
    supported, the result is returned with a clear note that validation was not possible.
    """
    is_drug = request.domain_id is not None and request.domain_id.lower() in _DRUG_DOMAINS
    fallback = _get_fallback(request.provider.value if request.provider else None)

    if is_drug:
        df = await fallback.search_drugs(
            search_text=request.query,
            vocabulary=request.vocabulary,
            standard_concept=request.standard_concept,
            top_k=request.top_k,
            custom_instructions=request.custom_instructions,
        )
    else:
        df = await fallback.search_medical_concepts(
            search_text=request.query,
            domain_id=request.domain_id,
            vocabulary=request.vocabulary,
            standard_concept=request.standard_concept,
            top_k=request.top_k,
            custom_instructions=request.custom_instructions,
        )

    if df.empty:
        return GroundedSearchResponse(query=request.query, results=[], total_results=0)

    # Offload: _build_results runs a blocking Snowflake validation query.
    results = await asyncio.to_thread(_build_results, df)
    return GroundedSearchResponse(
        query=request.query,
        results=results,
        total_results=len(results),
    )


def _build_results(df: pd.DataFrame) -> list[GroundedSearchConcept]:
    """Split rows into supported/unsupported, validate supported ones, assemble output."""
    supported_rows: list[tuple[int, dict]] = []
    unsupported_results: list[GroundedSearchConcept] = []

    for idx, row in df.iterrows():
        vocab = str(row.get("VOCABULARY_ID") or "")
        if vocab in _SUPPORTED_VOCABULARIES:
            supported_rows.append((idx, row))
        else:
            unsupported_results.append(
                _make_concept(
                    row,
                    ValidationStatus.NOT_SUPPORTED,
                    f"Vocabulary '{vocab}' is not available in internal ontologies; result is unvalidated",
                )
            )

    if not supported_rows:
        return unsupported_results

    pairs = [(str(row.get("CONCEPT_CODE") or ""), str(row.get("VOCABULARY_ID") or "")) for _, row in supported_rows]
    validated_map = _validate_concepts(pairs)

    validated_results: list[GroundedSearchConcept] = []
    for _, row in supported_rows:
        code = str(row.get("CONCEPT_CODE") or "")
        vocab = str(row.get("VOCABULARY_ID") or "")
        hit = validated_map.get((code, vocab))
        if hit:
            concept = _make_concept(
                row,
                ValidationStatus.VALIDATED,
                f"Code validated against internal OMOP ontology (vocabulary: {vocab})",
            )
            concept.validated_concept_id = hit.get("concept_id")
            concept.validated_concept_name = hit.get("concept_name")
            concept.validated_concept_code = hit.get("concept_code")
            concept.is_valid = hit.get("is_valid")
        else:
            concept = _make_concept(
                row,
                ValidationStatus.NOT_FOUND,
                f"Code not found in internal OMOP ontology for vocabulary '{vocab}'",
            )
        validated_results.append(concept)

    return validated_results + unsupported_results


def _validate_concepts(
    pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], dict]:
    """
    Batch lookup (concept_code, vocabulary_id) pairs in CONCEPT.
    Returns a dict keyed by (concept_code, vocabulary_id) with authoritative metadata.
    """
    if not pairs:
        return {}

    db = SnowflakeConnector(database="ASCENT", schema="PUBLIC")

    in_clause = ", ".join(f"('{_escape(code)}', '{_escape(vocab)}')" for code, vocab in pairs)
    sql = f"""
        SELECT concept_id, concept_name, concept_code, vocabulary_id, invalid_reason
        FROM CONCEPT
        WHERE (concept_code, vocabulary_id) IN ({in_clause})
    """

    try:
        cursor = db.conn.cursor()
        try:
            result_df: pd.DataFrame = cursor.execute(sql).df()
        finally:
            cursor.close()
        result_df.columns = [str(c).upper() for c in result_df.columns]
    except Exception:
        logger.exception("concept validation query failed; skipping validation")
        return {}

    mapping: dict[tuple[str, str], dict] = {}
    for _, row in result_df.iterrows():
        key = (str(row["CONCEPT_CODE"]), str(row["VOCABULARY_ID"]))
        mapping[key] = {
            "concept_id": int(row["CONCEPT_ID"]),
            "concept_name": str(row["CONCEPT_NAME"]),
            "concept_code": str(row["CONCEPT_CODE"]),
            "is_valid": bool(pd.isna(row.get("INVALID_REASON"))),
        }
    return mapping


def _escape(value: str) -> str:
    """Escape single quotes for safe SQL string interpolation."""
    return value.replace("'", "''")


def _make_concept(
    row: pd.Series,
    status: ValidationStatus,
    message: str,
) -> GroundedSearchConcept:
    return GroundedSearchConcept(
        concept_name=str(row.get("CONCEPT_NAME") or ""),
        concept_code=str(row.get("CONCEPT_CODE") or ""),
        vocabulary_id=str(row.get("VOCABULARY_ID") or ""),
        domain_id=str(row.get("DOMAIN_ID") or ""),
        standard_concept=row.get("STANDARD_CONCEPT") or None,
        score=float(row.get("SCORE") or 0.0),
        source=str(row.get("SOURCE") or get_settings().llm.grounded_search_provider.title()),
        validation_status=status,
        validation_message=message,
    )
