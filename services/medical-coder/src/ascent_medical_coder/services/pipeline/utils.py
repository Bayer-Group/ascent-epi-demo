"""Generic pipeline helpers (ported from ``utils.py``).

Response-shaping and DataFrame→concept assembly helpers used by the coding
pipeline. OpenSearch helpers, the sparse-vector builder (moved to
``vocabulary.py``), and the encoder cutoff (moved to ``encoder.py``) are not
here. ``convert_hits_to_df`` is not ported: the Qdrant connector owns its own
retrieval-shaping copy. ``_norm_domain_id`` is dropped as dead (no callers).
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException, status

from ascent_medical_coder.schemas.coding import ConceptData, ScoredConcept

INT64_MIN, INT64_MAX = np.iinfo(np.int64).min, np.iinfo(np.int64).max

logger = logging.getLogger(__name__)


def parse_db_name_and_schema(database: str, schema: str | None) -> tuple[str, str | None]:
    """Extract the schema from the database string; checks edge cases."""
    if not database:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Database is required.",
        )

    parsed = database.split(".")
    if len(parsed) == 1:
        return parsed[0], schema

    elif len(parsed) == 2:
        if schema:
            raise HTTPException(
                status_code=status.HTTP_406_NOT_ACCEPTABLE,
                detail="The database argument should not have a dot (.) if the schema arguemnt is set",
            )
        return parsed[0], parsed[1]

    else:
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            detail="The database argument should have zero or one dots (.)",
        )


def _clean_value(val: Any) -> Any:
    """Convert pandas nan to None, keep other values as-is."""
    if pd.isna(val):
        return None
    return val


def _dataframe_to_concepts(df: pd.DataFrame, query: str) -> dict[str, list[dict]]:
    """Convert a DataFrame of hits (Qdrant / LLM fallback) into the expected
    ``dict[str, list[concept_dict]]`` structure used downstream (patient counts,
    response).

    Each concept_dict has keys: CONCEPT_DATA (dict of fields), SCORE (float).
    Empty DataFrame -> ``{query: []}`` (so a domain without results contributes
    nothing).
    """
    if df is None or df.empty:
        return {query: []}

    score_col = None
    if "SCORE" in df.columns:
        score_col = "SCORE"
    elif "score" in df.columns:
        score_col = "score"

    concepts: list[dict] = []
    for _, row in df.iterrows():
        if "CONCEPT_ID" not in row or "CONCEPT_NAME" not in row:
            continue
        try:
            concept_id = int(row["CONCEPT_ID"]) if row["CONCEPT_ID"] not in (None, "") else None
        except Exception:
            continue
        if concept_id is None:
            continue
        score_val = 0.0
        if score_col:
            with contextlib.suppress(Exception):
                score_val = float(row[score_col])

        concept_data = {
            "CONCEPT_ID": concept_id,
            "CONCEPT_CODE": _clean_value(row.get("CONCEPT_CODE")),
            "CONCEPT_NAME": _clean_value(row.get("CONCEPT_NAME")),
            "DOMAIN_ID": row.get("DOMAIN_ID") or row.get("domain_id") or "",
            "VOCABULARY_ID": _clean_value(row.get("VOCABULARY_ID")),
            "STANDARD_CONCEPT": _clean_value(row.get("STANDARD_CONCEPT")),
            # OMOP validity: INVALID_REASON is set ('D'/'U') for invalid concepts.
            "IS_VALID": _clean_value(row.get("INVALID_REASON")) is None,
            "SOURCE": _clean_value(row.get("SOURCE")),
            "MEDCODE": _clean_value(row.get("MEDCODE")),
            "READCODE": _clean_value(row.get("READCODE")),
            "DMDID": _clean_value(row.get("DMDID")),
            "DMDCODE": _clean_value(row.get("DMDCODE")),
            "ORIGINALREADCODE": _clean_value(row.get("ORIGINALREADCODE")),
            "SNOMEDCTCONCEPTID": _clean_value(row.get("SNOMEDCTCONCEPTID")),
            "GEMSCRIPTCODE": _clean_value(row.get("GEMSCRIPTCODE")),
            "PRODCODE": _clean_value(row.get("PRODCODE")),
            # PATIENT_COUNT may be injected later by calculate_patient_counts
        }
        concepts.append({"CONCEPT_DATA": concept_data, "SCORE": score_val})

    return {query: concepts} if concepts else {query: []}


def _convert_concepts_to_models(concepts: dict[str, list[dict]]) -> dict[str, list[ScoredConcept]]:
    """Convert a dict of concept dicts to a dict of ``ScoredConcept`` lists.

    Each concept dict is validated and converted into a ``ScoredConcept`` model.
    Invalid concepts are skipped and logged. A missing ``DOMAIN_ID`` is coerced
    to an empty string (pydantic requires a str). Empty lists are preserved so
    callers receive ``{key: []}`` where applicable.
    """
    converted: dict[str, list[ScoredConcept]] = {}
    for key, concept_list in concepts.items():
        model_list: list[ScoredConcept] = []
        for c in concept_list:
            try:
                data_dict = c.get("CONCEPT_DATA", {})
                # pydantic requires a str DOMAIN_ID.
                if data_dict.get("DOMAIN_ID") is None:
                    data_dict["DOMAIN_ID"] = ""
                concept_data_model = ConceptData(**data_dict)
                model_list.append(ScoredConcept(CONCEPT_DATA=concept_data_model, SCORE=float(c.get("SCORE", 0.0))))
            except Exception as e:
                logger.warning(f"Skipping concept due to validation error: {e}: {c.get('CONCEPT_DATA', None)}")
                continue
        converted[key] = model_list
    return converted
