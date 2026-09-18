"""Schemas for bulk concept search. Byte-identical to the current service."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class SearchType(Enum):
    CONCEPT_ID = "concept_id"
    CONCEPT_CODE = "concept_code"
    CONCEPT_NAME = "concept_name"


class BulkConceptSearchRequest(BaseModel):
    query: str
    database: str | None = None
    search_type: SearchType


class Concept(BaseModel):
    CONCEPT_ID: int
    CONCEPT_NAME: str
    CONCEPT_CODE: str
    VOCABULARY_ID: str
    # False when the concept is flagged invalid in OMOP/Athena (INVALID_REASON set).
    IS_VALID: bool = True
    PATIENT_COUNT: int = 0
