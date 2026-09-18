"""Schemas for the validate-codes endpoint. Byte-identical to the current service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ValidateCodeItem(BaseModel):
    code: str = Field(..., description="CONCEPT_CODE value (e.g. 'E11.9', '44054006')")
    vocabulary_id: str = Field(..., description="Vocabulary, e.g. 'ICD10CM', 'SNOMED', 'RxNorm'")


class ValidateCodesRequest(BaseModel):
    codes: list[ValidateCodeItem] = Field(..., description="List of (code, vocabulary) pairs to validate")


class ValidatedCode(BaseModel):
    code: str
    vocabulary_id: str
    valid: bool = False
    concept_id: int | None = None
    concept_name: str | None = None
    standard_concept: str | None = None
    invalid_reason: str | None = None
    domain_id: str | None = None
    # OMOP/Athena validity of the matched concept (INVALID_REASON IS NULL).
    # None when the code was not found at all ('valid' above means "exists in vocabulary").
    is_valid: bool | None = None


class ValidateCodesResponse(BaseModel):
    results: list[ValidatedCode]
    codes_submitted: int
    codes_valid: int
    codes_invalid: int
