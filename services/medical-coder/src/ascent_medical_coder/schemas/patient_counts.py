"""Schemas for the patient-counts endpoint. Byte-identical to the current service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CodeVocabularyPair(BaseModel):
    code: str = Field(..., description="CONCEPT_CODE value")
    vocabulary_id: str = Field(..., description="Vocabulary, e.g. 'ICD10CM', 'SNOMED', 'RxNorm'")


class PatientCountRequest(BaseModel):
    codes: list[CodeVocabularyPair] = Field(..., description="List of (code, vocabulary) pairs")
    database: str = Field(
        ...,
        description="Database name, optionally with schema, e.g. 'SYNTHETIC_EHR_OMOP.CDM'",
    )
    schema: str | None = Field(default=None, description="Schema name, e.g. 'CDM'")
    standard_concept: str | None = Field(
        default=None,
        description="If set, query standard_counts table; otherwise source_counts",
    )


class CodePatientCount(BaseModel):
    concept_code: str
    concept_id: int | None = None
    vocabulary_id: str
    patient_count: int | None = None
    found: bool = True


class PatientCountResponse(BaseModel):
    results: list[CodePatientCount]
    database: str
    codes_submitted: int
    codes_found: int
    generated_sql: str | None = None
