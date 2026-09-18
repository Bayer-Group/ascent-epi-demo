"""Pydantic schemas for medical concepts returned by LLM (Gemini) responses.

Used as the ``response_schema`` for grounded Gemini search. Byte-identical to
the current service.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MedicalConcept(BaseModel):
    """A single medical concept (condition, procedure, drug, observation, ...)."""

    CONCEPT_NAME: str = Field(..., description="The official medical term name")
    CONCEPT_ID: int = Field(..., description="Unique identifier from the medical vocabulary.", ge=0)
    DOMAIN_ID: str = Field(
        ...,
        description="The domain of the concept (e.g., 'Condition', 'Procedure', 'Drug', 'Observation')",
        min_length=1,
    )
    VOCABULARY_ID: str = Field(
        ...,
        description="The vocabulary system (e.g., 'SNOMED', 'ICD10CM', 'RxNorm', 'LOINC')",
        min_length=1,
    )
    STANDARD_CONCEPT: str | None = Field(
        ...,
        description="Whether it's a standard concept. 'S' for standard, null/None otherwise",
    )
    CONCEPT_CODE: str = Field(..., description="The official code in the vocabulary system", min_length=1)
    score: float = Field(
        ...,
        description="Relevance score between 0.0 and 1.0",
        ge=0.0,
        le=1.0,
        alias="RELEVANCE_SCORE",
    )


class MedicalConceptList(BaseModel):
    """A list of medical concepts."""

    concepts: list[MedicalConcept]
