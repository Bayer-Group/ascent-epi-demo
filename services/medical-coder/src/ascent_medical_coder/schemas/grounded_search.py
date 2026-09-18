"""Schemas for the grounded-search endpoint (web-search-grounded LLM lookup)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class GroundedSearchProvider(str, Enum):
    """Providers implementing web-search-grounded completion.

    Mirrors the factory registry in ``connectors/llm/factory.py`` — add the
    provider there first, then expose it here.
    """

    GEMINI = "gemini"


class GroundedSearchRequest(BaseModel):
    query: str
    domain_id: str | None = None
    vocabulary: str | None = None
    standard_concept: str | None = None
    top_k: int = 10
    custom_instructions: str | None = None
    # Which grounded-search provider to use. Omitted -> the deployment default
    # (settings.llm.grounded_search_provider).
    provider: GroundedSearchProvider | None = Field(
        default=None,
        description="Grounded-search provider; defaults to the deployment-configured provider.",
    )


class ValidationStatus(str, Enum):
    VALIDATED = "validated"
    NOT_FOUND = "not_found"
    NOT_SUPPORTED = "not_supported"


class GroundedSearchConcept(BaseModel):
    concept_name: str
    concept_code: str
    vocabulary_id: str
    domain_id: str
    standard_concept: str | None
    score: float
    source: str
    validation_status: ValidationStatus
    validation_message: str
    validated_concept_id: int | None = None
    validated_concept_name: str | None = None
    validated_concept_code: str | None = None
    # OMOP/Athena validity of the validated concept (INVALID_REASON IS NULL).
    # None when the code could not be validated against internal ontologies.
    is_valid: bool | None = None


class GroundedSearchResponse(BaseModel):
    query: str
    results: list[GroundedSearchConcept]
    total_results: int
