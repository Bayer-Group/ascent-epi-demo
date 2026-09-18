"""Request/response schemas for the medical-coding endpoints.

Field names, types, and defaults are byte-identical to the current service —
these models are the public API contract consumed by callers of this service.
Only import style is modernized.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MedicalCodingRequest(BaseModel):
    query: str
    top_k: int = 10000
    llm_filter: Literal["default", "gemini", "chatgpt", "haiku"] | None = None
    encoder: Literal["bge", "sap", "gemini", "biolord"] = "bge"
    domain_ids: list[str] = Field(default_factory=list)
    vocabulary: list[str] | None = None
    standard_concept: str | None = None
    cosine_similarity: float | None = None
    database: str | None = None
    source: str | None = None
    allow_lts: bool = False
    # See the MCP tool docstrings: the fallback returns real-world codes
    # that do not exist in a synthetic vocabulary. Callers opt in.
    google_search_fallback: bool = False
    custom_instructions: str | None = None
    use_lts: bool = False
    use_hybrid: bool = False
    # False returns only raw matches, skipping descendant enrichment (ICD prefix, OMOP ancestor, NDC siblings).
    include_descendants: bool = True


# qdrant.ScoredPoint without id, to keep the same interface.
class ScoredRow(BaseModel):
    score: float
    payload: dict


class ConceptData(BaseModel):
    CONCEPT_ID: int
    CONCEPT_CODE: str | None = None
    CONCEPT_NAME: str
    DOMAIN_ID: str
    VOCABULARY_ID: str | None = None
    STANDARD_CONCEPT: str | None = None
    # False when INVALID_REASON is set in OMOP; defaults True for sources lacking validity info (LLM fallback, cache).
    IS_VALID: bool = True
    SOURCE: str | None = None
    # Seed concept a descendant enricher expanded this row from; None for original matches.
    ANCESTOR_CONCEPT_ID: int | None = None
    MEDCODE: str | None = None
    READCODE: str | None = None
    DMDID: str | None = None
    DMDCODE: str | None = None
    ORIGINALREADCODE: str | None = None
    SNOMEDCTCONCEPTID: str | None = None
    GEMSCRIPTCODE: str | None = None
    PATIENT_COUNT: int | None = None
    PRODCODE: str | None = None


class ScoredConcept(BaseModel):
    CONCEPT_DATA: ConceptData
    SCORE: float


class DrugEntry(BaseModel):
    DRUG_NAME: str
    DRUG_DATA: list[ScoredConcept]
