"""Router for bulk concept search. Thin shell over ``services.concept_search``."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from ascent_medical_coder.schemas.concept_search import BulkConceptSearchRequest, Concept
from ascent_medical_coder.services.concept_search import execute_concept_search

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/bulk-concept-search")
async def search_concepts(request: BulkConceptSearchRequest) -> list[Concept]:
    return await execute_concept_search(request)
