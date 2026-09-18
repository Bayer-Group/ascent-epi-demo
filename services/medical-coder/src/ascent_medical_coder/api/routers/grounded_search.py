"""Router for grounded-search. Thin shell over ``services.grounded_search``."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from ascent_medical_coder.schemas.grounded_search import GroundedSearchRequest, GroundedSearchResponse
from ascent_medical_coder.services.grounded_search import grounded_search as _grounded_search

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/grounded-search", response_model=GroundedSearchResponse)
async def grounded_search(request: GroundedSearchRequest) -> GroundedSearchResponse:
    return await _grounded_search(request)
