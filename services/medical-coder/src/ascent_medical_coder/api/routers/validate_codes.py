"""Router for validate-codes. Thin shell over ``services.validate_codes``."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ascent_medical_coder.core.security import get_current_user
from ascent_medical_coder.schemas.user import User
from ascent_medical_coder.schemas.validate_codes import ValidateCodesRequest, ValidateCodesResponse
from ascent_medical_coder.services.validate_codes import validate_codes as _validate_codes

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/validate-codes", response_model=ValidateCodesResponse)
async def validate_codes(
    request: ValidateCodesRequest,
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> ValidateCodesResponse:
    return await _validate_codes(request)
