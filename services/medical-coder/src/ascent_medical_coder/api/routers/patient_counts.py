"""Router for patient-counts. Thin shell over ``services.patient_counts``."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ascent_medical_coder.core.security import get_current_user
from ascent_medical_coder.schemas.patient_counts import PatientCountRequest, PatientCountResponse
from ascent_medical_coder.schemas.user import User
from ascent_medical_coder.services.patient_counts import get_patient_counts as _get_patient_counts

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/patient-counts", response_model=PatientCountResponse)
async def get_patient_counts(
    request: PatientCountRequest,
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> PatientCountResponse:
    return await _get_patient_counts(request)
