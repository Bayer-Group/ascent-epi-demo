"""Aggregated business routers.

``business_router`` bundles the five authenticated REST routers. ``main.create_app``
mounts it once under the service-path prefix with the Azure Security dependency.
The open system routes (health/version/test) and the self-authenticating WebSocket
are wired separately in ``main`` so they are NOT placed behind Security — matching
the current service exactly.
"""

from __future__ import annotations

from fastapi import APIRouter

from ascent_medical_coder.api.routers import (
    coding,
    concept_search,
    grounded_search,
    patient_counts,
    validate_codes,
)

business_router = APIRouter()
business_router.include_router(coding.router)
business_router.include_router(concept_search.router)
business_router.include_router(grounded_search.router)
business_router.include_router(patient_counts.router)
business_router.include_router(validate_codes.router)
