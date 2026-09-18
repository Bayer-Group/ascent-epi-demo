"""System / meta routes: health, version, and the test harness endpoints.

Ported faithfully from the legacy ``api/main.py`` app-level routes. Response
bodies and behavior are preserved verbatim; the only change is that config now
comes from typed settings instead of the flat ``config`` module.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter

from ascent_medical_coder.core.settings import get_settings
from ascent_medical_coder.schemas.system import VersionInfo

router = APIRouter()

logger = logging.getLogger(__name__)


@router.get("/")
async def get_health_signal() -> dict:
    """API server health signal."""
    return {"msg": "Server is up and running"}


@router.post("/version")
async def get_service_version() -> VersionInfo:
    settings = get_settings().service
    return VersionInfo(api_version=settings.api_version, release_version=settings.release_version)


@router.get("/test-timeout")
async def sleep(time: int = 5) -> None:
    """Returns a static response after the given time period in seconds"""
    await asyncio.sleep(time)
    return {"Hello": "World"}


@router.post("/test-error")
async def error() -> None:
    """Provokes a crash by 0 / 0"""
    0 / 0  # noqa: B018 -- intentional ZeroDivisionError to exercise the 500 handler
    return {"Hello": "World"}
