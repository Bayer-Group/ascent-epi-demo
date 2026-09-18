"""Clients for services outside this codebase."""

from ascent_platform.external.ascent_client import AscentClient
from ascent_platform.external.medical_coder import MedicalCoder

__all__ = ["AscentClient", "MedicalCoder"]
