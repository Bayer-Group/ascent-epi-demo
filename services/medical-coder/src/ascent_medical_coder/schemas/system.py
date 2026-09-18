"""System / meta schemas. Byte-identical to the current service."""

from __future__ import annotations

from pydantic import BaseModel


class VersionInfo(BaseModel):
    api_version: str
    release_version: str
