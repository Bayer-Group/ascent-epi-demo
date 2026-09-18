"""Authenticated-user schema. Byte-identical to the current service."""

from __future__ import annotations

from pydantic import BaseModel


class User(BaseModel):
    email: str
    name: str | None = None
    family_name: str | None = None
    given_name: str | None = None
    sub: str | None = None
    picture: str | None = None
    exp: int | None = None
    iss: str | None = None
