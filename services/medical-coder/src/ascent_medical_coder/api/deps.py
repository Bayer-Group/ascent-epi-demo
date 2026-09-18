"""Shared FastAPI dependencies.

Thin, reusable dependency providers for the routers. Business services construct
the connectors they need per call (matching current behavior); these helpers
cover auth and settings injection.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from ascent_medical_coder.core.security import get_current_user
from ascent_medical_coder.core.settings import Settings, get_settings
from ascent_medical_coder.schemas.user import User

CurrentUser = Annotated[User, Depends(get_current_user)]
AppSettings = Annotated[Settings, Depends(get_settings)]
