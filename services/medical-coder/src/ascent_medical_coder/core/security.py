"""Authentication, with the directory removed.

This service runs inside the compose stack, reachable only by the backend on
the internal network, and there is no directory to validate against.

The names ``azure_scheme``, ``get_current_user`` and
``validate_websocket_token`` are what ``main.py`` and five routers depend on.

======================================================================
THIS PERFORMS NO AUTHENTICATION. The service is not published on the host
by docker compose; the backend reaches it over the compose network. Do not
expose its port without putting a real authenticating proxy in front.
======================================================================
"""

from __future__ import annotations

import logging

from ascent_medical_coder.core.settings import get_settings
from ascent_medical_coder.schemas.user import User

# Re-exported: routers import User from here, not from schemas.
__all__ = ["User", "azure_scheme", "get_current_user", "validate_websocket_token"]

logger = logging.getLogger(__name__)

_settings = get_settings()

# The single local caller. Mirrors what the Azure user provided, because the
# coder records it against cached codelists and long-term-storage rows.
_LOCAL_USER = User(
    email="demo@localhost",
    name="Local User",
    sub="local-user",
    iss="local",
)


# FastAPI introspects a dependency's signature and turns any parameter it does
# not recognise into a required request field, so these take no parameters
# beyond the ones the callers actually pass.


async def azure_scheme() -> User:
    """The dependency every router registers. Authenticates nothing."""
    return _LOCAL_USER


async def get_current_user() -> User:
    """The caller. There is only one."""
    return _LOCAL_USER


async def validate_websocket_token(token: str | None = None) -> User:
    """Authenticates nothing; the coding router calls this with a token."""
    return _LOCAL_USER
