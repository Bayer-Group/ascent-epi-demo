"""Translate domain errors into HTTP responses.

The domain raises DomainError subclasses that know nothing about transport;
this is where they become status codes. The mapping reproduces exactly what the
HTTPException subclasses returned before -- see
tests/contracts/test_error_surface.py, which pins both the code and the body
shape for all fourteen.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from ascent_domain.errors import (
    AccessDeniedError,
    BadRequestError,
    ConflictError,
    DomainError,
    InternalError,
    InvalidRequestError,
    NotAuthenticatedError,
    NotFoundError,
    UpstreamError,
    UpstreamStatusError,
    UpstreamUnavailableError,
)
from ascent_platform.errors import AccessDenied, NotAuthenticated, PlatformError, UpstreamTimeout

# The platform raises its own hierarchy -- it cannot import the domain, so it
# cannot raise a DomainError. These are the statuses its five former
# HTTPExceptions used, preserved exactly.
STATUS_BY_PLATFORM_ERROR: dict[type[PlatformError], int] = {
    NotAuthenticated: status.HTTP_401_UNAUTHORIZED,
    AccessDenied: status.HTTP_403_FORBIDDEN,
    UpstreamTimeout: status.HTTP_504_GATEWAY_TIMEOUT,
}


STATUS_BY_ERROR: dict[type[DomainError], int] = {
    NotFoundError: status.HTTP_404_NOT_FOUND,
    BadRequestError: status.HTTP_400_BAD_REQUEST,
    AccessDeniedError: status.HTTP_403_FORBIDDEN,
    ConflictError: status.HTTP_409_CONFLICT,
    InvalidRequestError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    NotAuthenticatedError: status.HTTP_401_UNAUTHORIZED,
    UpstreamError: status.HTTP_502_BAD_GATEWAY,
    UpstreamUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
    InternalError: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


def status_for(error: DomainError) -> int:
    """Most specific match wins, so a subclass can override its parent."""
    if isinstance(error, UpstreamStatusError):
        # Carries the upstream's own status; see the class docstring.
        return error.status
    for cls in type(error).__mro__:
        if cls in STATUS_BY_ERROR:
            return STATUS_BY_ERROR[cls]
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def platform_status_for(error: PlatformError) -> int:
    """Most specific match wins, as with domain errors."""
    for cls in type(error).__mro__:
        if cls in STATUS_BY_PLATFORM_ERROR:
            return STATUS_BY_PLATFORM_ERROR[cls]
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def register_domain_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def _handle(_: Request, exc: DomainError) -> JSONResponse:
        # `detail` keeps the same key the HTTPException handler used, so the
        # response body is byte-identical to what callers received before.
        return JSONResponse(status_code=status_for(exc), content={"detail": exc.detail})

    @app.exception_handler(PlatformError)
    async def _handle_platform(_: Request, exc: PlatformError) -> JSONResponse:
        # Same body shape the HTTPExceptions produced: a bare string under
        # "detail". These are user-facing messages about Snowflake access.
        return JSONResponse(
            status_code=platform_status_for(exc), content={"detail": exc.detail}
        )
