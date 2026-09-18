"""Failures the platform can express without naming HTTP.

``ascent_platform.snowflake`` raised ``fastapi.HTTPException`` in five places.
That is infrastructure deciding a status code, and it had a concrete cost: the
ACL rules in ``ascent/db/snowflake_acl.py`` could not move into the domain,
because they CATCH those exceptions and the domain must not import fastapi.
Every surface that wanted the rule therefore imported it from the app package.

These carry the same meanings the status codes did. The surfaces translate:

    ascent/error_mapping.py     -> HTTP status + JSON body, unchanged
    ascent_mcp/error_handling   -> the structured-dict convention MCP tools use

Deliberately a separate hierarchy from ``ascent_domain.errors``: the platform
cannot import the domain, so it cannot raise a DomainError. The app maps both.

Status codes and detail strings are pinned by tests/contracts/test_error_surface.py
-- these are user-facing messages, so the wording is contract, not commentary.
"""

from __future__ import annotations


class PlatformError(Exception):
    """Base for infrastructure failures that reach a caller."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class NotAuthenticated(PlatformError):
    """The caller's credentials were rejected by the upstream system."""


class AccessDenied(PlatformError):
    """The caller is known but lacks the grant needed."""


class UpstreamTimeout(PlatformError):
    """A dependency did not answer within its budget."""
