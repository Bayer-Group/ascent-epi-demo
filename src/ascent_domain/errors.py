"""Domain errors, expressed without reference to HTTP.

Fourteen exception classes across services, data_queries, data_definitions and
util/polling subclassed fastapi.HTTPException. That is the domain not merely
raising HTTP errors but *defining* them, and it is what pins those modules to
the application layer -- a cohort rule cannot move to ascent_domain while its
failure mode is an HTTP 403.

Each class here carries what the caller needs to know (which cohort, which
ids) and nothing about transport. The surfaces translate:

    ascent/error_mapping.py     -> HTTP status + JSON body, unchanged
    ascent_mcp/error_handling   -> the structured-dict convention MCP tools use

Status codes and body shapes are pinned by tests/contracts/test_error_surface.py
so the translation can be shown to preserve them rather than merely intended to.
"""

from __future__ import annotations

from typing import Any


class DomainError(Exception):
    """Base for every failure the domain can express.

    `detail` is the body a caller receives. It stays a plain dict because that
    is what the current HTTPException subclasses produce, and preserving the
    shape is the point.
    """

    def __init__(self, detail: dict[str, Any] | str | None = None) -> None:
        # str as well as dict: some failures carry a bare message rather than a
        # structured body, and FastAPI renders both under "detail". Preserving
        # the existing shape matters more than making it uniform.
        self.detail: dict[str, Any] | str = detail if detail is not None else {}
        super().__init__(str(self.detail))


class NotFoundError(DomainError):
    """The thing asked for does not exist, or is not visible to this caller."""


class AccessDeniedError(DomainError):
    """The caller is known but not permitted."""


class ConflictError(DomainError):
    """The thing exists but is not in a state that allows this action."""


class BadRequestError(DomainError):
    """The request itself is malformed or contradictory.

    Distinct from InvalidRequestError: that one is 422 (understood but
    unprocessable), this is 400. Both exist because the current API returns
    both, and collapsing them would change what callers see.
    """


class InvalidRequestError(DomainError):
    """The request is well-formed but cannot be satisfied as asked."""


class NotAuthenticatedError(DomainError):
    """The caller could not be identified at all.

    Distinct from AccessDeniedError, which is a known caller without the right.
    """


class UpstreamError(DomainError):
    """A service this one depends on failed."""


class UpstreamUnavailableError(UpstreamError):
    """A dependency could not be reached at all -- connection or timeout."""


class UpstreamStatusError(UpstreamError):
    """A dependency answered with a status the caller should see verbatim.

    The only reason a domain error carries a number: an upstream's status is
    passed straight through, because a caller distinguishing 404 from 429
    would break if both collapsed to 502. `status` is the upstream's, not
    this API's opinion of it.
    """

    def __init__(self, status: int, detail: dict[str, Any] | str | None = None) -> None:
        super().__init__(detail)
        self.status = status


class InternalError(DomainError):
    """A failure with no better classification."""
