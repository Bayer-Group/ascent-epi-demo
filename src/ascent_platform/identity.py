"""Who is calling, when there is only ever one caller.

This install serves its own synthetic data to a single local user, so there is
nobody to authenticate and nothing to authorise against. The application still
needs a user object: ``user.email`` is the owner of cohorts, the key for saved
queries and the audit field on studies.

The MCP layer builds that object from the access-token claims (see
``ascent_mcp.tools._cohort_helpers.get_mcp_user``); the per-request predicates
below answer for the one identity that exists.
"""

from __future__ import annotations

from typing import Awaitable, Callable

# Kept as a type so the warehouse and domain signatures that thread a token
# fetcher still typecheck. Nothing produces a token any more; the callables
# passed around are no-ops and every consumer already tolerates ``None``.
TokenFetcher = Callable[[], Awaitable[str]]


def ci_service_account_mode() -> bool:
    """An escape hatch for CI runs with no user bearer to exchange.

    There is no exchange to skip, so the question is moot and the answer is
    False -- the machine-user path it selected does not exist either.
    """
    return False


def is_service_account_request() -> bool:
    """Whether an allowlisted service principal is calling."""
    return False


def is_reviewer_request() -> bool:
    """Whether the caller used the reviewer static key.

    Reviewer requests were confined to synthetic databases. Everything here is
    synthetic, so the distinction has no work left to do.
    """
    return False
