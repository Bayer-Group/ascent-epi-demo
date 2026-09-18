from __future__ import annotations

from ascent_domain.database_access import get_user_databases
from ascent_mcp._synthetic import is_synthetic_database
from ascent_platform.identity import is_reviewer_request


async def get_user_snowflake_token() -> str | None:
    """Return the warehouse token for the current request — always ``None``.

    No per-user authentication is performed: this deployment has a single local
    identity and the warehouse ignores the token entirely. The function exists so
    callers and helpers that expect a token fetcher keep a stable signature.

    Returns:
        ``None``, always. Never raises.
    """
    return None


async def assert_user_permission_db(database: str, sf_token: str | None = None) -> None:
    """Raise ValueError if ``database`` is not one the caller may access.

    Reviewer static-key requests (see ``is_reviewer_request``) are additionally
    confined to databases flagged ``IS_SYNTHETIC``.
    """
    if is_reviewer_request() and not await is_synthetic_database(database):
        raise ValueError("Reviewer access is restricted to synthetic databases.")

    if sf_token is None:
        sf_token = await get_user_snowflake_token()
    database_names = await get_user_databases(sf_token)

    if database not in database_names:
        raise ValueError("User does not have access to the specified db.")


async def authorize_user_db(database: str, sf_token: str | None = None) -> str | None:
    """Verify access to ``database`` and return the warehouse token.

    Convenience wrapper around ``get_user_snowflake_token`` +
    ``assert_user_permission_db`` so MCP tools can replace the two-line pattern
    with a single call. No authentication happens: ``get_user_snowflake_token``
    always returns ``None``, so the return value is ``None`` unless the caller
    passed an ``sf_token``. The access check on ``database`` is real and does
    raise.
    """
    if sf_token is None:
        sf_token = await get_user_snowflake_token()
    await assert_user_permission_db(database, sf_token=sf_token)
    return sf_token
