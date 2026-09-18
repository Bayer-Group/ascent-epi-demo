import asyncio
from typing import Set

from ascent_domain.synthetic import get_synthetic_databases
from ascent_platform.cache.aiocache_backend import cache
from ascent_platform.constants import ExpirationTTL
from ascent_platform.identity import is_reviewer_request, is_service_account_request
from ascent_platform.warehouse.catalog import PUBLIC_DATABASES

ACL_CACHE_KEY = "snowflake_acl"

# The curated set of databases this server will surface: the databases the
# warehouse was built with. It is what stops a database being offered before
# its catalog exists.


async def _get_allowlist(rwd_db=None) -> Set[str]:
    from ascent_platform.warehouse.bootstrap import DATABASES

    return {db.name for db in DATABASES}


async def _get_allowlist_cached() -> Set[str]:
    if allowlist := await cache.get(ACL_CACHE_KEY):
        return allowlist

    allowlist = await _get_allowlist()
    await cache.set(ACL_CACHE_KEY, allowlist, ttl=ExpirationTTL.HOUR * 3)

    return allowlist


async def get_user_databases(oauth_token: str) -> Set[str]:
    """Return the databases the caller may access in Ascent.

    The result is the curated allowlist of medical RWD databases the warehouse
    was built with, narrowed for reviewer requests to the synthetic ones.
    ``oauth_token`` is accepted for signature compatibility and is not used to
    compute grants.
    """
    # Reviewer requests are confined to synthetic-only databases so the listing
    # itself doesn't advertise real-data databases (querying them is blocked
    # separately in assert_user_permission_db, but a listing that named them
    # anyway would be misleading for a "synthetic-only" grant).
    if is_reviewer_request():
        allowlist, synthetic = await asyncio.gather(_get_allowlist_cached(), get_synthetic_databases())
        return allowlist & synthetic

    # Service-account requests get the curated allowlist directly.
    if is_service_account_request():
        return await _get_allowlist_cached()

    allowlist, sf_databases = await asyncio.gather(_get_allowlist_cached(), _show_databases_for_user(oauth_token))
    return allowlist & sf_databases


async def _show_databases_for_user(oauth_token: str) -> Set[str]:
    """Return the database names visible to the caller: the ``PUBLIC_DATABASES`` set.

    There is a single identity and no per-database grants, so every shipped
    database is visible. ``oauth_token`` is unused.

    The caller intersects this with the allowlist, which is what stops a
    database appearing before its metadata exists.
    """
    return set(PUBLIC_DATABASES)
