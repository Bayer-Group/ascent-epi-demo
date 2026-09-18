"""The domain's view of configuration -- which is now the same object as
everyone else's.

This module used to declare ``DomainSettings``, one of five settings classes.
Between them they declared 92 distinct fields of which 43 appeared in more than
one class and 25 disagreed on type or default. ``AZURE_CLIENT_ID`` and
``AZURE_CLIENT_SECRET`` were in all four of the non-base classes. Keeping them
consistent needed parity tests, a field snapshot, and a hand comparison before
every move -- and drift still got through three times in one refactor
(SNOWFLAKE_OAUTH_SCOPE, REDIS_HOST, USER_WAREHOUSE).

There is one ``Settings`` now, in ``ascent_platform.config.runtime``, at the
bottom of the layering so every package can read it. The names here are kept so
call sites did not all have to change in the same commit; prefer importing from
the platform in new code.
"""

from __future__ import annotations

from ascent_platform.config.runtime import Settings, get_settings, set_settings

# Historical names. DomainSettings was never a distinct schema -- every field it
# declared also existed on at least one other class, with the same value.
DomainSettings = Settings
get_domain_settings = get_settings
set_domain_settings = set_settings

__all__ = ["DomainSettings", "Settings", "get_domain_settings", "get_settings", "set_domain_settings"]
