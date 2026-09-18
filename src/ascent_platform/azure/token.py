"""One service-principal token provider.

There were two, against the same Azure app registration and the same
medical-coder endpoint, differing only in how they spelled things:

    AscentClient   AZURE_CLIENT_ID / AZURE_CLIENT_SECRET / AZURE_TENANT_ID
                   authority built from the tenant id
                   scopes=[f"{client_id}/.default"]

    MSALClient     MSAL_CLIENT_ID / MSAL_CLIENT_SECRET / MSAL_AUTHORITY_URL
                   authority given as a full URL
                   scopes=[f"api://{client_id}/.default"]

The MSAL_* variables resolve to the same values as the AZURE_* ones (they are
now aliases on the same RuntimeSettings fields), and the two scope forms were
checked against Azure AD rather than assumed: both return a token with the same
`aud`, `tid`, `roles` and `iss`. So the split was two names for one thing.

`token` asks MSAL on every read. MSAL caches in memory and only calls Azure AD
near expiry, so it is cheap -- and it avoids the failure mode where a token was
snapshotted once and every request 401'd in a burst after ~60 minutes.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Could not obtain a service-principal token."""


class ServicePrincipalTokenProvider:
    """Client-credentials tokens for one Azure app registration.

    The medical coder runs in the compose stack with no authentication in
    front of it, so there is no token to obtain and none is sent. The class,
    its constructor signature and ``token`` are kept because callers construct
    it eagerly at import and hold it for the process lifetime.
    """

    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        tenant_id: str | None = None,
        authority: str | None = None,
        http_timeout: float | None = None,
    ) -> None:
        self.client_id = client_id

    @property
    def scope(self) -> str:
        return f"{self.client_id}/.default"

    # A non-empty placeholder rather than "". Callers cache on truthiness --
    # `if not self.token_cache` -- so an empty string re-acquires on every
    # single request. expires_in must be present too: the caller computes an
    # expiry from it unconditionally, and its absence raised KeyError
    # ('expires_in') from inside every medical-coder call.
    _PLACEHOLDER = "local-no-auth"

    def acquire(self) -> dict:
        return {"access_token": self._PLACEHOLDER, "expires_in": 3600}

    @property
    def token(self) -> str:
        """A placeholder bearer. Callers send it; the local coder ignores it."""
        return self._PLACEHOLDER

    def force_refresh(self) -> None:
        """Nothing is cached, so there is nothing to invalidate."""
