"""Azure constants shared across the packages.

The tenant id comes from ``settings.AZURE_TENANT_ID``. A tenant id is not a
secret -- it appears in every OAuth URL -- but a specific organisation's tenant
hardcoded into a public repository is a fingerprint of that organisation, and
it is the wrong default for everyone else.

Resolve it by calling this at construction time rather than using it as an
argument default, so importing a client does not read the environment.
"""

from typing import Optional


def default_azure_tenant_id() -> Optional[str]:
    """The configured tenant, or None when Azure auth is not set up.

    None is a legitimate answer here: this distribution talks to a local
    medical-coder service that does not authenticate, so nothing on the default
    path needs a tenant. The callers that do need one fail on a missing tenant
    with a clearer message than a wrong one would produce.
    """
    from ascent_platform.config.runtime import get_runtime_settings

    return get_runtime_settings().AZURE_TENANT_ID
