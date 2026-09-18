"""The TLS decision shared by every Redis client in the system.

* ElastiCache uses in-transit encryption in deployed environments, so TLS must
  be on. ``CACHE`` is ``"redis"`` when deployed and ``"redis-local"`` for local
  dev; talking plaintext to a TLS endpoint produces "Timeout reading from
  socket" failures.
* Certificate verification is disabled because the endpoint presents a cert for
  the AWS-internal hostname.

Only that decision is shared. Timeouts, pool sizes, the RESP2 protocol pin and
sync-vs-async are per-client and stay with their callers — they are tuned for
different call sites (a 2s connect on the auth hot path versus 5s for a
best-effort cache).
"""

from __future__ import annotations

import os
from typing import Any


def tls_enabled(cache_mode: str | None = None) -> bool:
    """True when talking to a TLS-terminated Redis (deployed ElastiCache)."""
    mode = cache_mode if cache_mode is not None else os.environ.get("CACHE")
    return mode == "redis"


def tls_kwargs(cache_mode: str | None = None) -> dict[str, Any]:
    """Redis client kwargs implementing the TLS decision."""
    use_ssl = tls_enabled(cache_mode)
    return {
        "ssl": use_ssl,
        "ssl_cert_reqs": "none" if use_ssl else "required",
        "ssl_check_hostname": False,
    }
