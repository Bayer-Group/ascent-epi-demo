"""Bounds for outbound calls to the medical-coder service.

``sock_read`` is the gap between received bytes, and the service is silent
while it runs LLM filtering, so a read timeout that is too tight severs slow
but healthy calls. ``total`` must be set as well, or ``sock_*`` alone never
trips on a response that drips.
"""

from __future__ import annotations

import aiohttp

from ascent_platform.config.runtime import get_runtime_settings

MEDICAL_CODER_CONNECT_TIMEOUT = get_runtime_settings().MEDICAL_CODER_CONNECT_TIMEOUT
MEDICAL_CODER_READ_TIMEOUT = get_runtime_settings().MEDICAL_CODER_READ_TIMEOUT
# Hard ceiling on one call, including a slow-drip response body.
MEDICAL_CODER_TOTAL_TIMEOUT = get_runtime_settings().MEDICAL_CODER_TOTAL_TIMEOUT
# Azure AD token fetch for these clients.
MEDICAL_CODER_TOKEN_HTTP_TIMEOUT = get_runtime_settings().MEDICAL_CODER_TOKEN_HTTP_TIMEOUT


def medical_coder_timeout() -> aiohttp.ClientTimeout:
    """The aiohttp timeout every medical-coder client should use."""
    return aiohttp.ClientTimeout(
        total=MEDICAL_CODER_TOTAL_TIMEOUT,
        sock_connect=MEDICAL_CODER_CONNECT_TIMEOUT,
        sock_read=MEDICAL_CODER_READ_TIMEOUT,
    )
