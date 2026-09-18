"""Shared HTTP client bounds."""

from ascent_platform.http.timeouts import (
    MEDICAL_CODER_CONNECT_TIMEOUT,
    MEDICAL_CODER_READ_TIMEOUT,
    MEDICAL_CODER_TOKEN_HTTP_TIMEOUT,
    MEDICAL_CODER_TOTAL_TIMEOUT,
    medical_coder_timeout,
)

__all__ = [
    "MEDICAL_CODER_CONNECT_TIMEOUT",
    "MEDICAL_CODER_READ_TIMEOUT",
    "MEDICAL_CODER_TOKEN_HTTP_TIMEOUT",
    "MEDICAL_CODER_TOTAL_TIMEOUT",
    "medical_coder_timeout",
]
