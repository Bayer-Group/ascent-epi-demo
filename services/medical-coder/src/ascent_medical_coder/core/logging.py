"""Logging configuration.

Ported from the legacy ``api/core/custom_logging.py``. The dead ``post_mcs_log``
HTTP sink (which depended on a nonexistent ``USR_SERVICE_URL``) is dropped.
"""

from __future__ import annotations

import logging
import sys

from ascent_medical_coder.core.settings import get_settings

logger = logging.getLogger(__name__)


class FilterDocs(logging.Filter):
    """Suppress uvicorn access-log noise from health/docs polling only.

    The ALB health check hits ``/{service_path}/docs`` every few seconds, which
    would flood the access log. Only those documentation/health probe lines are
    dropped — real API request logs (4xx/5xx visibility) are kept. (The legacy
    filter matched the whole service prefix, which in deployed envs would have
    silenced every access line.)
    """

    def __init__(self, service_path: str) -> None:
        super().__init__()
        prefix = f"/{service_path}"
        self._needles = (f"{prefix}/docs", f"{prefix}/openapi.json", f"GET {prefix}/ ")

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(needle in message for needle in self._needles)


def configure_logging() -> None:
    """Configure root logging and install the uvicorn.access docs filter."""
    settings = get_settings()

    handler = logging.StreamHandler(sys.stdout)
    log_format = "%(asctime)s [%(levelname)s] - %(name)s: %(message)s"

    logging.basicConfig(
        format=log_format,
        level=logging.DEBUG if settings.service.debug else logging.INFO,
        handlers=[handler],
    )

    access_logger = logging.getLogger("uvicorn.access")
    access_logger.addFilter(FilterDocs(settings.service.service_path))
