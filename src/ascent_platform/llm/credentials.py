"""One resolution order for Google/Gemini credentials.

The same credential has two accepted spellings, GEMINI_API_KEY and
GOOGLE_API_KEY, and every call site resolves them here so that they cannot
disagree. If a site preferred one and a site preferred the other, a deployment
that sets both -- two projects, two quotas, a rotated key -- would authenticate
as different principals on different paths, with nothing reporting it.

**GEMINI_API_KEY wins.** It is the name deployments actually set.

Values come from RuntimeSettings rather than os.getenv so they are typed,
defaulted and visible in one place; see tests/seams/test_layering_rule.py.
"""

from __future__ import annotations

from typing import Optional

from ascent_platform.config.runtime import get_runtime_settings


def gemini_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """The Gemini/Google AI key: explicit argument, then GEMINI_API_KEY, then
    GOOGLE_API_KEY."""
    if explicit:
        return explicit
    settings = get_runtime_settings()
    return settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY


def google_cloud_project(explicit: Optional[str] = None) -> Optional[str]:
    """The Vertex AI project id: explicit argument, then GOOGLE_CLOUD_PROJECT."""
    if explicit:
        return explicit
    return get_runtime_settings().GOOGLE_CLOUD_PROJECT
