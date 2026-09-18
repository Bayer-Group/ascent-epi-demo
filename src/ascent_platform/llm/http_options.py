"""One place that decides the Gemini SDK's HTTP timeout.

google-genai applies no request timeout by default: a stalled call hangs until
the peer gives up. That cost ~17 minutes of a single pipeline run before
GeminiClient started passing HttpOptions.

That fix landed in one of four construction sites. The other three -- the OMOP
QA assistant, Google Search grounding, and uncertainty embeddings -- kept the
SDK default, and the OMOP one is on the hot path. Same SDK, same failure mode,
same repo, so the bound belongs in one function rather than in each caller.

Env-tunable via GEMINI_HTTP_TIMEOUT_SECONDS so deployments can move it without
a code change; the default matches what GeminiClient already used.
"""

from __future__ import annotations

from google.genai.types import HttpOptions

from ascent_platform.config.runtime import get_runtime_settings

DEFAULT_GEMINI_TIMEOUT_SECONDS = get_runtime_settings().GEMINI_HTTP_TIMEOUT_SECONDS


def gemini_http_options(timeout_seconds: float | None = None) -> HttpOptions:
    """HttpOptions carrying a request timeout. HttpOptions takes milliseconds."""
    seconds = DEFAULT_GEMINI_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    return HttpOptions(timeout=int(seconds * 1000))
