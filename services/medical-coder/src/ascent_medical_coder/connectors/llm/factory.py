"""LLM connector factories.

Two provider-agnostic entry points — services never name a vendor:

- :func:`get_llm_connector` — JSON-completion connector by key (the
  ``llm_filter`` request values: ``chatgpt`` / ``gemini`` / ``haiku``).
- :func:`get_grounded_search_connector` — the web-search-grounded connector,
  selected by ``settings.llm.grounded_search_provider``.

Instances are cached per key. Transport only — the include/exclude filter JSON
is built and parsed in the services layer, not here.
"""

from __future__ import annotations

import logging

from ascent_medical_coder.core.settings import get_settings

from .anthropic import BedrockAnthropicConnector
from .base import GroundedSearchConnector, LLMConnector
from .gemini import GeminiConnector
from .openai import OpenAIConnector

logger = logging.getLogger(__name__)

_CONNECTORS: dict[str, LLMConnector] = {}
_GROUNDED: dict[str, GroundedSearchConnector] = {}

# Registry of JSON-completion providers. Adding a provider = one entry here.
_LLM_PROVIDERS: dict[str, type] = {
    "chatgpt": OpenAIConnector,
    "gemini": GeminiConnector,
    "haiku": BedrockAnthropicConnector,
}

# Registry of grounded-search providers; Gemini (Google Search grounding) is the only one today.
_GROUNDED_PROVIDERS: dict[str, type] = {
    "gemini": GeminiConnector,
}


# What each provider needs before it can be constructed. Bedrock authenticates
# through boto3's own chain, so any AWS credential or region counts.
_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "chatgpt": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "haiku": ("AWS_REGION", "AWS_DEFAULT_REGION", "AWS_ACCESS_KEY_ID"),
}
_PREFERENCE = ("gemini", "chatgpt", "haiku")
_substituted: set[str] = set()


def _configured(key: str) -> bool:
    import os

    required = _REQUIREMENTS.get(key)
    return True if required is None else any(os.environ.get(v) for v in required)


def _resolve(key: str) -> str | None:
    """The provider to actually use, given what this deployment can reach.

    Callers name a provider directly -- ``llm_filter="haiku"`` is the common
    one -- which is a hard dependency on Bedrock for anybody who configured a
    different key. The request is honoured when it is configured, and
    otherwise substituted with a note, rather than failing at the call site.
    """
    if _configured(key):
        return key
    usable = [k for k in _LLM_PROVIDERS if _configured(k)]
    chosen = next((p for p in _PREFERENCE if p in usable), None) or (usable[0] if usable else None)
    if chosen and key not in _substituted:
        _substituted.add(key)
        logger.warning("LLM filter %r is not configured; using %r instead.", key, chosen)
    return chosen


def get_llm_connector(name: str) -> LLMConnector | None:
    """Return (and cache) the JSON-completion connector for a given key."""
    requested = name.lower().strip()
    key = _resolve(requested) or requested
    provider = _LLM_PROVIDERS.get(key)
    if provider is None:
        logger.error("Unknown LLM connector requested: %s", name)
        return None
    if key not in _CONNECTORS:
        _CONNECTORS[key] = provider()
    return _CONNECTORS[key]


def get_grounded_search_connector(name: str | None = None) -> GroundedSearchConnector:
    """Return (and cache) the grounded-search connector.

    Provider defaults to ``settings.llm.grounded_search_provider``; raises for
    unknown providers (misconfiguration should fail loudly, not fall back).
    """
    key = (name or get_settings().llm.grounded_search_provider).lower().strip()
    provider = _GROUNDED_PROVIDERS.get(key)
    if provider is None:
        raise ValueError(
            f"Unknown grounded-search provider {key!r}; "
            f"available: {sorted(_GROUNDED_PROVIDERS)}"
        )
    if key not in _GROUNDED:
        _GROUNDED[key] = provider()
    return _GROUNDED[key]


__all__ = ["get_grounded_search_connector", "get_llm_connector"]
