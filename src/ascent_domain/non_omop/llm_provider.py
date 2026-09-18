"""The injection point for the non-OMOP workflows' LLM service.

Seven workflow classes each built their own ``LLMRouter`` with a byte-identical
argument list. Two of them accepted an override; five hard-coded it, so those
five could not be exercised without a live Gemini key and could not be pointed
at a different model without editing them.

This module owns that construction once. Every workflow now takes
``llm_service=None`` and resolves through ``get_llm_service()``, so the app can
inject at wiring time and tests can inject a fake.

Two consequences worth stating plainly:

* **The router is now shared.** Its only state is a per-model client cache, and
  the clients themselves hold configuration plus an SDK handle (``genai.Client``,
  ``AzureOpenAI``) that is built for reuse. Previously every workflow instance
  built its own client for the same model; now they share one. That removes
  duplicate client construction rather than adding shared mutable state.
* **The concurrency bound is unchanged.** ``LLMRouter`` acquires its semaphore
  from a per-event-loop global, not from the router instance, so collapsing
  seven routers into one does not widen or narrow the in-flight limit.

Deliberately not routed through ``ascent_platform.llm``: that layer's protocol
returns a normalised ``LLMResponse``, while these call sites consume the raw
dict-or-string the two client families return. Adapting them is a behaviour
change on that path and needs its own commit.
"""

from __future__ import annotations

import threading
from typing import Optional

from ascent_platform.config.runtime import get_runtime_settings
from ascent_platform.llm.router import LLMRouter

_lock = threading.Lock()
_instance: Optional[LLMRouter] = None


def get_llm_service() -> LLMRouter:
    """Return the shared router, building it on first use."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = LLMRouter(
                    gemini_api_key=get_runtime_settings().GEMINI_API_KEY,
                    fallback_model_name=get_runtime_settings().FALLBACK_MODEL_NAME,
                )
    return _instance


def set_llm_service(service: Optional[LLMRouter]) -> None:
    """Override the shared router. ``None`` restores lazy construction."""
    global _instance
    with _lock:
        _instance = service


def reset_llm_service() -> None:
    """Drop the cached router so the next call rebuilds it from settings."""
    set_llm_service(None)
