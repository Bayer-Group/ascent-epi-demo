"""Embedder factory.

Ports the old ``determine_collection_and_generator`` encoder switch. Maps an
encoder name to its ``(collection_name, Embedder)`` pair. Model ids and Qdrant
collections are preserved verbatim; the sparse/BM25 concern lives in the Qdrant
connector, not here.

Unlike the old module (which eagerly instantiated all four generators at
import), embedders are built lazily on first use and cached per encoder.
"""

from __future__ import annotations

import logging
import threading

from ascent_medical_coder.connectors.embeddings.base import Embedder
from ascent_medical_coder.connectors.embeddings.local import LocalEmbedder
from ascent_medical_coder.connectors.embeddings.remote import GeminiEmbedder
from ascent_medical_coder.core.settings import get_settings

logger = logging.getLogger(__name__)

_cache: dict[str, Embedder] = {}
_COLLECTIONS: dict[str, str] = {}
# Serializes lazy construction so concurrent first requests don't each load the same heavy model.
_build_lock = threading.Lock()

ENCODERS = ("bge", "sap", "biolord", "gemini")


def _build(encoder: str) -> tuple[str | None, Embedder | None]:
    s = get_settings()
    if encoder == "sap":
        return "concept", LocalEmbedder(s.llm.sap_embed_model)
    if encoder == "bge":
        return s.qdrant.bge_collection, LocalEmbedder(s.llm.bge_embed_model)
    if encoder == "gemini":
        return s.qdrant.gemini_collection, GeminiEmbedder(s.llm.gemini_embed_model)
    if encoder == "biolord":
        return s.qdrant.biolord_collection, LocalEmbedder(s.llm.biolord_embed_model)
    return None, None


def get_embedder(encoder: str) -> tuple[str | None, Embedder | None]:
    """Return the ``(collection_name, Embedder)`` pair for an encoder name.

    Returns ``(None, None)`` for an unknown encoder, matching the old switch.
    Construction is lazy, cached, and lock-protected (model loads are heavy).
    """
    if encoder in _cache:
        return _COLLECTIONS[encoder], _cache[encoder]

    with _build_lock:
        # Double-check: another thread may have built it while we waited.
        if encoder in _cache:
            return _COLLECTIONS[encoder], _cache[encoder]
        collection, embedder = _build(encoder)
        if embedder is not None and collection is not None:
            _cache[encoder] = embedder
            _COLLECTIONS[encoder] = collection
    return collection, embedder


def warm_embedders() -> None:
    """Eagerly build every known embedder (call off the event loop at startup).

    Restores the legacy operational profile — models load at boot, not inside
    the first user request. Per-encoder failures are logged, not raised, so a
    missing optional credential (e.g. Gemini) doesn't block startup.
    """
    for encoder in ENCODERS:
        try:
            collection, embedder = get_embedder(encoder)
            if embedder is not None:
                logger.info("Warmed embedder %r (collection=%s)", encoder, collection)
        except Exception as e:
            logger.warning("Could not warm embedder %r: %s", encoder, e)
