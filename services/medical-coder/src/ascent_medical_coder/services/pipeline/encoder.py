"""Encoder-related pipeline helpers (ported from ``utils.py``).

Hosts the cosine-cutoff resolution. The encoder→(collection, embedder)
resolution already lives in ``connectors/embeddings/factory.get_embedder`` and
is re-exported here so the pipeline has a single import site for encoder
concerns rather than duplicating the switch.
"""

from __future__ import annotations

import logging

from ascent_medical_coder.connectors.embeddings.factory import get_embedder

logger = logging.getLogger(__name__)

__all__ = ["determine_cosine_cutoff", "get_embedder"]


def determine_cosine_cutoff(encoder: str, payload_cutoff: float) -> float | None:
    """Determine the cosine cutoff based on the encoder or an explicit override.

    If ``payload_cutoff`` is set it takes precedence and is returned. Otherwise
    the ``gemini`` encoder gets a predefined cutoff of ``0.84``; any
    other encoder returns ``None``.

    Args:
        encoder (str): The encoder type used to pick a default cutoff when no
            payload cutoff is provided.
        payload_cutoff (float): An explicit cutoff supplied with the request.

    Returns:
        float | None: The cutoff to use.
    """
    if payload_cutoff:  # if the request has a cutoff value set use it
        return payload_cutoff
    elif encoder == "gemini":
        return 0.84
    return None
