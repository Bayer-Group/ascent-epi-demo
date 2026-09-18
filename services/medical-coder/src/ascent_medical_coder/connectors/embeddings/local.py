"""Local SentenceTransformer embedder.

Ports the ``else`` branch of the old ``EmbeddingsGenerator`` (SapBERT / BGE /
BioLORD): a ``SentenceTransformer`` loaded on CUDA when available, otherwise
CPU, with the blocking ``.encode`` offloaded to a thread.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """Wraps a ``sentence_transformers.SentenceTransformer`` model."""

    def __init__(self, model_name: str) -> None:
        # Imported lazily so the heavy ML deps are only required when a local
        # encoder is actually instantiated.
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    async def encode(self, text: str) -> list[float]:
        return (await asyncio.to_thread(self.model.encode, text, convert_to_tensor=True)).tolist()

    async def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return (await asyncio.to_thread(self.model.encode, texts, convert_to_tensor=True)).tolist()

    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
