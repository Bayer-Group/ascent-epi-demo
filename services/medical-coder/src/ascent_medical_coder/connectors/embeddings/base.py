"""Embedder protocol.

The old ``EmbeddingsGenerator`` was a single class branching on ``model_name``.
It is split here into a small ``Embedder`` protocol implemented by local
(SentenceTransformer) and remote (Azure OpenAI / Gemini) backends.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Async embedding backend."""

    async def encode(self, text: str) -> list[float]:
        """Embed a single text into one vector."""
        ...

    async def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into one vector each."""
        ...

    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...
