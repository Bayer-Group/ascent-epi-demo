"""Qdrant client construction and search primitives.

Retrieval only: the raw hits are returned and the caller applies any LLM
filter, which belongs to the services layer.

Hybrid score: ``0.85 * cosine + 0.15 * sparse``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging

import numpy as np
import pandas as pd
from qdrant_client import AsyncQdrantClient, models
from tenacity import retry, stop_after_attempt, wait_fixed

from ascent_medical_coder.core.settings import get_settings

# Deliberate layering exception (no import cycle): the connector resolves vocabulary via the services layer.
from ascent_medical_coder.services.pipeline.vocabulary import match_vocabulary_ids

logger = logging.getLogger(__name__)

# Backpressure: process-wide cap so fan-out queues instead of stampeding the single Qdrant instance.
_search_semaphore: asyncio.Semaphore | None = None


def _get_search_semaphore() -> asyncio.Semaphore:
    global _search_semaphore
    if _search_semaphore is None:
        _search_semaphore = asyncio.Semaphore(get_settings().qdrant.max_concurrent_searches)
    return _search_semaphore


def _stable_hash(token: str) -> int:
    """Stable MD5-derived hash bucketed into ``[0, 999_999]`` (ported verbatim)."""
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % 1_000_000


def _build_sparse_vector(text: str) -> models.SparseVector:
    """Builds a SparseVector from token frequencies keyed by a stable hash."""
    if not text:
        return models.SparseVector(indices=[], values=[])
    tokens = [t for t in str(text).lower().split() if t]
    if not tokens:
        return models.SparseVector(indices=[], values=[])
    freq: dict[int, float] = {}
    for t in tokens:
        idx = _stable_hash(t)
        freq[idx] = freq.get(idx, 0.0) + 1.0
    return models.SparseVector(indices=list(freq.keys()), values=list(freq.values()))


async def _convert_hits_to_df(hits) -> pd.DataFrame:
    """Flatten Qdrant hit payloads into a DataFrame (retrieval-only shape)."""
    rows = [hit.payload.copy() for hit in (hits or []) if hit.payload]
    return pd.DataFrame(rows)


def get_qdrant_client() -> AsyncQdrantClient:
    """Construct an ``AsyncQdrantClient`` from settings."""
    s = get_settings().qdrant
    return AsyncQdrantClient(host=s.host, port=s.port, timeout=s.timeout)


async def create_search_filter(
    domain_id: str | None,
    vocabulary: list[str] | None,
    standard_concept: str | None,
    source: str | None,
) -> models.Filter:
    """Creates search filter based on provided criteria."""
    must = []
    if domain_id:
        must.append(models.FieldCondition(key="DOMAIN_ID", match=models.MatchValue(value=domain_id.title())))
    if source:
        must.append(models.FieldCondition(key="SOURCE", match=models.MatchValue(value=source)))
    if vocabulary:
        matched_vocab_ids = await match_vocabulary_ids(vocabulary)

        if matched_vocab_ids:
            unique_vocab_matches = list(set(matched_vocab_ids))
            must.append(models.FieldCondition(key="VOCABULARY_ID", match=models.MatchAny(any=unique_vocab_matches)))
        else:
            must.append(models.FieldCondition(key="VOCABULARY_ID", match=models.MatchAny(any=vocabulary)))
    if standard_concept:
        must.append(
            models.FieldCondition(key="STANDARD_CONCEPT", match=models.MatchValue(value=standard_concept.upper()))
        )
    return models.Filter(must=must)


class QdrantConnector:
    """Async Qdrant retrieval adapter wrapping a single ``AsyncQdrantClient``."""

    def __init__(self, client: AsyncQdrantClient | None = None) -> None:
        self.client = client or get_qdrant_client()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def search_qdrant(
        self,
        search_text: str,
        collection_name: str,
        query_vector: list[float],
        domain_id: str | None = None,
        vocabulary: list[str] | None = None,
        standard_concept: str | None = None,
        top_k: int = 1000,
        cosine_similarity: float = 0.5,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Search the vector database via ``query_points``.

        Returns the raw retrieved hits as a DataFrame; applying any LLM filter
        is the caller's responsibility.
        """
        query_filter = await create_search_filter(domain_id, vocabulary, standard_concept, source)
        try:
            async with _get_search_semaphore():
                result = await self.client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=top_k,
                    score_threshold=cosine_similarity,
                    query_filter=query_filter,
                    search_params=models.SearchParams(exact=True),
                    timeout=140,
                )
        except Exception:
            logger.exception("Qdrant Database error")
            raise

        hits_df = await _convert_hits_to_df(result.points)
        if not hits_df.empty:
            hits_df = hits_df.dropna(subset=["CONCEPT_ID"])
            hits_df = hits_df.where(pd.notna(hits_df), None).replace({np.nan: None})
        return hits_df

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def search_qdrant_hybrid(
        self,
        search_string: str,
        collection: str,
        cosine_cutoff: float | None,
        dense: list[float],
        top_k: int = 10_000,
        vocabulary: list[str] | None = None,
        standard_concept: str | None = None,
        domain_id: str | None = None,
        source: str | None = None,
        use_hybrid: bool = True,
    ) -> pd.DataFrame:
        """Dense-only semantic search with a hard cosine cutoff, followed by
        sparse-score re-ranking when ``use_hybrid`` is True.

        If ``use_hybrid`` is False, only dense vector search is performed.
        """
        cosine_cutoff = cosine_cutoff or 0.65

        query_filter = await create_search_filter(domain_id, vocabulary, standard_concept, source)
        query_filter.must_not = [
            models.FieldCondition(key="DOMAIN_ID", match=models.MatchValue(value="Drug")),
        ]

        # Permit acquired per network call, not across the dense+sparse pair, so one request can't hold two slots.
        async with _get_search_semaphore():
            dense_result = await self.client.query_points(
                collection_name=collection,
                query=models.NearestQuery(nearest=dense),
                using="dense",
                query_filter=query_filter,
                limit=top_k,
                score_threshold=cosine_cutoff,
                with_payload=True,
                with_vectors=False,
                timeout=140,
            )

        dense_hits = dense_result.points or []
        if not dense_hits:
            return pd.DataFrame()

        sparse_scores: dict = {}
        if use_hybrid:
            sparse = _build_sparse_vector(search_string)

            async with _get_search_semaphore():
                sparse_result = await self.client.query_points(
                    collection_name=collection,
                    query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                    using="sparse",
                    query_filter=query_filter,
                    limit=len(dense_hits),
                    with_payload=False,
                    with_vectors=False,
                )

            sparse_scores = {p.id: p.score for p in sparse_result.points or []}

        rows = []
        for hit in dense_hits:
            row = hit.payload.copy()
            row["COSINE_SIMILARITY"] = hit.score
            row["SPARSE_SCORE"] = sparse_scores.get(hit.id, 0.0)
            rows.append(row)

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        if use_hybrid:
            df["SCORE"] = 0.85 * df["COSINE_SIMILARITY"] + 0.15 * df["SPARSE_SCORE"]
        else:
            df["SCORE"] = df["COSINE_SIMILARITY"]

        df = df.dropna(subset=["CONCEPT_ID"])
        df = df.sort_values("SCORE", ascending=False)
        df = df.head(top_k)

        return df

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def search_drugs(
        self,
        keyword: str,
        top_k: int,
        domain_id: str | None,
        vocabulary: list[str] | None,
        standard_concept: str | None,
        source: str | None,
    ) -> pd.DataFrame:
        """Keyword-only drug search using ``MatchText`` on the BGE collection."""
        try:
            query_filter = await create_search_filter(domain_id, vocabulary, standard_concept, source)
            query_filter.must.append(models.FieldCondition(key="CONCEPT_NAME", match=models.MatchText(text=keyword)))
            async with _get_search_semaphore():
                hits, _ = await self.client.scroll(
                    collection_name=get_settings().qdrant.bge_collection,
                    scroll_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                )
        except Exception as err:
            logger.error("Qdrant keyword search error: %s", err)
            raise

        df = await _convert_hits_to_df(hits)
        if not df.empty:
            df = df.dropna(subset=["CONCEPT_ID"])
            df = df.where(pd.notna(df), None).replace({np.nan: None})
        return df
