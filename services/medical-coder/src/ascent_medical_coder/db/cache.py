"""Postgres-backed cache for medical-coder responses.

Ported from the legacy ``api/core/pg_cache.py``. All session/engine access flows
through ``db.session`` — there is no second sessionmaker here.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ascent_medical_coder.schemas.coding import MedicalCodingRequest

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from ascent_medical_coder.db.models import CacheEntry
from ascent_medical_coder.db.session import init_models_once, session_factory

logger = logging.getLogger(__name__)


def canonical_cache_key(
    *,
    query: str,
    domain_id: str | None,
    top_k: int,
    encoder: str | None,
    vocabulary: list[str] | str | None,
    standard_concept: str | None,
    source: str | None,
    llm_filter: str | None,
    allow_lts: bool | None,
    database: str | None,
    custom_instructions: str | None = None,
    cosine_similarity: float | None = None,
    include_descendants: bool | None = None,
    google_search_fallback: bool = False,
    use_hybrid: bool = False,
) -> str:
    """Versioned key containing every result-affecting search option.

    JSON avoids delimiter collisions and preserves nulls and booleans. The
    version invalidates old entries made before fallback/hybrid were included.
    Preserve query whitespace: the cached response is keyed by the exact query.
    """
    return json.dumps(
        {
            "version": 2,
            "query": query,
            "domain_id": domain_id.lower() if domain_id is not None else None,
            "top_k": top_k,
            "encoder": encoder,
            "vocabulary": sorted(set(vocabulary)) if isinstance(vocabulary, list) else vocabulary,
            "standard_concept": standard_concept,
            "source": source,
            "llm_filter": llm_filter,
            "allow_lts": allow_lts,
            "database": database,
            "custom_instructions": custom_instructions,
            "cosine_similarity": cosine_similarity,
            "include_descendants": include_descendants,
            "google_search_fallback": google_search_fallback,
            "use_hybrid": use_hybrid,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def cache_key_for_request(payload: MedicalCodingRequest, domain_id: str | None) -> str:
    """Keep request-to-key mapping shared by the pipeline and cache tests.

    domain_ids is replaced by the individual domain being cached. use_lts
    controls whether to access the cache, not the concepts generated.
    """
    return canonical_cache_key(
        domain_id=domain_id,
        **payload.model_dump(exclude={"domain_ids", "use_lts"}),
    )


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


async def _ensure_table(session: AsyncSession) -> None:
    try:
        await session.execute(select(CacheEntry).limit(1))
    except ProgrammingError:
        await init_models_once()


async def get_cached(key: str) -> Any | None:
    factory = session_factory()
    if factory is None:
        return None
    async with factory() as session:
        await _ensure_table(session)
        now = datetime.now(UTC)
        hkey = hash_key(key)
        stmt = select(CacheEntry).where(CacheEntry.hashed_key == hkey)
        res = await session.execute(stmt)
        entry = res.scalars().first()
        if not entry:
            return None
        if entry.expires_at <= now:
            try:
                await session.delete(entry)
                await session.commit()
            except Exception:
                await session.rollback()
            return None
        entry.hits = (entry.hits or 0) + 1
        try:
            await session.commit()
        except Exception:
            await session.rollback()
        try:
            return json.loads(entry.value)
        except Exception:
            try:
                await session.delete(entry)
                await session.commit()
            except Exception:
                await session.rollback()
            return None


async def set_cached(key: str, value: Any, ttl_seconds: int) -> None:
    factory = session_factory()
    if factory is None:
        return
    payload = json.dumps(value, ensure_ascii=False)
    async with factory() as session:
        await _ensure_table(session)
        now = datetime.now(UTC)
        expires = now + timedelta(seconds=ttl_seconds)
        hkey = hash_key(key)

        stmt = insert(CacheEntry).values(
            key=key,
            hashed_key=hkey,
            value=payload,
            ttl_seconds=ttl_seconds,
            expires_at=expires,
            hits=0,
        )

        stmt = stmt.on_conflict_do_update(
            index_elements=["key"],
            set_={
                "value": payload,
                "ttl_seconds": ttl_seconds,
                "expires_at": expires,
                "updated_at": now,
            },
        )

        try:
            await session.execute(stmt)
            await session.commit()
        except Exception:
            await session.rollback()
            # We don't want to crash the whole request if cache write fails
            logger.warning("pg cache write failed (suppressed)", exc_info=True)


async def invalidate(key: str) -> None:
    factory = session_factory()
    if factory is None:
        return
    async with factory() as session:
        await _ensure_table(session)
        hkey = hash_key(key)
        await session.execute(delete(CacheEntry).where(CacheEntry.hashed_key == hkey))
        await session.commit()
