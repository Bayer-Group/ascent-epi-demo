"""Async engine / session management for the app Postgres database.

Single source of truth for the async engine and sessionmaker (the legacy code
had a duplicate sessionmaker in ``pg_cache.py`` — collapsed here). The engine is
built lazily and only when Postgres is actually configured, so an unconfigured
deployment (``USR_DB_HOST`` unset) never opens a connection pool.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from ascent_medical_coder.core.settings import get_settings
from ascent_medical_coder.db.models import Base

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_initialized: bool = False


def get_engine() -> AsyncEngine | None:
    """Return the lazily-built async engine, or ``None`` if Postgres is unconfigured."""
    global _engine, _session_factory
    settings = get_settings()
    if not settings.postgres.configured:
        return None
    if _engine is None:
        _engine = create_async_engine(settings.postgres.async_url, echo=False)
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)
    return _engine


def session_factory() -> async_sessionmaker[AsyncSession] | None:
    """Return the shared async sessionmaker, or ``None`` if Postgres is unconfigured."""
    get_engine()
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an ``AsyncSession``.

    Raises ``RuntimeError`` when Postgres is unconfigured.
    """
    factory = session_factory()
    if factory is None:
        raise RuntimeError("Database host not configured")
    async with factory() as session:
        yield session


async def init_models_once() -> None:
    """Create tables once. No-op if already initialized or Postgres unconfigured."""
    global _initialized
    if _initialized:
        return
    engine = get_engine()
    if engine is None:
        return
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    _initialized = True


async def dispose_engine() -> None:
    """Dispose the engine on shutdown. Safe to call when unconfigured."""
    global _engine, _session_factory, _initialized
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        _initialized = False
