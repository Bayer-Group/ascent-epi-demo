"""SQLAlchemy ORM models for the medical-coder app database.

Ported from the legacy ``api/core/models.py``. Table and column names are kept
byte-for-byte identical so the rewrite binds to the existing Postgres schema.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class CacheEntry(Base):
    __tablename__ = "mcs_cache_entries"

    id = Column(Integer, primary_key=True)
    key = Column(String(1024), nullable=False, unique=True)
    hashed_key = Column(String(64), nullable=False)
    value = Column(Text, nullable=False)
    ttl_seconds = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    hits = Column(Integer, nullable=False, default=0)

    __table_args__ = (Index("ix_mcs_cache_entries_hashed_key", "hashed_key"),)


class MCSLog(Base):
    __tablename__ = "mcs_user_analytics"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String)
    query = Column(String)
    top_k = Column(Integer)
    standard_concept = Column(String, nullable=True)
    domain_id = Column(String, nullable=True)
    vocabulary = Column(String, nullable=True)
    llm_filter = Column(Boolean)
    lts_used = Column(Boolean)
    llm_n_requests = Column(Integer)
    execution_time_s = Column(Float)
    created_date = Column(DateTime(timezone=True), server_default=func.now())
