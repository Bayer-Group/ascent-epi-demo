"""
SQLite-based persistent cache for medical coder API results.

This module provides a database cache to store medical code lookups, enabling:
1. Faster evaluation by avoiding redundant API calls
2. Reproducibility by freezing codes at creation time
3. Offline evaluation when medical coder API is unavailable

The cache is keyed by all parameters that affect medical coder results.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class MedicalCodesCache:
    """Manages SQLite database for caching medical coder API results."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize medical codes cache database.

        Args:
            db_path: Path to SQLite database file (will be created if doesn't exist)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()
        logger.info(f"Initialized medical codes cache: {self.db_path}")

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _initialize_schema(self):
        """Create database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Schema version tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Insert schema version if not exists
            cursor.execute(
                """
                INSERT OR IGNORE INTO schema_info (version) VALUES (?)
            """,
                (self.SCHEMA_VERSION,),
            )

            # Cache metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_entries INTEGER DEFAULT 0,
                    description TEXT
                )
            """)

            # Initialize metadata if not exists
            cursor.execute("""
                INSERT OR IGNORE INTO cache_metadata (id, description)
                VALUES (1, 'Medical codes cache for evaluation pipeline')
            """)

            # Medical codes cache table
            # Key: (query, domain_id, coding_system, database, vocabulary, top_k, llm_filter, cosine_similarity)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS medical_codes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    domain_id TEXT NOT NULL,
                    coding_system TEXT NOT NULL,
                    database_name TEXT NOT NULL,
                    vocabulary TEXT,
                    top_k INTEGER NOT NULL,
                    llm_filter TEXT,
                    cosine_similarity REAL,
                    codes_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1
                )
            """)

            # Create index for fast lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_medical_codes_lookup
                ON medical_codes (
                    query, domain_id, coding_system, database_name,
                    vocabulary, top_k, llm_filter, cosine_similarity
                )
            """)

            conn.commit()

    def _update_metadata(self, conn: sqlite3.Connection):
        """Update cache metadata after modifications."""
        cursor = conn.cursor()

        # Update last_modified_at and total_entries
        cursor.execute("""
            UPDATE cache_metadata
            SET last_modified_at = CURRENT_TIMESTAMP,
                total_entries = (SELECT COUNT(*) FROM medical_codes)
            WHERE id = 1
        """)

        conn.commit()

    def _normalize_key(
        self,
        query: str,
        domain_id: str,
        coding_system: str,
        database_name: str,
        vocabulary: Optional[List[str]],
        top_k: int,
        llm_filter: Optional[str],
        cosine_similarity: Optional[float],
    ) -> Tuple:
        """
        Normalize cache key parameters for consistent lookups.

        Args:
            query: Search query for medical codes
            domain_id: Domain ID (e.g., "Condition", "Drug")
            coding_system: Coding system ("Standard" or "Source")
            database_name: Snowflake database name
            vocabulary: List of vocabularies (for source coding)
            top_k: Number of results to return
            llm_filter: LLM filter name
            cosine_similarity: Similarity threshold

        Returns:
            Tuple of normalized parameters for cache key
        """
        # Normalize query (lowercase, strip whitespace)
        normalized_query = query.lower().strip()

        # Normalize domain_id (lowercase)
        normalized_domain = domain_id.lower().strip()

        # Normalize coding system
        normalized_coding = coding_system.upper().strip()

        # Normalize database name
        normalized_db = database_name.upper().strip()

        # Normalize vocabulary (sort for consistency, convert to JSON string)
        if vocabulary:
            normalized_vocab = json.dumps(sorted([v.upper().strip() for v in vocabulary]))
        else:
            normalized_vocab = None

        # Keep numeric values as-is
        normalized_top_k = top_k
        normalized_llm = llm_filter.lower().strip() if llm_filter else None
        normalized_similarity = cosine_similarity

        return (
            normalized_query,
            normalized_domain,
            normalized_coding,
            normalized_db,
            normalized_vocab,
            normalized_top_k,
            normalized_llm,
            normalized_similarity,
        )

    def get_codes(
        self,
        query: str,
        domain_id: str,
        coding_system: str,
        database_name: str,
        vocabulary: Optional[List[str]] = None,
        top_k: int = 500,
        llm_filter: Optional[str] = None,
        cosine_similarity: Optional[float] = None,
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Lookup medical codes from cache.

        Args:
            query: Search query for medical codes
            domain_id: Domain ID (e.g., "Condition", "Drug")
            coding_system: Coding system ("Standard" or "Source")
            database_name: Snowflake database name
            vocabulary: List of vocabularies (for source coding)
            top_k: Number of results to return
            llm_filter: LLM filter name
            cosine_similarity: Similarity threshold

        Returns:
            Cached medical codes dict, or None if not found
        """
        normalized = self._normalize_key(query, domain_id, coding_system, database_name, vocabulary, top_k, llm_filter, cosine_similarity)

        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT codes_json, created_at
                FROM medical_codes
                WHERE query = ? AND domain_id = ? AND coding_system = ?
                    AND database_name = ? AND vocabulary IS ?
                    AND top_k = ? AND llm_filter IS ? AND cosine_similarity IS ?
            """,
                normalized,
            )

            row = cursor.fetchone()

            if row:
                # Update access statistics
                cursor.execute(
                    """
                    UPDATE medical_codes
                    SET last_accessed_at = CURRENT_TIMESTAMP,
                        access_count = access_count + 1
                    WHERE query = ? AND domain_id = ? AND coding_system = ?
                        AND database_name = ? AND vocabulary IS ?
                        AND top_k = ? AND llm_filter IS ? AND cosine_similarity IS ?
                """,
                    normalized,
                )
                conn.commit()

                codes_dict = json.loads(row["codes_json"])
                created_at = row["created_at"]

                logger.debug(f"Cache HIT for query='{query}', domain='{domain_id}' (created: {created_at})")
                return codes_dict
            else:
                logger.debug(f"Cache MISS for query='{query}', domain='{domain_id}'")
                return None

    def store_codes(
        self,
        query: str,
        domain_id: str,
        coding_system: str,
        database_name: str,
        codes: Dict[str, List[Dict[str, Any]]],
        vocabulary: Optional[List[str]] = None,
        top_k: int = 500,
        llm_filter: Optional[str] = None,
        cosine_similarity: Optional[float] = None,
    ):
        """
        Store medical codes in cache.

        Args:
            query: Search query for medical codes
            domain_id: Domain ID (e.g., "Condition", "Drug")
            coding_system: Coding system ("Standard" or "Source")
            database_name: Snowflake database name
            codes: Medical codes dictionary from API
            vocabulary: List of vocabularies (for source coding)
            top_k: Number of results to return
            llm_filter: LLM filter name
            cosine_similarity: Similarity threshold
        """
        normalized = self._normalize_key(query, domain_id, coding_system, database_name, vocabulary, top_k, llm_filter, cosine_similarity)

        codes_json = json.dumps(codes)

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Insert or replace (in case of duplicate keys)
            cursor.execute(
                """
                INSERT OR REPLACE INTO medical_codes (
                    query, domain_id, coding_system, database_name,
                    vocabulary, top_k, llm_filter, cosine_similarity,
                    codes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (*normalized, codes_json),
            )

            # Update metadata
            self._update_metadata(conn)

            conn.commit()

        logger.info(f"Stored codes in cache for query='{query}', domain='{domain_id}'")

    def clear_cache(self):
        """Clear all cached codes (for testing or reset)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM medical_codes")
            self._update_metadata(conn)
            conn.commit()

        logger.warning(f"Cleared all cached codes from {self.db_path}")
