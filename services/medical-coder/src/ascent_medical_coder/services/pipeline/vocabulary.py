"""Vocabulary resolution helpers.

Resolves free-form vocabulary strings (as emitted by the LLM) to canonical OMOP
``VOCABULARY_ID`` values, using the set of vocabulary ids present in the OMOP
concept table. Also hosts the sparse-vector builder that the Qdrant connector
uses for hybrid search.

Resolutions are memoized in memory for three hours.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time

import pandas as pd
from rapidfuzz import fuzz, process

from ascent_medical_coder.connectors.warehouse import SnowflakeConnector

logger = logging.getLogger(__name__)

query_get_all_unique_vocabs = """SELECT distinct VOCABULARY_ID
                                 FROM CONCEPT"""


# LLM shorthands (UPPERCASED keys) expanded to their real IDs; e.g. ambiguous "ICD10" fans out to CM+PCS.
VOCAB_ALIASES: dict[str, tuple[str, ...]] = {
    "ICD10": ("ICD10", "ICD10CM", "ICD10PCS"),
    "ICD-10": ("ICD10", "ICD10CM", "ICD10PCS"),
    "ICD9": ("ICD9CM", "ICD9Proc"),
    "ICD-9": ("ICD9CM", "ICD9Proc"),
    "ICD": ("ICD10", "ICD10CM", "ICD10PCS", "ICD9CM", "ICD9Proc"),
    "SNOMED": ("SNOMED",),
    "SNOMEDCT": ("SNOMED",),
    "SNOMED-CT": ("SNOMED",),
    "SNOMED CT": ("SNOMED",),
    "RXNORM": ("RxNorm", "RxNorm Extension"),
    "RX-NORM": ("RxNorm", "RxNorm Extension"),
}

# Per-process 3-hour memoization of the vocabulary-id set, guarded by an asyncio lock.
_VOCAB_CACHE_TTL_SECONDS = 3600 * 3
_vocab_cache: set[str] | None = None
_vocab_cache_expires_at: float = 0.0
_vocab_cache_lock = asyncio.Lock()


def _stable_hash(token: str) -> int:
    """Stable MD5-derived hash bucketed into ``[0, 999_999]``."""
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % 1_000_000


async def get_vocabulary_ids() -> set[str]:
    """Fetch vocabulary IDs and cache the result for three hours in memory.

    Queries the OMOP concept table (``CONCEPT``)
    for the distinct set of ``VOCABULARY_ID`` values. The result is memoized in
    process for three hours to reduce database round-trips.

    Returns:
        set[str]: The set of vocabulary IDs found in the concept table.
    """
    global _vocab_cache, _vocab_cache_expires_at

    now = time.monotonic()
    if _vocab_cache is not None and now < _vocab_cache_expires_at:
        return _vocab_cache

    async with _vocab_cache_lock:
        now = time.monotonic()
        if _vocab_cache is not None and now < _vocab_cache_expires_at:
            return _vocab_cache

        db = SnowflakeConnector()  # the local vocabulary database
        result_df: pd.DataFrame = await db.fetch_data_with_cursor(query_get_all_unique_vocabs)

        _vocab_cache = set(result_df["VOCABULARY_ID"])
        _vocab_cache_expires_at = time.monotonic() + _VOCAB_CACHE_TTL_SECONDS
        return _vocab_cache


async def match_vocabulary_ids(vocabulary: list[str]) -> set[str]:
    """Match vocabulary strings to their canonical OMOP vocabulary IDs.

    Resolution order per input token:
      1. Exact (case-sensitive) match against known vocabulary IDs.
      2. Case-insensitive exact match.
      3. Alias expansion via ``VOCAB_ALIASES`` — a single shorthand can map to
         multiple canonical IDs (e.g. ``ICD10`` → ``{ICD10CM, ICD10PCS, ICD10}``).
      4. Fuzzy matching that returns **all** candidates scoring >= 80 on
         ``partial_ratio``, not just the single best. This prevents silently
         losing sibling vocabularies when multiple are equally good matches.

    Parameters:
        vocabulary (list[str]): A list of vocabulary strings to be matched.

    Returns:
        set[str]: All canonical vocabulary IDs matched. Empty when the input
        list is empty or nothing resolves.
    """
    if not vocabulary:
        return set()

    possible_vocab_ids: set[str] = await get_vocabulary_ids()
    possible_by_upper: dict[str, str] = {v.upper(): v for v in possible_vocab_ids}

    logger.info(f"Input vocabularies {vocabulary}")
    matched_vocab_ids: set[str] = set()

    for raw in vocabulary:
        vocab = (raw or "").strip()
        if not vocab:
            continue

        if vocab in possible_vocab_ids:
            matched_vocab_ids.add(vocab)
            continue

        canonical = possible_by_upper.get(vocab.upper())
        if canonical is not None:
            matched_vocab_ids.add(canonical)
            continue

        alias_targets = VOCAB_ALIASES.get(vocab.upper())
        if alias_targets:
            added_any = False
            for target in alias_targets:
                if target in possible_vocab_ids:
                    matched_vocab_ids.add(target)
                    added_any = True
                else:
                    canonical_target = possible_by_upper.get(target.upper())
                    if canonical_target is not None:
                        matched_vocab_ids.add(canonical_target)
                        added_any = True
            if added_any:
                continue

        # Fuzzy fallback returns ALL matches above threshold; extractOne would drop equally-scoring siblings.
        fuzzy_matches = process.extract(
            vocab,
            possible_vocab_ids,
            scorer=fuzz.partial_ratio,
            score_cutoff=80,
            limit=10,
        )
        logger.info(f"fuzzy_matches for {vocab!r}: {fuzzy_matches}")
        for match_value, _score, _index in fuzzy_matches:
            matched_vocab_ids.add(match_value)

    return matched_vocab_ids
