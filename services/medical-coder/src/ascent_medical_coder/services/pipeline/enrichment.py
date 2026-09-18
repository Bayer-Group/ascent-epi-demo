"""Descendant enrichment strategies.

Faithful port of the old ``descendant_resolver.py``. Given a concepts
structure in the API shape (``dict[query, list[{CONCEPT_DATA: {...}, SCORE:
float, ...}]]``), each enricher appends descendant concepts discovered via a
particular vocabulary hierarchy. Descendants are tagged with the seed's
``ANCESTOR_CONCEPT_ID`` so downstream consumers can trace expansions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import pandas as pd

from ascent_medical_coder.connectors.warehouse import SnowflakeConnector

logger = logging.getLogger(__name__)

# Max ids per SQL IN-list chunk (Snowflake caps expression lists at ~16,384).
_SQL_IN_CHUNK = 10_000


ProgressCb = Callable[[str], Awaitable[None]]


class BaseDescendantsEnricher(ABC):
    """Base contract for descendants enrichment strategies.

    Implementations should:
      - accept and return the same concepts structure used by the API
      - be safe to call even when there are no applicable concepts
      - use `progress` for user-visible state messages when provided
      - optionally write stats into `state`
    """

    @abstractmethod
    async def enrich(
        self,
        concepts: dict[str, list[dict]],
        *,
        progress: ProgressCb | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, list[dict]]:
        raise NotImplementedError


@dataclass(frozen=True)
class ICDDescendantsEnricher(BaseDescendantsEnricher):
    """Enriches an existing concepts structure with ICD descendants.

    Contract:
      - Input: concepts in the API shape: dict[query, list[{CONCEPT_DATA: {...}, SCORE: float, ...}]]
      - Output: same shape, with additional items appended (descendants) for ICD vocabularies only.
      - Snowflake source: CONCEPT

    Notes:
      - Descendants here means: concepts in *same vocabulary* whose normalized concept_code starts
        with the ancestor's normalized concept_code (dot removed, uppercased), excluding exact match.
      - This is intentionally ICD-only to avoid exploding other vocabularies.
    """

    snowflake_database: str = "ASCENT"
    snowflake_schema: str = "PUBLIC"
    icd_vocabularies: tuple[str, ...] = ("ICD10CM", "ICD9CM", "ICD10PCS", "ICD9Proc", "ICD10")
    batch_size: int = 1000
    max_descendants_per_ancestor: int = 200
    # Global cap: high-fanout vocabs can balloon past the LLM filter's skip threshold downstream.
    max_total_descendants: int = 1000

    async def enrich(
        self,
        concepts: dict[str, list[dict]],
        *,
        progress: ProgressCb | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, list[dict]]:
        """Append ICD descendants to each query's concept list.

        Returns the same `concepts` mapping (mutated in-place) to stay fully backward compatible.
        """
        if not concepts:
            if state is not None:
                state["icd_descendants_added"] = 0
            return concepts

        icd_vocab_set = {v.upper() for v in self.icd_vocabularies}
        total_added = 0

        for query_key, concept_list in concepts.items():
            if total_added >= self.max_total_descendants:
                break
            if not concept_list:
                continue

            ancestor_ids: list[int] = []
            ancestor_domain_by_id: dict[int, str] = {}
            template_item: dict[str, Any] | None = None

            for item in concept_list:
                if not isinstance(item, dict):
                    continue
                if template_item is None:
                    template_item = item

                data = item.get("CONCEPT_DATA") or {}
                if not isinstance(data, dict):
                    continue

                vocab = (data.get("VOCABULARY_ID") or "").upper()
                if vocab not in icd_vocab_set:
                    continue

                concept_id_raw = data.get("CONCEPT_ID")
                if concept_id_raw is None:
                    continue

                try:
                    concept_id = int(concept_id_raw)
                except Exception:
                    continue

                ancestor_ids.append(concept_id)
                domain_val = data.get("DOMAIN_ID")
                if isinstance(domain_val, str) and domain_val:
                    ancestor_domain_by_id[concept_id] = domain_val

            if not ancestor_ids:
                continue

            if progress:
                await progress(
                    f"Enriching ICD concepts with descendants (query='{query_key}', n={len(ancestor_ids)})..."
                )

            descendants_df_parts: list[pd.DataFrame] = []
            db = SnowflakeConnector(database=self.snowflake_database, schema=self.snowflake_schema)

            for i in range(0, len(ancestor_ids), self.batch_size):
                batch_ids = ancestor_ids[i : i + self.batch_size]
                id_list = ",".join(str(x) for x in batch_ids)

                sql_query = f"""
WITH seeds AS (
    SELECT
        c.concept_id AS concept_id,
        c.vocabulary_id AS vocabulary_id,
        UPPER(REPLACE(c.concept_code, '.', '')) AS seed_code
    FROM CONCEPT AS c
    WHERE c.invalid_reason IS NULL
      AND c.concept_id IN ({id_list})
      AND c.vocabulary_id IN ({",".join([f"'{v}'" for v in self.icd_vocabularies])})
), descendants AS (
    SELECT
        s.concept_id AS ancestor_id,
        c.concept_id AS descendant_id,
        c.concept_code AS descendant_code,
        c.concept_name AS descendant_name,
        c.vocabulary_id AS descendant_vocabulary
    FROM seeds AS s
    JOIN CONCEPT AS c
      ON c.vocabulary_id = s.vocabulary_id
     AND c.invalid_reason IS NULL
     AND UPPER(REPLACE(c.concept_code, '.', '')) LIKE s.seed_code || '%'
     AND UPPER(REPLACE(c.concept_code, '.', '')) <> s.seed_code
    QUALIFY ROW_NUMBER() OVER (PARTITION BY s.concept_id ORDER BY c.concept_id)
        <= {int(self.max_descendants_per_ancestor)}
)
SELECT ancestor_id, descendant_id, descendant_code, descendant_name, descendant_vocabulary
FROM descendants
"""

                if progress:
                    await progress(f"Fetching ICD descendants batch {i // self.batch_size + 1}...")

                df = await db.fetch_data_with_cursor(sql_query)
                if df is not None and not df.empty:
                    descendants_df_parts.append(df)

            if not descendants_df_parts:
                continue

            descendants_df = pd.concat(descendants_df_parts, ignore_index=True)

            existing_ids: set[int] = set()
            for itm in concept_list:
                try:
                    cid = (itm.get("CONCEPT_DATA") or {}).get("CONCEPT_ID")
                    if cid is not None:
                        existing_ids.add(int(cid))
                except Exception:
                    continue

            template_item = template_item or {"CONCEPT_DATA": {}, "SCORE": 0.0}

            added_for_query = 0
            for _, row in descendants_df.iterrows():
                if total_added + added_for_query >= self.max_total_descendants:
                    break
                try:
                    desc_id = int(row.get("DESCENDANT_ID") if "DESCENDANT_ID" in row else row.get("descendant_id"))
                except Exception:
                    continue

                if desc_id in existing_ids:
                    continue

                ancestor_id_val = row.get("ANCESTOR_ID") if "ANCESTOR_ID" in row else row.get("ancestor_id")
                ancestor_domain = None
                try:
                    if ancestor_id_val is not None:
                        ancestor_domain = ancestor_domain_by_id.get(int(ancestor_id_val))
                except Exception:
                    ancestor_domain = None

                try:
                    ancestor_id_int = int(ancestor_id_val) if ancestor_id_val is not None else None
                except Exception:
                    ancestor_id_int = None

                concept_data = {
                    "CONCEPT_ID": desc_id,
                    "CONCEPT_CODE": row.get("DESCENDANT_CODE")
                    if "DESCENDANT_CODE" in row
                    else row.get("descendant_code"),
                    "CONCEPT_NAME": row.get("DESCENDANT_NAME")
                    if "DESCENDANT_NAME" in row
                    else row.get("descendant_name"),
                    "DOMAIN_ID": ancestor_domain or "",
                    "VOCABULARY_ID": row.get("DESCENDANT_VOCABULARY")
                    if "DESCENDANT_VOCABULARY" in row
                    else row.get("descendant_vocabulary"),
                    "STANDARD_CONCEPT": None,
                    "IS_VALID": True,
                    "SOURCE": "descendant_resolver",
                    "ANCESTOR_CONCEPT_ID": ancestor_id_int,
                }

                new_item = dict(template_item)
                new_item["CONCEPT_DATA"] = concept_data
                new_item["SCORE"] = 0.0

                concept_list.append(new_item)
                existing_ids.add(desc_id)
                added_for_query += 1

            if progress:
                await progress(f"Added {added_for_query} ICD descendants for query='{query_key}'")

            total_added += added_for_query

        if state is not None:
            state["icd_descendants_added"] = total_added

        return concepts


@dataclass(frozen=True)
class OmopAncestorDescendantsEnricher(BaseDescendantsEnricher):
    """Enriches concepts with descendants via the OMOP CONCEPT_ANCESTOR table.

    Targets standard vocabularies whose codes are not string-hierarchical
    (SNOMED, RxNorm, ATC, MedDRA, LOINC), where the ICD prefix trick does
    not work. ICD vocabularies are intentionally excluded — the prefix-based
    ICDDescendantsEnricher already handles them and also covers non-standard
    ICD source codes that are absent from CONCEPT_ANCESTOR.

    Source: CONCEPT_ANCESTOR joined to
    CONCEPT for descendant metadata.
    """

    snowflake_database: str = "ASCENT"
    snowflake_schema: str = "PUBLIC"
    ancestor_table: str = "CONCEPT_ANCESTOR"
    concept_table: str = "CONCEPT"
    supported_vocabularies: tuple[str, ...] = (
        "SNOMED",
        "RxNorm",
        "RxNorm Extension",
        "ATC",
        "LOINC",
        "MedDRA",
    )
    batch_size: int = 1000
    max_descendants_per_ancestor: int = 200
    max_levels_of_separation: int = 5
    # Global cap: last-resort safety net over the per-ancestor caps for pathological fanout.
    max_total_descendants: int = 1000
    # Per-vocab {max_levels, max_descendants} overrides; SNOMED/RxNorm tightened as broad parents fan out hugely.
    per_vocab_caps: Mapping[str, tuple[int, int]] = field(
        default_factory=lambda: MappingProxyType(
            {
                "SNOMED": (4, 150),
                "RXNORM": (4, 200),
                "RXNORM EXTENSION": (4, 200),
            }
        )
    )

    def _caps_for(self, vocab: str) -> tuple[int, int]:
        """Return (max_levels, max_descendants) for a vocabulary, falling back to defaults."""
        override = self.per_vocab_caps.get(vocab.upper())
        if override is not None:
            return override
        return self.max_levels_of_separation, self.max_descendants_per_ancestor

    async def enrich(
        self,
        concepts: dict[str, list[dict]],
        *,
        progress: ProgressCb | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, list[dict]]:
        if not concepts:
            if state is not None:
                state["omop_ancestor_descendants_added"] = 0
            return concepts

        supported_set = {v.upper() for v in self.supported_vocabularies}
        total_added = 0

        for query_key, concept_list in concepts.items():
            if not concept_list:
                continue

            # One query per distinct caps pair: WHERE/QUALIFY thresholds must be constants within a query.
            ancestors_by_caps: dict[tuple[int, int], list[int]] = {}
            ancestor_domain_by_id: dict[int, str] = {}
            ancestor_score_by_id: dict[int, float] = {}
            template_item: dict[str, Any] | None = None

            # Sort seeds by score so the global cap keeps descendants of the most relevant seeds.
            def _score_of(itm: Any) -> float:
                if not isinstance(itm, dict):
                    return 0.0
                try:
                    return float(itm.get("SCORE") or 0.0)
                except Exception:
                    return 0.0

            sorted_concept_list = sorted(
                (item for item in concept_list if isinstance(item, dict)),
                key=_score_of,
                reverse=True,
            )

            for item in sorted_concept_list:
                if template_item is None:
                    template_item = item

                data = item.get("CONCEPT_DATA") or {}
                if not isinstance(data, dict):
                    continue

                vocab = (data.get("VOCABULARY_ID") or "").upper()
                if vocab not in supported_set:
                    continue

                concept_id_raw = data.get("CONCEPT_ID")
                if concept_id_raw is None:
                    continue

                try:
                    concept_id = int(concept_id_raw)
                except Exception:
                    continue

                caps = self._caps_for(vocab)
                ancestors_by_caps.setdefault(caps, []).append(concept_id)
                ancestor_score_by_id[concept_id] = _score_of(item)
                domain_val = data.get("DOMAIN_ID")
                if isinstance(domain_val, str) and domain_val:
                    ancestor_domain_by_id[concept_id] = domain_val

            total_ancestors = sum(len(v) for v in ancestors_by_caps.values())
            if total_ancestors == 0:
                continue

            if progress:
                await progress(
                    f"Enriching standard OMOP concepts with descendants (query='{query_key}', n={total_ancestors})..."
                )

            descendants_df_parts: list[pd.DataFrame] = []
            db = SnowflakeConnector(
                database=self.snowflake_database,
                schema=self.snowflake_schema,
            )

            vocab_list_sql = ",".join(f"'{v}'" for v in self.supported_vocabularies)

            for (max_levels, max_descendants), ancestor_ids in ancestors_by_caps.items():
                for i in range(0, len(ancestor_ids), self.batch_size):
                    batch_ids = ancestor_ids[i : i + self.batch_size]
                    id_list = ",".join(str(x) for x in batch_ids)

                    sql_query = f"""
SELECT
    ca.ancestor_concept_id AS ancestor_id,
    c.concept_id           AS descendant_id,
    c.concept_code         AS descendant_code,
    c.concept_name         AS descendant_name,
    c.vocabulary_id        AS descendant_vocabulary,
    c.domain_id            AS descendant_domain,
    c.standard_concept     AS descendant_standard_concept
FROM {self.ancestor_table} AS ca
JOIN {self.concept_table} AS c
  ON c.concept_id = ca.descendant_concept_id
WHERE ca.ancestor_concept_id IN ({id_list})
  AND ca.min_levels_of_separation BETWEEN 1 AND {int(max_levels)}
  AND c.invalid_reason IS NULL
  AND c.vocabulary_id IN ({vocab_list_sql})
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY ca.ancestor_concept_id
    ORDER BY ca.min_levels_of_separation, c.concept_id
) <= {int(max_descendants)}
"""

                    if progress:
                        await progress(
                            f"Fetching OMOP ancestor descendants batch "
                            f"{i // self.batch_size + 1} "
                            f"(levels<={max_levels}, limit={max_descendants})..."
                        )

                    df = await db.fetch_data_with_cursor(sql_query)
                    if df is not None and not df.empty:
                        descendants_df_parts.append(df)

            if not descendants_df_parts:
                continue

            descendants_df = pd.concat(descendants_df_parts, ignore_index=True)

            # Order descendants by seed ancestor's score so the global cap preserves the most relevant rows.
            ancestor_col = "ANCESTOR_ID" if "ANCESTOR_ID" in descendants_df.columns else "ancestor_id"
            descendants_df["_ancestor_score"] = descendants_df[ancestor_col].map(
                lambda aid, _scores=ancestor_score_by_id: _scores.get(int(aid), 0.0) if pd.notna(aid) else 0.0
            )
            descendants_df = descendants_df.sort_values(
                "_ancestor_score", ascending=False, kind="mergesort"
            ).reset_index(drop=True)

            existing_ids: set[int] = set()
            for itm in concept_list:
                try:
                    cid = (itm.get("CONCEPT_DATA") or {}).get("CONCEPT_ID")
                    if cid is not None:
                        existing_ids.add(int(cid))
                except Exception:
                    continue

            template_item = template_item or {"CONCEPT_DATA": {}, "SCORE": 0.0}

            added_for_query = 0
            for _, row in descendants_df.iterrows():
                try:
                    desc_id = int(row.get("DESCENDANT_ID") if "DESCENDANT_ID" in row else row.get("descendant_id"))
                except Exception:
                    continue

                if desc_id in existing_ids:
                    continue

                descendant_domain = (
                    row.get("DESCENDANT_DOMAIN") if "DESCENDANT_DOMAIN" in row else row.get("descendant_domain")
                )
                if not (isinstance(descendant_domain, str) and descendant_domain):
                    ancestor_id_val = row.get("ANCESTOR_ID") if "ANCESTOR_ID" in row else row.get("ancestor_id")
                    try:
                        if ancestor_id_val is not None:
                            descendant_domain = ancestor_domain_by_id.get(int(ancestor_id_val), "")
                    except Exception:
                        descendant_domain = ""

                ancestor_id_val_for_data = row.get("ANCESTOR_ID") if "ANCESTOR_ID" in row else row.get("ancestor_id")
                try:
                    ancestor_id_int = int(ancestor_id_val_for_data) if ancestor_id_val_for_data is not None else None
                except Exception:
                    ancestor_id_int = None

                concept_data = {
                    "CONCEPT_ID": desc_id,
                    "CONCEPT_CODE": row.get("DESCENDANT_CODE")
                    if "DESCENDANT_CODE" in row
                    else row.get("descendant_code"),
                    "CONCEPT_NAME": row.get("DESCENDANT_NAME")
                    if "DESCENDANT_NAME" in row
                    else row.get("descendant_name"),
                    "DOMAIN_ID": descendant_domain or "",
                    "VOCABULARY_ID": row.get("DESCENDANT_VOCABULARY")
                    if "DESCENDANT_VOCABULARY" in row
                    else row.get("descendant_vocabulary"),
                    "STANDARD_CONCEPT": row.get("DESCENDANT_STANDARD_CONCEPT")
                    if "DESCENDANT_STANDARD_CONCEPT" in row
                    else row.get("descendant_standard_concept"),
                    "IS_VALID": True,
                    "SOURCE": "omop_ancestor_resolver",
                    "ANCESTOR_CONCEPT_ID": ancestor_id_int,
                }

                new_item = dict(template_item)
                new_item["CONCEPT_DATA"] = concept_data
                new_item["SCORE"] = 0.0

                concept_list.append(new_item)
                existing_ids.add(desc_id)
                added_for_query += 1
                total_added += 1

                if total_added >= self.max_total_descendants:
                    break

            if progress:
                await progress(f"Added {added_for_query} OMOP ancestor descendants for query='{query_key}'")

            if total_added >= self.max_total_descendants:
                if progress:
                    await progress(
                        f"Global descendant cap reached ({self.max_total_descendants}); stopping enrichment."
                    )
                break

        if state is not None:
            state["omop_ancestor_descendants_added"] = total_added
            state["omop_ancestor_global_cap_hit"] = total_added >= self.max_total_descendants

        return concepts


@dataclass(frozen=True)
class NDCDescendantsEnricher(BaseDescendantsEnricher):
    """Enriches NDC seeds with all 11-digit NDC siblings via RxNorm bridging.

    NDC concepts are non-standard and have no useful CONCEPT_ANCESTOR rows of
    their own, so the standard OmopAncestorDescendantsEnricher cannot expand
    them. This enricher closes the gap by traversing:

        NDC seed
          --(Maps to)-->            RxNorm standard concept
          --(CONCEPT_ANCESTOR)-->   RxNorm descendants (self + branded variants)
          --(Mapped from)-->        all NDC siblings
          filter concept_class_id = '11-digit NDC'

    Why 11-digit only:
      Claims tables commonly store NDC as 11-digit zero-padded.
      9-digit NDC concepts exist in OMOP but never appear in claims data, so
      emitting them produces silent zero-match misses.

    Why this matters:
      A single Clinical Drug typically has hundreds of 11-digit NDC siblings
      across packagers. Vector search over raw NDC concept names only surfaces
      a tiny fraction; without this expansion non-OMOP drug cohorts under-count
      patients by an order of magnitude relative to OMOP (CONCEPT_ANCESTOR-based)
      pipelines.
    """

    snowflake_database: str = "ASCENT"
    snowflake_schema: str = "PUBLIC"
    ancestor_table: str = "CONCEPT_ANCESTOR"
    relationship_table: str = "CONCEPT_RELATIONSHIP"
    concept_table: str = "CONCEPT"
    target_concept_class_id: str = "11-digit NDC"
    max_descendants_per_ancestor: int = 200
    max_levels_of_separation: int = 4
    # Global cap: NDC fanout is heavier than RxNorm (hundreds of packaged variants per Clinical Drug).
    max_total_descendants: int = 1000

    async def enrich(
        self,
        concepts: dict[str, list[dict]],
        *,
        progress: ProgressCb | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, list[dict]]:
        if not concepts:
            if state is not None:
                state["ndc_descendants_added"] = 0
            return concepts

        total_added = 0

        for query_key, concept_list in concepts.items():
            if total_added >= self.max_total_descendants:
                break
            if not concept_list:
                continue

            ancestor_ids: list[int] = []
            ancestor_domain_by_id: dict[int, str] = {}
            ancestor_score_by_id: dict[int, float] = {}
            template_item: dict[str, Any] | None = None

            def _score_of(itm: Any) -> float:
                if not isinstance(itm, dict):
                    return 0.0
                try:
                    return float(itm.get("SCORE") or 0.0)
                except Exception:
                    return 0.0

            sorted_concept_list = sorted(
                (item for item in concept_list if isinstance(item, dict)),
                key=_score_of,
                reverse=True,
            )

            for item in sorted_concept_list:
                if template_item is None:
                    template_item = item

                data = item.get("CONCEPT_DATA") or {}
                if not isinstance(data, dict):
                    continue

                vocab = (data.get("VOCABULARY_ID") or "").upper()
                if vocab != "NDC":
                    continue

                concept_id_raw = data.get("CONCEPT_ID")
                if concept_id_raw is None:
                    continue

                try:
                    concept_id = int(concept_id_raw)
                except Exception:
                    continue

                ancestor_ids.append(concept_id)
                ancestor_score_by_id[concept_id] = _score_of(item)
                domain_val = data.get("DOMAIN_ID")
                if isinstance(domain_val, str) and domain_val:
                    ancestor_domain_by_id[concept_id] = domain_val

            if not ancestor_ids:
                continue

            if progress:
                await progress(
                    f"Enriching NDC concepts with descendants (query='{query_key}', n={len(ancestor_ids)})..."
                )

            db = SnowflakeConnector(
                database=self.snowflake_database,
                schema=self.snowflake_schema,
            )

            # Batch partitioning doesn't change output: seeds are score-sorted,
            # so skipping post-cap batches is output-identical.
            existing_ids: set[int] = set()
            for itm in concept_list:
                try:
                    cid = (itm.get("CONCEPT_DATA") or {}).get("CONCEPT_ID")
                    if cid is not None:
                        existing_ids.add(int(cid))
                except Exception:
                    continue

            template_item = template_item or {"CONCEPT_DATA": {}, "SCORE": 0.0}
            added_for_query = 0

            # Fetch siblings once per DISTINCT Clinical Drug, not per seed; per-seed ranking/caps reconstructed
            # client-side with the same rules the old SQL applied.
            rx_by_seed: dict[int, list[int]] = {}
            rx_order: dict[int, int] = {}
            for start in range(0, len(ancestor_ids), _SQL_IN_CHUNK):
                chunk = ancestor_ids[start : start + _SQL_IN_CHUNK]
                map_df = await db.fetch_data_with_cursor(
                    f"""
SELECT cr.concept_id_1 AS ancestor_id, cr.concept_id_2 AS rx_id
FROM {self.relationship_table} AS cr
WHERE cr.concept_id_1 IN ({",".join(str(x) for x in chunk)})
  AND cr.relationship_id = 'Maps to'
  AND cr.invalid_reason IS NULL
"""
                )
                if map_df is None or map_df.empty:
                    continue
                for anc, rx in zip(map_df["ANCESTOR_ID"].tolist(), map_df["RX_ID"].tolist(), strict=False):
                    try:
                        anc_i, rx_i = int(anc), int(rx)
                    except Exception:
                        continue
                    rx_by_seed.setdefault(anc_i, []).append(rx_i)
                    rx_order.setdefault(rx_i, len(rx_order))

            if not rx_order:
                if progress:
                    await progress(f"Added 0 NDC descendants for query='{query_key}'")
                continue

            if progress:
                await progress(
                    f"Fetching NDC siblings for {len(rx_order)} distinct drugs "
                    f"(levels<={self.max_levels_of_separation}, "
                    f"limit={self.max_descendants_per_ancestor})..."
                )

            # Per-rx QUALIFY cap is a safe superset of the per-seed cap (seed top-N ⊆ drug top-N by concept_id).
            siblings_by_rx: dict[int, list[tuple]] = {}
            rx_ids = list(rx_order)
            for start in range(0, len(rx_ids), _SQL_IN_CHUNK):
                chunk = rx_ids[start : start + _SQL_IN_CHUNK]
                sib_df = await db.fetch_data_with_cursor(
                    f"""
WITH rx_descendants AS (
    SELECT ca.ancestor_concept_id AS rx_id,
           ca.descendant_concept_id AS rx_desc_id
    FROM {self.ancestor_table} ca
    WHERE ca.ancestor_concept_id IN ({",".join(str(x) for x in chunk)})
      AND ca.min_levels_of_separation
          BETWEEN 0 AND {int(self.max_levels_of_separation)}
)
SELECT
    rd.rx_id                  AS rx_id,
    c.concept_id              AS descendant_id,
    c.concept_code            AS descendant_code,
    c.concept_name            AS descendant_name,
    c.vocabulary_id           AS descendant_vocabulary,
    c.domain_id               AS descendant_domain,
    c.standard_concept        AS descendant_standard_concept
FROM rx_descendants rd
JOIN {self.relationship_table} cr2
  ON cr2.concept_id_1 = rd.rx_desc_id
 AND cr2.relationship_id = 'Mapped from'
 AND cr2.invalid_reason IS NULL
JOIN {self.concept_table} c
  ON c.concept_id = cr2.concept_id_2
 AND c.vocabulary_id = 'NDC'
 AND c.concept_class_id = '{self.target_concept_class_id}'
 AND c.invalid_reason IS NULL
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY rd.rx_id
    ORDER BY c.concept_id
) <= {int(self.max_descendants_per_ancestor)}
"""
                )
                if sib_df is None or sib_df.empty:
                    continue
                for r in sib_df.itertuples(index=False):
                    try:
                        rx_i = int(r.RX_ID)
                        desc_id = int(r.DESCENDANT_ID)
                    except Exception:
                        continue
                    siblings_by_rx.setdefault(rx_i, []).append(
                        (
                            desc_id,
                            r.DESCENDANT_CODE,
                            r.DESCENDANT_NAME,
                            r.DESCENDANT_VOCABULARY,
                            r.DESCENDANT_DOMAIN,
                            r.DESCENDANT_STANDARD_CONCEPT,
                        )
                    )
            for rows in siblings_by_rx.values():
                rows.sort(key=lambda t: t[0])  # rank by concept_id, as the old SQL did

            for seed_id in ancestor_ids:
                if total_added >= self.max_total_descendants:
                    break

                seed_rx = rx_by_seed.get(seed_id)
                if not seed_rx:
                    continue
                if len(seed_rx) == 1:
                    candidate_rows = siblings_by_rx.get(seed_rx[0], [])
                else:
                    candidate_rows = sorted(
                        (row for rx in seed_rx for row in siblings_by_rx.get(rx, [])),
                        key=lambda t: t[0],
                    )
                # Per-ancestor cap counts ranked slots incl. rows later skipped as duplicates (matching legacy SQL).
                candidate_rows = candidate_rows[: int(self.max_descendants_per_ancestor)]

                for desc_id, code, name, vocab_id, domain, standard in candidate_rows:
                    if desc_id in existing_ids:
                        continue

                    descendant_domain = domain
                    if not (isinstance(descendant_domain, str) and descendant_domain):
                        descendant_domain = ancestor_domain_by_id.get(seed_id, "")

                    concept_data = {
                        "CONCEPT_ID": desc_id,
                        "CONCEPT_CODE": code,
                        "CONCEPT_NAME": name,
                        "DOMAIN_ID": descendant_domain or "",
                        "VOCABULARY_ID": vocab_id,
                        "STANDARD_CONCEPT": standard,
                        "IS_VALID": True,
                        "SOURCE": "ndc_descendants_resolver",
                        "ANCESTOR_CONCEPT_ID": seed_id,
                    }

                    new_item = dict(template_item)
                    new_item["CONCEPT_DATA"] = concept_data
                    new_item["SCORE"] = 0.0

                    concept_list.append(new_item)
                    existing_ids.add(desc_id)
                    added_for_query += 1
                    total_added += 1

                    if total_added >= self.max_total_descendants:
                        break

            if progress:
                await progress(f"Added {added_for_query} NDC descendants for query='{query_key}'")

            if total_added >= self.max_total_descendants:
                if progress:
                    await progress(
                        f"Global NDC descendant cap reached ({self.max_total_descendants}); stopping enrichment."
                    )
                break

        if state is not None:
            state["ndc_descendants_added"] = total_added
            state["ndc_global_cap_hit"] = total_added >= self.max_total_descendants

        return concepts


@dataclass(frozen=True)
class ChainedDescendantsEnricher(BaseDescendantsEnricher):
    """Runs multiple enrichers in order, accumulating state.

    Use this to combine the ICD prefix enricher with the OMOP ancestor
    enricher so callers can keep a single integration point.
    """

    enrichers: tuple[BaseDescendantsEnricher, ...] = ()

    async def enrich(
        self,
        concepts: dict[str, list[dict]],
        *,
        progress: ProgressCb | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, list[dict]]:
        for enricher in self.enrichers:
            concepts = await enricher.enrich(concepts, progress=progress, state=state)
        return concepts
