"""Medical-coding pipeline orchestration.

Business logic ported near-verbatim from the legacy ``api/api/medical_coding.py``.
FastAPI transport (routing, dependencies, WebSocket) lives in
``api/routers/coding.py``; this module is framework-free. Functions take plain
arguments (the request model, the authenticated user, and an optional progress
callback used by the streaming/WebSocket path).

Pipeline order, thresholds, cache-key inputs, descendant-enrichment gating,
google-search fallback gating, patient-count gating, LTS gating, progress
message strings, and the final response structure are preserved exactly.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import pandas as pd
from fastapi import HTTPException
from tenacity import retry, stop_after_attempt, wait_fixed

from ascent_medical_coder.connectors.qdrant import QdrantConnector
from ascent_medical_coder.core.settings import get_settings
from ascent_medical_coder.db.analytics import log_request
from ascent_medical_coder.db.cache import cache_key_for_request, get_cached, set_cached
from ascent_medical_coder.schemas.coding import MedicalCodingRequest, ScoredConcept
from ascent_medical_coder.schemas.user import User
from ascent_medical_coder.services.pipeline.counts import calculate_patient_counts
from ascent_medical_coder.services.pipeline.drug_expand import expand_drug_class_names
from ascent_medical_coder.services.pipeline.encoder import determine_cosine_cutoff, get_embedder
from ascent_medical_coder.services.pipeline.enrichment import (
    ChainedDescendantsEnricher,
    ICDDescendantsEnricher,
    NDCDescendantsEnricher,
    OmopAncestorDescendantsEnricher,
)
from ascent_medical_coder.services.pipeline.fallback import LLMSearchFallback
from ascent_medical_coder.services.pipeline.filtering import (
    LLM_FILTER_MAX_RESULTS,
    LLMFilterResult,
    apply_llm_filter,
)
from ascent_medical_coder.services.pipeline.utils import (
    _convert_concepts_to_models,
    _dataframe_to_concepts,
    parse_db_name_and_schema,
)
from ascent_medical_coder.services.request_validation import (
    DRUG_DOMAINS,
    normalize_domain_id,
    validate_mixed_domains,
)

logger = logging.getLogger(__name__)

PG_CACHE_TTL_SECONDS = 86400

ProgressCallback = Callable[[str], Awaitable[Any]]

_qdrant = QdrantConnector()

# Built lazily so a missing Gemini key becomes a per-request error, not a boot failure.
_llm_fallback: LLMSearchFallback | None = None


def _get_llm_fallback() -> LLMSearchFallback:
    global _llm_fallback
    if _llm_fallback is None:
        _llm_fallback = LLMSearchFallback()
    return _llm_fallback


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def _encode_with_retry(generator: Any, text: str) -> list[float]:
    """Encode the query, retrying transient embedding failures (3x, 2s)."""
    return await generator.encode(text)


icd_descendants_enricher = ICDDescendantsEnricher()
omop_ancestor_descendants_enricher = OmopAncestorDescendantsEnricher()
ndc_descendants_enricher = NDCDescendantsEnricher()
descendants_enricher = ChainedDescendantsEnricher(
    enrichers=(
        icd_descendants_enricher,
        omop_ancestor_descendants_enricher,
        ndc_descendants_enricher,
    ),
)

# Higher NDC cap: drug/drug_class queries skip the LLM filter, so larger fan-out is safe.
drug_ndc_descendants_enricher = NDCDescendantsEnricher(
    max_descendants_per_ancestor=1500,
    max_total_descendants=6000,
)
drug_descendants_enricher = ChainedDescendantsEnricher(
    enrichers=(
        icd_descendants_enricher,
        omop_ancestor_descendants_enricher,
        drug_ndc_descendants_enricher,
    ),
)

_EMPTY_REASONING_META = {
    "filter_used": False,
    "lts_used": False,
    "reasons_include": None,
    "reasons_exclude": None,
}


def _is_missing_collection(err: BaseException) -> bool:
    """Whether *err* was ultimately caused by an absent Qdrant collection.

    The search is wrapped in tenacity, so what surfaces is a RetryError whose
    own str() is just ``RetryError[<Future ...>]`` -- the Qdrant 404 is only
    visible on the cause chain, and on ``last_attempt`` for the retry itself.
    """
    seen: set[int] = set()
    stack: list[BaseException | None] = [err]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if "doesn't exist" in str(current) or "Not found" in str(current):
            return True
        stack.extend([current.__cause__, current.__context__])
        last_attempt = getattr(current, "last_attempt", None)
        if last_attempt is not None:
            with contextlib.suppress(Exception):
                stack.append(last_attempt.exception())
    return False


def prepare_request(payload: MedicalCodingRequest) -> tuple[list[str | None], dict | None]:
    """Shared payload normalisation: default llm_filter and domain validation.

    Mutates ``payload.llm_filter`` (``"default"`` -> ``settings.llm.default_llm_filter``)
    and returns the normalised domain list plus an error-detail dict (``None`` when
    valid). The caller decides how to surface the error (HTTP 400 vs WebSocket
    error frame).
    """
    if payload.llm_filter == "default":
        payload.llm_filter = get_settings().llm.default_llm_filter

    normalized = [normalize_domain_id(domain) for domain in payload.domain_ids]
    error_detail = validate_mixed_domains(normalized)
    return normalized, error_detail


def resolve_context(
    payload: MedicalCodingRequest,
) -> tuple[str | None, Any, float | None, str | None, str | None]:
    """Resolve the per-request search context.

    Returns ``(collection, generator, cosine_cutoff, db, schema)``.
    """
    collection, generator = get_embedder(payload.encoder)
    cosine_cutoff = determine_cosine_cutoff(payload.encoder, payload.cosine_similarity)
    logger.info(f"Resolved collection={collection}, generator={generator}")

    db = schema = None
    if payload.database:
        db, schema = parse_db_name_and_schema(payload.database, None)
    return collection, generator, cosine_cutoff, db, schema


async def _run_drug_search_core(payload: MedicalCodingRequest, domain_id: str) -> tuple[pd.DataFrame, dict]:
    """Shared core for drug/drug_class retrieval (keyword + fallback).

    Returns (df, reasoning_meta_stub).
    """
    semaphore = asyncio.Semaphore(5)

    async def safe_search(drug_name: str):
        async with semaphore:
            return await _qdrant.search_drugs(
                keyword=drug_name,
                top_k=payload.top_k,
                domain_id="Drug",
                vocabulary=payload.vocabulary,
                standard_concept=payload.standard_concept,
                source=payload.source,
            )

    logger.info("Performing keyword-only drug search")
    logger.info(domain_id)

    if domain_id.lower() == "drug_class":
        drug_names = await expand_drug_class_names(payload.query)
        logger.info(f"LLM returned drug names: {drug_names}")
        df = pd.DataFrame()
        if drug_names:
            tasks = [safe_search(name) for name in drug_names]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            dfs = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]
            if dfs:
                df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    else:
        df = await _qdrant.search_drugs(
            keyword=payload.query,
            top_k=payload.top_k,
            domain_id=domain_id,
            vocabulary=payload.vocabulary,
            standard_concept=payload.standard_concept,
            source=payload.source,
        )

    if df is not None and df.empty and payload.google_search_fallback:
        logger.info(f"No results from keyword search for '{payload.query}', falling back to LLM web drug search")
        fallback_df = await _get_llm_fallback().search_drugs(
            search_text=payload.query,
            is_drug_class=(domain_id.lower() == "drug_class"),
            vocabulary=payload.vocabulary,
            standard_concept=payload.standard_concept,
            top_k=payload.top_k,
            custom_instructions=payload.custom_instructions,
        )
        if fallback_df is not None and not fallback_df.empty:
            df = fallback_df
            logger.info(f"LLM fallback returned {len(df)} results")
        else:
            logger.info("LLM fallback returned NO results")

    return df, dict(_EMPTY_REASONING_META)


async def _handle_drug_search(
    payload: MedicalCodingRequest,
    domain_id: str,
) -> tuple[dict[str, list[dict]], dict]:
    """Unified drug search returning (concepts_dict, reasoning_meta)."""
    df, reasoning_meta = await _run_drug_search_core(payload, domain_id)

    if (df is None or df.empty) and not payload.google_search_fallback:
        return {payload.query: []}, reasoning_meta

    concepts = _dataframe_to_concepts(df, payload.query)
    # Expand NDC seeds to all 11-digit siblings; without it non-OMOP drug cohorts severely under-count.
    if payload.include_descendants:
        concepts = await drug_descendants_enricher.enrich(concepts)
    return concepts, reasoning_meta


async def _handle_general_search(
    payload: MedicalCodingRequest,
    collection,
    generator,
    domain_id,
    cosine_cutoff,
    include_reasoning: bool = False,
    send_progress: ProgressCallback | None = None,
) -> tuple[bool, dict[str, list[dict]], dict]:
    """Unified general/LTS search returning (lts_hit, concepts_dict, reasoning_meta).

    When include_reasoning is True, ICD descendants enrichment and detailed
    reasoning metadata (counts, reasons) are included.
    """

    async def progress(message: str):
        if send_progress:
            await send_progress(message)

    await progress(f"Performing hybrid search in Qdrant (collection: {collection})...")

    # Retry transient encode failures (Azure/Gemini blips) here since the connector split moved it out of hybrid-search.
    dense = await _encode_with_retry(generator, payload.query)
    try:
        hits = await _qdrant.search_qdrant_hybrid(
            search_string=payload.query,
            collection=collection,
            cosine_cutoff=cosine_cutoff,
            dense=dense,
            top_k=payload.top_k,
            vocabulary=payload.vocabulary,
            standard_concept=payload.standard_concept,
            domain_id=domain_id,
            source=payload.source,
            use_hybrid=payload.use_hybrid,
        )
    except Exception as err:
        # A missing collection retried three times and surfaced as a bare 500,
        # which reads like the service is broken rather than like the chosen
        # encoder has no index. Only "bge" is seeded by scripts/seed_qdrant.py.
        if _is_missing_collection(err):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Qdrant collection {collection!r} does not exist, so encoder "
                    f"{payload.encoder!r} cannot be used. This stack seeds only the 'bge' "
                    f"index (scripts/seed_qdrant.py); pass encoder='bge', or build the "
                    f"collection for another encoder first."
                ),
            ) from err
        raise

    qdrant_initial_count = len(hits)
    post_descendants_total = qdrant_initial_count

    await progress(f"Vector DB search completed. Found {len(hits)} initial results.")

    # Runs regardless of include_reasoning: non-reasoning callers depend on descendants to resolve parent codes.
    if not hits.empty and payload.include_descendants:
        await progress("Enriching results with descendants...")
        concepts = _dataframe_to_concepts(hits, payload.query)
        enriched_concepts = await descendants_enricher.enrich(concepts, progress=progress)

        enriched_rows = []
        for concept_dict in enriched_concepts.get(payload.query, []):
            concept_data = concept_dict.get("CONCEPT_DATA", {})
            concept_data["SCORE"] = concept_dict.get("SCORE", 0.0)
            enriched_rows.append(concept_data)

        if enriched_rows:
            hits = pd.DataFrame(enriched_rows)
            hits = hits.where(pd.notna(hits), None)
            post_descendants_total = len(hits)
            await progress(f"Enrichment complete. Total results including descendants: {len(hits)}")

    # Trim descendants (lowest-parent-score first) past the filter cap so the filter runs; originals preserved.
    descendant_sources = {
        "descendant_resolver",
        "omop_ancestor_resolver",
        "ndc_descendants_resolver",
    }
    if payload.llm_filter is not None and len(hits) > LLM_FILTER_MAX_RESULTS and "SOURCE" in hits.columns:
        descendant_mask = hits["SOURCE"].isin(descendant_sources)
        descendant_count = int(descendant_mask.sum())
        if descendant_count > 0:
            overflow = len(hits) - LLM_FILTER_MAX_RESULTS
            drop_count = min(overflow, descendant_count)

            originals = hits.loc[~descendant_mask, ["CONCEPT_ID", "SCORE"]]
            parent_score: dict[int, float] = {}
            for cid, score in zip(originals["CONCEPT_ID"], originals["SCORE"], strict=False):
                try:
                    parent_score[int(cid)] = float(score) if score is not None else 0.0
                except Exception:
                    continue

            def _ancestor_score(aid: Any) -> float:
                try:
                    return parent_score.get(int(aid), 0.0) if aid is not None else 0.0
                except Exception:
                    return 0.0

            ancestor_col = (
                hits.loc[descendant_mask, "ANCESTOR_CONCEPT_ID"]
                if "ANCESTOR_CONCEPT_ID" in hits.columns
                else pd.Series([None] * descendant_count, index=hits.index[descendant_mask])
            )
            ranked = pd.DataFrame({"_score": ancestor_col.map(_ancestor_score)}).sort_values(
                "_score", ascending=True, kind="mergesort"
            )
            indices_to_drop = ranked.index[:drop_count].tolist()
            hits = hits.drop(indices_to_drop).reset_index(drop=True)
            post_descendants_total = len(hits)
            await progress(
                f"Trimmed {drop_count} descendants (lowest-parent-score first) "
                f"to fit LLM filter cap ({LLM_FILTER_MAX_RESULTS}). "
                f"Total now: {post_descendants_total}"
            )

    reasoning_meta: dict[str, Any] = dict(_EMPTY_REASONING_META)
    if include_reasoning:
        reasoning_meta["counts"] = {
            "qdrant_initial": qdrant_initial_count,
            "descendants_added": max(0, post_descendants_total - qdrant_initial_count),
            "llm_removed": 0,
            "final_total": post_descendants_total,
        }

    lts_hit = False
    if payload.llm_filter is not None and not hits.empty:
        await progress(f"Applying LLM filter ({payload.llm_filter}) to {len(hits)} results...")
        try:
            pre_filter_total = len(hits)
            filter_result: LLMFilterResult = await apply_llm_filter(
                payload.query,
                hits,
                payload.llm_filter,
                allow_lts=payload.allow_lts,
                include_reasons=include_reasoning,
                custom_instructions=payload.custom_instructions,
                send_progress=send_progress,
            )
            lts_hit = filter_result.lts_hit
            filtered = filter_result.filtered_df

            if include_reasoning:
                final_total = len(filtered)
                reasoning_meta["counts"]["llm_removed"] = max(0, pre_filter_total - final_total)
                reasoning_meta["counts"]["final_total"] = final_total
                reasoning_meta["filter_used"] = True
                reasoning_meta["lts_used"] = bool(lts_hit)
                reasoning_meta["reasons_include"] = filter_result.reasons_include
                reasoning_meta["reasons_exclude"] = filter_result.reasons_exclude

            await progress(f"LLM filter complete. {len(filtered)} results after filtering. LTS used: {lts_hit}")
            return lts_hit, _dataframe_to_concepts(filtered, payload.query), reasoning_meta
        except Exception as e:
            # Do not turn failed validation into successful, cacheable medical
            # codes. REST returns an explicit error; WebSocket reports it too.
            logger.exception("LLM filter failed")
            raise HTTPException(
                status_code=502,
                detail="Medical concept filtering failed. Retry the request; no unfiltered result was returned.",
            ) from e

    logger.info(f"Qdrant hybrid search returned {len(hits)} results")

    if hits.empty and payload.google_search_fallback:
        logger.info(
            f"No results from traditional search for query: {payload.query}, using LLM fallback with Google Search"
        )
        await progress("No results found. Trying LLM fallback with Google Search...")

        hits = await _get_llm_fallback().search_medical_concepts(
            search_text=payload.query,
            domain_id=domain_id,
            vocabulary=payload.vocabulary,
            standard_concept=payload.standard_concept,
            top_k=payload.top_k,
            custom_instructions=payload.custom_instructions,
        )

        lts_hit = False

        if not hits.empty:
            logger.info(f"LLM fallback returned {len(hits)} results")
            await progress(f"LLM fallback returned {len(hits)} results.")

            concepts = _dataframe_to_concepts(hits, payload.query)
            if payload.include_descendants:
                await progress("Enriching fallback results with descendants...")
                enriched_concepts = await descendants_enricher.enrich(concepts, progress=progress)
            else:
                enriched_concepts = concepts

            if include_reasoning:
                fallback_total = len(hits)
                enriched_total = len(enriched_concepts.get(payload.query, []))
                reasoning_meta["counts"]["final_total"] = enriched_total
                reasoning_meta["counts"]["descendants_added"] = max(0, enriched_total - fallback_total)
                reasoning_meta["counts"]["llm_removed"] = 0
            return lts_hit, enriched_concepts, reasoning_meta
        else:
            await progress("No results found even with LLM fallback.")
            if include_reasoning:
                reasoning_meta["counts"]["final_total"] = 0
                reasoning_meta["counts"]["descendants_added"] = 0
                reasoning_meta["counts"]["llm_removed"] = 0
            return lts_hit, {payload.query: []}, reasoning_meta

    elif hits.empty and not payload.google_search_fallback:
        await progress("No results found for query.")
        if include_reasoning:
            reasoning_meta["counts"]["final_total"] = 0
            reasoning_meta["counts"]["descendants_added"] = 0
            reasoning_meta["counts"]["llm_removed"] = 0
        return lts_hit, {payload.query: []}, reasoning_meta

    await progress(f"Returning {len(hits)} results.")
    if include_reasoning:
        reasoning_meta["counts"]["final_total"] = len(hits)
        reasoning_meta["counts"]["llm_removed"] = 0

    return lts_hit, _dataframe_to_concepts(hits, payload.query), reasoning_meta


async def run_for_domain(
    domain_id: str | None,
    *,
    payload: MedicalCodingRequest,
    collection,
    generator,
    cosine_cutoff,
    db: str | None,
    schema: str | None,
    current_user: User,
    start_time: float,
    include_reasoning: bool = False,
    send_progress: ProgressCallback | None = None,
) -> tuple[bool, dict[str, list[dict]], dict | None]:
    """Unified domain runner for backend, frontend, and WebSocket paths.

    Returns (lts_flag, concepts_dict, reasoning_meta_or_None).
    """
    d = normalize_domain_id(domain_id)

    if send_progress:
        await send_progress(f"Processing domain: {d or 'general'}...")

    logger.info(f"Running search for domain_id={d}")

    key = cache_key_for_request(payload, d)

    # Backend always caches; frontend/ws only when use_lts
    use_cache = not include_reasoning or payload.use_lts

    if use_cache:
        try:
            cached_val = await get_cached(key)
        except Exception as e:
            logger.warning(f"pg cache read failed: {e}")
            cached_val = None
        if cached_val is not None:
            logger.info("pg cache HIT")
            if send_progress:
                await send_progress(f"Cache hit for domain: {d or 'general'}")
            reasoning_meta = (
                {"filter_used": False, "lts_used": False, "raw_response": None} if include_reasoning else None
            )
            return False, cached_val, reasoning_meta

    reasoning_meta: dict | None = (
        {"filter_used": False, "lts_used": False, "raw_response": None} if include_reasoning else None
    )

    if d in DRUG_DOMAINS:
        if send_progress:
            await send_progress(f"Searching drugs for: {payload.query}...")
        concepts, domain_reasoning = await _handle_drug_search(payload, domain_id)
        lts_flag = False
        if include_reasoning and domain_reasoning is not None:
            reasoning_meta.update(domain_reasoning)
    else:
        lts_flag, concepts, domain_reasoning = await _handle_general_search(
            payload,
            collection,
            generator,
            domain_id,
            cosine_cutoff,
            include_reasoning=include_reasoning,
            send_progress=send_progress,
        )
        if include_reasoning and domain_reasoning is not None:
            reasoning_meta.update(domain_reasoning)

    if db and d is not None and concepts:
        if send_progress:
            await send_progress("Calculating patient counts...")
        concepts = await calculate_patient_counts(
            concepts=concepts,
            database=db,
            schema=schema,
            domain_id=d,
            standard_concept=payload.standard_concept,
        )

    try:
        await log_request(
            user_id=current_user.email,
            payload=payload,
            exec_time=time.time() - start_time,
            domain_id=domain_id,
            lts_used=lts_flag,
        )
    except Exception as e:
        # Analytics must never fail a request that already produced results.
        logger.warning(f"user-analytics write failed: {e}")

    if use_cache:
        try:
            await set_cached(key, concepts, PG_CACHE_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"pg cache write failed: {e}")

    return lts_flag, concepts, reasoning_meta


def aggregate_domain_results(
    unique_domains: list[str | None],
    results: list[tuple],
) -> tuple[bool, dict[str, list[dict]], dict[str | None, dict]]:
    """Merge multi-domain results into (any_lts_used, all_concepts, llm_reasoning)."""
    any_lts_used = False
    all_concepts: dict[str, list[dict]] = {}
    llm_reasoning: dict[str | None, dict] = {}

    for domain_id, result in zip(unique_domains, results, strict=False):
        lts_flag = result[0]
        concepts = result[1]
        reasoning_meta = result[2] if len(result) > 2 else None

        any_lts_used = any_lts_used or lts_flag
        if concepts:
            for key, val in concepts.items():
                if key not in all_concepts:
                    all_concepts[key] = val
                else:
                    all_concepts[key].extend(val)
        if reasoning_meta is not None:
            llm_reasoning[domain_id] = reasoning_meta

    return any_lts_used, all_concepts, llm_reasoning


async def run_domains_parallel(
    payload: MedicalCodingRequest,
    current_user: User,
    normalized: list[str | None],
    start_time: float,
    *,
    include_reasoning: bool,
) -> tuple[bool, dict[str, list[ScoredConcept]], dict[str | None, dict]]:
    """Fan out unique domains concurrently (REST paths).

    Returns ``(any_lts_used, converted_results, llm_reasoning)``.
    """
    # Offload: resolve_context may load a SentenceTransformer encoder, which must not block the event loop.
    collection, generator, cosine_cutoff, db, schema = await asyncio.to_thread(resolve_context, payload)

    domain_kwargs = {
        "payload": payload,
        "collection": collection,
        "generator": generator,
        "cosine_cutoff": cosine_cutoff,
        "db": db,
        "schema": schema,
        "current_user": current_user,
        "start_time": start_time,
        "include_reasoning": include_reasoning,
    }

    unique_domains = list(dict.fromkeys(normalized))
    if not unique_domains:
        lts_flag, concepts, reasoning_meta = await run_for_domain(None, **domain_kwargs)
        converted = _convert_concepts_to_models(concepts)
        llm_reasoning = {None: reasoning_meta} if include_reasoning else {}
        return lts_flag, converted, llm_reasoning

    tasks = [asyncio.create_task(run_for_domain(d, **domain_kwargs)) for d in unique_domains]
    results = await asyncio.gather(*tasks)

    any_lts_used, all_concepts, llm_reasoning = aggregate_domain_results(unique_domains, results)
    return any_lts_used, _convert_concepts_to_models(all_concepts), llm_reasoning


async def run_domains_sequential(
    payload: MedicalCodingRequest,
    current_user: User,
    normalized: list[str | None],
    start_time: float,
    *,
    send_progress: ProgressCallback,
) -> tuple[bool, dict[str, list[ScoredConcept]], dict[str | None, dict]]:
    """Process unique domains sequentially with progress updates (WebSocket path).

    Always includes reasoning metadata. Returns
    ``(any_lts_used, converted_results, llm_reasoning)``.
    """
    await send_progress("Resolving collection and generator...")

    # Offload: resolve_context may load a SentenceTransformer encoder, which must not block the event loop.
    collection, generator, cosine_cutoff, db, schema = await asyncio.to_thread(resolve_context, payload)

    domain_kwargs = {
        "payload": payload,
        "collection": collection,
        "generator": generator,
        "cosine_cutoff": cosine_cutoff,
        "db": db,
        "schema": schema,
        "current_user": current_user,
        "start_time": start_time,
        "include_reasoning": True,
        "send_progress": send_progress,
    }

    unique_domains = list(dict.fromkeys(normalized))

    if unique_domains:
        results = []
        for d in unique_domains:
            results.append(await run_for_domain(d, **domain_kwargs))
    else:
        result = await run_for_domain(None, **domain_kwargs)
        unique_domains = [None]
        results = [result]

    any_lts_used, all_concepts, llm_reasoning = aggregate_domain_results(unique_domains, results)

    await send_progress("Finalizing results...")

    converted_results = _convert_concepts_to_models(all_concepts)
    return any_lts_used, converted_results, llm_reasoning
