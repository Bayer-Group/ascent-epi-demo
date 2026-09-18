"""LLM-based filtering of medical-concept search results.

Ported from the old ``llm_filter.py``. Transport now goes through the thin LLM
connectors (``get_llm_connector(...).complete_json``); the include/exclude filter
prompt is built here and the raw JSON is parsed here — previously this lived in
the provider handler's ``generate_filter_json``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time as _time
from collections.abc import Awaitable, Callable
from typing import Any

import pandas as pd
from pydantic import BaseModel, StrictInt

from ascent_medical_coder.connectors.llm.factory import get_llm_connector
from ascent_medical_coder.prompts.prompts_llm_filter import get_prompt
from ascent_medical_coder.services.pipeline import lts

logger = logging.getLogger(__name__)

LLM_FILTER_MAX_RESULTS = 3000


class LLMFilterError(ValueError):
    """Filtering failed; callers must not return or cache unfiltered concepts."""


class _FilterDecision(BaseModel):
    include: list[StrictInt]
    exclude: list[StrictInt]
    reasons_include: list[str] | None = None
    reasons_exclude: list[str] | None = None


def _validate_decision(raw: dict[str, Any], expected: set[int]) -> _FilterDecision:
    decision = _FilterDecision.model_validate(raw)
    include, exclude = set(decision.include), set(decision.exclude)
    if raw.get("error") or include & exclude or include | exclude != expected:
        raise LLMFilterError("Include/exclude must partition exactly the expected indices")
    if len(include) != len(decision.include) or len(exclude) != len(decision.exclude):
        raise LLMFilterError("Duplicate filter indices are not supported")
    return decision


class LLMFilterResult(BaseModel):
    """Structured result of applying the LLM filter.

    Attributes
    ----------
    lts_hit: bool
        Whether LTS cache was used to answer without hitting the LLM.
    filtered_df: pd.DataFrame
        DataFrame after applying a validated filter. Failures raise LLMFilterError.
    reasons_include: Optional[List[str]]
        Optional list of textual reasons for included indices, as returned by the LLM.
    reasons_exclude: Optional[List[str]]
        Optional list of textual reasons for excluded indices, as returned by the LLM.
    """

    lts_hit: bool
    filtered_df: pd.DataFrame
    reasons_include: list[str] | None = None
    reasons_exclude: list[str] | None = None

    model_config = {
        "arbitrary_types_allowed": True,
    }


def _safe_parse_json(raw: str | None) -> dict[str, Any]:
    """Parse the raw JSON string from a connector, falling back on bad payloads.

    Mirrors the old per-handler ``_safe_parse_json``: on a decode failure OR a
    top-level value that is not an object (e.g. the model emits a bare array),
    returns a filter-shaped dict carrying the error rather than raising.
    """
    try:
        parsed = json.loads(raw or "{}")
        if not isinstance(parsed, dict):
            return {"include": [], "exclude": [], "error": "Top level JSON is not an object"}
        return parsed
    except json.JSONDecodeError as e:
        return {"include": [], "exclude": [], "error": str(e)}


async def _build_context(df: pd.DataFrame) -> str:
    """Build a compact textual context list for the prompt.
    Columns: index, CONCEPT_CODE (if available), CONCEPT_NAME, score.
    """
    if df.empty:
        return ""
    try:
        # Preserve the original index so row_index values are GLOBAL indices
        context_df = df.reset_index().rename(columns={"index": "row_index"})
        display_cols = [c for c in ["row_index", "CONCEPT_CODE", "CONCEPT_NAME", "score"] if c in context_df.columns]
        return context_df[display_cols].to_string(index=False)
    except Exception as e:
        raise LLMFilterError("Failed to build filter context") from e


async def _call_llm_filter_with_retry(
    connector,
    search_text: str,
    batch_df: pd.DataFrame,
    include_reasons: bool = False,
    custom_instructions: str | None = None,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Retry malformed decisions, then raise instead of returning partial decisions.

    Each successful batch is an exact, disjoint partition of its global labels.
    A failure is not an empty successful search and must never enter the cache.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    try:
        context_str = await _build_context(batch_df)
        prompt_text = await get_prompt(
            search_text,
            context_str,
            include_reasons=include_reasons,
            custom_instructions=custom_instructions,
        )
    except Exception as error:
        raise LLMFilterError("Failed to build filter prompt") from error

    for attempt in range(max_retries + 1):
        try:
            raw = await connector.complete_json(prompt_text)
            decision = _validate_decision(_safe_parse_json(raw), set(batch_df.index))
            result = decision.model_dump()
            if not include_reasons:
                result.update(reasons_include=None, reasons_exclude=None)
            return result
        except Exception as error:
            logger.warning("LLM filter batch failed, attempt %s/%s: %s", attempt + 1, max_retries + 1, error)
            if attempt == max_retries:
                raise LLMFilterError("No valid filtering decision after retries") from error
    raise LLMFilterError("No filtering attempt was made")


async def _apply_index_filter(
    hits_df: pd.DataFrame,
    include_indices: list[int],
    exclude_indices: list[int],
    progress: Callable[[str], Awaitable[Any]],
    *,
    scope: str = "",
) -> pd.DataFrame:
    """Apply only a complete, validated partition of the DataFrame labels."""
    try:
        _validate_decision({"include": include_indices, "exclude": exclude_indices}, set(hits_df.index))
        selected = sorted(include_indices) if scope else include_indices
        filtered = hits_df.loc[selected]
    except Exception as error:
        raise LLMFilterError("Invalid index-based filter") from error
    await progress(f"{scope or 'Single-batch'} filter applied: {len(filtered)} results remaining")
    return filtered


async def _update_lts_best_effort(
    search_text: str,
    filtered_df: pd.DataFrame,
    all_concept_ids: list[int],
    progress: Callable[[str], Awaitable[Any]] | None = None,
) -> None:
    """Update LTS without allowing a cache failure to fail the search."""
    try:
        accepted = filtered_df["CONCEPT_ID"].astype(int).tolist()
        await lts.update_lts(search_text, accepted, all_concept_ids)
    except Exception as error:
        logger.error("Failed updating LTS: %s", error)
        if progress:
            await progress(f"WARNING: LTS cache update failed: {error}")


def _aggregate_batch_results(batch_results: list[dict[str, Any]], include_reasons: bool):
    include: list[int] = []
    exclude: list[int] = []
    reasons_include: list[str] = []
    reasons_exclude: list[str] = []
    for result in batch_results:
        include.extend(result.get("include", []))
        exclude.extend(result.get("exclude", []))
        if include_reasons and isinstance(result.get("reasons_include"), list):
            reasons_include.extend(result["reasons_include"])
        if include_reasons and isinstance(result.get("reasons_exclude"), list):
            reasons_exclude.extend(result["reasons_exclude"])
    return include, exclude, reasons_include, reasons_exclude


async def apply_llm_filter(
    search_text: str,
    hits_df: pd.DataFrame,
    llm_name: str | None,
    allow_lts: bool = True,
    include_reasons: bool = True,
    custom_instructions: str | None = None,
    send_progress: Callable[[str], Awaitable[Any]] | None = None,
) -> LLMFilterResult:
    """Apply LLM-based filtering to search results DataFrame.

    Process:
      1. Optional LTS lookup; if complete hit, use cached accept/reject mask.
      2. Build prompt using indices + concept names only.
      3. Invoke selected LLM connector; parse JSON include/exclude *indices*.
      4. Filter DataFrame; update LTS if not previously cached.

    Returns an LLMFilterResult model instead of a raw tuple for clarity.
    Requested filtering must succeed or raise LLMFilterError; failures are not
    returned as successful (and potentially cacheable) concept lists.

    Args:
        send_progress: Optional async callback to send progress updates (for WebSocket).
    """

    async def progress(message: str):
        """Helper to send progress if callback is available."""
        logger.info(f"[LLM Filter] {message}")
        if send_progress:
            await send_progress(message)

    await progress(
        f"Starting LLM filter for query: '{search_text}' with {len(hits_df) if hits_df is not None else 0} results"
    )

    if hits_df is None:
        raise LLMFilterError("Filter input must be a DataFrame")
    if hits_df.empty or not llm_name:
        await progress("Skipping LLM filter: empty results or no LLM specified")
        return LLMFilterResult(lts_hit=False, filtered_df=hits_df, reasons_include=None, reasons_exclude=None)

    if len(hits_df) > LLM_FILTER_MAX_RESULTS:
        raise LLMFilterError(f"Result count exceeds filter limit of {LLM_FILTER_MAX_RESULTS}")
    if "CONCEPT_ID" not in hits_df.columns:
        raise LLMFilterError("Filter input is missing CONCEPT_ID")
    if not hits_df.index.is_unique or not pd.api.types.is_integer_dtype(hits_df.index.dtype):
        raise LLMFilterError("Filter input must have unique integer indices")

    concept_ids_series = hits_df["CONCEPT_ID"].astype(int).tolist()
    await progress(f"Extracted {len(concept_ids_series)} concept IDs for filtering")

    # 1. LTS cache check
    lts_hit = False
    # accepted_mask: List[bool]
    # if allow_lts:
    #     logger.info("Checking LTS cache for %s", search_text)
    #     try:
    #         lts_hit, accepted_mask = await lts.apply_lts(search_text, concept_ids_series)
    #         if lts_hit and accepted_mask:
    #             logger.info("LTS cache hit for %s", search_text)
    #             # Build filtered df from mask
    #             try:
    #                 filtered_df = hits_df[[bool(v) for v in accepted_mask]]
    #                 return LLMFilterResult(lts_hit=True, filtered_df=filtered_df)
    #             except Exception as e:
    #                 logger.error("Failed applying LTS mask, falling back to LLM: %s", e)
    #                 lts_hit = False  # treat as miss
    #     except Exception as e:
    #         logger.error("LTS apply failed: %s", e)
    #         lts_hit = False

    await progress(f"Initializing LLM handler: {llm_name}")
    connector = get_llm_connector(llm_name)
    if connector is None:
        raise LLMFilterError(f"Unknown LLM handler: {llm_name}")

    await progress(f"LLM handler '{llm_name}' initialized successfully")

    max_batch_size = 60
    if len(hits_df) <= max_batch_size:
        await progress(f"Using single-batch mode ({len(hits_df)} rows <= {max_batch_size} max batch size)")
        logger.info("Using single-batch LLM filtering (rows=%s)", len(hits_df))

        await progress("Calling LLM for filtering...")
        batch_result = await _call_llm_filter_with_retry(
            connector,
            search_text,
            hits_df,
            include_reasons=include_reasons,
            custom_instructions=custom_instructions,
        )
        include_indices = batch_result["include"]
        exclude_indices = batch_result["exclude"]
        reasons_include = batch_result.get("reasons_include")
        reasons_exclude = batch_result.get("reasons_exclude")

        await progress(f"LLM response received: {len(include_indices)} to include, {len(exclude_indices)} to exclude")
        logger.info("LLM include=%s, exclude=%s", include_indices, exclude_indices)

        filtered_df = await _apply_index_filter(hits_df, include_indices, exclude_indices, progress)

        if allow_lts and not lts_hit:
            await _update_lts_best_effort(search_text, filtered_df, concept_ids_series)

        await progress(f"LLM filter complete: {len(filtered_df)} final results")
        return LLMFilterResult(
            lts_hit=lts_hit,
            filtered_df=filtered_df,
            reasons_include=reasons_include,
            reasons_exclude=reasons_exclude,
        )

    num_batches = (len(hits_df) + max_batch_size - 1) // max_batch_size
    await progress(
        f"Using batched mode: {len(hits_df)} rows will be split into {num_batches} batches "
        f"(max {max_batch_size} per batch)"
    )
    logger.info("Using batched LLM filtering (rows=%s, batch_size=%s)", len(hits_df), max_batch_size)

    # Keep original indices per batch so results can be reassembled globally.
    batches = []
    for start in range(0, len(hits_df), max_batch_size):
        end = min(start + max_batch_size, len(hits_df))
        batch_df = hits_df.iloc[start:end]
        batches.append(batch_df)

    await progress(f"Created {len(batches)} batches, starting parallel processing (max 5 concurrent)...")

    semaphore = asyncio.Semaphore(5)
    _batch_start_time = _time.time()
    completed_batches = [0]  # list for a mutable counter shared with the closure

    async def _process_batch(batch_idx: int, batch_df: pd.DataFrame) -> dict[str, Any]:
        async with semaphore:
            start_t = _time.time()
            logger.info("Batch %d STARTED (rows=%d) at T+%.1fs", batch_idx, len(batch_df), start_t - _batch_start_time)
            result = await _call_llm_filter_with_retry(
                connector,
                search_text,
                batch_df,
                include_reasons=include_reasons,
                custom_instructions=custom_instructions,
            )
            end_t = _time.time()
            completed_batches[0] += 1
            logger.info("Batch %d DONE in %.1fs (T+%.1fs)", batch_idx, end_t - start_t, end_t - _batch_start_time)
            await progress(f"Batch {completed_batches[0]}/{len(batches)} completed")
            return result

    tasks = [_process_batch(i, batch_df) for i, batch_df in enumerate(batches)]
    batch_results = await asyncio.gather(*tasks)

    await progress(f"All {len(batches)} batches completed, aggregating results...")

    global_include, global_exclude, all_reasons_include, all_reasons_exclude = _aggregate_batch_results(
        batch_results, include_reasons
    )

    await progress(f"Aggregated results: {len(global_include)} to include, {len(global_exclude)} to exclude")
    logger.info("LLM include=%s, exclude=%s", global_include, global_exclude)

    filtered_df = await _apply_index_filter(hits_df, global_include, global_exclude, progress, scope="global")

    if allow_lts and not lts_hit:
        await _update_lts_best_effort(search_text, filtered_df, concept_ids_series, progress)

    total_time = _time.time() - _batch_start_time
    await progress(f"LLM filter complete: {len(filtered_df)} final results (total time: {total_time:.1f}s)")

    return LLMFilterResult(
        lts_hit=lts_hit,
        filtered_df=filtered_df,
        reasons_include=all_reasons_include or None,
        reasons_exclude=all_reasons_exclude or None,
    )


__all__ = ["LLMFilterError", "LLMFilterResult", "apply_llm_filter"]
