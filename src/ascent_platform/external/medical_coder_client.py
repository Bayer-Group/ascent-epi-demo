"""The non-OMOP entry point to the medical coder.

This was a second HTTP client to a service the codebase already had one for:
`requests` where everything else used aiohttp, its own MSALClient, its own
timeouts. The token client went first (see ascent_platform.azure.token); this
removes the transport, so there is now one way out to the coder.

What survives the move is the resilience the sync client had and the async one
did not -- retry with exponential backoff and jitter, on exactly the failures a
retry can fix. That policy now sits on the shared client, so the MCP tools get
it too; they had none.

The public surface is unchanged. ``execute_medical_coder_full_result`` is still
a synchronous call returning the same envelope, because its caller reaches it
through ``asyncio.to_thread`` (sql_postprocessing -> process_entity_reference)
and converting that chain is a separate job: it also holds a synchronous Redis
client. Inside that worker thread there is no running loop, so asyncio.run is
the correct bridge -- and if someone later calls this from a coroutine, it says
so instead of deadlocking.
"""

import asyncio
import logging
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

import aiohttp
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from ascent_platform.azure.token import AuthenticationError
from ascent_platform.external.medical_coder import MedicalCoder

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Custom exception for API request failures."""
    pass


class RetryableAPIError(APIError):
    """API failure worth retrying: connect/timeout errors and 401-after-refresh.

    5xx responses are deliberately NOT retryable — hammering a struggling
    medical coder with fixed-interval retries amplifies the outage.
    """
    pass


DOMAINS_NOT_TO_SEND = ["procedure", "condition", "measurement"]

# Bounds come from ascent_platform.http via the shared client; this module no
# longer sets its own. The old note is worth keeping: the read timeout must
# stay ABOVE legitimate drug lookups (slowest observed success ~198s) or it
# severs working calls, and the server sends nothing until the response is
# fully built, so it is effectively the server-compute budget.

_MAX_ATTEMPTS = 3


@lru_cache(maxsize=1)
def _client() -> MedicalCoder:
    """One client for this path, as MSALClient was one instance before.

    Constructing MedicalCoder builds an MSAL application, so a fresh one per
    call would re-do that work on every entity in the fan-out.
    """
    return MedicalCoder()


def _classify(exc: BaseException, client: MedicalCoder) -> Exception:
    """Map an aiohttp failure onto the retry policy the sync client had."""
    if isinstance(exc, (aiohttp.ClientConnectionError, asyncio.TimeoutError)):
        logger.error("Request failed (transient): %s", exc)
        return RetryableAPIError(f"Request failed: {exc}")
    if isinstance(exc, aiohttp.ClientResponseError) and exc.status == 401:
        # The provider normally keeps the token fresh; a real 401 means
        # something unexpected — force a fresh token and retry once.
        logger.info("Received 401 Unauthorized. Forcing token refresh.")
        client.invalidate_token()
        return RetryableAPIError(f"Request failed: {exc}")
    logger.error("Request failed: %s", exc)
    return APIError(f"Request failed: {exc}")


async def aquery_medical_codes(
    query: str,
    top_k: int = 1000,
    llm_filter: str = "chatgpt",
    domain_id: Optional[str] = None,
    vocabulary: Optional[List[str]] = None,
    standard_concept: Optional[str] = None,
    cosine_similarity: Optional[float] = None,
) -> Dict[str, Any]:
    """The raw coder response, retried on what a retry can fix.

    cosine_similarity: when None, the server applies its encoder-specific
    default (0.84 for gemini, 0.65 for bge/sap/biolord). Passing a literal
    value overrides that — avoid unless you specifically want to widen or
    tighten the cutoff regardless of encoder.
    """
    payload: Dict[str, Any] = {
        "query": query,
        "top_k": top_k,
        "llm_filter": llm_filter,
        "domain_ids": [domain_id] if domain_id else [],
        "vocabulary": vocabulary or [],
        "standard_concept": standard_concept,
    }
    if cosine_similarity is not None:
        payload["cosine_similarity"] = cosine_similarity

    client = _client()
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        wait=wait_exponential_jitter(initial=1, max=15),
        retry=retry_if_exception_type(RetryableAPIError),
        reraise=True,
    ):
        with attempt:
            try:
                # query_ascent_api, not get_medical_codes: this path returns the
                # coder's raw JSON, where get_medical_codes reshapes it through
                # construct_response. Callers read response_data directly.
                return await client.query_ascent_api("get-medical-codes", payload=payload)
            except (APIError, AuthenticationError):
                raise
            except Exception as exc:  # noqa: BLE001 — classified immediately below
                raise _classify(exc, client) from exc
    raise APIError("unreachable: AsyncRetrying exhausted without raising")


async def afetch_medical_coder_full_result(
    query: str,
    top_k: int = 1000,
    llm_filter: str = "chatgpt",
    domain_id: Optional[str] = None,
    vocabulary: Optional[List[str]] = None,
    standard_concept: Optional[str] = None,
    cosine_similarity: Optional[float] = None,
) -> Dict:
    """The coder call plus the envelope its callers expect."""
    time_start = time.time()
    try:
        if domain_id and domain_id.lower() in DOMAINS_NOT_TO_SEND:
            logger.info(f"Domain {domain_id} will not be sent to Medical Coder.")
            domain_id = None
        response_data = await aquery_medical_codes(
            query=query,
            top_k=top_k,
            llm_filter=llm_filter,
            domain_id=domain_id,
            vocabulary=vocabulary,
            standard_concept=standard_concept,
            cosine_similarity=cosine_similarity,
        )
        elapsed_time = time.time() - time_start
        logger.info(f"Time elapsed for {query}: {round(elapsed_time, 2)} seconds")
        return {
            "query": query,
            "cosine_similarity": cosine_similarity,
            "top_k": top_k,
            "response_data": response_data,
            "elapsed_time": f"{round(elapsed_time, 2)} sec",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    except (APIError, AuthenticationError) as e:
        logger.error(f"An error occurred: {e}")
        return _error_envelope(query, top_k, cosine_similarity, e)
    except ValueError as e:  # domain_id not an integer
        logger.error(f"domain_id must be an integer: {e}")
        return _error_envelope(query, top_k, cosine_similarity, e)


def _error_envelope(query, top_k, cosine_similarity, exc) -> Dict:
    return {
        "query": query,
        "cosine_similarity": cosine_similarity,
        "top_k": top_k,
        "response_data": None,
        "error": str(exc),
        "elapsed_time": "N/A",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def execute_medical_coder_full_result(
    query: str,
    top_k: int = 1000,
    llm_filter: str = "chatgpt",
    domain_id: Optional[str] = None,
    vocabulary: Optional[List[str]] = None,
    standard_concept: Optional[str] = None,
    cosine_similarity: Optional[float] = None,
) -> Dict:
    """Synchronous bridge for callers already running in a worker thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "execute_medical_coder_full_result is synchronous and was called from a "
            "running event loop, which would block it. Await "
            "afetch_medical_coder_full_result instead, or keep the asyncio.to_thread "
            "hop the non-OMOP pipeline uses."
        )

    return asyncio.run(
        afetch_medical_coder_full_result(
            query=query,
            top_k=top_k,
            llm_filter=llm_filter,
            domain_id=domain_id,
            vocabulary=vocabulary,
            standard_concept=standard_concept,
            cosine_similarity=cosine_similarity,
        )
    )
