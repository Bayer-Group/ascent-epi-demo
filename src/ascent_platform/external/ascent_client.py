import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import aiohttp

from ascent_platform.azure import default_azure_tenant_id
from ascent_platform.azure.token import ServicePrincipalTokenProvider
from ascent_platform.config.runtime import get_runtime_settings
from ascent_platform.http import (
    MEDICAL_CODER_TOKEN_HTTP_TIMEOUT,
    medical_coder_timeout,
)

logger = logging.getLogger(__name__)

# Dedicated bulkhead for the blocking MSAL token call — never the default
# executor, which is shared with every other blocking call in the process.
_TOKEN_EXECUTOR = ThreadPoolExecutor(
    max_workers=get_runtime_settings().ASCENT_CLIENT_TOKEN_WORKERS,
    thread_name_prefix="ascent-client-token",
)


class AscentClient:
    def __init__(
        self,
        base_url: str,
        azure_client_id: str,
        azure_client_secret: str,
        azure_tenant_id: str | None = None,
    ):
        self.base_url = base_url
        azure_tenant_id = azure_tenant_id if azure_tenant_id is not None else default_azure_tenant_id()
        # The MSAL application itself now lives behind the shared provider, so
        # this client and the non-OMOP one mint tokens the same way. The local
        # token_cache below stays: MSAL caches too, but this layer is what
        # keeps the retry/expiry behaviour of both methods unchanged.
        self.tokens = ServicePrincipalTokenProvider(
            client_id=azure_client_id,
            client_secret=azure_client_secret,
            tenant_id=azure_tenant_id,
            http_timeout=MEDICAL_CODER_TOKEN_HTTP_TIMEOUT,
        )
        self.token_cache = None
        self.token_expiry = 0

    def invalidate_token(self) -> None:
        """Drop the cached token and mint a fresh one on the next call.

        Needed for the 401-after-success case: the local cache and MSAL's cache
        would both keep handing back the same rejected token until expiry.
        """
        self.token_cache = None
        self.token_expiry = 0
        self.tokens.force_refresh()

    def get_headers(self, max_retries=4):
        current_time = time.time()
        if not self.token_cache or current_time >= self.token_expiry:
            for attempt in range(max_retries):
                try:
                    # Token is not cached or has expired, so acquire a new one
                    result = self.tokens.acquire()
                    if "access_token" in result:
                        self.token_cache = result["access_token"]
                        # Set the expiry time 5 minutes before
                        self.token_expiry = current_time + result["expires_in"] - 300
                        logger.debug("Acquired service-principal token (len=%s)", len(self.token_cache or ""))
                        break
                    else:
                        logger.error(f"Failed to acquire token: {result.get('error_description')}")
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed with error: {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2**attempt)  # Exponential backoff

        headers = {"Authorization": f"Bearer {self.token_cache}"}
        return headers

    async def aget_headers(self, max_retries: int = 4) -> dict:
        """Async counterpart to :meth:`get_headers`.

        ``get_headers`` is synchronous and sleeps between retries. Calling it
        from a coroutine — as ``query_ascent_api`` did — blocks the event loop
        for the whole backoff (up to ~7s across four attempts) plus the MSAL
        network call, stalling every other request on the worker. Here the
        blocking MSAL call goes to a dedicated executor and the backoff uses
        ``asyncio.sleep``.
        """
        if self.token_cache and time.time() < self.token_expiry:
            return {"Authorization": f"Bearer {self.token_cache}"}

        loop = asyncio.get_running_loop()
        for attempt in range(max_retries):
            requested_at = time.time()
            try:
                result = await loop.run_in_executor(_TOKEN_EXECUTOR, self.tokens.acquire)
            except Exception as exc:
                logger.error("Attempt %d to acquire token failed: %s", attempt + 1, exc)
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2**attempt)
                continue

            if "access_token" in result:
                self.token_cache = result["access_token"]
                # Refresh 5 minutes early, measured from when the token was
                # requested rather than from entry to this method.
                self.token_expiry = requested_at + result["expires_in"] - 300
                break

            logger.error("Failed to acquire token: %s", result.get("error_description"))
            if attempt == max_retries - 1:
                raise RuntimeError(f"Token acquisition failed: {result.get('error_description')}")
            await asyncio.sleep(2**attempt)

        return {"Authorization": f"Bearer {self.token_cache}"}

    async def query_ascent_api(self, endpoint: str, payload: dict = None, params: dict = None, headers=None, method: str = "POST"):
        try:
            url = f"{self.base_url}/{endpoint}"
            if not headers:
                headers = await self.aget_headers()

            # `total` must be set: with only sock_* bounds a response that
            # drips a byte at a time never trips a timeout. The old values were
            # total=None with 1000s socket timeouts — effectively unbounded on
            # the tool path. The helper is the single definition of the bound.
            timeout = medical_coder_timeout()

            # Start timing the request
            start_time = time.time()

            async with aiohttp.ClientSession(trust_env=True, timeout=timeout) as session:
                if method.upper() == "GET":
                    async with session.get(url, headers=headers, params=params) as response:
                        response_data = await self.handle_response(response, url, start_time, payload)
                else:
                    async with session.post(url, headers=headers, json=payload, params=params) as response:
                        response_data = await self.handle_response(response, url, start_time, payload)

                return response_data
        except Exception as e:
            # Handle the exception in whatever way makes sense for your application
            logger.exception(f"An error occurred: {e}")
            raise

    @staticmethod
    async def _error_detail(response) -> str | None:
        """FastAPI's ``detail`` from an error body, if it has one.

        Best-effort: an error path must not fail on a body it cannot parse,
        and the body is capped because nothing useful to a human is longer.
        """
        try:
            body = await response.json()
        except Exception:  # noqa: BLE001 - not JSON, or no body at all
            return None
        detail = body.get("detail") if isinstance(body, dict) else None
        if detail is None:
            return None
        return str(detail)[:500]

    async def handle_response(self, response, url, start_time, payload):
        # Calculate the elapsed time
        elapsed_time = time.time() - start_time

        # raise_for_status() reports only the status line, so a service that
        # explains itself in the body ("collection X does not exist, pass
        # encoder='bge'") reached the caller as a bare "404, message='Not
        # Found'". Carry the detail across.
        if response.status >= 400:
            detail = await self._error_detail(response)
            response.release()
            raise aiohttp.ClientResponseError(
                response.request_info,
                response.history,
                status=response.status,
                message=detail or response.reason,
                headers=response.headers,
            )
        result = await response.json()
        # Log the timing and payload
        logger.info(f"Request to {url} took {elapsed_time:.2f} seconds")
        logger.debug(f"Request payload: {payload}")

        return result

    def get_version(self):
        return self.query_ascent_api("get-version")
