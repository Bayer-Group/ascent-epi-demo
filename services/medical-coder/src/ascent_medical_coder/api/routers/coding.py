"""FastAPI transport for the medical-coding endpoints.

Thin HTTP/WebSocket layer over :mod:`ascent_medical_coder.services.coding`.
Routes only handle request/response plumbing (dependencies, headers, WebSocket
transport, keepalive); all pipeline logic lives in the service module.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, WebSocket, WebSocketDisconnect

from ascent_medical_coder.core.security import User, get_current_user, validate_websocket_token
from ascent_medical_coder.schemas.coding import MedicalCodingRequest, ScoredConcept
from ascent_medical_coder.services import coding

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/get-medical-codes")
async def get_medical_codes(
    payload: MedicalCodingRequest,
    response: Response,
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> dict[str, list[ScoredConcept]]:
    logger.info("/get-medical-codes input arguments: %r", payload)
    start_time = time.time()

    normalized, error_detail = coding.prepare_request(payload)
    if error_detail is not None:
        raise HTTPException(status_code=400, detail=error_detail)

    any_lts_used, results, _ = await coding.run_domains_parallel(
        payload, current_user, normalized, start_time, include_reasoning=False
    )
    response.headers["x-lts-used"] = str(any_lts_used)
    return results


@router.post("/get-medical-codes-reasoning")
async def get_medical_codes_reasoning(
    payload: MedicalCodingRequest,
    response: Response,
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> dict:
    """Frontend-oriented variant of get_medical_codes that also returns LLM reasoning.

    Response shape:
    {
        "results": { ...as in get-medical-codes... },
        "llm_reasoning": {
            "per_domain": {
                <domain_id_or_null>: {
                    "filter_used": bool,
                    "lts_used": bool,
                    "raw_response": str | None,
                },
                ...
            }
        }
    }
    """
    logger.info("/get-medical-codes-frontend input arguments: %r", payload)
    start_time = time.time()

    normalized, error_detail = coding.prepare_request(payload)
    if error_detail is not None:
        raise HTTPException(status_code=400, detail=error_detail)

    any_lts_used, results, llm_reasoning = await coding.run_domains_parallel(
        payload, current_user, normalized, start_time, include_reasoning=True
    )
    response.headers["x-lts-used"] = str(any_lts_used)
    return {
        "results": results,
        "llm_reasoning": {"per_domain": llm_reasoning},
    }


# Registered directly on the app in main.py to skip the router-wide Security dependency.
async def websocket_get_medical_codes_reasoning(
    websocket: WebSocket,
    token: str = Query(None),
):
    """WebSocket endpoint for medical codes with reasoning to avoid load balancer timeouts.

    Connection flow:
    1. Client connects with ?token=<jwt_token> query parameter
    2. Client sends JSON payload matching MedicalCodingRequest schema
    3. Server sends progress updates as JSON: {"type": "progress", "message": "..."}
    4. Server sends final result as JSON: {"type": "result", "data": {...}}
    5. Server sends error if any: {"type": "error", "message": "..."}

    The WebSocket keeps the connection alive with progress messages to prevent
    load balancer timeouts during long-running operations.
    """
    if not token:
        await websocket.close(code=4001, reason="Missing authentication token")
        return

    current_user = await validate_websocket_token(token)
    if not current_user:
        await websocket.close(code=4001, reason="Invalid authentication token")
        return

    await websocket.accept()

    async def send_progress(message: str):
        """Send a progress update with actual status information."""
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "progress", "message": message, "timestamp": time.time()})

    async def send_keepalive():
        """Send a keepalive ping to prevent connection timeout (frontend should not display this)."""
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "keepalive", "timestamp": time.time()})

    async def send_error(message: str, detail: Any = None):
        """Send an error message."""
        error_data = {"type": "error", "message": message}
        if detail:
            error_data["detail"] = detail
        await websocket.send_json(error_data)

    async def send_result(data: dict):
        """Send the final result."""
        await websocket.send_json({"type": "result", "data": data})

    try:
        raw_data = await websocket.receive_text()

        try:
            payload_dict = json.loads(raw_data)
            payload = MedicalCodingRequest(**payload_dict)
        except json.JSONDecodeError as e:
            await send_error("Invalid JSON payload", str(e))
            await websocket.close()
            return
        except Exception as e:
            await send_error("Invalid request payload", str(e))
            await websocket.close()
            return

        logger.info("WebSocket /ws/get-medical-codes-reasoning input: %r", payload)
        start_time = time.time()

        await send_progress("Processing request...")

        normalized, error_detail = coding.prepare_request(payload)
        if error_detail is not None:
            await send_error("MixedDomainIdsNotAllowed", error_detail)
            await websocket.close()
            return

        keepalive_active = True

        async def keepalive_task():
            while keepalive_active:
                await asyncio.sleep(5)
                if keepalive_active:
                    await send_keepalive()

        keepalive = asyncio.create_task(keepalive_task())

        try:
            any_lts_used, converted_results, llm_reasoning = await coding.run_domains_sequential(
                payload, current_user, normalized, start_time, send_progress=send_progress
            )

            serializable_results = {
                k: [c.model_dump() if hasattr(c, "model_dump") else c for c in v]
                for k, v in converted_results.items()
            }

            await send_result(
                {
                    "results": serializable_results,
                    "llm_reasoning": {"per_domain": llm_reasoning},
                    "lts_used": any_lts_used,
                }
            )

        finally:
            keepalive_active = False
            keepalive.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await keepalive

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        with contextlib.suppress(Exception):
            await send_error("Internal server error", str(e))
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()
