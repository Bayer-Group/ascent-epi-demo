import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Literal

from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette import status

from ascent_http.error_mapping import register_domain_error_handlers
from ascent_http.lifespan import combined_lifespan
from ascent_http.settings import settings
from ascent_http.util.correlation_id import get_correlation_id_short
from ascent_http.util.error import build_enhanced_traceback_from_exc, on_error
from ascent_http.util.logging import setup_logging
from ascent_mcp.app_experimental import mcp_experimental, mcp_experimental_app
from ascent_mcp.app_ui import mcp_ui, mcp_ui_app
from ascent_mcp.app_v1 import mcp_v1, mcp_v1_app
from ascent_mcp.authentication import redis_client

setup_logging()

path_prefix = "/api"

print("Initializing with env variables")
print(f"NON_OMOP_QUESTION_SANITY_CHECK_MODEL_NAME: {settings.NON_OMOP_QUESTION_SANITY_CHECK_MODEL_NAME}")
print(f"NON_OMOP_QUESTION_ANALYSIS_MODEL_NAME: {settings.NON_OMOP_QUESTION_ANALYSIS_MODEL_NAME}")
print(f"NON_OMOP_SQL_PREPARATION_MODEL_NAME: {settings.NON_OMOP_SQL_PREPARATION_MODEL_NAME}")
print(f"NON_OMOP_SQL_RESULTS_HEALING_MODEL_NAME: {settings.NON_OMOP_SQL_RESULTS_HEALING_MODEL_NAME}")
print(f"NON_OMOP_SQL_TO_QUESTION_MODEL_NAME: {settings.NON_OMOP_SQL_TO_QUESTION_MODEL_NAME}")

app = FastAPI(
    title="Ascent Backend",
    description="API server for the ascent application",
    openapi_url=f"{path_prefix}/openapi.json",
    docs_url=f"{path_prefix}/docs",
    redoc_url=None,
    swagger_ui_oauth2_redirect_url=f"{path_prefix}/oauth2-redirect",
    swagger_ui_init_oauth={
        "usePkceWithAuthorizationCodeGrant": True,
        "clientId": settings.AZURE_CLIENT_ID,
    },
    swagger_ui_parameters={"persistAuthorization": True},
    lifespan=combined_lifespan,
)

# Domain errors carry no HTTP status; this is where they acquire one.
register_domain_error_handlers(app)


# The REST routers were removed: this distribution exposes its capability over
# MCP only, and every /api/* route existed for the (not open-sourced) frontend.
# This one liveness probe is kept because it answers a question /mcp/health
# cannot -- whether the process is up at all, without touching Redis or
# enumerating tools. Deployment scripts and `docker compose` healthchecks want
# exactly that.
@app.get(f"{path_prefix}/public/health")
async def health() -> Literal["ok"]:
    """Liveness. Says nothing about dependencies -- see /mcp/health for those."""
    return "ok"


app.mount("/mcp/ascent-mcp-v1", mcp_v1_app)


@app.get("/mcp/health")
async def mcp_health():
    """
    Public health probe for the MCP servers.

    Reports in-process, non-sensitive signals only: whether each MCP server
    is mounted, how many tools are registered on each server, and whether
    Redis (used for session state) is reachable. No external probes
    (Snowflake / Gemini / etc.) and no tool invocations are performed here.

    The response is intentionally sanitized — no hostnames, error strings,
    or tool names — so it is safe to expose without authentication. Returns
    HTTP 200 when ``status == "ok"`` and 503 otherwise so ECS/k8s probes
    work without parsing JSON.

    Registered before the MCP mounts so the router resolves this path
    before delegating to a mounted MCP app.
    """
    import logging

    health_logger = logging.getLogger(__name__)
    checked_at = datetime.now(timezone.utc).isoformat()

    mounted_paths = {getattr(route, "path", None) for route in app.routes}

    async def _safe_tool_count(mcp_instance):
        try:
            tools = await mcp_instance.list_tools(run_middleware=False)
            return len(tools)
        except Exception:
            health_logger.exception("MCP health: list_tools failed")
            return None

    async def _safe_redis_ping():
        try:
            await asyncio.wait_for(redis_client.ping(), timeout=0.5)
            return True
        except Exception:
            health_logger.exception("MCP health: Redis ping failed")
            return False

    # Every served surface must be counted here: a server left out can fail to
    # mount while health stays green, and the reported count then disagrees with
    # what is actually mounted.
    v1_count, experimental_count, ui_count, redis_ok = await asyncio.gather(
        _safe_tool_count(mcp_v1),
        _safe_tool_count(mcp_experimental),
        _safe_tool_count(mcp_ui),
        _safe_redis_ping(),
    )

    servers = {
        "ascent-mcp-v1": {
            "mounted": "/mcp/ascent-mcp-v1" in mounted_paths,
            "tool_count": v1_count,
        },
        "ascent-experimental": {
            "mounted": "/mcp/ascent-experimental" in mounted_paths,
            "tool_count": experimental_count,
        },
        "ascent-ui-kit": {
            "mounted": "/mcp/ascent-ui-kit" in mounted_paths,
            "tool_count": ui_count,
        },
    }
    dependencies = {"redis": {"reachable": redis_ok}}

    not_mounted = any(not s["mounted"] for s in servers.values())
    list_failed = any(s["tool_count"] is None for s in servers.values())
    zero_tools = any(s["tool_count"] == 0 for s in servers.values())

    if not_mounted:
        rollup = "down"
    elif list_failed or zero_tools or not redis_ok:
        rollup = "degraded"
    else:
        rollup = "ok"

    return JSONResponse(
        status_code=200 if rollup == "ok" else 503,
        content={
            "status": rollup,
            "checked_at": checked_at,
            "servers": servers,
            "dependencies": dependencies,
        },
    )


app.mount("/mcp/ascent-experimental", mcp_experimental_app)
app.mount("/mcp/ascent-ui-kit", mcp_ui_app)

# No OAuth discovery routes: the MCP endpoints are unauthenticated here, so
# there is no authorization server for a client to discover. Clients connect
# directly.


# Add RFC 9470 Protected Resource Metadata endpoint for MCP OAuth discovery
@app.get("/.well-known/oauth-protected-resource/mcp")
async def oauth_protected_resource_mcp():
    """
    RFC 9470 Protected Resource Metadata endpoint.

    This endpoint advertises the MCP resource server's OAuth configuration,
    allowing clients to discover which authorization server protects this resource.
    """
    return {
        "resource": f"{settings.BASE_URL}/mcp",
        "authorization_servers": [settings.BASE_URL],
        "scopes_supported": ["user_impersonation"],
        "bearer_methods_supported": ["header"],
        "resource_documentation": f"{settings.BASE_URL}/api/docs",
    }


@app.get("/.well-known/oauth-protected-resource/mcp/ascent-mcp-v1")
async def oauth_protected_resource_mcp_v1():
    """RFC 9470 Protected Resource Metadata for the Ascent MCP V1 server."""
    return {
        "resource": f"{settings.BASE_URL}/mcp/ascent-mcp-v1",
        "authorization_servers": [settings.BASE_URL],
        "scopes_supported": ["user_impersonation"],
        "bearer_methods_supported": ["header"],
        "resource_documentation": f"{settings.BASE_URL}/api/docs",
    }


@app.get("/.well-known/oauth-protected-resource/mcp/ascent-experimental")
async def oauth_protected_resource_mcp_experimental():
    """RFC 9470 Protected Resource Metadata for the Experimental MCP server."""
    return {
        "resource": f"{settings.BASE_URL}/mcp/ascent-experimental",
        "authorization_servers": [settings.BASE_URL],
        "scopes_supported": ["user_impersonation"],
        "bearer_methods_supported": ["header"],
        "resource_documentation": f"{settings.BASE_URL}/api/docs",
    }


@app.middleware("http")
async def mcp_session_and_logging_middleware(request: Request, call_next):
    """
    MCP request/response logging middleware.

    Logs MCP requests and responses with session ID for debugging.

    NOTE: Cookie-based sticky session sync is NO LONGER NEEDED with Redis-backed
    session state. All session data is in Redis, accessible by any ECS task.
    The mcp-session-id header is still logged for debugging but not synced to cookies.
    """
    import logging

    logger = logging.getLogger(__name__)

    is_mcp = request.url.path.startswith("/mcp")

    # Log MCP requests
    if is_mcp or "well-known" in request.url.path:
        session_id = request.headers.get("mcp-session-id", "NONE")
        logger.warning(f"Request: {request.method} {request.url.path} - Session: {session_id}")

    response = await call_next(request)

    # Log MCP responses
    if is_mcp:
        session_id_response = response.headers.get("mcp-session-id", "NONE")
        logger.warning(f"Response: {response.status_code} - Session: {session_id_response}")
    elif "well-known" in request.url.path:
        logger.warning(f"Response: {response.status_code} for {request.url.path}")

    return response


_ALLOWED_ORIGINS = frozenset(origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip())


def _cors_headers(request: Request) -> dict[str, str]:
    """CORS headers for handlers that build a response outside the middleware.

    These paths run when CORSMiddleware does not, so they set the header
    themselves. They used to echo the request Origin unconditionally, which
    handed every error body -- including the traceback below -- to any site that
    asked, whatever CORS_ORIGINS said. Echo only what the policy already allows.
    """
    origin = request.headers.get("origin", "")
    if origin and ("*" in _ALLOWED_ORIGINS or origin in _ALLOWED_ORIGINS):
        return {"Access-Control-Allow-Origin": origin}
    return {}


@app.middleware("http")
async def mcp_error_protection_middleware(request: Request, call_next):
    """
    Middleware to catch and handle MCP-specific errors gracefully.

    This prevents server crashes from client disconnections and provides
    better error responses for MCP-related failures.
    """
    import logging

    from anyio import ClosedResourceError

    logger = logging.getLogger(__name__)

    try:
        return await call_next(request)
    except ClosedResourceError:
        if request.url.path.startswith("/mcp"):
            logger.warning(f"MCP stream closed for {request.url.path}. Client likely disconnected or timed out.")
            # Return 499 Client Closed Request
            return JSONResponse(
                status_code=499,
                content={"detail": "Client closed connection"},
                headers=_cors_headers(request),
            )
        # Re-raise for non-MCP routes
        raise
    except Exception:
        # Let other errors propagate to the main exception handler
        raise


@app.middleware("http")
async def save_request_body(request: Request, call_next):
    from starlette.requests import ClientDisconnect

    try:
        body = await request.body()
    except ClientDisconnect:
        return JSONResponse(
            status_code=499,
            content={"detail": "Client disconnected"},
            headers=_cors_headers(request),
        )
    request.state.body = body.decode("utf-8", errors="replace")
    response = await call_next(request)
    return response


if settings.TELEMETRY_ENABLED:
    # Imported here, not at module scope: importing the module builds an OTLP
    # exporter and installs the X-Ray propagator as a side effect, which is
    # work with nowhere to go when no collector is running.
    from ascent_platform.observability import configure_opentelemetry, configure_opentelemetry_local

    if settings.RELEASE_VERSION != "local":
        configure_opentelemetry(app)
    else:
        configure_opentelemetry_local(app)


@app.exception_handler(ValueError)
async def on_value_error(request: Request, exc: ValueError):
    # ascent-ai raises ValueError when required inputs (e.g. db_name) are missing.
    # Surface these as 400s so the frontend snackbar shows a clean message instead
    # of a 500 with a traceback.
    if "db_name is required" in str(exc):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "No database was specified. Please select a database and try again."},
            headers=_cors_headers(request),
        )
    return await on_exception(request, exc)


@app.exception_handler(Exception)
async def on_exception(request: Request, exc: Exception):
    # Special handling for MCP-related errors
    from anyio import ClosedResourceError

    # If this is an MCP request and the stream is closed, log and return 499 (client closed connection)
    if request.url.path.startswith("/mcp") and isinstance(exc, ClosedResourceError):
        logger = logging.getLogger(__name__)
        logger.warning(f"MCP client disconnected during request to {request.url.path}. This is expected when clients timeout or cancel requests.")
        # Return 499 Client Closed Request (non-standard but widely used)
        return JSONResponse(
            status_code=499,
            content={"detail": "Client disconnected"},
            headers=_cors_headers(request),
        )

    trace_id = getattr(request.state, "trace_id", "0" * 32)
    correlation_id = get_correlation_id_short()

    # The traceback carries absolute paths, source lines and the repr() of every
    # project frame's arguments, and its redaction is by argument *name* -- so a
    # parameter called dsn, query or email is rendered verbatim. That is a
    # debugging aid, not something to hand to whoever provoked the error. Log it
    # server-side always; return it only when DEBUG is explicitly on.
    enhanced_tb = build_enhanced_traceback_from_exc(exc)
    logger = logging.getLogger(__name__)
    logger.error(
        "Unhandled exception on %s (correlation_id=%s trace_id=%s)\n%s",
        request.url.path,
        correlation_id,
        trace_id,
        enhanced_tb,
    )

    if settings.DEBUG:
        detail = {
            "error": str(exc),
            "traceback": enhanced_tb,
            "correlation_id": correlation_id,
            "trace_id": trace_id,
        }
    else:
        detail = {
            "error": "Internal server error.",
            "correlation_id": correlation_id,
            "trace_id": trace_id,
        }
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=dict(detail=detail),
        headers={"X-Trace-Id": trace_id, **_cors_headers(request)},
    )


@app.middleware(middleware_type="http")
async def error_notification_middleware(request: Request, call_next):
    from anyio import ClosedResourceError
    from starlette.requests import ClientDisconnect

    try:
        return await call_next(request)
    except Exception as exception:
        if not isinstance(exception, (ClientDisconnect, ClosedResourceError)):
            await on_error(exception, request)
        raise exception


@app.middleware(middleware_type="http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["x-processing-time-ms"] = str(process_time * 1000.0)
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-amzn-trace-id", "x-trace-id", "mcp-session-id", "x-processing-time-ms"],
)
app.add_middleware(CorrelationIdMiddleware)
