import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

# "Local" means the EMF library writes each metric document to stdout rather
# than to a sidecar. That is the right default for a stack that runs neither,
# but it is only reached when telemetry is enabled at all.
os.environ.setdefault("AWS_EMF_ENVIRONMENT", "Local")

from aws_embedded_metrics import metric_scope
from fastmcp.server.dependencies import get_access_token, get_http_headers
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.telemetry import get_tracer

from ascent_platform.config.runtime import get_runtime_settings

logger = logging.getLogger(__name__)


def _emf_enabled() -> bool:
    """Whether to emit CloudWatch metrics at all.

    Read per call rather than at import so a test can flip the setting without
    reimporting the middleware.
    """
    return get_runtime_settings().TELEMETRY_ENABLED


# Tiny dedicated executor: EMF metric emission must never compete with (or
# wait on) the default thread pool, and must never block session initialize.
_TELEMETRY_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mcp-telemetry")

# Hard ceiling on the initialize handshake. A healthy init is milliseconds —
# it does auth plus a tools/list build, no I/O of consequence. 15s is far above
# any legitimate case and far below the client's cell cap.
_INIT_TIMEOUT_SECONDS = float(os.environ.get("MCP_INIT_TIMEOUT_SECONDS", "15"))
# Above this, the handshake is logged at WARNING so slow inits are greppable
# before they become hangs.
_INIT_SLOW_MS = float(os.environ.get("MCP_INIT_SLOW_MS", "1000"))


def _log_emf_errors(future) -> None:
    exc = future.exception()
    if exc is not None:
        logger.warning("EMF metric emission failed: %s", exc)


def getTelemetryMiddleware(server_name: str) -> type[Middleware]:

    class TelemetryMiddleware(Middleware):
        async def on_initialize(self, context: MiddlewareContext, call_next):
            # Access as an object attribute, not a dict key
            params = context.message.params
            client_info = params.clientInfo

            # Identify the User from Auth Token
            try:
                username = get_access_token().claims.get("preferred_username", "unknown")
            except Exception:
                username = "unknown"

            # Safely get the name
            client_name = client_info.name if client_info else "unknown"
            # Fire-and-forget on the dedicated executor: initialize must never
            # wait on metric emission (or on a saturated default pool).
            if _emf_enabled():
                loop = asyncio.get_running_loop()
                future = loop.run_in_executor(_TELEMETRY_EXECUTOR, self._log_emf_initialize_metric, server_name, client_name, username)
                future.add_done_callback(_log_emf_errors)

            # Bound the handshake. Without this the client has no signal that
            # initialize is wedged and waits out its whole cell cap (observed:
            # 20 min, zero tool calls). Failing fast lets it reconnect.
            start = time.perf_counter()
            try:
                async with asyncio.timeout(_INIT_TIMEOUT_SECONDS):
                    response = await call_next(context)
            except TimeoutError:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                logger.error(
                    "MCP session initialize exceeded %.0fs — failing fast so the client can retry",
                    _INIT_TIMEOUT_SECONDS,
                    extra={
                        "mcp.server.name": server_name,
                        "mcp.client.name": client_name,
                        "user.name": username,
                        "mcp_event": "initialize_timeout",
                        "duration_ms": elapsed_ms,
                    },
                )
                raise

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            # Structured, not print(): a bare print writes to stdout on the event
            # loop, so a stalled log pipe blocks every coroutine on the worker.
            log = logger.warning if elapsed_ms > _INIT_SLOW_MS else logger.info
            log(
                "MCP session initialized in %.0fms (client=%s)",
                elapsed_ms,
                client_name,
                extra={
                    "mcp.server.name": server_name,
                    "mcp.client.name": client_name,
                    "user.name": username,
                    "mcp_event": "initialize",
                    "duration_ms": elapsed_ms,
                },
            )
            return response

        async def on_call_tool(self, context: MiddlewareContext, call_next):
            # 1. IDENTIFY ALL DATA POINTS
            # Get the Client
            client_name = await context.fastmcp_context.get_state("client_id")
            if not client_name:
                try:
                    headers = get_http_headers()
                    client_name = headers.get("x-mcp-client")
                    if not client_name:
                        ua = headers.get("user-agent")
                        if ua:
                            # Keep only the product token so version bumps don't
                            # split one client across dashboard categories.
                            client_name = ua.split("/")[0].strip().lower()
                except Exception:
                    client_name = None
            client_name = client_name or "unknown-mcp-client"

            # Identify the User from Auth Token
            try:
                username = get_access_token().claims.get("preferred_username", "unknown")
            except Exception:
                username = "unknown"

            # Get the Tool name
            tool_name = context.message.name

            # TRACE EXECUTION (X-Ray)
            tracer = get_tracer()
            with tracer.start_as_current_span("tool_call") as span:
                span.set_attribute("mcp.server.name", server_name)
                span.set_attribute("mcp.tool.name", tool_name)
                span.set_attribute("mcp.client.name", client_name)
                span.set_attribute("user.name", username)

                # Link these attributes to X-Ray for easy searching in the X-Ray console
                span.set_attribute("aws.xray.annotations", ["user.name", "mcp.server.name", "mcp.tool.name", "mcp.client.name"])

                # Time + observe the tool call. Exceptions propagate through
                # FastMCP middleware (see fastmcp.server.server.call_tool),
                # so a try/except here is the correct pattern. We re-raise
                # to preserve the original behavior — we observe, not swallow.
                start = time.perf_counter()
                status = "success"
                error_type: str | None = None
                try:
                    return await call_next(context)
                except Exception as exc:
                    status = "error"
                    error_type = type(exc).__name__
                    # Single canonical structured error log for the failed tool
                    # call. FastMCP's own logger has propagate=False, so without
                    # this, the traceback never reaches CloudWatch's JSON pipeline
                    # for tools that don't go through @resilient_tool.
                    logger.exception(
                        "MCP tool failed: %s",
                        tool_name,
                        extra={
                            "mcp.tool.name": tool_name,
                            "mcp.server.name": server_name,
                            "mcp.client.name": client_name,
                            "user.name": username,
                            "mcp_tool_name": tool_name,
                            "mcp_server_name": server_name,
                            "error_type": error_type,
                        },
                    )
                    raise
                finally:
                    duration_ms = (time.perf_counter() - start) * 1000.0
                    # Fire-and-forget on the dedicated executor — metric
                    # emission must not extend the tool call or touch the
                    # shared default pool.
                    if _emf_enabled():
                        loop = asyncio.get_running_loop()
                        future = loop.run_in_executor(
                            _TELEMETRY_EXECUTOR,
                            self._log_emf_metric,
                            server_name,
                            tool_name,
                            client_name,
                            username,
                            status,
                            error_type,
                            duration_ms,
                        )
                        future.add_done_callback(_log_emf_errors)

        @metric_scope
        def _log_emf_initialize_metric(self, server, client, user, metrics):
            """
            Internal helper to generate the AWS Embedded Metric Format (EMF) log for initialize.
            """
            metrics.set_namespace("ECS/AWSOTel/Application")

            metrics.set_dimensions({"mcp.server.name": server, "mcp.event": "initialize"})

            metrics.put_metric("mcp.initialize_calls_total", 1, "Count")

            metrics.set_property("mcp.client.name", client)
            metrics.set_property("mcp_server_name", server)
            metrics.set_property("user.name", user)
            metrics.set_property("mcp_event", "initialize")

        @metric_scope
        def _log_emf_metric(self, server, tool, client, user, status, error_type, duration_ms, metrics):
            """
            Internal helper to generate the AWS Embedded Metric Format (EMF) log.

            Emits the original ``mcp.tool_calls_total`` counter plus
            success/error/latency metrics so CloudWatch can surface per-tool
            success rate and p50/p95 latency without an external probe.
            """
            # Define where these metrics appear in CloudWatch
            metrics.set_namespace("ECS/AWSOTel/Application")

            # --- DIMENSIONS ---
            # Kept to these three so metric cardinality stays bounded.
            metrics.set_dimensions({"mcp.server.name": server, "mcp.tool.name": tool})

            # --- THE METRICS ---
            metrics.put_metric("mcp.tool_calls_total", 1, "Count")
            metrics.put_metric("mcp.tool_call_success", 1 if status == "success" else 0, "Count")
            metrics.put_metric("mcp.tool_call_error", 1 if status == "error" else 0, "Count")
            metrics.put_metric("mcp.tool_call_latency_ms", duration_ms, "Milliseconds")

            # --- PROPERTIES ---
            # These appear in your log group but do not create custom metrics.
            metrics.set_property("user.name", user)
            metrics.set_property("mcp.client.name", client)
            metrics.set_property("mcp_server_name", server)  # Split for easier log filtering
            metrics.set_property("mcp_tool_name", tool)  # Split for easier log filtering
            metrics.set_property("full_context", f"{server}:{tool}")  # For human readability
            metrics.set_property("status", status)
            if error_type is not None:
                metrics.set_property("error_type", error_type)

    return TelemetryMiddleware
