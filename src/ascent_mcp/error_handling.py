"""
Error handling and resilience for MCP server.

This module provides robust error handling for the MCP server to prevent crashes
from closed connections, timeouts, and other transient failures.
"""

import asyncio
import functools
import logging
from typing import Any, Callable, ParamSpec, TypeVar

from anyio import ClosedResourceError
from fastapi import HTTPException

from ascent_domain.errors import DomainError

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def _domain_error_to_tool_result(func_name: str, exc: "DomainError") -> dict:
    """Same translation for domain errors.

    Tools used to see these as HTTPException subclasses and this module caught
    them by that type. Now that the domain raises transport-free errors, the
    MCP surface maps them itself -- otherwise an agent would get an opaque
    JSON-RPC failure where it used to get {"error": "access_denied", ...}.
    """
    from ascent_http.error_mapping import status_for

    code = status_for(exc)
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    logger.warning("Tool '%s' raised %s (%s): %s", func_name, type(exc).__name__, code, detail)
    return {
        "error": "access_denied" if code in (401, 403) else "request_failed",
        "status_code": code,
        "message": detail,
    }


def _http_exception_to_tool_result(func_name: str, exc: HTTPException) -> dict:
    """Translate an HTTPException raised inside a tool into a structured
    MCP-friendly result so the user-facing message reaches the client
    instead of being lost in an opaque JSON-RPC error."""
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    logger.warning("Tool '%s' raised HTTPException %s: %s", func_name, exc.status_code, detail)
    return {
        "error": "access_denied" if exc.status_code in (401, 403) else "request_failed",
        "status_code": exc.status_code,
        "message": detail,
    }


def resilient_tool(timeout_seconds: int = 30):
    """
    Decorator to make MCP tools resilient to common failures.

    This decorator:
    1. Adds timeout protection to prevent long-running async operations
    2. Handles ClosedResourceError gracefully when client disconnects
    3. Provides better error logging for debugging
    4. Works with both sync and async functions

    Args:
        timeout_seconds: Maximum time allowed for async tool execution (default: 30s)

    Example:
        @mcp.tool
        @resilient_tool(timeout_seconds=45)
        async def my_slow_tool():
            # This will timeout after 45 seconds
            await some_long_operation()

        @mcp.tool
        @resilient_tool()  # No timeout for sync functions
        def my_sync_tool():
            return {"data": "value"}
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Check if function is async or sync
        is_async = asyncio.iscoroutinefunction(func)

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    # Apply timeout to prevent indefinite execution
                    result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
                    return result

                except asyncio.TimeoutError:
                    error_msg = (
                        f"Tool '{func.__name__}' exceeded timeout of {timeout_seconds}s. "
                        f"This may indicate a slow database query or external API call. "
                        f"Consider optimizing the operation or increasing the timeout."
                    )
                    logger.error(error_msg)
                    raise TimeoutError(error_msg)

                except ClosedResourceError:
                    # Client disconnected - log but don't crash
                    logger.warning(
                        f"Client disconnected while executing tool '{func.__name__}'. "
                        f"This is usually caused by client timeout or cancellation. "
                        f"The operation may have taken too long (limit: {timeout_seconds}s)."
                    )
                    # Re-raise to let MCP SDK handle it gracefully
                    raise

                except DomainError as e:
                    return _domain_error_to_tool_result(func.__name__, e)
                except HTTPException as e:
                    return _http_exception_to_tool_result(func.__name__, e)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return func(*args, **kwargs)

                except ClosedResourceError:
                    # Client disconnected - log but don't crash
                    logger.warning(f"Client disconnected while executing tool '{func.__name__}'.")
                    raise

                except DomainError as e:
                    return _domain_error_to_tool_result(func.__name__, e)
                except HTTPException as e:
                    return _http_exception_to_tool_result(func.__name__, e)

            return sync_wrapper

    return decorator
