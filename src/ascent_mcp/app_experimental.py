from fastmcp import FastMCP
from fastmcp.server.event_store import EventStore

from ascent_mcp.authentication import combined_auth_provider, redis_client
from ascent_mcp.redis_storage_wrapper import RawRedisStorage
from ascent_mcp.telemetry import getTelemetryMiddleware

mcp_experimental = FastMCP(
    "Ascent Experimental MCP",
    auth=combined_auth_provider,
)

mcp_experimental.add_middleware(getTelemetryMiddleware("Ascent Experimental MCP")())
# NOTE: do NOT add fastmcp's PingMiddleware here — incompatible with
# stateless_http=True (crashes the session on ping). See app_v1.py.

# Import tools to register them with the experimental MCP instance.
# Must happen AFTER mcp_experimental is created but BEFORE http_app().
import ascent_mcp.tools_experimental  # noqa: F401, E402

experimental_event_store = EventStore(
    storage=RawRedisStorage(client=redis_client),
    max_events_per_stream=100,
    ttl=3600,
)
mcp_experimental_app = mcp_experimental.http_app(
    event_store=experimental_event_store, transport="http", path="/", stateless_http=True
)
