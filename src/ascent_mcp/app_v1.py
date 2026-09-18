from fastmcp import FastMCP
from fastmcp.server.event_store import EventStore

from ascent_mcp.authentication import combined_auth_provider, redis_client
from ascent_mcp.redis_storage_wrapper import RawRedisStorage
from ascent_mcp.telemetry import getTelemetryMiddleware

mcp_v1 = FastMCP(
    "Ascent MCP V1",
    auth=combined_auth_provider,
)

mcp_v1.add_middleware(getTelemetryMiddleware("Ascent MCP V1")())
# NOTE: do NOT add fastmcp's PingMiddleware here. This app runs with
# stateless_http=True; send_ping() awaits a client response on the per-request
# stream, which is gone by the time the ping fires -> ClosedResourceError tears
# down the session with "Stateless session crashed".
# Liveness for long tool calls comes from progress notifications instead
# (see ascent_mcp/tools/_keepalive.py).

# Import tools to register them with the MCP V1 instance.
# Must happen AFTER mcp_v1 is created but BEFORE http_app().
import ascent_mcp.tools_v1  # noqa: F401, E402
import ascent_mcp.tools_v1_clues  # noqa: F401, E402
import ascent_mcp.tools_v1_cohort  # noqa: F401, E402
import ascent_mcp.tools_v1_skills  # noqa: F401, E402

v1_event_store = EventStore(
    storage=RawRedisStorage(client=redis_client),
    max_events_per_stream=100,
    ttl=3600,
)
mcp_v1_app = mcp_v1.http_app(event_store=v1_event_store, transport="http", path="/", stateless_http=True)
