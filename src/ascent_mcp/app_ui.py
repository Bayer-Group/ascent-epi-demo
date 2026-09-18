from fastmcp import FastMCP

from ascent_mcp.telemetry import getTelemetryMiddleware

# Unauthenticated presentation server for generic MCP-UI widgets.
#
# The UI tools only format HTML from their arguments — no user context, no
# warehouse, no data access — so they require no authentication. LibreChat
# connects to this server with `requiresOAuth: false`, so users never have to
# authorize it separately (unlike the data servers, which gate on real auth).
# Injected into the agent's context by LibreChat when this server is connected
# (librechat.yaml: `serverInstructions: true`). Tells the model to *proactively*
# surface widgets for constrained choices instead of asking in plain text.
_INSTRUCTIONS = (
    "When you need the user to choose from a known, finite set of options — a "
    "database, a cohort, a code list, or a yes/no confirmation — present it as an "
    "interactive widget instead of asking in plain text:\n"
    "- ui_select: pick ONE value from a list (renders a dropdown).\n"
    "- ui_buttons: a quick one-tap choice (e.g. Yes / No, or a few options).\n"
    "- ui_form: collect several inputs at once.\n"
    "Pass the options you already have (e.g. from get_omop_databases) into the "
    "widget, then place its \\ui{resource-id} marker in your reply so it renders "
    "inline. The user's selection comes back as a follow-up message — continue from "
    "there. Prefer a widget over a free-text question whenever the valid answers are "
    "a known set."
)

mcp_ui = FastMCP("Ascent UI Kit", instructions=_INSTRUCTIONS)

mcp_ui.add_middleware(getTelemetryMiddleware("Ascent UI Kit")())

# Import tools to register them with the UI Kit instance.
# Must happen AFTER mcp_ui is created but BEFORE http_app().
import ascent_mcp.tools_ui  # noqa: F401, E402

mcp_ui_app = mcp_ui.http_app(transport="http", path="/", stateless_http=True)
