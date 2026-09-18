# Ascent UI Kit (MCP-UI widget server)

The Ascent UI Kit is a small MCP server that lets agents present **interactive
widgets** — a dropdown, choice buttons, or a form — inline in the chat instead
of asking questions in plain text. The user's answer flows back to the agent as
a normal follow-up message, so the conversation continues without any custom
frontend work per use case.

It is **generic and domain-agnostic**: the widgets know nothing about ASCENT,
Snowflake, or cohorts. An agent fetches real data with a real tool (e.g.
`get_omop_databases` on the data server) and passes the values into a widget at
call time.

**Code:** `src/ascent_mcp/app_ui.py` (server) and `src/ascent_mcp/tools_ui.py` (tools)
**Mounted at:** `/mcp/ascent-ui-kit` (see `src/main.py`)
**Rendered by:** an MCP client with [MCP-UI](https://mcpui.dev/) support

---

## Why it exists

Agents frequently need the user to choose from a known, finite set — a
database, a cohort, a yes/no confirmation. Free-text questions are error-prone
(typos, ambiguous answers) and a poor experience. The UI kit turns those
questions into one-tap widgets with zero backend changes per feature: any agent
connected to both a data server and the UI kit can compose them.

## Example flow

1. User: *"Run this on an OMOP database."*
2. Agent calls `get_omop_databases` (data server) → gets `["SYNTHETIC_EHR_OMOP"]`.
3. Agent calls `ui_select(title="Choose a database", options=[...])` (UI kit).
4. The tool returns a `ui://` HTML resource; the agent places its
   `\ui{resource-id}` marker in the reply, and LibreChat renders the dropdown
   inline at that spot.
5. User picks *Synthetic EHR* and hits Submit → the widget posts an MCP-UI `prompt`
   action → LibreChat sends *"Synthetic EHR"* as a follow-up user message.
6. The agent continues with the selected database.

## The contract

Three pieces, all standard MCP-UI — nothing bespoke between server and host:

| Step | Mechanism |
|---|---|
| Tool → host | Tool returns an `EmbeddedResource` with a `ui://` URI and `text/html` content. LibreChat detects the URI, assigns the resource a stable id. |
| Placement | The model puts `\ui{resource-id}` in its reply; LibreChat's markdown plugin (`client/src/components/MCPUIResource/`) swaps the marker for the rendered iframe. `\ui{id1,id2}` renders a carousel. |
| Widget → agent | On submit, the widget `postMessage`s an MCP-UI `prompt` action to the host; LibreChat turns it into a follow-up user message. |

Widgets also post `ui-size-change` messages so the host sizes the iframe to
the content (in addition to mcp-ui's own ResizeObserver).

Each tool docstring instructs the model to place the marker — keep that
instruction when editing docstrings, or widgets stop appearing inline.

## Tools

All tools return a UI resource to be rendered via its `\ui{resource-id}` marker.

### `ui_select` — pick one value from a dropdown

| Arg | Type | Notes |
|---|---|---|
| `title` | str | Heading above the dropdown |
| `options` | list[str] | Selectable values |
| `description` | str | Optional sub-text |
| `submit_label` | str | Button label (default "Submit") |
| `prompt_template` | str | Text sent back on submit; `{selection}` is replaced with the choice, e.g. `"Use database {selection}."` |

### `ui_buttons` — one-tap choice buttons

| Arg | Type | Notes |
|---|---|---|
| `title` | str | Heading above the buttons |
| `options` | list[str] | One button per value; the clicked label is sent back immediately |
| `description` | str | Optional sub-text |

### `ui_form` — collect several inputs at once

| Arg | Type | Notes |
|---|---|---|
| `title` | str | Heading; also prefixes the submitted message |
| `fields` | list[dict] | Each: `name` (required), `label`, `type` (`"text"` \| `"select"`), `options` (for select), `placeholder` (for text) |
| `description` | str | Optional sub-text |
| `submit_label` | str | Button label (default "Submit") |

On submit the form sends back `"<title> — name1: value1; name2: value2"`.

## Architecture & security posture

The UI kit runs as its **own unauthenticated FastMCP server** (`mcp_ui` in
`app_ui.py`), separate from the authenticated data servers. That is deliberate:

- The tools are **presentation-only** — they format HTML from their arguments.
  No Snowflake access, no user context, no shared state, no secrets.
- All user-supplied strings are HTML-escaped before rendering.
- Because there is nothing to protect, LibreChat connects with
  `requiresOAuth: false`, so users never see an authorization prompt just to
  render a widget.

The server ships `instructions` (injected into the agent's context via
`serverInstructions: true` in `librechat.yaml`) telling the model to
*proactively* prefer widgets over free-text questions whenever the valid
answers are a known set.

## Wiring into LibreChat

The shipped `librechat/librechat.yaml` registers only `ascent-mcp`, so the
widgets are not active in the chat UI out of the box. To enable them, add a
second server alongside it:

```yaml
# librechat/librechat.yaml
mcpServers:
  ascent-ui-kit:
    type: streamable-http
    url: http://backend:8000/mcp/ascent-ui-kit/
    timeout: 600000
    requiresOAuth: false
    serverInstructions: true
```

`backend` is the compose service name. Use `http://localhost:8000/...` instead
if the client runs outside the compose network. No other configuration is
needed; rendering is handled by the MCP-UI support in the client.

## Adding a new widget

1. Add an `async` tool in `tools_ui.py` decorated with `@mcp_ui.tool()`.
2. Build the card body as escaped HTML and a submit script that sets `P` (the
   prompt text) and appends `_POST`; wrap both with the `_ui(...)` helper.
3. In the docstring, state *when* to use the widget and include the
   `\ui{resource-id}` marker instruction — docstrings are the model's only
   usage guide.
4. Keep it presentation-only: data comes in through arguments, never fetched
   inside the tool. If a widget needs data access, it belongs on an
   authenticated server instead.
5. If the widget class is worth proactive use, mention it in `_INSTRUCTIONS`
   in `app_ui.py`.

## Running locally

The UI kit is part of the main backend app — running the backend as usual
serves it at `http://localhost:8000/mcp/ascent-ui-kit/`. Point a local MCP
client at that URL.
