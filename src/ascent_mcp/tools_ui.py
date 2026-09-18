"""
UI Kit tools — generic, domain-agnostic interactive widgets.

These tools return MCP-UI resources (``ui://`` HTML documents) that MCP-UI
hosts such as LibreChat render inline in the chat. On submit, each widget posts
an MCP-UI ``prompt`` action back to the host, which the host turns into a
follow-up message — so the agent receives the user's choice and continues.

They are presentation-only: no warehouse access, no auth-sensitive data, no
shared state. An agent fetches data with a real tool (e.g. ``get_omop_databases``
on the data server) and passes the values in here. They live on their own
unauthenticated ``mcp_ui`` server (see ``app_ui``) so users never have to
authorize a separate endpoint just to render a widget.

Rendering is handled entirely by the host. The only contract is that the tool
returns a ``ui://`` ``EmbeddedResource``; LibreChat detects the ``ui://`` URI,
assigns it a stable id, and renders it where the model places the
``\\ui{resource-id}`` marker. Keep the marker instruction in each docstring so
the model knows to surface the widget inline.
"""

import html as _html
import json
from typing import Any

from fastmcp.tools.tool import ToolResult
from mcp.types import EmbeddedResource, TextResourceContents
from pydantic import AnyUrl

from ascent_mcp.app_ui import mcp_ui

_CSS = """
 :root{--card:#1b2436;--ink:#e8eef6;--mut:#9aa7bd;--acc:#3b9eff;--field:#0e1626;--br:#2c3a52}
 *{box-sizing:border-box}
 html,body{margin:0;padding:0;background:transparent;overflow:hidden}
 body{font:13.5px/1.4 -apple-system,system-ui,Segoe UI,Roboto,sans-serif;color:var(--ink)}
 .card{width:100%;max-width:380px;background:var(--card);border:1px solid var(--br);
   border-radius:12px;padding:14px 15px}
 h3{margin:0 0 2px;font-size:14.5px;font-weight:600}
 p.sub{margin:0 0 10px;color:var(--mut);font-size:12px}
 label{display:block;font-size:10.5px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;
   color:var(--mut);margin:8px 0 4px}
 select,input{width:100%;padding:8px 10px;border-radius:8px;border:1px solid var(--br);
   background:var(--field);color:var(--ink);font:inherit;outline:none}
 select:focus,input:focus{border-color:var(--acc)}
 button.primary{width:100%;margin-top:12px;padding:9px 12px;border-radius:8px;border:none;
   background:var(--acc);color:#04233f;font-weight:700;font-size:13px;cursor:pointer}
 button.primary:hover{filter:brightness(1.06)}
 .btns{display:flex;flex-wrap:wrap;gap:8px;margin-top:4px}
 .btns button{padding:8px 14px;border-radius:8px;border:1px solid var(--br);background:var(--field);
   color:var(--ink);font:inherit;cursor:pointer}
 .btns button:hover{border-color:var(--acc);color:#cfe6ff}
"""

# Reports content height to the host so the iframe sizes exactly (belt-and-suspenders
# on top of mcp-ui's own ResizeObserver). Posts the common mcp-ui size message shapes.
_RESIZE_JS = (
    "function _rh(){var h=document.documentElement.scrollHeight;"
    "try{window.parent.postMessage({type:'ui-size-change',payload:{height:h}},'*');}catch(e){}"
    "try{window.parent.postMessage({type:'ui-size-change',height:h},'*');}catch(e){}}"
    "new ResizeObserver(_rh).observe(document.documentElement);"
    "window.addEventListener('load',_rh);setTimeout(_rh,50);"
)

# Posts an MCP-UI `prompt` action back to the host chat (handled by LibreChat's handleUIAction).
_POST = "window.parent.postMessage({type:'prompt',payload:{prompt:P}},'*');"


def _ui(html_body: str, script: str) -> ToolResult:
    """Wrap a component body + script into an MCP-UI resource ToolResult."""
    doc = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<style>{_CSS}</style></head><body><div class="card">{html_body}</div>'
        f"<script>{script}\n{_RESIZE_JS}</script></body></html>"
    )
    resource = EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=AnyUrl("ui://ascent-ui-kit/component"),
            mimeType="text/html",
            text=doc,
        ),
    )
    return ToolResult(content=[resource])


def _head(title: str, description: str) -> str:
    """Render the card heading and optional sub-text."""
    heading = f"<h3>{_html.escape(title)}</h3>"
    if description:
        heading += f'<p class="sub">{_html.escape(description)}</p>'
    return heading


@mcp_ui.tool()
async def ui_select(
    title: str,
    options: list[str],
    description: str = "",
    submit_label: str = "Submit",
    prompt_template: str = "{selection}",
) -> ToolResult:
    """
    Render an interactive single-choice DROPDOWN widget and return the user's pick.

    Use when the user must choose ONE value from a known list (e.g. an OMOP
    database from `get_omop_databases`). Render the returned UI resource inline by
    placing its `\\ui{resource-id}` marker in your reply. When the user submits,
    their choice comes back as a follow-up message for you to act on.

    Args:
        title: Heading shown above the dropdown.
        options: The selectable values.
        description: Optional sub-text under the title.
        submit_label: Button label.
        prompt_template: Text sent back on submit; `{selection}` is replaced with
            the chosen value (e.g. "Use database {selection}.").

    Returns:
        A UI resource to render inline via its `\\ui{resource-id}` marker.
    """
    opts = "".join(f"<option>{_html.escape(o)}</option>" for o in options)
    body = (
        _head(title, description)
        + f'<label>Choose</label><select id="sel">{opts}</select>'
        + f'<button class="primary" id="go">{_html.escape(submit_label)}</button>'
    )
    script = (
        f"var T={json.dumps(prompt_template)};"
        "document.getElementById('go').onclick=function(){"
        "var v=document.getElementById('sel').value;"
        "var P=T.indexOf('{selection}')>=0?T.replace('{selection}',v):(T+' '+v);"
        f"{_POST}}};"
    )
    return _ui(body, script)


@mcp_ui.tool()
async def ui_buttons(
    title: str,
    options: list[str],
    description: str = "",
) -> ToolResult:
    """
    Render a row of CHOICE BUTTONS; clicking one immediately submits it.

    Use for quick one-tap choices (e.g. Yes / No, or a short option set). Render
    the returned UI resource inline by placing its `\\ui{resource-id}` marker. The
    clicked label comes back as a follow-up message for you to act on.

    Args:
        title: Heading shown above the buttons.
        options: One button per value; the clicked label is sent back.
        description: Optional sub-text under the title.

    Returns:
        A UI resource to render inline via its `\\ui{resource-id}` marker.
    """
    btns = "".join(f'<button data-v="{_html.escape(o)}">{_html.escape(o)}</button>' for o in options)
    body = _head(title, description) + f'<div class="btns">{btns}</div>'
    script = f"document.querySelectorAll('.btns button').forEach(function(b){{b.onclick=function(){{var P=b.getAttribute('data-v');{_POST}}};}});"
    return _ui(body, script)


@mcp_ui.tool()
async def ui_form(
    title: str,
    fields: list[dict[str, Any]],
    description: str = "",
    submit_label: str = "Submit",
) -> ToolResult:
    """
    Render a multi-field FORM widget and return the filled values.

    Use when you need several inputs at once. Render the returned UI resource
    inline by placing its `\\ui{resource-id}` marker. On submit, the values come
    back as a readable follow-up message for you to act on.

    Args:
        title: Heading shown above the form.
        fields: Field definitions. Each is a dict:
            - name (str, required): key used in the result
            - label (str): shown to the user (defaults to name)
            - type ("text" | "select"): input kind (default "text")
            - options (list[str]): required when type is "select"
            - placeholder (str): optional, for text inputs
        description: Optional sub-text under the title.
        submit_label: Button label.

    Returns:
        A UI resource to render inline via its `\\ui{resource-id}` marker.
    """
    rows: list[str] = []
    for field in fields:
        name = str(field.get("name", "")).strip()
        if not name:
            continue
        label = _html.escape(str(field.get("label", name)))
        if field.get("type", "text") == "select":
            opts = "".join(f"<option>{_html.escape(str(o))}</option>" for o in field.get("options", []))
            ctrl = f'<select data-name="{_html.escape(name)}">{opts}</select>'
        else:
            placeholder = _html.escape(str(field.get("placeholder", "")))
            ctrl = f'<input data-name="{_html.escape(name)}" placeholder="{placeholder}">'
        rows.append(f"<label>{label}</label>{ctrl}")
    body = _head(title, description) + "".join(rows) + f'<button class="primary" id="go">{_html.escape(submit_label)}</button>'
    script = (
        f"var TITLE={json.dumps(title)};"
        "document.getElementById('go').onclick=function(){"
        "var parts=[];document.querySelectorAll('[data-name]').forEach(function(el){"
        "parts.push(el.getAttribute('data-name')+': '+el.value);});"
        "var P=TITLE+' — '+parts.join('; ');"
        f"{_POST}}};"
    )
    return _ui(body, script)
