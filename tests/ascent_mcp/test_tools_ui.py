"""Unit tests for the UI Kit MCP tools (ui_select / ui_buttons / ui_form).

These tools are pure presentation: they format a self-contained ``ui://`` HTML
resource from their arguments. The tests assert the MCP-UI contract (resource
shape, the ``prompt`` back-channel, the prompt template) and that all
user-supplied text is HTML-escaped.
"""

import pytest

from ascent_mcp.tools_ui import ui_buttons, ui_form, ui_select


async def _call(tool, /, **kwargs):
    """Invoke a tool whether it's a bare coroutine fn or a FastMCP-wrapped Tool."""
    fn = getattr(tool, "fn", None) or tool
    return await fn(**kwargs)


def _resource(result):
    """Extract the single embedded UI resource from a ToolResult."""
    assert len(result.content) == 1
    return result.content[0].resource


@pytest.mark.asyncio
async def test_ui_select_returns_ui_resource_with_options_and_back_channel():
    result = await _call(
        ui_select,
        title="Select an OMOP database",
        options=["SYNTHETIC_EHR_OMOP", "CDM_OMOP"],
        description="Choose the database to build your cohort in.",
        submit_label="Use database",
        prompt_template="Use database {selection}.",
    )
    resource = _resource(result)

    assert str(resource.uri).startswith("ui://")
    assert resource.mimeType == "text/html"
    assert "<select" in resource.text
    assert "SYNTHETIC_EHR_OMOP" in resource.text
    assert "CDM_OMOP" in resource.text
    assert "Use database" in resource.text
    # Prompt template (with the {selection} placeholder) is embedded for the click handler.
    assert "Use database {selection}." in resource.text
    # The submit posts an MCP-UI `prompt` action back to the host.
    assert "type:'prompt'" in resource.text


@pytest.mark.asyncio
async def test_ui_buttons_renders_one_button_per_option():
    result = await _call(ui_buttons, title="Proceed?", options=["Yes", "No"])
    text = _resource(result).text

    assert 'data-v="Yes"' in text
    assert 'data-v="No"' in text
    assert "type:'prompt'" in text


@pytest.mark.asyncio
async def test_ui_form_renders_text_and_select_fields():
    result = await _call(
        ui_form,
        title="New cohort",
        fields=[
            {"name": "cohort_name", "label": "Cohort name", "placeholder": "e.g. T2D adults"},
            {"name": "db", "label": "Database", "type": "select", "options": ["A_OMOP", "B_OMOP"]},
        ],
    )
    text = _resource(result).text

    assert '<input data-name="cohort_name"' in text
    assert '<select data-name="db"' in text
    assert "A_OMOP" in text and "B_OMOP" in text


@pytest.mark.asyncio
async def test_ui_form_skips_fields_without_a_name():
    result = await _call(
        ui_form,
        title="Partial",
        fields=[{"label": "no name here"}, {"name": "kept"}],
    )
    text = _resource(result).text

    assert 'data-name="kept"' in text
    assert "no name here" not in text


@pytest.mark.asyncio
async def test_user_supplied_text_is_html_escaped():
    result = await _call(
        ui_buttons,
        title="<script>alert(1)</script>",
        options=["<img src=x onerror=alert(1)>"],
    )
    # Exclude the static <style> block; only inspect the rendered body.
    body = _resource(result).text.split("</style>", 1)[1]

    assert "<script>alert(1)</script>" not in body
    assert "<img src=x onerror=alert(1)>" not in body
    assert "&lt;script&gt;" in body
    assert "&lt;img" in body
