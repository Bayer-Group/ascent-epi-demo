"""Tests for the ~318s generate_epidemiological_sql_non_omop failure.

The failure was not an infrastructure deadline: it was the 300s
MCP_STEP_BUDGET_SQL_TEMPLATE ceiling firing on step 3 of
the non-OMOP chain. Observed wall-clock = (steps 1+2) + 300s, which is why the
latencies clustered at 311-320s and trailed out to 607s.
"""

import asyncio
from unittest.mock import patch

import pytest

import ascent_domain.models.data_definitions  # noqa: F401  (import-cycle order)
from ascent_mcp.tools._capacity import ServerBusy
from ascent_mcp.tools._keepalive import StepTimeout, progress_keepalive, step_budget
from ascent_mcp.tools_v1_clues import _error_result


class _Ctx:
    async def report_progress(self, *_args, **_kwargs):
        return None


def test_non_omop_sql_budget_is_separate_and_larger():
    """Non-OMOP SQL prep reasons over a full M-schema; 300s was too tight."""
    assert step_budget("SQL_TEMPLATE_NON_OMOP", 480) == 480
    assert step_budget("SQL_TEMPLATE_NON_OMOP", 480) > step_budget("SQL_TEMPLATE", 300)


def test_budget_is_env_overridable():
    with patch.dict("os.environ", {"MCP_STEP_BUDGET_SQL_TEMPLATE_NON_OMOP": "540"}):
        assert step_budget("SQL_TEMPLATE_NON_OMOP", 480) == 540


# --- The error contract that produced KeyError: 'sql_template' -------------


def test_timeout_frame_carries_the_success_keys():
    """A scripted client indexes gen["sql_template"]; the frame must have it."""
    frame = _error_result(
        StepTimeout("Generating SQL template", 480),
        sql_template="",
        entities=[],
        analytical_objective="",
    )

    assert frame["sql_template"] == ""  # no KeyError for the caller
    assert frame["entities"] == []
    assert frame["error"] == "timeout"
    assert frame["retryable"] is True
    # Enough to tell which ceiling was hit, without reading logs.
    assert frame["failed_step"] == "Generating SQL template"
    assert frame["budget_seconds"] == 480


def test_busy_frame_also_carries_the_success_keys():
    frame = _error_result(ServerBusy("busy"), sql_template="", entities=[], analytical_objective="")
    assert frame["sql_template"] == ""
    assert frame["error"] == "server_busy"
    # Not a step timeout, so no step attribution.
    assert "failed_step" not in frame


def test_every_clues_error_path_supplies_a_success_shape():
    """A bare _error_result(e) would reintroduce the KeyError for that tool."""
    import inspect
    import re

    from ascent_mcp import tools_v1_clues

    source = inspect.getsource(tools_v1_clues)
    bare = re.findall(r"_error_result\(\s*e\s*\)", source)
    assert not bare, f"{len(bare)} _error_result call(s) return no success-shape keys"


# --- Step timing instrumentation ------------------------------------------


@pytest.mark.asyncio
async def test_step_timeout_reports_step_and_budget():
    with pytest.raises(StepTimeout) as excinfo:
        async with progress_keepalive(_Ctx(), 3, 4, "Generating SQL template", budget=0.05):
            await asyncio.sleep(5)

    assert excinfo.value.step_message == "Generating SQL template"
    assert excinfo.value.budget_seconds == 0.05


@pytest.mark.asyncio
async def test_step_duration_is_logged_on_success(caplog):
    """Without a duration we cannot tell a tight budget from a wedged step."""
    with caplog.at_level("INFO", logger="ascent_mcp.tools._keepalive"):
        async with progress_keepalive(_Ctx(), 3, 4, "Generating SQL template", budget=10):
            await asyncio.sleep(0.01)

    records = [r for r in caplog.records if "Step finished" in r.message]
    assert records, "no step-completion timing was logged"
    assert getattr(records[0], "duration_ms", None) is not None
    assert getattr(records[0], "mcp_event", None) == "step_complete"


@pytest.mark.asyncio
async def test_step_nearing_its_budget_is_logged_at_warning(caplog):
    """Surfaces a budget about to become a failure, before it does."""
    with caplog.at_level("INFO", logger="ascent_mcp.tools._keepalive"):
        async with progress_keepalive(_Ctx(), 3, 4, "Generating SQL template", budget=0.02):
            await asyncio.sleep(0.015)

    warnings = [r for r in caplog.records if r.levelname == "WARNING" and "Step finished" in r.message]
    assert warnings, "a step past half its budget should warn"
