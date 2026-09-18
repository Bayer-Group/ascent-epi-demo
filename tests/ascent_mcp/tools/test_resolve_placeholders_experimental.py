"""Unit tests for the experimental ``resolve_placeholders`` tool.

Mirrors ``test_resolve_placeholders_v1.py`` — the experimental MCP server
now supports both placeholder dialects via the same routing logic.
"""

from unittest.mock import AsyncMock, patch

import pytest

# See test_resolve_placeholders_v1.py for the circular-import rationale.
import ascent_domain.models.data_definitions  # noqa: F401, E402
from ascent_mcp.tools_experimental import (  # noqa: E402
    OMOP_PLACEHOLDER_PATTERN,
    PLACEHOLDER_PATTERN,
    list_placeholder_concepts,
    resolve_placeholders,
)


def _ctx():
    ctx = AsyncMock()
    ctx.report_progress = AsyncMock()
    return ctx


# ─── placeholder-pattern routing ──────────────────────────────────────────────


def test_three_segment_pattern_matches_three_only():
    assert PLACEHOLDER_PATTERN.findall("x IN ([condition@diabetes@ICD10CM,ICD9CM])") == [("condition", "diabetes", "ICD10CM,ICD9CM")]
    assert PLACEHOLDER_PATTERN.findall("x IN ([condition@diabetes])") == []


def test_omop_pattern_matches_two_only():
    assert OMOP_PLACEHOLDER_PATTERN.findall("x IN ([drug_class@statins])") == [("drug_class", "statins")]
    assert OMOP_PLACEHOLDER_PATTERN.findall("x IN ([condition@diabetes@ICD10CM])") == []


def test_two_adjacent_omop_do_not_bridge_into_false_three_segment():
    sql = "a IN ([drug_class@statins]) AND b IN ([condition@chronic kidney disease])"
    assert PLACEHOLDER_PATTERN.findall(sql) == []
    assert OMOP_PLACEHOLDER_PATTERN.findall(sql) == [
        ("drug_class", "statins"),
        ("condition", "chronic kidney disease"),
    ]


def test_mixed_sql_routes_each_format_once():
    mixed = "a IN ([drug_class@statins]) AND b IN ([condition@diabetes@ICD10CM])"
    assert PLACEHOLDER_PATTERN.findall(mixed) == [("condition", "diabetes", "ICD10CM")]
    assert OMOP_PLACEHOLDER_PATTERN.findall(mixed) == [("drug_class", "statins")]


# ─── resolve_placeholders: no-op when nothing to resolve ─────────────────────


async def test_no_placeholders_returns_unchanged():
    sql = "SELECT 1"
    result = await resolve_placeholders(sql, _ctx())
    assert result["resolved_sql"] == sql
    assert "No placeholders found" in result["message"]


# ─── resolve_placeholders: 3-segment non-OMOP path ───────────────────────────


async def test_three_segment_resolved_with_concept_ids():
    sql = "WHERE condition_concept_id IN ([condition@diabetes@ICD10CM])"
    fake_entities = [
        {
            "placeholder": "condition@diabetes@ICD10CM",
            "concepts": [{"CONCEPT_ID": 201826, "CONCEPT_CODE": "E11"}],
            "codes_str": "'E11'",
        }
    ]
    with patch(
        "ascent_mcp._tools_shared.get_medical_concepts_from_sql_async",
        return_value=fake_entities,
    ):
        result = await resolve_placeholders(sql, _ctx(), database="SYNTHETIC_CLAIMS", use_concept_ids=True)
    assert "201826" in result["resolved_sql"]
    assert result["placeholders_resolved"] == 1
    assert result["resolutions"][0]["format"] == "non-omop"
    assert result["resolutions"][0]["code_count"] == 1


async def test_three_segment_with_no_codes_uses_source_sentinel():
    sql = "x IN ([condition@xyz@ICD10CM])"
    fake_entities = [{"placeholder": "condition@xyz@ICD10CM", "concepts": [], "codes_str": ""}]
    with patch(
        "ascent_mcp._tools_shared.get_medical_concepts_from_sql_async",
        return_value=fake_entities,
    ):
        result = await resolve_placeholders(sql, _ctx(), database="SYNTHETIC_CLAIMS")
    assert "__NO_CODES__" in result["resolved_sql"]
    assert result["resolutions"][0]["code_count"] == 0


async def test_non_omop_forwards_database_to_coder():
    """``database`` is forwarded to the coder so the library can apply IS_WITH_DOTS."""
    sql = "x IN ([condition@diabetes@ICD10CM])"
    fake_entities = [{"placeholder": "condition@diabetes@ICD10CM", "concepts": [], "codes_str": ""}]
    with patch(
        "ascent_mcp._tools_shared.get_medical_concepts_from_sql_async",
        return_value=fake_entities,
    ) as coder:
        await resolve_placeholders(sql, _ctx(), database="SYNTHETIC_CLAIMS", use_concept_ids=False)
    assert coder.await_args.kwargs.get("database") == "SYNTHETIC_CLAIMS"


async def test_non_omop_requires_database():
    """Resolving 3-segment placeholders without a database is rejected."""
    sql = "x IN ([condition@diabetes@ICD10CM])"
    with pytest.raises(ValueError, match="`database` is required"):
        await resolve_placeholders(sql, _ctx())


# ─── resolve_placeholders: 2-segment OMOP path ───────────────────────────────


async def test_omop_placeholders_require_database():
    sql = "WHERE condition_concept_id IN ([condition@chronic kidney disease])"
    with pytest.raises(ValueError, match="`database` is required"):
        await resolve_placeholders(sql, _ctx())


async def test_omop_placeholders_resolved_via_qa_service():
    sql = "WHERE drug_concept_id IN ([drug_class@statins]) AND condition_concept_id IN ([condition@chronic kidney disease])"
    resolved = "WHERE drug_concept_id IN (1539403) AND condition_concept_id IN (46271022)"

    qa_service = AsyncMock()
    qa_service.post_process_query = AsyncMock(return_value=resolved)
    with (
        patch(
            "ascent_mcp.tools._clues_helpers.init_qa_system",
            AsyncMock(return_value=(qa_service, "DB")),
        ),
        patch("ascent_mcp._tools_shared.assert_user_permission_db", AsyncMock()),
    ):
        result = await resolve_placeholders(sql, _ctx(), database="SYNTHETIC_EHR_OMOP")

    assert result["resolved_sql"] == resolved
    assert result["placeholders_resolved"] == 2
    assert {r["placeholder"] for r in result["resolutions"]} == {
        "[drug_class@statins]",
        "[condition@chronic kidney disease]",
    }
    assert all(r["format"] == "omop" and r["resolved"] for r in result["resolutions"])
    qa_service.post_process_query.assert_awaited_once()


async def test_omop_coding_system_source_forwarded_to_post_process_query():
    from ascent_domain.omop.schemas.constants import CodingType

    sql = "WHERE condition_source_concept_id IN ([condition@chronic kidney disease])"

    qa_service = AsyncMock()
    qa_service.post_process_query = AsyncMock(return_value="WHERE … IN (44831337)")
    with (
        patch(
            "ascent_mcp.tools._clues_helpers.init_qa_system",
            AsyncMock(return_value=(qa_service, "DB")),
        ),
        patch("ascent_mcp._tools_shared.assert_user_permission_db", AsyncMock()),
    ):
        await resolve_placeholders(sql, _ctx(), database="SYNTHETIC_EHR_OMOP", coding_system="Source")

    kwargs = qa_service.post_process_query.call_args.kwargs
    assert kwargs["preferred_coding_system"] is CodingType.SOURCE_CODING


async def test_omop_concept_not_found_reports_warning_and_keeps_sql():
    from ascent_domain.omop.data.processing.sql_post_processor import ConceptNotFoundError

    sql = "WHERE condition_concept_id IN ([condition@nonexistent])"

    qa_service = AsyncMock()
    qa_service.post_process_query = AsyncMock(side_effect=ConceptNotFoundError("condition", "nonexistent"))
    with (
        patch(
            "ascent_mcp.tools._clues_helpers.init_qa_system",
            AsyncMock(return_value=(qa_service, "DB")),
        ),
        patch("ascent_mcp._tools_shared.assert_user_permission_db", AsyncMock()),
    ):
        result = await resolve_placeholders(sql, _ctx(), database="SYNTHETIC_EHR_OMOP")

    assert result["resolved_sql"] == sql
    assert result["resolutions"][0]["resolved"] is False
    assert "warning" in result["resolutions"][0]


# ─── resolve_placeholders: mixed (both formats in one SQL) ───────────────────


async def test_mixed_sql_resolves_both_paths():
    sql = "a IN ([drug@metformin@NDC]) AND b IN ([condition@chronic kidney disease])"
    fake_entities = [
        {
            "placeholder": "drug@metformin@NDC",
            "concepts": [{"CONCEPT_ID": 11, "CONCEPT_CODE": "11"}],
            "codes_str": "'11'",
        }
    ]
    qa_service = AsyncMock()
    qa_service.post_process_query = AsyncMock(return_value="a IN (11) AND b IN (46271022)")
    with (
        patch(
            "ascent_mcp._tools_shared.get_medical_concepts_from_sql_async",
            return_value=fake_entities,
        ),
        patch(
            "ascent_mcp.tools._clues_helpers.init_qa_system",
            AsyncMock(return_value=(qa_service, "DB")),
        ),
        patch("ascent_mcp._tools_shared.assert_user_permission_db", AsyncMock()),
    ):
        result = await resolve_placeholders(sql, _ctx(), database="SYNTHETIC_EHR_OMOP")

    assert result["placeholders_resolved"] == 2
    assert sorted(r["format"] for r in result["resolutions"]) == ["non-omop", "omop"]


# ─── list_placeholder_concepts ───────────────────────────────────────────────


async def test_list_lists_three_segment():
    result = await list_placeholder_concepts("a IN ([condition@diabetes@ICD10CM,ICD9CM])")
    assert result["placeholder_count"] == 1
    p = result["placeholders"][0]
    assert p["format"] == "non-omop"
    assert p["entity_type"] == "condition"
    assert p["coding_systems"] == "ICD10CM,ICD9CM"


async def test_list_lists_two_segment_omop():
    result = await list_placeholder_concepts("a IN ([condition@chronic kidney disease])")
    assert result["placeholder_count"] == 1
    p = result["placeholders"][0]
    assert p["format"] == "omop"
    assert p["entity_name"] == "chronic kidney disease"
    assert p["coding_systems"] is None


async def test_list_lists_both_formats():
    sql = "a IN ([drug_class@statins]) AND b IN ([condition@diabetes@ICD10CM])"
    result = await list_placeholder_concepts(sql)
    assert result["placeholder_count"] == 2
    assert sorted(p["format"] for p in result["placeholders"]) == ["non-omop", "omop"]


async def test_list_lists_empty():
    result = await list_placeholder_concepts("SELECT 1")
    assert result["placeholder_count"] == 0
