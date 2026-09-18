"""Unit tests for the compare_rwd_with_literature MCP tool."""

from unittest.mock import AsyncMock, patch

import pytest

from ascent_mcp.tools_v1_clues import compare_rwd_with_literature

SAMPLE_DOCUMENTS = [
    {
        "doc_index": 0,
        "title": "Alteplase for Acute Ischemic Stroke",
        "abstract": "Review of alteplase efficacy.",
        "pid": "EMBASE:L46745164",
        "authors": ["Smith J"],
        "year": "2022",
    },
]

SAMPLE_VERIFIED_LIT = {
    "all_answers": [
        {
            "question": "What is the prevalence?",
            "answer": "Around 2% [1].",
            "verified_claims_result": [
                {
                    "claim": "2% prevalence",
                    "attribution": "LITERATURE_ANSWER",
                    "explanation": "",
                    "references": [
                        {
                            "doc_index": 0,
                            "title": "Alteplase for Acute Ischemic Stroke",
                            "pid": "EMBASE:L46745164",
                            "abstract": "Review of alteplase efficacy.",
                        }
                    ],
                }
            ],
        }
    ]
}


def _make_arun_result(
    *,
    final_answer="Literature summary [1].",
    documents=None,
    verified=None,
    google=None,
    raw_state=None,
):
    return {
        "final_answer": final_answer,
        "execution_trace": [],
        "agent_state": {
            "literature_documents": documents,
            "verified_literature_results": verified,
            "google_search_results": google,
        },
        "agent_state_raw": raw_state or {"messages": ["raw_state"]},
    }


def _mock_ctx():
    return AsyncMock()


def _make_mock_agent(arun_return=None, combine_return=None):
    agent = AsyncMock()
    agent.arun.return_value = arun_return or _make_arun_result()
    agent.combine.return_value = combine_return or {
        "final_answer": "Combined narrative comparing [A1] with [L1] and [G1].",
        "execution_trace": [],
        "agent_state": {},
    }
    return agent


_AGENT_CLS = "ascent_domain.omop.models.agents.langgraph.contextual_literature_agent.LiteratureAscentComparisonAgent"
_LIT_PROMPT = "ascent_domain.omop.models.agents.langgraph.prompts_agents.literature_only_prompt"
_GOOGLE_TOOL = "ascent_domain.omop.models.agents.langgraph.google_search_tool.google_search_grounding"
_PERMISSION = "ascent_mcp.tools_v1_clues.assert_user_permission_db"


@pytest.mark.asyncio
async def test_success_returns_comparison_and_literature():
    ctx = _mock_ctx()
    agent = _make_mock_agent(
        arun_return=_make_arun_result(
            documents=SAMPLE_DOCUMENTS,
            verified=SAMPLE_VERIFIED_LIT,
        ),
    )
    with (
        patch(_PERMISSION, AsyncMock()),
        patch(_AGENT_CLS, return_value=agent),
        patch(_LIT_PROMPT, "{country_context}"),
        patch(_GOOGLE_TOOL, "fake_google_tool"),
    ):
        result = await compare_rwd_with_literature(
            original_question="How many patients have stroke?",
            rwd_answer="Found 1,234 patients.",
            database_name="REAL_DB",
            ctx=ctx,
        )

    assert result["status"] == "success"
    assert result["original_question"] == "How many patients have stroke?"
    assert result["comparison"] == "Combined narrative comparing [A1] with [L1] and [G1]."
    assert result["publication_count"] == 1
    assert result["publications"][0]["pid"] == "EMBASE:L46745164"
    assert len(result["literature_answers"]) == 1


@pytest.mark.asyncio
async def test_permission_failure_propagates():
    ctx = _mock_ctx()
    with patch(_PERMISSION, AsyncMock(side_effect=ValueError("no access"))):
        with pytest.raises(ValueError, match="no access"):
            await compare_rwd_with_literature(
                original_question="Q",
                rwd_answer="A",
                database_name="FORBIDDEN_DB",
                ctx=ctx,
            )


@pytest.mark.asyncio
async def test_agent_constructed_with_tools_override():
    """compare_rwd_with_literature constrains Phase 1 to literature tools only."""
    ctx = _mock_ctx()
    agent = _make_mock_agent()
    with (
        patch(_PERMISSION, AsyncMock()),
        patch(_AGENT_CLS, return_value=agent) as MockAgent,
        patch(_LIT_PROMPT, "{country_context}"),
        patch(_GOOGLE_TOOL, "fake_google_tool"),
    ):
        await compare_rwd_with_literature(
            original_question="Q",
            rwd_answer="RWD",
            database_name="REAL_DB",
            ctx=ctx,
        )

    kwargs = MockAgent.call_args.kwargs
    assert "tools_override" in kwargs
    assert len(kwargs["tools_override"]) == 1
    assert kwargs["database"] == "REAL_DB"


@pytest.mark.asyncio
async def test_combine_called_with_phase1_state_and_rwd_answer():
    """agent.combine receives the original question, RWD answer, db, and phase1_state."""
    ctx = _mock_ctx()
    raw_state = {"messages": ["m1", "m2"]}
    agent = _make_mock_agent(
        arun_return=_make_arun_result(raw_state=raw_state),
    )
    with (
        patch(_PERMISSION, AsyncMock()),
        patch(_AGENT_CLS, return_value=agent),
        patch(_LIT_PROMPT, "{country_context}"),
        patch(_GOOGLE_TOOL, "fake_google_tool"),
    ):
        await compare_rwd_with_literature(
            original_question="Original Q",
            rwd_answer="RWD answer text",
            database_name="REAL_DB",
            ctx=ctx,
        )

    agent.combine.assert_awaited_once_with(
        question="Original Q",
        main_pipeline_answer="RWD answer text",
        database="REAL_DB",
        phase1_state=raw_state,
    )


@pytest.mark.asyncio
async def test_comparison_field_comes_from_combine_final_answer():
    ctx = _mock_ctx()
    agent = _make_mock_agent(
        combine_return={"final_answer": "Custom comparison narrative"},
    )
    with (
        patch(_PERMISSION, AsyncMock()),
        patch(_AGENT_CLS, return_value=agent),
        patch(_LIT_PROMPT, "{country_context}"),
        patch(_GOOGLE_TOOL, "fake_google_tool"),
    ):
        result = await compare_rwd_with_literature(
            original_question="Q",
            rwd_answer="A",
            database_name="REAL_DB",
            ctx=ctx,
        )

    assert result["comparison"] == "Custom comparison narrative"


@pytest.mark.asyncio
async def test_progress_reported():
    """Tool reports progress at every step (5 steps + step 0)."""
    ctx = _mock_ctx()
    agent = _make_mock_agent()
    with (
        patch(_PERMISSION, AsyncMock()),
        patch(_AGENT_CLS, return_value=agent),
        patch(_LIT_PROMPT, "{country_context}"),
        patch(_GOOGLE_TOOL, "fake_google_tool"),
    ):
        await compare_rwd_with_literature(
            original_question="Q",
            rwd_answer="A",
            database_name="REAL_DB",
            ctx=ctx,
        )

    # Steps 0..5 are reported; keepalive may add more.
    assert ctx.report_progress.await_count >= 6


@pytest.mark.asyncio
async def test_empty_phase1_results_still_runs_combine():
    """If Phase 1 returns no literature, comparison still runs (RWD-only narrative)."""
    ctx = _mock_ctx()
    agent = _make_mock_agent(
        arun_return=_make_arun_result(documents=[], verified=None, google=None),
    )
    with (
        patch(_PERMISSION, AsyncMock()),
        patch(_AGENT_CLS, return_value=agent),
        patch(_LIT_PROMPT, "{country_context}"),
        patch(_GOOGLE_TOOL, "fake_google_tool"),
    ):
        result = await compare_rwd_with_literature(
            original_question="Q",
            rwd_answer="A",
            database_name="REAL_DB",
            ctx=ctx,
        )

    assert result["status"] == "success"
    assert result["publications"] == []
    assert result["literature_answers"] == []
    agent.combine.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_exception_propagates():
    ctx = _mock_ctx()
    agent = AsyncMock()
    agent.arun.side_effect = RuntimeError("literature search down")
    with (
        patch(_PERMISSION, AsyncMock()),
        patch(_AGENT_CLS, return_value=agent),
        patch(_LIT_PROMPT, "{country_context}"),
        patch(_GOOGLE_TOOL, "fake_google_tool"),
    ):
        with pytest.raises(RuntimeError, match="literature search down"):
            await compare_rwd_with_literature(
                original_question="Q",
                rwd_answer="A",
                database_name="REAL_DB",
                ctx=ctx,
            )
