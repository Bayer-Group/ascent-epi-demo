"""Unit tests for ascent_domain.literature_agent_service."""

from unittest.mock import AsyncMock, patch

import pytest

from ascent_domain.literature_agent_service import (
    LITERATURE_ONLY_DB_PLACEHOLDER,
    LiteraturePhaseHandle,
    build_country_context,
    combine_with_rwd,
    extract_literature_outputs,
    run_literature_phase,
)

# ---------------------------------------------------------------------------
# Patch targets — patching at the source module path lets the lazy imports
# inside run_literature_phase / combine_with_rwd resolve to the mock.
# ---------------------------------------------------------------------------

_AGENT_CLS = "ascent_domain.omop.models.agents.langgraph.contextual_literature_agent.LiteratureAscentComparisonAgent"
_LIT_PROMPT = "ascent_domain.omop.models.agents.langgraph.prompts_agents.literature_only_prompt"
_GOOGLE_TOOL = "ascent_domain.omop.models.agents.langgraph.google_search_tool.google_search_grounding"


# ---------------------------------------------------------------------------
# build_country_context
# ---------------------------------------------------------------------------


class TestBuildCountryContext:
    def test_empty_when_no_country(self):
        assert build_country_context(None) == ""
        assert build_country_context("") == ""

    def test_includes_country_name(self):
        result = build_country_context("Germany")
        assert "Germany" in result
        assert "IMPORTANT" in result
        assert result.count("Germany") >= 3


# ---------------------------------------------------------------------------
# extract_literature_outputs
# ---------------------------------------------------------------------------


def _arun_result(
    *,
    final_answer="Summary [1].",
    verified=None,
    google=None,
    documents=None,
    agent_state_raw=None,
):
    return {
        "final_answer": final_answer,
        "execution_trace": [],
        "agent_state": {
            "literature_documents": documents,
            "verified_literature_results": verified,
            "google_search_results": google,
        },
        "agent_state_raw": agent_state_raw or {"messages": ["raw1"]},
    }


class TestExtractLiteratureOutputs:
    def test_handles_missing_verified_and_google(self):
        outputs = extract_literature_outputs(_arun_result(verified=None, google=None))
        assert outputs["literature_answers"] == []
        assert outputs["web_search_answers"] == []
        assert outputs["sources"] == []
        assert outputs["literature_documents"] == []

    def test_parses_json_string_final_answer(self):
        outputs = extract_literature_outputs(_arun_result(final_answer='{"k": "v"}'))
        assert outputs["summary"] == {"k": "v"}

    def test_keeps_non_json_string_final_answer_as_text(self):
        outputs = extract_literature_outputs(_arun_result(final_answer="plain text"))
        assert outputs["summary"] == "plain text"

    def test_extracts_literature_answers(self):
        verified = {
            "all_answers": [
                {
                    "question": "Q?",
                    "answer": "A.",
                    "verified_claims_result": [],
                }
            ]
        }
        outputs = extract_literature_outputs(_arun_result(verified=verified))
        assert outputs["literature_answers"][0]["question"] == "Q?"
        assert outputs["verified_literature_results"] is verified

    def test_builds_web_answers_from_google_results(self):
        google = [{"question": "G?", "answer": "Web answer.", "sources": []}]
        outputs = extract_literature_outputs(_arun_result(google=google))
        assert outputs["web_search_answers"][0]["question"] == "G?"

    def test_returns_documents_for_publication_building(self):
        docs = [{"pid": "EMBASE:L1", "title": "Doc"}]
        outputs = extract_literature_outputs(_arun_result(documents=docs))
        assert outputs["literature_documents"] == docs


# ---------------------------------------------------------------------------
# LiteraturePhaseHandle
# ---------------------------------------------------------------------------


class TestLiteraturePhaseHandle:
    def test_state_defaults_to_empty(self):
        handle = LiteraturePhaseHandle(agent=object(), arun_result={})
        assert handle.agent_state == {}
        assert handle.agent_state_raw == {}


# ---------------------------------------------------------------------------
# run_literature_phase
# ---------------------------------------------------------------------------


def _make_mock_agent(arun_return=None):
    agent = AsyncMock()
    agent.arun.return_value = arun_return or _arun_result()
    return agent


@pytest.mark.asyncio
async def test_run_literature_phase_always_overrides_tools_with_web_search():
    """tools_override is set on every path, to web search alone.

    Deliberately unconditional: a path that skipped it would fall back to the
    agent's default tool list, which can reach EHR tools during Phase 1.
    """
    agent = _make_mock_agent()
    with patch(_AGENT_CLS, return_value=agent) as MockAgent, patch(_LIT_PROMPT, "prompt {country_context}"):
        await run_literature_phase(
            question="Q",
            database_name=LITERATURE_ONLY_DB_PLACEHOLDER,
        )

    kwargs = MockAgent.call_args.kwargs
    assert len(kwargs["tools_override"]) == 1
    assert kwargs["database"] == LITERATURE_ONLY_DB_PLACEHOLDER
    assert kwargs["enable_claim_attribution"] is False
    assert kwargs["enable_final_summary_verification"] is False


@pytest.mark.asyncio
async def test_constrain_to_literature_tools_no_longer_changes_the_tool_list():
    """The flag is accepted for call-site compatibility and selects nothing.

    Asserted rather than assumed, because the name suggests it still narrows
    the tool list.
    """
    calls = []
    for constrain in (False, True):
        agent = _make_mock_agent()
        with patch(_AGENT_CLS, return_value=agent) as MockAgent, patch(_LIT_PROMPT, "prompt {country_context}"):
            await run_literature_phase(
                question="Q",
                database_name="REAL_DB",
                constrain_to_literature_tools=constrain,
            )
        calls.append(MockAgent.call_args.kwargs["tools_override"])

    assert calls[0] == calls[1]
    assert len(calls[0]) == 1


@pytest.mark.asyncio
async def test_run_literature_phase_appends_cohort_description():
    agent = _make_mock_agent()
    with patch(_AGENT_CLS, return_value=agent), patch(_LIT_PROMPT, "prompt {country_context}"):
        await run_literature_phase(
            question="Stroke prevalence",
            database_name=LITERATURE_ONLY_DB_PLACEHOLDER,
            cohort_description="Patients aged 65+",
        )

    arun_kwargs = agent.arun.call_args.kwargs
    assert "Stroke prevalence" in arun_kwargs["question"]
    assert "Patients aged 65+" in arun_kwargs["question"]


@pytest.mark.asyncio
async def test_run_literature_phase_passes_country_context_into_prompt():
    agent = _make_mock_agent()
    with patch(_AGENT_CLS, return_value=agent), patch(_LIT_PROMPT, "{country_context}"):
        await run_literature_phase(
            question="Q",
            database_name=LITERATURE_ONLY_DB_PLACEHOLDER,
            country="Germany",
        )

    arun_kwargs = agent.arun.call_args.kwargs
    assert "Germany" in arun_kwargs["custom_prompt"]


@pytest.mark.asyncio
async def test_run_literature_phase_returns_handle_with_state_extracted():
    """The handle pre-extracts agent_state and agent_state_raw with safe defaults."""
    agent = _make_mock_agent(arun_return=_arun_result(agent_state_raw={"raw": "state"}))
    with patch(_AGENT_CLS, return_value=agent), patch(_LIT_PROMPT, "{country_context}"):
        handle = await run_literature_phase(
            question="Q",
            database_name=LITERATURE_ONLY_DB_PLACEHOLDER,
        )

    assert handle.agent is agent
    assert handle.agent_state_raw == {"raw": "state"}
    assert isinstance(handle.agent_state, dict)


# ---------------------------------------------------------------------------
# combine_with_rwd
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_combine_with_rwd_forwards_phase1_state():
    """combine_with_rwd passes phase1_state from the handle's agent_state_raw."""
    agent = AsyncMock()
    agent.combine.return_value = {"final_answer": "Combined narrative"}
    handle = LiteraturePhaseHandle(
        agent=agent,
        arun_result={},
        agent_state_raw={"raw": "phase1_state_payload"},
    )

    result = await combine_with_rwd(
        handle,
        question="Q",
        rwd_answer="RWD answer text",
        database_name="REAL_DB",
    )

    agent.combine.assert_awaited_once_with(
        question="Q",
        main_pipeline_answer="RWD answer text",
        database="REAL_DB",
        phase1_state={"raw": "phase1_state_payload"},
    )
    assert result["final_answer"] == "Combined narrative"
