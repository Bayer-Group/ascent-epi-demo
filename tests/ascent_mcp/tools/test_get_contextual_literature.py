"""Unit tests for the get_contextual_literature MCP tool."""

from unittest.mock import AsyncMock, patch

import pytest

from ascent_domain.literature_result_utils import (
    build_publications as _build_publications,
)
from ascent_domain.literature_result_utils import (
    extract_source_db as _extract_source_db,
)
from ascent_mcp.tools_v1_clues import (
    get_contextual_literature,
)

# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------

SAMPLE_VERIFIED_LIT = {
    "all_answers": [
        {
            "question": "What is the prevalence of acute ischemic stroke?",
            "answer": "Studies show prevalence of 2-3% [1].",
            "verified_claims_result": [
                {
                    "claim": "Prevalence is 2-3%.",
                    "attribution": "LITERATURE_ANSWER",
                    "explanation": "Direct literature search result",
                    "references": [
                        {
                            "doc_index": 0,
                            "title": "Alteplase for Acute Ischemic Stroke: A Systematic Review",
                            "pid": "EMBASE:L46745164",
                            "abstract": "This review examines the efficacy of alteplase...",
                        }
                    ],
                }
            ],
        },
        {
            "question": "What are the treatment options for acute ischemic stroke?",
            "answer": "Alteplase is the primary thrombolytic [1][2].",
            "verified_claims_result": [
                {
                    "claim": "Alteplase is primary thrombolytic.",
                    "attribution": "LITERATURE_ANSWER",
                    "explanation": "Direct literature search result",
                    "references": [
                        {
                            "doc_index": 0,
                            "title": "Alteplase for Acute Ischemic Stroke: A Systematic Review",
                            "pid": "EMBASE:L46745164",
                            "abstract": "This review examines the efficacy of alteplase...",
                        },
                        {
                            "doc_index": 1,
                            "title": "Thrombolysis in Stroke: An Updated Meta-Analysis",
                            "pid": "PUBMED:34567890",
                            "abstract": "Meta-analysis of thrombolytic therapy outcomes...",
                        },
                    ],
                }
            ],
        },
    ]
}

SAMPLE_GOOGLE_RESULTS = [
    {
        "question": "Acute ischemic stroke prevalence globally",
        "answer": "Globally, stroke affects 13.7 million people yearly [1].",
        "sources": [
            {"uri": "https://who.int/stroke", "title": "WHO Stroke Fact Sheet"},
        ],
    }
]


def _make_agent_result(
    *,
    final_answer: str = "Literature summary with [L1] references.",
    literature_documents: list[dict] | None = None,
    verified_literature_results: dict | None = None,
    google_search_results: list[dict] | None = None,
) -> dict:
    """Build a mock return value for LiteratureAscentComparisonAgent.arun()."""
    return {
        "final_answer": final_answer,
        "execution_trace": [{"role": "user", "content": "question"}],
        "agent_state": {
            "literature_documents": literature_documents,
            "literature_claims": None,
            "refined_summary": None,
            "verified_literature_results": verified_literature_results,
            "google_search_results": google_search_results,
            "ehr_answers": None,
            "unified_documents": None,
            "reference_mapping": None,
            "final_summary_text": None,
            "final_summary_claims": None,
            "final_summary_refined": None,
        },
    }


SAMPLE_DOCUMENTS = [
    {
        "doc_index": 0,
        "title": "Alteplase for Acute Ischemic Stroke: A Systematic Review",
        "abstract": "This review examines the efficacy of alteplase in acute ischemic stroke...",
        "text": "This review examines the efficacy of alteplase in acute ischemic stroke...",
        "pid": "EMBASE:L46745164",
        "authors": ["Smith J", "Doe A"],
        "year": "2022",
        "source": "Lancet Neurology",
    },
    {
        "doc_index": 1,
        "title": "Thrombolysis in Stroke: An Updated Meta-Analysis",
        "abstract": "Meta-analysis of thrombolytic therapy outcomes...",
        "text": "Meta-analysis of thrombolytic therapy outcomes...",
        "pid": "PUBMED:34567890",
        "authors": ["Jones B"],
        "year": "2023",
        "source": "Stroke",
    },
]


# ---------------------------------------------------------------------------
# Pure-function tests (no mocking needed)
# ---------------------------------------------------------------------------


class TestExtractSourceDb:
    def test_extracts_embase(self):
        assert _extract_source_db("EMBASE:L46745164") == "EMBASE"

    def test_extracts_pubmed(self):
        assert _extract_source_db("PUBMED:34567890") == "PUBMED"

    def test_returns_unknown_for_no_colon(self):
        assert _extract_source_db("unknown_format") == "Unknown"


class TestBuildPublications:
    def test_builds_from_documents(self):
        pubs = _build_publications(SAMPLE_DOCUMENTS)
        assert len(pubs) == 2
        assert pubs[0]["title"] == "Alteplase for Acute Ischemic Stroke: A Systematic Review"
        assert pubs[0]["source"] == "EMBASE"
        assert pubs[0]["publication_year"] == "2022"
        assert pubs[0]["pid"] == "EMBASE:L46745164"
        # None by design: a pid-only record has nothing to link to.
        assert pubs[0]["url"] is None
        assert pubs[0]["authors"] == ["Smith J", "Doe A"]

    def test_returns_empty_for_none(self):
        assert _build_publications(None) == []

    def test_returns_empty_for_empty_list(self):
        assert _build_publications([]) == []

    def test_deduplicates_by_pid(self):
        dup_docs = [SAMPLE_DOCUMENTS[0], SAMPLE_DOCUMENTS[0].copy()]
        pubs = _build_publications(dup_docs)
        assert len(pubs) == 1

    def test_handles_missing_fields_gracefully(self):
        minimal = [{"doc_index": 0}]
        pubs = _build_publications(minimal)
        assert len(pubs) == 1
        assert pubs[0]["title"] == "Untitled"
        assert pubs[0]["source"] == "Unknown"
        assert pubs[0]["publication_year"] is None


# ---------------------------------------------------------------------------
# Tool integration tests (mock the agent)
# ---------------------------------------------------------------------------


_AGENT_CLS = "ascent_domain.omop.models.agents.langgraph.contextual_literature_agent.LiteratureAscentComparisonAgent"
_LIT_PROMPT = "ascent_domain.omop.models.agents.langgraph.prompts_agents.literature_only_prompt"


def _make_mock_ctx():
    """Create a mock FastMCP Context with async report_progress."""
    return AsyncMock()


def _make_mock_agent(arun_return_value):
    """Create a mock agent instance with the given arun return value."""
    agent_instance = AsyncMock()
    agent_instance.arun.return_value = arun_return_value
    return agent_instance


@pytest.mark.asyncio
async def test_success_with_publications():
    """Tool returns publications and summary on a successful search."""
    ctx = _make_mock_ctx()
    agent_instance = _make_mock_agent(
        _make_agent_result(
            literature_documents=SAMPLE_DOCUMENTS,
        )
    )

    with patch(_AGENT_CLS, return_value=agent_instance) as MockAgent, patch(_LIT_PROMPT, "mock prompt {country_context}"):
        result = await get_contextual_literature(
            research_question="How many patients have acute ischemic stroke treated with alteplase?",
            ctx=ctx,
        )

    assert result["status"] == "success"
    assert result["publication_count"] == 2
    assert len(result["publications"]) == 2
    assert result["publications"][0]["title"] == "Alteplase for Acute Ischemic Stroke: A Systematic Review"
    assert result["publications"][0]["source"] == "EMBASE"
    assert result["summary"] == "Literature summary with [L1] references."
    assert result["research_question"] == "How many patients have acute ischemic stroke treated with alteplase?"

    # Verify agent was constructed with literature-only settings
    MockAgent.assert_called_once()
    call_kwargs = MockAgent.call_args.kwargs
    assert call_kwargs["enable_claim_attribution"] is False
    assert call_kwargs["enable_final_summary_verification"] is False


@pytest.mark.asyncio
async def test_returns_literature_answers_from_verified_results():
    """Tool returns per-question literature answers in literature_answers."""
    ctx = _make_mock_ctx()
    agent_instance = _make_mock_agent(
        _make_agent_result(
            literature_documents=SAMPLE_DOCUMENTS,
            verified_literature_results=SAMPLE_VERIFIED_LIT,
        )
    )

    with patch(_AGENT_CLS, return_value=agent_instance), patch(_LIT_PROMPT, "mock prompt {country_context}"):
        result = await get_contextual_literature(
            research_question="Stroke treatment options",
            ctx=ctx,
        )

    assert result["status"] == "success"
    assert len(result["literature_answers"]) == 2
    assert result["literature_answers"][0]["question"] == "What is the prevalence of acute ischemic stroke?"
    assert "answer" in result["literature_answers"][0]
    assert "verified_claims_result" in result["literature_answers"][0]


@pytest.mark.asyncio
async def test_returns_web_search_answers_from_google_results():
    """Tool returns per-question Google answers in web_search_answers."""
    ctx = _make_mock_ctx()
    agent_instance = _make_mock_agent(
        _make_agent_result(
            literature_documents=SAMPLE_DOCUMENTS,
            google_search_results=SAMPLE_GOOGLE_RESULTS,
        )
    )

    with patch(_AGENT_CLS, return_value=agent_instance), patch(_LIT_PROMPT, "mock prompt {country_context}"):
        result = await get_contextual_literature(
            research_question="Stroke prevalence globally",
            ctx=ctx,
        )

    assert result["status"] == "success"
    assert len(result["web_search_answers"]) == 1
    assert result["web_search_answers"][0]["question"] == "Acute ischemic stroke prevalence globally"
    assert "answer" in result["web_search_answers"][0]


@pytest.mark.asyncio
async def test_google_references_normalised_to_google_links():
    """[N] citation markers in Google answers are replaced with [G#](url) links."""
    ctx = _make_mock_ctx()
    google_results = [
        {
            "question": "Stroke statistics",
            "answer": "Stroke affects millions yearly [1].",
            "sources": [{"uri": "https://who.int/stroke", "title": "WHO Stroke"}],
        }
    ]
    agent_instance = _make_mock_agent(
        _make_agent_result(
            literature_documents=[],
            google_search_results=google_results,
        )
    )

    with patch(_AGENT_CLS, return_value=agent_instance), patch(_LIT_PROMPT, "mock prompt {country_context}"):
        result = await get_contextual_literature(
            research_question="Stroke statistics",
            ctx=ctx,
        )

    web_answers = result["web_search_answers"]
    assert len(web_answers) == 1
    # [1] should have been replaced with [G1](url)
    assert "[1]" not in web_answers[0]["answer"]
    assert "G1" in web_answers[0]["answer"]


@pytest.mark.asyncio
async def test_empty_literature_answers_when_no_verified_results():
    """literature_answers is empty when verified_literature_results is absent."""
    ctx = _make_mock_ctx()
    agent_instance = _make_mock_agent(
        _make_agent_result(
            literature_documents=SAMPLE_DOCUMENTS,
            verified_literature_results=None,
        )
    )

    with patch(_AGENT_CLS, return_value=agent_instance), patch(_LIT_PROMPT, "mock prompt {country_context}"):
        result = await get_contextual_literature(
            research_question="Any question",
            ctx=ctx,
        )

    assert result["literature_answers"] == []


@pytest.mark.asyncio
async def test_empty_web_answers_when_no_google_results():
    """web_search_answers is empty when google_search_results is absent."""
    ctx = _make_mock_ctx()
    agent_instance = _make_mock_agent(
        _make_agent_result(
            literature_documents=SAMPLE_DOCUMENTS,
            google_search_results=None,
        )
    )

    with patch(_AGENT_CLS, return_value=agent_instance), patch(_LIT_PROMPT, "mock prompt {country_context}"):
        result = await get_contextual_literature(
            research_question="Any question",
            ctx=ctx,
        )

    assert result["web_search_answers"] == []
    assert result["sources"] == []


@pytest.mark.asyncio
async def test_no_results_returned():
    """Tool returns no_results status when no literature is found."""
    ctx = _make_mock_ctx()
    agent_instance = _make_mock_agent(
        _make_agent_result(
            final_answer="",
            literature_documents=[],
        )
    )

    with patch(_AGENT_CLS, return_value=agent_instance), patch(_LIT_PROMPT, "mock prompt {country_context}"):
        result = await get_contextual_literature(
            research_question="A very niche question with no literature",
            ctx=ctx,
        )

    assert result["status"] == "no_results"
    assert result["publication_count"] == 0
    assert result["publications"] == []
    assert result["literature_answers"] == []
    assert result["web_search_answers"] == []
    assert result["sources"] == []
    assert "No relevant literature" in result["summary"]


@pytest.mark.asyncio
async def test_cohort_description_appended_to_question():
    """When cohort_description is provided, it's appended to the research question."""
    ctx = _make_mock_ctx()
    agent_instance = _make_mock_agent(
        _make_agent_result(
            literature_documents=SAMPLE_DOCUMENTS[:1],
        )
    )

    with patch(_AGENT_CLS, return_value=agent_instance), patch(_LIT_PROMPT, "mock prompt {country_context}"):
        await get_contextual_literature(
            research_question="Stroke treatment outcomes",
            ctx=ctx,
            cohort_description="Patients aged 65+ with acute ischemic stroke",
        )

    # Verify the effective question includes cohort context
    call_kwargs = agent_instance.arun.call_args.kwargs
    assert "Stroke treatment outcomes" in call_kwargs["question"]
    assert "Patients aged 65+ with acute ischemic stroke" in call_kwargs["question"]


@pytest.mark.asyncio
async def test_agent_exception_propagates():
    """Exceptions from the agent propagate to the caller."""
    ctx = _make_mock_ctx()
    agent_instance = AsyncMock()
    agent_instance.arun.side_effect = Exception("literature service unavailable")

    with patch(_AGENT_CLS, return_value=agent_instance), patch(_LIT_PROMPT, "mock prompt {country_context}"):
        with pytest.raises(Exception, match="literature service unavailable"):
            await get_contextual_literature(
                research_question="Any question",
                ctx=ctx,
            )


@pytest.mark.asyncio
async def test_json_final_answer_is_parsed():
    """When final_answer is a JSON string, it's parsed into a dict."""
    ctx = _make_mock_ctx()
    agent_instance = _make_mock_agent(
        _make_agent_result(
            final_answer='{"key": "value"}',
            literature_documents=SAMPLE_DOCUMENTS[:1],
        )
    )

    with patch(_AGENT_CLS, return_value=agent_instance), patch(_LIT_PROMPT, "mock prompt {country_context}"):
        result = await get_contextual_literature(
            research_question="Any question",
            ctx=ctx,
        )

    assert result["summary"] == {"key": "value"}


@pytest.mark.asyncio
async def test_country_context_injected_into_prompt():
    """When country is provided, country-specific guidance is injected into the prompt."""
    ctx = _make_mock_ctx()
    agent_instance = _make_mock_agent(
        _make_agent_result(
            literature_documents=SAMPLE_DOCUMENTS[:1],
        )
    )

    with patch(_AGENT_CLS, return_value=agent_instance), patch(_LIT_PROMPT, "{country_context}"):
        await get_contextual_literature(
            research_question="Stroke prevalence",
            ctx=ctx,
            country="Germany",
        )

    call_kwargs = agent_instance.arun.call_args.kwargs
    assert "Germany" in call_kwargs["custom_prompt"]


@pytest.mark.asyncio
async def test_no_country_context_when_country_not_provided():
    """When country is omitted, no country-specific guidance is added to the prompt."""
    ctx = _make_mock_ctx()
    agent_instance = _make_mock_agent(
        _make_agent_result(
            literature_documents=SAMPLE_DOCUMENTS[:1],
        )
    )

    with patch(_AGENT_CLS, return_value=agent_instance), patch(_LIT_PROMPT, "{country_context}"):
        await get_contextual_literature(
            research_question="Stroke prevalence",
            ctx=ctx,
        )

    call_kwargs = agent_instance.arun.call_args.kwargs
    assert call_kwargs["custom_prompt"].strip() == ""


@pytest.mark.asyncio
async def test_progress_reported():
    """Tool reports progress via ctx.report_progress."""
    ctx = _make_mock_ctx()
    agent_instance = _make_mock_agent(
        _make_agent_result(
            literature_documents=SAMPLE_DOCUMENTS[:1],
        )
    )

    with patch(_AGENT_CLS, return_value=agent_instance), patch(_LIT_PROMPT, "mock prompt {country_context}"):
        await get_contextual_literature(
            research_question="Any question",
            ctx=ctx,
        )

    # At minimum, progress should be reported at start and end
    assert ctx.report_progress.await_count >= 4
