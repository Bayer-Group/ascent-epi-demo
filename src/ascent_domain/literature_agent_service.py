"""Shared service for literature search + literature/RWD comparison.

Wraps :class:`ascent_domain.omop.LiteratureAscentComparisonAgent` (Phase 1 ``arun`` and
Phase 2 ``combine``) for reuse across:

- ``ascent_mcp.tools.get_contextual_literature`` (Phase 1 only),
- ``ascent_mcp.tools.compare_rwd_with_literature`` (Phase 1 + Phase 2),
- ``ascent_http.services.clues_ws_service.run_literature_agent_pipeline`` (uses the
  pure helpers; the WS choreography keeps its own agent calls inline).

The agent is lazily imported inside functions so importing this module does
not drag LangGraph/Gemini into every router.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ascent_domain.config import get_domain_settings
from ascent_domain.literature_result_utils import (
    build_google_answers,
    build_sources,
    normalize_google_references,
    normalize_literature_references,
)

logger = logging.getLogger(__name__)


# Placeholder database name used by callers that only need the literature
# phase (no RWD/EHR queries). The literature_only_prompt never triggers EHR
# tools so the value is unused at runtime — the agent constructor just needs
# a non-empty string.
LITERATURE_ONLY_DB_PLACEHOLDER = "LITERATURE_ONLY"


def build_country_context(country: str | None) -> str:
    """Build the country-specific guidance block for ``literature_only_prompt``.

    When *country* is provided the agent is instructed to include at least one
    country-specific sub-question alongside global ones. When *country* is
    ``None`` or empty, an empty string is returned (the prompt template expects
    a string substitution).
    """
    if not country:
        return ""
    return (
        f"\n    IMPORTANT: The database being queried contains data from {country}.\n"
        f"    When formulating literature questions:\n"
        f"     - Include at least one question specific to {country} "
        f'(e.g., "prevalence of X in {country}")\n'
        f"     - Include at least one general/global question for broader context\n"
        f"     - Prioritize {country}-specific data when available"
    )


@dataclass
class LiteraturePhaseHandle:
    """Result of :func:`run_literature_phase`.

    Holds the agent instance and the raw ``arun`` result, with the two pieces
    of agent state pre-extracted (with safe ``{}`` defaults). The agent must
    be retained because Phase 2 (``combine_with_rwd``) reuses it.

    .. note::
        Single-use after Phase 2: ``agent.combine()`` mutates ``self.database``
        and may rebuild the agent's tool graph. Do not reuse the handle's
        agent for further calls once :func:`combine_with_rwd` has run.
    """

    agent: Any
    arun_result: dict[str, Any]
    agent_state: dict[str, Any] = field(default_factory=dict)
    agent_state_raw: dict[str, Any] = field(default_factory=dict)


async def run_literature_phase(
    *,
    question: str,
    database_name: str,
    country: str | None = None,
    cohort_description: str | None = None,
    constrain_to_literature_tools: bool = False,
) -> LiteraturePhaseHandle:
    """Run the literature + Google search agent (Phase 1).

    Args:
        question: The clinical / research question to search literature for.
        database_name: The OMOP database name. Pass
            :data:`LITERATURE_ONLY_DB_PLACEHOLDER` for literature-only flows
            that never hit a RWD database.
        country: Optional country to bias literature questions toward. See
            :func:`build_country_context`.
        cohort_description: Optional cohort context appended to the question.
        constrain_to_literature_tools: Accepted for call-site compatibility
            but not load-bearing. The tool list is always overridden to web
            search alone, so the agent cannot reach EHR tools during Phase 1
            either way.

    Returns:
        A :class:`LiteraturePhaseHandle` carrying the agent and raw arun result.
    """
    from ascent_domain.omop.models.agents.langgraph.contextual_literature_agent import (
        LiteratureAscentComparisonAgent,
    )
    from ascent_domain.omop.models.agents.langgraph.prompts_agents import literature_only_prompt

    agent_kwargs: dict[str, Any] = {
        "model_name": get_domain_settings().CONTEXT_AGENT_MODEL_NAME,
        "database": database_name,
        "enable_claim_attribution": False,
        "enable_final_summary_verification": False,
    }
    # Web search only, set unconditionally rather than left to the caller: any
    # path that skipped it would fall back to the agent's default tool list.
    from ascent_domain.omop.models.agents.langgraph.google_search_tool import (
        google_search_grounding,
    )

    agent_kwargs["tools_override"] = [google_search_grounding]

    agent = LiteratureAscentComparisonAgent(**agent_kwargs)

    effective_question = question
    if cohort_description:
        effective_question = f"{question}\n\nAdditional cohort context: {cohort_description}"

    custom_prompt = literature_only_prompt.format(country_context=build_country_context(country))

    arun_result = await agent.arun(
        question=effective_question,
        custom_prompt=custom_prompt,
        database=database_name,
    )

    return LiteraturePhaseHandle(
        agent=agent,
        arun_result=arun_result,
        agent_state=arun_result.get("agent_state") or {},
        agent_state_raw=arun_result.get("agent_state_raw") or {},
    )


def extract_literature_outputs(arun_result: dict[str, Any]) -> dict[str, Any]:
    """Extract structured outputs from a Phase 1 ``arun`` result.

    Performs reference normalisation in place on the raw
    ``verified_literature_results`` and ``google_search_results``.

    Returns a dict containing:

    - ``summary``: parsed ``final_answer`` (JSON-decoded if it's a JSON string).
    - ``literature_answers``: ``verified_literature_results["all_answers"]``
      with ``[N]`` markers replaced by ``[H#](url)`` links.
    - ``web_search_answers``: list of ``{"question", "answer"}`` from Google
      results with ``[N]`` markers replaced by ``[G#](url)`` links.
    - ``sources``: unified list of literature + web source dicts.
    - ``verified_literature_results``: raw (mutated) literature dict.
    - ``google_search_results``: raw (mutated) Google list.
    - ``literature_documents``: raw documents (used to build the publication list).
    """
    raw_answer = arun_result.get("final_answer", "")
    try:
        summary = json.loads(raw_answer) if isinstance(raw_answer, str) else raw_answer
    except json.JSONDecodeError as e:
        logger.warning("Failed to JSON-decode literature agent final_answer; falling back to raw string. Error: %s", e)
        summary = raw_answer

    agent_state = arun_result.get("agent_state") or {}
    literature_documents = agent_state.get("literature_documents") or []
    verified_lit = agent_state.get("verified_literature_results") or {"all_answers": []}
    google_results = agent_state.get("google_search_results") or []

    sources = build_sources(verified_lit, google_results)
    normalize_literature_references(verified_lit, sources)
    normalize_google_references(google_results, sources)

    return {
        "summary": summary,
        "literature_answers": verified_lit.get("all_answers", []),
        "web_search_answers": build_google_answers(google_results),
        "sources": sources,
        "verified_literature_results": verified_lit,
        "google_search_results": google_results,
        "literature_documents": literature_documents,
    }


async def combine_with_rwd(
    handle: LiteraturePhaseHandle,
    *,
    question: str,
    rwd_answer: str,
    database_name: str,
) -> dict[str, Any]:
    """Run Phase 2: combine the literature evidence with a RWD answer.

    Delegates to ``handle.agent.combine``, which applies
    the ``literature_combine_prompt`` and uses ``get_patient_count_scaling``
    to extrapolate raw counts to national estimates.

    Args:
        handle: The handle returned from :func:`run_literature_phase`.
        question: The original clinical question.
        rwd_answer: The text answer produced by the RWD pipeline (e.g. the
            ``main_answer`` field from ``answer_question``).
        database_name: The OMOP database name. Must be a real database — the
            combine step looks up its country / population metadata.

    Returns:
        The raw ``combine`` result dict, containing ``final_answer``,
        ``execution_trace``, and ``agent_state``.
    """
    return await handle.agent.combine(
        question=question,
        main_pipeline_answer=rwd_answer,
        database=database_name,
        phase1_state=handle.agent_state_raw,
    )
