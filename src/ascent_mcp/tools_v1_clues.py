"""
CLUES pipeline tools for ascent-mcp-v1 — epidemiological question answering.

Provides composable tools that split the monolithic answer_question pipeline
into individual steps the agent can orchestrate interactively.
"""

import logging
import re

from fastmcp import Context

from ascent_domain.omop.models.uncertainty import clues_api
from ascent_mcp.app_v1 import mcp_v1
from ascent_mcp.tools._capacity import ServerBusy, heavy_tool_slot
from ascent_mcp.tools._clues_helpers import (
    init_qa_system,
    run_interpretations,
    run_template_generation,
)
from ascent_mcp.tools._keepalive import StepTimeout, progress_keepalive, step_budget
from ascent_mcp.tools.permissions import assert_user_permission_db, authorize_user_db, get_user_snowflake_token

logger = logging.getLogger(__name__)


def _error_result(error: Exception, **success_shape: object) -> dict:
    """Structured error frame for step-timeout / capacity failures.

    Returned (not raised) so the client and harness always receive a fast,
    visible, machine-readable failure instead of a silent hang.

    ``success_shape`` carries the keys the tool returns on success, with empty
    values. Scripted clients index the result directly — the original frame
    omitted them, so a timed-out ``generate_epidemiological_sql_non_omop``
    surfaced to the caller as ``KeyError: 'sql_template'`` rather than as the
    timeout it was. Callers can now read the key unconditionally and branch on
    ``error``.
    """
    kind = "timeout" if isinstance(error, StepTimeout) else "server_busy"
    frame: dict = {
        **success_shape,
        "error": kind,
        "message": str(error),
        "retryable": True,
    }
    if isinstance(error, StepTimeout):
        # Which step ran out, and against what ceiling — enough for a caller to
        # decide between retrying and raising the budget.
        frame["failed_step"] = error.step_message
        frame["budget_seconds"] = error.budget_seconds
    return frame


@mcp_v1.tool(task=True)
async def disambiguate_epidemiological_question(
    question: str,
    ctx: Context,
    num_interpretations: int = 3,
) -> dict:
    """
    Disambiguate an epidemiological research question.

    Takes an ambiguous medical/epidemiological question and returns a primary
    interpretation along with alternative interpretations. The agent should
    present these alternatives to the user for selection before proceeding
    with SQL generation.

    Args:
        question: The epidemiological research question to disambiguate.
            Examples: "How many patients have diabetes?", "What is the prevalence of heart failure?"
        num_interpretations: Number of alternative interpretations to generate (default: 3).

    Returns:
        Dictionary with:
        - primary_interpretation: The most likely interpretation of the question
        - alternative_interpretations: List of other valid interpretations the user may have meant
    """
    await ctx.report_progress(0, 2, "Disambiguating question")

    try:
        async with heavy_tool_slot("disambiguate_epidemiological_question"):
            async with progress_keepalive(
                ctx,
                0,
                2,
                "Disambiguating question",
                budget=step_budget("INTERPRETATIONS", 180),
            ):
                primary, all_interps, interp_result = await run_interpretations(
                    question=question,
                    num_interpretations=num_interpretations,
                    disambiguate=True,
                )
    except (StepTimeout, ServerBusy) as e:
        return _error_result(e, primary_interpretation={}, alternative_interpretations=[])

    await ctx.report_progress(2, 2, "Disambiguation complete")

    # The primary is the disambiguated version; alternatives are the rest
    alternatives = [i for i in all_interps if i != primary]

    return {
        "primary_interpretation": primary,
        "alternative_interpretations": alternatives,
    }


@mcp_v1.tool(task=True)
async def generate_epidemiological_sql_with_entities_omop(
    question: str,
    database_name: str,
    ctx: Context,
    database_schema: str | None = None,
    coding_system: str = "Standard",
) -> dict:
    """
    Generate a SQL template from an epidemiological question for an OMOP database.

    Uses the query library (RAG) to produce a SQL template with medical entity
    placeholders like [condition@type 2 diabetes mellitus]. Also extracts the
    medical entities found in the template and provides a human-readable
    explanation of the query logic.

    The agent can then use resolve_placeholders to fill the template with
    actual concept codes, or use lookup_medical_codes for fine-grained control
    over individual entities.

    Args:
        question: The epidemiological question (ideally already disambiguated).
            Example: "How many patients have Type 2 Diabetes Mellitus?"
        database_name: The OMOP database to target (e.g. "SYNTHETIC_EHR_OMOP").
        database_schema: Optional schema override within the database.
        coding_system: "Standard" (default) or "Source" concept coding.

    Returns:
        Dictionary with:
        - query_template: SQL with entity placeholders (e.g. [condition@diabetes])
        - entities: List of extracted medical entities [{type, value}, ...]
        - query_explanation: Human-readable explanation of what the SQL does
    """
    await ctx.report_progress(0, 4, "Initializing QA system")
    await authorize_user_db(database_name)

    try:
        async with heavy_tool_slot("generate_epidemiological_sql_with_entities_omop"):
            async with progress_keepalive(
                ctx,
                0,
                4,
                "Initializing QA system",
                budget=step_budget("INIT_QA", 180),
            ):
                qa_service, db_key = await init_qa_system(
                    database_name=database_name,
                    database_schema=database_schema,
                    token_fetcher=get_user_snowflake_token,
                )

            await ctx.report_progress(1, 4, "Generating SQL template from query library")

            # Generate template for the single question (no multi-interpretation matrix)
            async with progress_keepalive(
                ctx,
                1,
                4,
                "Generating SQL template from query library",
                budget=step_budget("SQL_TEMPLATE", 300),
            ):
                template_result = await run_template_generation(
                    qa_service=qa_service,
                    all_interps=[question],
                    num_sql_samples=1,
                )

            await ctx.report_progress(2, 4, "Extracting medical entities")

            # Extract entities from the generated template
            entities = clues_api.extract_medical_entities(template_result["template_results"])

            await ctx.report_progress(3, 4, "Generating query explanation")

            # Get the primary template text
            template_results = template_result.get("template_results", [])
            primary_template = template_results[0].get("query_template", "") if template_results else ""

            # If Source coding, rewrite *_concept_id → *_source_concept_id in the template
            if coding_system == "Source":
                pattern = re.compile(r"(condition|drug|drug_class|procedure|measurement|observation)_concept_id", re.IGNORECASE)
                primary_template = pattern.sub(
                    lambda m: f"{m.group(1)}_source_concept_id" if m.group().islower() else f"{m.group(1).upper()}_SOURCE_CONCEPT_ID",
                    primary_template,
                )

            # Generate explanation
            async with progress_keepalive(
                ctx,
                3,
                4,
                "Generating query explanation",
                budget=step_budget("EXPLANATION", 120),
            ):
                query_explanation = await clues_api.get_query_explanation(
                    qa_system=qa_service,
                    primary_template=primary_template,
                    primary_question=question,
                )
    except (StepTimeout, ServerBusy) as e:
        return _error_result(e, query_template="", entities=[], query_explanation="")

    await ctx.report_progress(4, 4, "SQL generation complete")

    # Format entities as list of dicts for JSON serialization
    entities_list = [{"type": etype, "value": value} for etype, value in entities]

    return {
        "query_template": primary_template,
        "entities": entities_list,
        "query_explanation": query_explanation,
    }


@mcp_v1.tool(task=True)
async def compare_rwd_with_literature(
    original_question: str,
    rwd_answer: str,
    database_name: str,
    ctx: Context,
) -> dict:
    """
    Compare a real-world data (RWD) answer with published literature evidence.

    Searches the web for the original question,
    then produces a narrative comparing the RWD result against published
    evidence. Includes patient-count scaling and quantifies over/under-
    representation in the RWD relative to literature.

    Use after getting query results to validate findings against published data.

    Args:
        original_question: The clinical question (e.g. "How many patients have
            atopic dermatitis?"). Used for literature search.
        rwd_answer: The answer from the database query (e.g. "There are 45,231
            patients with atopic dermatitis"). This is compared against literature.
        database_name: The OMOP database the answer came from. Used for
            country/population metadata and access control.

    Returns:
        Dictionary with:
        - comparison: Narrative comparing RWD with literature ([A#], [L#], [G#] citations)
        - summary: Literature-only summary
        - publications: List of publications found
        - publication_count: Number of publications
        - status: "success"
    """
    from ascent_domain.literature_agent_service import (
        combine_with_rwd,
        extract_literature_outputs,
        run_literature_phase,
    )
    from ascent_domain.literature_result_utils import build_publications

    total_steps = 5

    await ctx.report_progress(0, total_steps, "Validating database access")
    await assert_user_permission_db(database_name)

    try:
        async with heavy_tool_slot("compare_rwd_with_literature"):
            await ctx.report_progress(1, total_steps, "Searching literature and web sources")
            async with progress_keepalive(
                ctx,
                1,
                total_steps,
                "Searching literature and web sources",
                budget=step_budget("LITERATURE", 600),
            ):
                handle = await run_literature_phase(
                    question=original_question,
                    database_name=database_name,
                    constrain_to_literature_tools=True,
                )

            await ctx.report_progress(2, total_steps, "Processing literature results")
            outputs = extract_literature_outputs(handle.arun_result)
            publications = build_publications(outputs["literature_documents"])

            await ctx.report_progress(3, total_steps, "Comparing RWD with literature")
            async with progress_keepalive(
                ctx,
                3,
                total_steps,
                "Comparing RWD with literature",
                budget=step_budget("COMBINE", 300),
            ):
                combine_result = await combine_with_rwd(
                    handle,
                    question=original_question,
                    rwd_answer=rwd_answer,
                    database_name=database_name,
                )
    except (StepTimeout, ServerBusy) as e:
        return _error_result(e, original_question=original_question, comparison="", summary="", literature_answers=[], web_search_answers=[])

    # Step 4 was missing: the sequence ran 0,1,2,3,5, so a client's progress
    # jumped straight from "comparing" to "complete" and never showed the
    # assembly phase, which is where a large literature payload is collated.
    await ctx.report_progress(4, total_steps, "Assembling comparison")

    comparison = combine_result.get("final_answer") or ""
    combine_error = combine_result.get("error")
    # An empty narrative is a failure too. The agent can terminate without
    # producing one -- it does so when it declines to call a tool the router
    # asked for and the router eventually stops asking -- and reporting that as
    # a success hands the caller an empty string where a comparison should be.
    if not combine_error and not comparison.strip():
        combine_error = "the agent produced no comparison narrative"

    await ctx.report_progress(5, total_steps, "Comparison complete")

    # The searched material is still worth returning when only the narrative
    # failed, but the caller has to be able to tell the difference.
    return {
        "original_question": original_question,
        "comparison": comparison,
        "comparison_error": combine_error,
        "summary": outputs["summary"],
        "literature_answers": outputs["literature_answers"],
        "web_search_answers": outputs["web_search_answers"],
        "sources": outputs["sources"],
        "publications": publications,
        "publication_count": len(publications),
        "status": "comparison_failed" if combine_error else "success",
    }


@mcp_v1.tool(task=True)
async def get_contextual_literature(
    research_question: str,
    ctx: Context,
    cohort_description: str | None = None,
    country: str | None = None,
) -> dict:
    """
    Search published literature for a clinical research question.

    Searches the web via Google to find relevant publications and evidence.
    Does NOT query any database.

    Use standalone for background research, or after getting RWD results
    to find published comparisons.

    Args:
        research_question: The clinical question to search literature for
            (e.g. "What is the prevalence of atopic dermatitis in the US?").
        cohort_description: Optional cohort context to refine the search.
        country: Optional country to focus on (e.g. "United States", "Germany").

    Returns:
        Dictionary with:
        - summary: Narrative synthesis of findings
        - literature_answers: Per-question answers with [H#](url) citations
        - web_search_answers: Per-question web answers with [G#](url) citations
        - sources: Unified source list with URLs
        - publications: List of publications (title, authors, year, url)
        - publication_count: Number of publications found
        - status: "success" or "no_results"
    """
    from ascent_domain.literature_agent_service import (
        LITERATURE_ONLY_DB_PLACEHOLDER,
        extract_literature_outputs,
        run_literature_phase,
    )
    from ascent_domain.literature_result_utils import build_publications

    total_steps = 4

    await ctx.report_progress(0, total_steps, "Preparing literature search")
    await ctx.report_progress(1, total_steps, "Searching literature and web sources")

    try:
        async with heavy_tool_slot("get_contextual_literature"):
            async with progress_keepalive(
                ctx,
                1,
                total_steps,
                "Searching literature and web sources",
                budget=step_budget("LITERATURE", 600),
            ):
                handle = await run_literature_phase(
                    question=research_question,
                    database_name=LITERATURE_ONLY_DB_PLACEHOLDER,
                    country=country,
                    cohort_description=cohort_description,
                )
    except (StepTimeout, ServerBusy) as e:
        return _error_result(e, research_question=research_question, summary="", literature_answers=[], web_search_answers=[], sources=[])

    await ctx.report_progress(3, total_steps, "Processing results")

    outputs = extract_literature_outputs(handle.arun_result)
    publications = build_publications(outputs["literature_documents"])

    await ctx.report_progress(4, total_steps, "Literature search complete")

    if not publications and not outputs["summary"]:
        return {
            "research_question": research_question,
            "summary": "No relevant literature was found for this research question.",
            "literature_answers": [],
            "web_search_answers": [],
            "sources": [],
            "publications": [],
            "publication_count": 0,
            "status": "no_results",
        }

    return {
        "research_question": research_question,
        "summary": outputs["summary"],
        "literature_answers": outputs["literature_answers"],
        "web_search_answers": outputs["web_search_answers"],
        "sources": outputs["sources"],
        "publications": publications,
        "publication_count": len(publications),
        "status": "success",
    }


@mcp_v1.tool(task=True)
async def generate_epidemiological_sql_non_omop(
    question: str,
    database_name: str,
    database_schema: str,
    ctx: Context,
) -> dict:
    """
    Generate a SQL template from an epidemiological question for a non-OMOP database.

    Uses LLM-based SQL generation from the database M-schema (not a query library).
    Extracts medical entities, analyzes the question, and produces a SQL template
    with [entity@name@coding_systems] placeholders.

    The agent can then use resolve_placeholders(use_concept_ids=False) to fill
    the template with source codes, and execute_sql to run it.

    Args:
        question: The epidemiological question (ideally already disambiguated).
            Example: "How many patients have Type 2 Diabetes Mellitus?"
        database_name: The non-OMOP database name (e.g. "SYNTHETIC_CLAIMS").
        database_schema: The schema within the database (e.g. "DATA_202601").

    Returns:
        Dictionary with:
        - sql_template: SQL with entity placeholders (e.g. [condition@diabetes@ICD10CM])
        - entities: List of extracted medical entities
        - analytical_objective: The detected analysis type (e.g. "cohort_count")
    """
    from ascent_domain.non_omop.metadata_queries import (
        get_m_schema,
        get_metadata,
        get_tables_with_medical_coding_ontologies,
    )
    from ascent_domain.non_omop.workflows import QuestionAnalysis, SanityCheck, SQLPreparation
    from ascent_http.settings import settings

    await authorize_user_db(database_name)

    total_steps = 4

    try:
        async with heavy_tool_slot("generate_epidemiological_sql_non_omop"):
            # Step 1: Sanity check — extract medical entities
            await ctx.report_progress(1, total_steps, "Extracting medical entities from question")
            async with progress_keepalive(
                ctx,
                1,
                total_steps,
                "Extracting medical entities",
                budget=step_budget("ENTITIES", 240),
            ):
                sanity_check_handler = SanityCheck(model_name=settings.NON_OMOP_QUESTION_SANITY_CHECK_MODEL_NAME)
                sc = await sanity_check_handler.run_sanity_check(question)

            query_analysis = getattr(sc, "query_analysis", None) or {}
            medical_entities = getattr(sc, "medical_entities", None) or getattr(query_analysis, "medical_entities", None)
            intent = getattr(query_analysis, "intent_classification", None)
            analytical_objective = getattr(intent, "analytical_objective", None) if intent else None
            analytical_objective = analytical_objective or "cohort_count"

            # Step 2: Question analysis
            await ctx.report_progress(2, total_steps, "Analysing question")
            async with progress_keepalive(
                ctx,
                2,
                total_steps,
                "Analysing question",
                budget=step_budget("ANALYSIS", 180),
            ):
                qa_handler = QuestionAnalysis(model_name=settings.NON_OMOP_QUESTION_ANALYSIS_MODEL_NAME)
                question_to_analysis = await qa_handler.run_question_analysis(
                    analytical_objective,
                    question,
                    medical_entities,
                )

            # Step 3: SQL preparation — generate template from M-schema
            await ctx.report_progress(3, total_steps, "Generating SQL template from database schema")
            async with progress_keepalive(
                ctx,
                3,
                total_steps,
                "Generating SQL template",
                budget=step_budget("SQL_TEMPLATE_NON_OMOP", 480),
            ):
                sql_prep = SQLPreparation(model_name=settings.NON_OMOP_SQL_PREPARATION_MODEL_NAME)
                templated = await sql_prep.prepare_sql(
                    database_name=database_name,
                    database_schema=database_schema,
                    question_to_analysis=question_to_analysis,
                    medical_entities=medical_entities,
                    tables_with_medical_coding_ontologies=await get_tables_with_medical_coding_ontologies(
                        executor=None,
                        database_name=database_name,
                        database_schema=database_schema,
                    ),
                    m_schema_string=await get_m_schema(
                        executor=None,
                        database_name=database_name,
                        database_schema=database_schema,
                    )
                    or "",
                    metadata=await get_metadata(
                        executor=None,
                        database_name=database_name,
                        database_schema=database_schema,
                    ),
                )
    except (StepTimeout, ServerBusy) as e:
        return _error_result(e, sql_template="", entities=[], analytical_objective="")

    await ctx.report_progress(4, total_steps, "SQL generation complete")

    # Extract entity info for the response
    entity_refs = templated.extracted_concepts.entity_references if templated.extracted_concepts else []
    entities_list = [{"placeholder": e.placeholder, "entity": e.entity, "entity_name": e.entity_name} for e in entity_refs]

    return {
        "sql_template": templated.sql_code_snippet,
        "entities": entities_list,
        "analytical_objective": analytical_objective,
    }
