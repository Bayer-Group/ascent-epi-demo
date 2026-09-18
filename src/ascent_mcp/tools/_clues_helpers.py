"""Shared helpers for CLUES pipeline MCP tools.

Centralises QA-system initialisation, interpretation generation, template
building, entity extraction/coding, and template filling so that
generate_query_template, generate_query_filled, and answer_question all
share the same code paths.

Each helper wraps its work in an OpenTelemetry span using the
``pipeline.step.*`` naming convention, which is distinct from the
``mcp.*`` tool-level spans created by the telemetry middleware.
"""

import logging
from typing import Optional

from opentelemetry import metrics, trace

from ascent_domain.omop.models.uncertainty import clues_api
from ascent_domain.omop.schemas.constants import CodingType
from ascent_http.services.ai_services import _SearchService, load_query_library
from ascent_platform.identity import TokenFetcher

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# ── OTEL Metrics ──────────────────────────────────────────────────────────
meter = metrics.get_meter("ascent_mcp.clues")

tool_invocations = meter.create_counter(
    "clues.tool.invocations",
    description="Number of CLUES tool invocations",
    unit="1",
)

step_duration = meter.create_histogram(
    "clues.step.duration",
    description="Duration of individual CLUES pipeline steps",
    unit="ms",
)

pipeline_duration = meter.create_histogram(
    "clues.pipeline.duration",
    description="Total duration of the full CLUES pipeline",
    unit="ms",
)


# ── Helpers ───────────────────────────────────────────────────────────────
def to_coding_type(coding_system: str) -> CodingType:
    """Convert user-facing coding system string to CodingType enum."""
    return CodingType.STANDARD_CODING if coding_system == "Standard" else CodingType.SOURCE_CODING


async def init_qa_system(
    database_name: str,
    database_schema: Optional[str] = None,
    token_fetcher: Optional[TokenFetcher] = None,
):
    """Initialise the QA system (query library + search client).

    Args:
        database_name: Snowflake database to use.
        database_schema: Optional schema override.
        token_fetcher: Zero-arg async callable that returns a fresh Snowflake
            OAuth token (e.g. ``get_user_snowflake_token``). Required in
            backend mode — all Snowflake queries run under the user's own
            identity. The provider re-fetches on each connection open so
            multi-minute pipelines never serve a stale token.

    Returns:
        (qa_service, db_key) tuple.
    """
    if token_fetcher is None:
        raise RuntimeError("init_qa_system requires a Snowflake token_fetcher: SSO is mandatory for user RWD queries.")
    with tracer.start_as_current_span("pipeline.step.init_qa_system") as span:
        span.set_attribute("db.name", database_name)
        query_library = load_query_library()
        service = _SearchService(query_library=query_library)
        db_key = f"{database_name}.{database_schema}" if database_schema else database_name
        qa_service = await service.search_client(database=db_key, token_fetcher=token_fetcher)
        return qa_service, db_key


async def run_interpretations(
    question: str,
    num_interpretations: int = 3,
    disambiguate: bool = True,
):
    """Generate question interpretations (CLUES step 1).

    When *disambiguate* is ``True`` (default), ``primary_interpretation``
    is set to ``None`` so the upstream ``clues_api`` runs full recursive
    disambiguation before generating alternative interpretations.

    When ``False``, the raw *question* is passed as the primary
    interpretation, skipping the disambiguation round-trip.

    Returns:
        (primary, all_interps, interp_result) tuple.
    """
    with tracer.start_as_current_span("pipeline.step.interpret_question") as span:
        span.set_attribute("clues.disambiguate", disambiguate)
        span.set_attribute("clues.num_interpretations", num_interpretations)

        primary_interpretation = None if disambiguate else question

        interp_result = await clues_api.generate_question_interpretations(
            original_question=question,
            num_interpretations=num_interpretations,
            interpretation_mode="explore_meanings",
            primary_interpretation=primary_interpretation,
        )
        primary = interp_result["primary_interpretation"]
        all_interps = interp_result["all_interpretations"]
        span.set_attribute("clues.num_generated", len(all_interps))
        return primary, all_interps, interp_result


async def run_template_generation(qa_service, all_interps, num_sql_samples=3):
    """Generate SQL templates for all interpretations (CLUES step 2).

    Returns:
        template_result dict from ``clues_api.generate_query_templates``.
    """
    with tracer.start_as_current_span("pipeline.step.build_sql_templates") as span:
        span.set_attribute("clues.num_interpretations", len(all_interps))
        span.set_attribute("clues.num_sql_samples", num_sql_samples)
        template_result = await clues_api.generate_query_templates(
            qa_system=qa_service,
            all_interpretations=all_interps,
            num_sql_samples=num_sql_samples,
        )
        span.set_attribute(
            "clues.num_templates",
            len(template_result.get("template_results", [])),
        )
        return template_result
