"""Shared helpers for cohort generation MCP tools.

Centralises CriteriaService initialisation, text-to-criteria parsing,
criteria disambiguation, and the preview/SQL-generation flow so that
all cohort MCP tools share the same code paths.

Each helper wraps its work in an OpenTelemetry span using the
``pipeline.step.*`` naming convention.
"""

import logging
from typing import List, Optional

from fastmcp.server.dependencies import get_access_token
from opentelemetry import trace

from ascent_domain.models.data_definitions import (
    CodingSystem,
    Criteria,
    MedicalConcept,
    User,
)
from ascent_domain.omop.data.processing.sql_post_processor import ConceptNotFoundError
from ascent_domain.omop.models.inference.criteria_to_counts import CriteriaProcessor
from ascent_domain.services import make_criteria_processor

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


# ── Service initialisation ────────────────────────────────────────────────


def init_criteria_service() -> CriteriaProcessor:
    """Create a ``CriteriaProcessor`` with the same config as the FastAPI DI.

    This mirrors ``make_criteria_service()`` in
    ``ascent/services/ai_services.py`` but without requiring FastAPI
    ``Depends``.
    """
    return make_criteria_processor()


def get_mcp_user() -> User:
    """Extract a ``User`` from the current MCP auth token.

    Returns a ``User`` with ``email`` and ``oid`` populated from the
    access-token claims.  Falls back to ``"unknown"`` if claims are
    missing.
    """
    try:
        access_token = get_access_token()
        claims = access_token.claims or {}
    except Exception:
        claims = {}

    return User(
        email=claims.get("preferred_username", claims.get("email", "unknown")),
        sub=claims.get("oid", claims.get("sub", "unknown")),
        groups=claims.get("groups", []),
    )


# ── Pipeline steps ────────────────────────────────────────────────────────


async def run_text_to_criteria(
    user_input: str,
    service: CriteriaProcessor,
) -> Criteria:
    """Parse free-form text into structured ``Criteria``.

    Wraps ``service.text_to_criteria()``.
    """
    with tracer.start_as_current_span("pipeline.step.text_to_criteria") as span:
        span.set_attribute("cohort.input_length", len(user_input))
        prepared = await service.text_to_criteria(user_input)
        criteria = Criteria.model_validate(prepared)
        span.set_attribute("cohort.num_include", len(criteria.include))
        span.set_attribute("cohort.num_exclude", len(criteria.exclude))
        return criteria


async def run_cohort_preview(
    database: str,
    database_schema: Optional[str],
    coding_system: CodingSystem,
    criteria: Criteria,
    service: CriteriaProcessor,
) -> dict:
    """Generate SQL template + filled SQL + medical concepts from criteria.

    Mirrors the retry-loop logic in ``cohort.preview_cohort_query()``.
    Returns ``{"query_template": str, "query": str, "concepts": list}``.
    """
    with tracer.start_as_current_span("pipeline.step.cohort_preview") as span:
        span.set_attribute("cohort.database", database)
        span.set_attribute("cohort.coding_system", coding_system)

        attempts = 5
        missing_concept_mappings: dict = {}
        query_template = ""
        query = ""

        for i in range(attempts):
            try:
                (_, query_template, query, *_) = await service.criteria_to_data(
                    criteria_dict=criteria.model_dump(),
                    preferred_coding_system=coding_system,
                    snowflake_database=database,
                    snowflake_database_schema=database_schema,
                    main_path=None,
                    log_folder=None,
                    querylib_file=None,
                    verify=False,
                    run_query=False,
                    custom_code_mappings=missing_concept_mappings,
                )
                break
            except ConceptNotFoundError as e:
                if i + 1 < attempts:
                    missing_concept_mappings[(e.category, e.name)] = ""
                else:
                    raise

        concepts_raw = service.find_medical_concepts(
            query_template=query_template or "",
            database=database,
        )

        concepts: List[MedicalConcept] = []
        for raw_item in concepts_raw:
            if raw_item["category"] == "drug_class":
                for drug_name, drug_concepts in raw_item["value"].items():
                    concepts.append(MedicalConcept(name=drug_name, category=raw_item["name"], concepts=drug_concepts))
            else:
                concepts.append(MedicalConcept.from_medical_coder(raw_item))

        span.set_attribute("cohort.num_concepts", len(concepts))
        span.set_attribute("cohort.template_length", len(query_template or ""))

        return {
            "query_template": query_template or "",
            "query": query,
            "concepts": concepts,
        }


def build_criteria(
    criteria_include: list[str],
    criteria_exclude: list[str],
    criteria_index_date: Optional[str],
) -> Criteria:
    """Construct a ``Criteria`` model from flat MCP parameters."""
    return Criteria(
        include=criteria_include,
        exclude=criteria_exclude,
        index_date=criteria_index_date,
    )


def serialize_concepts(concepts: List[MedicalConcept]) -> list[dict]:
    """Convert a list of ``MedicalConcept`` to JSON-safe dicts."""
    return [c.model_dump() for c in concepts]
