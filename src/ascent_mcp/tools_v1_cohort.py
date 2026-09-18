"""
Cohort generation tools for ascent-mcp-v1.

Composable tools that split the monolithic cohort_create pipeline into
individual steps the agent can orchestrate interactively:
    text_to_criteria → generate_sql → resolve_placeholders → persist → verify

All tools are registered on the `mcp_v1` FastMCP instance.
"""

import asyncio
import logging
import re
import uuid
from uuid import UUID

import pandas as pd
import pandas.api.types as ptypes
from fastmcp import Context
from opentelemetry import trace

from ascent_domain.cohort import CannotShareStudyCohortsException, create_cohort_funnel
from ascent_domain.cohort import share_cohort as _share_cohort
from ascent_domain.data_queries.cohort import (
    CohortAccessDeniedError,
    CohortNotFoundError,
    _sanitize_identifier,
    authorize_cohort,
    default_cohort_name,
    delete_cohort_descriptors,
    fetch_cohort_dataframe,
    get_cohort_descriptor,
    get_cohort_descriptors_with_roles,
    replace_cohort_columns,
    store_cohort,
    store_cohort_descriptor,
    update_cohort_descriptor,
)
from ascent_domain.data_queries.cohort import revoke_cohort_access as _revoke_cohort_access
from ascent_domain.models.data_definitions import (
    CohortRole,
    CovariateInfo,
    UserCohortDescriptor,
    UserCohortDescriptorPatch,
)
from ascent_domain.non_omop.metadata_queries import (
    get_custom_instructions,
    get_m_schema,
    get_metadata,
    get_tables_with_medical_coding_ontologies,
)
from ascent_domain.non_omop.schemas.question_to_analysis import (
    AnalyticalSteps,
    CohortDefinition,
    QuestionToAnalysis,
)
from ascent_domain.non_omop.workflows import SanityCheck, SQLPreparation
from ascent_domain.omop.models.inference.criteria_to_counts import CriteriaProcessor
from ascent_domain.omop.models.uncertainty.disambiguation import disambiguate_criteria_dict, recursive_disambiguation
from ascent_domain.services import make_criteria_processor
from ascent_http.settings import settings
from ascent_mcp._synthetic import inject_synthetic_warning
from ascent_mcp.app_v1 import mcp_v1
from ascent_mcp.tools._cohort_helpers import (
    build_criteria,
    get_mcp_user,
    init_criteria_service,
    run_cohort_preview,
    run_text_to_criteria,
    serialize_concepts,
)
from ascent_mcp.tools._keepalive import progress_keepalive
from ascent_mcp.tools.permissions import authorize_user_db, get_user_snowflake_token
from ascent_platform.db.postgresql_session import AsyncSessionLocal
from ascent_platform.warehouse.db_type import is_omop_database
from ascent_platform.warehouse.session import get_db, get_db_schema
from ascent_platform.warehouse.user_session import get_db_as_user, user_sql_executor

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

_ID_COLUMNS = {"subject_id", "person_id"}
_BASE_COLUMNS = _ID_COLUMNS | {"index_date"}
_FORBIDDEN_SQL = re.compile(r"\b(insert|update|delete|merge|drop|alter|create|truncate|grant|revoke|call|use)\b", re.I)


def _find_id_column(columns) -> str | None:
    return next((c for c in columns if str(c).lower() in _ID_COLUMNS), None)


def _covariate_columns(columns) -> list[str]:
    return [c for c in columns if str(c).lower() not in _BASE_COLUMNS]


def _as_text_key(series: pd.Series) -> pd.Series:
    """Render a key column as text without float artefacts.

    ``str(1.0)`` is ``"1.0"``, which matches no id anywhere, so integral floats
    are formatted as integers before comparison.
    """
    if ptypes.is_numeric_dtype(series):
        return series.map(lambda v: pd.NA if pd.isna(v) else (str(int(v)) if float(v).is_integer() else str(v))).astype("string")
    return series.astype("string").str.strip()


def _aligned_join_keys(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Put two id columns into a dtype that can actually match.

    Both sides used to be pushed through ``pd.to_numeric(errors="coerce")``.
    That is safe for OMOP person_id, but the source layer keys on MRN
    ("RBM0000001"): every value coerces to NaN, and pandas *matches NaN to
    NaN* on a merge, so a 90-patient cohort joined to 1,000 covariate rows
    produced 90,000 -- a cartesian product reported as success, surfacing
    only later as an unrelated-looking cast error.

    Numeric only when both sides are wholly numeric; text otherwise.
    """
    left_num, right_num = pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")
    if len(left) and len(right) and left_num.notna().all() and right_num.notna().all():
        return left_num, right_num
    return _as_text_key(left), _as_text_key(right)


def _assert_read_only_select(sql: str) -> None:
    statement = sql.strip().rstrip(";").strip()
    if ";" in statement:
        raise ValueError("SQL must be a single statement.")
    if not re.match(r"(?is)^\s*(select|with)\b", statement):
        raise ValueError("SQL must be a read-only SELECT statement.")
    if _FORBIDDEN_SQL.search(statement):
        raise ValueError("SQL must not contain DML/DDL statements.")


def _covariates_from_table_info(table_info, source: str, definition: str | None) -> list[CovariateInfo]:
    return [
        CovariateInfo(name=col.name, type=col.type, source=source, definition=definition)
        for col in table_info.columns
        if col.name.upper() not in {"SUBJECT_ID", "INDEX_DATE"}
    ]


# =============================================================================
# Tool 1: cohort_text_to_criteria
# =============================================================================


async def _disambiguate_criterion_with_alternatives(criterion: str, mode: str = "criteria") -> dict:
    """Disambiguate a single criterion and return preferred + alternatives."""
    try:
        result = await recursive_disambiguation(criterion, mode=mode)
        last_round = result.get("rounds", [])[-1] if result.get("rounds") else None
        last_result = last_round["result"] if last_round else None
        alternatives = []
        if last_result and hasattr(last_result, "final_interpretations"):
            alternatives = [a for a in last_result.final_interpretations if a != result["final_interpretation"]]
        return {
            "original": criterion,
            "preferred": result["final_interpretation"],
            "alternatives": alternatives,
        }
    except Exception as e:
        logger.warning("Disambiguation failed for criterion '%s': %s", criterion, e)
        return {
            "original": criterion,
            "preferred": criterion,
            "alternatives": [],
        }


@mcp_v1.tool(task=True)
async def cohort_text_to_criteria(
    input_text: str,
    ctx: Context,
    disambiguate: bool = True,
) -> dict:
    """
    Parse natural language text into structured patient cohort criteria.

    Uses AI to extract inclusion criteria, exclusion criteria, and an index
    date definition from free-form text describing a patient cohort.

    When disambiguation is enabled (default), each criterion is refined for
    precision and alternative interpretations are provided so you can present
    choices to the user.

    Args:
        input_text: Free-form text describing the patient cohort.
            Examples:
            - "Women aged 16-45 with at least two diagnoses of endometriosis between 2010-2019"
            - "Patients with type 2 diabetes who have been prescribed metformin, excluding those with CKD"
        disambiguate: If True (default), disambiguate the extracted criteria
            to resolve ambiguous medical terms and provide alternatives.

    Returns:
        Dictionary containing:
        - include: List of inclusion criteria strings (disambiguated if enabled)
        - exclude: List of exclusion criteria strings (disambiguated if enabled)
        - index_date: Index date definition string, or null
        - disambiguation_applied: Whether disambiguation was performed
        - disambiguation_details: Per-criterion details with original, preferred,
          and alternative interpretations (null if disambiguation was not applied)
    """
    with tracer.start_as_current_span("mcp_v1.cohort_text_to_criteria") as span:
        span.set_attribute("cohort.input_length", len(input_text))
        span.set_attribute("cohort.disambiguate", disambiguate)

        total_steps = 3 if disambiguate else 2

        await ctx.report_progress(0, total_steps, "Initializing criteria service")
        service = init_criteria_service()

        await ctx.report_progress(1, total_steps, "Extracting criteria from text")
        criteria = await run_text_to_criteria(input_text, service)

        disambiguation_details = None
        if disambiguate:
            await ctx.report_progress(2, total_steps, "Disambiguating criteria")
            async with progress_keepalive(ctx, 2, total_steps, "Disambiguating criteria"):
                # Disambiguate all criteria in parallel, collecting alternatives
                all_criteria = []
                modes = []
                for c in criteria.include or []:
                    all_criteria.append(c)
                    modes.append("criteria")
                for c in criteria.exclude or []:
                    all_criteria.append(c)
                    modes.append("criteria")
                if criteria.index_date:
                    all_criteria.append(criteria.index_date)
                    modes.append("index_date")

                results = await asyncio.gather(*[_disambiguate_criterion_with_alternatives(c, m) for c, m in zip(all_criteria, modes)])

                # Split results back into include/exclude/index_date
                n_include = len(criteria.include or [])
                n_exclude = len(criteria.exclude or [])

                include_results = results[:n_include]
                exclude_results = results[n_include : n_include + n_exclude]
                index_date_results = results[n_include + n_exclude :]

                # Build disambiguated criteria
                criteria.include = [r["preferred"] for r in include_results]
                criteria.exclude = [r["preferred"] for r in exclude_results]
                if index_date_results:
                    criteria.index_date = index_date_results[0]["preferred"]

                # Collect details for all criteria
                disambiguation_details = list(include_results) + list(exclude_results) + list(index_date_results)

        await ctx.report_progress(total_steps, total_steps, "Done")

        logger.info("cohort_text_to_criteria completed: %d include, %d exclude", len(criteria.include), len(criteria.exclude))

        return {
            "include": criteria.include,
            "exclude": criteria.exclude,
            "index_date": criteria.index_date,
            "disambiguation_applied": disambiguate,
            "disambiguation_details": disambiguation_details,
        }


# =============================================================================
# Tool 2: cohort_generate_sql_omop
# =============================================================================


@mcp_v1.tool(task=True)
@inject_synthetic_warning
async def cohort_generate_sql_omop(
    criteria_include: list[str],
    criteria_exclude: list[str],
    database_name: str,
    ctx: Context,
    criteria_index_date: str | None = None,
    database_schema: str | None = None,
    coding_system: str = "Standard",
) -> dict:
    """
    Generate a SQL query template for a patient cohort from structured criteria (OMOP databases).

    Returns a SQL template with medical-entity placeholders like
    ``[condition@atopic dermatitis]`` that can be resolved into concept codes
    using ``resolve_placeholders``. The SQL is NOT filled with codes and NOT
    executed.

    Args:
        criteria_include: List of inclusion criteria strings.
            Example: ["Women aged 16-45", "At least two diagnoses of endometriosis"]
        criteria_exclude: List of exclusion criteria strings.
            Example: ["Prior history of hysterectomy"]
        database_name: The OMOP database (e.g. "SYNTHETIC_EHR_OMOP").
        criteria_index_date: Index date definition (optional).
        database_schema: Optional schema. Resolved automatically if omitted.
        coding_system: "Standard" (default) or "Source".

    Returns:
        Dictionary containing:
        - query_template: SQL with entity placeholders (e.g. [condition@diabetes])
        - entities: List of medical entities found [{type, value}, ...]
        - concepts: List of medical concepts with vocabulary details
        - criteria: The structured criteria used
    """
    with tracer.start_as_current_span("mcp_v1.cohort_generate_sql_omop") as span:
        span.set_attribute("cohort.database", database_name)
        span.set_attribute("cohort.num_include", len(criteria_include))
        span.set_attribute("cohort.num_exclude", len(criteria_exclude))

        await authorize_user_db(database_name)

        total_steps = 3

        await ctx.report_progress(0, total_steps, "Initializing criteria service")
        service = init_criteria_service()

        criteria = build_criteria(criteria_include, criteria_exclude, criteria_index_date)

        await ctx.report_progress(1, total_steps, "Generating SQL template")
        async with progress_keepalive(ctx, 1, total_steps, "Generating SQL template"):
            preview = await run_cohort_preview(
                database=database_name,
                database_schema=database_schema,
                coding_system=coding_system,
                criteria=criteria,
                service=service,
            )

        await ctx.report_progress(2, total_steps, "Generating query explanation")
        explanation = None
        if preview["query_template"]:
            async with progress_keepalive(ctx, 2, total_steps, "Generating query explanation"):
                explanation = await service.get_query_explanation(
                    preview["query_template"],
                    criteria.model_dump(),
                )

        await ctx.report_progress(3, total_steps, "Done")

        # Extract entities from [entity_type@entity_value] patterns
        entity_pattern = re.compile(r"\[(\w+)@([^\]]+)\]")
        entities = [{"type": m.group(1), "value": m.group(2)} for m in entity_pattern.finditer(preview["query_template"])]

        logger.info("cohort_generate_sql_omop completed for database=%s", database_name)

        return {
            "query_template": preview["query_template"],
            "entities": entities,
            "concepts": serialize_concepts(preview["concepts"]),
            "explanation": explanation,
            "criteria": {
                "include": criteria.include,
                "exclude": criteria.exclude,
                "index_date": criteria.index_date,
            },
        }


# =============================================================================
# Tool 3: cohort_generate_sql_non_omop
# =============================================================================


@mcp_v1.tool(task=True)
@inject_synthetic_warning
async def cohort_generate_sql_non_omop(
    criteria_include: list[str],
    criteria_exclude: list[str],
    database_name: str,
    database_schema: str,
    ctx: Context,
    criteria_index_date: str | None = None,
) -> dict:
    """
    Generate a SQL query template for a patient cohort from structured criteria (non-OMOP databases).

    Returns a SQL template with medical-entity placeholders like
    ``[condition@diabetes@ICD10CM,ICD9CM]`` that can be resolved into source
    codes using ``resolve_placeholders(sql, use_concept_ids=False)``.

    Args:
        criteria_include: List of inclusion criteria strings.
        criteria_exclude: List of exclusion criteria strings.
        database_name: The non-OMOP database (e.g. "SYNTHETIC_CLAIMS").
        database_schema: Schema within the database (e.g. "DATA_202601").
        criteria_index_date: Index date definition (optional).

    Returns:
        Dictionary containing:
        - sql_template: SQL with entity placeholders (e.g. [condition@diabetes@ICD10CM])
        - entities: List of entities found in the template
        - criteria: The structured criteria used
    """
    with tracer.start_as_current_span("mcp_v1.cohort_generate_sql_non_omop") as span:
        span.set_attribute("cohort.database", database_name)
        span.set_attribute("cohort.database_schema", database_schema)
        span.set_attribute("cohort.num_include", len(criteria_include))
        span.set_attribute("cohort.num_exclude", len(criteria_exclude))

        await authorize_user_db(database_name)

        criteria = build_criteria(criteria_include, criteria_exclude, criteria_index_date)
        total_steps = 4

        # ── Step 0: Disambiguate criteria ──
        await ctx.report_progress(0, total_steps, "Disambiguating criteria")
        async with progress_keepalive(ctx, 0, total_steps, "Disambiguating criteria"):
            criteria_dict_in = {
                "include": list(criteria.include or []),
                "exclude": list(criteria.exclude or []),
                "index_date": [criteria.index_date] if criteria.index_date else [],
            }
            try:
                disambiguated = await disambiguate_criteria_dict(criteria_dict=criteria_dict_in)
                criteria.include = disambiguated.get("include", criteria.include) or criteria.include
                criteria.exclude = disambiguated.get("exclude", criteria.exclude) or criteria.exclude
                idx = disambiguated.get("index_date") or []
                if idx:
                    criteria.index_date = idx[0]
            except Exception as e:
                logger.warning("cohort_generate_sql_non_omop: disambiguation failed, continuing: %s", e)

        # ── Build pseudo-question from criteria ──
        parts = []
        if criteria.include:
            parts.append("Inclusion criteria: " + "; ".join(criteria.include) + ".")
        if criteria.index_date:
            parts.append(f"Index date: {criteria.index_date}.")
        if criteria.exclude:
            parts.append("Exclusion criteria: " + "; ".join(criteria.exclude) + ".")
        cohort_question = "Create a patient cohort. " + " ".join(parts)

        # ── Step 1: Entity extraction via SanityCheck ──
        await ctx.report_progress(1, total_steps, "Extracting medical entities")
        async with progress_keepalive(ctx, 1, total_steps, "Extracting medical entities"):
            sanity_check_handler = SanityCheck(model_name=settings.NON_OMOP_QUESTION_SANITY_CHECK_MODEL_NAME)
            sc = await sanity_check_handler.run_sanity_check(cohort_question)

        query_analysis = getattr(sc, "query_analysis", None) or {}
        medical_entities = getattr(sc, "medical_entities", None) or getattr(query_analysis, "medical_entities", None)

        # ── Step 2: Construct QuestionToAnalysis ──
        question_to_analysis = QuestionToAnalysis(
            original_question=cohort_question,
            cohort_definition=CohortDefinition(
                inclusion_criteria=criteria.include,
                exclusion_criteria=criteria.exclude,
                index_date=criteria.index_date or "",
            ),
            analytical_steps=AnalyticalSteps(
                steps=["Select patients matching all criteria with their index dates"],
            ),
            suggested_visualizations=[],
        )

        # ── Step 3: Generate cohort SQL template ──
        await ctx.report_progress(2, total_steps, "Generating cohort SQL template")
        async with progress_keepalive(ctx, 2, total_steps, "Generating cohort SQL template"):
            (
                tables_with_ontologies,
                custom_instruction,
                m_schema_string,
                metadata,
            ) = await asyncio.gather(
                get_tables_with_medical_coding_ontologies(executor=None, database_name=database_name, database_schema=database_schema),
                get_custom_instructions(executor=None, database_name=database_name, database_schema=database_schema),
                get_m_schema(executor=None, database_name=database_name, database_schema=database_schema),
                get_metadata(executor=None, database_name=database_name, database_schema=database_schema),
            )

            sql_preparation = SQLPreparation(model_name=settings.NON_OMOP_SQL_PREPARATION_MODEL_NAME)
            sql_template = await sql_preparation.prepare_cohort_sql(
                database_name=database_name,
                database_schema=database_schema,
                inclusion_criteria=criteria.include,
                exclusion_criteria=criteria.exclude,
                index_date=criteria.index_date or "",
                question_to_analysis=question_to_analysis,
                medical_entities=medical_entities,
                tables_with_medical_coding_ontologies=tables_with_ontologies,
                m_schema_string=m_schema_string or "",
                metadata=metadata,
                custom_instruction=custom_instruction,
                top_k=500,
            )

        await ctx.report_progress(total_steps, total_steps, "Done")

        # Extract entities from the template
        placeholder_pattern = re.compile(r"\[(\w+)@([^@]+)@([^\]]+)\]")
        entities = [
            {
                "placeholder": m.group(0),
                "entity_type": m.group(1),
                "entity_name": m.group(2),
                "coding_systems": m.group(3),
            }
            for m in placeholder_pattern.finditer(sql_template.sql_code_snippet or "")
        ]

        logger.info("cohort_generate_sql_non_omop completed for database=%s", database_name)

        return {
            "sql_template": sql_template.sql_code_snippet or "",
            "entities": entities,
            "criteria": {
                "include": criteria.include,
                "exclude": criteria.exclude,
                "index_date": criteria.index_date,
            },
        }


# =============================================================================
# Tool 4: cohort_persist
# =============================================================================


@mcp_v1.tool(task=True)
@inject_synthetic_warning
async def cohort_persist(
    query_filled: str,
    database_name: str,
    ctx: Context,
    name: str | None = None,
    description: str | None = None,
    database_schema: str | None = None,
    criteria_include: list[str] | None = None,
    criteria_exclude: list[str] | None = None,
    criteria_index_date: str | None = None,
    query_template: str | None = None,
    coding_system: str = "Standard",
    explanation: str | None = None,
) -> dict:
    """
    Execute a filled cohort SQL query and persist the result as a cohort table.

    Takes a fully resolved SQL query (output of ``resolve_placeholders``),
    executes it against the user's database, and persists the resulting patient
    cohort to the cohort store. The caller (you) becomes the cohort owner.

    Returns metadata only — patient IDs stay in the cohort table.

    Args:
        query_filled: The complete, executable SQL query with all medical codes filled in.
        database_name: The database to execute against.
        name: Human-readable cohort name. Pass a short, descriptive name from the
            user's request (e.g. "Type 2 diabetes — synthetic EHR"). A friendly name is
            generated if omitted.
        description: Optional longer description of the cohort.
        database_schema: Schema within the database. Required for non-OMOP databases.
        criteria_include: Original inclusion criteria (stored for audit/attrition).
        criteria_exclude: Original exclusion criteria (stored for audit/attrition).
        criteria_index_date: Original index date definition.
        query_template: Original SQL template before code resolution (stored for attrition).
        coding_system: "Standard" or "Source" (for OMOP databases).
        explanation: Human-readable explanation of the query logic.

    Returns:
        Dictionary containing:
        - cohort_id: UUID of the created cohort
        - patient_count: Number of patients in the cohort
        - table_name: Cohort table name (in the ASCENT.ASCENT_COHORTS schema)
        - database: Database used
    """
    with tracer.start_as_current_span("mcp_v1.cohort_persist") as span:
        span.set_attribute("cohort.database", database_name)

        await authorize_user_db(database_name)
        user = get_mcp_user()

        is_omop = is_omop_database(database_name)
        span.set_attribute("cohort.pipeline", "omop" if is_omop else "non_omop")

        total_steps = 3
        cohort_id = uuid.uuid4()

        # ── Step 1: Execute query ──
        await ctx.report_progress(0, total_steps, "Executing cohort query")
        async with progress_keepalive(ctx, 0, total_steps, "Executing cohort query"):
            if is_omop:
                sf_token = await get_user_snowflake_token()
                async with get_db_as_user(database_name, sf_token, database_schema=database_schema) as rwd_db:
                    cohort_df = await fetch_cohort_dataframe(rwd_db, query_filled)
            else:
                # Non-OMOP: includes SQL healing retry
                from ascent_domain.non_omop.workflows import SQLResults

                max_heal_attempts = 3
                sql_healer = SQLResults(model_name=settings.NON_OMOP_SQL_PREPARATION_MODEL_NAME)
                final_sql = query_filled
                m_schema_string = None

                for heal_attempt in range(max_heal_attempts):
                    try:
                        sf_token = await get_user_snowflake_token()
                        async with get_db_as_user(database_name, sf_token, database_schema=database_schema) as rwd_db:
                            cohort_df = await fetch_cohort_dataframe(rwd_db, final_sql)
                        break
                    except Exception as exec_err:
                        if heal_attempt + 1 >= max_heal_attempts:
                            raise
                        logger.warning(
                            "cohort_persist (non_omop): SQL execution failed (attempt %d/%d), healing: %s",
                            heal_attempt + 1,
                            max_heal_attempts,
                            exec_err,
                        )
                        try:
                            if m_schema_string is None:
                                m_schema_string = await get_m_schema(
                                    executor=None,
                                    database_name=database_name,
                                    database_schema=database_schema,
                                )
                            healed = await sql_healer.handle_invalid_sql(
                                query=final_sql,
                                error=str(exec_err),
                                m_schema=m_schema_string or "",
                            )
                            if healed:
                                final_sql = healed
                        except Exception as heal_err:
                            logger.error("cohort_persist (non_omop): SQL healing failed: %s", heal_err)
                            raise exec_err from heal_err

                # Update query_filled if it was healed
                query_filled = final_sql

        if _find_id_column(cohort_df.columns) is None:
            return {"error": "The cohort query must return a patient id column (person_id or subject_id)."}
        covariate_columns = _covariate_columns(cohort_df.columns)

        # ── Step 2: Persist to the cohort store ──
        await ctx.report_progress(1, total_steps, "Persisting cohort table")
        async with progress_keepalive(ctx, 1, total_steps, "Persisting cohort table"):
            async with get_db("ASCENT") as ascent_db:
                table_info = await store_cohort(
                    cohort_id=cohort_id,
                    ascent_db=ascent_db,
                    cohort=cohort_df,
                    covariate_columns=covariate_columns,
                )

        # ── Step 3: Store descriptor in PostgreSQL ──
        await ctx.report_progress(2, total_steps, "Storing cohort descriptor")
        resolved_schema = database_schema
        if resolved_schema is None:
            resolved_schema = await get_db_schema(database_name)

        criteria = None
        if criteria_include is not None:
            criteria = build_criteria(
                criteria_include or [],
                criteria_exclude or [],
                criteria_index_date,
            )

        descriptor = UserCohortDescriptor(
            id=cohort_id,
            name=name or default_cohort_name(database_name, criteria_include),
            description=description,
            state="active",
            database=database_name,
            database_schema=resolved_schema,
            table=table_info.name,
            size=table_info.size,
            attributes=table_info.columns,
            owner_id=user.email if user.email else "",
            criteria=criteria,
            coding_system=coding_system if is_omop else None,
            query_template=query_template,
            query=query_filled,
            explanation=explanation,
            covariates=_covariates_from_table_info(table_info, source="inline", definition=None) or None,
        )

        async with AsyncSessionLocal() as app_db:
            await store_cohort_descriptor(app_db=app_db, descriptor=descriptor)

        await ctx.report_progress(total_steps, total_steps, "Done")

        span.set_attribute("cohort.id", str(cohort_id))
        span.set_attribute("cohort.patient_count", table_info.size)

        logger.info(
            "cohort_persist completed: cohort_id=%s, patients=%d, table=%s",
            cohort_id,
            table_info.size,
            table_info.name,
        )

        return {
            "cohort_id": str(cohort_id),
            "name": descriptor.name,
            "state": descriptor.state,
            "patient_count": table_info.size,
            "database": database_name,
            "database_schema": resolved_schema,
        }


# =============================================================================
# Tool 5: list_cohorts
# =============================================================================


@mcp_v1.tool()
async def list_cohorts(include_anonymous: bool = False) -> dict:
    """
    List cohorts available to the current user (owned + shared with you).

    Each entry shows the cohort's author (creator), your access role
    (owner/editor/viewer), and whether it's saved (state ``active``) or a draft
    (``pending``).

    Args:
        include_anonymous: Include unnamed draft cohorts (default False).

    Returns:
        Dictionary with ``total`` and ``cohorts`` (id, name, description, author,
        your_role, state, origin, database, size, created).
    """
    user = get_mcp_user()

    async with AsyncSessionLocal() as app_db:
        descriptors = await get_cohort_descriptors_with_roles(app_db, user, include_pending=include_anonymous)

    cohorts = [
        {
            "id": str(d.id),
            "name": d.name,
            "description": d.description,
            "author": getattr(d, "owner_id", None),
            "your_role": role,
            "state": d.state,
            "origin": d.origin,
            "database": d.database,
            "database_schema": d.database_schema,
            "size": d.size,
            "created": d.created.isoformat() if d.created else None,
        }
        for d, role in descriptors
    ]

    return {
        "total": len(cohorts),
        "cohorts": cohorts,
    }


# =============================================================================
# Tool 6: get_cohort
# =============================================================================


@mcp_v1.tool()
async def get_cohort(cohort_id: str) -> dict:
    """
    Retrieve an existing cohort by its ID and return its full definition.

    Returns cohort metadata including criteria, SQL query, coding system,
    and medical concepts used to build it.

    Args:
        cohort_id: UUID of the cohort to retrieve.

    Returns:
        Dictionary containing the full cohort definition:
        - id, name, description, state, origin
        - database, database_schema, table, size, created
        - attributes: list of column definitions
        - For user cohorts: criteria, coding_system, query_template, query,
          medical_concepts, funnel, explanation
        - For study cohorts: study_id
    """
    from uuid import UUID

    try:
        parsed_id = UUID(cohort_id)
    except ValueError:
        return {"error": f"Invalid cohort_id format: {cohort_id!r}. Expected a UUID."}

    user = get_mcp_user()

    try:
        async with AsyncSessionLocal() as app_db:
            descriptor, role = await authorize_cohort(app_db, user, parsed_id, "viewer")
    except CohortNotFoundError:
        return {"error": f"Cohort {cohort_id} not found or you do not have access to it."}

    result = {
        "id": str(descriptor.id),
        "name": descriptor.name,
        "description": descriptor.description,
        "author": getattr(descriptor, "owner_id", None),
        "your_role": role,
        "state": descriptor.state,
        "origin": descriptor.origin,
        "database": descriptor.database,
        "database_schema": descriptor.database_schema,
        "size": descriptor.size,
        "created": descriptor.created.isoformat() if descriptor.created else None,
        "attributes": [{"name": attr.name, "type": attr.type} for attr in descriptor.attributes],
    }

    if descriptor.origin == "user":
        result["criteria"] = descriptor.criteria.model_dump() if descriptor.criteria else None
        result["coding_system"] = descriptor.coding_system
        result["query_template"] = descriptor.query_template
        result["query"] = descriptor.query
        result["medical_concepts"] = [c.model_dump() for c in descriptor.user_concepts] if descriptor.user_concepts else None
        result["funnel"] = [f.model_dump() for f in descriptor.funnel] if descriptor.funnel else []
        result["explanation"] = descriptor.explanation
        result["covariates"] = [c.model_dump(mode="json") for c in descriptor.covariates] if descriptor.covariates else []
    elif descriptor.origin == "study":
        result["study_id"] = str(descriptor.study_id)

    return result


# =============================================================================
# Tool: cohort_add_covariates
# =============================================================================


@mcp_v1.tool(task=True)
async def cohort_add_covariates(cohort_id: str, covariate_sql: str, ctx: Context) -> dict:
    """
    Extend an existing cohort with per-patient covariate columns.

    Runs a read-only SELECT (returning a patient id column plus one or more
    covariate columns, one row per patient) under the user's warehouse access,
    joins it onto the cohort on the patient id, and adds the new columns to the
    cohort table. Add-only: a covariate whose column name already exists on the
    cohort is rejected.

    Args:
        cohort_id: UUID of the cohort to extend.
        covariate_sql: A single SELECT returning person_id/subject_id plus the
            covariate columns to add.

    Returns:
        Dictionary containing cohort_id, added_covariates, patient_count, columns.
    """
    with tracer.start_as_current_span("mcp_v1.cohort_add_covariates") as span:
        span.set_attribute("cohort.id", cohort_id)

        try:
            parsed_id = UUID(cohort_id)
        except ValueError:
            return {"error": f"Invalid cohort_id format: {cohort_id!r}. Expected a UUID."}

        try:
            _assert_read_only_select(covariate_sql)
        except ValueError as err:
            return {"error": str(err)}

        user = get_mcp_user()

        async with AsyncSessionLocal() as app_db:
            try:
                descriptor, _ = await authorize_cohort(app_db, user, parsed_id, "editor")
            except CohortNotFoundError:
                return {"error": f"Cohort {cohort_id} not found or you do not have access to it."}
            except CohortAccessDeniedError:
                return {"error": "You need editor access to add covariates to this cohort."}

        if descriptor.origin != "user":
            return {"error": "Covariates can only be added to user cohorts."}

        await authorize_user_db(descriptor.database)
        total_steps = 3

        # ── Step 1: Execute covariate query under the user's warehouse access ──
        await ctx.report_progress(0, total_steps, "Executing covariate query")
        async with progress_keepalive(ctx, 0, total_steps, "Executing covariate query"):
            sf_token = await get_user_snowflake_token()
            async with get_db_as_user(descriptor.database, sf_token, database_schema=descriptor.database_schema) as rwd_db:
                cov_df = await fetch_cohort_dataframe(rwd_db, covariate_sql)

        id_col = _find_id_column(cov_df.columns)
        if id_col is None:
            return {"error": "covariate_sql must return a patient id column (person_id or subject_id)."}
        new_cov_cols = _covariate_columns(cov_df.columns)
        if not new_cov_cols:
            return {"error": "covariate_sql returned no covariate columns beyond the patient id."}

        try:
            rename_map = {c: _sanitize_identifier(c) for c in new_cov_cols}
        except ValueError as err:
            return {"error": str(err)}
        added = list(rename_map.values())
        cov_df = cov_df.rename(columns={id_col: "SUBJECT_ID", **rename_map})[["SUBJECT_ID", *added]]

        if cov_df["SUBJECT_ID"].duplicated().any():
            dupes = int(cov_df["SUBJECT_ID"].duplicated().sum())
            return {
                "error": (
                    f"covariate_sql returned {dupes} duplicate patient id(s). It must return "
                    "one row per patient, otherwise the cohort would gain a row per duplicate."
                )
            }

        # ── Step 2: Merge onto the existing cohort table and rebuild ──
        await ctx.report_progress(1, total_steps, "Merging covariates into cohort")
        async with progress_keepalive(ctx, 1, total_steps, "Merging covariates into cohort"):
            async with get_db("ASCENT") as ascent_db:
                base_df = await fetch_cohort_dataframe(ascent_db, rf"""SELECT * FROM ASCENT_COHORTS.{descriptor.table}""")
                base_df.columns = [str(c).upper() for c in base_df.columns]
                if collisions := [c for c in added if c in set(base_df.columns)]:
                    return {"error": f"Covariate column(s) already exist on the cohort: {collisions}. Adding covariates is add-only."}
                base_key, cov_key = _aligned_join_keys(base_df["SUBJECT_ID"], cov_df["SUBJECT_ID"])
                merged = base_df.assign(SUBJECT_ID=base_key).merge(cov_df.assign(SUBJECT_ID=cov_key), on="SUBJECT_ID", how="left")
                # A left join on unique right-hand keys cannot change the row
                # count. If it did, the keys did not line up and the cohort is
                # about to be replaced with a cartesian product.
                if len(merged) != len(base_df):
                    return {
                        "error": (
                            f"Covariate join did not line up: {len(base_df)} cohort rows became "
                            f"{len(merged)}. Check that covariate_sql returns the same patient id "
                            "as the cohort."
                        )
                    }
                matched = int(merged[added[0]].notna().sum())
                if not matched:
                    return {
                        "error": (
                            "Covariate join matched no patients. The id returned by covariate_sql "
                            "does not correspond to the cohort's SUBJECT_ID values."
                        )
                    }
                logger.info("Covariates matched %d of %d cohort patients", matched, len(base_df))
                # SUBJECT_ID must go back as it came: the join key is a
                # comparison form, not the stored value.
                merged["SUBJECT_ID"] = base_df["SUBJECT_ID"].to_numpy()
                table_info = await replace_cohort_columns(parsed_id, ascent_db, merged)

        # ── Step 3: Update descriptor ──
        await ctx.report_progress(2, total_steps, "Updating cohort descriptor")
        type_lookup = {col.name: col.type for col in table_info.columns}
        new_covariates = [
            CovariateInfo(name=name, type=type_lookup.get(name, "VARCHAR"), source="extended", definition=covariate_sql) for name in added
        ]
        patch = UserCohortDescriptorPatch(
            attributes=table_info.columns,
            size=table_info.size,
            covariates=list(descriptor.covariates or []) + new_covariates,
        )
        async with AsyncSessionLocal() as app_db:
            await update_cohort_descriptor(app_db, user, parsed_id, patch)

        await ctx.report_progress(total_steps, total_steps, "Done")
        span.set_attribute("cohort.added_covariates", len(added))

        return {
            "cohort_id": cohort_id,
            "added_covariates": added,
            "patient_count": table_info.size,
            "columns": [col.name for col in table_info.columns],
        }


# =============================================================================
# Tool: share_cohort
# =============================================================================


@mcp_v1.tool()
async def share_cohort(cohort_id: str, recipient_email: str, role: CohortRole = "viewer") -> dict:
    """
    Grant another user access to a cohort (owner only).

    Grants ``viewer`` (read-only) or ``editor`` (can rename/add covariates)
    access to the recipient. Ownership is unchanged — only the creator may share
    or delete. Re-sharing updates the recipient's role.

    Args:
        cohort_id: UUID of the cohort to share.
        recipient_email: Email address of the user to share with.
        role: "viewer" (default) or "editor".

    Returns:
        Dictionary containing success, cohort_id, shared_with, role.
    """
    with tracer.start_as_current_span("mcp_v1.share_cohort") as span:
        span.set_attribute("cohort.id", cohort_id)

        if role not in ("viewer", "editor"):
            return {"error": "role must be 'viewer' or 'editor'."}

        try:
            parsed_id = UUID(cohort_id)
        except ValueError:
            return {"error": f"Invalid cohort_id format: {cohort_id!r}. Expected a UUID."}

        user = get_mcp_user()

        try:
            async with AsyncSessionLocal() as app_db:
                await _share_cohort(parsed_id, recipient_email, app_db, user, role=role)
        except CohortNotFoundError:
            return {"error": f"Cohort {cohort_id} not found or you do not have access to it."}
        except CohortAccessDeniedError:
            return {"error": "Only the cohort owner can share it."}
        except CannotShareStudyCohortsException:
            return {"error": "Study cohorts cannot be shared. Only user-created cohorts can be shared."}

        return {
            "success": True,
            "cohort_id": cohort_id,
            "shared_with": recipient_email,
            "role": role,
        }


# =============================================================================
# Tool: rename_cohort
# =============================================================================


@mcp_v1.tool()
async def rename_cohort(cohort_id: str, name: str | None = None, description: str | None = None) -> dict:
    """
    Rename a cohort or update its description (requires editor or owner access).

    Args:
        cohort_id: UUID of the cohort.
        name: New cohort name (optional).
        description: New description (optional).

    Returns:
        Dictionary with id, name, description, state.
    """
    with tracer.start_as_current_span("mcp_v1.rename_cohort") as span:
        span.set_attribute("cohort.id", cohort_id)

        try:
            parsed_id = UUID(cohort_id)
        except ValueError:
            return {"error": f"Invalid cohort_id format: {cohort_id!r}. Expected a UUID."}
        if name is None and description is None:
            return {"error": "Provide a new name and/or description."}

        user = get_mcp_user()
        patch = UserCohortDescriptorPatch(state="active")
        if name is not None:
            patch.name = name
        if description is not None:
            patch.description = description

        try:
            async with AsyncSessionLocal() as app_db:
                descriptor = await update_cohort_descriptor(app_db, user, parsed_id, patch)
        except CohortNotFoundError:
            return {"error": f"Cohort {cohort_id} not found or you do not have access to it."}
        except CohortAccessDeniedError:
            return {"error": "You need editor access to rename this cohort."}

        return {
            "id": str(descriptor.id),
            "name": descriptor.name,
            "description": descriptor.description,
            "state": descriptor.state,
        }


# =============================================================================
# Tool: delete_cohort
# =============================================================================


@mcp_v1.tool()
async def delete_cohort(cohort_id: str) -> dict:
    """
    Delete a cohort (owner only).

    Removes the cohort and any access grants. The underlying cohort table is
    garbage-collected later. Only the creator (owner) may delete.

    Args:
        cohort_id: UUID of the cohort to delete.

    Returns:
        Dictionary with success, cohort_id.
    """
    with tracer.start_as_current_span("mcp_v1.delete_cohort") as span:
        span.set_attribute("cohort.id", cohort_id)

        try:
            parsed_id = UUID(cohort_id)
        except ValueError:
            return {"error": f"Invalid cohort_id format: {cohort_id!r}. Expected a UUID."}

        user = get_mcp_user()

        try:
            async with AsyncSessionLocal() as app_db:
                await delete_cohort_descriptors(app_db, user, {parsed_id})
        except CohortNotFoundError:
            return {"error": f"Cohort {cohort_id} not found or you do not have access to it."}
        except CohortAccessDeniedError:
            return {"error": "Only the cohort owner can delete it."}

        return {"success": True, "cohort_id": cohort_id}


# =============================================================================
# Tool: revoke_cohort_access
# =============================================================================


@mcp_v1.tool()
async def revoke_cohort_access(cohort_id: str, recipient_email: str) -> dict:
    """
    Revoke a user's access to a cohort (owner only).

    Args:
        cohort_id: UUID of the cohort.
        recipient_email: The user whose access to remove.

    Returns:
        Dictionary with success, cohort_id, revoked.
    """
    with tracer.start_as_current_span("mcp_v1.revoke_cohort_access") as span:
        span.set_attribute("cohort.id", cohort_id)

        try:
            parsed_id = UUID(cohort_id)
        except ValueError:
            return {"error": f"Invalid cohort_id format: {cohort_id!r}. Expected a UUID."}

        user = get_mcp_user()

        try:
            async with AsyncSessionLocal() as app_db:
                await _revoke_cohort_access(app_db, user, parsed_id, recipient_email)
        except CohortNotFoundError:
            return {"error": f"Cohort {cohort_id} not found or you do not have access to it."}
        except CohortAccessDeniedError:
            return {"error": "Only the cohort owner can manage access."}

        return {"success": True, "cohort_id": cohort_id, "revoked": recipient_email}


# =============================================================================
# Tool 7: generate_cohort_attrition
# =============================================================================


def _build_funnel_sql(cohort_sql: str) -> str | None:
    """Build deterministic funnel SQL from a non-OMOP cohort query.

    Extracts CTE names and the patient identifier column by parsing the
    stored cohort SQL, then generates UNION ALL COUNT(DISTINCT ...) selects
    for each CTE step. No LLM required.
    """
    patient_col_match = re.search(
        r"\bSELECT\s+DISTINCT\s+(?:\w+\.)?(\w+)\s+AS\s+SUBJECT_ID",
        cohort_sql,
        re.IGNORECASE,
    )
    if not patient_col_match:
        return None
    patient_col = patient_col_match.group(1)

    cte_names = re.findall(r"\b(\w+)\s+AS\s*\(", cohort_sql, re.IGNORECASE)
    if not cte_names:
        return None

    depth = 0
    final_select_pos = None
    sql_lower = cohort_sql.lower()
    i = 0
    while i < len(cohort_sql):
        ch = cohort_sql[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and sql_lower[i : i + 6] == "select":
            final_select_pos = i
        i += 1

    if final_select_pos is None:
        return None

    with_clause = cohort_sql[:final_select_pos].rstrip()

    union_parts = [(f"SELECT '{name}' AS funnel_step, COUNT(DISTINCT {patient_col}) AS distinct_patient_count FROM {name}") for name in cte_names]

    return with_clause + "\n" + "\nUNION ALL\n".join(union_parts)


def _make_criteria_service() -> CriteriaProcessor:
    return make_criteria_processor()


@mcp_v1.tool(task=True)
async def generate_cohort_attrition(cohort_id: str, ctx: Context) -> dict:
    """
    Generate an attrition table (patient funnel) for an existing cohort.

    For OMOP cohorts the query is split by inclusion/exclusion criteria and
    cumulative patient counts are computed. For non-OMOP cohorts the CTE
    structure is parsed deterministically from stored SQL.

    Only user-created cohorts are supported.

    Args:
        cohort_id: UUID of the existing cohort.

    Returns:
        Dictionary containing:
        - cohort: basic cohort info (id, name, size, database, table)
        - pipeline: "omop" or "non_omop"
        - funnel: list of attrition steps with counts
    """
    from uuid import UUID

    with tracer.start_as_current_span("mcp_v1.generate_cohort_attrition") as span:
        span.set_attribute("cohort_id", cohort_id)

        try:
            parsed_id = UUID(cohort_id)
        except ValueError:
            return {"error": f"Invalid cohort_id format: {cohort_id!r}. Expected a UUID."}

        user = get_mcp_user()
        try:
            async with AsyncSessionLocal() as app_db:
                descriptor = await get_cohort_descriptor(app_db, user, parsed_id)
        except CohortNotFoundError:
            return {"error": f"Cohort {cohort_id} not found or you do not have access to it."}

        if descriptor.origin != "user":
            return {
                "error": (
                    "Attrition is only available for user-created cohorts. "
                    f"Cohort {cohort_id} is a study cohort and does not have "
                    "the query definition needed to compute a funnel."
                )
            }

        database_name = descriptor.database
        database_schema = descriptor.database_schema
        span.set_attribute("database", database_name)

        await authorize_user_db(database_name)

        cohort_info = {
            "cohort_id": str(descriptor.id),
            "cohort_name": getattr(descriptor, "name", None),
            "cohort_size": descriptor.size,
            "database": database_name,
            "table_name": descriptor.table,
        }

        is_omop = is_omop_database(database_name)
        span.set_attribute("pipeline", "omop" if is_omop else "non_omop")

        if is_omop:
            return await _run_omop_attrition(descriptor, cohort_info, user, ctx)
        else:
            return await _run_non_omop_attrition(
                descriptor,
                cohort_info,
                database_name,
                database_schema,
                ctx,
            )


async def _run_omop_attrition(descriptor, cohort_info: dict, user, ctx: Context) -> dict:
    """OMOP attrition via criteria splitting."""
    if not descriptor.criteria or not descriptor.query_template:
        return {
            "error": (f"Cohort {cohort_info['cohort_id']} does not have criteria or a query template, so an attrition funnel cannot be computed.")
        }

    existing_funnel = getattr(descriptor, "funnel", None)
    if existing_funnel:
        return {
            "cohort": cohort_info,
            "pipeline": "omop",
            "cached": True,
            "funnel": [
                {
                    "index": f.index,
                    "description": f.description,
                    "type": f.type,
                    "size": f.size,
                    "query_template": f.query_template,
                    "query": f.query,
                }
                for f in existing_funnel
            ],
        }

    await ctx.report_progress(1, 2, "Computing patient attrition funnel")
    async with progress_keepalive(ctx, 1, 2, "Computing patient attrition funnel"):
        criteria_service = _make_criteria_service()
        async with AsyncSessionLocal() as app_db:
            funnel = await create_cohort_funnel(
                descriptor=descriptor,
                app_db=app_db,
                user=user,
                service=criteria_service,
            )

    await ctx.report_progress(2, 2, "Done")

    return {
        "cohort": cohort_info,
        "pipeline": "omop",
        "cached": False,
        "funnel": [
            {
                "index": f.index,
                "description": f.description,
                "type": f.type,
                "size": f.size,
                "query_template": f.query_template,
                "query": f.query,
            }
            for f in funnel
        ],
    }


async def _run_non_omop_attrition(
    descriptor,
    cohort_info: dict,
    database_name: str,
    database_schema: str | None,
    ctx: Context,
) -> dict:
    """Non-OMOP attrition via deterministic CTE parsing."""
    if not descriptor.query:
        return {"error": (f"Cohort {cohort_info['cohort_id']} does not have a stored SQL query, so an attrition funnel cannot be computed.")}

    funnel_sql = _build_funnel_sql(descriptor.query)
    if not funnel_sql:
        return {
            "error": (
                "Could not parse the cohort SQL to build a funnel. "
                "The query may not have the expected CTE structure with a "
                "SUBJECT_ID alias in the final SELECT."
            )
        }

    await ctx.report_progress(1, 2, "Executing attrition funnel SQL")
    async with progress_keepalive(ctx, 1, 2, "Executing attrition funnel SQL"):
        executor = user_sql_executor(database_name, database_schema, get_user_snowflake_token)
        result_df = await executor.execute_sql_query(funnel_sql)

    await ctx.report_progress(2, 2, "Done")

    if not hasattr(result_df, "columns") or "DISTINCT_PATIENT_COUNT" not in result_df.columns:
        return {
            "error": (
                "Funnel SQL executed but result is missing DISTINCT_PATIENT_COUNT column. "
                f"Columns returned: {getattr(result_df, 'columns', result_df)}"
            )
        }

    sorted_df = result_df.sort_values(by="DISTINCT_PATIENT_COUNT", ascending=False)

    return {
        "cohort": cohort_info,
        "pipeline": "non_omop",
        "funnel": [
            {
                "funnel_step": row["FUNNEL_STEP"],
                "distinct_patient_count": row["DISTINCT_PATIENT_COUNT"],
            }
            for row in sorted_df.to_dict(orient="records")
        ],
    }
