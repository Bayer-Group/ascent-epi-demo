"""Implementation shared by the v1 and experimental MCP tool surfaces.

These two servers expose the same tools with different docstrings and a
different tool set: v1 additionally registers get_omop_databases and
get_non_omop_databases.

The @mcp.tool registrations deliberately stay in the two surface modules: the
docstrings are the model-facing contract and are legitimately per-server. What
lives here is only the logic, which must never differ.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List

import pandas as pd
from fastapi import HTTPException

from ascent_domain.errors import DomainError
from ascent_domain.non_omop.medical_coding.sql_postprocessing import (
    get_medical_concepts_from_sql_async,
)
from ascent_domain.non_omop.metadata_queries import (
    get_m_schema,
    get_tables_with_medical_coding_ontologies,
)
from ascent_mcp.tools.permissions import assert_user_permission_db, authorize_user_db, get_user_snowflake_token
from ascent_platform.warehouse.user_session import execute_user_sql_to_df, user_sql_executor

logger = logging.getLogger(__name__)


# Regex for safe warehouse identifiers — alphanumeric + underscore, max 255
# chars.
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,254}$")

# Maximum number of rows returned from execute_sql to prevent OOM on large
# tables.
_MAX_RESULT_ROWS = 100

# Non-OMOP (3-segment): vocabulary is explicit, e.g.
# [condition@diabetes@ICD10CM]. The middle/right groups exclude '[' and ']'
# so two adjacent 2-segment OMOP placeholders can't greedily bridge into a
# single bogus 3-segment match.
PLACEHOLDER_PATTERN = re.compile(r"\[(\w+)@([^@\[\]]+)@([^\[\]]+)\]")

# OMOP (2-segment): vocabulary implied by the domain, e.g.
# [condition@diabetes]. Value class mirrors
# ascent_domain.omop.MedicalSQLProcessor and excludes '@', so this pattern is
# disjoint from the 3-segment one.
OMOP_PLACEHOLDER_PATTERN = re.compile(r"\[([a-zA-Z_]+)@([a-zA-Z0-9_/\-\(\)\'\\ ]+)\]")

# Substituted for 3-segment placeholders that resolve to zero codes so the
# SQL still parses (matches nothing) instead of producing an invalid ``IN ()``.
_NO_CODES_SENTINEL_SOURCE = "'__NO_CODES__'"

_medical_coder = None

_medical_coder_lock = asyncio.Lock()


async def _get_medical_coder():
    """Return the shared MedicalCoder singleton, creating it on first use.

    Uses an asyncio.Lock to prevent concurrent requests from racing to
    create multiple instances (the check-then-act on the global is not
    atomic without a lock).
    """
    global _medical_coder
    if _medical_coder is not None:  # fast path — no lock needed once initialised
        return _medical_coder
    async with _medical_coder_lock:
        if _medical_coder is None:  # double-check after acquiring the lock
            from ascent_platform.external import MedicalCoder

            _medical_coder = MedicalCoder()
    return _medical_coder


def _df_to_serializable(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a DataFrame to a list of dicts with JSON-safe values.

    ``date_format="iso"`` is not cosmetic. pandas serialises datetime columns as
    epoch numbers by default, which would present a DATE column to the model as
    1443571200 rather than "2015-09-30" -- and the model reads sample rows to
    learn what a column holds, so it would write SQL comparing dates to integers.
    """
    return json.loads(df.to_json(orient="records", date_format="iso", default_handler=str))


def _validate_identifier(name: str, label: str) -> str:
    """Validate that a warehouse identifier is safe for SQL interpolation.

    Only allows alphanumeric characters and underscores, starting with a letter
    or underscore, up to 255 characters. Raises ValueError for invalid input.
    """
    if not _SAFE_IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid {label}: only alphanumeric characters and underscores are allowed.")
    return name


def _identifier_sql_literal(name: str, label: str) -> str:
    """Return a validated identifier encoded as a SQL string literal.

    INFORMATION_SCHEMA filters compare against string values, not identifiers.
    ``SqlExecutor`` currently accepts raw SQL only, so safety here comes from
    constraining the interpolated value to ``_SAFE_IDENTIFIER_RE`` first.
    """
    return f"'{_validate_identifier(name, label)}'"


def _identifier_matches(column: str, name: str, label: str) -> str:
    """Return a case-insensitive INFORMATION_SCHEMA predicate for an identifier.

    The catalog does not agree with itself about case. Snowflake folds unquoted
    identifiers to upper, so a table is ``PERSON`` there; the DuckDB warehouse
    this repo ships preserves the case it was created with, so the same table is
    ``person`` while its schema is still ``CDM``. An exact ``=`` would match the
    schema and miss the table, making ``describe_table('PERSON')`` return an
    empty column list with no error -- indistinguishable, to a caller, from a
    table that genuinely has no columns.

    Comparing both sides folded keeps one predicate correct on either backend.
    """
    return f"UPPER({column}) = UPPER({_identifier_sql_literal(name, label)})"


# Verbs that reach outside the attached warehouse. enable_external_access=false
# closes the filesystem and the network, but the postgres extension is loaded
# before the lockdown and ATTACH ... (TYPE postgres) still dials out, which
# would let caller SQL exfiltrate to an arbitrary host and write to the cohort
# store. Nothing a query tool legitimately does needs these.
_ESCAPE_VERBS = re.compile(r"(?is)\b(attach|detach|install|load|copy|export|import)\b")


def _assert_no_escape(query: str) -> None:
    match = _ESCAPE_VERBS.search(query)
    if match:
        raise ValueError(f"`{match.group(1).upper()}` is not allowed in a warehouse query.")


async def _execute_sql_impl(database: str, schema: str, query: str) -> dict:
    """Run a warehouse query and return a serialisable result dict.

    Shared by the ``execute_sql`` tool and other tools (e.g. ``get_sample_data``)
    that need to run SQL without the ``Context`` parameter that the task-mode
    tool requires.

    The allowlist gate is unconditional: the warehouse has no role model of its
    own, so this check is the only thing bounding which databases a tool call
    can reach.
    """
    await authorize_user_db(database)
    try:
        _assert_no_escape(query)
    except ValueError as err:
        return {"success": False, "error": str(err), "query": query, "duration_seconds": 0.0}
    start = time.time()
    try:
        result = await execute_user_sql_to_df(database, schema, get_user_snowflake_token, query)
    except (DomainError, HTTPException) as e:
        duration = round(time.time() - start, 2)
        return {"success": False, "error": e.detail, "query": query, "duration_seconds": duration}
    except Exception as e:  # noqa: BLE001 — surface any execution error to the tool caller
        duration = round(time.time() - start, 2)
        return {"success": False, "error": str(e), "query": query, "duration_seconds": duration}
    duration = round(time.time() - start, 2)

    if isinstance(result, pd.DataFrame):
        row_count = len(result)
        truncated = row_count > _MAX_RESULT_ROWS
        if truncated:
            result = result.head(_MAX_RESULT_ROWS)

        return {
            "success": True,
            "row_count": row_count,
            "truncated": truncated,
            "duration_seconds": duration,
            "columns": list(result.columns),
            "data": _df_to_serializable(result),
        }

    return {"success": True, "result": str(result), "duration_seconds": duration}


async def _resolve_non_omop_placeholders(
    sql: str,
    codes_with_dots: bool,
    use_concept_ids: bool,
    database: str | None = None,
) -> tuple[str, list[dict]]:
    """Resolve 3-segment ``[entity@name@vocab]`` placeholders via the MedicalCoder.

    Vocabulary is explicit in each placeholder, so coding does not need a
    database. Uses the async fan-out helper (parallel medical-coder calls
    per placeholder) instead of the sync version that would block the event
    loop for the duration of every coder roundtrip. Returns
    ``(resolved_sql, resolutions)``.

    When ``database`` is given, the library checks that database's recorded
    ``IS_WITH_DOTS`` flag and uses it to drive dot formatting; the explicit
    ``codes_with_dots`` is only the fallback when the flag is unset.
    """
    entities = await get_medical_concepts_from_sql_async(sql, codes_with_dots, database=database)

    resolved_sql = sql
    resolutions: list[dict] = []
    for entity in entities:
        placeholder = f"[{entity['placeholder']}]"
        concepts = entity.get("concepts", [])
        if concepts:
            if use_concept_ids:
                codes_str = ", ".join(str(c["CONCEPT_ID"]) for c in concepts if c.get("CONCEPT_ID"))
            else:
                codes_str = entity.get("codes_str", "")
            resolved_sql = resolved_sql.replace(placeholder, codes_str or _NO_CODES_SENTINEL_SOURCE)
            sample = [c.get("CONCEPT_ID", "") for c in concepts[:5]] if use_concept_ids else [c.get("CONCEPT_CODE", "") for c in concepts[:5]]
            resolutions.append({"placeholder": placeholder, "format": "non-omop", "code_count": len(concepts), "sample": sample})
        else:
            resolved_sql = resolved_sql.replace(placeholder, _NO_CODES_SENTINEL_SOURCE)
            resolutions.append({"placeholder": placeholder, "format": "non-omop", "code_count": 0, "warning": "No codes found"})
    return resolved_sql, resolutions


async def _resolve_omop_placeholders(
    sql: str,
    database: str,
    coding_system: str,
    omop_matches: list[tuple[str, str]],
) -> tuple[str, list[dict]]:
    """Resolve 2-segment ``[entity@value]`` placeholders to OMOP CONCEPT_IDs.

    Reuses the CLUES QA wiring (``init_qa_system`` + ``post_process_query``) so
    warehouse session, concept caching, and Standard/Source coding behave
    identically to ``generate_query_filled``.
    On ``ConceptNotFoundError`` the original SQL is returned with a per-
    placeholder warning rather than half-filled SQL.
    """
    from ascent_domain.omop.data.processing.sql_post_processor import ConceptNotFoundError
    from ascent_mcp.tools._clues_helpers import init_qa_system, to_coding_type

    await assert_user_permission_db(database)
    unique_matches = list(dict.fromkeys(omop_matches))
    qa_service, _ = await init_qa_system(database, database_schema=None, token_fetcher=get_user_snowflake_token)

    try:
        resolved_sql = await qa_service.post_process_query(sql, preferred_coding_system=to_coding_type(coding_system))
        resolutions = [{"placeholder": f"[{etype}@{value}]", "format": "omop", "resolved": True} for etype, value in unique_matches]
        return resolved_sql, resolutions
    except ConceptNotFoundError as exc:
        logger.warning("OMOP placeholder resolution failed for database=%s: %s", database, exc)
        resolutions = [
            {
                "placeholder": f"[{etype}@{value}]",
                "format": "omop",
                "resolved": False,
                "warning": f"Could not resolve one or more OMOP concepts: {exc}",
            }
            for etype, value in unique_matches
        ]
        return sql, resolutions


async def _get_database_schema_handler(
    database: str,
    schema: str,
    tables: list[str] | None = None,
) -> dict:
    await assert_user_permission_db(database)

    try:
        m_schema_text = await get_m_schema(
            executor=None,
            database_name=database,
            database_schema=schema,
        )
    except Exception as e:
        return {"database": database, "schema": schema, "error": str(e)}

    if not m_schema_text:
        return {"database": database, "schema": schema, "m_schema": f"No schema found for {database}.{schema}"}

    if tables:
        table_set = {t.upper() for t in tables}
        lines = m_schema_text.split("\n")
        filtered: list[str] = []
        include = False
        for line in lines:
            if line.startswith("# Table:"):
                table_part = line.split(":")[1].split(",")[0].strip() if ":" in line else ""
                table_name = table_part.split(".")[-1].upper()
                include = table_name in table_set
            # The 【DB_ID】/【Schema】 header names the database a filtered
            # subset still belongs to, so it survives the filter.
            if include or line.startswith("【") or not line.strip():
                filtered.append(line)
        m_schema_text = "\n".join(filtered)

    return {"database": database, "schema": schema, "m_schema": m_schema_text}


async def _get_ontology_info_handler(database: str, schema: str) -> dict:
    await assert_user_permission_db(database)

    try:
        ontology_tables = await get_tables_with_medical_coding_ontologies(
            executor=None,
            database_name=database,
            database_schema=schema,
        )
    except Exception as e:
        return {"database": database, "schema": schema, "error": str(e)}

    return {"database": database, "schema": schema, "ontology_tables": ontology_tables}


async def _execute_sql_handler(
    database: str,
    schema: str,
    query: str,
    ctx: Any,
) -> dict:
    await ctx.report_progress(0, 2, "Authorising and executing SQL")
    result = await _execute_sql_impl(database, schema, query)
    await ctx.report_progress(2, 2, "SQL execution complete")
    return result


async def _list_tables_handler(database: str, schema: str) -> dict:
    await authorize_user_db(database)
    safe_database = _validate_identifier(database, "database")
    schema_matches = _identifier_matches("TABLE_SCHEMA", schema, "schema")
    executor = user_sql_executor(safe_database, schema, get_user_snowflake_token)
    result = await executor.execute_sql_query(f"SELECT TABLE_NAME FROM {safe_database}.INFORMATION_SCHEMA.TABLES WHERE {schema_matches}")

    if isinstance(result, dict) and "error" in result:
        return {"error": result["error"]}

    if isinstance(result, pd.DataFrame):
        tables = result["TABLE_NAME"].tolist() if "TABLE_NAME" in result.columns else []
        return {"tables": tables, "count": len(tables)}

    return {"tables": [], "count": 0}


async def _describe_table_handler(database: str, schema: str, table_name: str) -> dict:
    await authorize_user_db(database)
    safe_database = _validate_identifier(database, "database")
    schema_matches = _identifier_matches("TABLE_SCHEMA", schema, "schema")
    table_matches = _identifier_matches("TABLE_NAME", table_name, "table_name")
    executor = user_sql_executor(safe_database, schema, get_user_snowflake_token)
    result = await executor.execute_sql_query(
        f"SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE "
        f"FROM {safe_database}.INFORMATION_SCHEMA.COLUMNS "
        f"WHERE {schema_matches} AND {table_matches} "
        f"ORDER BY ORDINAL_POSITION"
    )

    if isinstance(result, dict) and "error" in result:
        return {"table": table_name, "error": result["error"]}

    if isinstance(result, pd.DataFrame):
        columns = [
            {"name": row.get("COLUMN_NAME", ""), "type": row.get("DATA_TYPE", ""), "nullable": row.get("IS_NULLABLE", "YES")}
            for row in _df_to_serializable(result)
        ]
        return {"table": table_name, "columns": columns}

    return {"table": table_name, "columns": []}


async def _get_sample_data_handler(database: str, schema: str, table_name: str, limit: int = 5) -> dict:
    safe_limit = min(limit, 20)
    await assert_user_permission_db(database)
    _validate_identifier(database, "database")
    _validate_identifier(schema, "schema")
    _validate_identifier(table_name, "table_name")
    return await _execute_sql_impl(database, schema, f"SELECT * FROM {database}.{schema}.{table_name} LIMIT {safe_limit}")


async def _lookup_medical_codes_handler(
    query: str,
    ctx: Any,
    domain_ids: list[str] | None = None,
    vocabulary: list[str] | None = None,
    encoder: str = "bge",
    llm_filter: str | None = None,
    top_k: int = 10000,
    standard_concept: str | None = None,
    cosine_similarity: float | None = None,
    database: str | None = None,
    source: str | None = None,
    allow_lts: bool = False,
    google_search_fallback: bool = False,
    custom_instructions: str | None = None,
    use_lts: bool = False,
    use_hybrid: bool = False,
    with_reasoning: bool = False,
) -> dict:
    logger.info("lookup_medical_codes invoked: query=%r, domain_ids=%s, encoder=%s", query, domain_ids, encoder)

    await ctx.report_progress(0, 3, "Preparing medical coding request")

    if database is not None:
        await assert_user_permission_db(database)

    kwargs: dict[str, Any] = {
        "domain_ids": domain_ids or [],
        "encoder": encoder,
        "top_k": top_k,
        "allow_lts": allow_lts,
        "google_search_fallback": google_search_fallback,
        "use_lts": use_lts,
        "use_hybrid": use_hybrid,
    }

    if llm_filter is not None:
        kwargs["llm_filter"] = llm_filter
    if vocabulary is not None:
        kwargs["vocabulary"] = vocabulary
    if standard_concept is not None:
        kwargs["standard_concept"] = standard_concept
    if cosine_similarity is not None:
        kwargs["cosine_similarity"] = cosine_similarity
    if database is not None:
        kwargs["database"] = database
    if source is not None:
        kwargs["source"] = source
    if custom_instructions is not None:
        kwargs["custom_instructions"] = custom_instructions

    await ctx.report_progress(1, 3, "Calling medical coder service")

    coder = await _get_medical_coder()
    if with_reasoning:
        result = await coder.get_medical_codes_reasoning(query, **kwargs)
        concept_count = sum(len(v) for v in result.get("results", {}).values())
    else:
        result = await coder.get_medical_codes(query, **kwargs)
        concept_count = sum(len(v) for v in result.values())

    await ctx.report_progress(3, 3, "Medical coding complete")
    logger.info("lookup_medical_codes completed: query=%r, with_reasoning=%s, result_count=%d", query, with_reasoning, concept_count)

    return result


async def _resolve_placeholders_handler(
    sql_with_placeholders: str,
    ctx: Any,
    database: str | None = None,
    coding_system: str = "Standard",
    codes_with_dots: bool = False,
    use_concept_ids: bool = True,
) -> dict:
    three_segment = PLACEHOLDER_PATTERN.findall(sql_with_placeholders)
    omop_matches = OMOP_PLACEHOLDER_PATTERN.findall(sql_with_placeholders)

    if not three_segment and not omop_matches:
        return {"message": "No placeholders found", "resolved_sql": sql_with_placeholders}

    if not database:
        raise ValueError(
            "`database` is required to resolve placeholders. For 2-segment OMOP "
            "placeholders it selects the OMOP vocabulary; for 3-segment non-OMOP "
            "placeholders it selects the database's ICD dot convention "
            "(ASCENT_NON_OMOP_METADATA.IS_WITH_DOTS) so codes are emitted the way "
            "the target database stores them. Pass the database being queried."
        )

    total = len(set(three_segment)) + len(dict.fromkeys(omop_matches))
    await ctx.report_progress(0, total, "Resolving placeholders")

    resolved_sql = sql_with_placeholders
    resolutions: list[dict] = []

    if three_segment:
        resolved_sql, non_omop_resolutions = await _resolve_non_omop_placeholders(resolved_sql, codes_with_dots, use_concept_ids, database)
        resolutions.extend(non_omop_resolutions)
        await ctx.report_progress(len(resolutions), total, f"Resolved {len(resolutions)}/{total} placeholders")

    if omop_matches:
        resolved_sql, omop_resolutions = await _resolve_omop_placeholders(resolved_sql, database, coding_system, omop_matches)
        resolutions.extend(omop_resolutions)
        await ctx.report_progress(total, total, f"Resolved {total}/{total} placeholders")

    return {
        "placeholders_resolved": len(resolutions),
        "resolutions": resolutions,
        "resolved_sql": resolved_sql,
    }


async def _list_placeholder_concepts_handler(sql: str) -> dict:
    # 3-segment non-OMOP placeholders (bounded pattern avoids bridging across
    # adjacent placeholders — see PLACEHOLDER_PATTERN comment).
    placeholders = []
    seen_three: set[tuple[str, str, str]] = set()
    for entity_type, entity_name, coding_systems in PLACEHOLDER_PATTERN.findall(sql):
        if (entity_type, entity_name, coding_systems) in seen_three:
            continue
        seen_three.add((entity_type, entity_name, coding_systems))
        placeholders.append(
            {
                "placeholder": f"[{entity_type}@{entity_name}@{coding_systems}]",
                "format": "non-omop",
                "entity_type": entity_type,
                "entity_name": entity_name,
                "coding_systems": coding_systems,
            }
        )

    # 2-segment OMOP placeholders (disjoint from the 3-segment matches above).
    seen_omop: set[tuple[str, str]] = set()
    for entity_type, value in OMOP_PLACEHOLDER_PATTERN.findall(sql):
        if (entity_type, value) in seen_omop:
            continue
        seen_omop.add((entity_type, value))
        placeholders.append(
            {
                "placeholder": f"[{entity_type}@{value}]",
                "format": "omop",
                "entity_type": entity_type,
                "entity_name": value,
                "coding_systems": None,
            }
        )

    return {"placeholder_count": len(placeholders), "placeholders": placeholders}
