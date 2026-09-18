"""
Experimental MCP tools — database operations and medical code lookup.

All tools are registered on the `mcp_experimental` FastMCP instance so they
appear exclusively in the Experimental MCP server.
"""

from typing import List, Literal, Optional

from fastmcp import Context

from ascent_mcp._tools_shared import (
    OMOP_PLACEHOLDER_PATTERN as OMOP_PLACEHOLDER_PATTERN,
)
from ascent_mcp._tools_shared import (
    PLACEHOLDER_PATTERN as PLACEHOLDER_PATTERN,
)
from ascent_mcp._tools_shared import (
    _describe_table_handler,
    _execute_sql_handler,
    _get_database_schema_handler,
    _get_ontology_info_handler,
    _get_sample_data_handler,
    _list_placeholder_concepts_handler,
    _list_tables_handler,
    _lookup_medical_codes_handler,
    _resolve_placeholders_handler,
)
from ascent_mcp.app_experimental import mcp_experimental
from ascent_mcp.error_handling import resilient_tool

# =============================================================================
# Database tools
# =============================================================================


@mcp_experimental.tool
@resilient_tool(timeout_seconds=60)
async def get_database_schema(
    database: str,
    schema: str,
    tables: Optional[List[str]] = None,
) -> dict:
    """
    Get the M-schema representation of database tables and columns.

    Essential for understanding database structure before writing SQL queries.
    Returns table names, column types, primary keys, and sample values.

    Args:
        database: Database name (e.g. SYNTHETIC_CLAIMS)
        schema: Schema name (e.g. DATA_202601)
        tables: Optional list of specific table names to include

    Returns:
        Dictionary with database, schema, and the m_schema text.
    """
    return await _get_database_schema_handler(database, schema, tables)


# Task-mode tool: long-running queries are managed by FastMCP's
# Docket so they don't get capped by the asyncio.wait_for timeout in
# resilient_tool. Progress events are streamed back to the client.
@mcp_experimental.tool(task=True)
async def execute_sql(database: str, schema: str, query: str, ctx: Context) -> dict:
    """
    Execute a SQL query and return results.

    Args:
        database: Database name
        schema: Schema name
        query: SQL query to execute. Write Snowflake dialect -- it is
            transpiled to the dialect the warehouse actually speaks.

    Returns:
        Dictionary with success flag, row_count, columns, data, and duration.
        At most 100 rows are returned; row_count reports the full size and
        truncated says whether data was cut, so do not compute totals from the
        rows you can see.
    """
    return await _execute_sql_handler(database, schema, query, ctx)


@mcp_experimental.tool
@resilient_tool(timeout_seconds=60)
async def get_ontology_info(database: str, schema: str) -> dict:
    """
    Get information about which columns contain medical coding ontologies.

    Returns tables and columns tagged with their coding systems (ICD-10, NDC, SNOMED, etc.).
    Essential for knowing which columns need medical code lookups.

    Args:
        database: Database name
        schema: Schema name

    Returns:
        Dictionary with a list of tables containing medical coding columns.
    """
    return await _get_ontology_info_handler(database, schema)


@mcp_experimental.tool
@resilient_tool(timeout_seconds=30)
async def list_tables(database: str, schema: str) -> dict:
    """
    List all tables in a schema.

    Args:
        database: Database name
        schema: Schema name

    Returns:
        Dictionary with the list of table names and count.
    """
    return await _list_tables_handler(database, schema)


@mcp_experimental.tool
@resilient_tool(timeout_seconds=30)
async def describe_table(database: str, schema: str, table_name: str) -> dict:
    """
    Get detailed table structure including column names, types, and nullability.

    Args:
        database: Database name
        schema: Schema name
        table_name: Table to describe

    Returns:
        Dictionary with the table name and its column definitions.
    """
    return await _describe_table_handler(database, schema, table_name)


@mcp_experimental.tool
@resilient_tool(timeout_seconds=30)
async def get_sample_data(database: str, schema: str, table_name: str, limit: int = 5) -> dict:
    """
    Preview first N rows from a table.

    Args:
        database: Database name
        schema: Schema name
        table_name: Table to sample
        limit: Number of rows (default 5, max 20)

    Returns:
        Dictionary with the sample data rows.
    """
    return await _get_sample_data_handler(database, schema, table_name, limit)


# =============================================================================
# Medical Coder tools
# =============================================================================


# Task-mode tool: vector search + LLM filter against the remote medical-coder
# service can take well over 90s for broad queries; Docket lifts that cap.
@mcp_experimental.tool(task=True)
async def lookup_medical_codes(
    query: str,
    ctx: Context,
    domain_ids: list[str] | None = None,
    vocabulary: list[str] | None = None,
    # "bge" is the only index this stack builds: seed_qdrant.py embeds the
    # shipped vocabulary with BAAI/bge-base-en-v1.5 on CPU. Any other default
    # would point lookups at a collection that does not exist.
    encoder: Literal["bge", "sap", "gemini", "biolord"] = "bge",
    llm_filter: Literal["default", "gemini", "chatgpt", "llama3", "haiku"] | None = None,
    top_k: int = 10000,
    standard_concept: str | None = None,
    cosine_similarity: float | None = None,
    database: str | None = None,
    source: str | None = None,
    allow_lts: bool = False,
    # Off by default in this distribution. The fallback asks Gemini +
    # Google for codes when the local index finds none, which
    # searches a real OMOP vocabulary. Here the vocabulary is synthetic
    # (PHARMLEX, DXCODE, ...), so the fallback answers with real-world
    # RxNorm and ATC codes that match zero patients in the shipped data --
    # authoritative-looking and wrong, which is worse than an empty result.
    # Pass True explicitly if you want it.
    google_search_fallback: bool = False,
    custom_instructions: str | None = None,
    use_lts: bool = False,
    use_hybrid: bool = False,
    with_reasoning: bool = False,
) -> dict:
    """
    Translate a plain-language medical concept description into standardized medical codes.

    Uses the Ascent Medical Coder service, which performs vector search over the
    vocabulary this stack ships combined with optional LLM-based filtering to
    produce high-precision code lists.

    The shipped vocabulary is synthetic: its coding systems are DXCODE and
    DXCODE9 (conditions), PHARMLEX, MEDLEX and DRUGPKG (drugs), PXCODE
    (procedures), LABLEX and LABLOCAL (measurements) and ENCTYPE. Asking for
    ICD10CM, RxNorm, LOINC or CPT returns nothing, because no concept here
    carries them.

    Examples:
    - lookup_medical_codes("type 2 diabetes", domain_ids=["Condition"], vocabulary=["DXCODE", "DXCODE9"])
    - lookup_medical_codes("metformin", domain_ids=["Drug"], vocabulary=["PHARMLEX"])
    - lookup_medical_codes("colonoscopy", domain_ids=["Procedure"], vocabulary=["PXCODE"])
    - lookup_medical_codes("hypertension", with_reasoning=True)

    Args:
        query: The medical concept to search for (e.g. "diabetes", "metformin", "blood pressure").
        domain_ids: Filter by OMOP domain. Examples: ["Condition"], ["Procedure"], ["Drug"],
                    ["Drug_class"], ["Measurement"], ["Observation"]. Leave empty for all domains.
                    Note: Drug/Drug_class cannot be mixed with non-drug domains in a single request.
        vocabulary: Filter by vocabulary. Examples: ["DXCODE"], ["DXCODE9"],
                    ["PHARMLEX"], ["DXCODE", "DXCODE9"]. None for all vocabularies.
        encoder: Embedding model for vector search.
                 - "bge": BGE base (general-purpose) — the only index this stack ships
                 - "sap": SapBERT (biomedical-specialized)
                 - "gemini": Gemini embeddings (higher cosine cutoff) — requires a
                   separately built concepts-gemini-hybrid-new collection
                 - "biolord": BioLORD (biomedical)
        llm_filter: LLM model to post-filter results for relevance. None disables filtering.
                    - "default": configured DEFAULT_LLM_FILTER (shipped: "gemini")
                    - "haiku", "chatgpt", "gemini": configured filter connectors
                    - "llama3": advertised here but unsupported by the bundled service
                    Filtering applies to general search, not explicit Drug/Drug_class lookup.
        top_k: Retrieval candidate limit (default: 10000), not a final result cap:
               descendant enrichment can add concepts.
        standard_concept: Filter by OMOP standard concept flag.
                         - "S": Standard concepts only
                         - "C": Classification concepts only
                         - None: All concepts
        cosine_similarity: Custom cosine similarity threshold (0.0-1.0). Overrides the
                          encoder's default cutoff (0.65 for bge/sap/biolord, 0.84 for gemini).
        database: Optional OMOP database for precomputed patient-count enrichment.
                  Count tables are not shipped; PATIENT_COUNT may be null. Use backend
                  SQL for counts against the demo patient warehouse.
        source: Filter by source vocabulary (e.g. "DXCODE", "PHARMLEX").
        allow_lts: Attempt reuse of previous LLM filter decisions. The required
                   decision-cache table is absent from the shipped vocabulary.
        google_search_fallback: Fall back to Gemini + Google Search when vector search
                                returns no results (default: False). Be aware this path
                                is slow for broad, class-level terms such as "NSAID"
                                (minutes, versus seconds for a specific product name
                                like "celecoxib"). If a lookup is taking too long, pass
                                False or narrow ``domain_ids`` / ``vocabulary``.
        custom_instructions: Custom instructions appended to the LLM filter prompt to guide
                            code selection (e.g. "Only include codes for Type 2 diabetes").
        use_lts: Enable the optional response cache for reasoning requests. Basic
                 lookups use it whenever configured. Default Compose leaves it disabled.
        use_hybrid: Use hybrid search combining dense vectors + sparse BM25-like matching
                    (default: False, dense only).
        with_reasoning: Return per-domain counts and available filter explanations
                        (default: False). Does not enable llm_filter; explanations can
                        be absent if no filter ran or a response cache was used.

    Returns:
        When with_reasoning=False (default):
            Dictionary mapping the query to a list of coded concepts. Each concept contains:
            - CONCEPT_ID, CONCEPT_NAME, CONCEPT_CODE, VOCABULARY_ID
            - IS_VALID (false when the concept is marked invalid — deprecated or upgraded —
              in the OMOP/Athena vocabulary; review or exclude such codes as needed)
            - PATIENT_COUNT (if database provided, else null)
            - SIMILARITY_SCORE

        When with_reasoning=True:
            Dictionary with:
            - results: coded concepts grouped by query/domain
            - llm_reasoning: reasoning metadata keyed by domain under ``per_domain``,
              including counts and any filter include/exclude explanations
            - lts_used: whether the response reused long-term-storage cache results
    """
    return await _lookup_medical_codes_handler(
        query=query,
        ctx=ctx,
        domain_ids=domain_ids,
        vocabulary=vocabulary,
        encoder=encoder,
        llm_filter=llm_filter,
        top_k=top_k,
        standard_concept=standard_concept,
        cosine_similarity=cosine_similarity,
        database=database,
        source=source,
        allow_lts=allow_lts,
        google_search_fallback=google_search_fallback,
        custom_instructions=custom_instructions,
        use_lts=use_lts,
        use_hybrid=use_hybrid,
        with_reasoning=with_reasoning,
    )


# Task-mode: each placeholder may trigger a medical-coder lookup, so total
# runtime scales with placeholder count and easily exceeds a 60s wait_for cap.
@mcp_experimental.tool(task=True)
async def resolve_placeholders(
    sql_with_placeholders: str,
    ctx: Context,
    database: Optional[str] = None,
    coding_system: str = "Standard",
    codes_with_dots: bool = False,
    use_concept_ids: bool = True,
) -> dict:
    """
    Replace medical code placeholders in SQL with real codes. Both dialects:

    - **OMOP** 2-segment ``[entity@value]`` (e.g. ``[condition@chronic kidney disease]``)
      — vocabulary implied by the domain; values resolve via the OMOP CONCEPT
      table. **Requires** ``database``. Honours ``coding_system``.
    - **Non-OMOP** 3-segment ``[entity@name@coding_systems]`` (e.g.
      ``[condition@diabetes@DXCODE,DXCODE9]``) — vocabulary explicit; values
      resolve via the medical coder. ``database`` selects the dot convention.

    Both formats may co-exist in one query; each placeholder is routed by arity.

    Args:
        sql_with_placeholders: SQL with ``[entity@value]`` and/or
            ``[entity@name@coding_systems]`` placeholders.
        database: Target database being queried (e.g. ``SYNTHETIC_EHR_OMOP`` or
            ``SYNTHETIC_CLAIMS``). **Required.** Selects the OMOP vocabulary for
            2-segment placeholders and the ICD dot convention
            (``ASCENT_NON_OMOP_METADATA.IS_WITH_DOTS``) for 3-segment ones.
        coding_system: (OMOP only) ``"Standard"`` (default) emits standard
            CONCEPT_IDs into ``*_concept_id`` columns; ``"Source"`` emits source
            concept IDs into ``*_source_concept_id`` columns.
        codes_with_dots: (3-segment only) Keep dots in source codes
            (``'E11.9'`` vs ``'E119'``).
        use_concept_ids: (3-segment only) When True, emit integer OMOP
            CONCEPT_IDs; when False, emit quoted source codes.
    """
    return await _resolve_placeholders_handler(
        sql_with_placeholders=sql_with_placeholders,
        ctx=ctx,
        database=database,
        coding_system=coding_system,
        codes_with_dots=codes_with_dots,
        use_concept_ids=use_concept_ids,
    )


@mcp_experimental.tool
@resilient_tool()
async def list_placeholder_concepts(sql: str) -> dict:
    """
    Extract and list all medical placeholders from SQL, in both supported formats.

    - OMOP (2-segment): ``[entity@value]`` — vocabulary implied by domain.
    - Non-OMOP (3-segment): ``[entity@name@coding_systems]`` — vocabulary explicit;
      the coding_systems segment may be a single system (e.g. DXCODE) or a
      comma-separated list (e.g. DXCODE,DXCODE9).

    Args:
        sql: SQL to analyze

    Returns:
        Dictionary with the count and list of placeholders found. Each entry has a
        ``format`` of ``"non-omop"`` or ``"omop"``; ``coding_systems`` is null for
        OMOP placeholders.
    """
    return await _list_placeholder_concepts_handler(sql)
