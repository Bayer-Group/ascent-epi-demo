import re

from ascent_domain.non_omop.medical_coding.sql_postprocessing import filter_medical_coding_ontology
from ascent_platform.warehouse.sql_executor import SqlExecutor

# These four queries interpolate caller-supplied identifiers into SQL and run on
# the MACHINE-USER session (SqlExecutor with no connection_provider falls back to
# the key-pair principal), so they execute with the service account's grants
# rather than the caller's OBO grants -- the per-user access model does not
# contain them.
#
# Reachable unvalidated from GET /api/metadata/{database_name}/{database_schema},
# GET /api/database-columns-search, POST /api/sql-preparation, and the MCP
# get_database_schema / get_ontology_info tools. Those MCP tools already validate
# `database` with the same pattern but never `schema`.
#
# This is a containment fix, not the real one: the executor should take bound
# parameters (the machine pool is already paramstyle="qmark").
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,254}$")


def _validate_identifier(name: str, label: str) -> str:
    if not isinstance(name, str) or not _SAFE_IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid {label}: {name!r}")
    return name


async def get_tables_with_medical_coding_ontologies(executor: SqlExecutor | None, database_name: str, database_schema: str):
    if executor is None:
        executor = SqlExecutor("ASCENT", "PUBLIC")

    _validate_identifier(database_name, "database name")
    _validate_identifier(database_schema, "database schema")
    # Annotations are inferred from the data: a column whose sampled values are
    # concept codes in the shipped vocabulary is reported with the vocabularies
    # those codes belong to. See warehouse.catalog.
    from ascent_platform.warehouse import catalog
    from ascent_platform.warehouse.executor import run_db

    document = await run_db(catalog.ontology_metadata, database_name, database_schema, _catalog_dir())
    # filter_medical_coding_ontology reads json_data["tables"].
    return filter_medical_coding_ontology({"tables": document.get("ontology_tables", [])})


def _catalog_dir():
    from pathlib import Path

    from ascent_platform.config.runtime import get_runtime_settings

    return Path(get_runtime_settings().WAREHOUSE_WORK_DIR) / "catalog"


async def get_m_schema(executor: SqlExecutor | None, database_name: str, database_schema: str):
    if executor is None:
        executor = SqlExecutor("ASCENT", "PUBLIC")

    _validate_identifier(database_name, "database name")
    _validate_identifier(database_schema, "database schema")
    # Generated from the profiled catalog; see ascent_platform.warehouse.catalog.
    from ascent_platform.warehouse import catalog
    from ascent_platform.warehouse.executor import run_db

    return await run_db(catalog.m_schema, database_name, database_schema, _catalog_dir())


async def get_metadata(executor: SqlExecutor | None, database_name: str, database_schema: str):
    if executor is None:
        executor = SqlExecutor("ASCENT", "PUBLIC")

    _validate_identifier(database_name, "database name")
    _validate_identifier(database_schema, "database schema")
    # The full profiled document: tables and their columns with sampled
    # values. Not the ontology annotations -- extract_metadata indexes
    # json_metadata["tables"] to peek at the columns the planner asked for.
    from ascent_platform.warehouse import catalog
    from ascent_platform.warehouse.executor import run_db

    return await run_db(catalog.load, database_name, database_schema, _catalog_dir())


async def get_custom_instructions(executor: SqlExecutor | None, database_name: str, database_schema: str):
    """
    This asynchronous function queries a non-OMOP custom instructions table and fetches the associated instruction
    for a given database name. An optional SQL executor can be provided; otherwise,
    a default executor with pre-set configurations is used.

    Args:
        executor (SqlExecutor | None): The SQL executor used to query the database.
                                       If None, a default executor pointing to the
                                       "ASCENT" database and "PUBLIC" schema is used.
        database_name (str): Name of the database for which instructions are
                             requested.
        database_schema (str): Name of the schema within the database. Currently not
                               used in the query logic but included as part of the
                               arguments.

    Returns:
        str: The custom instruction string retrieved from the database. If no
             instruction is available, an empty string is returned.
    """
    if executor is None:
        executor = SqlExecutor("ASCENT", "PUBLIC")

    _validate_identifier(database_name, "database name")
    instruction_query = f"select INSTRUCTION from ASCENT_CUSTOM_INSTRUCTIONS_NON_OMOP where database = '{database_name}'"

    instruction = await executor.execute_sql_query(instruction_query)

    try:
        return instruction["INSTRUCTION"][0]
    except Exception:
        return ""
