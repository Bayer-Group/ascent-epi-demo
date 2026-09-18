"""The data catalog: what tables exist, and what the columns look like.

Discovery metadata is built from the warehouse itself rather than read from
out-of-band control-plane tables. An absent catalog returns no rows rather
than failing, which would leave every tool answering "no databases" while the
stack looked healthy, so the artefacts are committed and checked on startup.

Here the catalog is computed from the warehouse instead of maintained beside
it. Row counts, distinct counts, null counts and top values are all facts
about the data that DuckDB can answer directly, so profiling once at build
time is both cheaper and impossible to get out of sync -- switch to the 10k
dataset and the statistics follow.

Column *descriptions* cannot be derived, and they matter: they are what the
model reads when deciding which column holds a diagnosis code. Those come
from a curated overlay keyed by table and column, applied on top of the
generated structure.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from ascent_platform.warehouse.bootstrap import DATABASES

logger = logging.getLogger(__name__)

# Profiling a very wide column's distinct values is pointless and slow; these
# bounds keep a build over the 10k dataset in the same order of magnitude.
_TOP_VALUES = 5
_MAX_DISTINCT_SCAN = 10_000


def _profile_column(con: Any, table: str, column: str, dtype: str) -> dict[str, Any]:
    q = f'"{column}"'
    stats = con.execute(f'SELECT COUNT(DISTINCT {q}), COUNT(*) - COUNT({q}) FROM "{table}"').fetchone()
    distinct_count, null_count = int(stats[0]), int(stats[1])

    empty_count = 0
    if dtype.upper() in {"VARCHAR", "TEXT"}:
        empty_count = int(con.execute(f"SELECT COUNT(*) FROM \"{table}\" WHERE {q} = ''").fetchone()[0])

    top_values: list[tuple[str, int]] = []
    if 0 < distinct_count <= _MAX_DISTINCT_SCAN:
        rows = con.execute(
            f'SELECT CAST({q} AS VARCHAR), COUNT(*) c FROM "{table}" WHERE {q} IS NOT NULL GROUP BY 1 ORDER BY c DESC LIMIT {_TOP_VALUES}'
        ).fetchall()
        top_values = [(str(v), int(c)) for v, c in rows]

    return {
        "name": column,
        "data_type": dtype,
        "nullable": null_count > 0,
        "distinct_count": distinct_count,
        "null_count": null_count,
        "empty_count": empty_count,
        "top_values": top_values or None,
    }


def profile(database: str, schema: str, descriptions: Optional[dict] = None) -> dict[str, Any]:
    """Build the metadata document for one database and schema."""
    from ascent_platform.warehouse.session import root_connection

    con = root_connection().cursor()
    con.execute(f'USE "{database}"."{schema}";')
    tables = [
        r[0]
        for r in con.execute(
            "SELECT table_name FROM duckdb_tables() WHERE database_name = ? AND schema_name = ? ORDER BY table_name",
            [database, schema],
        ).fetchall()
    ]

    described = descriptions or {}
    documents = []
    for table in tables:
        columns = con.execute(
            "SELECT column_name, data_type FROM duckdb_columns() "
            "WHERE database_name = ? AND schema_name = ? AND table_name = ? "
            "ORDER BY column_index",
            [database, schema, table],
        ).fetchall()
        row_count = int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])

        profiled = []
        for name, dtype in columns:
            entry = _profile_column(con, table, name, dtype)
            entry["description"] = described.get(table, {}).get(name)
            profiled.append(entry)

        documents.append(
            {
                "name": table,
                "row_count": row_count,
                "columns_count": len(profiled),
                "columns": profiled,
            }
        )

    return {
        "database_name": database,
        "schema_name": schema,
        "tables": documents,
        "tables_count": len(documents),
    }


def build(out_dir: Path, descriptions_path: Optional[Path] = None) -> dict[str, Path]:
    """Profile every database and write one metadata document apiece."""
    descriptions: dict[str, Any] = {}
    if descriptions_path and descriptions_path.exists():
        descriptions = json.loads(descriptions_path.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for db in DATABASES:
        document = profile(db.name, db.schema, descriptions.get(db.name, {}))
        target = out_dir / f"{db.name}.{db.schema}.json"
        target.write_text(json.dumps(document, indent=1))
        written[db.name] = target
        logger.info("catalogued %s.%s: %d tables", db.name, db.schema, document["tables_count"])
    return written


# Artefacts written by scripts/build_catalog.py and committed. Prefer them over
# profiling: the build step samples coded columns per vocabulary, which
# request-time profiling cannot do cheaply, and it is that sampling the SQL
# generator depends on to see that DX_CD holds both ICD-10 and ICD-9.
_PREBUILT = Path(__file__).resolve().parents[3] / "data" / "catalog"


def _scale() -> str:
    from ascent_platform.config.runtime import get_runtime_settings

    return get_runtime_settings().WAREHOUSE_SCALE


def prebuilt_metadata(database: str, schema: str) -> dict[str, Any] | None:
    """The committed metadata document for the *current* scale, if there is one.

    Keyed on the scale, and deliberately returning None rather than another
    scale's file: the counts inside are facts about one dataset, so serving the
    1k catalog to a 10k warehouse would answer with confident wrong numbers.
    Falling through to live profiling is slower but always describes the data
    actually loaded.
    """
    target = _find(f"{database}.{schema}.{_scale()}.metadata.json")
    return json.loads(target.read_text()) if target else None


def prebuilt_m_schema(database: str, schema: str) -> str | None:
    target = _find(f"{database}.{schema}.{_scale()}.mschema.txt")
    return target.read_text() if target else None


def _find(name: str) -> Path | None:
    """Locate a catalog artefact: committed set first, then the generated one."""
    from ascent_platform.warehouse.catalog_build import search_dirs

    for directory in search_dirs():
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def load(database: str, schema: str, out_dir: Path) -> dict[str, Any]:
    """Read a metadata document, profiling it first if absent."""
    document = prebuilt_metadata(database, schema)
    if document is not None:
        return document
    target = out_dir / f"{database}.{schema}.json"
    if not target.exists():
        build(out_dir)
    return json.loads(target.read_text())


# Databases any caller may list without an explicit grant. Every shipped
# database is in that position.
PUBLIC_DATABASES = frozenset(db.name for db in DATABASES)


M_SCHEMA_EXAMPLES = 6
"""Sampled values shown per column.

Six rather than three: with a smaller budget the shown values of a
two-vocabulary column all still come from the first vocabulary, which defeats
the point of sampling it.
"""


def render_m_schema(document: dict[str, Any], examples: int = M_SCHEMA_EXAMPLES) -> str:
    """The M-Schema text the prompts embed, in the layout upstream emits.

    One ``【DB_ID】``/``【Schema】`` header, then one block per table, one line
    per column, carrying the description and the sampled values that tell a
    model what the column holds.
    """
    blocks = [f"【DB_ID】 {document['database_name']}", "【Schema】"]
    for table in document["tables"]:
        lines = [f"# Table: {table['name'].lower()}", "["]
        for column in table["columns"]:
            described = f", {column['description']}" if column.get("description") else ""
            samples = [str(v) for v, _ in (column.get("top_values") or [])][:examples]
            shown = f", Examples: [{', '.join(samples)}]" if samples else ""
            lines.append(f"({column['name'].lower()}:{column['data_type']}{described}{shown}),")
        lines.append("]")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def m_schema(database: str, schema: str, out_dir: Path) -> str:
    """The M-Schema for one database, preferring a prebuilt artefact."""
    prebuilt = prebuilt_m_schema(database, schema)
    if prebuilt is not None:
        return prebuilt

    return render_m_schema(load(database, schema, out_dir))


def ontology_metadata(database: str, schema: str, out_dir: Path) -> dict[str, Any]:
    """The per-column coding-system annotations get_ontology_info returns.

    Which columns hold medical codes, and in which vocabulary, is inferred
    from the data rather than curated: a
    column whose sampled values match concept codes in the shipped vocabulary
    is reported with the vocabularies those codes come from. Inference, and
    labelled as such, rather than an empty answer that reads as "no ontologies
    here".
    """
    document = load(database, schema, out_dir)
    tables = []
    for table in document["tables"]:
        columns = []
        for column in table["columns"]:
            samples = [str(v) for v, _ in (column.get("top_values") or [])]
            vocabularies = _vocabularies_for(samples)
            if vocabularies:
                columns.append(
                    {
                        "column_name": column["name"],
                        "coding_system": "/".join(sorted(vocabularies)),
                        "is_medical_coding_ontology": True,
                        "examples": ", ".join(samples[:3]) or None,
                    }
                )
        if columns:
            tables.append({"table_name": table["name"], "columns": columns})
    return {"database": database, "schema": schema, "ontology_tables": tables}


def _vocabularies_for(samples: list[str]) -> set[str]:
    """Which shipped vocabularies contain these values as concept codes."""
    if not samples:
        return set()
    from ascent_platform.warehouse.session import root_connection

    con = root_connection().cursor()
    placeholders = ", ".join("?" for _ in samples)
    try:
        rows = con.execute(
            f"SELECT DISTINCT vocabulary_id FROM SYNTHETIC_EHR_OMOP.CDM.concept WHERE concept_code IN ({placeholders})",
            samples,
        ).fetchall()
    except Exception:  # noqa: BLE001 - inference must never break discovery
        return set()
    return {r[0] for r in rows if r[0]}
