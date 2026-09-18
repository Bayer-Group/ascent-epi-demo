"""Generate the catalog artefacts the non-OMOP pipeline reads.

Profiles the built DuckDB warehouses and writes, per database and scale, a
METADATA document (``*.metadata.json``), an M-Schema rendering
(``*.mschema.txt``) and a ``manifest.<scale>.json``, into ``data/catalog/``.
``ascent_platform.warehouse.catalog`` reads those files, so the pipeline gets
prepared artefacts instead of profiling the warehouse at request time.

Run it after the warehouse exists:

    python scripts/build_catalog.py
    python scripts/build_catalog.py --scale 10k

The 1k output is committed, so a clone needs neither this script nor a
warehouse to start. Re-run it when the data or the descriptions overlay
changes.

Two choices worth recording, both because the build has to be reproducible
offline:

* **Sampling is stratified, not purely by frequency.** See ``_top_values``.
* **IS_WITH_DOTS is derived, not inferred by an LLM.** Whether a code has a
  dot in it is a fact about the string, and the data is right here, so a scan
  is exact, free, and needs no API key to build.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

from ascent_platform.warehouse.bootstrap import DATABASES, warehouse_path  # noqa: E402

# The renderer lives with the reader, so a built artefact and one rendered at
# request time cannot drift apart.
from ascent_platform.warehouse.catalog import render_m_schema  # noqa: E402

logger = logging.getLogger(__name__)

# The committed 1k catalog. Read-only in practice: it lives in the image.
OUT_DIR = REPO_ROOT / "data" / "catalog"


def generated_dir() -> Path:
    """Where a catalog built at startup is written.

    ``data/catalog`` when that is writable, which under compose it is: the
    directory is bind-mounted, so the artefacts land beside the committed 1k
    set and are visible from the host and the editor.

    When it is not writable -- a read-only mount, or the image's own copy with
    no bind mount, where anything written is discarded on the next recreate --
    it falls back to the warehouse work directory, which is a volume and
    already holds the DuckDB files being described.
    """
    if _is_writable(OUT_DIR):
        return OUT_DIR

    from ascent_platform.config.runtime import get_runtime_settings

    return Path(get_runtime_settings().WAREHOUSE_WORK_DIR) / "catalog"


def _is_writable(directory: Path) -> bool:
    """Whether *directory* can be created and written to.

    Probed rather than assumed: ``os.access`` reports on the mount, not on
    whether Docker has made it read-only, and a wrong answer here means a
    catalog silently written somewhere it will not be found.
    """
    import os

    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write-probe"
        probe.touch()
        os.unlink(probe)
        return True
    except OSError:
        return False


def search_dirs() -> list[Path]:
    """Where to look for a catalog, committed set first."""
    return [OUT_DIR, generated_dir()]


DESCRIPTIONS = REPO_ROOT / "data" / "catalog_descriptions.json"

# How many sampled values per column reach the metadata document. The M-Schema
# renderer shows only the first few of them.
_TOP_VALUES = 20

# Profiling every distinct value of a very wide column is slow and useless.
_MAX_DISTINCT_SCAN = 10_000

# A column whose name matches this is a coded column: its values come from a
# vocabulary, so which vocabularies appear matters more than which values are
# most frequent. Covers the code families the claims feed carries.
_CODE_COLUMN = re.compile(r"(DIAG|DX|ICD|COND|PROC|CPT|HCPCS|NDC|RX|DRUG|CODE|_CD$)", re.I)

# ICD-style code with a decimal point: a letter-or-digit stem, a dot, then more.
_DOTTED_CODE = re.compile(r"^[A-Za-z]?\d{2,3}\.\w+$")
_UNDOTTED_CODE = re.compile(r"^[A-Za-z]?\d{3,}$")


def _companion_type_column(column: str, columns: set[str]) -> str | None:
    """The ``*_TYPE`` column that says which vocabulary a coded column holds.

    ``DX.DX_CD`` carries both ICD-10 (``Q41.x``) and ICD-9 (``R90.x``) values,
    and ``DX_CD_TYPE`` is what distinguishes them. Finding this pairing is what
    lets the sampler show the model both.
    """
    lowered = {c.lower(): c for c in columns}
    for suffix in ("_type", "_typ", "_system", "_vocab"):
        candidate = lowered.get(f"{column.lower()}{suffix}")
        if candidate:
            return candidate
    return None


def _top_values(con: Any, table: str, column: str, dtype: str, columns: set[str]):
    """Sampled values for one column, as ``[(value, frequency)]``.

    Plain frequency ordering is wrong for coded columns and quietly so. In the
    shipped claims data ICD-10 rows outnumber ICD-9, so the three most frequent
    values of ``DX_CD`` are all ``Q41.x``; nothing in the M-Schema then hints
    that ``R90.x`` values exist at all. The SQL generator, reading only those
    examples, emits a single-vocabulary placeholder and silently loses every
    patient coded in the other system -- 16% of the hypertension cohort, with
    no error and a plausible-looking number.

    So when a coded column has a companion type column, the budget is split
    evenly across the distinct types. Both vocabularies reach the prompt, and
    the model can see it has to ask for both.
    """
    quoted = f'"{column}"'

    if dtype.upper() in {"DATE", "TIMESTAMP"}:
        row = con.execute(f'SELECT MIN({quoted}), MAX({quoted}) FROM "{table}"').fetchone()
        out = []
        if row and row[0] is not None:
            out.append((f"MIN: {row[0]}", 1))
        if row and row[1] is not None:
            out.append((f"MAX: {row[1]}", 1))
        return out

    distinct = int(con.execute(f'SELECT COUNT(DISTINCT {quoted}) FROM "{table}"').fetchone()[0])
    if not distinct or distinct > _MAX_DISTINCT_SCAN:
        return []

    type_column = None
    if _CODE_COLUMN.search(column):
        type_column = _companion_type_column(column, columns)

    if type_column is None:
        rows = con.execute(
            f'SELECT CAST({quoted} AS VARCHAR), COUNT(*) c FROM "{table}" '
            f"WHERE {quoted} IS NOT NULL GROUP BY 1 ORDER BY c DESC, 1 LIMIT {_TOP_VALUES}"
        ).fetchall()
        return [(str(v), int(c)) for v, c in rows]

    # Stratified: an equal share per vocabulary, largest vocabularies first so
    # a rounding remainder favours the dominant one.
    types = [
        r[0]
        for r in con.execute(
            f'SELECT CAST("{type_column}" AS VARCHAR), COUNT(*) c FROM "{table}" WHERE "{type_column}" IS NOT NULL GROUP BY 1 ORDER BY c DESC, 1'
        ).fetchall()
    ]
    if not types:
        return []

    per_type = max(1, _TOP_VALUES // len(types))
    by_type: list[list[tuple[str, int]]] = []
    for value_type in types:
        rows = con.execute(
            f'SELECT CAST({quoted} AS VARCHAR), COUNT(*) c FROM "{table}" '
            f'WHERE {quoted} IS NOT NULL AND CAST("{type_column}" AS VARCHAR) = ? '
            f"GROUP BY 1 ORDER BY c DESC, 1 LIMIT {per_type}",
            [value_type],
        ).fetchall()
        by_type.append([(str(v), int(c)) for v, c in rows])

    # Round-robin, not concatenated. The renderer shows only the first handful
    # of examples, so appending group after group puts the whole display budget
    # in the largest vocabulary -- which is the exact failure the stratification
    # is here to prevent, reintroduced one step later.
    # Deduplicated as well: type codes overlap (this feed uses both "DX10" and
    # "D10" for ICD-10), so the same value arrives from two groups and would
    # otherwise spend the budget twice on one vocabulary.
    sampled: list[tuple[str, int]] = []
    seen: set[str] = set()
    for rank in range(per_type):
        for group in by_type:
            if rank >= len(group):
                continue
            value, frequency = group[rank]
            if value in seen:
                continue
            seen.add(value)
            sampled.append((value, frequency))
    return sampled[:_TOP_VALUES]


def _profile_column(con: Any, table: str, column: str, dtype: str, columns: set[str]):
    stats = con.execute(f'SELECT COUNT(DISTINCT "{column}"), COUNT(*) - COUNT("{column}") FROM "{table}"').fetchone()
    distinct_count, null_count = int(stats[0]), int(stats[1])

    empty_count = 0
    if dtype.upper() in {"VARCHAR", "TEXT"}:
        empty_count = int(con.execute(f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" = \'\'').fetchone()[0])

    return {
        "name": column,
        "data_type": dtype,
        "nullable": null_count > 0,
        "null_count": null_count,
        "empty_count": empty_count,
        "distinct_count": distinct_count,
        "top_values": _top_values(con, table, column, dtype, columns),
        "description": None,
    }


def profile(con: Any, database: str, schema: str, descriptions: dict) -> dict[str, Any]:
    """The METADATA document for one database and schema."""
    con.execute(f'USE "{database}"."{schema}";')
    tables = [
        r[0]
        for r in con.execute(
            "SELECT table_name FROM duckdb_tables() WHERE database_name = ? AND schema_name = ? ORDER BY table_name",
            [database, schema],
        ).fetchall()
    ]

    documents = []
    for table in tables:
        columns = con.execute(
            "SELECT column_name, data_type FROM duckdb_columns() "
            "WHERE database_name = ? AND schema_name = ? AND table_name = ? "
            "ORDER BY column_index",
            [database, schema, table],
        ).fetchall()
        names = {name for name, _ in columns}
        row_count = int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])

        profiled = []
        for name, dtype in columns:
            entry = _profile_column(con, table, name, dtype, names)
            entry["description"] = descriptions.get(table, {}).get(name)
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


def detect_is_with_dots(document: dict[str, Any]) -> bool:
    """Whether this database stores ICD codes with their decimal point.

    Every sampled diagnosis code is in hand, so the answer is counted rather
    than guessed: a majority vote over the values of the coded columns, which
    is exact whenever the database is internally consistent and honest about
    it when it is not.
    """
    dotted = undotted = 0
    for table in document["tables"]:
        for column in table["columns"]:
            if not _CODE_COLUMN.search(column["name"]):
                continue
            for value, _ in column.get("top_values") or []:
                if _DOTTED_CODE.match(value):
                    dotted += 1
                elif _UNDOTTED_CODE.match(value):
                    undotted += 1
    return dotted > undotted


def build(scale: str, out_dir: Path | None = None) -> int:
    import duckdb

    from ascent_platform.config.runtime import get_runtime_settings

    work_dir = Path(get_runtime_settings().WAREHOUSE_WORK_DIR)
    descriptions = json.loads(DESCRIPTIONS.read_text()) if DESCRIPTIONS.exists() else {}
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for db in DATABASES:
        path = warehouse_path(work_dir, db, scale)
        if not path.exists():
            print(f"{path} is missing -- build the warehouse first", file=sys.stderr)
            return 1

        con = duckdb.connect(str(path), read_only=True)
        try:
            document = profile(con, db.name, db.schema, descriptions.get(db.name, {}))
        finally:
            con.close()

        is_omop = db.name.endswith("_OMOP")
        # The scale is in the filename for the same reason it is in the
        # warehouse filename: row counts, distinct counts and the sampled
        # frequencies are all facts about *this* dataset. Without it the 10k
        # stack loads the 1k catalog, reports 1,000-patient statistics for a
        # 10,000-patient warehouse, and raises nothing.
        stem = f"{db.name}.{db.schema}.{scale}"
        (out_dir / f"{stem}.metadata.json").write_text(json.dumps(document, indent=1))
        m_schema = render_m_schema(document)
        (out_dir / f"{stem}.mschema.txt").write_text(m_schema)

        entry = {
            "database": db.name,
            "schema": db.schema,
            "is_omop": is_omop,
            "is_synthetic": True,
            "metadata_file": f"{stem}.metadata.json",
            "mschema_file": f"{stem}.mschema.txt",
        }
        if not is_omop:
            entry["is_with_dots"] = detect_is_with_dots(document)
        manifest.append(entry)

        print(
            f"  {db.name}.{db.schema}: {document['tables_count']} tables, "
            f"{len(m_schema)} chars of M-Schema" + ("" if is_omop else f", is_with_dots={entry['is_with_dots']}")
        )

    (out_dir / f"manifest.{scale}.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nWrote {len(manifest)} catalogs to {out_dir}")
    return 0


def is_present(scale: str) -> bool:
    """Whether a complete catalog for *scale* is already on disk, anywhere."""
    for directory in search_dirs():
        if not (directory / f"manifest.{scale}.json").exists():
            continue
        if all((directory / f"{db.name}.{db.schema}.{scale}.{suffix}").exists() for db in DATABASES for suffix in ("metadata.json", "mschema.txt")):
            return True
    return False


def ensure(scale: str) -> bool:
    """Generate the catalog for *scale* unless it is already there.

    Called from the warehouse bootstrap, which runs once before the workers
    fork. That placement is deliberate on both counts: it needs the warehouse
    to exist (so it cannot go in coder-init), and it must not run per-worker
    (four workers writing the same files would race exactly as the warehouse
    build does).

    The 1k catalog ships in the repository, so the common path is a few
    ``exists()`` calls and nothing else. Larger scales have no committed
    catalog and generate one here, into the same volume as the warehouse, so
    the cost is paid once per dataset rather than per boot.

    Returns True when it generated, False when it skipped.
    """
    if is_present(scale):
        logger.info("catalog for %s already built", scale)
        return False
    logger.info("no catalog for %s -- generating", scale)
    build(scale, generated_dir())
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", default=None, help="dataset scale (default: WAREHOUSE_SCALE)")
    args = parser.parse_args()

    from ascent_platform.config.runtime import get_runtime_settings

    return build(args.scale or get_runtime_settings().WAREHOUSE_SCALE)


if __name__ == "__main__":
    raise SystemExit(main())
