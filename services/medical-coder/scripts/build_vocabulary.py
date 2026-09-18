"""Build the coder's vocabulary database from the shipped CSVs.

The coder needs only the concept tables -- CONCEPT, CONCEPT_ANCESTOR and
CONCEPT_RELATIONSHIP -- not patient data, so it reads them from a standalone
file rather than from the warehouse the analytics run on.

Here it gets its own small database built from ``data/synthetic/vocabulary``:
96 KB of CSV instead of opening a 59 MB patient file it has no business
reading, and the service boundary stays clean.

Of the three, CONCEPT_RELATIONSHIP matters most: its ``Maps to`` rows are the
source-to-standard mapping, so without it a raw code resolves to nothing.

Run:  python scripts/build_vocabulary.py [--data DIR] [--out FILE]
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("build-vocabulary")

# Columns the pipeline compares against integer literals. Everything else
# stays VARCHAR: names, codes and vocabulary ids are text, and the dates are
# YYYYMMDD strings that nothing here filters on by range.
_NUMERIC_COLUMNS = {
    ("CONCEPT", "concept_id"): "BIGINT",
    ("CONCEPT_ANCESTOR", "ancestor_concept_id"): "BIGINT",
    ("CONCEPT_ANCESTOR", "descendant_concept_id"): "BIGINT",
    ("CONCEPT_ANCESTOR", "min_levels_of_separation"): "INTEGER",
    ("CONCEPT_ANCESTOR", "max_levels_of_separation"): "INTEGER",
    ("CONCEPT_RELATIONSHIP", "concept_id_1"): "BIGINT",
    ("CONCEPT_RELATIONSHIP", "concept_id_2"): "BIGINT",
}

TABLES = ("CONCEPT", "CONCEPT_ANCESTOR", "CONCEPT_RELATIONSHIP")


def build(data_dir: Path, out: Path) -> dict[str, int]:
    import duckdb

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    con = duckdb.connect(str(out))
    extracted = out.parent / "_vocab_csv"
    extracted.mkdir(exist_ok=True)

    counts: dict[str, int] = {}
    for table in TABLES:
        archive = data_dir / f"{table}.csv.zip"
        if not archive.exists():
            raise FileNotFoundError(f"{archive} is missing")
        with zipfile.ZipFile(archive) as zf:
            member = next(n for n in zf.namelist() if n.endswith(".csv"))
            csv_path = extracted / f"{table}.csv"
            csv_path.write_bytes(zf.read(member))

        # The OMOP vocabulary files are tab-separated with a header row.
        # Read every column as text first -- the files are messy and letting
        # DuckDB sniff types fails the whole load on one odd row -- then cast
        # the columns that are compared numerically.
        con.execute(
            f"CREATE TABLE {table}_raw AS "
            f"SELECT * FROM read_csv_auto('{csv_path}', delim='\t', header=true, "
            f"all_varchar=true)"
        )
        # Typing these is not cosmetic. Loading everything as VARCHAR left
        # min_levels_of_separation as text, so the coder's drug enrichment --
        # `ca.min_levels_of_separation BETWEEN 1 AND 5` -- failed with
        # "Cannot mix values of type VARCHAR and INTEGER_LITERAL", and every
        # drug lookup returned a 500. Snowflake compared the two happily;
        # DuckDB does not. The concept ids have the same problem: they are
        # matched against integer literals from the vector search.
        projection = ", ".join(
            f"CAST({col} AS {_NUMERIC_COLUMNS[(table, col)]}) AS {col}" if (table, col) in _NUMERIC_COLUMNS else col
            for col in (
                r[0]
                for r in con.execute(
                    f"SELECT column_name FROM duckdb_columns() WHERE table_name = '{table}_raw'"
                ).fetchall()
            )
        )
        con.execute(f"CREATE TABLE {table} AS SELECT {projection} FROM {table}_raw")
        con.execute(f"DROP TABLE {table}_raw")
        counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logger.info("  %-22s %6d rows", table, counts[table])

    # The pipeline looks concepts up by code and by name, and walks the
    # hierarchy by ancestor; without these every lookup is a full scan.
    con.execute("CREATE INDEX idx_concept_code ON CONCEPT(concept_code)")
    con.execute("CREATE INDEX idx_concept_id ON CONCEPT(concept_id)")
    con.execute("CREATE INDEX idx_ancestor ON CONCEPT_ANCESTOR(ancestor_concept_id)")
    con.execute("CREATE INDEX idx_rel_source ON CONCEPT_RELATIONSHIP(concept_id_1)")
    con.close()
    for leftover in extracted.glob("*.csv"):
        leftover.unlink()
    extracted.rmdir()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/synthetic/vocabulary"))
    parser.add_argument("--out", type=Path, default=Path("/data/vocabulary.duckdb"))
    args = parser.parse_args()

    logger.info("Building vocabulary from %s", args.data)
    counts = build(args.data, args.out)
    logger.info("Wrote %s (%.1f MB)", args.out, args.out.stat().st_size / 1e6)
    if counts.get("CONCEPT_RELATIONSHIP", 0) == 0:
        logger.error("CONCEPT_RELATIONSHIP is empty; code lookup will resolve nothing")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
