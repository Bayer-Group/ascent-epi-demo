"""Run the QA questions from the query library against the new synthetic 1k OMOP data.

Ad-hoc evaluation script, not part of the shipped pipeline. It resolves
[domain@term] placeholders against the new vocabulary by name match + concept_ancestor
expansion, transpiles Snowflake SQL to DuckDB via sqlglot, and executes against
data/synthetic_new/omop_synthetic_1k.db directly (the live MCP server is still wired
to the old data/synthetic warehouse).
"""

import argparse
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import duckdb
import sqlglot

DEFAULT_QUERYLIB = Path("data/synthetic_new/querylib_20251216.db")
DEFAULT_OMOP_DB = Path("data/synthetic_new/omop_synthetic_1k.db")
DEFAULT_OUTPUT = Path("data/synthetic_new/qa_results_1k.json")

DOMAIN_MAP = {
    "condition": "Condition",
    "drug": "Drug",
    "procedure": "Procedure",
    "measurement": "Measurement",
    "observation": "Observation",
    "visit": "Visit",
}

PLACEHOLDER_RE = re.compile(r"\[([a-zA-Z_]+)@([^\]]+)\]")


def _require_input(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    return path


def _sqlite_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def load_qa_rows(query_library: Path):
    with closing(_sqlite_read_only(query_library)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT ID, QUESTION, QUERY_SNOWFLAKE_WITH_PLACEHOLDERS FROM queries WHERE QUESTION_TYPE='QA' ORDER BY ID").fetchall()
    return [dict(row) for row in rows]


def setup_duckdb(omop_db: Path):
    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    escaped_path = omop_db.as_posix().replace("'", "''")
    con.execute(f"ATTACH '{escaped_path}' AS omop (TYPE sqlite, READ_ONLY);")
    tables = [r[0] for r in con.execute("SELECT table_name FROM information_schema.tables WHERE table_catalog='omop'").fetchall()]

    try:
        with closing(_sqlite_read_only(omop_db)) as source:
            for table in tables:
                cols = [row[1] for row in source.execute("SELECT * FROM pragma_table_info(?)", [table]).fetchall()]
                selects = []
                for column in cols:
                    quoted = _quote_identifier(column)
                    lower_column = column.lower()
                    if lower_column.endswith("_datetime"):
                        selects.append(f"TRY_CAST({quoted} AS TIMESTAMP) AS {quoted}")
                    elif lower_column.endswith("_date"):
                        selects.append(f"TRY_CAST({quoted} AS DATE) AS {quoted}")
                    else:
                        selects.append(quoted)
                quoted_table = _quote_identifier(table)
                con.execute(f"CREATE VIEW {quoted_table} AS SELECT {', '.join(selects)} FROM omop.{quoted_table};")
        return con
    except Exception:
        con.close()
        raise


def resolve_term(con, domain_tag, term):
    domain = DOMAIN_MAP.get(domain_tag)
    if domain is None:
        return None, f"unknown domain tag '{domain_tag}'"

    term_norm = term.strip().lower()
    words = [w for w in re.split(r"\W+", term_norm) if w]

    # exact match
    q = """
        SELECT concept_id FROM omop.concept
        WHERE domain_id = ? AND lower(concept_name) = ?
    """
    matched = [r[0] for r in con.execute(q, [domain, term_norm]).fetchall()]
    reason = "exact"

    # substring match
    if not matched:
        q = """
            SELECT concept_id FROM omop.concept
            WHERE domain_id = ? AND lower(concept_name) LIKE ?
        """
        matched = [r[0] for r in con.execute(q, [domain, f"%{term_norm}%"]).fetchall()]
        reason = "substring"

    # all-words-present (any order) match
    if not matched and words:
        clauses = " AND ".join(["lower(concept_name) LIKE ?" for _ in words])
        params = [domain] + [f"%{w}%" for w in words]
        q = f"SELECT concept_id FROM omop.concept WHERE domain_id = ? AND {clauses}"
        matched = [r[0] for r in con.execute(q, params).fetchall()]
        reason = "all-words"

    if not matched:
        return None, "no matching concept in new vocabulary"

    # expand to descendants via concept_ancestor (self-inclusive per data invariants)
    placeholders = ",".join("?" * len(matched))
    q = f"""
        SELECT DISTINCT descendant_concept_id FROM omop.concept_ancestor
        WHERE ancestor_concept_id IN ({placeholders})
    """
    descendants = [r[0] for r in con.execute(q, matched).fetchall()]
    final_ids = sorted(set(descendants) or set(matched))
    return (
        final_ids,
        f"{reason} ({len(matched)} direct -> {len(final_ids)} incl. descendants)",
    )


def process_row(con, row):
    sql = row["QUERY_SNOWFLAKE_WITH_PLACEHOLDERS"]
    placeholders = PLACEHOLDER_RE.findall(sql or "")
    resolutions = []
    unresolved = []
    substituted = sql
    for domain_tag, term in placeholders:
        ids, reason = resolve_term(con, domain_tag, term)
        resolutions.append(
            {
                "domain": domain_tag,
                "term": term,
                "reason": reason,
                "n_ids": len(ids) if ids else 0,
            }
        )
        if ids is None:
            unresolved.append(f"{domain_tag}@{term}: {reason}")
            continue
        token = f"[{domain_tag}@{term}]"
        substituted = substituted.replace(token, ",".join(str(i) for i in ids))

    if unresolved:
        return {
            "id": row["ID"],
            "question": row["QUESTION"],
            "status": "skipped",
            "reason": "; ".join(unresolved),
            "resolutions": resolutions,
        }

    try:
        translated = sqlglot.transpile(substituted, read="snowflake", write="duckdb")[0]
    except Exception as e:
        return {
            "id": row["ID"],
            "question": row["QUESTION"],
            "status": "transpile_error",
            "reason": str(e),
            "resolutions": resolutions,
        }

    try:
        cur = con.execute(translated)
        cols = [d[0] for d in cur.description]
        row_count = 0
        result_rows = []
        while batch := cur.fetchmany(1_000):
            row_count += len(batch)
            remaining = 20 - len(result_rows)
            if remaining > 0:
                result_rows.extend(dict(zip(cols, row)) for row in batch[:remaining])
        return {
            "id": row["ID"],
            "question": row["QUESTION"],
            "status": "ok",
            "columns": cols,
            "row_count": row_count,
            "rows": result_rows,
            "resolutions": resolutions,
            "sql": translated,
        }
    except Exception as e:
        return {
            "id": row["ID"],
            "question": row["QUESTION"],
            "status": "exec_error",
            "reason": str(e),
            "resolutions": resolutions,
            "sql": translated,
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-library", type=Path, default=DEFAULT_QUERYLIB)
    parser.add_argument("--omop-db", type=Path, default=DEFAULT_OMOP_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        query_library = _require_input(args.query_library, "Query library")
        omop_db = _require_input(args.omop_db, "OMOP database")
    except FileNotFoundError as error:
        raise SystemExit(f"error: {error}") from None
    rows = load_qa_rows(query_library)
    with closing(setup_duckdb(omop_db)) as con:
        results = [process_row(con, row) for row in rows]

    summary = {}
    for result in results:
        summary[result["status"]] = summary.get(result["status"], 0) + 1

    print(json.dumps({"total": len(results), "summary": summary}, indent=2))

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        json.dump(results, file, indent=2, default=str)


if __name__ == "__main__":
    main()
