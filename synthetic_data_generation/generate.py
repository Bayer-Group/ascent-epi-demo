#!/usr/bin/env python3
"""Generate a fictitious but clinically coherent EMR and concept table for RWE demos."""

import argparse
import json
import os
import sys
from datetime import date
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)
load_dotenv(os.path.join(project_root, ".env"))

from catalog import load_catalog, validate_catalog  # noqa: E402
from expansion import expand_vocabulary  # noqa: E402
from ground_truth import build_ground_truth, write_ground_truth, write_known_quirks  # noqa: E402
from omop_writer import CDM_VERSION, write_omop_sqlite  # noqa: E402
from pathways import generate_events  # noqa: E402
from patients import assign_causes_of_death, sample_population, simulate_mortality  # noqa: E402
from source_writer import write_source_sqlite  # noqa: E402
from validate import print_report, validate_databases  # noqa: E402
from vocabulary import build_vocabulary, write_athena_zips  # noqa: E402

DEFAULT_OUT_DIR = os.path.join(project_root, "data")
DEFAULT_PATIENTS = 10000
DEFAULT_SEED = 42
DEFAULT_START = "2010-01-01"
# The extract runs to the labelled release period rather than stopping two years
# short of it. A file called 202601 whose last record is from December 2023 makes
# every recency, run-out and trend analysis wrong by construction, and it is the
# first thing anyone notices.
DEFAULT_END = "2026-01-31"

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", 5432),
    "database": os.getenv("POSTGRES_DATABASE"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic OMOP CDM database, a messy source EMR, and provable ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                  # 10,000 patients, seed 42, all outputs
  %(prog)s --patients 1000                  # Small dataset for a quick iteration
  %(prog)s --seed 7 --messiness nasty       # Different population, heavier data defects
  %(prog)s --stages concepts                # Only rebuild the vocabulary ZIPs
  %(prog)s --postgres                       # Also load into the docker-compose database

Outputs (in --out-dir, default data/). Patient files are stamped with the cohort
size, so a 1k and a 10k dataset can sit side by side:
  CONCEPT.csv.zip               Vocabulary in the Athena CSV layout OMOP loaders expect
  CONCEPT_RELATIONSHIP.csv.zip  Source-to-standard mappings
  CONCEPT_ANCESTOR.csv.zip      Hierarchy for descendant expansion
  omop_synthetic_10k.db         OMOP CDM 5.4 (SQLite)
  source_synthetic_10k.db       Messy non-OMOP source EMR (SQLite)
  ground_truth_10k.json         Cohort counts, prevalence and lab stats computed from the finished data

Everything clinical is fictitious: patients, codes and vocabularies are invented. Code formats,
naming grammar and vocabulary names are all original, and no licensed terminology's codes,
hierarchy or mappings are redistributed. Two structural exceptions: a small set of OMOP concept
ids (8532 and the gender/race/unit equivalents) is reproduced verbatim so the data reads
correctly in OHDSI tooling, and twenty observation/death concepts carry their real OMOP ids and
real standard names under vocabulary_id 'SNOMED'. See the README for what that means and how to
remove it.

To use a freshly generated dataset in the demo stack, zip the databases into data/synthetic/
under the scale the stack asks for:
  zip -j data/synthetic/omop_synthetic_1k.db.zip   <out-dir>/omop_synthetic_1k.db
  zip -j data/synthetic/source_synthetic_1k.db.zip <out-dir>/source_synthetic_1k.db
  cp <out-dir>/CONCEPT*.csv.zip data/synthetic/vocabulary/
Then: DATA_SCALE=1k docker compose up
""",
    )
    parser.add_argument("--patients", type=int, default=DEFAULT_PATIENTS, help=f"Number of patients (default: {DEFAULT_PATIENTS})")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help=f"Random seed; identical seeds reproduce identical data (default: {DEFAULT_SEED})"
    )
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory (default: data/)")
    parser.add_argument("--start-date", type=str, default=DEFAULT_START, help=f"Start of the observation window (default: {DEFAULT_START})")
    parser.add_argument("--end-date", type=str, default=DEFAULT_END, help=f"End of the observation window (default: {DEFAULT_END})")
    parser.add_argument(
        "--messiness",
        choices=["clean", "realistic", "nasty"],
        default="realistic",
        help="How many data-quality defects to inject into the source EMR (default: realistic)",
    )
    parser.add_argument("--stages", type=str, default="concepts,emr,ground-truth,validate", help="Comma-separated stages to run (default: all)")
    parser.add_argument("--postgres", action="store_true", help="Also load into PostgreSQL schemas omop and source_emr")
    parser.add_argument("--no-sqlite", action="store_true", help="Skip writing the SQLite databases")
    parser.add_argument("--no-expand", action="store_true", help="Emit only the curated concepts, without sibling code expansion")
    args = parser.parse_args()

    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if start >= end:
        print("Error: --start-date must precede --end-date")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading catalog...")
    cat = load_catalog()
    errors = validate_catalog(cat)
    if errors:
        print(f"Error: catalog has {len(errors)} problems:")
        for e in errors[:20]:
            print(f"  - {e}")
        return 1
    print(f"  {len(cat.conditions)} conditions, {len(cat.drugs)} drugs, {len(cat.labs)} labs, {len(cat.procedures)} procedures")

    # Independent child streams, so changing --patients does not perturb the
    # vocabulary and vice versa.
    root = np.random.default_rng(args.seed)
    rng_pop, rng_events, rng_source = root.spawn(3)

    vocab = build_vocabulary(cat)
    if not args.no_expand:
        added = expand_vocabulary(cat, vocab, root)
        print(f"  expanded vocabulary with {added:,} sibling concepts")

    if "concepts" in stages:
        print(f"\nWriting vocabulary ({len(vocab.concepts):,} concepts)...")
        written = write_athena_zips(vocab, args.out_dir)
        for name, path in written.items():
            print(f"  {name:22s} -> {os.path.basename(path)}")

    # Size-stamped names, so datasets of different scales coexist instead of
    # silently overwriting each other.
    tag = _scale_tag(args.patients)
    omop_db = os.path.join(args.out_dir, f"omop_synthetic_{tag}.db")
    source_db = os.path.join(args.out_dir, f"source_synthetic_{tag}.db")
    gt_path = os.path.join(args.out_dir, f"ground_truth_{tag}.json")
    gt = None
    log = None

    if "emr" in stages:
        print(f"\nSampling {args.patients:,} patients...")
        pop = sample_population(cat, args.patients, end, rng_pop)
        death_year = simulate_mortality(cat, pop, start.year, end.year, rng_pop)
        cause_key = assign_causes_of_death(cat, pop, death_year, rng_pop)
        print(f"  {int((death_year >= 0).sum()):,} deaths over {end.year - start.year + 1} years")

        print("Generating clinical events...")
        with tqdm(total=args.patients, desc="  patients", unit="pt", unit_scale=True) as pbar:
            log = generate_events(
                cat,
                pop,
                death_year,
                start,
                end,
                rng_events,
                progress=pbar.update,
                cause_key=cause_key,
            )

        print(
            f"  {len(log.visits):,} visits, {len(log.conditions):,} diagnoses, "
            f"{len(log.drugs):,} exposures, {len(log.measurements):,} measurements, "
            f"{len(log.observations):,} observations"
        )

        if not args.no_sqlite:
            print(f"\nWriting OMOP CDM {CDM_VERSION} -> {os.path.basename(omop_db)}")
            counts = write_omop_sqlite(cat, vocab, log, omop_db, args.seed)
            for table, n in counts.items():
                print(f"  {table:22s} {n:>10,}")

            print(f"\nWriting source EMR ({args.messiness}) -> {os.path.basename(source_db)}")
            src_counts = write_source_sqlite(cat, log, source_db, args.messiness, rng_source)
            for table, n in src_counts.items():
                print(f"  {table:22s} {n:>10,}")
            # Transposed dates are injected while rendering the extract, but the
            # manifest documents one error budget, so the realised count joins
            # the defects planted upstream in the event log.
            log.injected_errors["transposed_date_digits"] = src_counts.get("transposed_date_digits", 0)

        if log.injected_errors:
            planted = ", ".join(f"{k} {v:,}" for k, v in sorted(log.injected_errors.items()))
            print(f"\n  injected error budget: {planted}")

        if args.postgres:
            print("\nLoading into PostgreSQL...")
            _load_postgres(omop_db, source_db)

    if "ground-truth" in stages and os.path.exists(omop_db):
        print("\nComputing ground truth from the finished database...")
        gt = build_ground_truth(
            cat,
            omop_db,
            source_db,
            {
                "seed": args.seed,
                "n_patients": args.patients,
                "cdm_version": CDM_VERSION,
                "observation_window": [args.start_date, args.end_date],
                "messiness": args.messiness,
            },
            injected_errors=(log.injected_errors if log else _previous_injected(gt_path)),
        )
        path = write_ground_truth(gt, gt_path)
        print(f"  wrote {os.path.basename(path)} with {len(gt['cohorts'])} demo cohorts")

        # The quirks manifest ships with the data. A deliberate defect nobody
        # documented is indistinguishable from a bug, and the first reviewer to
        # find one stops trusting everything else in the file.
        quirks_path = write_known_quirks(cat, gt, os.path.join(args.out_dir, f"KNOWN_QUIRKS_{tag}.md"))
        print(f"  wrote {os.path.basename(quirks_path)}")

    if "validate" in stages and gt and os.path.exists(omop_db):
        print("\nValidating...")
        failures, warnings = validate_databases(cat, omop_db, source_db, gt)
        # Warnings are distributional: a small cohort can miss a target shape by
        # chance without anything being broken, so they print and do not fail.
        for w in warnings:
            print(f"    ! {w}")
        if failures:
            print(f"  FAILED with {len(failures)} broken invariants:")
            for f in failures:
                print(f"    - {f}")
            return 1
        print(f"  all invariants hold ({len(warnings)} distributional notes)")
        print_report(cat, gt, omop_db)

    print("\nDone.")
    return 0


def _previous_injected(gt_path: str) -> Optional[dict]:
    """Recover the planted-defect counts when the EMR stage did not run.

    The budget is a property of the databases on disk, not of this process.
    Rebuilding the ground truth alone used to forget it, which made the
    validation suite demand a defect-free file and fail against data that was
    exactly as documented.
    """
    if not os.path.exists(gt_path):
        return None
    try:
        with open(gt_path) as fh:
            return json.load(fh).get("injected_errors") or None
    except (OSError, ValueError):
        return None


def _scale_tag(n_patients: int) -> str:
    """Compact, sortable size label: 1000 -> 1k, 10000 -> 10k, 1500000 -> 1.5m."""
    for limit, suffix in ((1_000_000, "m"), (1_000, "k")):
        if n_patients >= limit:
            value = n_patients / limit
            text = f"{value:.1f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    return f"{n_patients}pt"


def _load_postgres(omop_db: str, source_db: str) -> None:
    """Copy the SQLite databases into PostgreSQL schemas.

    The schema is created here rather than in sql/ because docker-entrypoint-initdb.d
    only runs against an empty volume, so an existing database would never pick it up.
    """
    import sqlite3

    import psycopg2
    from psycopg2.extras import execute_values

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            for schema, path in [("omop", omop_db), ("source_emr", source_db)]:
                if not os.path.exists(path):
                    continue
                cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
                cur.execute(f"CREATE SCHEMA {schema}")

                lite = sqlite3.connect(path)
                try:
                    tables = [r[0] for r in lite.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
                    for table in tables:
                        cols = [r[1] for r in lite.execute(f"PRAGMA table_info({table})")]
                        types = [r[2] or "TEXT" for r in lite.execute(f"PRAGMA table_info({table})")]
                        ddl_cols = ", ".join(f'"{c}" {_pg_type(t)}' for c, t in zip(cols, types))
                        cur.execute(f'CREATE TABLE {schema}."{table}" ({ddl_cols})')

                        rows = lite.execute(f"SELECT * FROM {table}").fetchall()
                        if rows:
                            collist = ", ".join(f'"{c}"' for c in cols)
                            execute_values(
                                cur,
                                f'INSERT INTO {schema}."{table}" ({collist}) VALUES %s',
                                rows,
                                page_size=1000,
                            )
                        print(f"  {schema}.{table:<24s} {len(rows):>10,}")
                finally:
                    lite.close()
        conn.commit()
    finally:
        conn.close()


def _pg_type(sqlite_type: str) -> str:
    t = sqlite_type.upper()
    if "INT" in t:
        return "BIGINT"
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return "DOUBLE PRECISION"
    return "TEXT"


if __name__ == "__main__":
    exit(main())
