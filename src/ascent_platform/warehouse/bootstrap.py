"""Build the local warehouse from the shipped synthetic SQLite files.

Runs once, before the app serves anything, and produces one DuckDB file per
logical database so that three-part names resolve the way they did on
Snowflake::

    SYNTHETIC_EHR_OMOP.CDM.person
    SYNTHETIC_CLAIMS.DATA_202601.DX

That matters more than it looks: every caller threads ``database`` and
``schema`` separately, so preserving the shape means the swap lands at one
seam instead of rippling through 56 call sites.

The build is also where the source layer's dates get fixed. ``DX_DT`` and
friends arrive as text in four different formats -- ``YYYY-MM-DD``,
``MM/DD/YYYY``, ``DD-Mon-YYYY`` and unix epoch seconds -- and a plain
``CAST(... AS DATE)`` nulls 61.6% of them without raising anything. Parsing
once here, into a real DATE column, is the only place that fix stays fixed;
doing it per-query would mean trusting every generated statement to remember.

Row counts are asserted across the copy. A normalisation that silently drops
rows is exactly the failure this data is built to provoke, so it fails the
build rather than the analysis.
"""

from __future__ import annotations

import hashlib
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LogicalDatabase:
    """One warehouse database, as the rest of the platform names it."""

    name: str
    schema: str
    archive: str
    # Whether diagnosis/procedure codes are stored with their decimal point.
    # The coder strips or keeps dots based on this, and getting it wrong makes
    # every code lookup match nothing -- silently, since an empty match set is
    # indistinguishable from a genuinely absent condition.
    codes_with_dots: bool = True

    @property
    def sqlite_name(self) -> str:
        return self.archive.removesuffix(".zip")


# The ``_OMOP`` suffix is load-bearing: ascent_platform.warehouse.db_type
# routes on it, so a rename here silently sends OMOP questions down the
# non-OMOP pipeline.
DATABASES: tuple[LogicalDatabase, ...] = (
    LogicalDatabase("SYNTHETIC_EHR_OMOP", "CDM", "omop_synthetic_{scale}.db.zip", codes_with_dots=True),
    LogicalDatabase("SYNTHETIC_CLAIMS", "DATA_202601", "source_synthetic_{scale}.db.zip", codes_with_dots=True),
)

# Formats seen across both layers. The source layer mixes the first three in a
# single column; the OMOP vocabulary stores YYYYMMDD as an integer.
_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%Y%m%d")


def _normalised_date(column: str) -> str:
    """SQL that turns a date-ish column into a real DATE, whatever its format.

    Cast to VARCHAR first: some of these arrive as BIGINT (``20000101``), and
    TRY_STRPTIME only accepts text.
    """
    text = f"CAST({column} AS VARCHAR)"
    attempts = ", ".join(f"TRY_STRPTIME({text}, '{fmt}')" for fmt in _DATE_FORMATS)
    # 9-10 digits is epoch seconds; 8 digits is YYYYMMDD and handled above.
    epoch = f"CASE WHEN regexp_full_match({text}, '[0-9]{{9,10}}') THEN to_timestamp(CAST({text} AS BIGINT)) END"
    return f"CAST(COALESCE({attempts}, {epoch}) AS DATE)"


def _looks_like_a_date(column: str, sql_type: str) -> bool:
    return sql_type.upper() in {"VARCHAR", "TEXT", "BIGINT", "INTEGER"} and (column.upper().endswith(("_DT", "_DATE")))


def _unpack(archive: Path, into: Path, source_digest: str | None = None) -> Path:
    """Extract *archive*, reusing an earlier extraction of the same archive."""
    target = into / archive.name.removesuffix(".zip")
    marker = _source_marker(target)
    if target.exists() and (source_digest is None or (marker.exists() and marker.read_text().strip() == source_digest)):
        logger.debug("reusing %s", target.name)
        return target
    into.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        member = next(n for n in zf.namelist() if n.endswith(".db"))
        target.write_bytes(zf.read(member))
    if source_digest is not None:
        marker.write_text(source_digest)
    logger.info("unpacked %s", target.name)
    return target


def warehouse_path(work_dir: Path, db: LogicalDatabase, scale: str) -> Path:
    """Where the built DuckDB file for *db* at *scale* lives.

    The scale is in the filename, and that is load-bearing. It used to be
    ``{db.name}.duckdb``: switching DATA_SCALE from 1k to 10k found the
    existing file, reported "already built", and served the 1,000-patient
    warehouse while every setting and log line said 10k. Nothing errored --
    the answers were simply computed against the wrong dataset.

    Keying on the scale also means switching back does not rebuild: both
    warehouses can sit in the work directory at once.
    """
    return work_dir / f"{db.name}.{scale}.duckdb"


def build_database(db: LogicalDatabase, sqlite_path: Path, out_dir: Path, scale: str = "1k"):
    """Project one SQLite file into a DuckDB database, with typed dates.

    Returns a ``{table: row_count}`` mapping so the caller can report what was
    built. Raises if any table loses rows in the copy.
    """
    import duckdb

    out_dir.mkdir(parents=True, exist_ok=True)
    target = warehouse_path(out_dir, db, scale)
    if target.exists():
        target.unlink()

    con = duckdb.connect(str(target))
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{sqlite_path}' AS src (TYPE sqlite, READ_ONLY);")
    con.execute(f'CREATE SCHEMA IF NOT EXISTS "{db.schema}";')

    tables = [r[0] for r in con.execute("SELECT table_name FROM duckdb_tables() WHERE database_name = 'src'").fetchall()]

    built: dict[str, int] = {}
    for table in tables:
        columns = con.execute("SELECT column_name, data_type FROM duckdb_columns() WHERE database_name='src' AND table_name=?", [table]).fetchall()

        projection = ", ".join(
            f'{_normalised_date(chr(34) + name + chr(34))} AS "{name}"' if _looks_like_a_date(name, dtype) else f'"{name}"' for name, dtype in columns
        )
        con.execute(f'CREATE TABLE "{db.schema}"."{table}" AS SELECT {projection} FROM src."{table}";')

        before = con.execute(f'SELECT COUNT(*) FROM src."{table}"').fetchone()[0]
        after = con.execute(f'SELECT COUNT(*) FROM "{db.schema}"."{table}"').fetchone()[0]
        if before != after:
            raise RuntimeError(f"{db.name}.{db.schema}.{table}: copy lost rows ({before} -> {after})")
        built[table] = after

    con.execute("DETACH src;")
    con.close()
    return target, built


def _source_marker(target: Path) -> Path:
    return target.with_name(target.name + ".source")


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_complete(target: Path, db: LogicalDatabase, source_digest: str | None) -> bool:
    """Whether *target* is a finished build of the current source archive.

    A container killed mid-build leaves a valid but empty DuckDB file, and a
    file built from a different archive is indistinguishable from a current one
    by size or schema alone. Both are rebuilt rather than served.

    ``source_digest`` of None means the archive is unavailable to compare
    against, so only the structural checks apply.
    """
    if not target.exists() or target.stat().st_size < 100_000:
        return False
    if source_digest is not None:
        marker = _source_marker(target)
        if not marker.exists() or marker.read_text().strip() != source_digest:
            return False
    try:
        import duckdb

        con = duckdb.connect(str(target), read_only=True)
        try:
            tables = con.execute("SELECT COUNT(*) FROM duckdb_tables() WHERE schema_name = ?", [db.schema]).fetchone()[0]
        finally:
            con.close()
        return bool(tables)
    except Exception:  # noqa: BLE001 - unreadable means unusable means rebuild
        return False


# Where persisted cohorts live. cohort_persist writes the patient list to a
# real table and keeps only metadata in Postgres, so a cohort can be joined
# against later. The shipped data is attached READ_ONLY, so cohorts get their
# own writable database.
COHORT_DATABASE = "ASCENT"
COHORT_SCHEMA = "ASCENT_COHORTS"


def cohort_attach_sql() -> str:
    """ATTACH statement for the cohort store, which lives in Postgres.

    Cohorts were briefly a writable DuckDB file, and that cannot work: DuckDB
    allows a single read-write connection to a file across processes, so the
    second worker to persist a cohort died on "Conflicting lock is held".

    Postgres handles concurrent writers, is already in the stack for cohort
    metadata, and -- through DuckDB's postgres extension -- can still be
    joined against the DuckDB patient tables in one query. That join is the
    requirement that ruled out simply writing cohorts through SQLAlchemy:
    table_one and the cohort tools select FROM the cohort table alongside
    OMOP tables.
    """
    from ascent_platform.config.runtime import get_runtime_settings

    s = get_runtime_settings()
    dsn = f"host={s.DB_HOST} port={s.DB_PORT} dbname={s.DB_NAME} user={s.DB_APP_USR} password={s.DB_APP_PWD}"
    return f"ATTACH '{dsn}' AS \"{COHORT_DATABASE}\" (TYPE postgres);"


def bootstrap(data_dir: Path, work_dir: Path, scale: str = "1k") -> dict[str, Path]:
    """Build every logical database. Idempotent -- skips completed builds.

    Returns ``{database_name: duckdb_path}`` for the session layer to ATTACH.
    """
    built: dict[str, Path] = {}
    for db in DATABASES:
        target = warehouse_path(work_dir, db, scale)
        archive = data_dir / db.archive.format(scale=scale)
        source_digest = _digest(archive) if archive.exists() else None

        if _is_complete(target, db, source_digest):
            logger.info("%s already built", db.name)
            built[db.name] = target
            continue
        if target.exists():
            logger.warning("%s is stale or incomplete; rebuilding", target.name)
            target.unlink()
            _source_marker(target).unlink(missing_ok=True)

        if not archive.exists():
            raise FileNotFoundError(f"{archive} is missing. The {scale} dataset ships in data/synthetic/; larger scales are published separately.")
        sqlite_path = _unpack(archive, work_dir / "sqlite", source_digest)
        target, tables = build_database(db, sqlite_path, work_dir, scale)
        if source_digest is not None:
            _source_marker(target).write_text(source_digest)
        logger.info(
            "built %s.%s: %d tables, %d rows",
            db.name,
            db.schema,
            len(tables),
            sum(tables.values()),
        )
        built[db.name] = target
    return built


def main() -> int:
    """Build the warehouse as a one-shot step, before the server forks.

    The app runs several worker processes and each one opens the warehouse. If
    they also *build* it they race: one unlinks a stub and starts writing while
    another attaches the half-written file and caches the connection for the
    process lifetime. The symptom is not an error -- it is a worker that
    answers every query with zero rows, so the service looks healthy and the
    data looks empty.

    Running this before the workers start removes the race rather than
    coordinating it.
    """
    import logging

    from ascent_platform.config.runtime import get_runtime_settings

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_runtime_settings()
    built = bootstrap(
        Path(settings.WAREHOUSE_DATA_DIR),
        Path(settings.WAREHOUSE_WORK_DIR),
        settings.WAREHOUSE_SCALE,
    )
    if not built:
        return 1

    # The catalog describes the warehouse, so it is built here rather than in
    # coder-init (which runs before the warehouse exists) and here rather than
    # per-worker (which would race for the same reason the warehouse build
    # does). The 1k catalog ships in the repository and this is a no-op; a
    # larger scale has none and generates one now, once.
    from ascent_platform.warehouse import catalog_build

    try:
        catalog_build.ensure(settings.WAREHOUSE_SCALE)
    except Exception:
        # A missing catalog degrades to request-time profiling, which still
        # answers -- so this must not stop the server from starting.
        logger.exception("catalog generation failed; falling back to profiling")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
