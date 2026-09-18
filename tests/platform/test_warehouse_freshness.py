"""A built warehouse must match the archive it was built from.

Size and schema cannot tell a current build from one made against different
data, so a changed archive has to force a rebuild. The failure it prevents is
silent: every query answers, against the wrong dataset.
"""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path

import duckdb
import pytest

from ascent_platform.warehouse import bootstrap as b


@pytest.fixture
def warehouse(tmp_path):
    data, work = tmp_path / "data", tmp_path / "work"
    data.mkdir()
    work.mkdir()

    def write_archive(rows: int, db=None) -> None:
        for logical in [db] if db else b.DATABASES:
            staging = work / "mk"
            staging.mkdir(exist_ok=True)
            sqlite_file = staging / f"{logical.name}.db"
            sqlite_file.unlink(missing_ok=True)
            con = sqlite3.connect(sqlite_file)
            con.execute("CREATE TABLE person (person_id INTEGER)")
            con.executemany("INSERT INTO person VALUES (?)", [(i,) for i in range(rows)])
            con.commit()
            con.close()
            with zipfile.ZipFile(data / logical.archive.format(scale="1k"), "w") as zf:
                zf.write(sqlite_file, f"{logical.name}.db")

    return data, work, write_archive


def _rows(target: Path, schema: str) -> int:
    con = duckdb.connect(str(target), read_only=True)
    try:
        return con.execute(f'SELECT COUNT(*) FROM "{schema}".person').fetchone()[0]
    finally:
        con.close()


def test_a_changed_archive_rebuilds(warehouse):
    data, work, write_archive = warehouse
    db = b.DATABASES[0]
    target = b.warehouse_path(work, db, "1k")

    write_archive(10)
    b.bootstrap(data, work, "1k")
    assert _rows(target, db.schema) == 10

    write_archive(25, db)
    b.bootstrap(data, work, "1k")
    assert _rows(target, db.schema) == 25


def test_an_unchanged_archive_is_not_rebuilt(warehouse):
    data, work, write_archive = warehouse
    target = b.warehouse_path(work, b.DATABASES[0], "1k")

    write_archive(10)
    b.bootstrap(data, work, "1k")
    stamp = target.stat().st_mtime_ns

    b.bootstrap(data, work, "1k")
    assert target.stat().st_mtime_ns == stamp


def test_a_missing_marker_rebuilds(warehouse):
    data, work, write_archive = warehouse
    target = b.warehouse_path(work, b.DATABASES[0], "1k")

    write_archive(10)
    b.bootstrap(data, work, "1k")
    b._source_marker(target).unlink()

    b.bootstrap(data, work, "1k")
    assert b._source_marker(target).exists()


def test_the_extracted_sqlite_is_not_reused_across_archives(warehouse):
    """The unpacked copy is cached too, and caches the same staleness."""
    data, work, write_archive = warehouse
    db = b.DATABASES[0]

    write_archive(10)
    b.bootstrap(data, work, "1k")

    write_archive(31, db)
    digest = b._digest(data / db.archive.format(scale="1k"))
    unpacked = b._unpack(data / db.archive.format(scale="1k"), work / "sqlite", digest)

    con = sqlite3.connect(unpacked)
    try:
        assert con.execute("SELECT COUNT(*) FROM person").fetchone()[0] == 31
    finally:
        con.close()
