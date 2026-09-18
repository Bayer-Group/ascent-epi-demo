"""The warehouse connection must not be able to reach the filesystem.

``execute_sql`` forwards caller SQL to the shared DuckDB connection essentially
unchanged -- transpile() rewrites dialect, it does not restrict anything. DuckDB
leaves ``enable_external_access`` ON by default, and with it on, plain SQL reads
and writes files:

    SELECT content FROM read_text('.env')     -- returns the LLM key
    COPY (SELECT '...') TO '/app/src/main.py' -- overwrites source

That made an MCP tool into arbitrary host file read/write, reachable by anyone
who could call the server -- which, with no API_AUTH_TOKEN set, is the default.

The fix is one ``SET`` in ``_connect``, and its *position* is the whole point:
ATTACH and INSTALL both need external access, so the lockdown has to come after
them. A future edit that moves it earlier breaks the warehouse; one that drops
it reopens the hole silently, because nothing else observes the difference.
These tests are what notices.
"""

from __future__ import annotations

import duckdb
import pytest

from ascent_platform.warehouse import session


@pytest.fixture
def warehouse(tmp_path, monkeypatch):
    """A real ``_connect`` run against a throwaway one-table warehouse.

    Only ``bootstrap`` is faked, so the attach loop, the extension load and the
    lockdown all execute as they do in production. The cohort store is forced to
    fail: a test must not depend on a Postgres being up, and the except branch
    is the path a developer without one actually takes.
    """
    db_path = tmp_path / "patients.db"
    seed = duckdb.connect(str(db_path))
    seed.execute("CREATE TABLE person AS SELECT 1 AS person_id")
    seed.close()

    secret = tmp_path / "secret.env"
    secret.write_text("GOOGLE_API_KEY=super-secret-value\n")

    monkeypatch.setattr(session, "bootstrap", lambda *_a, **_k: {"OMOP": str(db_path)})
    monkeypatch.setattr(session, "cohort_attach_sql", lambda: "ATTACH 'no-postgres-here' AS X (TYPE postgres);")

    con = session._connect(tmp_path, tmp_path, "1k")
    try:
        yield con, secret
    finally:
        con.close()


def test_attached_data_is_still_queryable(warehouse):
    """The lockdown must not cost us the thing the connection exists for."""
    con, _ = warehouse
    assert con.execute('SELECT person_id FROM "OMOP".person').fetchone() == (1,)


def test_scratch_tables_still_work(warehouse):
    """Cohort building writes temp tables; those are in-memory, not external."""
    con, _ = warehouse
    con.execute("CREATE TEMP TABLE cohort AS SELECT 1 AS person_id")
    assert con.execute("SELECT count(*) FROM cohort").fetchone() == (1,)


def test_external_access_is_off(warehouse):
    con, _ = warehouse
    assert con.execute("SELECT current_setting('enable_external_access')").fetchone()[0] is False


def test_reading_a_local_file_is_refused(warehouse):
    """The headline exploit: the LLM key out of .env through a SELECT."""
    con, secret = warehouse
    with pytest.raises(duckdb.Error):
        con.execute(f"SELECT content FROM read_text('{secret}')").fetchall()


def test_reading_a_local_file_by_another_reader_is_refused(warehouse):
    """read_text is not the only reader; the setting has to cover the family."""
    con, secret = warehouse
    with pytest.raises(duckdb.Error):
        con.execute(f"SELECT * FROM read_csv_auto('{secret}')").fetchall()


def test_writing_a_local_file_is_refused(warehouse, tmp_path):
    con, _ = warehouse
    target = tmp_path / "written.csv"
    with pytest.raises(duckdb.Error):
        con.execute(f"COPY (SELECT 1 AS x) TO '{target}';")
    assert not target.exists()


def test_installing_an_extension_is_refused(warehouse):
    """httpfs would hand back the network egress the setting just removed."""
    con, _ = warehouse
    with pytest.raises(duckdb.Error):
        con.execute("INSTALL httpfs;")


def test_the_lockdown_cannot_be_undone_by_a_later_query(warehouse):
    """Without this, the guard is one caller-supplied SET away from nothing."""
    con, secret = warehouse
    with pytest.raises(duckdb.Error):
        con.execute("SET enable_external_access=true;")
    with pytest.raises(duckdb.Error):
        con.execute(f"SELECT content FROM read_text('{secret}')").fetchall()


def test_duckdb_still_defaults_to_unrestricted():
    """A negative control: a check that always passes is worth nothing.

    If a future DuckDB ships external access off by default, this fails and the
    comments above should be reworded -- rather than everyone assuming the
    lockdown is load-bearing when the engine already handles it.
    """
    plain = duckdb.connect(":memory:")
    try:
        assert plain.execute("SELECT current_setting('enable_external_access')").fetchone()[0] is True
    finally:
        plain.close()
