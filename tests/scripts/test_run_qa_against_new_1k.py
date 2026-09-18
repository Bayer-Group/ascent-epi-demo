from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_qa_against_new_1k.py"
SPEC = importlib.util.spec_from_file_location("run_qa_against_new_1k", SCRIPT)
assert SPEC and SPEC.loader
qa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qa)


def test_missing_input_is_reported_without_creating_a_database(tmp_path):
    missing = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError, match="Query library"):
        qa._require_input(missing, "Query library")
    assert not missing.exists()


def test_query_library_is_loaded_read_only(tmp_path):
    library = tmp_path / "querylib.db"
    with sqlite3.connect(library) as connection:
        connection.execute("CREATE TABLE queries (ID INTEGER, QUESTION TEXT, QUERY_SNOWFLAKE_WITH_PLACEHOLDERS TEXT, QUESTION_TYPE TEXT)")
        connection.execute("INSERT INTO queries VALUES (1, 'How many?', 'SELECT 1', 'QA')")

    assert qa.load_qa_rows(library) == [
        {
            "ID": 1,
            "QUESTION": "How many?",
            "QUERY_SNOWFLAKE_WITH_PLACEHOLDERS": "SELECT 1",
        }
    ]


class _Cursor:
    description = [("value",)]

    def __init__(self):
        self._rows = [(number,) for number in range(2_500)]

    def fetchmany(self, size):
        batch, self._rows = self._rows[:size], self._rows[size:]
        return batch


class _Connection:
    def execute(self, sql):
        assert sql
        return _Cursor()


def test_query_results_are_counted_in_batches_but_only_twenty_are_retained():
    result = qa.process_row(
        _Connection(),
        {
            "ID": 1,
            "QUESTION": "Values",
            "QUERY_SNOWFLAKE_WITH_PLACEHOLDERS": "SELECT 1 AS value",
        },
    )

    assert result["status"] == "ok"
    assert result["row_count"] == 2_500
    assert len(result["rows"]) == 20
