"""Population metadata must degrade to a note, not a TypeError.

The shipped datasets are generated rather than sampled, so they cover no
country and no real population. Three prompt-building sites formatted
``estimated_coverage`` and ``population`` straight into an f-string, which
raised "unsupported format string passed to NoneType.__format__" and took
compare_rwd_with_literature down *after* it had completed its literature
search -- 51 seconds of work discarded at the last step.
"""

import json

import pytest

from ascent_domain.omop.models.agents.langgraph.scaling_tools import (
    database_metadata_dict,
    get_database_metadata,
    population_context_note,
)


@pytest.fixture
def real_database():
    """A database that does sample a real population."""
    database_metadata_dict["TEST_REAL_DB"] = {
        "country": "United States",
        "total_patients": 92_000_000,
        "population": 331_000_000,
        "estimated_coverage": 0.2779,
    }
    yield "TEST_REAL_DB"
    del database_metadata_dict["TEST_REAL_DB"]


@pytest.mark.parametrize("database", ["SYNTHETIC_EHR_OMOP", "SYNTHETIC_CLAIMS"])
def test_synthetic_metadata_does_not_raise(database):
    meta = get_database_metadata(database)

    assert meta["coverage_description"]
    assert "synthetic" in meta["coverage_description"].lower()


@pytest.mark.parametrize("database", ["SYNTHETIC_EHR_OMOP", "SYNTHETIC_CLAIMS"])
def test_synthetic_context_note_warns_against_extrapolation(database):
    note = population_context_note(get_database_metadata(database))

    assert "must NOT be extrapolated" in note
    # No stray "None" rendered into a prompt the model will read.
    assert "None" not in note


def test_real_database_still_reports_coverage(real_database):
    note = population_context_note(get_database_metadata(real_database))

    assert "27.79% of United States population" in note
    assert "331,000,000" in note


def test_missing_database_returns_none():
    assert get_database_metadata("NO_SUCH_DATABASE") is None


def test_context_note_for_missing_metadata_is_empty():
    assert population_context_note(None) == ""


async def test_scaling_declines_to_extrapolate_synthetic_counts(monkeypatch):
    """The scaling tool must say why, not divide by None."""
    from ascent_domain.omop.models.agents.langgraph import scaling_tools

    # The count is extracted by an LLM. Stub it so this stays a unit test:
    # without the stub it reaches a live provider and needs an API key.
    monkeypatch.setattr(scaling_tools, "get_llm_response_sync", lambda prompt, temp: "368")

    result = await scaling_tools.get_patient_count_scaling.ainvoke({
        "condition_name": "essential hypertension",
        "previous_output": "There are 368 patients with essential hypertension.",
        "database": "SYNTHETIC_EHR_OMOP",
    })

    payload = json.loads(result)
    assert "error" not in payload
    assert "cannot be extrapolated" in payload["summary"]
