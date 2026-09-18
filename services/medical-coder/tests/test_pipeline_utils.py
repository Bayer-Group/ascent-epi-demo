from __future__ import annotations

import pandas as pd
import pytest
from fastapi import HTTPException

from ascent_medical_coder.services.pipeline.utils import (
    _convert_concepts_to_models,
    _dataframe_to_concepts,
    parse_db_name_and_schema,
)


def test_parse_database_and_schema_variants():
    assert parse_db_name_and_schema("database", None) == ("database", None)
    assert parse_db_name_and_schema("database.schema", None) == ("database", "schema")
    assert parse_db_name_and_schema("database", "schema") == ("database", "schema")


@pytest.mark.parametrize("database,schema", [("", None), ("a.b", "c"), ("a.b.c", None)])
def test_parse_database_and_schema_rejects_ambiguous_values(database, schema):
    with pytest.raises(HTTPException):
        parse_db_name_and_schema(database, schema)


def test_dataframe_conversion_skips_invalid_rows_and_preserves_validity():
    frame = pd.DataFrame(
        [
            {
                "CONCEPT_ID": "42",
                "CONCEPT_NAME": "Hypertension",
                "CONCEPT_CODE": "HYP",
                "DOMAIN_ID": "Condition",
                "INVALID_REASON": None,
                "score": "0.9",
            },
            {"CONCEPT_ID": "not-an-id", "CONCEPT_NAME": "Invalid"},
        ]
    )

    converted = _dataframe_to_concepts(frame, "hypertension")
    assert len(converted["hypertension"]) == 1
    assert converted["hypertension"][0]["CONCEPT_DATA"]["CONCEPT_ID"] == 42
    assert converted["hypertension"][0]["CONCEPT_DATA"]["IS_VALID"] is True
    assert converted["hypertension"][0]["SCORE"] == 0.9


def test_model_conversion_skips_invalid_concepts():
    concepts = {
        "query": [
            {
                "CONCEPT_DATA": {
                    "CONCEPT_ID": 42,
                    "CONCEPT_NAME": "Hypertension",
                    "DOMAIN_ID": None,
                },
                "SCORE": 0.9,
            },
            {"CONCEPT_DATA": {"CONCEPT_ID": "bad"}, "SCORE": 0},
        ]
    }

    converted = _convert_concepts_to_models(concepts)
    assert len(converted["query"]) == 1
    assert converted["query"][0].CONCEPT_DATA.DOMAIN_ID == ""
