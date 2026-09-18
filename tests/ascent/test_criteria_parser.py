from __future__ import annotations

import pytest

from ascent_domain.omop.models.inference.criteria_to_counts import (
    CriteriaParser,
    CriteriaParsingError,
)


def test_extracts_direct_criteria():
    criteria = {"include": ["adult", "hypertension"], "exclude": ["pregnancy"]}
    assert CriteriaParser.extract_criteria(criteria) == ["adult", "hypertension", "pregnancy"]


def test_extracts_nested_masked_criteria():
    criteria = {
        "masked_text": {
            "include": ["condition_1"],
            "exclude": ["condition_2"],
        }
    }
    assert CriteriaParser.extract_criteria(criteria) == ["condition_1", "condition_2"]


def test_parses_json_embedded_in_model_output():
    output = 'prefix {"include": ["adult"], "exclude": []} suffix'
    assert CriteriaParser.extract_criteria(output) == ["adult"]


def test_parses_safe_python_literal_fallback():
    output = "{'include': ['adult'], 'exclude': []}"
    assert CriteriaParser.extract_criteria(output) == ["adult"]


@pytest.mark.parametrize(
    "value,match",
    [
        ("not criteria", "No dictionary-like structure"),
        ("{not valid}", "invalid dictionary"),
        ({"include": [], "exclude": "none"}, "list-valued"),
        ({"masked_text": []}, "masked_text"),
        (None, "dictionary"),
    ],
)
def test_invalid_criteria_raise_a_specific_error(value, match):
    with pytest.raises(CriteriaParsingError, match=match):
        CriteriaParser.extract_criteria(value)
