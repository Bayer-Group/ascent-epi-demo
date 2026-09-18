from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from ascent_medical_coder.services.pipeline import filtering
from ascent_medical_coder.services.pipeline.filtering import (
    LLMFilterError,
    _aggregate_batch_results,
    _apply_index_filter,
    _call_llm_filter_with_retry,
    _safe_parse_json,
    apply_llm_filter,
)


def test_safe_parse_json_rejects_invalid_and_non_object_payloads():
    assert _safe_parse_json("not json")["error"]
    assert _safe_parse_json("[]")["error"] == "Top level JSON is not an object"
    assert _safe_parse_json('{"include": [1], "exclude": [2]}') == {"include": [1], "exclude": [2]}


class _Connector:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    async def complete_json(self, prompt):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def frame():
    return pd.DataFrame({"CONCEPT_ID": [1, 2], "CONCEPT_NAME": ["One", "Two"]}, index=[10, 20])


@pytest.mark.parametrize(
    "invalid",
    [
        {"include": [10], "exclude": []},  # incomplete
        {"include": [10, 20], "exclude": [20]},  # contradictory
        {"include": [10, 10], "exclude": [20]},  # duplicate
        {"include": [999], "exclude": [10, 20]},  # unknown label
        {"include": [True, 10], "exclude": [20]},  # booleans are not indices
        {"include": [10.0], "exclude": [20]},
        {"include": ["10"], "exclude": [20]},
        {"include": None, "exclude": [10, 20]},
        {"include": [], "exclude": [10, 20], "error": "upstream failure"},
    ],
)
async def test_invalid_decisions_retry_then_accept_valid_partition(frame, invalid):
    connector = _Connector([json.dumps(invalid), '{"include": [10], "exclude": [20]}'])
    result = await _call_llm_filter_with_retry(connector, "query", frame, max_retries=1)
    assert connector.calls == 2
    assert result["include"] == [10]
    assert result["exclude"] == [20]


async def test_index_filter_validates_partition_and_preserves_label_order():
    progress = AsyncMock()
    frame = pd.DataFrame({"CONCEPT_ID": [1, 2, 3]}, index=[10, 20, 30])
    included = await _apply_index_filter(frame, [30, 10], [20], progress)
    empty = await _apply_index_filter(frame, [], [10, 20, 30], progress)
    assert included.index.tolist() == [30, 10]
    assert empty.empty
    with pytest.raises(LLMFilterError):
        await _apply_index_filter(frame, [999], [], progress)


def test_batch_results_are_aggregated_with_optional_reasons():
    results = [
        {"include": [1], "exclude": [2], "reasons_include": ["yes"]},
        {"include": [3], "exclude": [4], "reasons_exclude": ["no"]},
    ]
    assert _aggregate_batch_results(results, True) == ([1, 3], [2, 4], ["yes"], ["no"])
    assert _aggregate_batch_results(results, False) == ([1, 3], [2, 4], [], [])


@pytest.mark.parametrize("failure", [ConnectionError("down"), '{"include": [10,20], "exclude": [20]}'])
async def test_exhausted_retries_are_an_error_not_an_empty_success(frame, failure):
    connector = _Connector([failure, failure])
    with pytest.raises(LLMFilterError, match="after retries"):
        await _call_llm_filter_with_retry(connector, "query", frame, max_retries=1)
    assert connector.calls == 2


@pytest.mark.parametrize("size", [2, 61])
async def test_failed_single_or_multi_batch_never_updates_lts(monkeypatch, size):
    frame = pd.DataFrame({"CONCEPT_ID": range(size), "CONCEPT_NAME": ["concept"] * size})
    connector = _Connector(['{"include": [true], "exclude": [0]}'] * 6)
    monkeypatch.setattr(filtering, "get_llm_connector", lambda _: connector)
    update = AsyncMock()
    monkeypatch.setattr(filtering.lts, "update_lts", update)
    with pytest.raises(LLMFilterError):
        await apply_llm_filter("query", frame, "haiku", allow_lts=True)
    update.assert_not_awaited()


async def test_valid_multi_batch_filter_retains_only_selected_concepts(monkeypatch):
    frame = pd.DataFrame({"CONCEPT_ID": range(61), "CONCEPT_NAME": ["concept"] * 61})
    connector = _Connector(
        [json.dumps({"include": [0], "exclude": list(range(1, 60))}), '{"include": [], "exclude": [60]}']
    )
    monkeypatch.setattr(filtering, "get_llm_connector", lambda _: connector)
    result = await apply_llm_filter("query", frame, "haiku", allow_lts=False)
    assert result.filtered_df["CONCEPT_ID"].tolist() == [0]


async def test_prompt_failure_is_explicit_and_does_not_call_connector(monkeypatch, frame):
    monkeypatch.setattr(filtering, "get_prompt", AsyncMock(side_effect=ValueError("bad prompt")))
    connector = _Connector([])
    with pytest.raises(LLMFilterError, match="prompt"):
        await _call_llm_filter_with_retry(connector, "query", frame)
    assert connector.calls == 0


async def test_duplicate_input_labels_are_rejected(frame):
    frame.index = [10, 10]
    with pytest.raises(LLMFilterError, match="unique integer"):
        await apply_llm_filter("query", frame, "haiku")
