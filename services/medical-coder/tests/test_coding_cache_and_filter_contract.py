from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest
from fastapi import HTTPException

from ascent_medical_coder.schemas.coding import MedicalCodingRequest
from ascent_medical_coder.services import coding
from ascent_medical_coder.services.pipeline import filtering


@pytest.fixture
def cache_backend(monkeypatch):
    stored = {}

    async def get(key):
        return stored.get(key)

    async def set_value(key, value, ttl):
        stored[key] = value

    monkeypatch.setattr(coding, "get_cached", AsyncMock(side_effect=get))
    monkeypatch.setattr(coding, "set_cached", AsyncMock(side_effect=set_value))
    monkeypatch.setattr(coding, "log_request", AsyncMock())
    return stored


async def _run(payload, *, include_reasoning=False):
    return await coding.run_for_domain(
        "condition",
        payload=payload,
        collection="test",
        generator=SimpleNamespace(encode=AsyncMock(return_value=[0.1])),
        cosine_cutoff=0.8,
        db=None,
        schema=None,
        current_user=SimpleNamespace(email="test@example.invalid"),
        start_time=0,
        include_reasoning=include_reasoning,
    )


@pytest.mark.parametrize("option", ["google_search_fallback", "use_hybrid"])
async def test_domain_cache_hit_respects_search_mode(monkeypatch, cache_backend, option):
    async def search(payload, *args, **kwargs):
        return False, {payload.query: [{"mode": getattr(payload, option)}]}, {}

    search_mock = AsyncMock(side_effect=search)
    monkeypatch.setattr(coding, "_handle_general_search", search_mock)
    baseline = MedicalCodingRequest(query="probe")
    changed = baseline.model_copy(update={option: True})
    original_result = (await _run(baseline))[1]
    assert (await _run(baseline))[1] == original_result
    assert search_mock.await_count == 1
    changed_result = (await _run(changed))[1]
    assert changed_result != original_result
    assert (await _run(changed))[1] == changed_result
    assert search_mock.await_count == 2
    assert len(cache_backend) == 2


@pytest.mark.parametrize("include_reasoning", [False, True])
async def test_invalid_filter_is_an_explicit_error_and_never_cached(monkeypatch, cache_backend, include_reasoning):
    frame = pd.DataFrame({"CONCEPT_ID": [1, 2], "CONCEPT_NAME": ["One", "Two"]})
    monkeypatch.setattr(coding, "_qdrant", SimpleNamespace(search_qdrant_hybrid=AsyncMock(return_value=frame)))
    connector = SimpleNamespace(complete_json=AsyncMock(return_value='{"include": [true], "exclude": [0]}'))
    monkeypatch.setattr(filtering, "get_llm_connector", lambda _: connector)
    lts_update = AsyncMock()
    monkeypatch.setattr(filtering.lts, "update_lts", lts_update)
    request = MedicalCodingRequest(
        query="probe", llm_filter="haiku", include_descendants=False, use_lts=True, allow_lts=True
    )

    with pytest.raises(HTTPException) as error:
        await _run(request, include_reasoning=include_reasoning)
    assert error.value.status_code == 502
    assert "no unfiltered result" in error.value.detail
    assert connector.complete_json.await_count == 3
    assert cache_backend == {}
    coding.set_cached.assert_not_awaited()
    lts_update.assert_not_awaited()
