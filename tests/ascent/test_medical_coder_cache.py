"""A coder result is only cached when it has codes.

The medical coder warms three embedding models at boot and the backend reports
healthy before it can answer, so the first lookups of a fresh stack can come
back empty. Caching that answer turns one slow start into a vocabulary that
resolves to nothing for the whole TTL, with no error anywhere.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ascent_domain.non_omop.medical_coding import sql_postprocessing as sp

CODES = {"response_data": {"asthma": [{"CONCEPT_CODE": "Q30.0"}]}}
EMPTY = {"response_data": {}}
EMPTY_LIST = {"response_data": {"asthma": []}}


@pytest.fixture
def redis_client():
    client = MagicMock()
    client.get.return_value = None
    with patch.object(sp, "_get_redis_client", return_value=client):
        yield client


def _fetch(result):
    return patch.object(sp, "execute_medical_coder_full_result", return_value=result)


@pytest.mark.parametrize("result", [EMPTY, EMPTY_LIST], ids=["no-keys", "empty-list"])
def test_an_empty_result_is_not_cached(redis_client, result):
    with _fetch(result):
        sp.get_cached_medical_coder_data("condition", "asthma", "DXCODE", top_k=10)

    redis_client.setex.assert_not_called()


def test_a_result_with_codes_is_cached(redis_client):
    with _fetch(CODES):
        sp.get_cached_medical_coder_data("condition", "asthma", "DXCODE", top_k=10)

    redis_client.setex.assert_called_once()


def test_a_later_call_still_reaches_the_coder_after_an_empty_one(redis_client):
    """The failure this guards: recovery must not need a cache flush."""
    with _fetch(EMPTY):
        sp.get_cached_medical_coder_data("condition", "asthma", "DXCODE", top_k=10)

    with _fetch(CODES) as fetch:
        out = sp.get_cached_medical_coder_data("condition", "asthma", "DXCODE", top_k=10)

    assert fetch.called
    assert out["response_data"]["asthma"]
