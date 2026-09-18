from __future__ import annotations

import json

import pytest

from ascent_medical_coder.db.cache import cache_key_for_request, canonical_cache_key, hash_key
from ascent_medical_coder.schemas.coding import MedicalCodingRequest
from ascent_medical_coder.services.request_validation import normalize_domain_id, validate_mixed_domains


def _cache_key(**overrides):
    values = {
        "query": " hypertension ",
        "domain_id": "Condition",
        "top_k": 100,
        "encoder": "bge",
        "vocabulary": "SNOMED",
        "standard_concept": "S",
        "source": None,
        "llm_filter": None,
        "allow_lts": False,
        "database": "db",
        "custom_instructions": None,
        "cosine_similarity": None,
        "include_descendants": True,
    }
    values.update(overrides)
    return canonical_cache_key(**values)


def test_cache_key_is_stable_and_separates_search_options():
    baseline = _cache_key()
    assert baseline == _cache_key()
    assert baseline != _cache_key(cosine_similarity=0.9)
    assert baseline != _cache_key(include_descendants=False)
    assert hash_key(baseline) == hash_key(baseline)
    assert len(hash_key(baseline)) == 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("query", "another query"),
        ("top_k", 200),
        ("encoder", "gemini"),
        ("vocabulary", ["ICD10CM"]),
        ("standard_concept", "S"),
        ("source", "test"),
        ("llm_filter", "haiku"),
        ("allow_lts", True),
        ("database", "db"),
        ("custom_instructions", "only exact matches"),
        ("cosine_similarity", 0.9),
        ("include_descendants", False),
        ("google_search_fallback", True),
        ("use_hybrid", True),
    ],
)
def test_request_keys_isolate_every_search_option(field, value):
    request = MedicalCodingRequest(query="probe")
    baseline = cache_key_for_request(request, "condition")
    changed = cache_key_for_request(request.model_copy(update={field: value}), "condition")
    assert baseline != changed
    assert json.loads(baseline)["version"] == 2


@pytest.mark.parametrize("option", ["google_search_fallback", "use_hybrid"])
def test_cached_result_cannot_cross_search_modes(option):
    request = MedicalCodingRequest(query="probe")
    cache = {cache_key_for_request(request, "condition"): {"probe": ["original result"]}}
    assert cache[cache_key_for_request(request, "condition")] == {"probe": ["original result"]}
    changed = request.model_copy(update={option: True})
    assert cache.get(cache_key_for_request(changed, "condition")) is None


def test_request_keys_preserve_response_query_and_normalize_vocabulary():
    assert _cache_key(query="probe") != _cache_key(query=" probe ")
    assert _cache_key(vocabulary=["SNOMED", "ICD10CM"]) == _cache_key(vocabulary=["ICD10CM", "SNOMED"])
    request = MedicalCodingRequest(query="probe")
    assert cache_key_for_request(request, "Condition") == cache_key_for_request(request, "condition")
    assert cache_key_for_request(request, "condition") != cache_key_for_request(request, "procedure")
    assert cache_key_for_request(request, None) == cache_key_for_request(
        request.model_copy(update={"use_lts": True}), None
    )


def test_domain_normalization_and_mixed_domain_validation():
    assert normalize_domain_id(" Condition ") == "condition"
    assert normalize_domain_id(" ") is None
    assert normalize_domain_id(None) is None

    assert validate_mixed_domains(["drug", "drug_class"]) is None
    assert validate_mixed_domains(["condition", "procedure"]) is None
    error = validate_mixed_domains(["drug", "condition"])
    assert error["error"] == "MixedDomainIdsNotAllowed"
    assert error["received"] == {
        "drug_like": ["drug"],
        "non_drug_like": ["condition"],
    }
