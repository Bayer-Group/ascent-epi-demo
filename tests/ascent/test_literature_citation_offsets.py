"""Citation markers must resolve against the answer that carries them.

``build_sources`` flattens every batch's web sources into one list, while each
answer numbers its own sources from 1, so each answer's markers must be
resolved against its own slice of that list rather than against the whole.
"""

from __future__ import annotations

from ascent_domain.literature_result_utils import build_sources, normalize_google_references


def _result(question: str, answer: str, urls: list[str]) -> dict:
    return {
        "question": question,
        "answer": answer,
        "sources": [{"uri": u, "title": u} for u in urls],
    }


def test_second_batch_markers_resolve_to_its_own_sources():
    google = [
        _result("first", "Prevalence is rising [1].", ["https://a.example/1"]),
        _result("second", "Incidence differs [1] and [2].", ["https://b.example/1", "https://b.example/2"]),
    ]
    sources = build_sources({}, google)
    normalize_google_references(google, sources)

    assert "https://a.example/1" in google[0]["answer"]
    assert "https://b.example/1" in google[1]["answer"]
    assert "https://b.example/2" in google[1]["answer"]
    assert "https://a.example/1" not in google[1]["answer"]


def test_single_batch_is_unchanged():
    google = [_result("only", "Result [1] and [2].", ["https://a.example/1", "https://a.example/2"])]
    sources = build_sources({}, google)
    normalize_google_references(google, sources)

    assert "https://a.example/1" in google[0]["answer"]
    assert "https://a.example/2" in google[0]["answer"]


def test_a_marker_beyond_the_answers_own_sources_is_left_alone():
    google = [
        _result("first", "Claim [2].", ["https://a.example/1"]),
        _result("second", "Other.", ["https://b.example/1"]),
    ]
    sources = build_sources({}, google)
    normalize_google_references(google, sources)

    assert google[0]["answer"] == "Claim [2]."


def test_sources_without_a_url_do_not_shift_the_offset():
    """build_sources skips a source with no URL, so the mapping must too."""
    google = [
        {"question": "first", "answer": "Claim [1].", "sources": [{"title": "no url"}, {"uri": "https://a.example/1"}]},
        _result("second", "Other [1].", ["https://b.example/1"]),
    ]
    sources = build_sources({}, google)
    normalize_google_references(google, sources)

    assert "https://b.example/1" in google[1]["answer"]
