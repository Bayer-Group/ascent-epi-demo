"""Unit tests for ascent_domain.literature_result_utils."""


from ascent_domain.literature_result_utils import (
    build_google_answers,
    build_sources,
    normalize_google_references,
    normalize_literature_references,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

VERIFIED_LIT_TWO_ANSWERS = {
    "all_answers": [
        {
            "question": "What is the prevalence of stroke?",
            "answer": "Prevalence is 2-3% [1].",
            "verified_claims_result": [
                {
                    "claim": "Prevalence is 2-3%.",
                    "attribution": "LITERATURE_ANSWER",
                    "explanation": "Direct literature search result",
                    "references": [
                        {
                            "doc_index": 0,
                            "title": "Stroke Prevalence Review",
                            "pid": "EMBASE:L100",
                            "abstract": "Abstract A",
                        }
                    ],
                }
            ],
        },
        {
            "question": "What treatments exist for stroke?",
            "answer": "Alteplase is used [1][2].",
            "verified_claims_result": [
                {
                    "claim": "Alteplase is used.",
                    "attribution": "LITERATURE_ANSWER",
                    "explanation": "Direct literature search result",
                    "references": [
                        {
                            "doc_index": 0,
                            "title": "Stroke Prevalence Review",
                            "pid": "EMBASE:L100",
                            "abstract": "Abstract A",
                        },
                        {
                            "doc_index": 1,
                            "title": "Alteplase Meta-Analysis",
                            "pid": "PUBMED:200",
                            "abstract": "Abstract B",
                        },
                    ],
                }
            ],
        },
    ]
}

GOOGLE_RESULTS_ONE = [
    {
        "question": "Stroke statistics globally",
        "answer": "Stroke affects millions [1].",
        "sources": [{"uri": "https://who.int/stroke", "title": "WHO Stroke"}],
    }
]




# ---------------------------------------------------------------------------
# build_sources
# ---------------------------------------------------------------------------


class TestBuildSources:
    def test_builds_literature_sources_from_verified_lit(self):
        sources = build_sources(VERIFIED_LIT_TWO_ANSWERS, None)
        lit_sources = [s for s in sources if s["source_type"] == "literature"]
        # Two unique PIDs: EMBASE:L100 and PUBMED:200
        assert len(lit_sources) == 2
        pids = {s["pid"] for s in lit_sources}
        assert pids == {"EMBASE:L100", "PUBMED:200"}

    def test_builds_web_sources_from_google_results(self):
        sources = build_sources({"all_answers": []}, GOOGLE_RESULTS_ONE)
        web_sources = [s for s in sources if s["source_type"] == "web"]
        assert len(web_sources) == 1
        assert web_sources[0]["url"] == "https://who.int/stroke"
        assert web_sources[0]["pid"] is None
        assert web_sources[0]["abstract"] is None

    def test_deduplicates_literature_sources_by_pid(self):
        # Same PID appears in two different answers — should appear once
        verified = {
            "all_answers": [
                {
                    "question": "Q1",
                    "answer": "A1",
                    "verified_claims_result": [
                        {
                            "claim": "C",
                            "references": [
                                {"doc_index": 0, "title": "T", "pid": "EMBASE:L1", "abstract": ""}
                            ],
                        }
                    ],
                },
                {
                    "question": "Q2",
                    "answer": "A2",
                    "verified_claims_result": [
                        {
                            "claim": "C",
                            "references": [
                                {"doc_index": 0, "title": "T", "pid": "EMBASE:L1", "abstract": ""}
                            ],
                        }
                    ],
                },
            ]
        }
        sources = build_sources(verified, None)
        assert len(sources) == 1

    def test_assigns_sequential_doc_indices(self):
        sources = build_sources(VERIFIED_LIT_TWO_ANSWERS, GOOGLE_RESULTS_ONE)
        indices = [s["doc_index"] for s in sources]
        assert indices == list(range(len(sources)))

    def test_returns_empty_list_for_empty_inputs(self):
        sources = build_sources({"all_answers": []}, None)
        assert sources == []

    def test_returns_empty_list_for_empty_all_answers(self):
        sources = build_sources({}, [])
        assert sources == []


# ---------------------------------------------------------------------------
# normalize_literature_references
# ---------------------------------------------------------------------------


class TestNormalizeLiteratureReferences:
    def test_leaves_answer_unchanged_when_no_sources(self):
        verified = {
            "all_answers": [
                {
                    "question": "Q",
                    "answer": "No citations here.",
                    "verified_claims_result": [],
                }
            ]
        }
        normalize_literature_references(verified, [])
        assert verified["all_answers"][0]["answer"] == "No citations here."

    def test_handles_empty_all_answers(self):
        verified: dict = {"all_answers": []}
        normalize_literature_references(verified, [])  # should not raise

    def test_handles_missing_answer_key(self):
        verified = {
            "all_answers": [
                {
                    "question": "Q",
                    # no "answer" key
                    "verified_claims_result": [],
                }
            ]
        }
        normalize_literature_references(verified, [])  # should not raise

    def test_citation_not_replaced_when_no_matching_mapping(self):
        """Citations that have no PID mapping are left as-is."""
        verified = {
            "all_answers": [
                {
                    "question": "Q",
                    "answer": "See reference [5].",
                    "verified_claims_result": [],
                }
            ]
        }
        normalize_literature_references(verified, [])
        assert "[5]" in verified["all_answers"][0]["answer"]


# ---------------------------------------------------------------------------
# normalize_google_references
# ---------------------------------------------------------------------------


class TestNormalizeGoogleReferences:
    def test_replaces_bare_citation_with_google_link(self):
        results = [
            {
                "question": "Q",
                "answer": "Stroke affects millions [1].",
                "sources": [{"uri": "https://who.int/stroke", "title": "WHO"}],
            }
        ]
        sources = build_sources({"all_answers": []}, results)
        normalize_google_references(results, sources)
        assert "[1]" not in results[0]["answer"]
        assert "G1" in results[0]["answer"]
        assert "who.int" in results[0]["answer"]

    def test_does_not_replace_in_sources_footer(self):
        """Citations in the **Sources:** footer section are left untouched."""
        results = [
            {
                "question": "Q",
                "answer": "Body text [1].\n\n---\n\n**Sources:**\n[1] WHO",
                "sources": [{"uri": "https://who.int/stroke", "title": "WHO"}],
            }
        ]
        sources = build_sources({"all_answers": []}, results)
        normalize_google_references(results, sources)
        # The footer [1] should remain unchanged
        assert "[1] WHO" in results[0]["answer"]
        # But the body citation should be replaced
        assert "G1" in results[0]["answer"].split("\n\n---\n\n")[0]

    def test_handles_none_google_results(self):
        normalize_google_references(None, [])  # should not raise

    def test_handles_empty_google_results(self):
        normalize_google_references([], [])  # should not raise

    def test_citation_not_replaced_when_index_out_of_range(self):
        """[5] with only 1 web source is left as-is."""
        results = [
            {
                "question": "Q",
                "answer": "See [5].",
                "sources": [{"uri": "https://who.int", "title": "WHO"}],
            }
        ]
        sources = build_sources({"all_answers": []}, results)
        normalize_google_references(results, sources)
        assert "[5]" in results[0]["answer"]


# ---------------------------------------------------------------------------
# build_google_answers
# ---------------------------------------------------------------------------


class TestBuildGoogleAnswers:
    def test_extracts_question_and_answer(self):
        results = [{"question": "Q1", "answer": "A1", "sources": []}]
        answers = build_google_answers(results)
        assert len(answers) == 1
        assert answers[0] == {"question": "Q1", "answer": "A1"}

    def test_strips_sources_footer(self):
        results = [
            {"question": "Q", "answer": "Body text.\n\n---\n\n**Sources:**\n[1] WHO", "sources": []}
        ]
        answers = build_google_answers(results)
        assert answers[0]["answer"] == "Body text."
        assert "**Sources:**" not in answers[0]["answer"]

    def test_skips_empty_answers(self):
        results = [
            {"question": "Q1", "answer": "", "sources": []},
            {"question": "Q2", "answer": "Has content.", "sources": []},
        ]
        answers = build_google_answers(results)
        assert len(answers) == 1
        assert answers[0]["question"] == "Q2"

    def test_returns_empty_for_none(self):
        assert build_google_answers(None) == []

    def test_returns_empty_for_empty_list(self):
        assert build_google_answers([]) == []

    def test_multiple_results(self):
        results = [
            {"question": "Q1", "answer": "A1", "sources": []},
            {"question": "Q2", "answer": "A2", "sources": []},
            {"question": "Q3", "answer": "A3", "sources": []},
        ]
        answers = build_google_answers(results)
        assert len(answers) == 3
        assert [a["question"] for a in answers] == ["Q1", "Q2", "Q3"]
