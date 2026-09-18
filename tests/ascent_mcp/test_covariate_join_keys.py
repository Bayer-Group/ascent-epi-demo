"""Regression tests for the covariate join key.

Both sides of the covariate merge used to be pushed through
``pd.to_numeric(errors="coerce")``. With OMOP person_id that is harmless, but
the source layer keys on MRN ("RBM0000001"): every value becomes NaN, and
pandas *matches NaN to NaN*, so the left join fanned a 90-patient cohort out
to 90,000 rows. It surfaced as an unrelated-looking INT128 cast error one
step later, having already replaced the cohort table.
"""

import pandas as pd
import pytest

from ascent_mcp.tools_v1_cohort import _aligned_join_keys


def _merge(base: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    base_key, cov_key = _aligned_join_keys(base["SUBJECT_ID"], cov["SUBJECT_ID"])
    return base.assign(SUBJECT_ID=base_key).merge(
        cov.assign(SUBJECT_ID=cov_key), on="SUBJECT_ID", how="left"
    )


def test_non_numeric_ids_do_not_fan_out():
    """The exact shape that produced 90,000 rows from 90 patients."""
    base = pd.DataFrame({"SUBJECT_ID": [f"RBM{i:07d}" for i in range(1, 91)]})
    cov = pd.DataFrame(
        {"SUBJECT_ID": [f"RBM{i:07d}" for i in range(1, 1001)], "SEX": ["F"] * 1000}
    )

    merged = _merge(base, cov)

    assert len(merged) == 90
    assert merged["SEX"].notna().all()


def test_numeric_ids_still_match():
    base = pd.DataFrame({"SUBJECT_ID": [1, 2, 3]})
    cov = pd.DataFrame({"SUBJECT_ID": [1, 2, 3], "YEAR": [1990, 1991, 1992]})

    merged = _merge(base, cov)

    assert merged["YEAR"].tolist() == [1990, 1991, 1992]


def test_numeric_cohort_matches_text_ids_from_the_query():
    """A driver returning ids as text must still join to a numeric cohort."""
    base = pd.DataFrame({"SUBJECT_ID": [1, 2, 3]})
    cov = pd.DataFrame({"SUBJECT_ID": ["1", "2", "3"], "X": [7, 8, 9]})

    merged = _merge(base, cov)

    assert merged["X"].tolist() == [7, 8, 9]


def test_float_ids_do_not_acquire_a_decimal_point():
    """str(1.0) is "1.0", which matches no id anywhere."""
    base = pd.DataFrame({"SUBJECT_ID": [1.0, 2.0]})
    cov = pd.DataFrame({"SUBJECT_ID": ["1", "MRN2"], "X": [5, 6]})

    merged = _merge(base, cov)

    assert len(merged) == 2
    assert merged["X"].tolist() == [5, pytest.approx(float("nan"), nan_ok=True)]


def test_unmatched_ids_yield_no_matches_rather_than_a_product():
    base = pd.DataFrame({"SUBJECT_ID": ["A1", "A2"]})
    cov = pd.DataFrame({"SUBJECT_ID": ["B1", "B2", "B3"], "X": [1, 2, 3]})

    merged = _merge(base, cov)

    assert len(merged) == 2
    assert merged["X"].isna().all()
