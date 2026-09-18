"""Every date format the generator emits has to be one the warehouse can read.

The two halves are deliberately separate: ``synthetic_data_generation`` imports
nothing from ``src``, so the generator can be run and shipped on its own. That
leaves the knowledge of which formats exist in two places -- the facility
catalog that decides what gets written, and ``_DATE_FORMATS`` that decides what
can be parsed back.

Asserting the relationship rather than sharing a constant keeps the separation
and still fails when the two drift. Drift is silent otherwise: an unparseable
date becomes NULL during normalisation, so the load succeeds, the column is
typed, and rows simply go missing from every date-filtered query.
"""

from __future__ import annotations

import json
import pathlib

from ascent_platform.warehouse.bootstrap import _DATE_FORMATS

CATALOG = pathlib.Path(__file__).resolve().parents[2] / "synthetic_data_generation" / "catalog" / "demographics.json"

# Neither layer treats this as a strptime pattern: the writer emits seconds
# since the epoch, and the warehouse matches the digits and calls to_timestamp.
_NOT_A_STRPTIME_PATTERN = {"epoch"}


def test_catalog_exists() -> None:
    """rglob-style guards pass vacuously; a moved catalog must fail loudly."""
    assert CATALOG.is_file(), f"facility catalog not found at {CATALOG}"


def test_every_emitted_date_format_can_be_parsed_back() -> None:
    facilities = json.loads(CATALOG.read_text())["facilities"]
    assert facilities, "catalog declares no facilities, so this test would assert nothing"

    emitted = {f["date_format"] for f in facilities}
    unreadable = {fmt for fmt in emitted - _NOT_A_STRPTIME_PATTERN if fmt not in _DATE_FORMATS}

    assert not unreadable, (
        f"facility catalog emits {sorted(unreadable)}, which "
        f"ascent_platform.warehouse.bootstrap._DATE_FORMATS cannot parse "
        f"({list(_DATE_FORMATS)}). Dates in that format normalise to NULL and "
        f"the rows disappear from every date-filtered query."
    )
