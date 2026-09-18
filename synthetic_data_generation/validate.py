#!/usr/bin/env python3
"""Referential-integrity checks and a realism report for the generated data.

Two kinds of check live here, and they are deliberately kept apart:

* **Failures** are invariants. A dataset that breaks one is wrong - a diagnosis
  before birth, a code that did not exist yet, a potassium of zero - and the
  build stops.
* **Warnings** are distributional. They compare the finished data against the
  shape the catalog asked for, and a small cohort can miss the target by chance
  without anything being broken. They print, they do not fail the build.

Anything the generator plants on purpose is checked against its documented
count rather than against zero. A suite that demands zero injected defects
would force the generator to produce data with no defects at all, which is the
one thing a test dataset must never do.
"""

import math
import sqlite3
from datetime import date
from typing import Dict, List, Tuple

from models import Catalog

# Bands used for the population-shape comparison, matching age_pyramid.bands.
KS_ALPHA_CRITICAL = 1.36  # two-sided 5% Kolmogorov-Smirnov coefficient


def validate_databases(cat: Catalog, omop_db: str, source_db: str, gt: dict) -> Tuple[List[str], List[str]]:
    """Return (hard failures, advisory warnings)."""
    failures: List[str] = []
    warnings: List[str] = []
    planted = gt.get("injected_errors", {})

    conn = sqlite3.connect(omop_db)
    conn.row_factory = sqlite3.Row
    try:

        def count(sql: str, params=()) -> int:
            return conn.execute(sql, params).fetchone()[0]

        def check(label: str, sql: str, params=()) -> None:
            n = count(sql, params)
            if n:
                failures.append(f"{label}: {n} offending rows")

        def check_equals(label: str, sql: str, expected: int, params=()) -> None:
            """A deliberately planted defect must appear exactly as often as documented."""
            n = count(sql, params)
            if n != expected:
                failures.append(f"{label}: {n} rows, expected the documented {expected}")

        def check_at_least(label: str, sql: str, expected: int, params=()) -> None:
            """For defects the data can also produce honestly, only the shortfall is an error."""
            n = count(sql, params)
            if n < expected:
                failures.append(f"{label}: {n} rows, fewer than the documented {expected}")
            elif n > expected:
                warnings.append(f"{label}: {n} rows, of which {expected} were planted")

        _structural_checks(conn, cat, check)
        _temporal_checks(conn, check)
        _plausibility_checks(conn, cat, check)
        _existence_window_checks(cat, check)
        _anachronism_checks(conn, check, warnings)
        _injected_checks(conn, check, check_equals, check_at_least, planted)

        _distribution_warnings(cat, conn, gt, warnings)
    finally:
        conn.close()

    _source_checks(omop_db, source_db, failures)
    return failures, warnings


# --------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------

_CLINICAL_TABLES = [
    ("visit_occurrence", "visit_start_date"),
    ("condition_occurrence", "condition_start_date"),
    ("drug_exposure", "drug_exposure_start_date"),
    ("measurement", "measurement_date"),
    ("procedure_occurrence", "procedure_date"),
    ("observation", "observation_date"),
]

_SOURCE_CONCEPT_COLUMNS = [
    ("condition_occurrence", "condition_start_date", "condition_source_concept_id"),
    ("drug_exposure", "drug_exposure_start_date", "drug_source_concept_id"),
    ("measurement", "measurement_date", "measurement_source_concept_id"),
    ("procedure_occurrence", "procedure_date", "procedure_source_concept_id"),
]


def _structural_checks(conn, cat: Catalog, check) -> None:
    for table, column in [
        ("visit_occurrence", "visit_concept_id"),
        ("condition_occurrence", "condition_concept_id"),
        ("drug_exposure", "drug_concept_id"),
        ("measurement", "measurement_concept_id"),
        ("procedure_occurrence", "procedure_concept_id"),
        ("observation", "observation_concept_id"),
        ("death", "cause_concept_id"),
    ]:
        check(
            f"{table}.{column} not in concept",
            f"SELECT COUNT(*) FROM {table} t LEFT JOIN concept c ON c.concept_id = t.{column} WHERE c.concept_id IS NULL",
        )

    for table, _ in _CLINICAL_TABLES + [("observation_period", ""), ("death", "")]:
        check(
            f"{table}.person_id orphaned",
            f"SELECT COUNT(*) FROM {table} t LEFT JOIN person p ON p.person_id = t.person_id WHERE p.person_id IS NULL",
        )

    # Descendant expansion is only correct if every standard concept is its own
    # ancestor at zero levels of separation. A missing self-row silently drops
    # the concept itself out of every "X and its descendants" query.
    check(
        "standard concept without a concept_ancestor self-row",
        """SELECT COUNT(*) FROM concept c WHERE c.standard_concept = 'S'
           AND NOT EXISTS (SELECT 1 FROM concept_ancestor a
                           WHERE a.ancestor_concept_id = c.concept_id
                           AND a.descendant_concept_id = c.concept_id
                           AND a.min_levels_of_separation = 0)""",
    )
    for column in ["ancestor_concept_id", "descendant_concept_id"]:
        check(
            f"concept_ancestor.{column} not in concept",
            f"SELECT COUNT(*) FROM concept_ancestor a LEFT JOIN concept c ON c.concept_id = a.{column} WHERE c.concept_id IS NULL",
        )
    # An "Is a" edge with no matching closure row means the closure was built
    # from a stale edge list, which is exactly the failure a materialised
    # ancestor table exists to make impossible.
    check(
        "'Is a' edge missing from concept_ancestor",
        """SELECT COUNT(*) FROM concept_relationship r WHERE r.relationship_id = 'Is a'
           AND NOT EXISTS (SELECT 1 FROM concept_ancestor a
                           WHERE a.ancestor_concept_id = r.concept_id_2
                           AND a.descendant_concept_id = r.concept_id_1)""",
    )

    # concept 0 is the sentinel: it maps to nothing by definition.
    check(
        "source concept without a Maps to relationship",
        """SELECT COUNT(*) FROM concept c WHERE c.standard_concept = '' AND c.concept_id != 0
           AND NOT EXISTS (SELECT 1 FROM concept_relationship r
                           WHERE r.concept_id_1 = c.concept_id AND r.relationship_id = 'Maps to')""",
    )

    for key, wrong in [("prostate_cancer", "F"), ("breast_cancer", "M"), ("osteoporosis", "M")]:
        check(
            f"{key} recorded in sex {wrong}",
            """SELECT COUNT(*) FROM condition_occurrence co
               JOIN concept c ON c.concept_id = co.condition_concept_id
               JOIN person p ON p.person_id = co.person_id
               WHERE c.concept_name = ? AND p.gender_source_value = ?""",
            (cat.conditions[key].name, wrong),
        )

    check(
        "measurement without a unit",
        "SELECT COUNT(*) FROM measurement WHERE unit_concept_id = 0 OR unit_concept_id IS NULL",
    )


def _temporal_checks(conn, check) -> None:
    check(
        "visit ends before it starts",
        "SELECT COUNT(*) FROM visit_occurrence WHERE visit_end_date < visit_start_date",
    )
    check(
        "observation period ends before it starts",
        "SELECT COUNT(*) FROM observation_period WHERE observation_period_end_date < observation_period_start_date",
    )

    # Nothing may happen before the patient was born. This is the check that
    # catches a coverage window opening before the date of birth.
    birth = "printf('%04d-%02d-%02d', p.year_of_birth, p.month_of_birth, p.day_of_birth)"
    for table, datecol in _CLINICAL_TABLES:
        check(
            f"{table} recorded before birth",
            f"SELECT COUNT(*) FROM {table} t JOIN person p ON p.person_id = t.person_id WHERE t.{datecol} < {birth}",
        )
    check(
        "observation period starts before birth",
        f"SELECT COUNT(*) FROM observation_period t JOIN person p ON p.person_id = t.person_id WHERE t.observation_period_start_date < {birth}",
    )
    check(
        "death recorded before birth",
        f"SELECT COUNT(*) FROM death t JOIN person p ON p.person_id = t.person_id WHERE t.death_date < {birth}",
    )

    # ...and nothing after the patient dies.
    for table, datecol in _CLINICAL_TABLES:
        check(
            f"{table} recorded after death",
            f"SELECT COUNT(*) FROM {table} t JOIN death d ON d.person_id = t.person_id WHERE t.{datecol} > d.death_date",
        )

    # Encounters and diagnoses must fall inside a coverage window. Measurements
    # are exempt here because the error budget deliberately pushes a counted
    # handful outside it; that count is asserted separately.
    for table, datecol in [
        ("visit_occurrence", "visit_start_date"),
        ("condition_occurrence", "condition_start_date"),
    ]:
        check(
            f"{table} outside observation period",
            f"""SELECT COUNT(*) FROM {table} t WHERE NOT EXISTS (
                    SELECT 1 FROM observation_period o WHERE o.person_id = t.person_id
                    AND t.{datecol} BETWEEN o.observation_period_start_date
                                        AND o.observation_period_end_date)""",
        )


def _plausibility_checks(conn, cat: Catalog, check) -> None:
    """No result may sit outside the analyte's physiologic range.

    The old generator floored lab draws at zero, so an unresulted test arrived
    as a serum potassium of 0 mmol/L - a value that is not merely abnormal but
    incompatible with life, and one that every downstream mean quietly absorbed.
    """
    for key, lab in sorted(cat.labs.items()):
        check(
            f"{key} outside its physiologic range",
            """SELECT COUNT(*) FROM measurement m JOIN concept c ON c.concept_id = m.measurement_concept_id
               WHERE c.concept_name = ? AND m.value_as_number IS NOT NULL
               AND (m.value_as_number < ? OR m.value_as_number > ?)""",
            (lab.name, lab.plausible_min, lab.plausible_max),
        )

    check(
        "measurement with a value but no unit source value",
        "SELECT COUNT(*) FROM measurement WHERE value_as_number IS NOT NULL AND value_source_value IS NULL",
    )
    check(
        "unresulted measurement carrying a value_source_value",
        "SELECT COUNT(*) FROM measurement WHERE value_as_number IS NULL AND value_source_value IS NOT NULL",
    )


def _existence_window_checks(cat: Catalog, check) -> None:
    """A disease, drug or procedure cannot be recorded before it existed.

    The vocabulary check below asks whether the *code* existed; this asks
    whether the *thing* did. COVID-19 in 2016 carries a code that was perfectly
    valid in 2016, so only this check catches it.
    """
    targets = [
        ("condition_occurrence", "condition_start_date", "condition_concept_id", cat.conditions),
        ("drug_exposure", "drug_exposure_start_date", "drug_concept_id", cat.drugs),
        ("procedure_occurrence", "procedure_date", "procedure_concept_id", cat.procedures),
    ]
    for table, datecol, concept_col, items in targets:
        for key, item in sorted(items.items()):
            if item.valid_from:
                check(
                    f"{key} recorded before {item.valid_from.isoformat()}, when it first existed",
                    f"""SELECT COUNT(*) FROM {table} t JOIN concept c ON c.concept_id = t.{concept_col}
                        WHERE c.concept_name = ? AND t.{datecol} < ?""",
                    (item.name, item.valid_from.isoformat()),
                )
            if item.valid_to:
                check(
                    f"{key} recorded after {item.valid_to.isoformat()}, when it was withdrawn",
                    f"""SELECT COUNT(*) FROM {table} t JOIN concept c ON c.concept_id = t.{concept_col}
                        WHERE c.concept_name = ? AND t.{datecol} > ?""",
                    (item.name, item.valid_to.isoformat()),
                )


def _anachronism_checks(conn, check, warnings: List[str]) -> None:
    """No code may be used before the day it was issued.

    A record dated 2012 carrying a code from a classification introduced in 2015
    is the cheapest possible proof that a dataset was fabricated, and it is the
    first thing a sceptical reviewer looks for.
    """
    for table, datecol, concept_col in _SOURCE_CONCEPT_COLUMNS:
        check(
            f"{table} coded before its vocabulary was issued",
            f"""SELECT COUNT(*) FROM {table} t JOIN concept c ON c.concept_id = t.{concept_col}
                WHERE CAST(REPLACE(t.{datecol}, '-', '') AS INTEGER) < c.valid_start_date""",
        )

    # Use AFTER retirement is not an error: it is the cutover straggler tail,
    # which is real and deliberate. It is surfaced so nobody mistakes it for one.
    n = conn.execute(
        """SELECT COUNT(*) FROM condition_occurrence t JOIN concept c ON c.concept_id = t.condition_source_concept_id
           WHERE CAST(REPLACE(t.condition_start_date, '-', '') AS INTEGER) > c.valid_end_date"""
    ).fetchone()[0]
    if n:
        warnings.append(f"cutover straggler tail: {n:,} diagnoses coded in the retired classification (expected)")


def _injected_checks(conn, check, check_equals, check_at_least, planted: Dict[str, int]) -> None:
    """Every planted defect must be present, and present exactly as often as documented."""
    check_equals(
        "orphaned visit references in condition_occurrence",
        "SELECT COUNT(*) FROM condition_occurrence WHERE visit_occurrence_id IS NULL",
        planted.get("orphan_visit_reference", 0),
    )
    check_equals(
        "measurements outside the observation period",
        """SELECT COUNT(*) FROM measurement t WHERE NOT EXISTS (
               SELECT 1 FROM observation_period o WHERE o.person_id = t.person_id
               AND t.measurement_date BETWEEN o.observation_period_start_date
                                          AND o.observation_period_end_date)""",
        planted.get("records_outside_observation_period", 0),
    )
    # Two genuine same-day visits of the same type are possible, so surplus rows
    # here are coincidence rather than error; only a shortfall means the
    # injection did not run.
    check_at_least(
        "duplicated encounters",
        """SELECT COALESCE(SUM(n - 1), 0) FROM (
               SELECT COUNT(*) n FROM visit_occurrence
               GROUP BY person_id, visit_concept_id, visit_start_date, visit_end_date HAVING n > 1)""",
        planted.get("duplicate_encounter", 0),
    )

    # A visit reference that survives must point at a visit that exists.
    for table in ["condition_occurrence", "drug_exposure", "measurement", "procedure_occurrence"]:
        check(
            f"{table}.visit_occurrence_id points at a missing visit",
            f"""SELECT COUNT(*) FROM {table} t
                LEFT JOIN visit_occurrence v ON v.visit_occurrence_id = t.visit_occurrence_id
                WHERE t.visit_occurrence_id IS NOT NULL AND v.visit_occurrence_id IS NULL""",
        )


def _source_checks(omop_db: str, source_db: str, failures: List[str]) -> None:
    src = sqlite3.connect(source_db)
    try:
        omop_conn = sqlite3.connect(omop_db)
        n_omop = omop_conn.execute("SELECT COUNT(*) FROM person").fetchone()[0]
        omop_conn.close()

        # MRNs are site-local, so the patient count is the count of MEMBER_IDs.
        # Counting distinct MRNs would double-count every patient seen at two
        # facilities.
        n_src = src.execute("SELECT COUNT(DISTINCT MEMBER_ID) FROM PATIENT_MASTER").fetchone()[0]
        if n_omop != n_src:
            failures.append(f"patient count mismatch: OMOP {n_omop} vs source {n_src}")

        orphan = src.execute("SELECT COUNT(*) FROM DX d LEFT JOIN PATIENT_MASTER p ON p.MRN = d.MRN WHERE p.MRN IS NULL").fetchone()[0]
        if orphan:
            failures.append(f"source DX rows with unknown MRN: {orphan}")

        # A result value with no status, or a status of ORDERED with a value,
        # would make the unresulted-test quirk unreadable.
        bad = src.execute("SELECT COUNT(*) FROM LAB_RESULTS WHERE (RESULT_STATUS = 'ORDERED') != (RESULT_VAL IS NULL)").fetchone()[0]
        if bad:
            failures.append(f"source LAB_RESULTS rows where RESULT_STATUS disagrees with RESULT_VAL: {bad}")
    finally:
        src.close()


# --------------------------------------------------------------------------
# Distributional shape: warn, never fail
# --------------------------------------------------------------------------


def _distribution_warnings(cat: Catalog, conn, gt: dict, warnings: List[str]) -> None:
    _monotonic_age_warnings(cat, gt, warnings)
    _pyramid_warning(cat, gt, warnings)
    _seasonality_warnings(cat, conn, warnings)
    _visit_periodicity_warnings(cat, conn, warnings)
    _visit_mix_warnings(cat, conn, warnings)


def _monotonic_age_warnings(cat: Catalog, gt: dict, warnings: List[str]) -> None:
    """Age-related disease must not become rarer as the population ages.

    A dip in the oldest band is the signature of a prevalence sampler that has
    saturated, and it is invisible in the marginal rate.
    """
    for key, cond in sorted(cat.conditions.items()):
        if not cond.monotonic_age:
            continue
        bands = gt["prevalence"].get(key, {}).get("by_age_band", {})
        previous = None
        for band, stats in bands.items():
            lo = int(band.split("-")[0].rstrip("+"))
            n = stats["denominator"]
            if n == 0 or lo < cond.age_min:
                continue
            rate = stats["rate"]
            if previous is not None:
                prev_band, prev_rate, prev_n = previous
                # Two-proportion test on the pooled rate. Using the observed rate
                # of the higher band alone gives a zero standard error whenever
                # that band happens to contain no cases, so a 60-person band with
                # one expected case would fail every time on a small cohort.
                pooled = (rate * n + prev_rate * prev_n) / (n + prev_n)
                se = math.sqrt(max(pooled * (1 - pooled), 1e-9) * (1 / n + 1 / prev_n))
                # A drop nobody could have detected is not evidence of one.
                detectable = pooled * (n + prev_n) >= 10
                if detectable and rate < prev_rate - 2 * se:
                    warnings.append(f"{key}: prevalence falls from {prev_rate:.3f} ({prev_band}) to {rate:.3f} ({band}) despite monotonic_age")
            previous = (band, rate, n)


def _pyramid_warning(cat: Catalog, gt: dict, warnings: List[str]) -> None:
    """Kolmogorov-Smirnov comparison against the sampling pyramid.

    The reference is share x utilization: an extract is not a census, it
    over-represents whoever uses healthcare. Infants born inside the window are
    in the file but not in the reference, so the threshold is loose.
    """
    bands = cat.demographics["age_pyramid"]["bands"]
    weights = [b["share"] * b.get("utilization", 1.0) for b in bands]
    total = sum(weights) or 1.0

    observed = gt["demographics"]["age_pyramid"]
    n = gt["demographics"]["n_patients"]

    # Collapse the 5-year reference bands onto the coarse bands the ground-truth
    # file publishes, so the two CDFs are comparable.
    coarse = [("0-17", 0, 17), ("18-44", 18, 44), ("45-64", 45, 64), ("65-79", 65, 79), ("80+", 80, 200)]
    expected = {name: 0.0 for name, _, _ in coarse}
    for band, w in zip(bands, weights):
        mid = (band["min"] + band["max"]) / 2
        for name, lo, hi in coarse:
            if lo <= mid <= hi:
                expected[name] += w / total
                break

    cdf_o = cdf_e = 0.0
    ks = 0.0
    for name, _, _ in coarse:
        cdf_o += observed.get(name, 0) / max(n, 1)
        cdf_e += expected[name]
        ks = max(ks, abs(cdf_o - cdf_e))

    critical = max(KS_ALPHA_CRITICAL / math.sqrt(max(n, 1)), 0.10)
    if ks > critical:
        warnings.append(
            f"age distribution differs from the reference pyramid: KS {ks:.3f} > {critical:.3f} "
            f"(observed {[observed.get(b, 0) for b, _, _ in coarse]})"
        )


MIN_PERIOD_VISITS = 300
# A seasonal profile is only tested where a perfect generator would clear this
# many standard deviations, so that a shallow profile on a small cohort is
# reported as untestable rather than as a defect.
MIN_SEASONAL_POWER = 4.0
SEASONAL_Z = 2.0
# Days of coverage a patient must have before a first diagnosis counts as an
# onset the file could have observed rather than a backlog code at enrolment.
INCIDENT_WASHOUT_DAYS = 365

# One first diagnosis per patient. Repeat rows are re-codings at follow-up
# visits and follow the visit calendar, not the disease, so counting them
# flattens every profile in the file.
_SQL_FIRST_DX = """SELECT CAST(strftime('%m', co.condition_start_date) AS INTEGER) m, COUNT(*) n
                   FROM (SELECT person_id, MIN(condition_start_date) condition_start_date
                         FROM condition_occurrence co
                         JOIN concept c ON c.concept_id = co.condition_concept_id
                         WHERE c.concept_name = ? GROUP BY person_id) co
                   GROUP BY 1"""

# Chronic disease additionally has to be incident: a patient who walked in with
# it already gets coded at their first visit, which says nothing about the
# season the disease actually started in.
_SQL_INCIDENT_DX = f"""SELECT CAST(strftime('%m', f.first_dx) AS INTEGER) m, COUNT(*) n
                       FROM (SELECT person_id, MIN(condition_start_date) first_dx
                             FROM condition_occurrence co
                             JOIN concept c ON c.concept_id = co.condition_concept_id
                             WHERE c.concept_name = ? GROUP BY person_id) f
                       JOIN (SELECT person_id, MIN(observation_period_start_date) entry
                             FROM observation_period GROUP BY person_id) o
                         ON o.person_id = f.person_id
                       WHERE julianday(f.first_dx) - julianday(o.entry) >= {INCIDENT_WASHOUT_DAYS}
                       GROUP BY 1"""


def _seasonality_warnings(cat: Catalog, conn, warnings: List[str]) -> None:
    """A seasonal disease must arrive in its season.

    The test is a likelihood ratio: how much better the observed month histogram
    is explained by the condition's profile than by a flat calendar, expressed in
    standard deviations of that statistic under a flat calendar. A correlation
    threshold was the wrong instrument - it fails a shallow profile on a small
    cohort, where month-to-month sampling noise is larger than the seasonal
    swing, and it cannot say whether an absent signal is a broken generator or a
    sample too small to show one. Here the power a perfect signal would have is
    computed first, and anything underpowered is reported as untested.
    """
    pooled: Dict[str, List[int]] = {}
    members: Dict[str, List[str]] = {}

    for key, cond in sorted(cat.conditions.items()):
        profile = cat.calendar.seasonality.get(cond.seasonality)
        if cond.seasonality == "none" or profile is None:
            continue

        counts = [0] * 12
        sql = _SQL_INCIDENT_DX if cond.chronic else _SQL_FIRST_DX
        for r in conn.execute(sql, (cond.name,)):
            counts[r["m"] - 1] = r["n"]

        z, power = _seasonal_z(counts, profile.monthly)
        if power >= MIN_SEASONAL_POWER:
            if z < SEASONAL_Z:
                peak = counts.index(max(counts)) + 1
                warnings.append(
                    f"{key}: monthly onsets do not track the '{cond.seasonality}' profile "
                    f"(z {z:+.1f} against {power:.1f} available, peak month {peak}, "
                    f"n={sum(counts):,})"
                )
            continue

        # Too rare to judge alone. Every condition on a profile is drawn from the
        # same twelve multipliers, so pooling them tests the profile itself and
        # recovers most of the power a single rare disease cannot supply.
        pool = pooled.setdefault(cond.seasonality, [0] * 12)
        for i, c in enumerate(counts):
            pool[i] += c
        members.setdefault(cond.seasonality, []).append(key)

    untestable: List[str] = []
    for name, counts in sorted(pooled.items()):
        profile = cat.calendar.seasonality[name]
        z, power = _seasonal_z(counts, profile.monthly)
        who = ", ".join(members[name])
        if power < MIN_SEASONAL_POWER:
            untestable.append(f"'{name}' ({who}): {sum(counts):,} onsets, power {power:.1f}")
        elif z < SEASONAL_Z:
            peak = counts.index(max(counts)) + 1
            warnings.append(
                f"pooled onsets for '{name}' ({who}) do not track the profile "
                f"(z {z:+.1f} against {power:.1f} available, peak month {peak}, "
                f"n={sum(counts):,})"
            )

    if untestable:
        # Saying nothing here would read as "seasonality verified".
        warnings.append(f"seasonality untestable for {len(untestable)} profiles at this cohort size: " + "; ".join(untestable))


def _visit_periodicity_warnings(cat: Catalog, conn, warnings: List[str]) -> None:
    """Monthly encounter counts must carry the annual cycle, not just condition onsets.

    This is the check with real statistical power behind it: there are two orders
    of magnitude more visits than onsets of any one disease, so a profile the
    conditions are too rare to demonstrate shows up here beyond doubt. Only whole
    calendar years count - the window opens and closes mid-year, and a partial
    year would put its own months ahead of the rest for reasons that have nothing
    to do with seasonality.
    """
    span = conn.execute("SELECT MIN(observation_period_start_date) lo, MAX(observation_period_end_date) hi FROM observation_period").fetchone()
    if not span or not span["lo"]:
        return
    lo, hi = date.fromisoformat(span["lo"]), date.fromisoformat(span["hi"])
    first = lo.year if (lo.month, lo.day) == (1, 1) else lo.year + 1
    last = hi.year if (hi.month, hi.day) == (12, 31) else hi.year - 1
    if last < first:
        warnings.append("visit periodicity untested: the window holds no whole calendar year")
        return

    by_type: Dict[str, List[int]] = {}
    for r in conn.execute(
        """SELECT c.concept_name name,
                  CAST(strftime('%m', v.visit_start_date) AS INTEGER) m,
                  COUNT(*) n
           FROM visit_occurrence v
           JOIN concept c ON c.concept_id = v.visit_concept_id
           WHERE CAST(strftime('%Y', v.visit_start_date) AS INTEGER) BETWEEN ? AND ?
           GROUP BY 1, 2""",
        (first, last),
    ):
        by_type.setdefault(r["name"], [0] * 12)[r["m"] - 1] = r["n"]

    untestable: List[str] = []
    for key, visit in sorted(cat.visits.items()):
        profile = cat.calendar.visit_seasonality.get(key)
        counts = by_type.get(visit.name)
        if profile is None or counts is None:
            continue
        if max(profile.monthly) - min(profile.monthly) < 1e-9:
            continue  # a deliberately flat profile has nothing to detect
        z, power = _seasonal_z(counts, profile.monthly)
        if power < MIN_SEASONAL_POWER:
            untestable.append(f"{key} ({sum(counts):,} visits, power {power:.1f})")
        elif z < SEASONAL_Z:
            peak = counts.index(max(counts)) + 1
            warnings.append(
                f"{key} visits in {first}-{last} do not track their monthly profile "
                f"(z {z:+.1f} against {power:.1f} available, peak month {peak}, "
                f"n={sum(counts):,})"
            )

    if untestable:
        warnings.append(f"visit periodicity untestable for {len(untestable)} visit types: " + ", ".join(untestable))


# Mean Gregorian month lengths. Events are drawn per day, so a 31-day month
# carries 10% more of them than February for reasons that are not seasonal.
_MONTH_DAYS = (31, 28.2425, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _seasonal_z(counts: List[int], monthly: List[float]) -> Tuple[float, float]:
    """Return (observed z, the z a perfectly seasonal sample of this size would give).

    The statistic is the per-event log-likelihood ratio of the profile against a
    calendar of constant daily intensity, standardised by its spread under that
    flat calendar. It is additive over events, so power grows as sqrt(n) and a
    shallow profile needs a proportionally larger sample - which is exactly what
    the caller has to know before it can read anything into a null result.
    """
    n = sum(counts)
    total_days = sum(_MONTH_DAYS)
    flat = [d / total_days for d in _MONTH_DAYS]
    seasonal_mass = sum(m * d for m, d in zip(monthly, _MONTH_DAYS))
    if n == 0 or seasonal_mass <= 0:
        return 0.0, 0.0
    seasonal = [m * d / seasonal_mass for m, d in zip(monthly, _MONTH_DAYS)]

    weights = [math.log(max(s, 1e-12) / f) for s, f in zip(seasonal, flat)]
    flat_mean = sum(f * w for f, w in zip(flat, weights))
    variance = sum(f * (w - flat_mean) ** 2 for f, w in zip(flat, weights))
    if variance <= 0:
        return 0.0, 0.0

    spread = math.sqrt(variance * n)
    observed = (sum(c * w for c, w in zip(counts, weights)) - n * flat_mean) / spread
    # The best any generator could do: every event drawn from the profile itself,
    # which makes the expected statistic the divergence between the two calendars.
    perfect = sum(s * w for s, w in zip(seasonal, weights)) - flat_mean
    return observed, perfect * n / spread


def _visit_mix_warnings(cat: Catalog, conn, warnings: List[str]) -> None:
    """The telehealth share must follow the anchored curve, not a flat line.

    A tool tested against a constant telehealth share silently fails on any real
    post-2020 extract, so the spike is the single most important calendar effect
    in the file to get right. The comparison is by quarter and the target is the
    curve evaluated on the dates the visits actually fall on: a mid-year point
    estimate is meaningless against a curve that triples and decays inside 2020.
    """
    tele_name = cat.visits["telehealth"].name
    rows = conn.execute(
        """SELECT v.visit_start_date d,
                  SUM(CASE WHEN c.concept_name = ? THEN 1 ELSE 0 END) tele,
                  COUNT(*) n
           FROM visit_occurrence v JOIN concept c ON c.concept_id = v.visit_concept_id
           GROUP BY 1""",
        (tele_name,),
    ).fetchall()

    quarters: Dict[str, List[float]] = {}
    years: Dict[str, List[float]] = {}
    for r in rows:
        when = date.fromisoformat(r["d"])
        mix = cat.calendar.visit_mix(when)
        share = mix.get("telehealth", 0.0) / (sum(mix.values()) or 1.0)
        for label, buckets in ((f"{when.year}Q{(when.month - 1) // 3 + 1}", quarters), (str(when.year), years)):
            bucket = buckets.setdefault(label, [0.0, 0.0, 0.0])
            bucket[0] += r["n"]
            bucket[1] += r["tele"]
            # The target is the curve evaluated on the dates the visits fell on,
            # so it stays exact whatever period the buckets cover.
            bucket[2] += share * r["n"]

    # A quarter of a 1,000-patient extract holds a few hundred visits, which
    # cannot distinguish a 12% share from an 18% one. Drop to annual buckets
    # rather than skipping the check and reporting nothing.
    testable = [b for b in quarters.values() if b[0] >= MIN_PERIOD_VISITS]
    buckets = quarters if len(testable) >= 8 else years

    checked = 0
    for label in sorted(buckets):
        n, tele, expected_rows = buckets[label]
        if n < MIN_PERIOD_VISITS:
            continue
        checked += 1
        observed, target = tele / n, expected_rows / n
        tolerance = max(0.02, target * 0.35)
        if abs(observed - target) > tolerance:
            warnings.append(f"telehealth share in {label}: {observed:.1%} against a target of {target:.1%} ({int(n):,} visits)")
    if not checked:
        warnings.append("telehealth curve untested: no period holds enough visits at this cohort size")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def print_report(cat: Catalog, gt: dict, omop_db: str) -> None:
    demo = gt["demographics"]
    print()
    print("=" * 74)
    print("REALISM REPORT")
    print("=" * 74)

    print(
        f"\nPatients: {demo['n_patients']:,}   Deaths: {demo['n_deaths']:,} "
        f"({100 * demo['n_deaths'] / max(demo['n_patients'], 1):.1f}%)   "
        f"Person-years: {demo['person_years']:,.0f}"
    )

    print(f"\nAge pyramid (age on {demo['reference_date']}):")
    total = max(sum(demo["age_pyramid"].values()), 1)
    for band, n in demo["age_pyramid"].items():
        bar = "#" * int(40 * n / total)
        print(f"  {band:>6s} {n:>7,} {bar}")

    print("\nPrevalence (recorded in the database vs the population target):")
    print(f"  {'condition':<22s} {'recorded':>9s} {'rate':>7s} {'target':>7s} {'capture':>8s}")
    for key, p in sorted(gt["prevalence"].items(), key=lambda kv: -kv[1]["observed_rate"]):
        # Obstetric conditions are assigned by the pregnancy episode generator,
        # not sampled to a marginal prevalence. Printing "0% capture" against a
        # target of zero would read as a failure of a target nobody set.
        if cat.conditions[key].assigned_by:
            print(f"  {key:<22s} {p['n_patients']:>9,} {p['observed_rate']:>7.3f} {'-':>7s} {'episode':>8s}")
            continue
        capture = p["observed_rate"] / p["target_rate"] if p["target_rate"] else 0
        print(f"  {key:<22s} {p['n_patients']:>9,} {p['observed_rate']:>7.3f} {p['target_rate']:>7.3f} {capture:>7.0%}")
    print("  Note: recorded rate sits below the population target by design - a database")
    print("  only captures disease that was coded while the patient was enrolled.")

    print("\nIncidence (new onset after a 1-year disease-free washout):")
    print(f"  {'condition':<22s} {'cases':>8s} {'py at risk':>12s} {'per 1000 py':>12s}")
    top = sorted(gt["incidence"].items(), key=lambda kv: -kv[1]["rate_per_1000_py"])[:12]
    for key, inc in top:
        print(f"  {key:<22s} {inc['n_incident_cases']:>8,} {inc['person_years_at_risk']:>12,.0f} {inc['rate_per_1000_py']:>12.2f}")

    print("\nLab distributions (mean [reference range], % out of range):")
    for key in ["hba1c", "glucose", "creatinine", "egfr", "ldl", "sbp", "hemoglobin", "tsh", "bnp", "psa"]:
        lab = gt["labs"].get(key)
        if not lab:
            continue
        lo, hi = lab["reference_range"]
        print(f"  {lab['name'][:38]:<38s} {lab['mean']:>8.2f} {lab['unit']:<12s} [{lo:g}-{hi:g}]  {lab['pct_above_range']:.0%} high")

    print("\nCohort attrition:")
    for c in gt["cohorts"]:
        print(f"\n  {c['question']}")
        for step in c["attrition"]:
            print(f"      {step['n']:>7,}  {step['step']}")

    art = gt["source_db_artifacts"]
    print("\nSource EMR data-quality artifacts (deliberate):")
    print(
        f"  patients / MRNs   : {art['n_patients']:,} people under {art['n_distinct_mrns']:,} MRNs "
        f"({art['n_patients_at_multiple_facilities']:,} at 2+ sites)"
    )
    print(f"  duplicate DX rows : {art['duplicate_dx_rows']:,}")
    print(f"  pkg code missing  : {art['pct_pkg_code_null']}%")
    print(f"  std lab code missing  : {art['pct_std_lab_code_null']}%")
    print(f"  ordered, no result: {art['lab_rows_ordered_not_resulted']:,}")
    print(f"  code systems      : {art['dx_code_systems']}")
    print(f"  date formats      : {art['facility_date_formats']}")
    birth = art["birth_date_agreement"]
    print(f"  birth dates       : {birth['rows_mismatched']:,} of {birth['rows_checked']:,} disagree with OMOP")

    if gt.get("injected_errors"):
        print("\nInjected error budget (documented in KNOWN_QUIRKS):")
        for name, n in sorted(gt["injected_errors"].items()):
            print(f"  {name:<36s} {n:>8,}")
