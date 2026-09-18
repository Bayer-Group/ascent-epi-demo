#!/usr/bin/env python3
"""Compute provable answers by querying the finished OMOP database."""

import json
import math
import sqlite3
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from catalog import catalog_hash
from models import Catalog

AGE_BANDS = [("0-17", 0, 17), ("18-44", 18, 44), ("45-64", 45, 64), ("65-79", 65, 79), ("80+", 80, 200)]

# Disease-free time a patient must be observed for before a first diagnosis
# counts as incident. Without it, every diagnosis coded in the first weeks of
# enrolment - which is when a new member's whole history gets entered - is
# counted as new-onset disease, and the incidence rate is really a measure of
# how fast the chart was transcribed.
INCIDENCE_WASHOUT_DAYS = 365

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _d(value: str) -> date:
    return date.fromisoformat(value)


def _wilson(k: int, n: int) -> List[float]:
    """Wilson score interval: behaves sensibly for the rare conditions here."""
    if n == 0:
        return [0.0, 0.0]
    z = 1.959963985
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return [round(max(center - margin, 0.0), 5), round(min(center + margin, 1.0), 5)]


def build_ground_truth(
    cat: Catalog,
    omop_db: str,
    source_db: str,
    meta: dict,
    injected_errors: Optional[Dict[str, int]] = None,
) -> dict:
    conn = sqlite3.connect(omop_db)
    conn.row_factory = sqlite3.Row
    try:
        gt = {
            "meta": {**meta, "catalog_hash": catalog_hash()},
            # Exactly how many deliberately wrong rows were planted. A data-quality
            # check that reports zero of these has a bug; one that reports these
            # numbers is correct. Without the counts they are just noise.
            "injected_errors": dict(injected_errors or {}),
            "demographics": _demographics(conn),
            "prevalence": _prevalence(cat, conn),
            "incidence": _incidence(cat, conn),
            "treatment": _treatment(cat, conn),
            "labs": _labs(cat, conn),
            "cohorts": _cohorts(cat, conn),
            "known_quirks": _quirks(cat, conn),
        }
    finally:
        conn.close()

    gt["source_db_artifacts"] = _source_artifacts(source_db, omop_db)
    return gt


# --------------------------------------------------------------------------
# Population
# --------------------------------------------------------------------------


def _reference_date(conn) -> date:
    return _d(conn.execute("SELECT MAX(observation_period_end_date) d FROM observation_period").fetchone()["d"])


def _birth_dates(conn) -> Dict[int, date]:
    return {
        r["person_id"]: date(r["year_of_birth"], r["month_of_birth"] or 7, r["day_of_birth"] or 1)
        for r in conn.execute("SELECT person_id, year_of_birth, month_of_birth, day_of_birth FROM person")
    }


def _age_on(birth: date, when: date) -> int:
    return max(int((when - birth).days // 365.25), 0)


def _band_of(age: int) -> str:
    for band, lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return band
    return AGE_BANDS[-1][0]


def _demographics(conn) -> dict:
    total = conn.execute("SELECT COUNT(*) n FROM person").fetchone()["n"]

    by_sex = {r["gender_source_value"]: r["n"] for r in conn.execute("SELECT gender_source_value, COUNT(*) n FROM person GROUP BY 1")}
    by_race = {r["race_source_value"]: r["n"] for r in conn.execute("SELECT race_source_value, COUNT(*) n FROM person GROUP BY 1")}

    ref = _reference_date(conn)
    births = _birth_dates(conn)
    pyramid = {band: 0 for band, _, _ in AGE_BANDS}
    for birth in births.values():
        pyramid[_band_of(_age_on(birth, ref))] += 1

    person_days = conn.execute(
        "SELECT SUM(julianday(observation_period_end_date) - julianday(observation_period_start_date)) d FROM observation_period"
    ).fetchone()["d"]

    return {
        "n_patients": total,
        "n_deaths": conn.execute("SELECT COUNT(*) n FROM death").fetchone()["n"],
        "by_sex": by_sex,
        "by_race": by_race,
        # Age is computed on the extract date from the full date of birth, not
        # from the year alone: a year-only age is wrong for half the cohort and
        # shifts the whole pyramid by six months.
        "age_pyramid": pyramid,
        "reference_date": ref.isoformat(),
        "reference_year": ref.year,
        "person_years": round((person_days or 0) / 365.25, 1),
    }


# --------------------------------------------------------------------------
# Occurrence
# --------------------------------------------------------------------------


def _condition_codes(cat: Catalog, conn, key: str) -> dict:
    """Every code that identifies this condition, in every vocabulary."""
    name = cat.conditions[key].name
    std = conn.execute(
        "SELECT concept_id, concept_code FROM concept WHERE concept_name = ? AND standard_concept = 'S'",
        (name,),
    ).fetchone()
    if not std:
        return {}

    rows = conn.execute(
        """SELECT c.vocabulary_id, c.concept_code FROM concept_relationship cr
           JOIN concept c ON c.concept_id = cr.concept_id_1
           WHERE cr.concept_id_2 = ? AND cr.relationship_id = 'Maps to'
           ORDER BY c.vocabulary_id, c.concept_code""",
        (std["concept_id"],),
    ).fetchall()

    by_vocab: Dict[str, List[str]] = {}
    for r in rows:
        by_vocab.setdefault(r["vocabulary_id"], []).append(r["concept_code"])

    return {"standard_concept_id": std["concept_id"], "source_codes": by_vocab}


def _patients_with(conn, concept_id: int) -> set:
    return {r["person_id"] for r in conn.execute("SELECT DISTINCT person_id FROM condition_occurrence WHERE condition_concept_id = ?", (concept_id,))}


def _prevalence(cat: Catalog, conn) -> dict:
    total = conn.execute("SELECT COUNT(*) n FROM person").fetchone()["n"]
    ref = _reference_date(conn)
    births = _birth_dates(conn)
    bands = {pid: _band_of(_age_on(birth, ref)) for pid, birth in births.items()}

    denom_by_band = {band: 0 for band, _, _ in AGE_BANDS}
    for band in bands.values():
        denom_by_band[band] += 1

    out = {}
    for key, cond in sorted(cat.conditions.items()):
        codes = _condition_codes(cat, conn, key)
        if not codes:
            continue
        cid = codes["standard_concept_id"]
        patients = _patients_with(conn, cid)
        n = len(patients)

        by_sex = {
            r["gender_source_value"]: r["n"]
            for r in conn.execute(
                """SELECT p.gender_source_value, COUNT(DISTINCT p.person_id) n
                   FROM person p JOIN condition_occurrence co ON co.person_id = p.person_id
                   WHERE co.condition_concept_id = ? GROUP BY 1""",
                (cid,),
            )
        }

        # Counts alone cannot answer "does prevalence rise with age" - the bands
        # hold different numbers of people. The denominator ships with the
        # numerator so the rate is checkable rather than merely plausible.
        hits = {band: 0 for band, _, _ in AGE_BANDS}
        for pid in patients:
            band = bands.get(pid)
            if band:
                hits[band] += 1
        by_age = {
            band: {
                "n": hits[band],
                "denominator": denom_by_band[band],
                "rate": round(hits[band] / denom_by_band[band], 5) if denom_by_band[band] else 0.0,
            }
            for band, _, _ in AGE_BANDS
        }

        out[key] = {
            "name": cond.name,
            "standard_concept_id": cid,
            "codes": codes["source_codes"],
            "n_patients": n,
            "denominator": total,
            "observed_rate": round(n / total, 5) if total else 0,
            "ci95": _wilson(n, total),
            "target_rate": cond.prevalence,
            "by_sex": by_sex,
            "by_age_band": by_age,
            "monotonic_age": cond.monotonic_age,
            "reference": cond.reference,
        }

    return out


def _incidence(cat: Catalog, conn) -> dict:
    """Incidence density: new-onset disease over disease-free person-time.

    The previous version divided the number of patients who ever carried a code
    by the whole file's person-time, which is prevalence wearing an incidence
    label: it counts patients who were already sick on the day they enrolled and
    it charges the denominator for time after diagnosis, when the patient is no
    longer at risk of a first event. Both errors push the rate the same way, so
    the result looked reasonable and was not.
    """
    spells: Dict[int, List[Tuple[date, date]]] = {}
    for r in conn.execute(
        "SELECT person_id, observation_period_start_date s, observation_period_end_date e "
        "FROM observation_period ORDER BY person_id, observation_period_start_date"
    ):
        spells.setdefault(r["person_id"], []).append((_d(r["s"]), _d(r["e"])))

    washout = timedelta(days=INCIDENCE_WASHOUT_DAYS)

    out = {}
    for key, cond in sorted(cat.conditions.items()):
        codes = _condition_codes(cat, conn, key)
        if not codes:
            continue
        cid = codes["standard_concept_id"]

        first = {
            r["person_id"]: _d(r["d"])
            for r in conn.execute(
                "SELECT person_id, MIN(condition_start_date) d FROM condition_occurrence WHERE condition_concept_id = ? GROUP BY person_id",
                (cid,),
            )
        }

        incident = 0
        prevalent = 0
        at_risk_days = 0.0
        for pid, spans in spells.items():
            entry = spans[0][0] + washout
            if entry >= spans[-1][1]:
                continue  # never observed long enough to contribute risk time

            onset = first.get(pid)
            if onset is not None and onset <= entry:
                prevalent += 1
                continue

            censor = onset if onset is not None else spans[-1][1]
            days = sum(max(0, (min(e, censor) - max(s, entry)).days) for s, e in spans)
            if days <= 0:
                continue

            at_risk_days += days
            if onset is not None:
                incident += 1

        person_years = at_risk_days / 365.25
        out[key] = {
            "name": cond.name,
            "n_incident_cases": incident,
            "n_prevalent_excluded": prevalent,
            "person_years_at_risk": round(person_years, 1),
            "rate_per_1000_py": round(1000 * incident / person_years, 3) if person_years else 0,
            "washout_days": INCIDENCE_WASHOUT_DAYS,
        }
    return out


def _treatment(cat: Catalog, conn) -> dict:
    out = {}
    for key, cond in sorted(cat.conditions.items()):
        if not cond.treatment_lines:
            continue
        codes = _condition_codes(cat, conn, key)
        if not codes:
            continue
        cid = codes["standard_concept_id"]
        with_dx = _patients_with(conn, cid)
        if not with_dx:
            continue

        lines = {}
        any_treated: set = set()
        for tl in cond.treatment_lines:
            drug = cat.drugs[tl.drug]
            treated = {
                r["person_id"]
                for r in conn.execute(
                    """SELECT DISTINCT de.person_id FROM drug_exposure de
                       JOIN concept c ON c.concept_id = de.drug_concept_id
                       WHERE c.concept_name = ?""",
                    (drug.name,),
                )
            }
            both = with_dx & treated
            any_treated |= both
            lines[tl.drug] = {
                "drug_name": drug.name,
                "n_treated": len(both),
                "pct_of_diagnosed": round(len(both) / len(with_dx), 4),
                "target_p_treated": tl.p_treated,
                # Exposures in patients who never carried the indication: the
                # injected off-label budget. A cohort defined by drug alone
                # picks these up, which is the point of publishing the number.
                "n_exposed_without_diagnosis": len(treated - with_dx),
            }

        out[key] = {
            "name": cond.name,
            "n_diagnosed": len(with_dx),
            "n_untreated": len(with_dx) - len(any_treated),
            "pct_untreated": round((len(with_dx) - len(any_treated)) / len(with_dx), 4),
            "by_drug": lines,
        }
    return out


def _labs(cat: Catalog, conn) -> dict:
    out = {}
    for key, lab in sorted(cat.labs.items()):
        row = conn.execute(
            """SELECT COUNT(*) n_rows,
                      SUM(CASE WHEN m.value_as_number IS NULL THEN 1 ELSE 0 END) n_null
               FROM measurement m JOIN concept c ON c.concept_id = m.measurement_concept_id
               WHERE c.concept_name = ?""",
            (lab.name,),
        ).fetchone()
        if not row["n_rows"]:
            continue

        # Ordered-but-unresulted tests are NULL, and NULL is not a small number:
        # averaging them in as zero is what turned a missing potassium into a
        # fatal one in the previous build.
        values = [
            r[0]
            for r in conn.execute(
                """SELECT m.value_as_number FROM measurement m
                   JOIN concept c ON c.concept_id = m.measurement_concept_id
                   WHERE c.concept_name = ? AND m.value_as_number IS NOT NULL
                   ORDER BY m.value_as_number""",
                (lab.name,),
            )
        ]
        n = len(values)
        if not n:
            continue

        out[key] = {
            "name": lab.name,
            "unit": lab.unit,
            "n_results": n,
            "n_ordered_no_result": row["n_null"],
            "pct_unresulted": round(row["n_null"] / row["n_rows"], 4),
            "mean": round(sum(values) / n, 3),
            "p05": values[max(int(n * 0.05) - 1, 0)],
            "p50": values[n // 2],
            "p95": values[min(int(n * 0.95), n - 1)],
            "min": values[0],
            "max": values[-1],
            "reference_range": [lab.range_low, lab.range_high],
            "plausible_range": [lab.plausible_min, lab.plausible_max],
            # Must be zero: anything outside the physiologic range is an artifact
            # of the generator, not a critical result.
            "n_outside_plausible": sum(1 for v in values if v < lab.plausible_min or v > lab.plausible_max),
            "pct_above_range": round(sum(1 for v in values if v > lab.range_high) / n, 4),
            "pct_below_range": round(sum(1 for v in values if v < lab.range_low) / n, 4),
        }
    return out


# --------------------------------------------------------------------------
# Cohorts
# --------------------------------------------------------------------------


def _cohorts(cat: Catalog, conn) -> List[dict]:
    """Run each attrition ladder step by step against the finished database."""
    results = []
    for spec in cat.cohorts:
        cohort = {"id": spec["id"], "question": spec["question"], "attrition": []}
        # `index` is fixed by the first qualifying diagnosis and never moves;
        # `anchor` is what a windowed step measures from, and a drug step
        # advances it, so "within 90 days of exposure" means what it says.
        state = {"survivors": None, "index": {}, "anchor": {}}

        for step in spec["attrition"]:
            state = _apply_step(cat, conn, step, state)
            cohort["attrition"].append({"step": step["step"], "n": len(state["survivors"] or ())})

        cohort["expected_n"] = len(state["survivors"]) if state["survivors"] is not None else 0
        results.append(cohort)
    return results


def _restrict(state: dict, keep: set) -> dict:
    survivors = keep if state["survivors"] is None else state["survivors"] & keep
    return {"survivors": survivors, "index": state["index"], "anchor": state["anchor"]}


def _apply_step(cat: Catalog, conn, step: dict, state: dict) -> dict:
    kind = step["type"]
    survivors = state["survivors"]

    if kind == "any_condition_dx":
        cid = _std_concept_id(cat, conn, step["condition"])
        rows = conn.execute(
            "SELECT person_id, MIN(condition_start_date) d FROM condition_occurrence WHERE condition_concept_id = ? GROUP BY person_id",
            (cid,),
        ).fetchall()
        found = {r["person_id"]: r["d"] for r in rows}
        new = _restrict(state, set(found))
        # The first qualifying diagnosis anchors the index date.
        for pid in new["survivors"]:
            if pid not in new["index"]:
                new["index"][pid] = found[pid]
                new["anchor"][pid] = found[pid]
        return new

    if kind == "repeat_condition_dx":
        cid = _std_concept_id(cat, conn, step["condition"])
        rows = conn.execute(
            """SELECT person_id, COUNT(DISTINCT condition_start_date) n,
                      MIN(condition_start_date) first_d, MAX(condition_start_date) last_d
               FROM condition_occurrence WHERE condition_concept_id = ? GROUP BY person_id""",
            (cid,),
        ).fetchall()
        keep = {r["person_id"] for r in rows if r["n"] >= step["min_count"] and _days_between(r["first_d"], r["last_d"]) >= step["min_gap_days"]}
        return _restrict(state, keep)

    if kind == "prior_observation":
        keep = set()
        for pid in survivors or set():
            idx = state["index"].get(pid)
            if not idx:
                continue
            r = conn.execute(
                """SELECT 1 FROM observation_period
                   WHERE person_id = ? AND observation_period_start_date <= date(?, ?)
                   AND observation_period_end_date >= ? LIMIT 1""",
                (pid, idx, f"-{step['days']} day", idx),
            ).fetchone()
            if r:
                keep.add(pid)
        return _restrict(state, keep)

    if kind in ("drug_exposure", "no_drug_exposure"):
        names = _drug_names(cat, step["ingredient"])
        placeholders = ",".join("?" for _ in names)
        exposed = {
            r["person_id"]
            for r in conn.execute(
                f"""SELECT DISTINCT de.person_id FROM drug_exposure de
                    JOIN concept c ON c.concept_id = de.drug_concept_id
                    WHERE c.concept_name IN ({placeholders})""",
                names,
            )
        }

        if kind == "no_drug_exposure":
            # "No record of therapy" is a statement about the whole record, not
            # about the window after index, so this one is deliberately unanchored.
            base = survivors if survivors is not None else _all_persons(conn)
            return {"survivors": base - exposed, "index": state["index"], "anchor": state["anchor"]}

        # Treatment starts at or after the index diagnosis: a prescription that
        # predates the diagnosis is not treatment for it. The qualifying
        # exposure then becomes the anchor, so a following "within N days"
        # window is measured from the exposure the step just selected.
        candidates = exposed if survivors is None else exposed & survivors
        first_after: Dict[int, str] = {}
        for pid in candidates:
            earliest = _first_exposure_after(conn, pid, names, state["index"].get(pid))
            if earliest:
                first_after[pid] = earliest

        new = _restrict(state, set(first_after))
        for pid in new["survivors"]:
            new["anchor"][pid] = first_after[pid]
        return new

    if kind == "lab_threshold":
        lab = cat.labs[step["lab"]]
        op = ">" if step["operator"] == ">" else "<"
        window = step.get("window_days")

        if window is None:
            rows = conn.execute(
                f"""SELECT DISTINCT m.person_id FROM measurement m
                    JOIN concept c ON c.concept_id = m.measurement_concept_id
                    WHERE c.concept_name = ? AND m.value_as_number {op} ?""",
                (lab.name, step["value"]),
            ).fetchall()
            return _restrict(state, {r["person_id"] for r in rows})

        # A windowed threshold is measured from the anchor - the qualifying drug
        # exposure if a drug step has run, otherwise the index diagnosis. The
        # window has to be applied, not just parsed: dropping it turns "HbA1c >
        # 8% within 90 days of exposure" into "at any point in the patient's
        # life".
        keep = set()
        for pid in survivors or set():
            anchor = state["anchor"].get(pid) or state["index"].get(pid)
            if not anchor:
                continue
            r = conn.execute(
                f"""SELECT 1 FROM measurement m
                    JOIN concept c ON c.concept_id = m.measurement_concept_id
                    WHERE m.person_id = ? AND c.concept_name = ? AND m.value_as_number {op} ?
                    AND m.measurement_date BETWEEN ? AND date(?, ?) LIMIT 1""",
                (pid, lab.name, step["value"], anchor, anchor, f"+{window} day"),
            ).fetchone()
            if r:
                keep.add(pid)
        return _restrict(state, keep)

    if kind == "age_range":
        # Age at index, not age today: a 63-year-old diagnosed at 61 does not
        # belong in a cohort defined as "65 or older at index", and dating the
        # age from the extract date quietly admitted them.
        fallback = _reference_date(conn)
        births = _birth_dates(conn)
        lo = step.get("min_age", 0)
        hi = step.get("max_age", 200)
        keep = set()
        for pid, birth in births.items():
            if survivors is not None and pid not in survivors:
                continue
            idx = state["index"].get(pid)
            age = _age_on(birth, _d(idx) if idx else fallback)
            if lo <= age <= hi:
                keep.add(pid)
        return _restrict(state, keep)

    if kind == "sex":
        code = "F" if step["sex"] == "female" else "M"
        rows = conn.execute("SELECT person_id FROM person WHERE gender_source_value = ?", (code,)).fetchall()
        return _restrict(state, {r["person_id"] for r in rows})

    if kind == "no_condition_dx":
        # The exclusion arm of a ladder: everyone who never carried the code.
        # This step type was documented and whitelisted but never implemented,
        # so a cohort that used it would have raised at generation time.
        cid = _std_concept_id(cat, conn, step["condition"])
        excluded = _patients_with(conn, cid)
        base = survivors if survivors is not None else _all_persons(conn)
        return {"survivors": base - excluded, "index": state["index"], "anchor": state["anchor"]}

    if kind == "condition_after_condition":
        cid = _std_concept_id(cat, conn, step["condition"])
        after = _std_concept_id(cat, conn, step["after_condition"])
        rows = conn.execute(
            """SELECT DISTINCT a.person_id FROM condition_occurrence a
               JOIN condition_occurrence b ON b.person_id = a.person_id
               WHERE a.condition_concept_id = ? AND b.condition_concept_id = ?
               AND julianday(a.condition_start_date) BETWEEN julianday(b.condition_start_date)
                   AND julianday(b.condition_start_date) + ?""",
            (cid, after, step["window_days"]),
        ).fetchall()
        return _restrict(state, {r["person_id"] for r in rows})

    raise ValueError(f"Unknown attrition step type: {kind}")


def _all_persons(conn) -> set:
    return {r["person_id"] for r in conn.execute("SELECT person_id FROM person")}


def _first_exposure_after(conn, pid: int, names: List[str], index_date: Optional[str]) -> Optional[str]:
    placeholders = ",".join("?" for _ in names)
    params: List = [pid, *names]
    clause = ""
    if index_date:
        clause = "AND de.drug_exposure_start_date >= ?"
        params.append(index_date)
    row = conn.execute(
        f"""SELECT MIN(de.drug_exposure_start_date) d FROM drug_exposure de
            JOIN concept c ON c.concept_id = de.drug_concept_id
            WHERE de.person_id = ? AND c.concept_name IN ({placeholders}) {clause}""",
        params,
    ).fetchone()
    return row["d"]


def _std_concept_id(cat: Catalog, conn, condition_key: str) -> int:
    row = conn.execute(
        "SELECT concept_id FROM concept WHERE concept_name = ? AND standard_concept = 'S'",
        (cat.conditions[condition_key].name,),
    ).fetchone()
    return row["concept_id"]


def _drug_names(cat: Catalog, ingredient) -> List[str]:
    wanted = [ingredient] if isinstance(ingredient, str) else list(ingredient)
    return [d.name for d in cat.drugs.values() if d.ingredient in wanted]


def _days_between(a: str, b: str) -> int:
    return (_d(b) - _d(a)).days


# --------------------------------------------------------------------------
# Known quirks: real numbers for the shipped manifest
# --------------------------------------------------------------------------


def _quirks(cat: Catalog, conn) -> dict:
    """Quantify the deliberate imperfections so they can be documented, not fixed."""
    out: dict = {}

    # Under-coded obesity: the BMI is in the chart, the diagnosis is not. This
    # is the single most common reason a claims-derived obesity prevalence is
    # half the measured one, and removing it would make the data easier and
    # wrong.
    obese_row = conn.execute(
        """SELECT COUNT(DISTINCT m.person_id) n FROM measurement m
           JOIN concept c ON c.concept_id = m.measurement_concept_id
           WHERE c.concept_name = ? AND m.value_as_number >= 30""",
        (cat.labs["bmi"].name,),
    ).fetchone()
    coded = conn.execute(
        """SELECT COUNT(DISTINCT co.person_id) n FROM condition_occurrence co
           JOIN concept c ON c.concept_id = co.condition_concept_id
           JOIN measurement m ON m.person_id = co.person_id
           JOIN concept mc ON mc.concept_id = m.measurement_concept_id
           WHERE c.concept_name = ? AND mc.concept_name = ? AND m.value_as_number >= 30""",
        (cat.conditions["obesity"].name, cat.labs["bmi"].name),
    ).fetchone()
    n_measured = obese_row["n"]
    out["undercoded_obesity"] = {
        "n_patients_bmi_ge_30": n_measured,
        "n_also_coded": coded["n"],
        "n_uncoded": n_measured - coded["n"],
        "pct_uncoded": round((n_measured - coded["n"]) / n_measured, 4) if n_measured else 0.0,
    }

    # Coverage gaps: patients disenrol and come back. Anything that assumes a
    # single continuous window per patient will silently drop the gap.
    spells: Dict[int, List[Tuple[date, date]]] = {}
    for r in conn.execute(
        "SELECT person_id, observation_period_start_date s, observation_period_end_date e "
        "FROM observation_period ORDER BY person_id, observation_period_start_date"
    ):
        spells.setdefault(r["person_id"], []).append((_d(r["s"]), _d(r["e"])))

    gaps = []
    for spans in spells.values():
        for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
            gaps.append((next_start - prev_end).days)
    gaps.sort()
    out["observation_period_gaps"] = {
        "n_patients": len(spells),
        "n_patients_with_gap": sum(1 for s in spells.values() if len(s) > 1),
        "n_gaps": len(gaps),
        "median_gap_days": gaps[len(gaps) // 2] if gaps else 0,
        "total_gap_days": sum(gaps),
    }

    # Cause-of-death mix, published because it is a known limitation rather than
    # a target: the competing-risk model concentrates deaths in the conditions
    # this catalog carries, so cancer's share is below the population figure.
    out["cause_of_death_mix"] = {
        r["concept_name"]: r["n"]
        for r in conn.execute(
            """SELECT c.concept_name, COUNT(*) n FROM death d
               JOIN concept c ON c.concept_id = d.cause_concept_id
               GROUP BY 1 ORDER BY 2 DESC"""
        )
    }

    unresulted = conn.execute("SELECT COUNT(*) n, SUM(CASE WHEN value_as_number IS NULL THEN 1 ELSE 0 END) nulls FROM measurement").fetchone()
    out["unresulted_measurements"] = {
        "n_measurements": unresulted["n"],
        "n_null_values": unresulted["nulls"] or 0,
        "pct_null": round((unresulted["nulls"] or 0) / unresulted["n"], 4) if unresulted["n"] else 0.0,
    }

    return out


def _source_artifacts(source_db: str, omop_db: str) -> dict:
    """Record the known delta between the clean OMOP data and the messy extract."""
    conn = sqlite3.connect(source_db)
    conn.row_factory = sqlite3.Row
    try:

        def one(sql):
            return conn.execute(sql).fetchone()[0]

        dx_total = one("SELECT COUNT(*) FROM DX")
        dx_distinct = one("SELECT COUNT(*) FROM (SELECT DISTINCT ENC_ID, MRN, DX_CD, DX_DT FROM DX)")

        by_type = {r["DX_CD_TYPE"]: r["n"] for r in conn.execute("SELECT DX_CD_TYPE, COUNT(*) n FROM DX GROUP BY 1")}
        formats = {r["FAC_CD"]: r["DATE_FMT"] for r in conn.execute("SELECT FAC_CD, DATE_FMT FROM FACILITY")}
        specialties = {r["FAC_CD"]: r["SPECIALTY"] for r in conn.execute("SELECT FAC_CD, SPECIALTY FROM FACILITY")}

        # Record linkage: one person, one member id, several site-local MRNs.
        n_members = one("SELECT COUNT(DISTINCT MEMBER_ID) FROM PATIENT_MASTER")
        n_mrns = one("SELECT COUNT(DISTINCT MRN) FROM PATIENT_MASTER")
        multi_site = one("SELECT COUNT(*) FROM (SELECT MEMBER_ID FROM PATIENT_MASTER GROUP BY MEMBER_ID HAVING COUNT(*) > 1)")

        # Lab codes: null standard codes and site-local mnemonics are the two
        # separate reasons a naive code join under-counts, so they are counted
        # separately.
        unmapped_by_test = {
            r["TEST_NM"]: r["n"]
            for r in conn.execute("SELECT TEST_NM, COUNT(*) n FROM LAB_RESULTS WHERE STD_LAB_CD IS NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 10")
        }

        artifacts = {
            "n_patient_master_rows": one("SELECT COUNT(*) FROM PATIENT_MASTER"),
            "n_patients": n_members,
            "n_distinct_mrns": n_mrns,
            "n_patients_at_multiple_facilities": multi_site,
            "duplicate_dx_rows": dx_total - dx_distinct,
            "dx_rows": dx_total,
            "med_rows": one("SELECT COUNT(*) FROM MED_ORDERS"),
            "lab_rows": one("SELECT COUNT(*) FROM LAB_RESULTS"),
            "coverage_rows": one("SELECT COUNT(*) FROM COVERAGE"),
            "social_hx_rows": one("SELECT COUNT(*) FROM SOCIAL_HX"),
            "lab_rows_ordered_not_resulted": one("SELECT COUNT(*) FROM LAB_RESULTS WHERE RESULT_STATUS = 'ORDERED'"),
            "pct_pkg_code_null": round(one("SELECT 100.0*SUM(CASE WHEN PKG_CD IS NULL THEN 1 ELSE 0 END)/COUNT(*) FROM MED_ORDERS"), 2),
            "pct_std_lab_code_null": round(one("SELECT 100.0*SUM(CASE WHEN STD_LAB_CD IS NULL THEN 1 ELSE 0 END)/COUNT(*) FROM LAB_RESULTS"), 2),
            "unmapped_lab_rows_by_test": unmapped_by_test,
            "dx_code_systems": by_type,
            "facility_date_formats": formats,
            "facility_specialties": specialties,
            "note": (
                "Counts here differ from the OMOP database by design: duplicate rows, null codes, "
                "per-facility date formats and the legacy/current code-system cutover are injected deliberately. "
                "A correct non-OMOP pipeline must reconcile to the OMOP cohort counts above."
            ),
        }
        artifacts["birth_date_agreement"] = _birth_date_agreement(conn, omop_db, formats)
        return artifacts
    finally:
        conn.close()


def _birth_date_agreement(conn, omop_db: str, formats: Dict[str, str]) -> dict:
    """Reconcile every source birth date against OMOP, in the site's own format.

    Date-of-birth disagreement between two renderings of the same person used to
    be an unexplained handful of rows. Here it is a closed set: the only rows
    that may disagree are the ones the transposed-digit budget planted, and this
    check states how many there are so the manifest can name the number.
    """
    omop = sqlite3.connect(omop_db)
    omop.row_factory = sqlite3.Row
    try:
        birth_by_person = {
            r["person_id"]: date(r["year_of_birth"], r["month_of_birth"], r["day_of_birth"])
            for r in omop.execute("SELECT person_id, year_of_birth, month_of_birth, day_of_birth FROM person")
        }
        by_mrn = {r["person_source_value"]: r["person_id"] for r in omop.execute("SELECT person_id, person_source_value FROM person")}
        by_member = {
            r["family_source_value"]: r["person_id"] for r in omop.execute("SELECT DISTINCT family_source_value, person_id FROM payer_plan_period")
        }
    finally:
        omop.close()

    checked = mismatched = unresolved = 0
    for r in conn.execute("SELECT MRN, MEMBER_ID, BIRTH_DT, FAC_CD FROM PATIENT_MASTER"):
        pid = by_mrn.get(r["MRN"]) or by_member.get(r["MEMBER_ID"])
        parsed = _parse_source_date(r["BIRTH_DT"], formats.get(r["FAC_CD"], "%Y-%m-%d"))
        if pid is None or parsed is None:
            unresolved += 1
            continue
        checked += 1
        if parsed != birth_by_person.get(pid):
            mismatched += 1

    return {
        "rows_checked": checked,
        "rows_unresolved": unresolved,
        "rows_mismatched": mismatched,
        "expectation": "equals the transposed_date_digits rows that landed on a birth date",
    }


def _parse_source_date(value: Optional[str], fmt: str) -> Optional[date]:
    """Read a date back in whichever dialect the emitting facility writes."""
    if not value:
        return None
    text = value.strip()
    try:
        if fmt == "epoch":
            return date(1970, 1, 1) + timedelta(seconds=int(text))
        if fmt == "%d-%b-%Y":
            day, mon, year = text.split("-")
            return date(int(year), MONTHS.index(mon) + 1, int(day))
        if fmt == "%m/%d/%Y":
            month, day, year = text.split("/")
            return date(int(year), int(month), int(day))
        return _d(text)
    except (ValueError, OverflowError):
        return None


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def write_ground_truth(gt: dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, sort_keys=False)
    return path


def write_known_quirks(cat: Catalog, gt: dict, path: str) -> str:
    """Ship the deliberate imperfections as documentation, with exact counts.

    An undocumented deliberate defect is indistinguishable from a bug. The first
    reviewer who finds one and cannot tell which it is stops trusting everything
    else in the file, so every quirk this generator plants on purpose is named
    here, counted, and paired with the query that finds it.
    """
    meta = gt["meta"]
    art = gt["source_db_artifacts"]
    quirks = gt["known_quirks"]
    budget = cat.calendar.error_budget

    lines: List[str] = []
    add = lines.append

    add("# Known data-quality quirks")
    add("")
    add(
        f"Dataset: {meta['n_patients']:,} patients, seed {meta['seed']}, "
        f"window {meta['observation_window'][0]} to {meta['observation_window'][1]}, "
        f"messiness `{meta['messiness']}`, OMOP CDM {meta['cdm_version']}."
    )
    add(f"Catalog hash `{meta['catalog_hash']}`. Regenerating with the same seed and catalog reproduces every number below exactly.")
    add("")
    add("Everything on this page is deliberate. Nothing here is a bug report, and a pipeline")
    add('that "corrects" the items in section 2 is removing the only parts of the dataset')
    add("that resemble a real extract.")
    add("")

    add("## 1. Injected error budget (genuinely wrong rows)")
    add("")
    add("These rows are wrong. They are planted at the rates in `catalog/calendar.json` so that a")
    add("data-quality check has something to find: a checker that reports zero of these is broken,")
    add("and one that reports these counts is correct.")
    add("")
    add("| Defect | Rows | Configured rate | Where it lands | How to find it |")
    add("|---|---:|---:|---|---|")
    detect = {
        "orphan_visit_reference": (
            "OMOP `condition_occurrence`",
            "`visit_occurrence_id IS NULL`",
        ),
        "records_outside_observation_period": (
            "OMOP `measurement`",
            "measurement_date after every `observation_period` of that person",
        ),
        "off_label_prescribing": (
            "OMOP `drug_exposure`",
            "exposure in a patient with no diagnosis the drug treats",
        ),
        "implausible_but_real_lab": (
            "OMOP `measurement`",
            "value pinned to the analyte's physiologic min or max (still valid, still alarming)",
        ),
        "duplicate_encounter": (
            "OMOP `visit_occurrence`",
            "same person, type and dates under a second `visit_occurrence_id`",
        ),
        "transposed_date_digits": (
            "source EMR date columns",
            "year with its last two digits swapped, e.g. 2019 keyed as 2091",
        ),
    }
    for name, n in sorted(gt.get("injected_errors", {}).items()):
        where, how = detect.get(name, ("-", "-"))
        add(f"| `{name}` | {n:,} | {budget.get(name, 0):g} | {where} | {how} |")
    add("")
    add("Set `--messiness clean` to suppress the source-side defects; the OMOP-side budget is")
    add("part of the event log and is always present.")
    add("")

    add('## 2. Structural quirks - do NOT "fix" these')
    add("")
    add("Each of these is what the corresponding real-world artifact looks like. They are")
    add("reconcilable, not random, and the ground-truth file gives the answer they reconcile to.")
    add("")

    add("### 2.1 Site-local MRNs and record linkage")
    add(
        f"{art['n_patient_master_rows']:,} `PATIENT_MASTER` rows describe "
        f"{art['n_patients']:,} people under {art['n_distinct_mrns']:,} distinct MRNs; "
        f"{art['n_patients_at_multiple_facilities']:,} people are registered at more than one facility."
    )
    add("MRNs are site-local. `MEMBER_ID` is the person-level key and is the intended join path")
    add("(`PATIENT_MASTER.MEMBER_ID` = OMOP `payer_plan_period.family_source_value`).")
    add("")

    add("### 2.2 Unmapped and site-local lab codes")
    add(f"{art['pct_std_lab_code_null']}% of `LAB_RESULTS` rows have a NULL `STD_LAB_CD`, and sites")
    add("running a local dictionary emit their own mnemonics in `LOCAL_CD` (`GLUA1C`, `SGPT`, `K`).")
    add("Vital signs are the worst affected. Rows with the most unmapped results:")
    add("")
    add("| Test | Unmapped rows |")
    add("|---|---:|")
    for test, n in list(art["unmapped_lab_rows_by_test"].items())[:8]:
        add(f"| {test} | {n:,} |")
    add("")
    add("Recover them by name and local code, not by standard code alone.")
    add("")

    add("### 2.3 Five date formats, including epoch seconds")
    add("")
    add("| Facility | Format | Specialty |")
    add("|---|---|---|")
    for fac, fmt in sorted(art["facility_date_formats"].items()):
        add(f"| {fac} | `{fmt}` | {art['facility_specialties'].get(fac, '')} |")
    add("")
    add("`epoch` is seconds since 1970-01-01 and is negative for births before then.")
    add("")

    add("### 2.4 Three labels for two code systems")
    add(f"`DX_CD_TYPE` takes the values {', '.join(f'`{k}` ({v:,})' for k, v in sorted(art['dx_code_systems'].items()))}.")
    add("`DX10` and `D10` are the same system under two labels, so `GROUP BY DX_CD_TYPE` over-counts")
    add(f"the number of systems. Codes cross over at {cat.calendar.cutover_date.isoformat()} with a")
    add(f"{cat.calendar.cutover_tail_days}-day straggler tail at rate {cat.calendar.cutover_tail_rate:g}.")
    add("Units are also inconsistently cased per site (`mg/dL`, `MG/DL`, `mg/dl`).")
    add("")

    add("### 2.5 Under-coded obesity")
    ob = quirks["undercoded_obesity"]
    add(f"{ob['n_patients_bmi_ge_30']:,} patients have a recorded BMI of 30 or above; only")
    add(
        f"{ob['n_also_coded']:,} of them carry an obesity diagnosis. "
        f"{ob['n_uncoded']:,} ({ob['pct_uncoded']:.1%}) are obese by measurement and silent in the"
    )
    add("claim. A claims-only obesity prevalence is supposed to under-count here.")
    add("")

    add("### 2.6 Treatment gaps")
    add("Diagnosed patients are not all treated, by design:")
    add("")
    add("| Condition | Diagnosed | Untreated | % untreated |")
    add("|---|---:|---:|---:|")
    untreated = sorted(gt["treatment"].items(), key=lambda kv: -kv[1]["pct_untreated"])
    for key, t in untreated[:10]:
        add(f"| {t['name']} | {t['n_diagnosed']:,} | {t['n_untreated']:,} | {t['pct_untreated']:.1%} |")
    add("")

    add("### 2.7 Coverage gaps")
    gap = quirks["observation_period_gaps"]
    add(f"{gap['n_patients_with_gap']:,} of {gap['n_patients']:,} patients have more than one")
    add(f"`observation_period`; the {gap['n_gaps']:,} gaps have a median of {gap['median_gap_days']:,} days.")
    add("A patient is invisible during a gap. Do not merge the spells into one continuous window.")
    add("")

    add("### 2.8 Ordered but unresulted tests")
    un = quirks["unresulted_measurements"]
    add(f"{un['n_null_values']:,} of {un['n_measurements']:,} measurements ({un['pct_null']:.1%}) have a")
    add("NULL `value_as_number` and `RESULT_STATUS = 'ORDERED'` in the source extract. They are")
    add("orders that never came back. NULL is the correct value: a potassium of 0 is not a missing")
    add("result, and treating one as the other is how a plausibility check learns the wrong lesson.")
    add("")

    add("## 3. Coverage limitations")
    add("")
    deaths = quirks["cause_of_death_mix"]
    total_deaths = sum(deaths.values()) or 1
    # The malignancies are identified through the hierarchy, not by looking for
    # "cancer" in the concept name: every one of them is named "Malignant
    # neoplasm of ...", so the string match found none of them and reported a
    # cancer share of zero against a file where cancer causes one death in ten.
    neoplasm = {k for k, g in cat.groupers.items() if "neoplas" in g.name.lower()}
    oncology = {cond.name for key, cond in cat.conditions.items() if neoplasm & set(cat.ancestors_of(key))}
    cancer = sum(n for name, n in deaths.items() if name in oncology)
    add(
        f"- **Cause of death is concentrated in this catalog.** {cancer:,} of {total_deaths:,} deaths "
        f"({cancer / total_deaths:.1%}) are attributed to a cancer, against the ~20% seen in national"
    )
    add("  statistics, because the competing-risk model can only assign causes the catalog carries.")
    add("  Treat the cause-of-death mix as an artifact of the catalog, not as an estimate.")
    add("- **No cancer staging, histology, biomarkers or genomics.** Oncology cohorts can be built by")
    add("  diagnosis and drug only.")
    add("- **No clinical notes, imaging, or waveform data.** The `SOCIAL_HX` table carries the")
    add("  structured facts (smoking, alcohol, family history) that would otherwise live in a note.")
    add("- **One geography and one currency.** All costs are USD and all facilities sit in the same")
    add("  synthetic region.")
    add("- **Every code is fictitious.** Vocabularies, code formats and concept names are invented,")
    add("  so nothing here can be cross-walked to a real terminology.")
    add("")

    add("## 4. Invariants the data must satisfy")
    add("")
    add("`validate.py` fails the build if any of these breaks; they are the counterpart to")
    add("section 1. Anything the file gets wrong that is not listed in section 1 or 2 is a bug.")
    add("")
    add("- No condition, drug, measurement, procedure or coverage window dated before the")
    add("  patient's date of birth.")
    add("- No clinical record dated after the patient's death.")
    add("- No visit or diagnosis outside the patient's observation period.")
    add("- No lab value outside its analyte's physiologic range (unresulted tests are NULL).")
    add("- No concept id referenced by a clinical table that is missing from `concept`.")
    add("- Every standard concept has a self-row in `concept_ancestor`, and every `Is a`")
    add("  relationship appears in the closure.")
    add("- No code used before the date its vocabulary was issued, and no record of a disease,")
    add("  drug or procedure before it existed - COVID-19 not before")
    add(f"  {cat.calendar.pandemic_start.isoformat()}, and no drug before its market date.")
    add("- Out-of-period and orphaned rows appear at exactly the counts in section 1 - not zero,")
    add("  and not more.")
    add("")
    add("These are checked too, but they are distributional: a small cohort can miss the target")
    add("shape by chance, so they print a note rather than failing the build.")
    add("")
    add("- Recorded prevalence rises with age for every age-related condition.")
    add("- Seasonal disease arrives in its season.")
    add("- The telehealth share of encounters tracks the anchored curve through 2020.")
    add("- The age distribution matches the reference pyramid weighted by utilisation.")
    add("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path
