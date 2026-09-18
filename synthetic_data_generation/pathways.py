#!/usr/bin/env python3
"""Emit longitudinal clinical events from patient latent state."""

from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from models import Catalog
from obstetrics import PregnancyEpisode, plan_pregnancies
from patients import Population

# Chronic disease is re-coded at follow-up visits, which is what makes
# "at least 2 diagnoses 30 days apart" phenotypes meaningful.
P_RECODE_CHRONIC = 0.6
# Fraction of visits at which a condition's lab panel is drawn.
P_ORDER_PANEL = 0.55
P_ORDER_PROCEDURE = 0.18
# How long an acute episode keeps perturbing lab values.
ACUTE_LAB_WINDOW_DAYS = 30
# Chance that disease whose onset falls outside the coverage window is
# nonetheless surfaced (and coded) during follow-up.
P_ONSET_OBSERVED = 0.75
# Gap tolerated when collapsing exposures/diagnoses into eras.
ERA_GAP_DAYS = 30
# An ordered test that never came back. Real EMR measurement tables are full of
# these; the value must be NULL, never 0, because a zero potassium is not a
# missing result, it is a corpse.
P_UNRESULTED = 0.028
# Once a patient stops a drug they stay stopped, so the refill chain ends here.
P_REFILL_AT_VISIT = 0.45
# How many providers the whole delivery network employs.
N_PROVIDERS = 200
# Mean Gregorian month, used to turn a month's day count into relative exposure.
_MEAN_MONTH_DAYS = 365.2425 / 12


@dataclass
class EventLog:
    """Canonical event stream, rendered once into OMOP and once into the source EMR."""

    persons: List[dict] = field(default_factory=list)
    observation_periods: List[dict] = field(default_factory=list)
    visits: List[dict] = field(default_factory=list)
    conditions: List[dict] = field(default_factory=list)
    drugs: List[dict] = field(default_factory=list)
    measurements: List[dict] = field(default_factory=list)
    procedures: List[dict] = field(default_factory=list)
    observations: List[dict] = field(default_factory=list)
    deaths: List[dict] = field(default_factory=list)
    providers: List[dict] = field(default_factory=list)
    payer_plan_periods: List[dict] = field(default_factory=list)
    fact_relationships: List[dict] = field(default_factory=list)
    # (person_id, facility, mrn) for every site a patient was ever seen at. A
    # patient with rows at two facilities is the record-linkage exercise.
    person_facilities: List[dict] = field(default_factory=list)
    # Defect name -> how many rows carry it, for the known-quirks manifest.
    injected_errors: Dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Calendar tables, precomputed once because they do not vary by person
# --------------------------------------------------------------------------


@dataclass
class _CalendarTables:
    """Month-resolution lookups for volume, censoring, visit mix and seasonality.

    Resolving these per day per person would mean tens of millions of
    interpolations for a large cohort; per month it is a few hundred entries
    shared by everyone, and no reader of the finished data can tell the
    difference because the within-month position is still uniform.
    """

    months: List[Tuple[int, int]]
    index: Dict[Tuple[int, int], int]
    weight: np.ndarray
    mix: List[Dict[str, float]]
    covid: np.ndarray


def _build_calendar_tables(cat: Catalog, start: date, end: date) -> _CalendarTables:
    months: List[Tuple[int, int]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append((year, month))
        month += 1
        if month == 13:
            year, month = year + 1, 1

    cal = cat.calendar
    weights, mixes, covid = [], [], []
    for y, m in months:
        mid = date(y, m, 15)
        mix = cal.visit_mix(mid)
        # The seasonal profile has to move the month's total volume, not only the
        # composition inside it. Applying it to the type draw alone made every
        # type's realised curve its own profile divided by the mix-weighted mean
        # of all of them - and since office and outpatient are three quarters of
        # the mix, that denominator is mostly their own shape, so it cancelled
        # ~80% of their amplitude and left monthly encounter totals with no
        # annual cycle whatsoever. Weighting the month by the blended profile
        # makes each type's expected count proportional to mix x profile again.
        shares = sum(mix.values()) or 1.0
        blended = sum(share * cal.visit_seasonality[k].factor(m) for k, share in mix.items()) / shares
        # A month's weight is a mass, not a daily rate, so the number of days it
        # holds belongs in it: February really does see ~9% fewer encounters than
        # January for no reason other than its length. Dividing by the mean month
        # keeps the mean weight at 1, which is what scales the visit rate.
        exposure = monthrange(y, m)[1] / _MEAN_MONTH_DAYS
        weights.append(cal.volume_factor(mid) * cal.censoring_factor(mid, end) * blended * exposure)
        mixes.append(mix)
        covid.append(cal.pandemic_intensity(mid))

    return _CalendarTables(
        months=months,
        index={ym: i for i, ym in enumerate(months)},
        weight=np.array(weights, dtype=float),
        mix=mixes,
        covid=np.array(covid, dtype=float),
    )


def _month_span(tables: _CalendarTables, p_start: date, p_end: date) -> Tuple[int, int]:
    lo = tables.index.get((p_start.year, p_start.month), 0)
    hi = tables.index.get((p_end.year, p_end.month), len(tables.months) - 1)
    return lo, max(hi, lo)


def _sample_visit_dates(rng, tables, p_start: date, p_end: date, n: int) -> List[date]:
    """Draw n visit dates in proportion to the calendar's monthly weight."""
    lo, hi = _month_span(tables, p_start, p_end)
    w = tables.weight[lo : hi + 1]
    if w.sum() <= 0:
        return []
    picks = rng.choice(hi - lo + 1, size=n, p=w / w.sum())

    out = []
    for offset in picks:
        y, m = tables.months[lo + int(offset)]
        # Uniform over the days the month actually has, clipped to the spell. A
        # flat 1-28 draw would give February the same
        # volume as January and left the 29th to the 31st of every month with no
        # routine encounter in the whole file, which is one query away from
        # obvious. It also broke the exposure correction every seasonality test
        # applies, by an amount the size of the profiles being tested.
        first = max(date(y, m, 1), p_start)
        last = min(date(y, m, monthrange(y, m)[1]), p_end)
        if last < first:
            first, last = p_start, max(p_start, p_end)
        out.append(first + timedelta(days=int(rng.integers(0, (last - first).days + 1))))
    return out


def _window_weight(tables: _CalendarTables, p_start: date, p_end: date) -> float:
    """Mean calendar weight over a spell, used to scale the expected visit count."""
    lo, hi = _month_span(tables, p_start, p_end)
    return float(tables.weight[lo : hi + 1].mean())


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


def _coverage_spells(rng, cat: Catalog, payer: str, start_limit: date, end_limit: date) -> List[Tuple[date, date]]:
    """Disjoint enrolment spells, with lengths and gaps driven by the payer.

    start_limit is clamped to the patient's date of birth by the caller. Passing
    the global window start instead - which is what this used to receive - let a
    coverage window, and therefore any onset re-dated into it, begin before the
    patient existed. That produced 22 conditions diagnosed before birth, and the
    only reason it was not worse is that most patients are older than the window.
    """
    coverage = cat.demographics["coverage"]
    span = (end_limit - start_limit).days
    if span <= 0:
        return [(start_limit, end_limit)]

    mean_years = coverage["mean_spell_years"].get(payer, 3.0)
    p_gap = coverage["p_gap_after_spell"].get(payer, 0.2)
    p_active = coverage.get("p_active_at_extract", {}).get(payer, 0.6)

    # Spells are built backwards from the date coverage last ran out, because for
    # most of the file that date is the extract date itself: the population is
    # sampled from whoever the payer covers now. Building forwards from a start
    # drawn in the first third of the window and stopping at the first lapse left
    # 85% of patients with no coverage after 2019 - an extract labelled 2026-01
    # whose last seven years were nearly empty.
    if rng.random() < p_active:
        anchor = end_limit
    else:
        anchor = start_limit + timedelta(days=int(rng.integers(int(span * 0.2), span + 1)))

    spells: List[Tuple[date, date]] = []
    cursor = anchor

    while cursor > start_limit and len(spells) < 4:
        duration = max(int(rng.exponential(mean_years * 365.25)), 200)
        spell_start = max(cursor - timedelta(days=duration), start_limit)

        # Plan years overwhelmingly start on 1 January, so spell boundaries pile
        # up there. It is a real artifact and a recognisable one: an analyst who
        # sees a January cliff in enrolment should not conclude the data is broken.
        if spell_start > start_limit and rng.random() < coverage["p_january_boundary"]:
            candidate = date(spell_start.year + (1 if spell_start.month > 6 else 0), 1, 1)
            if start_limit <= candidate < cursor:
                spell_start = candidate

        spells.append((spell_start, cursor))
        if spell_start <= start_limit or rng.random() >= p_gap:
            break
        gap = max(int(rng.exponential(coverage["gap_days_mean"])), 20)
        cursor = spell_start - timedelta(days=gap)

    spells.reverse()
    return spells or [(start_limit, end_limit)]


# --------------------------------------------------------------------------
# Labs
# --------------------------------------------------------------------------


def _draw_lab_value(rng, lab, state: str, intercept: float) -> float:
    mean, sd = lab.distribution.get(state, lab.distribution["default"])

    if lab.lognormal:
        # sd is expressed on the natural scale; convert to log-space spread.
        sigma = float(np.sqrt(np.log1p((sd / max(mean, 1e-6)) ** 2)))
        mu = float(np.log(max(mean, 1e-6)) - 0.5 * sigma**2)
        value = float(rng.lognormal(mu + intercept * sigma, sigma))
    else:
        value = float(rng.normal(mean + intercept, sd))

    # Clamp to the analyte's physiologic range, not to zero. The old floor of 0.0
    # turned every left-tail draw into a sentinel: 512 sodium results of 0 mmol/L,
    # 1,300 haemoglobins of 0 g/dL. Those are not low results, they are
    # incompatible with life, and any reference-range or outlier analysis run
    # against them is answering a question about a bug.
    return lab.clamp(round(value, lab.decimals))


def _dominant_state(lab, states: List[str]) -> str:
    """Pick the candidate state whose mean deviates most from the healthy baseline."""
    baseline = lab.distribution["default"][0]
    best, best_dev = "default", -1.0
    for state in states:
        if state not in lab.distribution:
            # A lab with no treated-state variant still reflects the disease:
            # treating CKD does not restore a healthy eGFR.
            state = state.removesuffix("_treated")
            if state not in lab.distribution:
                continue
        mean = lab.distribution[state][0]
        dev = abs(mean - baseline) / max(abs(baseline), 1e-6)
        if dev > best_dev:
            best, best_dev = state, dev
    return best


def _egfr(creatinine: float, age: int, is_female: bool) -> float:
    """CKD-EPI 2021, so eGFR stays consistent with the creatinine actually emitted."""
    kappa = 0.7 if is_female else 0.9
    alpha = -0.241 if is_female else -0.302
    scr = max(creatinine, 0.1)
    value = 142 * min(scr / kappa, 1.0) ** alpha * max(scr / kappa, 1.0) ** -1.200 * (0.9938**age) * (1.012 if is_female else 1.0)
    return round(value, 0)


# --------------------------------------------------------------------------
# Per-person emission context
# --------------------------------------------------------------------------


@dataclass
class _Ctx:
    """Everything the emitters need about one patient, bundled to keep signatures sane."""

    pid: int
    idx: int
    is_female: bool
    birth: date
    home_facility: str
    payer: str
    death_date: Optional[date]
    active: List[Tuple[str, date]]
    treated_since: Dict[str, date] = field(default_factory=dict)
    never_treat: Set[str] = field(default_factory=set)
    # A drug the patient started and permanently stopped.
    discontinued: Set[str] = field(default_factory=set)
    seen_facilities: Dict[str, str] = field(default_factory=dict)
    recorded_observations: Dict[str, List[Tuple[date, str]]] = field(default_factory=dict)


def generate_events(
    cat: Catalog,
    pop: Population,
    death_year: np.ndarray,
    start: date,
    end: date,
    rng,
    progress=None,
    cause_key: Optional[np.ndarray] = None,
) -> EventLog:
    log = EventLog()
    demo = cat.demographics
    tables = _build_calendar_tables(cat, start, end)

    care_sites = demo["care_sites"]
    site_by_facility: Dict[str, List[dict]] = {}
    for cs in care_sites:
        site_by_facility.setdefault(cs["facility"], []).append(cs)
    providers_by_facility = _build_providers(cat, log, rng)

    # Per-patient lab intercepts: repeated measures on one person correlate.
    lab_keys = sorted(cat.labs)
    intercepts = {k: rng.standard_normal(pop.n) * cat.labs[k].patient_intercept_sd for k in lab_keys}

    ids = {"visit": 1, "cond": 1, "drug": 1, "meas": 1, "proc": 1, "obs": 1, "period": 1, "plan": 1}
    given_f = demo["given_names_female"]
    given_m = demo["given_names_male"]
    family = demo["family_names"]

    episodes = plan_pregnancies(cat, pop, death_year, start, end, rng)
    infant_of: Dict[int, Tuple[int, PregnancyEpisode]] = {}
    for mother_idx, eps in episodes.items():
        for e in eps:
            if e.infant_person_id is not None:
                infant_of[e.infant_person_id] = (mother_idx, e)

    for idx in range(pop.n):
        pid = int(pop.person_id[idx])
        is_female = bool(pop.is_female[idx])
        birth = date(int(pop.birth_year[idx]), int(pop.birth_month[idx]), int(pop.birth_day[idx]))

        given = given_f if is_female else given_m
        name = f"{given[int(rng.integers(0, len(given)))]} {family[int(rng.integers(0, len(family)))]}"
        home = str(pop.facility[idx])
        mrn = f"{home}{pid:07d}"
        pop.mrn[idx] = mrn

        dyear = int(death_year[idx])
        death_date = None
        if dyear >= 0:
            death_date = date(dyear, int(rng.integers(1, 13)), int(rng.integers(1, 29)))
            death_date = max(death_date, birth + timedelta(days=1))

        person_end = min(end, death_date) if death_date else end
        # Coverage cannot begin before the patient exists, and a patient born
        # after the window closes has no data at all.
        person_start = max(start, birth)
        if person_end < person_start:
            # Someone who dies on the day their coverage would open keeps a
            # single-day window and no encounters. Padding it by a day instead
            # would move the end past the death date, dating every record in
            # that one day's stay after the patient died.
            person_end = person_start

        spells = _coverage_spells(rng, cat, str(pop.payer[idx]), person_start, person_end)
        if pid in infant_of and spells:
            # A newborn joins a parent's plan on the day of delivery. The
            # enrolment lag that applies to an adult signing up on their own
            # would leave the birth admission itself - the one encounter every
            # infant in this file is guaranteed to have - outside coverage.
            spells[0] = (birth, spells[0][1])

        log.persons.append(
            {
                "person_id": pid,
                "gender_concept_id": int(pop.gender_concept_id[idx]),
                "year_of_birth": birth.year,
                "month_of_birth": birth.month,
                "day_of_birth": birth.day,
                "birth_datetime": birth,
                "race_concept_id": int(pop.race_concept_id[idx]),
                "ethnicity_concept_id": int(pop.ethnicity_concept_id[idx]),
                "person_source_value": mrn,
                "gender_source_value": "F" if is_female else "M",
                "race_source_value": pop.race_source[idx],
                "ethnicity_source_value": pop.ethnicity_source[idx],
                "name": name,
                "facility": home,
                "member_id": pop.member_id[idx],
                "location_id": int(pop.location_id[idx]),
                "payer": str(pop.payer[idx]),
                "care_site_id": site_by_facility[home][0]["care_site_id"],
            }
        )

        for p_start, p_end in spells:
            log.observation_periods.append(
                {
                    "observation_period_id": ids["period"],
                    "person_id": pid,
                    "observation_period_start_date": p_start,
                    "observation_period_end_date": p_end,
                    "period_type_concept_id": demo["type_concepts"]["observation_period"],
                }
            )
            ids["period"] += 1
            _emit_plan_period(cat, log, ids, pop, idx, p_start, p_end)

        if death_date:
            key = str(cause_key[idx]) if cause_key is not None and cause_key[idx] else "natural"
            log.deaths.append(
                {
                    "person_id": pid,
                    "death_date": death_date,
                    "death_type_concept_id": demo["type_concepts"]["death"],
                    "cause_key": key,
                }
            )

        coverage_start, coverage_end = spells[0][0], spells[-1][1]
        active = _condition_timeline(cat, pop, idx, birth, coverage_start, coverage_end, rng)

        ctx = _Ctx(
            pid=pid,
            idx=idx,
            is_female=is_female,
            birth=birth,
            home_facility=home,
            payer=str(pop.payer[idx]),
            death_date=death_date,
            active=active,
        )
        _record_facility(log, ctx, home)

        for p_start, p_end in spells:
            _emit_period(
                cat,
                log,
                ids,
                rng,
                ctx,
                p_start,
                p_end,
                tables,
                site_by_facility,
                providers_by_facility,
                intercepts,
                demo,
            )

        for episode in episodes.get(idx, []):
            _emit_pregnancy(
                cat,
                log,
                ids,
                rng,
                ctx,
                episode,
                spells,
                site_by_facility,
                providers_by_facility,
                intercepts,
                demo,
            )

        if pid in infant_of:
            _emit_birth_encounter(
                cat,
                log,
                ids,
                rng,
                ctx,
                infant_of[pid][1],
                site_by_facility,
                providers_by_facility,
                demo,
                pop,
                infant_of[pid][0],
            )

        if progress:
            progress(1)

    _inject_error_budget(cat, log, ids, rng, end)
    return log


def _build_providers(cat: Catalog, log: EventLog, rng) -> Dict[str, List[dict]]:
    """Staff each facility from its own specialty mix.

    A tertiary cardiac centre whose roster is 40% family practice is an
    inconsistency a reviewer notices immediately, and provider specialty is a
    common cohort filter - so it has to agree with the facility.
    """
    demo = cat.demographics
    mixes = {k: v for k, v in demo["facility_specialty_mix"].items() if not k.startswith("_")}
    site_by_facility: Dict[str, List[dict]] = {}
    for cs in demo["care_sites"]:
        site_by_facility.setdefault(cs["facility"], []).append(cs)

    by_facility: Dict[str, List[dict]] = {}
    provider_id = 1
    for facility in demo["facilities"]:
        code = facility["code"]
        mix = mixes.get(facility["specialty"], {})
        names = list(mix) or demo["provider_specialties"]
        weights = np.array([mix.get(n, 1.0) for n in names], dtype=float)
        weights = weights / weights.sum()

        count = max(int(round(N_PROVIDERS * facility["share"])), 4)
        sites = site_by_facility.get(code, demo["care_sites"])
        for _ in range(count):
            specialty = names[int(rng.choice(len(names), p=weights))]
            site = sites[int(rng.integers(0, len(sites)))]
            record = {
                "provider_id": provider_id,
                "provider_name": f"Provider {provider_id:03d}",
                "specialty": specialty,
                "care_site_id": site["care_site_id"],
                "facility": code,
            }
            log.providers.append(record)
            by_facility.setdefault(code, []).append(record)
            provider_id += 1

    return by_facility


def _emit_plan_period(cat: Catalog, log: EventLog, ids, pop: Population, idx: int, p_start: date, p_end: date) -> None:
    payer_key = str(pop.payer[idx])
    payer = next((p for p in cat.payers if p.key == payer_key), cat.payers[0])
    log.payer_plan_periods.append(
        {
            "payer_plan_period_id": ids["plan"],
            "person_id": int(pop.person_id[idx]),
            "payer_plan_period_start_date": p_start,
            "payer_plan_period_end_date": p_end,
            "payer_concept_id": payer.concept_id,
            "payer_source_value": payer.key,
            "plan_source_value": payer.plan_source_value,
            "family_source_value": pop.member_id[idx],
        }
    )
    ids["plan"] += 1


# --------------------------------------------------------------------------
# Condition timeline
# --------------------------------------------------------------------------


def _condition_timeline(cat, pop, idx, birth, coverage_start, coverage_end, rng) -> List[Tuple[str, date]]:
    """Turn each carried condition's onset age into a dated, plausible onset.

    The order of operations matters and used to be wrong. The COVID guard fired
    first and the "surfaced during follow-up" re-date fired second, so a guarded
    2020 onset was immediately re-drawn uniformly across a coverage window that
    often began in 2010 - which is how 305 COVID records ended up dated before
    the disease existed. Every validity clamp now runs last, after seasonality
    and after re-dating, and a condition whose clamp pushes it outside the
    patient's coverage is dropped rather than silently moved.
    """
    active: List[Tuple[str, date]] = []

    for key in cat.conditions:
        if key not in pop.has or not pop.has[key][idx]:
            continue
        c = cat.conditions[key]

        # Counting forward from the birth date, not back from the extract date:
        # age is a whole number, so counting back put every onset on a grid of
        # anniversaries of the extract date and gave the last month of the file a
        # visible incidence spike.
        onset_age = float(pop.onset_age[key][idx])
        onset_date = birth + timedelta(days=int(onset_age * 365.25))
        onset_date = _seasonal_redraw(cat, c, onset_date, coverage_start, coverage_end, rng)

        if onset_date > coverage_end and rng.random() < P_ONSET_OBSERVED:
            onset_date = _seasonal_date_between(cat, c, coverage_start, coverage_end, rng)

        # --- clamps, applied last -------------------------------------------
        if onset_date < birth:
            onset_date = _seasonal_date_between(cat, c, birth, birth + timedelta(days=400), rng)
        window_start = c.valid_from
        if window_start and onset_date < window_start:
            # Push into the valid era rather than deleting the patient's disease:
            # they really do have it, the record just cannot predate the code.
            onset_date = _seasonal_date_between(cat, c, window_start, max(coverage_end, window_start), rng)
        if c.valid_to and onset_date > c.valid_to:
            onset_date = c.valid_to
        if onset_date > coverage_end or onset_date < birth:
            continue

        active.append((key, onset_date))

    return active


def _seasonal_redraw(cat: Catalog, cond, when: date, coverage_start: date, coverage_end: date, rng) -> date:
    """Move an onset to a seasonal month of its own year, without moving it in or
    out of the patient's coverage.

    A redraw free to cross a coverage boundary funnels unobservable onsets into
    whatever sliver of their year is observable. With an extract ending on
    31 January, every 2026 onset that redrew into January became a recorded
    January diagnosis, and January ran at twice its profile weight in the
    finished file. Redrawing within the region the onset already occupied keeps
    the observable months distributed as the profile says they should be.
    """
    year_start, year_end = date(when.year, 1, 1), date(when.year, 12, 31)
    if when < coverage_start:
        lo, hi = year_start, min(year_end, coverage_start - timedelta(days=1))
    elif when > coverage_end:
        lo, hi = max(year_start, coverage_end + timedelta(days=1)), year_end
    else:
        lo, hi = max(year_start, coverage_start), min(year_end, coverage_end)
    if hi < lo:
        return when
    return _seasonal_date_between(cat, cond, lo, hi, rng)


def _month_start_after(when: date) -> date:
    return date(when.year + (when.month == 12), when.month % 12 + 1, 1)


def _seasonal_date_between(cat: Catalog, cond, lo: date, hi: date, rng) -> date:
    """Draw a date in [lo, hi] that still respects the condition's seasonal profile.

    Every re-date of an onset used to be a uniform draw, which threw away the month
    the profile had just chosen. A large share of onsets get re-dated - anyone whose
    modelled onset falls outside their coverage or before their code existed - so a
    uniform redraw is enough on its own to flatten the curve in the finished file.
    """
    span = (hi - lo).days
    if span <= 0:
        return lo

    profile = cat.calendar.seasonality.get(cond.seasonality)
    if profile is None or cond.seasonality == "none":
        return lo + timedelta(days=int(rng.integers(0, span + 1)))

    monthly = np.array(profile.monthly, dtype=float)
    # One entry per calendar month the window touches, weighted by the profile and
    # by how much of the month the window actually contains. Weighting whole months
    # instead biases a window that ends mid-month towards its final month, which
    # showed up as a spurious November peak on conditions whose onsets cluster at a
    # coverage boundary.
    spans: List[Tuple[date, date]] = []
    weights: List[float] = []
    cursor = date(lo.year, lo.month, 1)
    while cursor <= hi:
        following = _month_start_after(cursor)
        first = max(cursor, lo)
        last = min(following - timedelta(days=1), hi)
        spans.append((first, last))
        weights.append(float(monthly[cursor.month - 1]) * ((last - first).days + 1))
        cursor = following

    total = sum(weights)
    if total <= 0:
        return lo + timedelta(days=int(rng.integers(0, span + 1)))

    first, last = spans[int(rng.choice(len(spans), p=np.array(weights) / total))]
    return first + timedelta(days=int(rng.integers(0, (last - first).days + 1)))


# --------------------------------------------------------------------------
# Visits
# --------------------------------------------------------------------------


def _emit_period(
    cat,
    log,
    ids,
    rng,
    ctx: _Ctx,
    p_start: date,
    p_end: date,
    tables,
    site_by_facility,
    providers_by_facility,
    intercepts,
    demo,
):
    """Emit visits and their attached clinical events across one coverage window."""
    span_days = (p_end - p_start).days
    if span_days <= 0:
        return

    years = span_days / 365.25
    conditions_in_window = [(k, d) for k, d in ctx.active if d <= p_end]

    age_at_start = max(int((p_start - ctx.birth).days / 365.25), 0)
    base_rate = 0.6 + 0.02 * age_at_start
    rate = base_rate + sum(cat.conditions[k].visit_rate for k, _ in conditions_in_window)
    # Utilisation is a function of calendar time as well as of the patient:
    # volume grows year on year, collapses in April 2020 and tails off inside the
    # claims run-out window at the end of the extract.
    rate *= _window_weight(tables, p_start, p_end)

    n_visits = int(rng.poisson(max(rate * years, 0.4)))
    n_visits = max(min(n_visits, 400), 1)

    dates = _sample_visit_dates(rng, tables, p_start, p_end, n_visits)

    # An acute event is itself a reason to present for care, so guarantee an
    # encounter on the day it happens. Otherwise infarcts and sepsis episodes
    # go unrecorded whenever no routine visit lands nearby.
    for key, onset in conditions_in_window:
        cond = cat.conditions[key]
        if not (p_start <= onset <= p_end):
            continue
        if not cond.chronic:
            dates.append(onset)
        elif cond.seasonality != "none" and not any(0 <= (d - onset).days <= 30 for d in dates):
            # A chronic disease with a seasonal profile is diagnosed when the
            # patient presents unwell - a COPD exacerbation in January, not a
            # routine review in June. Recording it at whatever visit came next
            # smeared a seasonal onset uniformly across the following months and
            # left the file with no detectable seasonality at all.
            presented = onset + timedelta(days=int(rng.integers(0, 15)))
            dates.append(min(presented, p_end))

    for visit_no, vdate in enumerate(sorted(dates)):
        newly_active = [(k, d) for k, d in conditions_in_window if d <= vdate]
        acute_now = [k for k, d in newly_active if not cat.conditions[k].chronic and abs((vdate - d).days) < 45]
        visit_key = _choose_visit_type(cat, rng, tables, vdate, acute_now)

        visit = _open_visit(
            cat,
            log,
            ids,
            rng,
            ctx,
            vdate,
            visit_key,
            newly_active,
            site_by_facility,
            providers_by_facility,
            demo,
            p_end,
        )
        _emit_conditions(cat, log, ids, rng, ctx, visit, newly_active, demo["type_concepts"])
        _emit_drugs(cat, log, ids, rng, ctx, visit, p_end, newly_active)
        _emit_labs(cat, log, ids, rng, ctx, visit, newly_active, intercepts, demo["type_concepts"])
        _emit_procedures(cat, log, ids, rng, ctx, visit, newly_active, demo["type_concepts"])
        _emit_observations(cat, log, ids, rng, ctx, visit, newly_active, visit_no, demo["type_concepts"])


def _choose_visit_type(cat: Catalog, rng, tables, vdate: date, acute_now) -> str:
    if acute_now:
        roll = rng.random()
        if roll < 0.45:
            return "inpatient"
        if roll < 0.70:
            return "emergency"
        if roll < 0.80:
            return "icu"
        return "observation_stay"

    # Routine care mix moves with the calendar: telehealth is 0.4% of encounters
    # in 2019 and 32% in May 2020. A flat mix across sixteen years makes the
    # single most legible event in recent health-care delivery invisible.
    slot = tables.index.get((vdate.year, vdate.month), 0)
    mix = tables.mix[slot]
    keys = sorted(mix)
    weights = np.array(
        [mix[k] * cat.calendar.visit_seasonality[k].factor(vdate.month) for k in keys],
        dtype=float,
    )
    total = weights.sum()
    if total <= 0:
        return "office"
    return keys[int(rng.choice(len(keys), p=weights / total))]


def _record_facility(log: EventLog, ctx: _Ctx, facility: str) -> str:
    """Return the patient's MRN at a facility, minting one on first contact."""
    if facility not in ctx.seen_facilities:
        # Every site issues its own identifier, and they do not agree. This is
        # the whole point of a member id: the MRN is site-local, the member id
        # follows the person, and linking on the wrong one is the classic
        # multi-site RWD error.
        mrn = f"{facility}{ctx.pid:07d}"
        ctx.seen_facilities[facility] = mrn
        log.person_facilities.append({"person_id": ctx.pid, "facility": facility, "mrn": mrn})
    return ctx.seen_facilities[facility]


def _choose_facility(cat: Catalog, rng, ctx: _Ctx, visit_key: str, newly_active) -> str:
    """Home facility most of the time, with emergencies and referrals leaking out."""
    network = cat.demographics["network"]
    p_out = network["p_out_of_network_emergency"] if visit_key == "emergency" else network["p_out_of_network"]
    if rng.random() >= p_out:
        return ctx.home_facility

    # A referral goes where the service is: oncology to the cancer centre,
    # ischaemic heart disease to the cardiac centre.
    for key, _ in newly_active:
        for grouper in cat.ancestors_of(key):
            targets = network["referral_targets"].get(grouper)
            if targets:
                names = sorted(targets)
                w = np.array([targets[n] for n in names], dtype=float)
                return names[int(rng.choice(len(names), p=w / w.sum()))]

    facilities = [f for f in cat.demographics["facilities"] if f["code"] != ctx.home_facility]
    if not facilities:
        return ctx.home_facility
    w = np.array([f["share"] for f in facilities], dtype=float)
    return facilities[int(rng.choice(len(facilities), p=w / w.sum()))]["code"]


def _open_visit(
    cat,
    log,
    ids,
    rng,
    ctx: _Ctx,
    vdate: date,
    visit_key: str,
    newly_active,
    site_by_facility,
    providers_by_facility,
    demo,
    p_end: date,
    los_override: Optional[int] = None,
) -> dict:
    vt = cat.visits[visit_key]
    los = los_override if los_override is not None else 0
    if los_override is None and vt.inpatient:
        los = max(int(rng.exponential(vt.los_mean)), 1)
    vend = vdate + timedelta(days=los)
    if ctx.death_date:
        vend = min(vend, ctx.death_date)

    facility = _choose_facility(cat, rng, ctx, visit_key, newly_active)
    mrn = _record_facility(log, ctx, facility)
    sites = site_by_facility.get(facility) or demo["care_sites"]
    site = _match_site(sites, vt.inpatient, rng)
    providers = providers_by_facility.get(facility) or log.providers
    provider = providers[int(rng.integers(0, len(providers)))]

    visit = {
        "visit_occurrence_id": ids["visit"],
        "person_id": ctx.pid,
        "visit_concept_key": visit_key,
        "visit_start_date": vdate,
        "visit_end_date": vend,
        "visit_type_concept_id": demo["type_concepts"]["visit"],
        "provider_id": provider["provider_id"],
        "care_site_id": site["care_site_id"],
        "visit_source_key": visit_key,
        "facility": facility,
        "mrn": mrn,
        "payer": ctx.payer,
    }
    _price_visit(cat, rng, visit)
    log.visits.append(visit)
    ids["visit"] += 1
    return visit


def _match_site(sites: List[dict], inpatient: bool, rng) -> dict:
    """Prefer a care site whose place of service matches the visit type."""
    wanted = "Inpatient Hospital" if inpatient else None
    if wanted:
        matches = [s for s in sites if s["place_of_service"] == wanted]
        if matches:
            return matches[int(rng.integers(0, len(matches)))]
    ambulatory = [s for s in sites if s["place_of_service"] != "Inpatient Hospital"]
    pool = ambulatory if (not inpatient and ambulatory) else sites
    return pool[int(rng.integers(0, len(pool)))]


def _price_visit(cat: Catalog, rng, visit: dict) -> None:
    """Attach charge, allowed and paid amounts driven by the patient's payer."""
    cost = cat.demographics["cost"]
    spec = cost["visit_charge"].get(visit["visit_concept_key"], {"mean": 200.0, "sigma": 0.7})
    charge = float(rng.lognormal(np.log(spec["mean"]), spec["sigma"]))

    payer = next((p for p in cat.payers if p.key == visit["payer"]), cat.payers[0])
    noise = float(rng.normal(1.0, cost["contract_noise_sd"]))
    paid = max(charge * payer.paid_ratio * max(noise, 0.2), 0.0)
    copay = 0.0 if payer.copay_mean <= 0 else float(rng.exponential(payer.copay_mean))

    visit["total_charge"] = round(charge, 2)
    visit["paid_by_payer"] = round(min(paid, charge), 2)
    visit["paid_by_patient"] = round(min(copay, max(charge - paid, 0.0)), 2)


# --------------------------------------------------------------------------
# Clinical events attached to a visit
# --------------------------------------------------------------------------


def _facility_of(cat: Catalog, code: str) -> dict:
    for f in cat.demographics["facilities"]:
        if f["code"] == code:
            return f
    return cat.demographics["facilities"][0]


def _emit_conditions(cat, log, ids, rng, ctx: _Ctx, visit, newly_active, type_concepts):
    facility = _facility_of(cat, visit["facility"])
    style = facility.get("coding_style", {})
    max_codes = max(int(round(style.get("codes_per_encounter", 4.0))), 1)
    p_unspecified = style.get("p_unspecified_code", 0.0)

    vdate = visit["visit_start_date"]
    candidates = []
    for key, onset in newly_active:
        c = cat.conditions[key]
        is_index = abs((vdate - onset).days) < 30

        if c.chronic:
            if not is_index and rng.random() > P_RECODE_CHRONIC:
                continue
        else:
            # Acute events are coded around the episode, not forever after.
            if not is_index and rng.random() > 0.08:
                continue
        candidates.append((key, onset, is_index))

    # How many codes a site puts on an encounter is a property of the site, not
    # of the patient: a tertiary centre bills four diagnoses where an ambulatory
    # clinic bills two. What truncation drops is the chronic padding - never the
    # diagnosis the patient came in with. Cutting the list blind dropped the
    # index code whenever a multimorbid patient's comorbidities happened to sort
    # first, which deleted acute onsets outright and erased the seasonal peak of
    # every acute disease along with them.
    index_codes = [c for c in candidates if c[2]]
    padding = [c for c in candidates if not c[2]]
    candidates = index_codes + padding[: max(max_codes - len(index_codes), 0)]
    if not (style.get("p_primary_first", 0.0) and rng.random() < style["p_primary_first"]):
        # A site that does not maintain the primary-diagnosis position emits
        # them in whatever order the chart happens to hold.
        candidates = [candidates[i] for i in rng.permutation(len(candidates))]

    for rank, (key, onset, is_index) in enumerate(candidates):
        c = cat.conditions[key]
        # Under-specific coding: the first source code in the catalog is the
        # least specific one, and some sites reach for it by habit.
        if rng.random() < p_unspecified:
            sc = c.source_codes[0]
        else:
            sc = c.source_codes[int(rng.integers(0, len(c.source_codes)))]

        _append_condition(cat, log, ids, rng, ctx, visit, key, sc, is_index, rank, type_concepts)


def _coded_as(cat: Catalog, sc, when: date, rng) -> Tuple[str, str]:
    """Which diagnosis code system was in force when the record was written.

    A 2012 diagnosis carrying a code from a classification that was not issued
    until 2015 is an anachronism, and it is the cheapest way for a reviewer to
    prove a file was fabricated. The choice is made once, here, so the OMOP
    rendering and the source extract agree on what was written down - deciding
    it separately in each writer made them disagree for no stated reason.
    """
    cal = cat.calendar
    if not sc.legacy_code:
        return sc.code, "DXCODE"
    if when < cal.cutover_date:
        return sc.legacy_code, "DXCODE9"
    # Sites finish remediation at different times, so legacy codes keep arriving
    # for months after the switch.
    if (when - cal.cutover_date).days <= cal.cutover_tail_days and rng.random() < cal.cutover_tail_rate:
        return sc.legacy_code, "DXCODE9"
    return sc.code, "DXCODE"


def _append_condition(cat, log, ids, rng, ctx: _Ctx, visit, key, sc, is_index, rank, type_concepts):
    c = cat.conditions[key]
    emitted_code, emitted_vocabulary = _coded_as(cat, sc, visit["visit_start_date"], rng)
    log.conditions.append(
        {
            "condition_occurrence_id": ids["cond"],
            "person_id": ctx.pid,
            "visit_occurrence_id": visit["visit_occurrence_id"],
            "condition_key": key,
            "source_code": sc.code,
            "legacy_code": sc.legacy_code,
            "emitted_code": emitted_code,
            "emitted_vocabulary": emitted_vocabulary,
            "condition_start_date": visit["visit_start_date"],
            "condition_end_date": visit["visit_end_date"] if not c.chronic else None,
            "condition_type_concept_id": type_concepts["condition"],
            "is_index": is_index,
            "condition_rank": rank + 1,
            "facility": visit["facility"],
        }
    )
    ids["cond"] += 1


def _emit_drugs(cat, log, ids, rng, ctx: _Ctx, visit, p_end, newly_active):
    vdate = visit["visit_start_date"]
    for key, onset in newly_active:
        c = cat.conditions[key]
        for tl in c.treatment_lines:
            state_key = f"{key}::{tl.drug}"
            drug = cat.drugs[tl.drug]

            if state_key in ctx.never_treat or state_key in ctx.discontinued:
                continue
            # A prescription for a drug that was not on the market yet is the
            # cheapest possible proof the data is fabricated.
            if drug.valid_from and vdate < drug.valid_from:
                continue
            if drug.valid_to and vdate > drug.valid_to:
                continue

            if state_key in ctx.treated_since:
                if rng.random() < tl.p_discontinue:
                    # Discontinuation is permanent. Re-rolling it at every refill
                    # produced chains that paused and silently resumed, so no
                    # persistence or gap analysis written against this data could
                    # be right; a stopped drug now stays stopped.
                    ctx.discontinued.add(state_key)
                    continue
                if rng.random() > P_REFILL_AT_VISIT:
                    continue
            else:
                # Whether this patient is ever treated is decided once. Re-rolling
                # at every visit would eventually treat everybody, erasing the
                # untreated group that treatment-gap analyses depend on.
                if rng.random() > tl.p_treated:
                    ctx.never_treat.add(state_key)
                    continue
                # Only start therapy at or after the diagnosis.
                if vdate < onset:
                    continue
                ctx.treated_since[state_key] = vdate

            _append_drug(cat, log, ids, rng, ctx.pid, visit, tl.drug, p_end, key)


def _append_drug(cat, log, ids, rng, pid: int, visit, drug_key, p_end, for_condition):
    drug = cat.drugs[drug_key]
    vdate = visit["visit_start_date"]
    days_supply = int(drug.days_supply[int(rng.integers(0, len(drug.days_supply)))])
    end_date = min(vdate + timedelta(days=days_supply), p_end)

    cost = cat.demographics["cost"]["drug_charge"]
    charge = float(rng.lognormal(np.log(cost["mean"]), cost["sigma"]))

    log.drugs.append(
        {
            "drug_exposure_id": ids["drug"],
            "person_id": pid,
            "visit_occurrence_id": visit["visit_occurrence_id"],
            "drug_key": drug_key,
            "pkg_code": drug.pkg_code,
            "drug_exposure_start_date": vdate,
            "drug_exposure_end_date": end_date,
            "days_supply": days_supply,
            "quantity": float(days_supply),
            "refills": int(rng.integers(0, 4)),
            "sig": f"Take as directed ({drug.strength} {drug.dose_form})",
            "route": drug.route,
            "for_condition": for_condition,
            "facility": visit["facility"],
            "total_charge": round(charge, 2),
        }
    )
    ids["drug"] += 1


def _emit_labs(cat, log, ids, rng, ctx: _Ctx, visit, newly_active, intercepts, type_concepts, extra: Optional[List[str]] = None):
    vdate = visit["visit_start_date"]

    # Which labs get ordered depends on which condition prompted the visit...
    ordered = set(extra or ())
    for key, _ in newly_active:
        c = cat.conditions[key]
        if c.lab_panel and rng.random() <= P_ORDER_PANEL:
            ordered.update(c.lab_panel)

    for vital in ("sbp", "dbp"):
        if rng.random() < 0.75:
            ordered.add(vital)
    if rng.random() < 0.35:
        ordered.add("bmi")
    if not newly_active and rng.random() < 0.12:
        ordered.update(("glucose", "hemoglobin", "creatinine"))

    # ...but the VALUE reflects the patient's whole clinical state, not whoever
    # happened to order it. Otherwise a mild driver (prediabetes) masks a severe
    # one (uncontrolled diabetes) whenever it wins the ordering roll.
    patient_states = []
    for key, onset in newly_active:
        # Acute illness perturbs labs around the episode only: troponin is high
        # during the infarct, not five years later.
        if not cat.conditions[key].chronic and (vdate - onset).days > ACUTE_LAB_WINDOW_DAYS:
            continue
        treated = any(k.startswith(f"{key}::") for k in ctx.treated_since)
        patient_states.append(f"{key}_treated" if treated else key)

    # Sorted, because set iteration order is not stable across runs and would
    # break reproducibility from a fixed seed.
    panel = {lab_key: _dominant_state(cat.labs[lab_key], patient_states) for lab_key in sorted(ordered)}

    age_at = max(int((vdate - ctx.birth).days / 365.25), 0)
    creatinine_value: Optional[float] = None

    # eGFR is derived from creatinine, so creatinine must be drawn first.
    if "egfr" in panel and "creatinine" not in panel:
        panel["creatinine"] = _dominant_state(cat.labs["creatinine"], patient_states)
    order = sorted(panel, key=lambda k: (k == "egfr", k))

    charge_spec = cat.demographics["cost"]["measurement_charge"]
    for lab_key in order:
        state = panel[lab_key]
        lab = cat.labs[lab_key]
        value: Optional[float] = _draw_lab_value(rng, lab, state, float(intercepts[lab_key][ctx.idx]))

        if lab_key == "creatinine":
            creatinine_value = value
        if lab_key == "egfr" and creatinine_value is not None:
            # CKD-EPI runs past 150 for a young patient with a low creatinine,
            # and a real analyser reports that as its ceiling rather than a
            # number. The derived value was skipping the clamp every drawn value
            # goes through, so it was the one lab that could leave its own
            # physiologic range.
            value = lab.clamp(_egfr(creatinine_value, age_at, ctx.is_female))

        # The order exists; the result never came back. NULL, not zero.
        if rng.random() < P_UNRESULTED:
            value = None

        log.measurements.append(
            {
                "measurement_id": ids["meas"],
                "person_id": ctx.pid,
                "visit_occurrence_id": visit["visit_occurrence_id"],
                "lab_key": lab_key,
                "source_code": lab.source_code,
                "measurement_date": vdate,
                "value_as_number": value,
                "unit": lab.unit,
                "range_low": lab.range_low,
                "range_high": lab.range_high,
                "measurement_type_concept_id": type_concepts["measurement"],
                "facility": visit["facility"],
                "total_charge": round(float(rng.lognormal(np.log(charge_spec["mean"]), charge_spec["sigma"])), 2),
            }
        )
        ids["meas"] += 1


def _emit_procedures(cat, log, ids, rng, ctx: _Ctx, visit, newly_active, type_concepts):
    vdate = visit["visit_start_date"]
    age_at = max(int((vdate - ctx.birth).days / 365.25), 0)

    for key, onset in newly_active:
        c = cat.conditions[key]
        for proc_key in c.procedures:
            if not _procedure_allowed(cat, proc_key, ctx.is_female, age_at, vdate):
                continue
            is_index = abs((vdate - onset).days) < 30
            # p_index scales BOTH paths. Gating only the index visit leaves the
            # recurring 18% per follow-up visit intact, and over a decade of
            # visits that alone reaches ~86% -- which is how every obese patient
            # ended up with a gastric bypass.
            p_proc = cat.procedures[proc_key].p_index
            threshold = p_proc if is_index else P_ORDER_PROCEDURE * p_proc
            if rng.random() > threshold:
                continue
            _append_procedure(cat, log, ids, rng, ctx, visit, proc_key, type_concepts)


def _procedure_allowed(cat: Catalog, proc_key: str, is_female: bool, age_at: int, vdate: date) -> bool:
    p = cat.procedures[proc_key]
    if p.sex == "female" and not is_female:
        return False
    if p.sex == "male" and is_female:
        return False
    if age_at < p.age_min or age_at > p.age_max:
        return False
    if p.valid_from and vdate < p.valid_from:
        return False
    if p.valid_to and vdate > p.valid_to:
        return False
    return True


def _append_procedure(cat, log, ids, rng, ctx: _Ctx, visit, proc_key, type_concepts):
    p = cat.procedures[proc_key]
    charge_spec = cat.demographics["cost"]["procedure_charge"]
    log.procedures.append(
        {
            "procedure_occurrence_id": ids["proc"],
            "person_id": ctx.pid,
            "visit_occurrence_id": visit["visit_occurrence_id"],
            "procedure_key": proc_key,
            "source_code": p.source_code,
            "procedure_date": visit["visit_start_date"],
            "procedure_type_concept_id": type_concepts["procedure"],
            "facility": visit["facility"],
            "total_charge": round(float(rng.lognormal(np.log(charge_spec["mean"]), charge_spec["sigma"])), 2),
        }
    )
    ids["proc"] += 1


# --------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------

_OBS_AGE_BANDS = [(0, 17, "0-17"), (18, 44, "18-44"), (45, 64, "45-64"), (65, 79, "65-79"), (80, 200, "80+")]


def _obs_band(age: int) -> str:
    for lo, hi, name in _OBS_AGE_BANDS:
        if lo <= age <= hi:
            return name
    return "80+"


def _emit_observations(cat, log, ids, rng, ctx: _Ctx, visit, newly_active, visit_no, type_concepts):
    """Record the non-measurement facts a chart actually carries.

    The observation table was empty. That is not a cosmetic gap: smoking status
    is a covariate in almost every cardiovascular or oncology study, and a
    database that cannot supply it forces every such analysis to proxy it from
    diagnosis codes, which is exactly the shortcut synthetic data should let you
    avoid rather than force.
    """
    vdate = visit["visit_start_date"]
    age_at = max(int((vdate - ctx.birth).days / 365.25), 0)
    present = {k for k, _ in newly_active}

    for item in cat.observations:
        if age_at < item.age_min or age_at > item.age_max:
            continue
        history = ctx.recorded_observations.setdefault(item.key, [])

        if item.cadence == "once" and history:
            continue
        if item.cadence == "annual" and history and (vdate - history[-1][0]).days < 300:
            continue
        if item.cadence == "on_change" and history and (vdate - history[-1][0]).days < 400:
            continue
        if rng.random() >= item.p_recorded:
            continue

        if item.value_type == "condition_concept":
            _emit_family_history(cat, log, ids, rng, ctx, visit, item, present, type_concepts)
            history.append((vdate, "family_history"))
            continue

        shares = _observation_shares(item, present, age_at)
        keys = sorted(shares)
        w = np.array([shares[k] for k in keys], dtype=float)
        if w.sum() <= 0:
            continue
        chosen = keys[int(rng.choice(len(keys), p=w / w.sum()))]
        value = next(v for v in item.values if v.key == chosen)

        log.observations.append(
            {
                "observation_id": ids["obs"],
                "person_id": ctx.pid,
                "visit_occurrence_id": visit["visit_occurrence_id"],
                "observation_concept_id": item.concept_id,
                "observation_date": vdate,
                "value_as_concept_id": value.concept_id,
                "value_as_string": value.name,
                "value_as_number": None,
                "observation_source_value": item.source_code,
                "observation_type_concept_id": type_concepts["observation"],
                "facility": visit["facility"],
            }
        )
        ids["obs"] += 1
        history.append((vdate, chosen))


def _observation_shares(item, present: Set[str], age: int) -> Dict[str, float]:
    """Marginal shares, overridden by any condition or age band that applies."""
    shares = {v.key: v.share for v in item.values}
    band = _obs_band(age)
    if band in item.age_bands:
        shares = dict(item.age_bands[band])
    # A condition-specific override wins: a patient coded for smoking should not
    # be recorded as a never-smoker, which is exactly what marginal shares do.
    for key, override in item.conditioned_on.items():
        if key in present:
            shares = dict(override)
            break
    return shares


def _emit_family_history(cat, log, ids, rng, ctx: _Ctx, visit, item, present, type_concepts):
    for spec in item.conditions:
        key = spec["condition"]
        p = spec["p_present_if_affected"] if key in present else spec["p_present"]
        if rng.random() >= p:
            continue
        log.observations.append(
            {
                "observation_id": ids["obs"],
                "person_id": ctx.pid,
                "visit_occurrence_id": visit["visit_occurrence_id"],
                "observation_concept_id": item.concept_id,
                "observation_date": visit["visit_start_date"],
                "value_as_concept_key": key,
                "value_as_concept_id": None,
                "value_as_string": f"Family history of {cat.conditions[key].name}",
                "value_as_number": None,
                "observation_source_value": item.source_code,
                "observation_type_concept_id": type_concepts["observation"],
                "facility": visit["facility"],
            }
        )
        ids["obs"] += 1


# --------------------------------------------------------------------------
# Pregnancy
# --------------------------------------------------------------------------


def _emit_pregnancy(
    cat,
    log,
    ids,
    rng,
    ctx: _Ctx,
    episode: PregnancyEpisode,
    spells,
    site_by_facility,
    providers_by_facility,
    intercepts,
    demo,
):
    """Render one planned obstetric episode as visits, codes, labs and procedures."""
    type_concepts = demo["type_concepts"]
    model = cat.pregnancy
    covered = [(s, e) for s, e in spells]

    def in_coverage(when: date) -> bool:
        return any(s <= when <= e for s, e in covered)

    complications_by_date = sorted(episode.complications, key=lambda t: t[1])
    screening_by_date: Dict[date, List[str]] = {}
    for lab_key, when in episode.screening:
        screening_by_date.setdefault(when, []).append(lab_key)
    procs_by_date: Dict[date, List[str]] = {}
    for proc_key, when in episode.procedure_dates:
        procs_by_date.setdefault(when, []).append(proc_key)

    prenatal_dates = sorted(set(episode.prenatal_dates) | set(screening_by_date) | set(procs_by_date))
    for when in prenatal_dates:
        if not in_coverage(when) or (ctx.death_date and when > ctx.death_date):
            continue
        active_now = [(k, d) for k, d, _ in complications_by_date if d <= when]
        visit = _open_visit(
            cat,
            log,
            ids,
            rng,
            ctx,
            when,
            model.prenatal_visit_type,
            active_now,
            site_by_facility,
            providers_by_facility,
            demo,
            when,
        )
        for key, onset in active_now:
            if abs((when - onset).days) < 45 or rng.random() < P_RECODE_CHRONIC:
                c = cat.conditions[key]
                _append_condition(
                    cat,
                    log,
                    ids,
                    rng,
                    ctx,
                    visit,
                    key,
                    c.source_codes[0],
                    abs((when - onset).days) < 30,
                    0,
                    type_concepts,
                )
        extra = screening_by_date.get(when, [])
        for spec_key, _onset, spec in episode.complications:
            if _onset <= when:
                extra.extend(spec.get("labs", []))
        _emit_labs(cat, log, ids, rng, ctx, visit, active_now, intercepts, type_concepts, extra=extra)

        age_at = max(int((when - ctx.birth).days / 365.25), 0)
        for proc_key in procs_by_date.get(when, []):
            if _procedure_allowed(cat, proc_key, ctx.is_female, age_at, when):
                _append_procedure(cat, log, ids, rng, ctx, visit, proc_key, type_concepts)

    # Complication treatment, e.g. insulin for gestational diabetes.
    for key, onset, spec in episode.complications:
        treatment = spec.get("treatment")
        if not treatment or not in_coverage(onset) or rng.random() >= treatment["p_treated"]:
            continue
        visit = _open_visit(
            cat,
            log,
            ids,
            rng,
            ctx,
            onset,
            "office",
            [(key, onset)],
            site_by_facility,
            providers_by_facility,
            demo,
            onset,
        )
        _append_drug(cat, log, ids, rng, ctx.pid, visit, treatment["drug"], episode.end_date, key)

    if not episode.delivered or not in_coverage(episode.end_date):
        return
    if ctx.death_date and episode.end_date > ctx.death_date:
        return

    delivery = _open_visit(
        cat,
        log,
        ids,
        rng,
        ctx,
        episode.end_date,
        model.delivery["visit_type"],
        [(k, d) for k, d, _ in episode.complications],
        site_by_facility,
        providers_by_facility,
        demo,
        episode.end_date,
        los_override=episode.delivery_los,
    )
    delivery_condition = cat.conditions[model.delivery["condition"]]
    _append_condition(
        cat,
        log,
        ids,
        rng,
        ctx,
        delivery,
        model.delivery["condition"],
        delivery_condition.source_codes[0],
        True,
        0,
        type_concepts,
    )
    for key, onset, _spec in episode.complications:
        _append_condition(
            cat,
            log,
            ids,
            rng,
            ctx,
            delivery,
            key,
            cat.conditions[key].source_codes[0],
            abs((episode.end_date - onset).days) < 30,
            1,
            type_concepts,
        )

    age_at = max(int((episode.end_date - ctx.birth).days / 365.25), 0)
    if episode.delivery_procedure and _procedure_allowed(cat, episode.delivery_procedure, ctx.is_female, age_at, episode.end_date):
        _append_procedure(cat, log, ids, rng, ctx, delivery, episode.delivery_procedure, type_concepts)

    # Gestational age is the observation obstetric research is actually built on.
    log.observations.append(
        {
            "observation_id": ids["obs"],
            "person_id": ctx.pid,
            "visit_occurrence_id": delivery["visit_occurrence_id"],
            "observation_concept_id": model.delivery["gestational_age_observation"],
            "observation_date": episode.end_date,
            "value_as_concept_id": None,
            "value_as_string": None,
            "value_as_number": round(episode.gestation_weeks, 1),
            "observation_source_value": "OBS-GA",
            "observation_type_concept_id": type_concepts["observation"],
            "facility": delivery["facility"],
        }
    )
    ids["obs"] += 1

    for when in episode.postpartum_dates:
        if not in_coverage(when) or (ctx.death_date and when > ctx.death_date):
            continue
        visit = _open_visit(
            cat,
            log,
            ids,
            rng,
            ctx,
            when,
            model.postpartum["visit_type"],
            [],
            site_by_facility,
            providers_by_facility,
            demo,
            when,
        )
        _emit_labs(cat, log, ids, rng, ctx, visit, [], intercepts, type_concepts)


def _emit_birth_encounter(
    cat,
    log,
    ids,
    rng,
    ctx: _Ctx,
    episode: PregnancyEpisode,
    site_by_facility,
    providers_by_facility,
    demo,
    pop: Population,
    mother_idx: int,
):
    """The infant's own admission, plus the mother-child link OMOP expects."""
    type_concepts = demo["type_concepts"]
    spec = cat.pregnancy.newborn["birth_encounter"]
    nicu = rng.random() < spec["p_nicu"]
    los = max(int(rng.normal(spec["nicu_los_mean"] if nicu else spec["los_mean"], spec["los_sd"])), 1)

    visit = _open_visit(
        cat,
        log,
        ids,
        rng,
        ctx,
        ctx.birth,
        "icu" if nicu else spec["visit_type"],
        [],
        site_by_facility,
        providers_by_facility,
        demo,
        ctx.birth,
        los_override=los,
    )
    newborn = cat.conditions[spec["condition"]]
    _append_condition(cat, log, ids, rng, ctx, visit, spec["condition"], newborn.source_codes[0], True, 0, type_concepts)

    link = cat.pregnancy.newborn["fact_relationship"]
    mother_id = int(pop.person_id[mother_idx])
    log.fact_relationships.append(
        {
            "domain_concept_id_1": link["domain_concept_id"],
            "fact_id_1": mother_id,
            "domain_concept_id_2": link["domain_concept_id"],
            "fact_id_2": ctx.pid,
            "relationship_concept_id": link["relationship_concept_id_parent_to_child"],
        }
    )
    log.fact_relationships.append(
        {
            "domain_concept_id_1": link["domain_concept_id"],
            "fact_id_1": ctx.pid,
            "domain_concept_id_2": link["domain_concept_id"],
            "fact_id_2": mother_id,
            "relationship_concept_id": link["relationship_concept_id_child_to_parent"],
        }
    )


# --------------------------------------------------------------------------
# Error budget
# --------------------------------------------------------------------------


def _inject_error_budget(cat: Catalog, log: EventLog, ids: Dict[str, int], rng, end: date) -> None:
    """Introduce a known, counted quantity of genuinely wrong records.

    A file with zero referential-integrity defects teaches a pipeline that
    defects do not happen, and every real extract has them. The rates come from
    calendar.json and the realised counts are written to the log so the shipped
    manifest states exactly how many of each were planted - a defect nobody can
    count is indistinguishable from a bug.

    Every defect here is planted so that it is the ONLY rule it breaks. An
    out-of-period record that also lands after the patient's death would make
    the two checks indistinguishable, and a validation suite that cannot tell
    which defect it caught cannot tell a planted one from a real one.
    """
    budget = cat.calendar.error_budget
    counts: Dict[str, int] = {}

    rate = budget.get("orphan_visit_reference", 0.0)
    if rate > 0 and log.conditions:
        picks = rng.random(len(log.conditions)) < rate
        n = 0
        for row, hit in zip(log.conditions, picks):
            if hit:
                row["visit_occurrence_id"] = None
                n += 1
        counts["orphan_visit_reference"] = n

    rate = budget.get("records_outside_observation_period", 0.0)
    if rate > 0 and log.measurements:
        # Move the result past the end of the patient's last coverage spell, so
        # the defect is deterministic: a blind +N days often lands back inside
        # a later spell, and then the planted count and the detectable count
        # disagree for no stated reason.
        last_cover: Dict[int, date] = {}
        for op in log.observation_periods:
            pid = op["person_id"]
            cover_end = op["observation_period_end_date"]
            if cover_end > last_cover.get(pid, date.min):
                last_cover[pid] = cover_end
        dead = {d["person_id"] for d in log.deaths}

        n = 0
        for row in log.measurements:
            pid = row["person_id"]
            cover_end = last_cover.get(pid)
            if pid in dead or cover_end is None or rng.random() >= rate:
                continue
            shifted = cover_end + timedelta(days=int(rng.integers(15, 200)))
            if shifted > end:
                continue
            row["measurement_date"] = shifted
            n += 1
        counts["records_outside_observation_period"] = n

    rate = budget.get("off_label_prescribing", 0.0)
    if rate > 0 and log.drugs and log.visits:
        counts["off_label_prescribing"] = _inject_off_label(cat, log, ids, rng, rate)

    rate = budget.get("implausible_but_real_lab", 0.0)
    if rate > 0 and log.measurements:
        n = 0
        for row in log.measurements:
            if row["value_as_number"] is None or rng.random() >= rate:
                continue
            lab = cat.labs[row["lab_key"]]
            # Still inside the physiologic range, but at its very edge: a real
            # critical value, not a sentinel. This is what a plausibility check
            # should tolerate, and it is the control for the zeros that were
            # removed.
            row["value_as_number"] = round(lab.plausible_max if rng.random() < 0.5 else lab.plausible_min, lab.decimals)
            n += 1
        counts["implausible_but_real_lab"] = n

    rate = budget.get("duplicate_encounter", 0.0)
    if rate > 0 and log.visits:
        extra = []
        max_id = max(v["visit_occurrence_id"] for v in log.visits)
        for row in log.visits:
            if rng.random() >= rate:
                continue
            max_id += 1
            clone = dict(row)
            clone["visit_occurrence_id"] = max_id
            clone["is_duplicate"] = True
            extra.append(clone)
        log.visits.extend(extra)
        counts["duplicate_encounter"] = len(extra)

    log.injected_errors = counts


def _inject_off_label(cat: Catalog, log: EventLog, ids: Dict[str, int], rng, rate: float) -> int:
    """Prescribe drugs to patients carrying none of the matching indications.

    Every exposure in this dataset otherwise follows a diagnosis, which teaches
    a phenotyping pipeline that a drug is a reliable proxy for its indication.
    Real prescribing is not that tidy, and a cohort defined by drug alone should
    pick up a small, known number of patients who never had the disease.
    """
    indications: Dict[str, Set[str]] = {}
    for key, cond in cat.conditions.items():
        for tl in cond.treatment_lines:
            indications.setdefault(tl.drug, set()).add(key)
    candidates = sorted(indications)
    if not candidates:
        return 0

    dx_by_person: Dict[int, Set[str]] = {}
    for c in log.conditions:
        dx_by_person.setdefault(c["person_id"], set()).add(c["condition_key"])

    person_of = {p["person_id"]: p for p in log.persons}
    last_cover: Dict[int, date] = {}
    for op in log.observation_periods:
        pid = op["person_id"]
        cover_end = op["observation_period_end_date"]
        if cover_end > last_cover.get(pid, date.min):
            last_cover[pid] = cover_end

    target = int(round(rate * len(log.drugs)))
    if target <= 0:
        return 0

    # Sample without replacement from the visit list: one spurious order per
    # encounter, drawn across the whole file rather than clustered in whoever
    # happens to come first.
    pool = rng.permutation(len(log.visits))[: target * 5]
    planted = 0
    for i in pool:
        if planted >= target:
            break
        visit = log.visits[int(i)]
        pid = visit["person_id"]
        person = person_of.get(pid)
        if person is None:
            continue
        vdate = visit["visit_start_date"]
        birth = person["birth_datetime"]
        age_at = max(int((vdate - birth).days / 365.25), 0)
        carried = dx_by_person.get(pid, set())

        for j in rng.permutation(len(candidates)):
            drug_key = candidates[int(j)]
            if indications[drug_key] & carried:
                continue
            if not _drug_allowed(cat, drug_key, bool(person["gender_source_value"] == "F"), age_at, vdate):
                continue
            _append_drug(
                cat,
                log,
                ids,
                rng,
                pid,
                visit,
                drug_key,
                last_cover.get(pid, vdate),
                "off_label",
            )
            planted += 1
            break

    return planted


def _drug_allowed(cat: Catalog, drug_key: str, is_female: bool, age_at: int, when: date) -> bool:
    d = cat.drugs[drug_key]
    if d.sex == "female" and not is_female:
        return False
    if d.sex == "male" and is_female:
        return False
    if age_at < d.age_min or age_at > d.age_max:
        return False
    if d.valid_from and when < d.valid_from:
        return False
    if d.valid_to and when > d.valid_to:
        return False
    return True


def build_eras(records: List[dict], person_key: str, concept_key: str, start_key: str, end_key: str) -> List[dict]:
    """Collapse overlapping/near-adjacent records into eras (30-day gap rule)."""
    grouped: Dict[Tuple[int, str], List[Tuple[date, date]]] = {}
    for r in records:
        start = r[start_key]
        end = r.get(end_key) or start
        grouped.setdefault((r[person_key], r[concept_key]), []).append((start, end))

    eras = []
    era_id = 1
    for (pid, key), spans in grouped.items():
        spans.sort()
        cur_start, cur_end = spans[0]
        count = 1
        for start, end in spans[1:]:
            if (start - cur_end).days <= ERA_GAP_DAYS:
                cur_end = max(cur_end, end)
                count += 1
            else:
                eras.append({"era_id": era_id, "person_id": pid, "key": key, "start": cur_start, "end": cur_end, "count": count})
                era_id += 1
                cur_start, cur_end, count = start, end, 1
        eras.append({"era_id": era_id, "person_id": pid, "key": key, "start": cur_start, "end": cur_end, "count": count})
        era_id += 1

    return eras
