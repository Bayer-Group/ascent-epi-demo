#!/usr/bin/env python3
"""Plan obstetric episodes, including mother-infant linkage.

Pregnancy is the largest single gap in a synthetic EHR that claims to support
real-world evidence work: maternal-fetal safety is one of the commonest reasons
anyone opens an RWD database at all, and a dataset with no deliveries, no
gestational ages and no mother-child links cannot rehearse any of it. It is also
the only clinical domain here that is genuinely episodic - a bounded window with
an entry, a schedule, an outcome and a second person attached - so it is modelled
as an explicit episode rather than as prevalent disease.

Episodes are planned here and emitted by pathways.py, which owns visits, labs and
codes. Nothing in this module writes to the event log.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from models import Catalog
from patients import Population

# Age bands used by the complication age_multipliers, matching pregnancy.json.
FERTILITY_BANDS = [(15, 19), (20, 24), (25, 29), (30, 34), (35, 39), (40, 44), (45, 49)]


@dataclass
class PregnancyEpisode:
    """One pregnancy from conception to the end of postpartum follow-up."""

    conception: date
    end_date: date
    outcome: str
    gestation_weeks: float
    # (condition key, onset date, spec) for each complication that occurred.
    complications: List[Tuple[str, date, dict]] = field(default_factory=list)
    delivery_procedure: Optional[str] = None
    delivery_los: int = 0
    prenatal_dates: List[date] = field(default_factory=list)
    # (lab key, date)
    screening: List[Tuple[str, date]] = field(default_factory=list)
    # (procedure key, date)
    procedure_dates: List[Tuple[str, date]] = field(default_factory=list)
    postpartum_dates: List[date] = field(default_factory=list)
    # Person id of an in-cohort infant, when one was matched.
    infant_person_id: Optional[int] = None

    @property
    def delivered(self) -> bool:
        return self.outcome in ("live_birth", "stillbirth")


def _band_of(age: int) -> Optional[str]:
    for lo, hi in FERTILITY_BANDS:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    return None


def plan_pregnancies(
    cat: Catalog,
    pop: Population,
    death_year: np.ndarray,
    start: date,
    end: date,
    rng,
) -> Dict[int, List[PregnancyEpisode]]:
    """Draw episodes for every eligible woman, then back-fit in-cohort infants.

    Infants are matched from the existing cohort rather than minted as new
    people, so `--patients N` still produces exactly N persons. Any person born
    inside the observation window is a candidate; the matcher finds a woman whose
    planned delivery is close to that birth date and links the pair.
    """
    model = cat.pregnancy
    episodes: Dict[int, List[PregnancyEpisode]] = {}

    for idx in range(pop.n):
        if not pop.is_female[idx]:
            continue
        drawn = _plan_for_person(cat, pop, death_year, idx, start, end, rng)
        if drawn:
            episodes[idx] = drawn

    _link_infants(cat, pop, episodes, start, end, rng)

    # Sanity: an episode must not outlive its mother or precede the window.
    for idx, eps in episodes.items():
        episodes[idx] = [e for e in eps if start <= e.end_date <= end]

    _ = model  # kept for symmetry; every field is read via helpers below
    return {k: v for k, v in episodes.items() if v}


def _plan_for_person(cat, pop, death_year, idx, start, end, rng) -> List[PregnancyEpisode]:
    model = cat.pregnancy
    birth = date(int(pop.birth_year[idx]), int(pop.birth_month[idx]), int(pop.birth_day[idx]))
    dyear = int(death_year[idx])
    person_end = min(end, date(dyear, 12, 31)) if dyear >= 0 else end

    episodes: List[PregnancyEpisode] = []
    # Walk the window year by year and roll the age-specific fertility rate.
    year = max(start.year, birth.year + 15)
    last_end: Optional[date] = None

    while year <= person_end.year and len(episodes) < model.max_episodes_per_patient:
        age = year - birth.year
        rate = _fertility_rate(model, age)
        if rate <= 0.0 or rng.random() >= rate:
            year += 1
            continue

        conception = date(year, 1, 1) + timedelta(days=int(rng.integers(0, 365)))
        if last_end and (conception - last_end).days < model.min_interpregnancy_days:
            year += 1
            continue

        episode = _draw_episode(cat, pop, idx, conception, rng)
        if episode is None or episode.end_date > person_end or episode.end_date < start:
            year += 1
            continue

        episodes.append(episode)
        last_end = episode.end_date
        # Skip forward past the minimum interpregnancy interval.
        year = (last_end + timedelta(days=model.min_interpregnancy_days)).year

    return episodes


def _fertility_rate(model, age: int) -> float:
    for (lo, hi), rate in model.fertility_age_rates.items():
        if lo <= age <= hi:
            return rate
    return 0.0


def _draw_episode(cat: Catalog, pop: Population, idx: int, conception: date, rng) -> Optional[PregnancyEpisode]:
    model = cat.pregnancy

    shares = np.array([o["share"] for o in model.outcomes], dtype=float)
    outcome = model.outcomes[int(rng.choice(len(shares), p=shares / shares.sum()))]
    weeks = max(
        float(rng.normal(outcome["gestation_weeks_mean"], outcome["gestation_weeks_sd"])),
        float(outcome["min_weeks"]),
    )

    episode = PregnancyEpisode(
        conception=conception,
        end_date=conception + timedelta(days=int(weeks * 7)),
        outcome=outcome["key"],
        gestation_weeks=weeks,
    )

    _draw_complications(cat, pop, idx, episode, conception, rng)

    # A complication that forces early delivery shortens the pregnancy, which is
    # what actually makes preeclampsia and preterm labour visible in the data:
    # without it every gestational age is drawn from the same normal and the
    # complication is a code with no consequence attached.
    for _key, _onset, spec in episode.complications:
        shift = spec.get("forces_early_delivery_weeks")
        if shift and episode.delivered:
            episode.gestation_weeks = max(episode.gestation_weeks - float(shift), 22.0)
    episode.end_date = conception + timedelta(days=int(episode.gestation_weeks * 7))

    _schedule(cat, episode, rng)
    return episode


def _draw_complications(cat: Catalog, pop: Population, idx: int, episode, conception: date, rng) -> None:
    age = int((conception - date(int(pop.birth_year[idx]), int(pop.birth_month[idx]), int(pop.birth_day[idx]))).days / 365.25)
    band = _band_of(age)
    present: Dict[str, bool] = {}

    for spec in cat.pregnancy.complications:
        if episode.outcome not in spec.get("requires_outcome", []):
            continue

        p = float(spec["p"])
        for key, mult in spec.get("risk_multipliers", {}).items():
            # A risk factor may be a chronic condition the patient already has or
            # an earlier complication in this same pregnancy.
            has = present.get(key)
            if has is None:
                has = bool(pop.has[key][idx]) if key in pop.has else False
            if has:
                p *= float(mult)
        if band:
            p *= float(spec.get("age_multipliers", {}).get(band, 1.0))

        if rng.random() >= min(p, 0.95):
            continue

        present[spec["condition"]] = True
        if spec.get("at_delivery"):
            onset = episode.end_date
        else:
            week = float(rng.normal(spec["onset_week_mean"], spec["onset_week_sd"]))
            week = min(max(week, 6.0), episode.gestation_weeks)
            onset = conception + timedelta(days=int(week * 7))
        episode.complications.append((spec["condition"], onset, spec))


def _schedule(cat: Catalog, episode: PregnancyEpisode, rng) -> None:
    """Fill in prenatal, screening, procedure and postpartum dates."""
    model = cat.pregnancy

    for week in model.prenatal_weeks:
        if week > episode.gestation_weeks:
            break
        if rng.random() < model.prenatal_p_attend:
            episode.prenatal_dates.append(episode.conception + timedelta(days=int(week * 7)))

    for item in model.screening:
        if item["week"] > episode.gestation_weeks:
            continue
        if rng.random() < item["p"]:
            episode.screening.append((item["key"], episode.conception + timedelta(days=int(item["week"] * 7))))

    delivery_candidates: List[Tuple[str, float]] = []
    for item in model.procedures:
        if item.get("at_delivery"):
            if episode.outcome == item.get("outcome"):
                delivery_candidates.append((item["key"], float(item["share"])))
            continue
        if item["week"] > episode.gestation_weeks:
            continue
        if rng.random() < item["p"]:
            episode.procedure_dates.append((item["key"], episode.conception + timedelta(days=int(item["week"] * 7))))

    if delivery_candidates:
        keys = [k for k, _ in delivery_candidates]
        w = np.array([s for _, s in delivery_candidates], dtype=float)
        episode.delivery_procedure = keys[int(rng.choice(len(keys), p=w / w.sum()))]
        mean = (
            model.delivery["los_cesarean_mean"]
            if episode.delivery_procedure == "cesarean_delivery"
            else model.delivery["los_vaginal_mean"]
        )
        episode.delivery_los = max(int(rng.normal(mean, model.delivery["los_sd"])), 1)

    if episode.delivered:
        for weeks, p in zip(model.postpartum["weeks_after"], model.postpartum["p_attend"]):
            if rng.random() < p:
                episode.postpartum_dates.append(episode.end_date + timedelta(days=int(weeks * 7)))


def _link_infants(cat, pop, episodes, start: date, end: date, rng) -> None:
    """Match in-cohort newborns to a planned live birth on (or near) their birth date."""
    p_link = float(cat.pregnancy.newborn["p_linked_to_mother"])

    # Deliveries available for linkage, bucketed by year so the search is local.
    by_year: Dict[int, List[Tuple[int, PregnancyEpisode]]] = {}
    for idx, eps in episodes.items():
        for e in eps:
            if e.outcome == "live_birth" and e.infant_person_id is None:
                by_year.setdefault(e.end_date.year, []).append((idx, e))

    for idx in range(pop.n):
        birth = date(int(pop.birth_year[idx]), int(pop.birth_month[idx]), int(pop.birth_day[idx]))
        if birth < start or birth > end:
            continue
        if rng.random() >= p_link:
            continue

        candidates = by_year.get(birth.year, [])
        if not candidates:
            continue
        # Nearest unlinked delivery in the same year, and only if it is close
        # enough that shifting it to the infant's birth date stays plausible.
        best_pos, best_gap = -1, 10**9
        for pos, (mother_idx, e) in enumerate(candidates):
            if mother_idx == idx or e.infant_person_id is not None:
                continue
            gap = abs((e.end_date - birth).days)
            if gap < best_gap:
                best_pos, best_gap = pos, gap
        if best_pos < 0 or best_gap > 120:
            continue

        mother_idx, episode = candidates[best_pos]
        shift = birth - episode.end_date
        episode.end_date = birth
        episode.conception += shift
        episode.prenatal_dates = [d + shift for d in episode.prenatal_dates]
        episode.screening = [(k, d + shift) for k, d in episode.screening]
        episode.procedure_dates = [(k, d + shift) for k, d in episode.procedure_dates]
        episode.postpartum_dates = [d + shift for d in episode.postpartum_dates]
        episode.complications = [(k, d + shift, s) for k, d, s in episode.complications]
        episode.infant_person_id = int(pop.person_id[idx])
