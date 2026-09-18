#!/usr/bin/env python3
"""Sample the patient population: demographics, comorbidities, onset ages, mortality."""

from datetime import date
from typing import Dict, List, Optional

import numpy as np

from models import Catalog

# Gompertz baseline annual mortality: h(age) = A * exp(B * age).
# Chosen so annual risk is ~0.1% at 40 and ~9% at 85, roughly matching US life tables.
GOMPERTZ_A = 2.2e-5
GOMPERTZ_B = 0.0925

# Exponent applied to the summed log hazard ratios; < 1 keeps multimorbid
# patients at plausible risk instead of near-certain death.
HR_DAMPING = 0.45

# Fallback cause-of-death concepts when a decedent carried nothing fatal.
CAUSE_NATURAL = 4306655
CAUSE_ACCIDENTAL = 4145711
P_ACCIDENTAL_DEATH = 0.06

# "Natural causes" competes against the decedent's own conditions rather than
# only catching patients who had none. Without it, anyone whose sole weighted
# condition was hypertension was certain to be recorded as dying of hypertension,
# which made it the leading cause in the file at 13% - roughly four times its
# real share. As a competing risk it now claims most of those deaths, while a
# lung-cancer patient (weight 9.0 against this 1.5) still nearly always dies of
# the cancer.
NATURAL_BASELINE_WEIGHT = 1.5


class Population:
    """Patient-level latent state, all arrays indexed by patient position."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.person_id = np.arange(1, n + 1, dtype=np.int64)
        self.age = np.zeros(n, dtype=np.int32)
        self.is_female = np.zeros(n, dtype=bool)
        self.gender_concept_id = np.zeros(n, dtype=np.int64)
        self.race_concept_id = np.zeros(n, dtype=np.int64)
        self.ethnicity_concept_id = np.zeros(n, dtype=np.int64)
        self.race_source = np.empty(n, dtype=object)
        self.ethnicity_source = np.empty(n, dtype=object)
        self.birth_year = np.zeros(n, dtype=np.int32)
        self.birth_month = np.zeros(n, dtype=np.int32)
        self.birth_day = np.zeros(n, dtype=np.int32)
        self.frailty = np.zeros(n)
        # condition key -> bool mask / onset age
        self.has: Dict[str, np.ndarray] = {}
        self.onset_age: Dict[str, np.ndarray] = {}
        # Home facility. Encounters may still occur elsewhere - see network in
        # demographics.json - so this is where the patient usually goes, not a
        # partition of the population.
        self.facility = np.empty(n, dtype=object)
        self.mrn = np.empty(n, dtype=object)
        # A single enterprise-wide identifier that survives moving between
        # facilities. Cross-site linkage is the central problem in multi-site RWD
        # and cannot be exercised without an identifier that spans sites.
        self.member_id = np.empty(n, dtype=object)
        self.location_id = np.zeros(n, dtype=np.int64)
        self.payer = np.empty(n, dtype=object)
        # Underlying cause of death, filled by assign_causes_of_death: a condition
        # key, the string "natural" or "accidental", or None for survivors. Kept as
        # a key rather than a concept id because the vocabulary belongs to the
        # writer, not to the population model.
        self.cause_key = np.empty(n, dtype=object)

    def condition_keys(self) -> List[str]:
        return list(self.has)


def _weighted_choice(rng, options, n, key="share"):
    weights = np.array([o[key] for o in options], dtype=float)
    weights = weights / weights.sum()
    return rng.choice(len(options), size=n, p=weights)


def _age_band_index(ages: np.ndarray, bands: List[dict]) -> np.ndarray:
    idx = np.zeros(len(ages), dtype=np.int32)
    for i, b in enumerate(bands):
        idx[(ages >= b["min"]) & (ages <= b["max"])] = i
    return idx


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def sample_population(cat: Catalog, n: int, extract_date: date, rng) -> Population:
    """Draw demographics, then comorbidities in dependency order with calibrated marginals.

    Ages are ages ON THE EXTRACT DATE, which is what every consumer of the file
    will compute from the date of birth.
    """
    pop = Population(n)
    demo = cat.demographics

    genders = demo["gender"]
    gi = _weighted_choice(rng, genders, n)
    pop.gender_concept_id = np.array([genders[i]["concept_id"] for i in gi], dtype=np.int64)
    pop.is_female = pop.gender_concept_id == 8532

    races = demo["race"]
    ri = _weighted_choice(rng, races, n)
    pop.race_concept_id = np.array([races[i]["concept_id"] for i in ri], dtype=np.int64)
    pop.race_source = np.array([races[i]["source_value"] for i in ri], dtype=object)

    eths = demo["ethnicity"]
    ei = _weighted_choice(rng, eths, n)
    pop.ethnicity_concept_id = np.array([eths[i]["concept_id"] for i in ei], dtype=np.int64)
    pop.ethnicity_source = np.array([eths[i]["source_value"] for i in ei], dtype=object)

    pop.age = _sample_ages(demo, pop.is_female, rng)

    pop.birth_month = rng.integers(1, 13, size=n).astype(np.int32)
    pop.birth_day = rng.integers(1, 29, size=n).astype(np.int32)
    # Subtracting the age from the extract YEAR gave anyone aged 0 a birth date
    # anywhere in that calendar year, including the eleven months after the
    # extract closed - patients born after the file ends, whose coverage
    # therefore started after it ended. Deciding the birth year from whether the
    # birthday has already come round makes age-at-extract exact instead.
    had_birthday = (pop.birth_month < extract_date.month) | (
        (pop.birth_month == extract_date.month) & (pop.birth_day <= extract_date.day)
    )
    pop.birth_year = (extract_date.year - pop.age - np.where(had_birthday, 0, 1)).astype(np.int32)

    locations = demo["locations"]
    li = rng.integers(0, len(locations), size=n)
    pop.location_id = np.array([locations[i]["location_id"] for i in li], dtype=np.int64)

    # Shared frailty: one latent health axis that makes disease cluster within patients.
    pop.frailty = rng.standard_normal(n)

    bands = demo["age_bands"]
    band_idx = _age_band_index(pop.age, bands)
    band_names = [b["band"] for b in bands]

    for key in cat.condition_order():
        c = cat.conditions[key]

        eligible = (pop.age >= c.age_min) & (pop.age <= c.age_max)
        if c.sex == "female":
            eligible &= pop.is_female
        elif c.sex == "male":
            eligible &= ~pop.is_female

        rel = np.ones(n)
        for i, name in enumerate(band_names):
            rel[band_idx == i] = c.age_multipliers.get(name, 1.0)

        linear = c.frailty_beta * pop.frailty
        for dep_key, gamma in c.depends_on.items():
            linear = linear + gamma * pop.has[dep_key].astype(float)

        probs = _calibrate(c.prevalence, rel, linear, eligible, band_idx, bands)
        pop.has[key] = rng.random(n) < probs

        # Onset age, resampled into the window where the condition is possible.
        floor = float(max(c.age_min, 0))
        ceiling = np.maximum(pop.age.astype(float), floor)
        pop.onset_age[key] = _truncated_onset_age(c, floor, ceiling, n, rng)

    # Conditions an episode generator owns still need array slots, so every
    # downstream consumer can iterate pop.has uniformly.
    for key, c in cat.conditions.items():
        if c.assigned_by is not None:
            pop.has[key] = np.zeros(n, dtype=bool)
            pop.onset_age[key] = np.zeros(n)

    pop.payer = _assign_payers(cat, pop, band_names, band_idx, rng)
    pop.facility = _assign_facilities(cat, pop, rng)
    # An enterprise identifier that is stable across facilities, unlike the
    # per-site MRN. Formatted unlike any MRN so the two cannot be confused.
    pop.member_id = np.array([f"M{100000000 + int(pid) * 7:09d}" for pid in pop.person_id], dtype=object)

    return pop


def _truncated_onset_age(cond, floor: float, ceiling: np.ndarray, n: int, rng) -> np.ndarray:
    """Draw an age at onset inside [floor, the patient's current age].

    Clipping the normal draw instead of resampling it put a point mass on the
    patient's current age: every patient whose modelled onset lay in the future
    was recorded as having developed the disease today. Anchored on the extract
    date, "today" is one calendar day, so 2.4% of all onsets in the file landed in
    its final month at five times the rate of any other. Rejection sampling keeps
    the same truncated distribution without the spike; the uniform fallback covers
    the patients an old-age disease reached decades early, where rejection would
    otherwise spin.
    """
    onset = rng.normal(cond.onset_mean, cond.onset_sd, size=n)
    for _ in range(8):
        bad = (onset < floor) | (onset > ceiling)
        if not bad.any():
            return onset
        onset[bad] = rng.normal(cond.onset_mean, cond.onset_sd, size=int(bad.sum()))
    bad = (onset < floor) | (onset > ceiling)
    onset[bad] = floor + rng.random(int(bad.sum())) * np.maximum(ceiling[bad] - floor, 0.0)
    return onset


def _assign_facilities(cat: Catalog, pop: Population, rng) -> np.ndarray:
    """Pick each patient's home facility, enriched by the facility's case mix.

    Facilities used to be assigned by market share alone, which made every site a
    uniform random sample of the same population - so the oncology centre and the
    primary-care group had identical disease profiles, and `case_mix` in
    demographics.json described nothing. Weighting each site by the multipliers
    for the groupers a patient's conditions roll up to gives the sites genuinely
    different denominators, which is what makes provider-level confounding, site
    selection and multi-site pooling exercises meaningful.
    """
    facilities = cat.demographics["facilities"]
    base = np.array([f["share"] for f in facilities], dtype=float)

    # Precompute, per facility, the multiplier that each condition attracts.
    per_condition: List[Dict[str, float]] = []
    for f in facilities:
        mix = f.get("case_mix", {})
        weights = {}
        for key in pop.has:
            m = 1.0
            for grouper in cat.ancestors_of(key):
                if grouper in mix:
                    m *= float(mix[grouper])
            if m != 1.0:
                weights[key] = m
        per_condition.append(weights)

    out = np.empty(pop.n, dtype=object)
    codes = [f["code"] for f in facilities]
    present = {key: mask for key, mask in pop.has.items() if mask.any()}

    log_w = np.tile(np.log(base), (pop.n, 1))
    for j, weights in enumerate(per_condition):
        for key, m in weights.items():
            mask = present.get(key)
            if mask is not None:
                # Log-additive so several conditions compound without one of them
                # driving the probability to 1.
                log_w[mask, j] += np.log(m) * 0.5

    probs = np.exp(log_w - log_w.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    draws = rng.random(pop.n)
    picks = (probs.cumsum(axis=1) < draws[:, None]).sum(axis=1).clip(0, len(codes) - 1)
    for i in range(pop.n):
        out[i] = codes[picks[i]]
    return out


def _sample_ages(demo: dict, is_female: np.ndarray, rng) -> np.ndarray:
    """Draw ages from a real pyramid weighted by healthcare utilization.

    Ages used to be drawn by picking one of five coarse bands and then drawing
    uniformly inside it. That made every age from 18 to 44 exactly equally likely
    and put a discontinuity at each band edge, which is not what any real age
    distribution looks like.

    Two separate things are being modelled here and conflating them is the usual
    mistake. `share` is how common that age is in the general population. But an
    EHR or claims extract is not a census - it contains whoever sought care, so it
    over-represents infants (well-child visits) and the old (chronic disease), and
    under-represents healthy adults in their twenties. `utilization` carries that
    second effect. Sampling proportional to share x utilization gives the
    three-humped shape real extracts have.
    """
    n = len(is_female)
    pyramid = demo["age_pyramid"]
    bands = pyramid["bands"]

    base = np.array([b["share"] * b["utilization"] for b in bands], dtype=float)

    sex_util = pyramid.get("sex_utilization", {})
    female_w = _sex_weights(bands, sex_util.get("female", {}))
    male_w = _sex_weights(bands, sex_util.get("male", {}))

    ages = np.zeros(n, dtype=np.int32)
    for mask, weights in ((is_female, base * female_w), (~is_female, base * male_w)):
        count = int(mask.sum())
        if count == 0:
            continue
        probs = weights / weights.sum()
        picked = rng.choice(len(bands), size=count, p=probs)
        lows = np.array([bands[i]["min"] for i in picked])
        highs = np.array([bands[i]["max"] for i in picked])
        ages[mask] = rng.integers(lows, highs + 1)

    return ages


def _sex_weights(bands: List[dict], spec: dict) -> np.ndarray:
    """Per-band utilization multiplier for one sex, from "lo-hi" range keys."""
    default = spec.get("default", 1.0)
    weights = np.full(len(bands), float(default))
    for band_key, multiplier in spec.items():
        if band_key == "default":
            continue
        lo, hi = (int(x) for x in band_key.split("-"))
        for i, b in enumerate(bands):
            # Overlap, not containment: a 15-49 rule must cover the 45-49 band.
            if b["min"] <= hi and b["max"] >= lo:
                weights[i] = float(multiplier)
    return weights


def _calibrate(
    target: float,
    rel: np.ndarray,
    linear: np.ndarray,
    eligible: np.ndarray,
    band_idx: np.ndarray,
    bands: List[dict],
) -> np.ndarray:
    """Fit an intercept per age band so realized prevalence matches the catalog.

    A single shared intercept was fitted here before, and it produced a real
    defect: hypertension prevalence fell in the 80+ band even though the catalog
    asks for its highest multiplier there. The mechanism is saturation. One
    intercept has to satisfy the whole population at once, so for a common
    condition it is pushed high enough that the oldest band's log-odds land far
    out on the flat part of the logistic curve. Once p is near 1 the age
    multiplier has almost no room left to act, and ordinary sampling noise in a
    small 80+ denominator is then free to push the realized rate below the band
    below it.

    Fitting one intercept per band removes the coupling entirely. The per-band
    targets are derived so that their share-weighted average is still exactly the
    catalog's marginal prevalence:

        target_b = prevalence * m_b / sum_b(share_b * m_b)

    so the overall figure a literature comparison checks is unchanged, while the
    age gradient becomes exactly what the catalog specifies rather than whatever
    survived saturation.
    """
    n = len(rel)
    probs = np.zeros(n)
    if not eligible.any():
        return probs

    n_bands = len(bands)
    band_eligible = [(band_idx == b) & eligible for b in range(n_bands)]
    counts = np.array([int(m.sum()) for m in band_eligible], dtype=float)
    if counts.sum() == 0:
        return probs

    multipliers = np.array([rel[m].mean() if m.any() else 0.0 for m in band_eligible])
    # Share of the WHOLE population that each band's eligible members make up.
    # Using the whole population as the denominator is what keeps the recovered
    # marginal comparable to the catalog target, which is also expressed over
    # everyone rather than over the eligible subset.
    shares = counts / n

    denominator = float((shares * multipliers).sum())
    if denominator <= 0:
        return probs

    targets = target * multipliers / denominator
    targets = _cap_and_redistribute(targets, shares, target)

    for b in range(n_bands):
        mask = band_eligible[b]
        if mask.any() and targets[b] > 0:
            probs[mask] = _fit_intercept(float(targets[b]), linear[mask])

    return probs


def _cap_and_redistribute(
    targets: np.ndarray, shares: np.ndarray, marginal: float, cap: float = 0.98
) -> np.ndarray:
    """Hold per-band targets below `cap` while preserving the overall marginal.

    A band whose multiplier is large enough can be asked for a prevalence above 1,
    which is unachievable. Clamping alone would quietly lose those cases and drag
    the overall marginal below the catalog figure, so whatever a capped band
    cannot absorb is pushed onto the bands that still have headroom. If every band
    saturates the marginal is genuinely unreachable and the capped values stand.
    """
    out = np.clip(targets, 0.0, cap)
    for _ in range(10):
        realized = float((out * shares).sum())
        deficit = marginal - realized
        if abs(deficit) < 1e-9:
            break
        headroom = (cap - out) * shares
        total_headroom = float(headroom.sum())
        if deficit > 0:
            if total_headroom <= 1e-12:
                break
            out = out + np.where(shares > 0, deficit * headroom / total_headroom / np.maximum(shares, 1e-12), 0.0)
        else:
            room = out * shares
            total_room = float(room.sum())
            if total_room <= 1e-12:
                break
            out = out + np.where(shares > 0, deficit * room / total_room / np.maximum(shares, 1e-12), 0.0)
        out = np.clip(out, 0.0, cap)
    return out


def _fit_intercept(target: float, linear: np.ndarray) -> np.ndarray:
    """Binary-search one intercept so the mean probability equals the target.

    The frailty and dependency terms shift every patient's log-odds, so applying
    the target prevalence directly would overshoot. Without this the ground-truth
    file would record intended rather than actual prevalence.
    """
    lo, hi = -30.0, 30.0
    probs = np.zeros(len(linear))
    for _ in range(80):
        alpha = (lo + hi) / 2
        probs = 1.0 / (1.0 + np.exp(-(alpha + linear)))
        realized = float(probs.mean())
        if abs(realized - target) < 1e-7:
            break
        if realized < target:
            lo = alpha
        else:
            hi = alpha
    return probs


def _assign_payers(cat: Catalog, pop: Population, band_names, band_idx, rng) -> np.ndarray:
    """Assign a primary payer, conditioned on age.

    Medicare eligibility at 65 is the sharpest discontinuity in US healthcare
    data. A payer drawn independently of age would put 19% of infants on Medicare
    and hide a structural feature every real analysis has to cope with.
    """
    n = pop.n
    payers = cat.payers
    out = np.empty(n, dtype=object)

    for b, band in enumerate(band_names):
        mask = band_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        weights = np.array(
            [p.age_shares.get(band, p.share) for p in payers], dtype=float
        )
        if weights.sum() <= 0:
            weights = np.array([p.share for p in payers], dtype=float)
        weights = weights / weights.sum()
        picked = rng.choice(len(payers), size=count, p=weights)
        out[mask] = [payers[i].key for i in picked]

    return out


def simulate_mortality(cat: Catalog, pop: Population, start_year: int, end_year: int, rng) -> np.ndarray:
    """Year-by-year survival; returns the year of death, or -1 for survivors.

    Hazard is Gompertz in age scaled by the product of comorbidity hazard ratios,
    so death concentrates in older and sicker patients rather than at a flat rate.
    """
    n = pop.n
    death_year = np.full(n, -1, dtype=np.int32)

    log_hr = {key: np.log(cat.conditions[key].mortality_hr) for key in pop.has}

    alive = np.ones(n, dtype=bool)
    for year in range(start_year, end_year + 1):
        age_then = pop.age - (end_year - year)

        # Comorbidity risk only counts once the disease has actually appeared.
        total_log_hr = np.zeros(n)
        for key, mask in pop.has.items():
            active = mask & (age_then >= pop.onset_age[key])
            total_log_hr += np.where(active, log_hr[key], 0.0)

        # Sub-multiplicative: stacking five diagnoses should not multiply five
        # hazard ratios outright, which would send annual risk past certainty.
        hr = np.exp(HR_DAMPING * total_log_hr)

        baseline = GOMPERTZ_A * np.exp(GOMPERTZ_B * np.maximum(age_then, 0))
        p_die = np.clip(baseline * hr, 0, 0.95)
        died = alive & (age_then >= 0) & (rng.random(n) < p_die)
        death_year[died] = year
        alive &= ~died

    return death_year


def assign_causes_of_death(cat: Catalog, pop: Population, death_year: np.ndarray, rng) -> np.ndarray:
    """Pick an underlying cause for each decedent from the disease they carried.

    death.cause_concept_id was a hardcoded 0 for every death, which makes the
    column useless: cause-specific mortality is a primary endpoint in most
    outcomes research and a constant 0 cannot support it. The cause is now drawn
    from the decedent's own conditions - only those that had already appeared by
    the year of death - weighted by cause_of_death_weight. CKD patients therefore
    die disproportionately of renal and cardiovascular causes, and a lung-cancer
    patient rarely dies of something incidental.

    Conditions with weight 0 (hyperlipidemia, GERD, prediabetes) are never
    selected however common they are, because they do not appear as an underlying
    cause on a real death certificate.

    Returns an object array of condition keys / "natural" / "accidental", with
    None for survivors. Concept-id resolution belongs to the writer, which owns
    the vocabulary.
    """
    causes = np.empty(pop.n, dtype=object)
    decedents = np.flatnonzero(death_year >= 0)
    if len(decedents) == 0:
        pop.cause_key = causes
        return causes

    fatal = [
        (key, cat.conditions[key].cause_of_death_weight)
        for key in sorted(pop.has)
        if cat.conditions[key].cause_of_death_weight > 0
    ]

    # A minority of deaths are unrelated to anything in the patient's chart.
    accidental = rng.random(pop.n) < P_ACCIDENTAL_DEATH

    for idx in decedents:
        if accidental[idx]:
            causes[idx] = "accidental"
            continue

        age_at_death = int(death_year[idx] - pop.birth_year[idx])

        candidates, weights = ["natural"], [NATURAL_BASELINE_WEIGHT]
        for key, weight in fatal:
            if pop.has[key][idx] and pop.onset_age[key][idx] <= age_at_death:
                candidates.append(key)
                weights.append(weight)

        w = np.array(weights, dtype=float)
        causes[idx] = candidates[int(rng.choice(len(candidates), p=w / w.sum()))]

    pop.cause_key = causes
    return causes
