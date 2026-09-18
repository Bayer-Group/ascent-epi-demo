#!/usr/bin/env python3
"""Frozen dataclasses describing the hand-authored clinical catalog."""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

SEX_ANY = "any"
SEX_MALE = "male"
SEX_FEMALE = "female"

# Sentinel bounds for concepts that were valid for the whole of recorded time.
EPOCH_START = date(1900, 1, 1)
EPOCH_END = date(2099, 12, 31)


@dataclass(frozen=True)
class SourceCode:
    """A non-standard source code (what a real EMR actually stores)."""

    code: str
    name: str
    legacy_code: Optional[str] = None


@dataclass(frozen=True)
class TreatmentLine:
    """A drug offered for a condition, ordered by line of therapy."""

    drug: str
    p_treated: float
    line: int = 1
    # Probability of stopping at each refill decision point. Discontinuation is
    # permanent for that drug: real non-persistence is not a coin flipped afresh
    # at every fill, and treating it that way produced refill chains that paused
    # and silently resumed.
    p_discontinue: float = 0.03


@dataclass(frozen=True)
class Grouper:
    """A roll-up concept that is never diagnosed directly but anchors the hierarchy."""

    key: str
    name: str
    std_code: str
    parent: Optional[str] = None


@dataclass(frozen=True)
class Condition:
    key: str
    name: str
    std_code: str
    source_codes: List[SourceCode]
    # Target marginal prevalence in the whole population.
    prevalence: float
    # Multiplicative prevalence adjustment per age band, e.g. {"65+": 2.4}.
    age_multipliers: Dict[str, float] = field(default_factory=dict)
    sex: str = SEX_ANY
    age_min: int = 0
    age_max: int = 120
    # Age at which the condition first appears, truncated to [age_min, current age].
    onset_mean: float = 50.0
    onset_sd: float = 15.0
    chronic: bool = True
    # Loading on the shared frailty term: higher means more clustered with other disease.
    frailty_beta: float = 0.4
    # Log-odds bumps applied when the named condition is already present.
    depends_on: Dict[str, float] = field(default_factory=dict)
    parent: Optional[str] = None
    treatment_lines: List[TreatmentLine] = field(default_factory=list)
    lab_panel: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)
    # Extra visits per year attributable to this condition.
    visit_rate: float = 0.5
    mortality_hr: float = 1.0
    # Published figure this prevalence was calibrated against.
    reference: Optional[str] = None

    # --- calendar-dependence -------------------------------------------------
    # Named profile in calendar.json; drives when in the year onsets fall.
    seasonality: str = "none"
    # The disease (and therefore its codes) could not exist outside this window.
    # Enforced as a hard clamp applied AFTER every other date adjustment.
    valid_from: Optional[date] = None
    valid_to: Optional[date] = None

    # --- mortality -----------------------------------------------------------
    # Relative propensity that a decedent carrying this condition died OF it.
    # Zero means it never appears as an underlying cause of death.
    cause_of_death_weight: float = 0.0

    # --- validation ----------------------------------------------------------
    # Assert that recorded prevalence must not fall as age rises. The validation
    # suite fails the build otherwise; this is what catches a calibration that
    # saturates and produces a dip in the oldest band.
    monotonic_age: bool = False

    # --- ownership -----------------------------------------------------------
    # Non-None means a dedicated episode generator assigns this condition and the
    # prevalence sampler must skip it entirely (currently only "pregnancy").
    assigned_by: Optional[str] = None


@dataclass(frozen=True)
class Drug:
    key: str
    name: str
    ingredient: str
    std_code: str
    pkg_code: str
    dose_form: str
    strength: str
    route: str
    days_supply: List[int]
    sex: str = SEX_ANY
    age_min: int = 0
    age_max: int = 120
    # Market availability. A prescription dated before the drug existed is the
    # cheapest possible proof that a dataset was fabricated.
    valid_from: Optional[date] = None
    valid_to: Optional[date] = None


@dataclass(frozen=True)
class LabAnalyte:
    key: str
    name: str
    std_code: str
    source_code: str
    unit: str
    # The REFERENCE range: what a normal result looks like.
    range_low: float
    range_high: float
    # Mean/sd keyed by clinical state; "default" is the healthy baseline.
    distribution: Dict[str, Tuple[float, float]]
    # Between-patient variation, so repeated measures on one person correlate.
    patient_intercept_sd: float = 0.0
    decimals: int = 1
    lognormal: bool = False
    # The PHYSIOLOGIC range: outside this a value is an artifact, not a result.
    # Distinct from the reference range - total cholesterol's reference range
    # starts at 0 because the convention is "<200", but a serum cholesterol of 0
    # is impossible. Draws are clamped here; unresulted tests are NULL, not 0.
    plausible_min: float = 0.0
    plausible_max: float = 1e9
    # Which specimen/method variants the expansion may legitimately generate.
    # Empty means the analyte takes no variant on that axis.
    specimens: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)

    def clamp(self, value: float) -> float:
        """Constrain a drawn value to the physiologically possible range."""
        return min(max(value, self.plausible_min), self.plausible_max)


@dataclass(frozen=True)
class Procedure:
    key: str
    name: str
    std_code: str
    source_code: str
    sex: str = SEX_ANY
    age_min: int = 0
    age_max: int = 120
    valid_from: Optional[date] = None
    valid_to: Optional[date] = None
    # Chance the procedure is performed when its indicating condition first
    # appears. 1.0 -- the default -- suits routine work-up that every new
    # diagnosis triggers; surgery needs a real rate, or every obese patient
    # ends up with a gastric bypass.
    p_index: float = 1.0


@dataclass(frozen=True)
class VisitType:
    key: str
    name: str
    std_code: str
    source_code: str
    # Inpatient stays span several days; everything else is same-day.
    inpatient: bool = False
    los_mean: float = 0.0


@dataclass(frozen=True)
class ObservationValue:
    """One coded answer to an observation question."""

    key: str
    name: str
    concept_id: int
    share: float


@dataclass(frozen=True)
class ObservationItem:
    """A non-measurement fact recorded about a patient (smoking, alcohol, family history)."""

    key: str
    name: str
    concept_id: int
    source_code: str
    # "concept" -> value_as_concept_id from `values`;
    # "condition_concept" -> value_as_concept_id is a standard condition concept.
    value_type: str
    # "once", "annual" or "on_change".
    cadence: str
    p_recorded: float
    values: List[ObservationValue] = field(default_factory=list)
    # condition key -> {value key: share}, overriding the marginal shares.
    conditioned_on: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # For value_type == "condition_concept": which conditions may be reported.
    conditions: List[dict] = field(default_factory=list)
    # Coarse age band -> {value key: share}, overriding the marginal shares.
    age_bands: Dict[str, Dict[str, float]] = field(default_factory=dict)
    age_min: int = 0
    age_max: int = 120


@dataclass(frozen=True)
class HierarchyGrouper:
    """A roll-up concept in a domain the condition catalog does not cover."""

    key: str
    name: str
    std_code: str
    parent: Optional[str] = None


@dataclass(frozen=True)
class Hierarchy:
    """Roll-up structure for one domain, so descendant queries reach its leaves."""

    domain: str
    vocabulary: str
    concept_class: str
    root: str
    groupers: Dict[str, HierarchyGrouper]
    # leaf catalog key -> grouper key. A leaf that is absent hangs off the root.
    members: Dict[str, str]

    def grouper_for(self, key: str) -> str:
        return self.members.get(key, self.root)

    def chain(self, grouper_key: str) -> List[str]:
        """The grouper and every grouper above it, nearest first."""
        chain: List[str] = []
        cursor: Optional[str] = grouper_key
        while cursor:
            chain.append(cursor)
            cursor = self.groupers[cursor].parent
        return chain


@dataclass(frozen=True)
class Payer:
    key: str
    name: str
    concept_id: int
    plan_source_value: str
    share: float
    paid_ratio: float
    copay_mean: float
    # Coarse age band -> share, overriding the marginal share.
    age_shares: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Seasonality:
    """Monthly intensity multipliers, January first, normalised to mean 1.0."""

    name: str
    monthly: Tuple[float, ...]

    def factor(self, month: int) -> float:
        return self.monthly[month - 1]


@dataclass(frozen=True)
class Calendar:
    """Everything that makes the data a function of calendar time."""

    seasonality: Dict[str, Seasonality]
    visit_seasonality: Dict[str, Seasonality]
    # (date, multiplier) anchors, ascending, linearly interpolated.
    volume_anchors: List[Tuple[date, float]]
    censoring_lag_days: int
    censoring_floor: float
    # (date, {visit key: share}) anchors, ascending, linearly interpolated.
    visit_mix_anchors: List[Tuple[date, Dict[str, float]]]
    vocabulary_windows: Dict[str, Tuple[date, date]]
    cutover_date: date
    cutover_tail_days: int
    cutover_tail_rate: float
    pandemic_start: date
    pandemic_waves: List[Tuple[date, float]]
    error_budget: Dict[str, float]

    def volume_factor(self, when: date) -> float:
        """Relative encounter volume on a date: secular growth plus the 2020 dip."""
        return _interpolate(self.volume_anchors, when)

    def censoring_factor(self, when: date, extract: date) -> float:
        """Claims run-out: records near the extract date are still arriving.

        A file whose last month is as full as the month before it has no run-out
        shelf, which is not what any real extract looks like, and it invites a
        reader to treat the final partial period as a genuine drop in utilisation.
        """
        remaining = (extract - when).days
        if remaining >= self.censoring_lag_days:
            return 1.0
        if remaining <= 0:
            return self.censoring_floor
        share = remaining / self.censoring_lag_days
        return self.censoring_floor + (1.0 - self.censoring_floor) * share

    def visit_mix(self, when: date) -> Dict[str, float]:
        """Visit-type shares on a date, interpolated between anchors."""
        anchors = self.visit_mix_anchors
        if when <= anchors[0][0]:
            return dict(anchors[0][1])
        if when >= anchors[-1][0]:
            return dict(anchors[-1][1])
        for (d0, m0), (d1, m1) in zip(anchors, anchors[1:]):
            if d0 <= when <= d1:
                span = (d1 - d0).days or 1
                t = (when - d0).days / span
                return {k: m0[k] + (m1.get(k, 0.0) - m0[k]) * t for k in m0}
        return dict(anchors[-1][1])

    def pandemic_intensity(self, when: date) -> float:
        """Relative COVID incidence on a date; zero before the disease existed."""
        if when < self.pandemic_start:
            return 0.0
        return _interpolate(self.pandemic_waves, when)


def _interpolate(anchors: List[Tuple[date, float]], when: date) -> float:
    """Linear interpolation over (date, value) anchors, flat outside the range."""
    if when <= anchors[0][0]:
        return anchors[0][1]
    if when >= anchors[-1][0]:
        return anchors[-1][1]
    for (d0, v0), (d1, v1) in zip(anchors, anchors[1:]):
        if d0 <= when <= d1:
            span = (d1 - d0).days or 1
            return v0 + (v1 - v0) * ((when - d0).days / span)
    return anchors[-1][1]


@dataclass(frozen=True)
class PregnancyModel:
    """Obstetric episode structure: cadence, outcomes and complications."""

    fertility_age_rates: Dict[Tuple[int, int], float]
    min_interpregnancy_days: int
    max_episodes_per_patient: int
    outcomes: List[dict]
    prenatal_weeks: List[int]
    prenatal_visit_type: str
    prenatal_p_attend: float
    screening: List[dict]
    procedures: List[dict]
    delivery: dict
    postpartum: dict
    complications: List[dict]
    newborn: dict


@dataclass(frozen=True)
class Catalog:
    conditions: Dict[str, Condition]
    groupers: Dict[str, Grouper]
    drugs: Dict[str, Drug]
    labs: Dict[str, LabAnalyte]
    procedures: Dict[str, Procedure]
    visits: Dict[str, VisitType]
    demographics: dict
    cohorts: List[dict]
    standard_concepts: dict
    # Domain name -> roll-up structure for labs, procedures, visits and drugs.
    hierarchy: Dict[str, Hierarchy]
    calendar: Calendar
    observations: List[ObservationItem]
    payers: List[Payer]
    pregnancy: PregnancyModel

    def sampled_conditions(self) -> List[str]:
        """Condition keys the prevalence sampler owns.

        Excludes anything an episode generator assigns, which would otherwise be
        sampled twice - once at its (zero) prevalence and once by the episode.
        """
        return [k for k, c in self.conditions.items() if c.assigned_by is None]

    def condition_order(self) -> List[str]:
        """Topologically ordered condition keys so dependencies resolve first."""
        ordered: List[str] = []
        seen = set()

        def visit(key: str, stack: Tuple[str, ...] = ()) -> None:
            if key in seen:
                return
            if key in stack:
                raise ValueError(f"Cycle in condition dependencies: {' -> '.join(stack + (key,))}")
            for dep in self.conditions[key].depends_on:
                visit(dep, stack + (key,))
            seen.add(key)
            ordered.append(key)

        for key in self.sampled_conditions():
            visit(key)
        return ordered

    def ancestors_of(self, key: str) -> List[str]:
        """Every grouper above a condition or grouper, nearest first."""
        chain: List[str] = []
        cursor = self.conditions[key].parent if key in self.conditions else (
            self.groupers[key].parent if key in self.groupers else None
        )
        while cursor:
            chain.append(cursor)
            cursor = self.groupers[cursor].parent if cursor in self.groupers else None
        return chain
