#!/usr/bin/env python3
"""Load and validate the hand-authored clinical catalog."""

import hashlib
import json
import os
from datetime import date
from typing import Dict, List, Optional, Tuple

from models import (
    Calendar,
    Catalog,
    Condition,
    Drug,
    Grouper,
    Hierarchy,
    HierarchyGrouper,
    LabAnalyte,
    ObservationItem,
    ObservationValue,
    Payer,
    PregnancyModel,
    Procedure,
    Seasonality,
    SourceCode,
    TreatmentLine,
    VisitType,
)

CATALOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalog")

AGE_BANDS = ["0-17", "18-44", "45-64", "65-79", "80+"]

# Specimen and method axes the expansion is allowed to draw from. Anything an
# analyte does not list is not generated for it.
SPECIMENS = ["serum", "plasma", "whole blood", "urine", "cerebrospinal fluid"]
METHODS = ["automated", "immunoassay", "calculated"]


def _read(name: str) -> dict:
    with open(os.path.join(CATALOG_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def _strip_comments(raw: dict) -> dict:
    """Drop the _comment keys used for documentation inside the JSON."""
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _date(value: Optional[str]) -> Optional[date]:
    return date.fromisoformat(value) if value else None


def catalog_hash() -> str:
    """Stable digest of catalog content, recorded in ground truth."""
    digest = hashlib.sha256()
    for name in sorted(os.listdir(CATALOG_DIR)):
        if name.endswith(".json"):
            with open(os.path.join(CATALOG_DIR, name), "rb") as f:
                digest.update(f.read())
    return digest.hexdigest()[:16]


def load_catalog() -> Catalog:
    calendar = _load_calendar()

    raw_conditions = _read("conditions.json")
    groupers = {
        g["key"]: Grouper(**_strip_comments(g)) for g in raw_conditions.get("groupers", [])
    }

    conditions = {}
    for raw in raw_conditions["conditions"]:
        conditions[raw["key"]] = Condition(
            key=raw["key"],
            name=raw["name"],
            std_code=raw["std_code"],
            source_codes=[SourceCode(**_strip_comments(sc)) for sc in raw["source_codes"]],
            prevalence=raw["prevalence"],
            age_multipliers=raw.get("age_multipliers", {}),
            sex=raw.get("sex") or "any",
            age_min=raw.get("age_min", 0),
            age_max=raw.get("age_max", 120),
            onset_mean=raw.get("onset_mean", 50.0),
            onset_sd=raw.get("onset_sd", 15.0),
            chronic=raw.get("chronic", True),
            frailty_beta=raw.get("frailty_beta", 0.4),
            depends_on=raw.get("depends_on") or {},
            parent=raw.get("parent"),
            treatment_lines=[
                TreatmentLine(**_strip_comments(tl)) for tl in raw.get("treatment_lines", [])
            ],
            lab_panel=raw.get("lab_panel", []),
            procedures=raw.get("procedures", []),
            visit_rate=raw.get("visit_rate", 0.5),
            mortality_hr=raw.get("mortality_hr", 1.0),
            reference=raw.get("reference"),
            seasonality=raw.get("seasonality", "none"),
            valid_from=_date(raw.get("valid_from")),
            valid_to=_date(raw.get("valid_to")),
            cause_of_death_weight=raw.get("cause_of_death_weight", 0.0),
            monotonic_age=raw.get("monotonic_age", False),
            assigned_by=raw.get("assigned_by"),
        )

    drugs = {}
    for raw in _read("drugs.json")["drugs"]:
        raw = _strip_comments(raw)
        drugs[raw["key"]] = Drug(
            **{k: v for k, v in raw.items() if k not in ("valid_from", "valid_to")},
            valid_from=_date(raw.get("valid_from")),
            valid_to=_date(raw.get("valid_to")),
        )

    labs = {}
    for raw in _read("labs.json")["labs"]:
        labs[raw["key"]] = LabAnalyte(
            key=raw["key"],
            name=raw["name"],
            std_code=raw["std_code"],
            source_code=raw["source_code"],
            unit=raw["unit"],
            range_low=raw["range_low"],
            range_high=raw["range_high"],
            distribution={k: tuple(v) for k, v in raw["distribution"].items()},
            patient_intercept_sd=raw.get("patient_intercept_sd", 0.0),
            decimals=raw.get("decimals", 1),
            lognormal=raw.get("lognormal", False),
            plausible_min=raw.get("plausible_min", 0.0),
            plausible_max=raw.get("plausible_max", 1e9),
            specimens=raw.get("specimens", []),
            methods=raw.get("methods", []),
        )

    procedures = {}
    for raw in _read("procedures.json")["procedures"]:
        raw = _strip_comments(raw)
        procedures[raw["key"]] = Procedure(
            **{k: v for k, v in raw.items() if k not in ("valid_from", "valid_to")},
            valid_from=_date(raw.get("valid_from")),
            valid_to=_date(raw.get("valid_to")),
        )

    visits = {v["key"]: VisitType(**_strip_comments(v)) for v in _read("visits.json")["visits"]}

    demographics = _read("demographics.json")
    payers = [
        Payer(
            key=p["key"], name=p["name"], concept_id=p["concept_id"],
            plan_source_value=p["plan_source_value"], share=p["share"],
            paid_ratio=p["paid_ratio"], copay_mean=p["copay_mean"],
            age_shares=p.get("age_shares", {}),
        )
        for p in demographics["payers"]
    ]

    return Catalog(
        conditions=conditions,
        groupers=groupers,
        drugs=drugs,
        labs=labs,
        procedures=procedures,
        visits=visits,
        demographics=demographics,
        cohorts=_read("cohorts.json")["cohorts"],
        standard_concepts=_read("standard_concepts.json"),
        hierarchy=_load_hierarchy(),
        calendar=calendar,
        observations=_load_observations(),
        payers=payers,
        pregnancy=_load_pregnancy(),
    )


def _load_hierarchy() -> Dict[str, Hierarchy]:
    raw = _strip_comments(_read("hierarchy.json"))
    out: Dict[str, Hierarchy] = {}
    for domain, block in raw.items():
        block = _strip_comments(block)
        out[domain] = Hierarchy(
            domain=domain,
            vocabulary=block["vocabulary"],
            concept_class=block["concept_class"],
            root=block["root"],
            groupers={
                g["key"]: HierarchyGrouper(**_strip_comments(g)) for g in block["groupers"]
            },
            members=block["members"],
        )
    return out


def _normalise(monthly: List[float]) -> Tuple[float, ...]:
    """Scale a monthly profile to mean 1.0.

    A profile must only move events around the year, never change how many there
    are. Without this, editing a seasonality curve would silently change disease
    prevalence, which is not what the author of the curve intended.
    """
    mean = sum(monthly) / len(monthly)
    return tuple(m / mean for m in monthly)


def _load_calendar() -> Calendar:
    raw = _read("calendar.json")

    seasonality = {
        name: Seasonality(name=name, monthly=_normalise(vals))
        for name, vals in _strip_comments(raw["seasonality_profiles"]).items()
    }
    visit_seasonality = {
        name: Seasonality(name=name, monthly=_normalise(vals))
        for name, vals in _strip_comments(raw["visit_seasonality"]).items()
    }

    volume = raw["volume"]
    volume_anchors = sorted(
        (date.fromisoformat(a["date"]), a["multiplier"]) for a in volume["anchors"]
    )
    # Sort on the date alone: the second element is a dict, which is not orderable,
    # so a bare sorted() would raise as soon as two anchors shared a date.
    visit_mix_anchors = sorted(
        ((date.fromisoformat(a["date"]), a["shares"]) for a in raw["visit_mix"]["anchors"]),
        key=lambda pair: pair[0],
    )

    windows = {
        vocab: (date.fromisoformat(w["valid_from"]), date.fromisoformat(w["valid_to"]))
        for vocab, w in _strip_comments(raw["vocabulary_windows"]).items()
    }

    pandemic = raw["pandemic"]
    waves = sorted(
        ((date.fromisoformat(w["date"]), w["intensity"]) for w in pandemic["waves"]),
        key=lambda pair: pair[0],
    )

    return Calendar(
        seasonality=seasonality,
        visit_seasonality=visit_seasonality,
        volume_anchors=volume_anchors,
        censoring_lag_days=volume["censoring"]["lag_days"],
        censoring_floor=volume["censoring"]["floor"],
        visit_mix_anchors=visit_mix_anchors,
        vocabulary_windows=windows,
        cutover_date=date.fromisoformat(raw["cutover"]["date"]),
        cutover_tail_days=raw["cutover"]["tail_days"],
        cutover_tail_rate=raw["cutover"]["tail_rate"],
        pandemic_start=date.fromisoformat(pandemic["emergence_date"]),
        pandemic_waves=waves,
        error_budget=_strip_comments(raw["error_budget"]),
    )


def _load_observations() -> List[ObservationItem]:
    items = []
    for raw in _read("observations.json")["items"]:
        items.append(
            ObservationItem(
                key=raw["key"],
                name=raw["name"],
                concept_id=raw["concept_id"],
                source_code=raw["source_code"],
                value_type=raw["value_type"],
                cadence=raw["cadence"],
                p_recorded=raw["p_recorded"],
                values=[ObservationValue(**v) for v in raw.get("values", [])],
                conditioned_on=_strip_comments(raw.get("conditioned_on", {})),
                conditions=raw.get("conditions", []),
                age_bands=_strip_comments(raw.get("age_bands", {})),
                age_min=raw.get("age_min", 0),
                age_max=raw.get("age_max", 120),
            )
        )
    return items


def _load_pregnancy() -> PregnancyModel:
    raw = _read("pregnancy.json")
    fert = raw["fertility"]

    age_rates = {}
    for band, rate in _strip_comments(fert["age_rates"]).items():
        lo, hi = band.split("-")
        age_rates[(int(lo), int(hi))] = rate

    prenatal = raw["prenatal_schedule"]
    return PregnancyModel(
        fertility_age_rates=age_rates,
        min_interpregnancy_days=fert["min_interpregnancy_days"],
        max_episodes_per_patient=fert["max_episodes_per_patient"],
        outcomes=raw["outcomes"],
        prenatal_weeks=prenatal["weeks"],
        prenatal_visit_type=prenatal["visit_type"],
        prenatal_p_attend=prenatal["p_attend"],
        screening=raw["screening"],
        procedures=raw["procedures"],
        delivery=raw["delivery"],
        postpartum=raw["postpartum"],
        complications=raw["complications"],
        newborn=raw["newborn"],
    )


def validate_catalog(cat: Catalog) -> List[str]:
    """Return a list of referential problems; empty means the catalog is coherent."""
    errors: List[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    seen_std: Dict[str, str] = {}
    seen_source: Dict[str, str] = {}

    for key, c in cat.conditions.items():
        check(bool(c.source_codes), f"condition '{key}' has no source codes")
        for dep in c.depends_on:
            check(dep in cat.conditions, f"condition '{key}' depends on unknown '{dep}'")
        if c.parent:
            check(
                c.parent in cat.conditions or c.parent in cat.groupers,
                f"condition '{key}' has unknown parent '{c.parent}'",
            )
        for tl in c.treatment_lines:
            check(tl.drug in cat.drugs, f"condition '{key}' prescribes unknown drug '{tl.drug}'")
            check(0.0 <= tl.p_treated <= 1.0, f"condition '{key}' drug '{tl.drug}' p_treated out of range")
            check(0.0 <= tl.p_discontinue <= 1.0, f"condition '{key}' drug '{tl.drug}' p_discontinue out of range")
        for lab_key in c.lab_panel:
            check(lab_key in cat.labs, f"condition '{key}' orders unknown lab '{lab_key}'")
        for proc_key in c.procedures:
            check(proc_key in cat.procedures, f"condition '{key}' performs unknown procedure '{proc_key}'")

        # A condition an episode generator owns is not prevalence-sampled, so its
        # prevalence is 0 by construction rather than a mis-specified target.
        if c.assigned_by is None:
            check(0.0 < c.prevalence < 1.0, f"condition '{key}' prevalence must be in (0,1)")
        else:
            check(
                c.assigned_by in ("pregnancy",),
                f"condition '{key}' assigned_by '{c.assigned_by}' has no generator",
            )
            check(
                c.prevalence == 0.0,
                f"condition '{key}' is assigned by '{c.assigned_by}' but also carries prevalence "
                f"{c.prevalence}; it would be generated twice",
            )

        # age_min == age_max is legitimate: a birth-admission code applies at
        # exactly age 0 and nowhere else.
        check(c.age_min <= c.age_max, f"condition '{key}' has an inverted age range")
        for band in c.age_multipliers:
            check(band in AGE_BANDS, f"condition '{key}' references unknown age band '{band}'")

        check(
            c.seasonality in cat.calendar.seasonality,
            f"condition '{key}' uses unknown seasonality profile '{c.seasonality}'",
        )
        if c.valid_from and c.valid_to:
            check(c.valid_from < c.valid_to, f"condition '{key}' has an inverted validity window")
        check(c.cause_of_death_weight >= 0.0, f"condition '{key}' has a negative cause_of_death_weight")

        # A condition asserted to rise monotonically with age must actually say so
        # in its own multipliers, otherwise the assertion contradicts the input and
        # the validation suite would fail a build that is behaving as specified.
        if c.monotonic_age and c.age_multipliers:
            ordered = [c.age_multipliers[b] for b in AGE_BANDS if b in c.age_multipliers]
            check(
                all(a <= b for a, b in zip(ordered, ordered[1:])),
                f"condition '{key}' declares monotonic_age but its age_multipliers "
                f"are not non-decreasing across bands: {ordered}",
            )

        check(c.std_code not in seen_std, f"duplicate standard code '{c.std_code}' ({key}, {seen_std.get(c.std_code)})")
        seen_std[c.std_code] = key
        for sc in c.source_codes:
            check(
                sc.code not in seen_source,
                f"duplicate source code '{sc.code}' ({key}, {seen_source.get(sc.code)})",
            )
            seen_source[sc.code] = key
            if sc.legacy_code:
                check(
                    sc.legacy_code not in seen_source,
                    f"duplicate source code '{sc.legacy_code}' ({key}, {seen_source.get(sc.legacy_code)})",
                )
                seen_source[sc.legacy_code] = key

    for key, g in cat.groupers.items():
        if g.parent:
            check(g.parent in cat.groupers, f"grouper '{key}' has unknown parent '{g.parent}'")
        check(g.std_code not in seen_std, f"duplicate standard code '{g.std_code}' ({key}, {seen_std.get(g.std_code)})")
        seen_std[g.std_code] = key

    for key in cat.groupers:
        walked = set()
        cursor = key
        while cursor:
            if cursor in walked:
                errors.append(f"cycle in grouper hierarchy at '{key}'")
                break
            walked.add(cursor)
            cursor = cat.groupers[cursor].parent if cursor in cat.groupers else None

    # Lab distributions may only key off states the simulator can actually produce:
    # a bare condition key, or "<condition>_treated".
    valid_states = {"default"}
    for key in cat.conditions:
        valid_states.add(key)
        valid_states.add(f"{key}_treated")
    for key, lab in cat.labs.items():
        check("default" in lab.distribution, f"lab '{key}' has no default distribution")
        for state in lab.distribution:
            check(state in valid_states, f"lab '{key}' has distribution for unknown state '{state}'")
        check(lab.range_low < lab.range_high, f"lab '{key}' has an inverted reference range")
        check(
            lab.plausible_min < lab.plausible_max,
            f"lab '{key}' has an inverted plausible range",
        )
        # The plausible range must contain the reference range, or a perfectly
        # normal result would be clamped away as an artifact.
        check(
            lab.plausible_min <= lab.range_high and lab.plausible_max >= lab.range_low,
            f"lab '{key}' plausible range [{lab.plausible_min}, {lab.plausible_max}] does not "
            f"overlap its reference range [{lab.range_low}, {lab.range_high}]",
        )
        # Every state mean must be reachable. A mean outside the clamp would pile
        # every draw for that state onto the boundary.
        for state, (mean, _sd) in lab.distribution.items():
            check(
                lab.plausible_min <= mean <= lab.plausible_max,
                f"lab '{key}' state '{state}' has mean {mean} outside its plausible range",
            )
        for spec in lab.specimens:
            check(spec in SPECIMENS, f"lab '{key}' names unknown specimen '{spec}'")
        for method in lab.methods:
            check(method in METHODS, f"lab '{key}' names unknown method '{method}'")

    for key, d in cat.drugs.items():
        check(bool(d.days_supply), f"drug '{key}' has no days_supply options")
        if d.valid_from and d.valid_to:
            check(d.valid_from < d.valid_to, f"drug '{key}' has an inverted validity window")

    # Every unit needs an OMOP concept, or measurements land with unit_concept_id 0.
    mapped_units = {u["unit"] for u in cat.demographics["units"]}
    for key, lab in cat.labs.items():
        check(lab.unit in mapped_units, f"lab '{key}' uses unit '{lab.unit}' with no concept mapping")

    errors.extend(_validate_standard_concepts(cat))
    errors.extend(_validate_hierarchy(cat, seen_std))
    errors.extend(_validate_calendar(cat))
    errors.extend(_validate_observations(cat))
    errors.extend(_validate_pregnancy(cat))
    errors.extend(_validate_cohorts(cat))

    # Tab/newline in a name would corrupt an unquoted COPY during vocabulary load.
    for group in (cat.conditions, cat.groupers, cat.drugs, cat.labs, cat.procedures, cat.visits):
        for key, item in group.items():
            check(
                "\t" not in item.name and "\n" not in item.name,
                f"'{key}' name contains a tab or newline, which breaks COPY",
            )

    return errors


def _validate_hierarchy(cat: Catalog, seen_std: Dict[str, str]) -> List[str]:
    """The non-condition roll-ups must be a rooted forest over real catalog keys."""
    errors: List[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    leaves = {
        "Measurement": cat.labs,
        "Procedure": cat.procedures,
        "Visit": cat.visits,
        "Drug": {},
    }
    for domain, h in cat.hierarchy.items():
        check(domain in leaves, f"hierarchy declares unknown domain '{domain}'")
        check(h.root in h.groupers, f"hierarchy '{domain}' root '{h.root}' is not a grouper")

        for key, g in h.groupers.items():
            check(
                g.std_code not in seen_std,
                f"duplicate standard code '{g.std_code}' ({key}, {seen_std.get(g.std_code)})",
            )
            seen_std[g.std_code] = key
            if g.parent:
                check(g.parent in h.groupers, f"hierarchy '{domain}' grouper '{key}' has unknown parent '{g.parent}'")
            else:
                check(key == h.root, f"hierarchy '{domain}' grouper '{key}' has no parent but is not the root")

        for key in h.groupers:
            walked = set()
            cursor: Optional[str] = key
            while cursor:
                if cursor in walked:
                    errors.append(f"cycle in hierarchy '{domain}' at '{key}'")
                    break
                walked.add(cursor)
                cursor = h.groupers[cursor].parent if cursor in h.groupers else None

        # A member naming a leaf that does not exist is a silent no-op: the leaf
        # would quietly fall back to the root and the intended grouping vanish.
        for leaf, grouper in h.members.items():
            check(leaf in leaves.get(domain, {}), f"hierarchy '{domain}' maps unknown leaf '{leaf}'")
            check(grouper in h.groupers, f"hierarchy '{domain}' maps '{leaf}' to unknown grouper '{grouper}'")

    return errors


def _validate_standard_concepts(cat: Catalog) -> List[str]:
    """Every structural concept_id the writers reference must exist as a concept row.

    This is the check that would have caught the empty-demographic-join defect:
    person.gender_concept_id held 8532 while the concept table contained only the
    fictitious clinical concepts, so the join every OMOP tool starts with returned
    nothing at all.
    """
    errors: List[str] = []
    sc = cat.standard_concepts

    known = {sc["sentinel"]["concept_id"]}
    for section in (
        "gender", "race", "ethnicity", "units", "type_concepts", "routes",
        "places_of_service", "specialties", "payer_concepts", "death_causes",
        "observation_concepts", "observation_values",
    ):
        for row in sc.get(section, []):
            known.add(row["concept_id"])

    demo = cat.demographics
    for section in ("gender", "race", "ethnicity"):
        for row in demo[section]:
            if row["concept_id"] not in known:
                errors.append(
                    f"demographics.{section} uses concept_id {row['concept_id']} "
                    f"({row['name']}) with no row in standard_concepts.json"
                )
    for row in demo["units"]:
        if row["concept_id"] not in known:
            errors.append(
                f"unit '{row['unit']}' uses concept_id {row['concept_id']} "
                f"with no row in standard_concepts.json"
            )
    for name, cid in demo["type_concepts"].items():
        if cid not in known:
            errors.append(f"type_concept '{name}' ({cid}) has no row in standard_concepts.json")
    for payer in cat.payers:
        if payer.concept_id not in known:
            errors.append(f"payer '{payer.key}' ({payer.concept_id}) has no row in standard_concepts.json")

    # Routes named by drugs must map to a route concept.
    route_names = {r["route"] for r in sc.get("routes", [])}
    for key, drug in cat.drugs.items():
        if drug.route not in route_names:
            errors.append(f"drug '{key}' uses route '{drug.route}' with no route concept")

    # Places of service named by care sites must map.
    pos_names = {p["concept_name"] for p in sc.get("places_of_service", [])}
    for site in demo["care_sites"]:
        if site["place_of_service"] not in pos_names:
            errors.append(
                f"care site '{site['name']}' uses place_of_service "
                f"'{site['place_of_service']}' with no concept"
            )

    # Provider specialties must map.
    spec_names = {s["specialty"] for s in sc.get("specialties", [])}
    for specialty in demo["provider_specialties"]:
        if specialty not in spec_names:
            errors.append(f"provider specialty '{specialty}' has no specialty concept")

    return errors


def _validate_calendar(cat: Catalog) -> List[str]:
    errors: List[str] = []
    cal = cat.calendar

    for name, profile in list(cal.seasonality.items()) + list(cal.visit_seasonality.items()):
        if len(profile.monthly) != 12:
            errors.append(f"seasonality profile '{name}' has {len(profile.monthly)} months, not 12")
        if any(m < 0 for m in profile.monthly):
            errors.append(f"seasonality profile '{name}' has a negative multiplier")

    for key in cat.visits:
        if key not in cal.visit_seasonality:
            errors.append(f"visit type '{key}' has no entry in calendar visit_seasonality")

    if not cal.volume_anchors:
        errors.append("calendar has no volume anchors")
    for _when, multiplier in cal.volume_anchors:
        if multiplier <= 0:
            errors.append("calendar volume anchor has a non-positive multiplier")

    for when, shares in cal.visit_mix_anchors:
        unknown = set(shares) - set(cat.visits)
        if unknown:
            errors.append(f"visit_mix anchor {when} names unknown visit types {sorted(unknown)}")
        missing = set(cat.visits) - set(shares)
        if missing:
            errors.append(f"visit_mix anchor {when} omits visit types {sorted(missing)}")
        if any(s < 0 for s in shares.values()):
            errors.append(f"visit_mix anchor {when} has a negative share")

    for vocab in ("DXCODE", "DXCODE9", "MEDLEX", "PHARMLEX", "LABLEX", "PXCODE"):
        if vocab not in cal.vocabulary_windows:
            errors.append(f"calendar has no vocabulary window for '{vocab}'")

    # The cutover must sit exactly where the legacy vocabulary ends, or source
    # records would carry a code system that was already retired.
    if "DXCODE9" in cal.vocabulary_windows:
        _, legacy_end = cal.vocabulary_windows["DXCODE9"]
        if (cal.cutover_date - legacy_end).days != 1:
            errors.append(
                f"cutover date {cal.cutover_date} does not follow the DXCODE9 window end "
                f"{legacy_end}; legacy codes would be emitted outside their validity"
            )

    for name, rate in cal.error_budget.items():
        if not 0.0 <= rate <= 1.0:
            errors.append(f"error budget '{name}' rate {rate} is not a probability")

    return errors


def _validate_observations(cat: Catalog) -> List[str]:
    errors: List[str] = []
    for item in cat.observations:
        if item.value_type == "concept":
            if not item.values:
                errors.append(f"observation '{item.key}' is value_type concept but lists no values")
            total = sum(v.share for v in item.values)
            if abs(total - 1.0) > 0.01:
                errors.append(f"observation '{item.key}' value shares sum to {total:.3f}, not 1")
            keys = {v.key for v in item.values}
            for cond, override in item.conditioned_on.items():
                if cond not in cat.conditions:
                    errors.append(f"observation '{item.key}' conditions on unknown condition '{cond}'")
                if set(override) != keys:
                    errors.append(
                        f"observation '{item.key}' override for '{cond}' does not cover "
                        f"exactly its value keys"
                    )
            for band, override in item.age_bands.items():
                if band not in AGE_BANDS:
                    errors.append(f"observation '{item.key}' references unknown age band '{band}'")
                if set(override) != keys:
                    errors.append(
                        f"observation '{item.key}' age-band override for '{band}' does not cover "
                        f"exactly its value keys"
                    )
        elif item.value_type == "condition_concept":
            for entry in item.conditions:
                if entry["condition"] not in cat.conditions:
                    errors.append(
                        f"observation '{item.key}' reports family history of unknown "
                        f"condition '{entry['condition']}'"
                    )
        else:
            errors.append(f"observation '{item.key}' has unknown value_type '{item.value_type}'")

        if item.cadence not in ("once", "annual", "on_change"):
            errors.append(f"observation '{item.key}' has unknown cadence '{item.cadence}'")
    return errors


def _validate_pregnancy(cat: Catalog) -> List[str]:
    errors: List[str] = []
    preg = cat.pregnancy

    total = sum(o["share"] for o in preg.outcomes)
    if abs(total - 1.0) > 0.01:
        errors.append(f"pregnancy outcome shares sum to {total:.3f}, not 1")

    for entry in preg.screening:
        if entry["key"] not in cat.labs:
            errors.append(f"pregnancy screening names unknown lab '{entry['key']}'")
    for entry in preg.procedures:
        if entry["key"] not in cat.procedures:
            errors.append(f"pregnancy names unknown procedure '{entry['key']}'")

    owned = {k for k, c in cat.conditions.items() if c.assigned_by == "pregnancy"}
    for comp in preg.complications:
        key = comp["condition"]
        if key not in cat.conditions:
            errors.append(f"pregnancy complication names unknown condition '{key}'")
            continue
        if key not in owned:
            errors.append(
                f"pregnancy complication '{key}' is not marked assigned_by=pregnancy, "
                f"so the prevalence sampler would generate it as well"
            )
        for risk in comp.get("risk_multipliers", {}):
            if risk not in cat.conditions:
                errors.append(f"complication '{key}' keys risk off unknown condition '{risk}'")
        for lab_key in comp.get("labs", []):
            if lab_key not in cat.labs:
                errors.append(f"complication '{key}' orders unknown lab '{lab_key}'")
        treatment = comp.get("treatment")
        if treatment and treatment["drug"] not in cat.drugs:
            errors.append(f"complication '{key}' prescribes unknown drug '{treatment['drug']}'")

    for section, field_name in ((preg.delivery, "condition"), (preg.newborn["birth_encounter"], "condition")):
        key = section[field_name]
        if key not in cat.conditions:
            errors.append(f"pregnancy references unknown condition '{key}'")

    for visit_key in (
        preg.prenatal_visit_type,
        preg.delivery["visit_type"],
        preg.postpartum["visit_type"],
        preg.newborn["birth_encounter"]["visit_type"],
    ):
        if visit_key not in cat.visits:
            errors.append(f"pregnancy references unknown visit type '{visit_key}'")

    # Every maternal condition must be female-only, or the generator would be free
    # to assign a delivery to a male patient.
    for key in owned:
        cond = cat.conditions[key]
        if key != cat.pregnancy.newborn["birth_encounter"]["condition"] and cond.sex != "female":
            errors.append(f"maternal condition '{key}' is not restricted to sex=female")

    return errors


def _validate_cohorts(cat: Catalog) -> List[str]:
    """Check that every attrition step names an operator the evaluator implements."""
    errors: List[str] = []

    known_steps = {
        "any_condition_dx", "no_condition_dx", "repeat_condition_dx", "prior_observation",
        "drug_exposure", "no_drug_exposure", "lab_threshold", "age_range", "sex",
        "condition_after_condition", "observation_value", "visit_type", "continuous_coverage",
    }

    for i, cohort in enumerate(cat.cohorts):
        label = cohort.get("key", f"#{i}")
        if "question" not in cohort:
            errors.append(f"cohort {label} has no question")
        steps = cohort.get("attrition")
        if not steps:
            errors.append(f"cohort {label} has no attrition ladder")
            continue
        for step in steps:
            op = step.get("type")
            if op not in known_steps:
                errors.append(f"cohort {label} uses unimplemented attrition step '{op}'")
            for key_field, universe, what in (
                ("condition", cat.conditions, "condition"),
                ("drug", cat.drugs, "drug"),
                ("lab", cat.labs, "lab"),
            ):
                name = step.get(key_field)
                if name and name not in universe and name not in cat.groupers:
                    errors.append(f"cohort {label} step references unknown {what} '{name}'")
    return errors
