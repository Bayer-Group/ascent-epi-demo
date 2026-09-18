#!/usr/bin/env python3
"""Render the event log into an OMOP CDM 5.4 database."""

import os
import sqlite3
from typing import Dict, List, Optional

from models import Catalog
from pathways import EventLog, build_eras
from vocabulary import Vocabulary

CDM_VERSION = "5.4"

DDL = [
    """CREATE TABLE person (
        person_id INTEGER PRIMARY KEY, gender_concept_id INTEGER, year_of_birth INTEGER,
        month_of_birth INTEGER, day_of_birth INTEGER, birth_datetime TEXT,
        race_concept_id INTEGER, ethnicity_concept_id INTEGER, location_id INTEGER,
        provider_id INTEGER, care_site_id INTEGER, person_source_value TEXT,
        gender_source_value TEXT, gender_source_concept_id INTEGER, race_source_value TEXT,
        race_source_concept_id INTEGER, ethnicity_source_value TEXT, ethnicity_source_concept_id INTEGER)""",
    """CREATE TABLE observation_period (
        observation_period_id INTEGER PRIMARY KEY, person_id INTEGER,
        observation_period_start_date TEXT, observation_period_end_date TEXT,
        period_type_concept_id INTEGER)""",
    """CREATE TABLE visit_occurrence (
        visit_occurrence_id INTEGER PRIMARY KEY, person_id INTEGER, visit_concept_id INTEGER,
        visit_start_date TEXT, visit_start_datetime TEXT, visit_end_date TEXT, visit_end_datetime TEXT,
        visit_type_concept_id INTEGER, provider_id INTEGER, care_site_id INTEGER,
        visit_source_value TEXT, visit_source_concept_id INTEGER,
        admitted_from_concept_id INTEGER, discharged_to_concept_id INTEGER,
        preceding_visit_occurrence_id INTEGER)""",
    """CREATE TABLE condition_occurrence (
        condition_occurrence_id INTEGER PRIMARY KEY, person_id INTEGER, condition_concept_id INTEGER,
        condition_start_date TEXT, condition_start_datetime TEXT, condition_end_date TEXT,
        condition_end_datetime TEXT, condition_type_concept_id INTEGER, condition_status_concept_id INTEGER,
        stop_reason TEXT, provider_id INTEGER, visit_occurrence_id INTEGER, visit_detail_id INTEGER,
        condition_source_value TEXT, condition_source_concept_id INTEGER, condition_status_source_value TEXT)""",
    """CREATE TABLE drug_exposure (
        drug_exposure_id INTEGER PRIMARY KEY, person_id INTEGER, drug_concept_id INTEGER,
        drug_exposure_start_date TEXT, drug_exposure_start_datetime TEXT, drug_exposure_end_date TEXT,
        drug_exposure_end_datetime TEXT, verbatim_end_date TEXT, drug_type_concept_id INTEGER,
        stop_reason TEXT, refills INTEGER, quantity REAL, days_supply INTEGER, sig TEXT,
        route_concept_id INTEGER, lot_number TEXT, provider_id INTEGER, visit_occurrence_id INTEGER,
        visit_detail_id INTEGER, drug_source_value TEXT, drug_source_concept_id INTEGER,
        route_source_value TEXT, dose_unit_source_value TEXT)""",
    """CREATE TABLE measurement (
        measurement_id INTEGER PRIMARY KEY, person_id INTEGER, measurement_concept_id INTEGER,
        measurement_date TEXT, measurement_datetime TEXT, measurement_time TEXT,
        measurement_type_concept_id INTEGER, operator_concept_id INTEGER, value_as_number REAL,
        value_as_concept_id INTEGER, unit_concept_id INTEGER, range_low REAL, range_high REAL,
        provider_id INTEGER, visit_occurrence_id INTEGER, visit_detail_id INTEGER,
        measurement_source_value TEXT, measurement_source_concept_id INTEGER, unit_source_value TEXT,
        unit_source_concept_id INTEGER, value_source_value TEXT, measurement_event_id INTEGER,
        meas_event_field_concept_id INTEGER)""",
    """CREATE TABLE procedure_occurrence (
        procedure_occurrence_id INTEGER PRIMARY KEY, person_id INTEGER, procedure_concept_id INTEGER,
        procedure_date TEXT, procedure_datetime TEXT, procedure_end_date TEXT, procedure_end_datetime TEXT,
        procedure_type_concept_id INTEGER, modifier_concept_id INTEGER, quantity INTEGER,
        provider_id INTEGER, visit_occurrence_id INTEGER, visit_detail_id INTEGER,
        procedure_source_value TEXT, procedure_source_concept_id INTEGER, modifier_source_value TEXT)""",
    """CREATE TABLE observation (
        observation_id INTEGER PRIMARY KEY, person_id INTEGER, observation_concept_id INTEGER,
        observation_date TEXT, observation_datetime TEXT, observation_type_concept_id INTEGER,
        value_as_number REAL, value_as_string TEXT, value_as_concept_id INTEGER,
        qualifier_concept_id INTEGER, unit_concept_id INTEGER, provider_id INTEGER,
        visit_occurrence_id INTEGER, visit_detail_id INTEGER, observation_source_value TEXT,
        observation_source_concept_id INTEGER, unit_source_value TEXT, qualifier_source_value TEXT,
        value_source_value TEXT, observation_event_id INTEGER, obs_event_field_concept_id INTEGER)""",
    """CREATE TABLE death (
        person_id INTEGER PRIMARY KEY, death_date TEXT, death_datetime TEXT,
        death_type_concept_id INTEGER, cause_concept_id INTEGER, cause_source_value TEXT,
        cause_source_concept_id INTEGER)""",
    """CREATE TABLE drug_era (
        drug_era_id INTEGER PRIMARY KEY, person_id INTEGER, drug_concept_id INTEGER,
        drug_era_start_date TEXT, drug_era_end_date TEXT, drug_exposure_count INTEGER,
        gap_days INTEGER)""",
    """CREATE TABLE condition_era (
        condition_era_id INTEGER PRIMARY KEY, person_id INTEGER, condition_concept_id INTEGER,
        condition_era_start_date TEXT, condition_era_end_date TEXT, condition_occurrence_count INTEGER)""",
    """CREATE TABLE care_site (
        care_site_id INTEGER PRIMARY KEY, care_site_name TEXT, place_of_service_concept_id INTEGER,
        location_id INTEGER, care_site_source_value TEXT, place_of_service_source_value TEXT)""",
    """CREATE TABLE provider (
        provider_id INTEGER PRIMARY KEY, provider_name TEXT, npi TEXT, dea TEXT,
        specialty_concept_id INTEGER, care_site_id INTEGER, year_of_birth INTEGER,
        gender_concept_id INTEGER, provider_source_value TEXT, specialty_source_value TEXT,
        specialty_source_concept_id INTEGER, gender_source_value TEXT, gender_source_concept_id INTEGER)""",
    """CREATE TABLE cdm_source (
        cdm_source_name TEXT, cdm_source_abbreviation TEXT, cdm_holder TEXT, source_description TEXT,
        source_documentation_reference TEXT, cdm_etl_reference TEXT, source_release_date TEXT,
        cdm_release_date TEXT, cdm_version TEXT, cdm_version_concept_id INTEGER, vocabulary_version TEXT)""",
    """CREATE TABLE concept (
        concept_id INTEGER PRIMARY KEY, concept_name TEXT, domain_id TEXT, vocabulary_id TEXT,
        concept_class_id TEXT, standard_concept TEXT, concept_code TEXT,
        valid_start_date INTEGER, valid_end_date INTEGER, invalid_reason TEXT)""",
    """CREATE TABLE concept_relationship (
        concept_id_1 INTEGER, concept_id_2 INTEGER, relationship_id TEXT,
        valid_start_date INTEGER, valid_end_date INTEGER, invalid_reason TEXT)""",
    """CREATE TABLE concept_ancestor (
        ancestor_concept_id INTEGER, descendant_concept_id INTEGER,
        min_levels_of_separation INTEGER, max_levels_of_separation INTEGER)""",
    """CREATE TABLE vocabulary (
        vocabulary_id TEXT PRIMARY KEY, vocabulary_name TEXT, vocabulary_reference TEXT,
        vocabulary_version TEXT, vocabulary_concept_id INTEGER)""",
    """CREATE TABLE location (
        location_id INTEGER PRIMARY KEY, address_1 TEXT, address_2 TEXT, city TEXT, state TEXT,
        zip TEXT, county TEXT, location_source_value TEXT, country_concept_id INTEGER,
        country_source_value TEXT, latitude REAL, longitude REAL)""",
    """CREATE TABLE payer_plan_period (
        payer_plan_period_id INTEGER PRIMARY KEY, person_id INTEGER,
        payer_plan_period_start_date TEXT, payer_plan_period_end_date TEXT,
        payer_concept_id INTEGER, payer_source_value TEXT, payer_source_concept_id INTEGER,
        plan_concept_id INTEGER, plan_source_value TEXT, plan_source_concept_id INTEGER,
        sponsor_concept_id INTEGER, sponsor_source_value TEXT, sponsor_source_concept_id INTEGER,
        family_source_value TEXT, stop_reason_concept_id INTEGER, stop_reason_source_value TEXT,
        stop_reason_source_concept_id INTEGER)""",
    """CREATE TABLE cost (
        cost_id INTEGER PRIMARY KEY, cost_event_id INTEGER, cost_domain_id TEXT,
        cost_type_concept_id INTEGER, currency_concept_id INTEGER, total_charge REAL,
        total_cost REAL, total_paid REAL, paid_by_payer REAL, paid_by_patient REAL,
        paid_patient_copay REAL, paid_patient_coinsurance REAL, paid_patient_deductible REAL,
        paid_by_primary REAL, paid_ingredient_cost REAL, paid_dispensing_fee REAL,
        payer_plan_period_id INTEGER, amount_allowed REAL, revenue_code_concept_id INTEGER,
        revenue_code_source_value TEXT, drg_concept_id INTEGER, drg_source_value TEXT)""",
    """CREATE TABLE fact_relationship (
        domain_concept_id_1 INTEGER, fact_id_1 INTEGER, domain_concept_id_2 INTEGER,
        fact_id_2 INTEGER, relationship_concept_id INTEGER)""",
]

INDEXES = [
    "CREATE INDEX idx_vo_person ON visit_occurrence(person_id)",
    "CREATE INDEX idx_vo_date ON visit_occurrence(visit_start_date)",
    "CREATE INDEX idx_co_person ON condition_occurrence(person_id)",
    "CREATE INDEX idx_co_concept ON condition_occurrence(condition_concept_id)",
    "CREATE INDEX idx_co_date ON condition_occurrence(condition_start_date)",
    "CREATE INDEX idx_de_person ON drug_exposure(person_id)",
    "CREATE INDEX idx_de_concept ON drug_exposure(drug_concept_id)",
    "CREATE INDEX idx_de_date ON drug_exposure(drug_exposure_start_date)",
    "CREATE INDEX idx_m_person ON measurement(person_id)",
    "CREATE INDEX idx_m_concept ON measurement(measurement_concept_id)",
    "CREATE INDEX idx_m_date ON measurement(measurement_date)",
    "CREATE INDEX idx_po_person ON procedure_occurrence(person_id)",
    "CREATE INDEX idx_po_concept ON procedure_occurrence(procedure_concept_id)",
    "CREATE INDEX idx_op_person ON observation_period(person_id)",
    "CREATE INDEX idx_ca_ancestor ON concept_ancestor(ancestor_concept_id)",
    "CREATE INDEX idx_ca_descendant ON concept_ancestor(descendant_concept_id)",
    "CREATE INDEX idx_cr_1 ON concept_relationship(concept_id_1)",
    "CREATE INDEX idx_concept_code ON concept(vocabulary_id, concept_code)",
    "CREATE INDEX idx_o_person ON observation(person_id)",
    "CREATE INDEX idx_o_concept ON observation(observation_concept_id)",
    "CREATE INDEX idx_ppp_person ON payer_plan_period(person_id)",
    "CREATE INDEX idx_cost_event ON cost(cost_event_id, cost_domain_id)",
    "CREATE INDEX idx_fr_1 ON fact_relationship(fact_id_1)",
]

# US dollars, so the cost table's amounts are not unitless.
CURRENCY_USD = 44818668
# "Cost record from claim" - what a payer-adjudicated amount is sourced from.
COST_TYPE = 5032


def _d(value) -> Optional[str]:
    return value.isoformat() if value else None


def write_omop_sqlite(
    cat: Catalog, vocab: Vocabulary, log: EventLog, db_path: str, seed: int
) -> Dict[str, int]:
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    try:
        for stmt in DDL:
            conn.execute(stmt)
        counts = _populate(cat, vocab, log, conn, seed)
        for stmt in INDEXES:
            conn.execute(stmt)
        conn.commit()
    finally:
        conn.close()
    return counts


def _populate(cat: Catalog, vocab: Vocabulary, log: EventLog, conn, seed: int) -> Dict[str, int]:
    demo = cat.demographics
    unit_concept = {u["unit"]: u["concept_id"] for u in demo["units"]}
    type_concepts = demo["type_concepts"]
    route_concept = vocab.structural.get("routes", {})
    counts: Dict[str, int] = {}

    conn.executemany(
        "INSERT INTO person VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                p["person_id"], p["gender_concept_id"], p["year_of_birth"], p["month_of_birth"],
                p["day_of_birth"], f"{_d(p['birth_datetime'])} 00:00:00", p["race_concept_id"],
                p["ethnicity_concept_id"],
                # location_id and care_site_id were NULL for every person. That is
                # not a cosmetic omission: geographic and provider-level variation
                # are two of the commonest stratifications in RWE, and both were
                # unavailable even though the generator already knew the answer.
                p["location_id"], None, p["care_site_id"], p["person_source_value"],
                p["gender_source_value"], 0, p["race_source_value"], 0,
                p["ethnicity_source_value"], 0,
            )
            for p in log.persons
        ],
    )
    counts["person"] = len(log.persons)

    conn.executemany(
        "INSERT INTO location VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (loc["location_id"], None, None, loc["city"], loc["state"], loc["zip"],
             loc["county"], f"LOC{loc['location_id']:03d}", 0, "US", None, None)
            for loc in demo["locations"]
        ],
    )
    counts["location"] = len(demo["locations"])

    conn.executemany(
        "INSERT INTO observation_period VALUES (?,?,?,?,?)",
        [
            (o["observation_period_id"], o["person_id"], _d(o["observation_period_start_date"]),
             _d(o["observation_period_end_date"]), o["period_type_concept_id"])
            for o in log.observation_periods
        ],
    )
    counts["observation_period"] = len(log.observation_periods)

    payer_concept = {p.key: p.concept_id for p in cat.payers}
    conn.executemany(
        "INSERT INTO payer_plan_period VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (pp["payer_plan_period_id"], pp["person_id"], _d(pp["payer_plan_period_start_date"]),
             _d(pp["payer_plan_period_end_date"]),
             payer_concept.get(pp["payer_source_value"], 0), pp["payer_source_value"], 0,
             0, pp["plan_source_value"], 0, 0, None, 0, pp["family_source_value"], 0, None, 0)
            for pp in log.payer_plan_periods
        ],
    )
    counts["payer_plan_period"] = len(log.payer_plan_periods)
    plan_by_person: Dict[int, List[dict]] = {}
    for pp in log.payer_plan_periods:
        plan_by_person.setdefault(pp["person_id"], []).append(pp)

    def plan_for(person_id: int, when) -> Optional[int]:
        for pp in plan_by_person.get(person_id, ()):
            if pp["payer_plan_period_start_date"] <= when <= pp["payer_plan_period_end_date"]:
                return pp["payer_plan_period_id"]
        return None

    cost_rows: List[tuple] = []

    def add_cost(event_id: int, domain: str, person_id: int, when, charge, paid_payer=None, paid_patient=None):
        """One cost row per billable event, linked to the plan in force that day."""
        if charge is None:
            return
        payer_share = charge * 0.72 if paid_payer is None else paid_payer
        patient_share = 0.0 if paid_patient is None else paid_patient
        cost_rows.append(
            (
                len(cost_rows) + 1, event_id, domain, COST_TYPE, CURRENCY_USD,
                round(charge, 2), None, round(payer_share + patient_share, 2),
                round(payer_share, 2), round(patient_share, 2), round(patient_share, 2), 0.0, 0.0,
                None, None, None, plan_for(person_id, when), round(payer_share + patient_share, 2),
                0, None, 0, None,
            )
        )

    visit_rows = []
    previous_visit: Dict[int, int] = {}
    for v in sorted(log.visits, key=lambda r: (r["person_id"], r["visit_start_date"], r["visit_occurrence_id"])):
        std = vocab.std_id[v["visit_concept_key"]]
        src_code = cat.visits[v["visit_source_key"]].source_code
        src = vocab.src_id[(v["visit_source_key"], src_code)]
        visit_rows.append(
            (
                v["visit_occurrence_id"], v["person_id"], std, _d(v["visit_start_date"]),
                f"{_d(v['visit_start_date'])} 09:00:00", _d(v["visit_end_date"]),
                f"{_d(v['visit_end_date'])} 17:00:00", v["visit_type_concept_id"],
                v["provider_id"], v["care_site_id"], src_code, src, 0, 0,
                previous_visit.get(v["person_id"]),
            )
        )
        previous_visit[v["person_id"]] = v["visit_occurrence_id"]
        add_cost(
            v["visit_occurrence_id"], "Visit", v["person_id"], v["visit_start_date"],
            v.get("total_charge"), v.get("paid_by_payer"), v.get("paid_by_patient"),
        )
    conn.executemany("INSERT INTO visit_occurrence VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", visit_rows)
    counts["visit_occurrence"] = len(visit_rows)

    cond_rows = []
    for c in log.conditions:
        std = vocab.std_id[c["condition_key"]]
        # The code that was actually written on the date of service, which for a
        # pre-cutover record is the legacy classification. Storing today's code
        # against a 2012 encounter is an anachronism that no real ETL produces.
        src = vocab.src_id[(c["condition_key"], c["emitted_code"])]
        # Rank 1 is the reason for the encounter; everything after it is a
        # secondary diagnosis, which is how a real claim is read.
        primary = c.get("condition_rank", 1) == 1
        status = type_concepts["condition_status_inpatient_primary"] if primary else type_concepts["condition_status_outpatient"]
        cond_rows.append(
            (
                c["condition_occurrence_id"], c["person_id"], std, _d(c["condition_start_date"]),
                f"{_d(c['condition_start_date'])} 09:00:00", _d(c["condition_end_date"]),
                f"{_d(c['condition_end_date'])} 17:00:00" if c["condition_end_date"] else None,
                c["condition_type_concept_id"], status, None, None, c["visit_occurrence_id"], None,
                c["emitted_code"], src, None,
            )
        )
    conn.executemany(
        "INSERT INTO condition_occurrence VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", cond_rows
    )
    counts["condition_occurrence"] = len(cond_rows)

    drug_rows = []
    for d in log.drugs:
        std = vocab.std_id[d["drug_key"]]
        src = vocab.src_id[(d["drug_key"], d["pkg_code"])]
        drug_rows.append(
            (
                d["drug_exposure_id"], d["person_id"], std, _d(d["drug_exposure_start_date"]),
                f"{_d(d['drug_exposure_start_date'])} 09:00:00", _d(d["drug_exposure_end_date"]),
                f"{_d(d['drug_exposure_end_date'])} 09:00:00", None, type_concepts["drug"],
                None, d["refills"], d["quantity"], d["days_supply"], d["sig"],
                route_concept.get(d["route"], 0), None, None,
                d["visit_occurrence_id"], None, d["pkg_code"], src, d["route"], None,
            )
        )
        add_cost(
            d["drug_exposure_id"], "Drug", d["person_id"], d["drug_exposure_start_date"],
            d.get("total_charge"),
        )
    conn.executemany(
        "INSERT INTO drug_exposure VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", drug_rows
    )
    counts["drug_exposure"] = len(drug_rows)

    meas_rows = []
    for m in log.measurements:
        std = vocab.std_id[m["lab_key"]]
        src = vocab.src_id[(m["lab_key"], m["source_code"])]
        meas_rows.append(
            (
                m["measurement_id"], m["person_id"], std, _d(m["measurement_date"]),
                f"{_d(m['measurement_date'])} 10:00:00", "10:00:00",
                m["measurement_type_concept_id"], 0, m["value_as_number"], 0,
                unit_concept.get(m["unit"], 0), m["range_low"], m["range_high"], None,
                m["visit_occurrence_id"], None, m["source_code"], src, m["unit"], 0,
                # An unresulted order carries no value_source_value either; the
                # string "None" would defeat every IS NULL check downstream.
                None if m["value_as_number"] is None else str(m["value_as_number"]), None, 0,
            )
        )
        add_cost(
            m["measurement_id"], "Measurement", m["person_id"], m["measurement_date"],
            m.get("total_charge"),
        )
    conn.executemany(
        "INSERT INTO measurement VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", meas_rows
    )
    counts["measurement"] = len(meas_rows)

    proc_rows = []
    for p in log.procedures:
        std = vocab.std_id[p["procedure_key"]]
        src = vocab.src_id[(p["procedure_key"], p["source_code"])]
        proc_rows.append(
            (
                p["procedure_occurrence_id"], p["person_id"], std, _d(p["procedure_date"]),
                f"{_d(p['procedure_date'])} 11:00:00", _d(p["procedure_date"]),
                f"{_d(p['procedure_date'])} 11:30:00", p["procedure_type_concept_id"], 0, 1,
                None, p["visit_occurrence_id"], None, p["source_code"], src, None,
            )
        )
        add_cost(
            p["procedure_occurrence_id"], "Procedure", p["person_id"], p["procedure_date"],
            p.get("total_charge"),
        )
    conn.executemany(
        "INSERT INTO procedure_occurrence VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", proc_rows
    )
    counts["procedure_occurrence"] = len(proc_rows)

    obs_rows = []
    for o in log.observations:
        # Family history names a condition, so its value is the condition's own
        # standard concept - which is what makes "family history of breast
        # cancer" answerable with the same concept set as the diagnosis itself.
        value_concept = o.get("value_as_concept_id")
        if value_concept is None and o.get("value_as_concept_key"):
            value_concept = vocab.std_id.get(o["value_as_concept_key"])
        obs_rows.append(
            (
                o["observation_id"], o["person_id"], o["observation_concept_id"],
                _d(o["observation_date"]), f"{_d(o['observation_date'])} 09:00:00",
                o["observation_type_concept_id"], o.get("value_as_number"),
                o.get("value_as_string"), value_concept or 0, 0, 0, None,
                o["visit_occurrence_id"], None, o["observation_source_value"], 0, None, None,
                o.get("value_as_string"), None, 0,
            )
        )
    conn.executemany(
        "INSERT INTO observation VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", obs_rows
    )
    counts["observation"] = len(obs_rows)

    conn.executemany(
        "INSERT INTO cost VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", cost_rows
    )
    counts["cost"] = len(cost_rows)

    conn.executemany(
        "INSERT INTO fact_relationship VALUES (?,?,?,?,?)",
        [
            (f["domain_concept_id_1"], f["fact_id_1"], f["domain_concept_id_2"],
             f["fact_id_2"], f["relationship_concept_id"])
            for f in log.fact_relationships
        ],
    )
    counts["fact_relationship"] = len(log.fact_relationships)

    # cause_concept_id was a literal 0 on every row, which makes cause-specific
    # mortality - a primary endpoint in most outcomes research - unanswerable.
    # The population model picks a cause; resolving it to a concept belongs here,
    # because the writer is what owns the vocabulary.
    causes = {c["concept_name"]: c["concept_id"] for c in cat.standard_concepts["death_causes"]}
    fallback = {"natural": causes["Death from natural causes"], "accidental": causes["Accidental death"]}
    death_rows = []
    for d in log.deaths:
        key = d.get("cause_key") or "natural"
        cause_id = fallback.get(key) or vocab.std_id.get(key, fallback["natural"])
        source = key if key in fallback else cat.conditions[key].source_codes[0].code
        death_rows.append(
            (d["person_id"], _d(d["death_date"]), f"{_d(d['death_date'])} 00:00:00",
             d["death_type_concept_id"], cause_id, source, 0)
        )
    conn.executemany("INSERT INTO death VALUES (?,?,?,?,?,?,?)", death_rows)
    counts["death"] = len(death_rows)

    # Era tables: how real RWE studies define continuous exposure and disease.
    drug_eras = build_eras(log.drugs, "person_id", "drug_key", "drug_exposure_start_date", "drug_exposure_end_date")
    conn.executemany(
        "INSERT INTO drug_era VALUES (?,?,?,?,?,?,?)",
        [(e["era_id"], e["person_id"], vocab.std_id[e["key"]], _d(e["start"]), _d(e["end"]), e["count"], 0)
         for e in drug_eras],
    )
    counts["drug_era"] = len(drug_eras)

    cond_eras = build_eras(log.conditions, "person_id", "condition_key", "condition_start_date", "condition_end_date")
    conn.executemany(
        "INSERT INTO condition_era VALUES (?,?,?,?,?,?)",
        [(e["era_id"], e["person_id"], vocab.std_id[e["key"]], _d(e["start"]), _d(e["end"]), e["count"])
         for e in cond_eras],
    )
    counts["condition_era"] = len(cond_eras)

    place_concept = vocab.structural.get("places_of_service", {})
    conn.executemany(
        "INSERT INTO care_site VALUES (?,?,?,?,?,?)",
        [
            (cs["care_site_id"], cs["name"], place_concept.get(cs["place_of_service"], 0),
             cs["location_id"], cs["facility"], cs["place_of_service"])
            for cs in demo["care_sites"]
        ],
    )
    counts["care_site"] = len(demo["care_sites"])

    specialty_concept = vocab.structural.get("specialties", {})
    conn.executemany(
        "INSERT INTO provider VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (p["provider_id"], p["provider_name"], None, None,
             specialty_concept.get(p["specialty"], 0), p["care_site_id"], None, 0,
             f"PRV{p['provider_id']:04d}", p["specialty"], 0, None, 0)
            for p in log.providers
        ],
    )
    counts["provider"] = len(log.providers)

    conn.executemany("INSERT INTO concept VALUES (?,?,?,?,?,?,?,?,?,?)", vocab.concepts)
    conn.executemany(
        "INSERT INTO concept_relationship VALUES (?,?,?,?,?,?)", sorted(set(vocab.relationships))
    )
    conn.executemany("INSERT INTO concept_ancestor VALUES (?,?,?,?)", sorted(vocab.ancestor_rows()))
    counts["concept"] = len(vocab.concepts)

    from vocabulary import VOCABULARIES

    conn.executemany(
        "INSERT INTO vocabulary VALUES (?,?,?,?,?)",
        [(v[0], v[1], "Synthetic - not a real vocabulary", "v1.0", 0) for v in VOCABULARIES],
    )

    conn.execute(
        "INSERT INTO cdm_source VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            "Synthetic RWE Demo Database", "SYNTH-RWE", "Medical Coder demo",
            f"Fictitious patient-level data generated from a hand-authored clinical catalog (seed={seed}). "
            "All patients, codes and vocabularies are invented.",
            "synthetic_data_generation/catalog", "synthetic_data_generation/generate.py",
            None, None, CDM_VERSION, 0, "SIM v1.0",
        ),
    )

    return counts
