#!/usr/bin/env python3
"""Render the same event log into a messy, non-OMOP source EMR extract."""

import os
import sqlite3
from datetime import date
from typing import Dict, List, Optional, Tuple

from models import Catalog
from pathways import EventLog

MESSINESS_SCALE = {"clean": 0.0, "realistic": 1.0, "nasty": 2.2}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

DDL = [
    """CREATE TABLE FACILITY (
        FAC_CD TEXT PRIMARY KEY, FAC_NM TEXT, EHR_VENDOR TEXT, DATE_FMT TEXT, SPECIALTY TEXT)""",
    """CREATE TABLE PATIENT_MASTER (
        MRN TEXT, MEMBER_ID TEXT, PAT_NM TEXT, BIRTH_DT TEXT, SEX_CD TEXT, RACE_TXT TEXT,
        ETHNICITY_TXT TEXT, DECEASED_FL TEXT, DEATH_DT TEXT, FAC_CD TEXT)""",
    """CREATE TABLE ENCOUNTER (
        ENC_ID TEXT, MRN TEXT, ENC_TYPE_TXT TEXT, ENC_TYPE_CD TEXT, ADMIT_DT TEXT,
        DISCH_DT TEXT, LOS_DAYS INTEGER, DEPT_NM TEXT, ATTENDING_ID TEXT, FAC_CD TEXT,
        TOTAL_CHARGE_AMT TEXT)""",
    """CREATE TABLE DX (
        ENC_ID TEXT, MRN TEXT, DX_CD TEXT, DX_CD_TYPE TEXT, DX_DESC TEXT,
        DX_DT TEXT, PRIMARY_FL TEXT, DX_SEQ INTEGER, FAC_CD TEXT)""",
    """CREATE TABLE MED_ORDERS (
        ORDER_ID TEXT, ENC_ID TEXT, MRN TEXT, MED_NM TEXT, PKG_CD TEXT, DOSE_TXT TEXT,
        ROUTE TEXT, ORDER_DT TEXT, END_DT TEXT, DAYS_SUPPLY INTEGER, REFILLS INTEGER, FAC_CD TEXT)""",
    """CREATE TABLE LAB_RESULTS (
        ENC_ID TEXT, MRN TEXT, TEST_NM TEXT, LOCAL_CD TEXT, STD_LAB_CD TEXT,
        RESULT_VAL TEXT, RESULT_UNIT TEXT, REF_RANGE_TXT TEXT, ABNORMAL_FL TEXT,
        COLLECT_DT TEXT, RESULT_STATUS TEXT, FAC_CD TEXT)""",
    """CREATE TABLE PROCEDURES (
        ENC_ID TEXT, MRN TEXT, PROC_CD TEXT, PROC_DESC TEXT, PROC_DT TEXT, FAC_CD TEXT)""",
    # Social history and family history live in their own table in most EMRs
    # rather than alongside numeric results, and that is where an analyst has to
    # go looking for smoking status.
    """CREATE TABLE SOCIAL_HX (
        ENC_ID TEXT, MRN TEXT, OBS_CD TEXT, OBS_NM TEXT, OBS_VAL TEXT, OBS_NUM TEXT,
        OBS_DT TEXT, FAC_CD TEXT)""",
    """CREATE TABLE COVERAGE (
        MEMBER_ID TEXT, MRN TEXT, PAYER_CD TEXT, PLAN_CD TEXT, EFF_DT TEXT, TERM_DT TEXT)""",
]

INDEXES = [
    "CREATE INDEX idx_pm_mrn ON PATIENT_MASTER(MRN)",
    "CREATE INDEX idx_pm_member ON PATIENT_MASTER(MEMBER_ID)",
    "CREATE INDEX idx_enc_mrn ON ENCOUNTER(MRN)",
    "CREATE INDEX idx_dx_mrn ON DX(MRN)",
    "CREATE INDEX idx_dx_cd ON DX(DX_CD)",
    "CREATE INDEX idx_med_mrn ON MED_ORDERS(MRN)",
    "CREATE INDEX idx_lab_mrn ON LAB_RESULTS(MRN)",
    "CREATE INDEX idx_lab_cd ON LAB_RESULTS(LOCAL_CD)",
    "CREATE INDEX idx_proc_mrn ON PROCEDURES(MRN)",
    "CREATE INDEX idx_sh_mrn ON SOCIAL_HX(MRN)",
    "CREATE INDEX idx_cov_member ON COVERAGE(MEMBER_ID)",
]


class Messiness:
    """Facility-specific data-quality profile.

    Real heterogeneity comes from sites running different software, not from
    uniform random noise, so every defect here is keyed to a facility.
    """

    def __init__(self, facilities: List[dict], level: str, error_budget: Dict[str, float], rng) -> None:
        self.scale = MESSINESS_SCALE[level]
        self.by_code = {f["code"]: f for f in facilities}
        self.rng = rng
        # Keyed-in dates with two digits the wrong way round. This is a source
        # defect, not a model one: the OMOP rendering of the same event keeps
        # the correct date, which is what makes it findable by reconciliation.
        self.transpose_rate = error_budget.get("transposed_date_digits", 0.0)
        self.transposed = 0

    def fmt_date(self, fac: str, value: Optional[date]) -> Optional[str]:
        if value is None:
            return None
        if self.scale and self.transpose_rate and self.rng.random() < self.transpose_rate:
            swapped = _transpose_year(value)
            if swapped is not None:
                value = swapped
                self.transposed += 1
        fmt = self.by_code[fac]["date_format"] if self.scale else "%Y-%m-%d"
        if fmt == "epoch":
            return str(int((value - date(1970, 1, 1)).days * 86400))
        if fmt == "%d-%b-%Y":
            return f"{value.day:02d}-{MONTHS[value.month - 1]}-{value.year}"
        if fmt == "%m/%d/%Y":
            return f"{value.month:02d}/{value.day:02d}/{value.year}"
        return value.isoformat()

    def p(self, fac: str, key: str) -> float:
        return min(self.by_code[fac][key] * self.scale, 0.95)

    def unit(self, fac: str, unit: str, rng) -> str:
        if not self.scale:
            return unit
        style = self.by_code[fac]["unit_case"]
        if style == "lower":
            return unit.lower()
        if style == "upper":
            return unit.upper()
        return unit.lower() if rng.random() < 0.4 else unit

    def maybe_pad(self, value: Optional[str], rng) -> Optional[str]:
        # Trailing whitespace from fixed-width interface feeds.
        if value and self.scale and rng.random() < 0.03 * self.scale:
            return f"{value}  "
        return value


def _transpose_year(value: date) -> Optional[date]:
    """Swap the last two digits of the year, or None if that changes nothing.

    A transposition that still parses is the dangerous kind: 2019 keyed as 2091
    survives every format check and only fails a range check, which is exactly
    the defect a date-quality rule is supposed to catch.
    """
    year = value.year
    tens, units = (year // 10) % 10, year % 10
    if tens == units:
        return None
    swapped = year - tens * 10 - units + units * 10 + tens
    try:
        return value.replace(year=swapped)
    except ValueError:  # 29 February in a target year that is not a leap year
        return None


def write_source_sqlite(cat: Catalog, log: EventLog, db_path: str, messiness: str, rng) -> Dict[str, int]:
    if os.path.exists(db_path):
        os.remove(db_path)

    demo = cat.demographics
    mess = Messiness(demo["facilities"], messiness, cat.calendar.error_budget, rng)

    conn = sqlite3.connect(db_path)
    try:
        for stmt in DDL:
            conn.execute(stmt)
        counts = _populate(cat, log, conn, mess, rng)
        for stmt in INDEXES:
            conn.execute(stmt)
        conn.commit()
    finally:
        conn.close()
    counts["transposed_date_digits"] = mess.transposed
    return counts


def _populate(cat: Catalog, log: EventLog, conn, mess: Messiness, rng) -> Dict[str, int]:
    demo = cat.demographics
    counts: Dict[str, int] = {"duplicates_injected": 0, "nulls_injected": 0, "legacy_code_rows": 0}

    conn.executemany(
        "INSERT INTO FACILITY VALUES (?,?,?,?,?)",
        [(f["code"], f["name"], f["ehr_vendor"], f["date_format"], f["specialty"]) for f in demo["facilities"]],
    )

    person_of = {p["person_id"]: p for p in log.persons}
    death_of = {d["person_id"]: d["death_date"] for d in log.deaths}
    care_site_name = {cs["care_site_id"]: cs["name"] for cs in demo["care_sites"]}

    # MRNs are site-local. A patient seen at two facilities has two chart numbers
    # and appears twice in PATIENT_MASTER, which is the record-linkage problem
    # this extract exists to pose. Every clinical row is keyed to the MRN of the
    # facility that produced it, not to the patient's home site - keying them all
    # to the home MRN would quietly solve the very problem the multi-facility
    # model exists to pose.
    mrn_at: Dict[Tuple[int, str], str] = {(pf["person_id"], pf["facility"]): pf["mrn"] for pf in log.person_facilities}

    def key_for(pid: int, facility: str) -> Tuple[str, str]:
        """(MRN, facility) for a row, falling back to the patient's home site."""
        home = person_of[pid]["facility"]
        fac = facility or home
        return mrn_at.get((pid, fac), person_of[pid]["person_source_value"]), fac

    patient_rows = []
    for pf in log.person_facilities:
        p = person_of[pf["person_id"]]
        fac = pf["facility"]
        dod = death_of.get(p["person_id"])
        patient_rows.append(
            (
                pf["mrn"],
                p["member_id"],
                p["name"],
                mess.fmt_date(fac, p["birth_datetime"]),
                p["gender_source_value"],
                mess.maybe_pad(p["race_source_value"], rng),
                p["ethnicity_source_value"],
                "Y" if dod else "N",
                mess.fmt_date(fac, dod),
                fac,
            )
        )
    conn.executemany("INSERT INTO PATIENT_MASTER VALUES (?,?,?,?,?,?,?,?,?,?)", patient_rows)
    counts["PATIENT_MASTER"] = len(patient_rows)

    conn.executemany(
        "INSERT INTO COVERAGE VALUES (?,?,?,?,?,?)",
        [
            (
                pp["family_source_value"],
                person_of[pp["person_id"]]["person_source_value"],
                pp["payer_source_value"],
                pp["plan_source_value"],
                mess.fmt_date(person_of[pp["person_id"]]["facility"], pp["payer_plan_period_start_date"]),
                mess.fmt_date(person_of[pp["person_id"]]["facility"], pp["payer_plan_period_end_date"]),
            )
            for pp in log.payer_plan_periods
        ],
    )
    counts["COVERAGE"] = len(log.payer_plan_periods)

    enc_id_of = {}
    enc_rows = []
    for v in log.visits:
        pid = v["person_id"]
        mrn, fac = key_for(pid, v.get("facility"))
        enc_id = f"E{v['visit_occurrence_id']:09d}"
        enc_id_of[v["visit_occurrence_id"]] = enc_id
        vt = cat.visits[v["visit_concept_key"]]
        los = (v["visit_end_date"] - v["visit_start_date"]).days
        enc_rows.append(
            (
                enc_id,
                mrn,
                vt.name,
                vt.source_code,
                mess.fmt_date(fac, v["visit_start_date"]),
                mess.fmt_date(fac, v["visit_end_date"]),
                los,
                care_site_name.get(v["care_site_id"], ""),
                f"PRV{v['provider_id']:04d}",
                fac,
                f"{v.get('total_charge', 0.0):.2f}",
            )
        )
    enc_rows, dups = _inject_duplicates(enc_rows, mess, rng, fac_index=9)
    counts["duplicates_injected"] += dups
    conn.executemany("INSERT INTO ENCOUNTER VALUES (?,?,?,?,?,?,?,?,?,?,?)", enc_rows)
    counts["ENCOUNTER"] = len(enc_rows)

    dx_rows = []
    for c in log.conditions:
        pid = c["person_id"]
        mrn, fac = key_for(pid, c.get("facility"))
        dt = c["condition_start_date"]

        code, code_type = _dx_code(c, mess, rng)
        if code_type == "DX9":
            counts["legacy_code_rows"] += 1

        desc = _source_description(cat, c["condition_key"], c["source_code"])
        dx_rows.append(
            (
                # A dangling encounter reference is one of the injected defects,
                # and it has to survive into the source extract rather than
                # crashing the writer that meets it.
                enc_id_of.get(c["visit_occurrence_id"]),
                mrn,
                code,
                code_type,
                mess.maybe_pad(desc, rng),
                mess.fmt_date(fac, dt),
                "Y" if c.get("condition_rank", 1) == 1 else "N",
                c.get("condition_rank", 1),
                fac,
            )
        )
    dx_rows, dups = _inject_duplicates(dx_rows, mess, rng, fac_index=8)
    counts["duplicates_injected"] += dups
    conn.executemany("INSERT INTO DX VALUES (?,?,?,?,?,?,?,?,?)", dx_rows)
    counts["DX"] = len(dx_rows)

    med_rows = []
    for i, d in enumerate(log.drugs):
        pid = d["person_id"]
        mrn, fac = key_for(pid, d.get("facility"))
        drug = cat.drugs[d["drug_key"]]

        pkg_code = drug.pkg_code
        if rng.random() < mess.p(fac, "null_rate_pkg_code"):
            # A missing package code forces the tool to fall back to name matching.
            pkg_code = None
            counts["nulls_injected"] += 1

        med_rows.append(
            (
                f"O{i + 1:09d}",
                enc_id_of.get(d["visit_occurrence_id"]),
                mrn,
                mess.maybe_pad(drug.name, rng),
                pkg_code,
                drug.strength,
                drug.route,
                mess.fmt_date(fac, d["drug_exposure_start_date"]),
                mess.fmt_date(fac, d["drug_exposure_end_date"]),
                d["days_supply"],
                d["refills"],
                fac,
            )
        )
    med_rows, dups = _inject_duplicates(med_rows, mess, rng, fac_index=11)
    counts["duplicates_injected"] += dups
    conn.executemany("INSERT INTO MED_ORDERS VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", med_rows)
    counts["MED_ORDERS"] = len(med_rows)

    lab_rows = []
    for m in log.measurements:
        pid = m["person_id"]
        mrn, fac = key_for(pid, m.get("facility"))
        lab = cat.labs[m["lab_key"]]

        std_lab_cd = lab.std_code
        if rng.random() < mess.p(fac, "null_rate_lab_code"):
            std_lab_cd = None
            counts["nulls_injected"] += 1

        local_cd = lab.source_code
        if mess.scale and mess.by_code[fac]["lab_dialect"] == "local_mnemonic":
            local_cd = _mnemonic(m["lab_key"])

        result, abnormal, status = _result_text(lab, m["value_as_number"], mess, rng)
        lab_rows.append(
            (
                enc_id_of.get(m["visit_occurrence_id"]),
                mrn,
                lab.name,
                local_cd,
                std_lab_cd,
                result,
                mess.unit(fac, lab.unit, rng),
                f"{lab.range_low:g}-{lab.range_high:g}",
                abnormal,
                mess.fmt_date(fac, m["measurement_date"]),
                status,
                fac,
            )
        )
    lab_rows, dups = _inject_duplicates(lab_rows, mess, rng, fac_index=11)
    counts["duplicates_injected"] += dups
    conn.executemany("INSERT INTO LAB_RESULTS VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", lab_rows)
    counts["LAB_RESULTS"] = len(lab_rows)

    proc_rows = []
    for p in log.procedures:
        pid = p["person_id"]
        mrn, fac = key_for(pid, p.get("facility"))
        proc = cat.procedures[p["procedure_key"]]
        proc_rows.append(
            (
                enc_id_of.get(p["visit_occurrence_id"]),
                mrn,
                proc.source_code,
                mess.maybe_pad(proc.name, rng),
                mess.fmt_date(fac, p["procedure_date"]),
                fac,
            )
        )
    conn.executemany("INSERT INTO PROCEDURES VALUES (?,?,?,?,?,?)", proc_rows)
    counts["PROCEDURES"] = len(proc_rows)

    obs_by_id = {o.concept_id: o for o in cat.observations}
    social_rows = []
    for o in log.observations:
        pid = o["person_id"]
        mrn, fac = key_for(pid, o.get("facility"))
        item = obs_by_id.get(o["observation_concept_id"])
        social_rows.append(
            (
                enc_id_of.get(o["visit_occurrence_id"]),
                mrn,
                o["observation_source_value"],
                item.name if item else "Gestational age at delivery",
                mess.maybe_pad(o.get("value_as_string"), rng),
                None if o.get("value_as_number") is None else f"{o['value_as_number']:g}",
                mess.fmt_date(fac, o["observation_date"]),
                fac,
            )
        )
    conn.executemany("INSERT INTO SOCIAL_HX VALUES (?,?,?,?,?,?,?,?)", social_rows)
    counts["SOCIAL_HX"] = len(social_rows)

    return counts


def _dx_code(c: dict, mess: Messiness, rng) -> Tuple[str, str]:
    """Label the code the event log already chose for this date of service.

    Which classification was in force - including the straggler tail after the
    cutover - is decided once, in pathways._coded_as, so that this extract and
    the OMOP database agree on what was written down. All that is left here is
    the labelling, where the same code system appears under two names and makes
    a naive GROUP BY DX_CD_TYPE wrong.
    """
    if c["emitted_vocabulary"] == "DXCODE9":
        return c["emitted_code"], "DX9"
    if mess.scale and rng.random() < 0.25:
        return c["emitted_code"], "D10"
    return c["emitted_code"], "DX10"


def _result_text(lab, value, mess: Messiness, rng) -> Tuple[Optional[str], Optional[str], str]:
    """Format a result, preserving unresulted orders as genuinely absent values.

    An ordered-but-unresulted test has no value, no flag and no reference
    comparison - only a status. Writing 0 here, which the old floor produced,
    turns a missing result into a critical one.
    """
    if value is None:
        return None, None, "ORDERED"

    if mess.scale and value < lab.range_low * 0.5 and rng.random() < 0.3:
        result = f"<{lab.range_low}"
    elif mess.scale and value > lab.range_high * 3 and rng.random() < 0.2:
        result = f">{lab.range_high * 3:g}"
    else:
        result = f"{value:g}"

    abnormal = "H" if value > lab.range_high else ("L" if value < lab.range_low else "N")
    return result, abnormal, "FINAL"


def _inject_duplicates(rows: List[tuple], mess: Messiness, rng, fac_index: int):
    """Replay a fraction of rows, mimicking a double-posted interface feed.

    The rate is the emitting facility's own `duplicate_rate`. It used to be a flat
    0.012 for every site, so demographics.json declared per-facility rates that
    nothing read and every facility duplicated at an identical rate - which
    removes the only signal that would let anyone find the misbehaving feed.
    """
    if not mess.scale or not rows:
        return rows, 0

    extra: List[tuple] = []
    for row in rows:
        fac = row[fac_index]
        rate = min(mess.by_code[fac]["duplicate_rate"] * mess.scale, 0.5)
        if rng.random() < rate:
            extra.append(row)

    return rows + extra, len(extra)


def _source_description(cat: Catalog, condition_key: str, code: str) -> str:
    for sc in cat.conditions[condition_key].source_codes:
        if sc.code == code:
            return sc.name
    return cat.conditions[condition_key].name


def _mnemonic(lab_key: str) -> str:
    """Site-local lab mnemonic, the kind that defeats naive code joins."""
    overrides = {
        "hba1c": "GLUA1C",
        "glucose": "GLUC",
        "creatinine": "CREA",
        "hemoglobin": "HGB",
        "hematocrit": "HCT",
        "potassium": "K",
        "sodium": "NA",
        "calcium": "CA",
        "wbc": "WBC",
        "platelets": "PLT",
        "ldl": "LDLCHOL",
        "hdl": "HDLCHOL",
        "total_cholesterol": "CHOL",
        "triglycerides": "TRIG",
        "alt": "SGPT",
        "ast": "SGOT",
        "tsh": "TSH",
        "psa": "PSA",
        "inr": "INR",
        "egfr": "EGFR",
    }
    return overrides.get(lab_key, lab_key.upper()[:8])
