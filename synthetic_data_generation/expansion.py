#!/usr/bin/env python3
"""Expand each code family with unused sibling codes.

The EMR only ever uses the curated catalog, but a code-lookup or semantic-search
tool needs enough density that finding the right concept is a real task. These
extra concepts share the family prefix of the condition they hang off, so
prefix-based neighbour expansion returns a plausible differential rather than a
handful of codes.
"""

from typing import List

from models import Catalog
from vocabulary import VALID_END, VALID_START, Vocabulary, _code_window, _omop_date

# Clinical qualifiers that generate genuinely distinct sibling concepts.
LATERALITY = ["right", "left", "bilateral", "unspecified side"]
SEVERITY = ["mild", "moderate", "severe"]
EPISODE = ["initial encounter", "subsequent encounter", "sequela"]

CONDITION_VARIANTS = {
    "t2dm": [
        "with hyperosmolarity", "with ketoacidosis", "with diabetic cataract",
        "with diabetic peripheral angiopathy", "with diabetic dermatitis",
        "with diabetic arthropathy", "with hypoglycemia", "with other specified complication",
    ],
    "ckd": ["with hypertension", "unspecified", "following acute kidney injury"],
    "heart_failure": ["acute on chronic", "acute", "chronic", "unspecified"],
    "afib": ["with rapid ventricular response", "longstanding persistent", "typical atrial flutter"],
    "copd": ["with emphysema", "with chronic bronchitis", "with bronchiectasis"],
    "asthma": ["with status asthmaticus", "cough variant", "exercise induced"],
    "stroke": ["of anterior cerebral artery", "of middle cerebral artery", "of posterior cerebral artery",
               "of cerebellar artery", "of basilar artery"],
    "mi": ["of lateral wall", "of posterior wall", "subsequent", "of unspecified site"],
    "pneumonia": ["due to Klebsiella", "due to Haemophilus influenzae", "aspiration", "viral", "ventilator associated"],
    "sepsis": ["due to Escherichia coli", "due to Staphylococcus aureus", "due to Streptococcus", "puerperal"],
    "anemia": ["due to chronic blood loss", "megaloblastic", "aplastic", "hemolytic", "sideroblastic"],
    "depression": ["in remission", "with psychotic features", "with anxious distress", "postpartum"],
    "osteoarthritis": ["of shoulder", "of ankle", "of wrist", "of spine", "of hand"],
    "chronic_pain": ["of shoulder", "of hip", "of knee", "neuropathic", "central"],
    "cirrhosis": ["with ascites", "with hepatic encephalopathy", "with portal hypertension", "biliary"],
    "lung_cancer": ["of main bronchus", "of middle lobe", "overlapping sites", "metastatic to brain",
                    "metastatic to bone"],
    "breast_cancer": ["of central portion", "of lower-outer quadrant", "of nipple and areola",
                      "metastatic to bone", "inflammatory"],
    "prostate_cancer": ["metastatic to bone", "castration resistant", "in situ"],
    "colorectal_cancer": ["of transverse colon", "of descending colon", "of rectosigmoid junction",
                          "of rectum", "metastatic to liver"],
    "dementia": ["with behavioral disturbance", "with early onset", "frontotemporal", "with Lewy bodies"],
    "hypertension": ["secondary", "renovascular", "resistant", "with chronic kidney disease"],
    "hypothyroidism": ["postprocedural", "subclinical", "congenital", "drug induced"],
    "gerd": ["with hemorrhage", "with stricture", "with Barrett esophagus"],
    "obesity": ["class II", "class III", "due to excess calories", "hypothalamic"],
    "aki": ["with medullary necrosis", "with cortical necrosis", "prerenal", "postrenal"],
    "smoking": ["in remission", "with withdrawal", "vaping related"],
    "anxiety": ["social phobia", "agoraphobia", "specific phobia", "with panic attacks"],
    "osteoporosis": ["with vertebral fracture", "with hip fracture", "drug induced", "postmenopausal"],
    "covid19": ["with acute respiratory failure", "post-acute sequelae", "asymptomatic", "suspected"],
    "cad": ["of bypass graft", "with angina pectoris", "with unstable angina", "of transplanted heart"],
    "prediabetes": ["with impaired glucose tolerance", "gestational history"],
    "t1dm": ["with hypoglycemia", "with diabetic nephropathy", "with diabetic retinopathy",
             "with hyperglycemia", "brittle"],
}


def expand_vocabulary(cat: Catalog, v: Vocabulary, rng) -> int:
    """Add sibling condition/drug/lab concepts; returns how many were added."""
    before = len(v.concepts)

    _expand_conditions(cat, v)
    _expand_drugs(cat, v)
    _expand_labs(cat, v)
    _expand_procedures(cat, v)

    return len(v.concepts) - before


def _next_ids(v: Vocabulary, block: int) -> int:
    """First free id at or above a block start, so expansion never collides."""
    used = {c[0] for c in v.concepts if block <= c[0] < block + 1000000}
    return max(used) + 1 if used else block


def _expand_conditions(cat: Catalog, v: Vocabulary) -> None:
    std_id = _next_ids(v, 20000000) + 5000
    src_id = _next_ids(v, 21000000) + 5000
    used_codes = {c[6] for c in v.concepts if c[3] == "DXCODE"}
    windows = cat.calendar.vocabulary_windows

    for key in sorted(cat.conditions):
        cond = cat.conditions[key]
        variants = CONDITION_VARIANTS.get(key, [])
        if not variants:
            continue

        # Share the family prefix so 3-character recall picks these up.
        family = cond.source_codes[0].code.split(".")[0]
        suffix = 20

        # Cross the clinical variants with severity/laterality the way a real
        # classification does, so each family has genuine coding depth.
        expanded: List[str] = list(variants)
        for variant in variants[:4]:
            for qualifier in SEVERITY:
                expanded.append(f"{variant}, {qualifier}")
        if any(w in cond.name.lower() for w in ("osteoarthritis", "fracture", "cancer", "neoplasm")):
            for variant in variants[:3]:
                for side in LATERALITY[:3]:
                    expanded.append(f"{variant}, {side}")
        if not cond.chronic:
            for episode in EPISODE:
                expanded.append(episode)

        for variant in expanded:
            code = f"{family}.{suffix}"
            while code in used_codes:
                suffix += 1
                code = f"{family}.{suffix}"
            used_codes.add(code)
            suffix += 1

            name = f"{cond.name} {variant}"
            # A variant of a disease cannot predate the disease: "COVID-19 with
            # acute respiratory failure, severe" inherits the 2020 window.
            v.add_concept(
                std_id, name, "Condition", "MEDLEX", "Disorder", "S",
                str(3900000 + std_id % 100000),
                _omop_date(cond.valid_from, VALID_START), _omop_date(cond.valid_to, VALID_END),
            )
            start, end = _code_window(windows, "DXCODE", cond)
            v.add_concept(src_id, name, "Condition", "DXCODE", "Diagnosis Code", "", code, start, end)
            v.add_mapping(src_id, std_id)
            # Roll up to the curated condition itself, not to its parent: the
            # variant genuinely is a kind of the condition, and hanging it a level
            # too high is what made "descendants of type 2 diabetes" miss every
            # complication code the expansion had just created.
            v.add_edge(v.std_id[key], std_id)
            std_id += 1
            src_id += 1


def _expand_drugs(cat: Catalog, v: Vocabulary) -> None:
    std_id = _next_ids(v, 22000000) + 5000
    src_id = _next_ids(v, 23000000) + 5000
    strengths = ["5 mg", "10 mg", "25 mg", "50 mg", "100 mg", "500 mg"]
    forms = ["Oral Tablet", "Oral Capsule", "Extended Release Oral Tablet"]

    existing = {d.name for d in cat.drugs.values()}
    pkg_seq = 5000

    for ingredient in sorted({d.ingredient for d in cat.drugs.values()}):
        for strength in strengths:
            for form in forms[:3]:
                name = f"{ingredient} {strength} {form}"
                if name in existing:
                    continue
                existing.add(name)
                pkg_code = f"99999-{pkg_seq:04d}-30"
                pkg_seq += 1

                v.add_concept(std_id, name, "Drug", "PHARMLEX", "Drug Product", "S", str(490000 + std_id % 100000))
                v.add_concept(src_id, name, "Drug", "DRUGPKG", "Package Code", "", pkg_code)
                v.add_mapping(src_id, std_id)
                ing_key = f"ingredient::{ingredient}"
                if ing_key in v.std_id:
                    v.add_edge(v.std_id[ing_key], std_id)
                std_id += 1
                src_id += 1


def _expand_labs(cat: Catalog, v: Vocabulary) -> None:
    """Cross each analyte with only the specimens and methods it can be run on.

    The previous version crossed every analyte with all five specimens and both
    methods unconditionally, which manufactured concepts like "Systolic blood
    pressure, cerebrospinal fluid, immunoassay" and "Body mass index, urine,
    automated". Nonsense concepts are worse than absent ones for a search demo:
    they are exactly the kind of near-miss a retrieval model happily ranks first,
    and no reviewer trusts a vocabulary once they have seen one. The permitted
    axes now live on the analyte itself (labs.json specimens/methods), and an
    analyte that takes no variant on an axis - a vital sign has no specimen -
    simply produces no rows for it.
    """
    std_id = _next_ids(v, 24000000) + 5000
    src_id = _next_ids(v, 25000000) + 5000

    existing = {lab.name for lab in cat.labs.values()}
    seq = 5000

    for key in sorted(cat.labs):
        lab = cat.labs[key]
        # An empty axis still yields one pass with no qualifier, so an analyte
        # with methods but no specimens keeps its method variants.
        specimens = lab.specimens or [""]
        methods = lab.methods or [""]
        if not lab.specimens and not lab.methods:
            continue

        for specimen in specimens:
            for method in methods:
                qualifiers = ", ".join(q for q in (specimen, method) if q)
                name = f"{lab.name}, {qualifiers}"
                if name in existing:
                    continue
                existing.add(name)
                v.add_concept(std_id, name, "Measurement", "LABLEX", "Lab Test", "S", f"LX{40000 + seq:05d}")
                v.add_concept(src_id, name, "Measurement", "LABLOCAL", "Local Lab Code", "", f"LC{200000 + seq}")
                v.add_mapping(src_id, std_id)
                # A specimen variant is a kind of the analyte, so a phenotype
                # written against "Creatinine" also picks up the plasma variant.
                v.add_edge(v.std_id[key], std_id)
                std_id += 1
                src_id += 1
                seq += 1


def _expand_procedures(cat: Catalog, v: Vocabulary) -> None:
    std_id = _next_ids(v, 26000000) + 5000
    src_id = _next_ids(v, 27000000) + 5000
    approaches = ["open approach", "percutaneous approach", "endoscopic approach", "robotic assisted"]
    # Laterality only means something for a paired structure. A "Colonoscopy,
    # left" is the procedural equivalent of a cerebrospinal blood pressure.
    PAIRED = ("knee", "breast", "mastectomy", "carotid", "lung", "renal", "hip", "shoulder")
    windows = cat.calendar.vocabulary_windows

    existing = {p.name for p in cat.procedures.values()}
    seq = 5000

    for key in sorted(cat.procedures):
        proc = cat.procedures[key]
        family = proc.source_code[:3]
        sides = ("", " right", " left") if any(w in proc.name.lower() for w in PAIRED) else ("",)
        start, end = _code_window(windows, "PXCODE", proc)
        for approach in approaches:
            for side in sides:
                name = f"{proc.name}{side}, {approach}"
                if name in existing:
                    continue
                existing.add(name)
                v.add_concept(
                    std_id, name, "Procedure", "MEDLEX", "Procedure", "S", str(3600000 + seq),
                    _omop_date(proc.valid_from, VALID_START), _omop_date(proc.valid_to, VALID_END),
                )
                v.add_concept(
                    src_id, name, "Procedure", "PXCODE", "Procedure Code", "",
                    f"{family}{seq:04d}", start, end,
                )
                v.add_mapping(src_id, std_id)
                v.add_edge(v.std_id[key], std_id)
                std_id += 1
                src_id += 1
                seq += 1
