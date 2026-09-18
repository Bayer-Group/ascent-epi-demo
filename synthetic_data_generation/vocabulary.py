#!/usr/bin/env python3
"""Derive OMOP vocabulary tables from the clinical catalog."""

import csv
import io
import os
import zipfile
from collections import deque
from datetime import date
from typing import Dict, Iterable, List, Optional, Set, Tuple

from models import EPOCH_END, EPOCH_START, Catalog

VALID_START = 20000101
VALID_END = 20991231

# Fictitious vocabularies. DXCODE and PXCODE carry a meaningful 3-character
# prefix, which is what makes prefix-based neighbour expansion return real
# clinical siblings, which prefix-based retrieval relies on.
VOCABULARIES = [
    ("MEDLEX", "Simulated standard clinical terminology", "MEDLEX"),
    ("PHARMLEX", "Simulated drug terminology", "PHARMLEX"),
    ("LABLEX", "Simulated laboratory terminology", "LABLEX"),
    ("DXCODE", "Simulated diagnosis classification", "DXCODE"),
    ("DXCODE9", "Simulated legacy diagnosis classification", "DXCODE9"),
    ("PXCODE", "Simulated procedure classification", "PXCODE"),
    ("DRUGPKG", "Simulated drug package code directory", "DRUGPKG"),
    ("LABLOCAL", "Simulated local laboratory codes", "LABLOCAL"),
    ("ENCTYPE", "Simulated encounter type codes", "ENCTYPE"),
    # Structural OHDSI concepts are reproduced verbatim rather than invented,
    # because a person row whose gender_concept_id is 8532 is only meaningful if
    # 8532 resolves the way every OHDSI tool expects it to.
    ("Gender", "OMOP Gender", "OMOP generated"),
    ("Race", "Race and Ethnicity Code Set", "OMOP generated"),
    ("Ethnicity", "OMOP Ethnicity", "OMOP generated"),
    ("UCUM", "Unified Code for Units of Measure", "UCUM"),
    ("Type Concept", "OMOP Type Concept", "OMOP generated"),
    ("Route", "OMOP Route", "OMOP generated"),
    ("Place of Service", "Place of Service Codes for Professional Claims", "CMS"),
    ("Specialty", "OMOP Specialty", "OMOP generated"),
    ("Payer", "OMOP Payer", "OMOP generated"),
    ("SNOMED", "Systematic Nomenclature of Medicine - Clinical Terms", "SNOMED"),
    ("None", "OMOP Standardized Vocabularies", "OMOP generated"),
]

CONCEPT_COLUMNS = [
    "concept_id", "concept_name", "domain_id", "vocabulary_id", "concept_class_id",
    "standard_concept", "concept_code", "valid_start_date", "valid_end_date", "invalid_reason",
]

# concept_id blocks, kept disjoint so ids are stable and human-readable.
ID_BASE = {
    "condition_std": 20000000,
    "condition_src": 21000000,
    "drug_std": 22000000,
    "drug_src": 23000000,
    "meas_std": 24000000,
    "meas_src": 25000000,
    "proc_std": 26000000,
    "proc_src": 27000000,
    "visit_std": 28000000,
    "visit_src": 29000000,
    # Roll-up concepts for labs, procedures, visits and drugs. Held apart from the
    # leaf blocks so expansion, which fills forward from the leaf blocks, can never
    # collide with a grouper.
    "hierarchy": 19000000,
}

# Which standard_concepts.json section supplies which domain and vocabulary. The
# sentinel and the relationship/domain metadata are handled separately.
STRUCTURAL_SECTIONS = {
    "gender": ("Gender", "Gender", "Gender"),
    "race": ("Race", "Race", "Race"),
    "ethnicity": ("Ethnicity", "Ethnicity", "Ethnicity"),
    "units": ("Unit", "UCUM", "Unit"),
    "type_concepts": ("Type Concept", "Type Concept", "Type Concept"),
    "routes": ("Route", "Route", "Route"),
    "places_of_service": ("Place of Service", "Place of Service", "Place of Service"),
    "specialties": ("Provider", "Specialty", "Specialty"),
    "payer_concepts": ("Payer", "Payer", "Payer"),
    "death_causes": ("Condition", "SNOMED", "Clinical Finding"),
    "observation_concepts": ("Observation", "SNOMED", "Observable Entity"),
    "observation_values": ("Meas Value", "SNOMED", "Clinical Finding"),
    "metadata_concepts": ("Metadata", "None", "Metadata"),
}


def _omop_date(value: Optional[date], default: int) -> int:
    return int(value.strftime("%Y%m%d")) if value else default


class Vocabulary:
    """Concept tables plus the lookups the EMR generator needs."""

    def __init__(self) -> None:
        self.concepts: List[tuple] = []
        self.relationships: List[tuple] = []
        # catalog key -> concept_id for the standard concept
        self.std_id: Dict[str, int] = {}
        # (catalog key, source code) -> concept_id for source concepts
        self.src_id: Dict[Tuple[str, str], int] = {}
        # concept_id -> (code, vocabulary_id, name)
        self.by_id: Dict[int, Tuple[str, str, str]] = {}
        # Structural OHDSI lookups, e.g. structural["gender"]["F"] -> 8532.
        self.structural: Dict[str, Dict[str, int]] = {}
        # Hierarchy edges, parent -> children. concept_ancestor is derived from
        # these by transitive closure rather than written directly, because a
        # hand-written ancestor table is where grandparent rows go missing.
        self.edges: Dict[int, Set[int]] = {}
        # Every standard concept that must own a self-row in concept_ancestor.
        self.hierarchical: Set[int] = set()

    def add_concept(
        self,
        concept_id: int,
        name: str,
        domain: str,
        vocab: str,
        concept_class: str,
        standard: str,
        code: str,
        valid_start: int = VALID_START,
        valid_end: int = VALID_END,
    ) -> None:
        # Vocabulary load COPYs with QUOTE E'\b' (i.e. unquoted), so a tab or
        # newline in a name would shift every downstream column.
        clean = name.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
        self.concepts.append(
            (concept_id, clean, domain, vocab, concept_class, standard, code, valid_start, valid_end, "")
        )
        self.by_id[concept_id] = (code, vocab, clean)

    def add_mapping(self, source_id: int, standard_id: int) -> None:
        """Record the bidirectional Maps to / Mapped from pair."""
        self.relationships.append((source_id, standard_id, "Maps to", VALID_START, VALID_END, ""))
        self.relationships.append((standard_id, source_id, "Mapped from", VALID_START, VALID_END, ""))

    def add_edge(self, parent_id: int, child_id: int) -> None:
        """Declare that child rolls up to parent, one level."""
        self.edges.setdefault(parent_id, set()).add(child_id)
        self.hierarchical.add(parent_id)
        self.hierarchical.add(child_id)
        self.relationships.append((child_id, parent_id, "Is a", VALID_START, VALID_END, ""))
        self.relationships.append((parent_id, child_id, "Subsumes", VALID_START, VALID_END, ""))

    def ancestor_rows(self) -> List[Tuple[int, int, int, int]]:
        """Full transitive closure of the hierarchy, plus a self-row per concept.

        OHDSI's concept_ancestor is a materialised closure, not an edge list: a
        query for descendants of "Disorder of endocrine system" is expected to
        return type 2 diabetes even though the two are three levels apart, and a
        concept is expected to be its own ancestor at distance zero. Emitting
        only parent-child edges - which is what this module used to do - makes
        every hierarchical query silently return a fraction of its true answer,
        and the failure is invisible unless you already know the right count.
        """
        rows: Dict[Tuple[int, int], Tuple[int, int]] = {}
        # Self-rows for every standard concept, not just the ones that sit in a
        # hierarchy. `WHERE ancestor_concept_id = X` is the idiomatic way to
        # express "this concept or anything below it", and it silently returns
        # nothing for a concept with no self-row - so a unit or a payer concept
        # missing here breaks a query that is correct.
        for concept in self.concepts:
            if concept[5] == "S":
                rows[(concept[0], concept[0])] = (0, 0)
        for cid in self.hierarchical:
            rows[(cid, cid)] = (0, 0)

        # Breadth-first descent from each ancestor. min and max levels differ
        # whenever a concept is reachable by two paths of different length, which
        # a diamond in the hierarchy produces; keeping both is what lets a
        # consumer ask for "direct children only" via max_levels_of_separation.
        for ancestor in self.edges:
            queue = deque((child, 1) for child in self.edges[ancestor])
            while queue:
                node, depth = queue.popleft()
                key = (ancestor, node)
                prev = rows.get(key)
                if prev is None:
                    rows[key] = (depth, depth)
                elif depth < prev[0] or depth > prev[1]:
                    rows[key] = (min(prev[0], depth), max(prev[1], depth))
                elif depth >= prev[0] and depth <= prev[1]:
                    # Already covered at this distance; descending again would
                    # revisit the whole subtree for nothing.
                    continue
                for grandchild in self.edges.get(node, ()):
                    queue.append((grandchild, depth + 1))

        return [(a, d, lo, hi) for (a, d), (lo, hi) in rows.items()]


def build_vocabulary(cat: Catalog) -> Vocabulary:
    v = Vocabulary()

    _add_standard_concepts(cat, v)
    _add_hierarchy_groupers(cat, v)
    _add_conditions(cat, v)
    _add_drugs(cat, v)
    _add_measurements(cat, v)
    _add_procedures(cat, v)
    _add_visits(cat, v)
    _add_condition_ancestry(cat, v)

    return v


def _add_standard_concepts(cat: Catalog, v: Vocabulary) -> None:
    """Insert the OHDSI concepts the CDM tables reference structurally.

    Every *_concept_id the writers emit outside the fictitious clinical
    vocabularies - gender, race, ethnicity, unit, type, route, place of service,
    specialty, payer, cause of death, observation - used to be a dangling
    reference. `person JOIN concept ON gender_concept_id = concept_id` returned
    nothing, so the first thing any OHDSI-shaped tool did with this data failed.
    """
    raw = cat.standard_concepts

    sentinel = raw["sentinel"]
    v.add_concept(
        sentinel["concept_id"], sentinel["concept_name"], sentinel["domain_id"],
        sentinel["vocabulary_id"], sentinel["concept_class_id"],
        sentinel.get("standard_concept", ""), sentinel["concept_code"],
    )

    for section, (domain, vocab, concept_class) in STRUCTURAL_SECTIONS.items():
        lookup: Dict[str, int] = {}
        for row in raw.get(section, []):
            cid = row["concept_id"]
            v.add_concept(
                cid,
                row["concept_name"],
                row.get("domain_id", domain),
                vocab,
                concept_class,
                "S",
                row.get("concept_code", str(cid)),
            )
            # Index by every handle the writers might hold: the code, the name,
            # and the catalog-side key (unit string, route name, specialty name).
            for handle in (row.get("concept_code"), row.get("route"), row.get("specialty"), row["concept_name"]):
                if handle:
                    lookup[str(handle)] = cid
        v.structural[section] = lookup

    # The unit lookup is keyed by the UCUM string the labs actually carry.
    v.structural["units"].update(
        {row["concept_code"]: row["concept_id"] for row in raw.get("units", [])}
    )


def _add_hierarchy_groupers(cat: Catalog, v: Vocabulary) -> None:
    """Create the roll-up concepts for labs, procedures, visits and drugs."""
    offset = 0
    for domain in sorted(cat.hierarchy):
        h = cat.hierarchy[domain]
        for key in sorted(h.groupers):
            g = h.groupers[key]
            cid = ID_BASE["hierarchy"] + offset
            offset += 1
            v.std_id[f"grouper::{key}"] = cid
            v.add_concept(cid, g.name, domain, h.vocabulary, h.concept_class, "S", g.std_code)

        for key in sorted(h.groupers):
            g = h.groupers[key]
            if g.parent:
                v.add_edge(v.std_id[f"grouper::{g.parent}"], v.std_id[f"grouper::{key}"])
            else:
                # Root groupers still need a self-row, which add_edge would
                # otherwise be the only thing to register them for.
                v.hierarchical.add(v.std_id[f"grouper::{key}"])


def _attach(cat: Catalog, v: Vocabulary, domain: str, leaf_key: str, concept_id: int) -> None:
    """Hang a leaf concept off its domain grouper."""
    h = cat.hierarchy.get(domain)
    if h is None:
        return
    v.add_edge(v.std_id[f"grouper::{h.grouper_for(leaf_key)}"], concept_id)


def _add_conditions(cat: Catalog, v: Vocabulary) -> None:
    for offset, (key, g) in enumerate(sorted(cat.groupers.items())):
        cid = ID_BASE["condition_std"] + offset
        v.std_id[key] = cid
        v.add_concept(cid, g.name, "Condition", "MEDLEX", "Disorder", "S", g.std_code)

    std_offset = ID_BASE["condition_std"] + 1000
    src_offset = ID_BASE["condition_src"]
    windows = cat.calendar.vocabulary_windows

    for i, (key, c) in enumerate(sorted(cat.conditions.items())):
        std_cid = std_offset + i
        v.std_id[key] = std_cid
        # A disease concept is valid from the day the disease was recognised.
        # COVID-19 dated 2013 is the single loudest tell that a dataset was
        # fabricated, and the concept row is where that fact belongs.
        v.add_concept(
            std_cid, c.name, "Condition", "MEDLEX", "Disorder", "S", c.std_code,
            _omop_date(c.valid_from, VALID_START), _omop_date(c.valid_to, VALID_END),
        )

        for j, sc in enumerate(c.source_codes):
            src_cid = src_offset + i * 100 + j
            v.src_id[(key, sc.code)] = src_cid
            start, end = _code_window(windows, "DXCODE", c)
            v.add_concept(src_cid, sc.name, "Condition", "DXCODE", "Diagnosis Code", "", sc.code, start, end)
            v.add_mapping(src_cid, std_cid)

            # Pre-2015 records use the legacy classification, so the demo has to
            # reason across a code-system transition.
            if sc.legacy_code:
                legacy_cid = src_cid + 50
                v.src_id[(key, sc.legacy_code)] = legacy_cid
                start, end = _code_window(windows, "DXCODE9", c)
                v.add_concept(
                    legacy_cid, sc.name, "Condition", "DXCODE9", "Diagnosis Code", "",
                    sc.legacy_code, start, end,
                )
                v.add_mapping(legacy_cid, std_cid)


def _code_window(windows: Dict[str, Tuple[date, date]], vocab: str, item) -> Tuple[int, int]:
    """Intersect a vocabulary's issuance window with the concept's own validity."""
    vocab_from, vocab_to = windows.get(vocab, (EPOCH_START, EPOCH_END))
    start = max(vocab_from, item.valid_from or EPOCH_START)
    end = min(vocab_to, item.valid_to or EPOCH_END)
    return _omop_date(start, VALID_START), _omop_date(end, VALID_END)


def _add_drugs(cat: Catalog, v: Vocabulary) -> None:
    ingredients: Dict[str, int] = {}
    ing_offset = ID_BASE["drug_std"]
    drug_root = v.std_id.get("grouper::" + cat.hierarchy["Drug"].root)

    for i, name in enumerate(sorted({d.ingredient for d in cat.drugs.values()})):
        cid = ing_offset + i
        ingredients[name] = cid
        v.std_id[f"ingredient::{name}"] = cid
        v.add_concept(cid, name, "Drug", "PHARMLEX", "Substance", "S", f"9{cid}")
        if drug_root is not None:
            v.add_edge(drug_root, cid)

    std_offset = ID_BASE["drug_std"] + 1000
    src_offset = ID_BASE["drug_src"]

    for i, (key, d) in enumerate(sorted(cat.drugs.items())):
        std_cid = std_offset + i
        v.std_id[key] = std_cid
        start = _omop_date(d.valid_from, VALID_START)
        end = _omop_date(d.valid_to, VALID_END)
        v.add_concept(std_cid, d.name, "Drug", "PHARMLEX", "Drug Product", "S", d.std_code, start, end)

        src_cid = src_offset + i
        v.src_id[(key, d.pkg_code)] = src_cid
        v.add_concept(src_cid, d.name, "Drug", "DRUGPKG", "Package Code", "", d.pkg_code, start, end)
        v.add_mapping(src_cid, std_cid)

        # Ingredient rolls up the clinical drug, so "any metformin" works.
        v.add_edge(ingredients[d.ingredient], std_cid)


def _add_measurements(cat: Catalog, v: Vocabulary) -> None:
    std_offset = ID_BASE["meas_std"]
    src_offset = ID_BASE["meas_src"]

    for i, (key, lab) in enumerate(sorted(cat.labs.items())):
        std_cid = std_offset + i
        v.std_id[key] = std_cid
        v.add_concept(std_cid, lab.name, "Measurement", "LABLEX", "Lab Test", "S", lab.std_code)
        _attach(cat, v, "Measurement", key, std_cid)

        src_cid = src_offset + i
        v.src_id[(key, lab.source_code)] = src_cid
        v.add_concept(src_cid, lab.name, "Measurement", "LABLOCAL", "Local Lab Code", "", lab.source_code)
        v.add_mapping(src_cid, std_cid)


def _add_procedures(cat: Catalog, v: Vocabulary) -> None:
    std_offset = ID_BASE["proc_std"]
    src_offset = ID_BASE["proc_src"]
    windows = cat.calendar.vocabulary_windows

    for i, (key, p) in enumerate(sorted(cat.procedures.items())):
        std_cid = std_offset + i
        v.std_id[key] = std_cid
        v.add_concept(
            std_cid, p.name, "Procedure", "MEDLEX", "Procedure", "S", p.std_code,
            _omop_date(p.valid_from, VALID_START), _omop_date(p.valid_to, VALID_END),
        )
        _attach(cat, v, "Procedure", key, std_cid)

        src_cid = src_offset + i
        v.src_id[(key, p.source_code)] = src_cid
        start, end = _code_window(windows, "PXCODE", p)
        v.add_concept(src_cid, p.name, "Procedure", "PXCODE", "Procedure Code", "", p.source_code, start, end)
        v.add_mapping(src_cid, std_cid)


def _add_visits(cat: Catalog, v: Vocabulary) -> None:
    std_offset = ID_BASE["visit_std"]
    src_offset = ID_BASE["visit_src"]

    for i, (key, vt) in enumerate(sorted(cat.visits.items())):
        std_cid = std_offset + i
        v.std_id[key] = std_cid
        v.add_concept(std_cid, vt.name, "Visit", "MEDLEX", "Visit", "S", vt.std_code)
        _attach(cat, v, "Visit", key, std_cid)

        src_cid = src_offset + i
        v.src_id[(key, vt.source_code)] = src_cid
        v.add_concept(src_cid, vt.name, "Visit", "ENCTYPE", "Visit Code", "", vt.source_code)
        v.add_mapping(src_cid, std_cid)


def _add_condition_ancestry(cat: Catalog, v: Vocabulary) -> None:
    """Declare each condition's edge to its parent; the closure is derived later."""

    def parent_of(key: str) -> Optional[str]:
        if key in cat.conditions:
            return cat.conditions[key].parent
        return cat.groupers[key].parent if key in cat.groupers else None

    for key in list(cat.conditions) + list(cat.groupers):
        cid = v.std_id[key]
        v.hierarchical.add(cid)
        parent = parent_of(key)
        if parent:
            v.add_edge(v.std_id[parent], cid)


def write_athena_zips(v: Vocabulary, out_dir: str) -> Dict[str, str]:
    """Write vocabulary ZIPs that an OMOP loader can COPY without changes."""
    written = {}

    written["concept"] = _write_zip(
        os.path.join(out_dir, "CONCEPT.csv.zip"), "CONCEPT.csv", CONCEPT_COLUMNS, sorted(v.concepts)
    )
    written["concept_relationship"] = _write_zip(
        os.path.join(out_dir, "CONCEPT_RELATIONSHIP.csv.zip"),
        "CONCEPT_RELATIONSHIP.csv",
        ["concept_id_1", "concept_id_2", "relationship_id", "valid_start_date", "valid_end_date", "invalid_reason"],
        sorted(set(v.relationships)),
    )
    written["concept_ancestor"] = _write_zip(
        os.path.join(out_dir, "CONCEPT_ANCESTOR.csv.zip"),
        "CONCEPT_ANCESTOR.csv",
        ["ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation", "max_levels_of_separation"],
        sorted(v.ancestor_rows()),
    )
    return written


def _write_zip(zip_path: str, member: str, columns: List[str], rows: Iterable[tuple]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar=None)
    writer.writerow(columns)
    writer.writerows(rows)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(member, buf.getvalue())
    return zip_path
