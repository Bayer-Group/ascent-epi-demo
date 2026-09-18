"""OMOP table/column detail maps (ported verbatim from ``table_details.py``).

``table_details`` maps a domain (or a tuple of domain aliases) to the OMOP
table plus its concept / source-concept columns. ``vocabulary_domain_mapping``
maps a ``VOCABULARY_ID`` to its default OMOP domain. Both are consumed by the
patient-count and Snowflake-concept lookups.
"""

from __future__ import annotations

table_details = {
    "condition": {
        "table_name": "condition_occurrence",
        "concept_column": "condition_concept_id",
        "source_concept_column": "condition_source_concept_id",
    },
    ("drug", "drug_class"): {
        "table_name": "drug_exposure",
        "concept_column": "drug_concept_id",
        "source_concept_column": "drug_source_concept_id",
    },
    "procedure": {
        "table_name": "procedure_occurrence",
        "concept_column": "procedure_concept_id",
        "source_concept_column": "procedure_source_concept_id",
    },
    "observation": {
        "table_name": "observation",
        "concept_column": "observation_concept_id",
        "source_concept_column": "observation_source_concept_id",
    },
    "measurement": {
        "table_name": "measurement",
        "concept_column": "measurement_concept_id",
        "source_concept_column": "measurement_source_concept_id",
    },
    "device": {
        "table_name": "device_exposure",
        "concept_column": "device_concept_id",
        "source_concept_column": "device_source_concept_id",
    },
    "visit": {
        "table_name": "visit_occurrence",
        "concept_column": "visit_concept_id",
        "source_concept_column": "visit_source_concept_id",
    },
    "visit_detail": {
        "table_name": "visit_detail",
        "concept_column": "visit_detail_concept_id",
        "source_concept_column": "visit_detail_source_concept_id",
    },
    "specimen": {
        "table_name": "specimen",
        "concept_column": "specimen_concept_id",
        "source_concept_column": "specimen_source_concept_id",
    },
    "death": {
        "table_name": "death",
        "concept_column": "cause_concept_id",
        "source_concept_column": "cause_source_concept_id",
    },
    "provider": {
        "table_name": "provider",
        "concept_column": "specialty_concept_id",
        "source_concept_column": "specialty_source_concept_id",
    },
}

vocabulary_domain_mapping = {
    "SNOMED": "condition",
    "ICD10CM": "condition",
    "ICD9CM": "condition",
    "RxNorm": "drug",
    "NDC": "drug",
    "ATC": "drug",
    "CPT4": "procedure",
    "HCPCS": "procedure",
    "ICD10PCS": "procedure",
    "ICD9Proc": "procedure",
    "LOINC": "measurement",
    "UCUM": "measurement",
    "CVX": "drug",
    "Device Type": "device",
    "Visit": "visit",
    "Visit Type": "visit",
    "Death Type": "death",
}
