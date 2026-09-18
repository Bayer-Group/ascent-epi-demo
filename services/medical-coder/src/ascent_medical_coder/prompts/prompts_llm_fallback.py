"""
Prompt templates and utilities for LLM fallback (medical concepts and drug search).
"""
# ruff: noqa: E501  -- prompt text is copied verbatim (long lines preserved)

from __future__ import annotations

MEDICAL_CONCEPT_PROMPT = """You are a medical terminology expert. Search for medical concepts related to the query below using Google Search to find the most accurate and up-to-date information from medical databases and resources.

Query: "{search_text}"

Filters:
{filters_text}

Return up to {top_k} relevant medical concepts in JSON format. Each concept should include:
- CONCEPT_NAME: The official medical term name
- CONCEPT_ID: A unique identifier (use a placeholder like 0 if not found)
- DOMAIN_ID: The domain (e.g., "Condition", "Procedure", "Drug", "Observation")
- VOCABULARY_ID: The vocabulary system (e.g., "SNOMED", "ICD10CM", "RxNorm")
- STANDARD_CONCEPT: Whether it's a standard concept ("S" for standard, null otherwise)
- CONCEPT_CODE: The official code in the vocabulary system. Do not return 0 always return meaningful data
- score: Relevance score (0.0 to 1.0)

Focus on finding concepts from standard medical terminologies like SNOMED CT, ICD-10, RxNorm, LOINC, etc.
{custom_instructions_text}
Return ONLY a JSON array of concept objects, nothing else. Example format:
[
  {{
    "CONCEPT_NAME": "Type 2 diabetes mellitus",
    "CONCEPT_ID": 0,
    "DOMAIN_ID": "Condition",
    "VOCABULARY_ID": "SNOMED",
    "STANDARD_CONCEPT": "S",
    "CONCEPT_CODE": "44054006",
    "score": 0.95
  }}
]
"""

DRUG_SEARCH_PROMPT = """You are a pharmaceutical expert. Search for drug concepts related to the query below using Google Search to find the most accurate and up-to-date information from drug databases and resources.

Query: "{search_text}"
Search Type: {search_type}

Filters:
{filters_text}

Return up to {top_k} relevant drug concepts in JSON format. Each concept should include:
- CONCEPT_NAME: The official drug name
- CONCEPT_ID: A unique identifier (use a placeholder like 0 if not found)
- DOMAIN_ID: Always "Drug" for drug concepts
- VOCABULARY_ID: The vocabulary system (e.g., "RxNorm", "NDC", "ATC")
- STANDARD_CONCEPT: Whether it's a standard concept ("S" for standard, null otherwise)
- CONCEPT_CODE: The official code in the vocabulary system, formatted per the rules below
- RELEVANCE_SCORE: Relevance score (0.0 to 1.0)

CONCEPT_CODE FORMAT RULES (HARD RULES):
- NDC: return 11-digit, no-dash, package-level codes only (e.g. "50419047505"). Do NOT return product-level 2-segment NDCs like "50419-475" or "50419475". Emit one row per real package SKU listed in the FDA NDC Directory or DailyMed, padding with leading zeros as needed (5-digit labeler + 4-digit product + 2-digit package).
- RxNorm: return the numeric RxCUI as a string (e.g. "2725516"). Do not include text labels.
- ATC: return the 7-character ATC code (e.g. "G02CX07") only when ATC is the requested vocabulary AND the code is published in the WHOCC ATC/DDD index. Do not invent ATC codes for newly approved drugs that have not yet been assigned one.

Do NOT return CONCEPT_CODE values of "0", empty strings, or fabricated identifiers. If you cannot find a real, verifiable code in the requested vocabulary, omit that row.

Focus on finding concepts from standard drug terminologies like RxNorm, NDC, ATC, etc.
{drug_class_note}
{custom_instructions_text}
Return ONLY a JSON array of concept objects, nothing else. Example format:
[
  {{
    "CONCEPT_NAME": "Metformin hydrochloride 500 MG Oral Tablet",
    "CONCEPT_ID": 0,
    "DOMAIN_ID": "Drug",
    "VOCABULARY_ID": "RxNorm",
    "STANDARD_CONCEPT": "S",
    "CONCEPT_CODE": "860975",
    "RELEVANCE_SCORE": 0.95
  }}
]
"""


def get_prompt(
    prompt_type: str,
    search_text: str,
    domain_id: str | None = None,
    vocabulary: str | None = None,
    standard_concept: str | None = None,
    top_k: int = 10,
    is_drug_class: bool = False,
    custom_instructions: str | None = None,
) -> str:
    """
    Returns a formatted prompt for the given type and parameters.
    prompt_type: 'medical_concept' or 'drug_search'
    """
    filters = []
    if domain_id:
        filters.append(f"- Domain: {domain_id}")
    if vocabulary:
        filters.append(f"- Vocabulary: {vocabulary}")
    if standard_concept:
        filters.append(f"- Standard Concept: {standard_concept}")
    filters_text = "\n".join(filters) if filters else "No specific filters"

    if custom_instructions:
        custom_instructions_text = (
            f"\nIMPORTANT — You MUST also follow these additional instructions "
            f"from the user when selecting concepts:\n{custom_instructions}\n"
        )
    else:
        custom_instructions_text = ""

    if prompt_type == "medical_concept":
        return MEDICAL_CONCEPT_PROMPT.format(
            search_text=search_text,
            filters_text=filters_text,
            top_k=top_k,
            custom_instructions_text=custom_instructions_text,
        )
    elif prompt_type == "drug_search":
        search_type = "drug class" if is_drug_class else "drug name"
        drug_class_note = (
            "If searching by drug class, include drugs that belong to this class." if is_drug_class else ""
        )
        return DRUG_SEARCH_PROMPT.format(
            search_text=search_text,
            search_type=search_type,
            filters_text=filters_text,
            top_k=top_k,
            drug_class_note=drug_class_note,
            custom_instructions_text=custom_instructions_text,
        )
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")
