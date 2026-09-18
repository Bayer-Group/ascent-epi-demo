"""Pure request-normalization rules for the medical-coding pipeline."""

from __future__ import annotations

DRUG_DOMAINS = {"drug", "drug_class"}

_MIXED_DOMAIN_ERROR = (
    "You cannot mix drug-related domains ('drug', 'drug_class') "
    "with non-drug/general domains in a single request. "
    "Submit separate requests for drug and non-drug domains."
)


def normalize_domain_id(domain_id: str | None) -> str | None:
    """Normalize a domain identifier to lowercase; preserve an absent value."""
    if domain_id is None:
        return None
    normalized = domain_id.strip().lower()
    return normalized or None


def validate_mixed_domains(normalized: list[str | None]) -> dict | None:
    """Describe an invalid mix of drug and non-drug domains, if present."""
    drug_like = [domain for domain in normalized if domain in DRUG_DOMAINS]
    non_drug_like = [domain for domain in normalized if domain not in DRUG_DOMAINS]
    if not drug_like or not non_drug_like:
        return None
    return {
        "error": "MixedDomainIdsNotAllowed",
        "message": _MIXED_DOMAIN_ERROR,
        "received": {
            "drug_like": drug_like,
            "non_drug_like": non_drug_like,
        },
    }
