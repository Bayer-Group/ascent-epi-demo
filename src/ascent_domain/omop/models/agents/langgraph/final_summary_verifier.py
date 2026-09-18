"""
Final summary claim verification module.

This module provides tools to verify claims in the final summary that combines
both literature and EHR sources. It leverages the existing claim verification
infrastructure to ensure all claims are properly attributed.
"""

import logging
from typing import Any, Dict, List

from ascent_domain.omop.models.agents.langgraph.factfinder_tools import LocalClaimProcessor, refine_summary_with_claim_verdicts
from ascent_domain.omop.models.agents.langgraph.reference_manager import (
    convert_references_to_doc_indices,
    extract_claims_with_references,
    format_reference_list,
    validate_references,
)

logger = logging.getLogger(__name__)


async def extract_claims_from_summary(
    summary_text: str,
    reference_mapping: Dict[str, int]
) -> Dict[str, List[int]]:
    """
    Extract claims from final summary with their referenced document indices.

    This adapts the claim extraction process to work with [L#]/[A#] reference
    markers instead of plain [#] markers.

    Args:
        summary_text: The final summary text with [L#] and [A#] references
        reference_mapping: Maps reference markers to document indices

    Returns:
        Dictionary mapping claim text to list of document indices
        Example: {
            "Diabetes affects 10% of adults": [0, 1],  # Refs [L1][L2]
            "Database contains 1,234 patients": [5]     # Ref [A1]
        }
    """
    # Extract claims with their reference markers
    claims_with_markers = extract_claims_with_references(summary_text)

    # Convert reference markers to document indices
    claims_with_doc_indices = {}
    for claim_text, ref_markers in claims_with_markers.items():
        doc_indices = convert_references_to_doc_indices(ref_markers, reference_mapping)
        if doc_indices:  # Only include claims that have valid references
            claims_with_doc_indices[claim_text] = doc_indices
        else:
            logger.warning(f"Claim has no valid document references: {claim_text[:100]}...")

    logger.info(f"Extracted {len(claims_with_doc_indices)} claims with valid references from summary")
    return claims_with_doc_indices


async def verify_final_summary_claims(
    final_summary: str,
    unified_documents: List[Dict[str, Any]],
    reference_mapping: Dict[str, int],
    include_partial: bool = False
) -> Dict[str, Any]:
    """
    Verify all claims in the final summary against unified documents.

    This is the core function that applies the existing claim verification
    pipeline to the final summary. It works with both literature and EHR sources.

    Args:
        final_summary: The final summary text with [L#] and [A#] references
        unified_documents: Combined list of literature docs and EHR pseudo-docs
        reference_mapping: Maps [L#]/[A#] markers to document indices
        include_partial: Whether to include partially attributable claims

    Returns:
        Dictionary with verification results:
        {
            "original_summary": str,
            "total_claims": int,
            "verified_claims": List[Dict],
            "excluded_claims": List[Dict],
            "failed_verifications": List[Dict],
            "validation": Dict  # Reference validation results
        }
    """
    logger.info("Starting final summary claim verification...")

    # Step 1: Validate references in summary
    validation = validate_references(final_summary, reference_mapping)
    if not validation["valid"]:
        logger.warning(f"Summary contains invalid references: {validation['invalid_references']}")

    # Step 2: Extract claims with their referenced documents
    claims_with_doc_indices = await extract_claims_from_summary(
        final_summary, reference_mapping
    )

    if not claims_with_doc_indices:
        logger.warning("No claims with valid references found in summary")
        return {
            "original_summary": final_summary,
            "total_claims": 0,
            "verified_claims": [],
            "excluded_claims": [],
            "failed_verifications": [],
            "validation": validation
        }

    logger.info(f"Extracted {len(claims_with_doc_indices)} claims for verification")

    # Step 3: Verify claims using existing infrastructure
    # This is where we REUSE the existing batch_verify_claims function!
    processor = LocalClaimProcessor()

    attribution_results = await processor.batch_verify_claims(
        claims_with_docs=claims_with_doc_indices,
        documents=unified_documents,
        max_concurrent=5
    )

    # Step 4: Categorize claims by attribution
    verified_claims = []
    excluded_claims = []
    failed_verifications = []

    for claim, attribution_info in attribution_results.items():
        label = attribution_info.get("label", "UNKNOWN")

        claim_data = {
            "claim": claim,
            "attribution": label,
            "explanation": attribution_info.get("explanation", ""),
            "references": attribution_info.get("references", [])
        }

        if label == "ATTRIBUTABLE":
            verified_claims.append(claim_data)
        elif label == "PARTIALLY_ATTRIBUTABLE" and include_partial:
            verified_claims.append(claim_data)
        elif label == "FAILED_VERIFICATION":
            failed_verifications.append(claim_data)
        else:
            excluded_claims.append(claim_data)

    result = {
        "original_summary": final_summary,
        "total_claims": len(claims_with_doc_indices),
        "verified_claims": verified_claims,
        "excluded_claims": excluded_claims,
        "failed_verifications": failed_verifications,
        "validation": validation
    }

    logger.info(f"Final summary verification complete: {len(verified_claims)} verified, "
                f"{len(excluded_claims)} excluded, {len(failed_verifications)} failed")

    return result


async def refine_final_summary(
    verification_result: Dict[str, Any],
    unified_documents: List[Dict[str, Any]],
    reference_mapping: Dict[str, int]
) -> Dict[str, Any]:
    """
    Refine the final summary based on claim verification results.

    This applies the existing summary refinement logic to the final summary.

    Args:
        verification_result: Results from verify_final_summary_claims
        unified_documents: Combined document list
        reference_mapping: Reference mapping

    Returns:
        Dictionary with refined summary and statistics
    """
    logger.info("Refining final summary based on verification results...")

    # Reuse existing refinement function
    refinement_result = await refine_summary_with_claim_verdicts(
        original_summary=verification_result["original_summary"],
        claims_data=verification_result
    )

    # Add reference list to refined summary
    used_refs = extract_claims_with_references(refinement_result["refined_summary"])
    all_used_markers = set()
    for markers in used_refs.values():
        all_used_markers.update(markers)

    reference_list = format_reference_list(
        unified_docs=unified_documents,
        reference_mapping=reference_mapping,
        used_references=list(all_used_markers)
    )

    # Append reference list to refined summary
    refined_with_refs = f"{refinement_result['refined_summary']}\n\n{reference_list}"
    refinement_result["refined_summary"] = refined_with_refs
    refinement_result["reference_list"] = reference_list
    refinement_result["used_references"] = list(all_used_markers)

    logger.info("Final summary refinement complete")
    return refinement_result


async def process_final_summary_with_verification(
    final_summary: str,
    unified_documents: List[Dict[str, Any]],
    reference_mapping: Dict[str, int],
    include_partial: bool = False
) -> Dict[str, Any]:
    """
    Complete pipeline: verify and refine final summary.

    This is a convenience function that combines verification and refinement
    into a single call.

    Args:
        final_summary: Summary text to verify
        unified_documents: Combined document list
        reference_mapping: Reference mapping
        include_partial: Whether to include partially attributable claims

    Returns:
        Complete results including verification and refinement data
    """
    # Verify claims
    verification_result = await verify_final_summary_claims(
        final_summary=final_summary,
        unified_documents=unified_documents,
        reference_mapping=reference_mapping,
        include_partial=include_partial
    )

    # Refine summary
    refinement_result = await refine_final_summary(
        verification_result=verification_result,
        unified_documents=unified_documents,
        reference_mapping=reference_mapping
    )

    # Combine results
    complete_result = {
        "verification": verification_result,
        "refinement": refinement_result,
        "final_summary_verified": refinement_result["refined_summary"],
        "statistics": {
            "total_claims": verification_result["total_claims"],
            "verified_claims": len(verification_result["verified_claims"]),
            "excluded_claims": len(verification_result["excluded_claims"]),
            "failed_verifications": len(verification_result["failed_verifications"]),
            "kept_claims": refinement_result["kept_claims"],
            "removed_claims": refinement_result["removed_claims"],
            "modified_claims": refinement_result["modified_claims"]
        }
    }

    return complete_result


def prepare_final_summary_for_display(result: Dict[str, Any]) -> str:
    """
    Format final summary verification results for display.

    Args:
        result: Complete result from process_final_summary_with_verification

    Returns:
        Formatted display string
    """
    output = []

    output.append("=" * 80)
    output.append("FINAL SUMMARY VERIFICATION RESULTS")
    output.append("=" * 80)
    output.append("")

    # Statistics
    stats = result["statistics"]
    output.append("📊 Verification Statistics:")
    output.append(f"  • Total claims extracted: {stats['total_claims']}")
    output.append(f"  • ✅ Verified (ATTRIBUTABLE): {stats['verified_claims']}")
    output.append(f"  • ❌ Excluded: {stats['excluded_claims']}")
    output.append(f"  • ⚠️  Failed verification: {stats['failed_verifications']}")
    output.append("")
    output.append("📈 Refinement Statistics:")
    output.append(f"  • Claims kept: {stats['kept_claims']}")
    output.append(f"  • Claims removed: {stats['removed_claims']}")
    output.append(f"  • Claims modified: {stats['modified_claims']}")
    output.append("")

    # Refined summary
    output.append("✨ VERIFIED AND CORRECTED FINAL SUMMARY:")
    output.append("-" * 80)
    output.append(result["final_summary_verified"])
    output.append("-" * 80)
    output.append("")

    # Changes made
    if "changes_made" in result["refinement"]:
        output.append("📝 Changes Made:")
        output.append(result["refinement"]["changes_made"])
        output.append("")

    return "\n".join(output)

