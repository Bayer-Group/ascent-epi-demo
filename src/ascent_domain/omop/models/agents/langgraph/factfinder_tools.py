"""
FactFinder claim extraction and attribution using local Gemini models.

This module provides tools for:
1. Extracting claims from answers using Gemini with structured output
2. Verifying claim attribution against documents using Gemini
"""

import asyncio
import logging
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from ascent_platform.llm.factory import create_assistant

logger = logging.getLogger(__name__)


# Pydantic models for structured outputs
class Claim(BaseModel):
    """A single claim with its document references."""
    claim_text: str = Field(description="The claim statement")
    document_indices: List[int] = Field(description="List of document indices this claim references")


class ClaimsExtraction(BaseModel):
    """Structured output for claim extraction."""
    claims: List[Claim] = Field(
        description="List of extracted claims with their referenced document indices"
    )


class AttributionResult(BaseModel):
    """Structured output for claim attribution."""
    content_review: str = Field(
        description="Analysis of how the claim relates to the reference text"
    )
    label: Literal["ATTRIBUTABLE", "PARTIALLY_ATTRIBUTABLE", "CONTRADICTORY", "UNRELATED"] = Field(
        description="Attribution label for the claim"
    )


class LocalClaimProcessor:
    """Local claim processor using Gemini with structured outputs."""

    def __init__(self):
        """Initialize Gemini assistant for claim processing."""
        self.assistant = None
        self.system_context = "You are an expert in biomedical field responsible for analyzing claims and verifying attribution."

    async def _get_assistant(self):
        """Lazy load assistant."""
        if self.assistant is None:
            # GeminiAssistant doesn't accept system_message parameter
            # We'll prepend the system context to each prompt instead
            # Use high max_output_tokens (12000) to avoid JSON truncation for large claim lists
            self.assistant = create_assistant(
                assistant_type="gemini",
                max_output_tokens=12000  # Increased from default 4096 to handle many claims
            )
        return self.assistant

    async def extract_claims(self, answer: str) -> Dict[str, Any]:
        """
        Extract claims from an answer text using Gemini with structured output.

        Args:
            answer: The answer text to extract claims from (with [n] reference markers)

        Returns:
            Dictionary with claims and their associated document indices
            Format: {"claims": {"claim_text": [doc_index1, doc_index2], ...}}
        """
        claim_extraction_prompt = """You are tasked with analyzing an answer that includes cited sources, denoted by indices (e.g., [1][2]). Your objective is to pinpoint individual assertions within the answer that may require verification. Each assertion should be associated with its corresponding source index or indices.

Extract claims as a list where each claim has:
- claim_text: The assertion/claim statement
- document_indices: List of document indices (numbers) that support this claim

Ensure that each assertion is self-contained and can be independently verified without dependence on other assertions. You may need to modify the given answer text to achieve this. In doing so, you must ensure the following:

- When referring to specific entities such as drugs, genes, diseases, or others, their names should be explicitly stated (e.g., 'Acetaminophen', 'BRCA1', 'cancer').
- Each assertion should not reference any information from preceding assertions. Therefore, phrases such as "other medications", "other genes" should be avoided. Some cases may be complex: for instance, if you encounter the phrase "{some_disease} can impact the small bowel and other parts of the colon", you should extract the following assertions: "{some_disease} can impact the small bowel" and "{some_disease} can impact parts of the colon other than the small bowel".

Additional constraints to adhere to include:

- If an assertion doesn't explicitly specify what sources it is referring to then it must be associated to empty document_indices (empty list). Never associate an assertion to all the sources unless they are clearly referenced.
- Whenever the answer states that a fact is supported by specific source(s), ensure to include this statement within an assertion.

It is critical that all the constraints are respected in order to maintain the accuracy and reliability of the analysis."""

        try:
            logger.info("Extracting claims from answer using Gemini...")
            assistant = await self._get_assistant()

            # Prepend system context to the prompt (GeminiAssistant doesn't support system_message)
            full_prompt = f"{self.system_context}\n\n{claim_extraction_prompt}\n\nANSWER: {answer}"

            result = await assistant.get_response(
                prompt=full_prompt,
                temperature=1.0,
                response_schema=ClaimsExtraction
            )

            if isinstance(result, ClaimsExtraction):
                # Convert list of Claim objects to dict format for backward compatibility
                claims_dict = {
                    claim.claim_text: claim.document_indices
                    for claim in result.claims
                }
                logger.info(f"Extracted {len(claims_dict)} claims")
                return {"claims": claims_dict}
            else:
                logger.error("Failed to get structured response from Gemini")
                return {"claims": {}, "error": "Failed to extract claims"}

        except Exception as e:
            logger.error(f"Error during claim extraction: {str(e)}")
            return {"claims": {}, "error": str(e)}

    async def verify_claim_attribution(self, claim: str, text: str) -> Dict[str, Any]:
        """
        Verify if a claim is attributable to a given text/document using Gemini.

        Note: Only pass the REFERENCED documents (those cited in the claim) as text.
        Gemini supports 1M+ input tokens, so full document text can be included.

        Args:
            claim: The claim to verify
            text: The reference text from documents cited in the claim (full text, no truncation)
        Returns:
            Dictionary with attribution result
            Format: {
                "claims": claim_text,
                "text": text,
                "claims_dictionary": {
                    claim_text: {
                        "label": "ATTRIBUTABLE" | "PARTIALLY_ATTRIBUTABLE" | "UNRELATED" | "CONTRADICTORY",
                        "explanation": "..."
                    }
                }
            }
        """
        attribution_prompt = """Begin by examining a specific claim and then thoroughly review the content and context of the given reference. Compare the details of the claim against the information presented in the reference. Consider if the reference fully supports the claim, partially supports it, contradicts it, or is unrelated to it. After analyzing and comparing the claim with the reference, conclude your assessment.

Assessment Labels:
ATTRIBUTABLE: Use this label if the reference fully supports the claim, indicating a direct and complete alignment. It is ok, if the reference supports the claim but with an additional detail, label it as ATTRIBUTABLE.
CONTRADICTORY: Use this label if the reference clearly opposes or is in direct conflict with the claim.
PARTIALLY_ATTRIBUTABLE: Use this label if the reference is related but we cannot firmly infer the preceding options, suggesting a possibility or partial alignment.
UNRELATED: Use this label if the reference is not related to the claim in the abstract.

IMPORTANT: Keep your 'content_review' explanation concise (2-3 sentences maximum). Focus only on the key evidence supporting your label decision."""

        try:
            logger.debug(f"Verifying claim attribution: '{claim[:50]}...' against {len(text):,} chars of reference text")
            assistant = await self._get_assistant()

            # NO truncation - Gemini supports 1M+ input tokens
            # We only pass the documents that are REFERENCED in the claim, not all documents
            # This provides complete context while being selective about what to include

            # Prepend system context to the prompt
            full_prompt = f"{self.system_context}\n\n{attribution_prompt}\n\nClaim: {claim}\n\nReference: {text}"

            result = await assistant.get_response(
                prompt=full_prompt,
                temperature=1.0,
                response_schema=AttributionResult
            )

            if isinstance(result, AttributionResult):
                logger.debug(f"Attribution result: {result.label}")
                return {
                    "claims": claim,
                    "text": text,
                    "claims_dictionary": {
                        claim: {
                            "label": result.label,
                            "explanation": result.content_review
                        }
                    }
                }
            else:
                logger.error("Failed to get structured response from Gemini")
                return {
                    "claims": claim,
                    "text": text,
                    "claims_dictionary": {
                        claim: {
                            "label": "ERROR",
                            "explanation": "Failed to verify attribution"
                        }
                    }
                }

        except Exception as e:
            logger.error(f"Error during claim verification: {str(e)}")
            return {
                "claims": claim,
                "text": text,
                "claims_dictionary": {
                    claim: {
                        "label": "ERROR",
                        "explanation": f"Verification failed: {str(e)}"
                    }
                }
            }

    async def batch_verify_claims(
        self,
        claims_with_docs: Dict[str, List[int]],
        documents: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Verify multiple claims against their referenced documents in batch.

        Args:
            claims_with_docs: Dictionary mapping claims to document indices
            documents: List of document dictionaries with 'abstract' or 'text' field
            max_concurrent: Maximum number of concurrent verification requests

        Returns:
            Dictionary mapping each claim to its attribution result
            Format: {
                claim_text: {
                    "label": "ATTRIBUTABLE" | ...,
                    "explanation": "...",
                    "references": [{"doc_index": 0, "title": "...", ...}]
                }
            }
        """
        results = {}
        semaphore = asyncio.Semaphore(max_concurrent)

        async def verify_claim_with_semaphore(claim: str, doc_indices: List[int]):
            async with semaphore:
                # Combine ONLY the referenced documents (those cited in the claim)
                # Gemini supports 1M+ input tokens, so we can include full document text
                combined_text = ""
                references = []
                total_chars = 0

                logger.debug(f"Verifying claim with doc_indices={doc_indices}, total_docs_available={len(documents)}")

                for idx in doc_indices:  # Process all referenced documents
                    if idx < len(documents):
                        doc = documents[idx]
                        # Extract text from document (try abstract first, then text, then content)
                        doc_text = doc.get('abstract') or doc.get('text') or doc.get('content', '')

                        if doc_text:
                            # Include FULL document text (no truncation) since model can handle 1M tokens
                            combined_text += f"\n\n[Document {idx}]\n{doc_text}"
                            total_chars += len(doc_text)

                            # Store full reference details for citation tracking
                            references.append({
                                "doc_index": idx,
                                "title": doc.get('title', 'Untitled'),
                                "pid": doc.get('pid', ''),
                                "abstract": doc.get('abstract', '')[:200] + "..." if doc.get('abstract') else "",
                            })
                        else:
                            logger.warning(f"Document {idx} has no text content (abstract/text/content all empty)")
                    else:
                        logger.warning(f"Document index {idx} out of range (max: {len(documents)-1})")

                if not combined_text:
                    logger.warning(f"NO_REFERENCE for claim: {claim[:100]}... (doc_indices={doc_indices})")
                    return claim, {
                        "label": "NO_REFERENCE",
                        "explanation": f"No document text found for referenced indices {doc_indices}",
                        "references": []
                    }

                logger.debug(f"Combined {len(references)} documents ({total_chars:,} chars total) for claim verification")

                # Verify the claim against combined documents
                try:
                    result = await self.verify_claim_attribution(claim, combined_text)

                    # Extract attribution info
                    claims_dict = result.get("claims_dictionary", {})
                    attribution_info = claims_dict.get(claim, {
                        "label": "UNKNOWN",
                        "explanation": "No attribution result returned"
                    })

                    # Add references to the result
                    attribution_info["references"] = references

                    return claim, attribution_info

                except Exception as e:
                    # If verification fails after all retries, mark as FAILED_VERIFICATION
                    logger.warning(f"Failed to verify claim after retries: {str(e)}")
                    return claim, {
                        "label": "FAILED_VERIFICATION",
                        "explanation": f"Could not verify claim due to technical error: {str(e)[:100]}",
                        "references": references
                    }

        # Create tasks for all claims
        tasks = [
            verify_claim_with_semaphore(claim, doc_indices)
            for claim, doc_indices in claims_with_docs.items()
        ]

        # Execute all verifications
        logger.info(f"Verifying {len(tasks)} claims in batch (max {max_concurrent} concurrent)...")
        verified_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in verified_results:
            if isinstance(result, Exception):
                logger.error(f"Claim verification failed: {str(result)}")
                continue

            claim, attribution_info = result
            results[claim] = attribution_info

        # Log summary
        labels_count = {}
        for info in results.values():
            label = info.get("label", "UNKNOWN")
            labels_count[label] = labels_count.get(label, 0) + 1

        logger.info(f"Claim verification complete. Summary: {labels_count}")

        return results


async def process_literature_answer_with_claims(
    answer: str,
    documents: List[Dict[str, Any]],
    include_partial: bool = False
) -> Dict[str, Any]:
    """
    Process a literature answer to extract and verify claims.

    Args:
        answer: The literature answer text (with [n] reference markers)
        documents: List of source documents
        include_partial: Whether to include partially attributable claims

    Returns:
        Dictionary with verified claims and their references
        Format: {
            "original_answer": str,
            "total_claims": int,
            "verified_claims": [
                {
                    "claim": str,
                    "attribution": "ATTRIBUTABLE" | "PARTIALLY_ATTRIBUTABLE",
                    "explanation": str,
                    "references": [{"doc_index": int, "title": str, "pid": str}, ...]
                },
                ...
            ],
            "excluded_claims": [
                {
                    "claim": str,
                    "attribution": "UNRELATED" | "CONTRADICTORY",
                    "explanation": str
                },
                ...
            ]
        }
    """
    client = LocalClaimProcessor()

    # Step 1: Extract claims from answer
    logger.info("Step 1: Extracting claims from literature answer using Gemini...")
    extraction_result = await client.extract_claims(answer)

    if "error" in extraction_result:
        logger.error(f"Claim extraction failed: {extraction_result['error']}")
        return {
            "original_answer": answer,
            "total_claims": 0,
            "verified_claims": [],
            "excluded_claims": [],
            "error": extraction_result["error"]
        }

    claims_with_docs = extraction_result.get("claims", {})

    if not claims_with_docs:
        logger.warning("No claims extracted from answer")
        return {
            "original_answer": answer,
            "total_claims": 0,
            "verified_claims": [],
            "excluded_claims": []
        }

    logger.info(f"Extracted {len(claims_with_docs)} claims")

    # Step 2: Verify each claim against its referenced documents
    logger.info("Step 2: Verifying claim attributions using Gemini...")
    attribution_results = await client.batch_verify_claims(claims_with_docs, documents)

    # Step 3: Categorize claims based on attribution
    verified_claims = []
    excluded_claims = []
    failed_verifications = []  # Track claims that couldn't be verified due to technical errors

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
            # Track failed verifications separately - exclude from both verified and regular excluded
            failed_verifications.append(claim_data)
            logger.warning(f"Claim verification failed after retries: {claim[:100]}...")
        else:
            # Exclude UNRELATED, CONTRADICTORY, PARTIALLY_ATTRIBUTABLE (if not including), etc.
            excluded_claims.append(claim_data)

    result = {
        "original_answer": answer,
        "total_claims": len(claims_with_docs),
        "verified_claims": verified_claims,
        "excluded_claims": excluded_claims,
        "failed_verifications": failed_verifications  # Include failed verifications in result
    }

    if failed_verifications:
        logger.warning(f"Claim processing complete: {len(verified_claims)} verified, {len(excluded_claims)} excluded, {len(failed_verifications)} failed verification")
    else:
        logger.info(f"Claim processing complete: {len(verified_claims)} verified, {len(excluded_claims)} excluded")

    return result


def format_claims_as_answer(verified_claims: List[Dict[str, Any]]) -> str:
    """
    Format verified claims into a readable answer with references.

    Args:
        verified_claims: List of verified claim dictionaries

    Returns:
        Formatted answer string with claims and references
    """
    if not verified_claims:
        return "No verified claims could be extracted from the literature."

    answer_parts = []

    for i, claim_data in enumerate(verified_claims, 1):
        claim = claim_data["claim"]
        references = claim_data.get("references", [])

        # Format references as [n, m, ...]
        ref_indices = [str(ref["doc_index"]) for ref in references[:100]]  # Limit to 100
        ref_str = f"[{', '.join(ref_indices)}]" if ref_indices else ""

        # Add attribution confidence if partially attributable
        confidence_note = ""
        if claim_data.get("attribution") == "PARTIALLY_ATTRIBUTABLE":
            confidence_note = " (partially verified)"

        answer_parts.append(f"• {claim} {ref_str}{confidence_note}")

    # Add reference list at the end
    answer_parts.append("\n\n=== References ===")

    # Collect all unique references
    all_refs = {}
    for claim_data in verified_claims:
        for ref in claim_data.get("references", []):
            doc_idx = ref["doc_index"]
            if doc_idx not in all_refs:
                all_refs[doc_idx] = ref

    # Sort by index and format
    for idx in sorted(all_refs.keys()):
        ref = all_refs[idx]
        answer_parts.append(f"[{idx}] {ref['title']} ({ref['pid']})")

    return "\n\n".join(answer_parts)


def prepare_claims_for_streamlit(claims_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare claims data in a structured format for Streamlit visualization.

    Args:
        claims_data: Claims data from process_literature_answer_with_claims

    Returns:
        Structured data optimized for Streamlit display with:
        - Numbered claims with attribution status
        - Reference details for each claim
        - Summary statistics
    """
    if not claims_data:
        return {"error": "No claims data available"}

    # Categorize claims by attribution status
    attributable = [c for c in claims_data.get("verified_claims", []) if c["attribution"] == "ATTRIBUTABLE"]
    partially_attr = [c for c in claims_data.get("excluded_claims", []) if c["attribution"] == "PARTIALLY_ATTRIBUTABLE"]
    unrelated = [c for c in claims_data.get("excluded_claims", []) if c["attribution"] == "UNRELATED"]
    contradictory = [c for c in claims_data.get("excluded_claims", []) if c["attribution"] == "CONTRADICTORY"]
    no_reference = [c for c in claims_data.get("excluded_claims", []) if c["attribution"] == "NO_REFERENCE"]
    failed_verification = [c for c in claims_data.get("excluded_claims", []) if c["attribution"] == "FAILED_VERIFICATION"]
    error = [c for c in claims_data.get("excluded_claims", []) if c["attribution"] == "ERROR"]
    unknown = [c for c in claims_data.get("excluded_claims", []) if c["attribution"] == "UNKNOWN"]

    # Build structured output
    structured = {
        "summary": {
            "total_claims": claims_data.get("total_claims", 0),
            "attributable_count": len(attributable),
            "partially_attributable_count": len(partially_attr),
            "unrelated_count": len(unrelated),
            "contradictory_count": len(contradictory),
            "no_reference_count": len(no_reference),
            "failed_verification_count": len(failed_verification),
            "error_count": len(error),
            "unknown_count": len(unknown)
        },
        "claims_by_status": {
            "attributable": [],
            "partially_attributable": [],
            "unrelated": [],
            "contradictory": [],
            "no_reference": [],
            "failed_verification": [],
            "error": [],
            "unknown": []
        },
        "all_references": {}
    }

    # Helper to format claim for display
    def format_claim_for_display(claim_data: Dict, claim_id: int) -> Dict:
        references = claim_data.get("references", [])
        return {
            "id": claim_id,
            "text": claim_data["claim"],
            "attribution": claim_data["attribution"],
            "explanation": claim_data.get("explanation", ""),
            "reference_count": len(references),
            "reference_ids": [ref["doc_index"] for ref in references],
            "references": references
        }

    # Process attributable claims
    claim_counter = 1
    for claim_data in attributable:
        structured["claims_by_status"]["attributable"].append(
            format_claim_for_display(claim_data, claim_counter)
        )
        # Collect references
        for ref in claim_data.get("references", []):
            structured["all_references"][ref["doc_index"]] = ref
        claim_counter += 1

    # Process partially attributable claims
    for claim_data in partially_attr:
        structured["claims_by_status"]["partially_attributable"].append(
            format_claim_for_display(claim_data, claim_counter)
        )
        for ref in claim_data.get("references", []):
            structured["all_references"][ref["doc_index"]] = ref
        claim_counter += 1

    # Process unrelated claims
    for claim_data in unrelated:
        structured["claims_by_status"]["unrelated"].append(
            format_claim_for_display(claim_data, claim_counter)
        )
        claim_counter += 1

    # Process contradictory claims
    for claim_data in contradictory:
        structured["claims_by_status"]["contradictory"].append(
            format_claim_for_display(claim_data, claim_counter)
        )
        claim_counter += 1

    # Process no_reference claims
    for claim_data in no_reference:
        structured["claims_by_status"]["no_reference"].append(
            format_claim_for_display(claim_data, claim_counter)
        )
        claim_counter += 1

    # Process failed_verification claims
    for claim_data in failed_verification:
        structured["claims_by_status"]["failed_verification"].append(
            format_claim_for_display(claim_data, claim_counter)
        )
        claim_counter += 1

    # Process error claims
    for claim_data in error:
        structured["claims_by_status"]["error"].append(
            format_claim_for_display(claim_data, claim_counter)
        )
        claim_counter += 1

    # Process unknown claims
    for claim_data in unknown:
        structured["claims_by_status"]["unknown"].append(
            format_claim_for_display(claim_data, claim_counter)
        )
        claim_counter += 1

    return structured


async def refine_summary_with_claim_verdicts(
    original_summary: str,
    claims_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Refine the original summary based on claim verification verdicts.

    This function uses an LLM to edit the original summary by:
    - Keeping statements aligned with ATTRIBUTABLE claims
    - Modifying PARTIALLY_ATTRIBUTABLE claims to make them correct (based on feedback)
    - Dropping PARTIALLY_ATTRIBUTABLE claims if they cannot be easily corrected
    - Removing UNRELATED/CONTRADICTORY claims
    - Maintaining the original structure and citations

    Args:
        original_summary: The original answer text from literature search
        claims_data: Full claims data with attributions from process_literature_answer_with_claims

    Returns:
        Dictionary with:
        - refined_summary: The edited summary text
        - changes_made: List of changes and their reasons
        - kept_claims: Count of attributable claims kept
        - removed_claims: Count of claims removed (includes dropped partial claims)
        - modified_claims: Count of partial claims that were successfully modified
    """
    processor = LocalClaimProcessor()

    # Prepare claims summary for the LLM
    all_claims = []

    # Add verified claims
    for claim in claims_data.get("verified_claims", []):
        all_claims.append({
            "claim": claim["claim"],
            "verdict": claim["attribution"],
            "explanation": claim.get("explanation", "Fully supported by evidence"),
            "action": "KEEP - This is well-supported"
        })

    # Add excluded claims
    for claim in claims_data.get("excluded_claims", []):
        attribution = claim["attribution"]
        explanation = claim.get("explanation", "")

        if attribution == "PARTIALLY_ATTRIBUTABLE":
            # Provide specific guidance based on the explanation
            action = f"MODIFY OR DROP - {explanation[:200]}. Try to correct the claim based on this feedback. If you cannot make it accurate with a simple change, drop it entirely."
        elif attribution == "CONTRADICTORY":
            action = "REMOVE - Contradicts evidence"
        else:  # UNRELATED
            action = "REMOVE - Not supported by evidence"

        all_claims.append({
            "claim": claim["claim"],
            "verdict": attribution,
            "explanation": explanation,
            "action": action
        })

    # Add failed verifications - these should be removed from summary
    for claim in claims_data.get("failed_verifications", []):
        all_claims.append({
            "claim": claim["claim"],
            "verdict": "FAILED_VERIFICATION",
            "explanation": claim.get("explanation", "Technical error during verification"),
            "action": "REMOVE - Could not be verified due to technical error"
        })

        all_claims.append({
            "claim": claim["claim"],
            "verdict": attribution,
            "explanation": explanation,
            "action": action
        })

    # Build the refinement prompt
    refinement_prompt = f"""You are tasked with refining a literature summary based on claim verification results.

ORIGINAL SUMMARY:
{original_summary}

CLAIM VERIFICATION RESULTS:
Below are the individual claims extracted from the summary, each with a verdict:

"""

    for i, claim_info in enumerate(all_claims, 1):
        refinement_prompt += f"""
Claim {i}: {claim_info['claim']}
Verdict: {claim_info['verdict']}
Explanation: {claim_info['explanation']}
Recommended Action: {claim_info['action']}
---
"""

    refinement_prompt += """

YOUR TASK:
Refine the ORIGINAL SUMMARY by making MINIMAL changes based on the claim verdicts:

1. For ATTRIBUTABLE claims (KEEP):
   - Keep these statements as they are
   - Maintain their citations [X]
   
2. For PARTIALLY_ATTRIBUTABLE claims (MODIFY OR DROP):
   - Read the verification explanation carefully - it tells you WHY it's only partial
   - Try to MODIFY the statement to make it correct based on the feedback
   - Examples:
     * If feedback says "doesn't confirm exact number" → remove the number, keep general statement
     * If feedback says "only mentions briefly" → soften the claim ("is mentioned", "is noted")
     * If feedback says "context is different" → add proper context or qualifier
   - If you CANNOT make it correct with a simple change → DROP it entirely
   - Maintain citations [X] for modified statements
   
3. For UNRELATED/CONTRADICTORY claims (REMOVE):
   - Remove these statements entirely
   - Adjust surrounding text for smooth flow
   
4. General rules:
   - Preserve the original structure and flow as much as possible
   - Maintain ALL citation numbers [X] for kept/modified statements
   - Do NOT add new information beyond what's in the feedback
   - Make edits smooth and natural
   - Keep the narrative coherent
   - When in doubt about PARTIALLY_ATTRIBUTABLE → DROP it (better safe than inaccurate)

OUTPUT FORMAT:
Provide your refined summary followed by a brief list of changes made.

REFINED SUMMARY:
[Your refined version here]

CHANGES MADE:
1. [Describe first change - kept/modified/dropped and why]
2. [Describe second change]
etc.
"""

    try:
        logger.info("Refining summary based on claim verdicts using Gemini...")
        assistant = await processor._get_assistant()

        # Add system context
        full_prompt = f"{processor.system_context}\n\n{refinement_prompt}"

        # Get refinement from Gemini (no structured output needed, just text)
        result = await assistant.get_response(
            prompt=full_prompt,
            temperature=1.0,
            json_format=False
        )

        # Parse the result to separate refined summary and changes
        result_text = str(result)

        # Try to split into refined summary and changes
        if "CHANGES MADE:" in result_text:
            parts = result_text.split("CHANGES MADE:")
            refined_summary = parts[0].replace("REFINED SUMMARY:", "").strip()
            changes_text = parts[1].strip()
        elif "REFINED SUMMARY:" in result_text:
            refined_summary = result_text.split("REFINED SUMMARY:")[1].strip()
            changes_text = "Changes applied based on claim verification"
        else:
            refined_summary = result_text
            changes_text = "Summary refined based on claim verdicts"

        # Count changes
        kept_count = len([c for c in all_claims if c["verdict"] == "ATTRIBUTABLE"])
        removed_count = len([c for c in all_claims if c["verdict"] in ["UNRELATED", "CONTRADICTORY"]])
        partial_count = len([c for c in all_claims if c["verdict"] == "PARTIALLY_ATTRIBUTABLE"])

        logger.info(f"Summary refinement complete: {kept_count} kept, {removed_count} removed, {partial_count} partial (modified or dropped)")

        return {
            "refined_summary": refined_summary,
            "changes_made": changes_text,
            "kept_claims": kept_count,
            "removed_claims": removed_count,
            "modified_claims": partial_count,  # These were either modified or dropped
            "original_summary": original_summary
        }

    except Exception as e:
        logger.error(f"Error during summary refinement: {str(e)}")
        return {
            "refined_summary": original_summary,  # Fallback to original
            "changes_made": f"Error during refinement: {str(e)}",
            "kept_claims": 0,
            "removed_claims": 0,
            "qualified_claims": 0,
            "original_summary": original_summary,
            "error": str(e)
        }


