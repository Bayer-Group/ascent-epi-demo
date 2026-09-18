"""
Reference management system for unified Literature and EHR citations.

This module provides utilities to:
1. Convert EHR answers into pseudo-documents for claim verification
2. Create unified document lists combining literature papers, EHR answers, and Google Search results
3. Map reference markers ([L1], [A1], [G1], etc.) to document indices
4. Parse and validate references in summaries
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def create_ehr_pseudo_document(
    question: str,
    answer: str,
    database: str,
    doc_index: int,
    ehr_index: int,
    timestamp: Optional[str] = None,
    database_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert an EHR answer into a pseudo-document structure for claim verification.

    This allows EHR answers to be treated like literature documents in the
    existing claim verification pipeline.

    Args:
        question: The original query asked to ASCENT
        answer: The full text response from the EHR system
        database: Database name (e.g., "SYNTHETIC_EHR_OMOP")
        doc_index: Index in the unified document list
        ehr_index: Index in the EHR-only sequence (for [A#] references)
        timestamp: Optional timestamp for the query
        database_metadata: Optional population metadata dict with country,
            total_patients, population, estimated_coverage, coverage_description

    Returns:
        Dictionary with document-like structure containing:
        - doc_index: Position in unified document list
        - type: "EHR_ANSWER" to distinguish from literature
        - title: Human-readable title
        - abstract: Answer text (for claim verification)
        - text: Full answer text
        - pid: Unique identifier
        - question: Original query
        - database: Source database
        - ehr_reference: The [A#] marker for this answer
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    pseudo_doc = {
        "doc_index": doc_index,
        "type": "EHR_ANSWER",
        "title": f"ASCENT EHR Query: {question}",
        "abstract": answer,  # Used for claim verification
        "text": answer,      # Full answer text
        "pid": f"EHR_{database}_{timestamp}",
        "database": database,
        "question": question,
        "ehr_reference": f"A{ehr_index}",  # [A1], [A2], etc.
        "metadata": {
            "source_type": "electronic_health_record",
            "query_timestamp": timestamp,
            "database_name": database,
            "country": database_metadata.get("country") if database_metadata else None,
            "total_patients_in_db": database_metadata.get("total_patients") if database_metadata else None,
            "country_population": database_metadata.get("population") if database_metadata else None,
            "estimated_coverage": database_metadata.get("estimated_coverage") if database_metadata else None,
            "coverage_description": database_metadata.get("coverage_description") if database_metadata else None,
        }
    }

    logger.debug(f"Created EHR pseudo-document [A{ehr_index}] for question: {question[:50]}...")
    return pseudo_doc


def prepare_unified_document_list(
    literature_docs: List[Dict[str, Any]],
    ehr_answers: List[Dict[str, Any]],
    google_search_docs: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Create unified document list combining literature papers, EHR answers,
    and optionally Google Search results.

    This function combines up to three types of sources:
    - Literature documents: Research papers with PMIDs, abstracts, etc. -> [L#]
    - EHR answers: Query results from ASCENT system -> [A#]
    - Google Search results: Web sources from Google Search grounding -> [G#]

    Args:
        literature_docs: List of literature document dictionaries
            Expected keys: 'title', 'abstract', 'text', 'pid', etc.
        ehr_answers: List of EHR answer dictionaries
            Expected keys: 'question', 'answer', 'database'
        google_search_docs: Optional list of Google Search result dictionaries
            Expected keys: 'title', 'url', 'answer'

    Returns:
        Tuple of:
        - unified_docs: Combined list of all documents
        - reference_mapping: Maps reference markers to document indices
            Format: {"L1": 0, "L2": 1, "A1": 2, "A2": 3, "G1": 4, ...}
    """
    unified_docs = []
    reference_mapping = {}

    # Add literature documents with [L1], [L2], ... references
    logger.info(f"Adding {len(literature_docs)} literature documents to unified list")
    for i, lit_doc in enumerate(literature_docs):
        # Ensure doc has doc_index set
        if 'doc_index' not in lit_doc:
            lit_doc['doc_index'] = len(unified_docs)

        unified_docs.append(lit_doc)
        lit_ref = f"L{i + 1}"
        reference_mapping[lit_ref] = lit_doc['doc_index']
        logger.debug(f"  [{lit_ref}] -> doc_index {lit_doc['doc_index']}: {lit_doc.get('title', 'Untitled')[:50]}...")

    # Add EHR answers as pseudo-documents with [A1], [A2], ... references
    logger.info(f"Adding {len(ehr_answers)} EHR answers as pseudo-documents")
    for i, ehr_data in enumerate(ehr_answers):
        doc_index = len(unified_docs)
        ehr_index = i + 1

        # Create pseudo-document from EHR answer
        pseudo_doc = create_ehr_pseudo_document(
            question=ehr_data.get('question', 'Unknown question'),
            answer=ehr_data.get('answer', ''),
            database=ehr_data.get('database', 'UNKNOWN_DB'),
            doc_index=doc_index,
            ehr_index=ehr_index,
            timestamp=ehr_data.get('timestamp'),
            database_metadata=ehr_data.get('database_metadata'),
        )

        unified_docs.append(pseudo_doc)
        ehr_ref = f"A{ehr_index}"
        reference_mapping[ehr_ref] = doc_index
        logger.debug(f"  [{ehr_ref}] -> doc_index {doc_index}: {ehr_data.get('question', '')[:50]}...")

    # Add Google Search results with [G1], [G2], ... references
    google_search_docs = google_search_docs or []
    if google_search_docs:
        logger.info(f"Adding {len(google_search_docs)} Google Search results to unified list")
        for i, gs_doc in enumerate(google_search_docs):
            doc_index = len(unified_docs)
            gs_index = i + 1

            pseudo_doc = {
                "doc_index": doc_index,
                "type": "GOOGLE_SEARCH",
                "title": gs_doc.get("title", "Web Source"),
                "abstract": gs_doc.get("answer", gs_doc.get("text", "")),
                "text": gs_doc.get("answer", gs_doc.get("text", "")),
                "pid": gs_doc.get("url", ""),
                "url": gs_doc.get("url", ""),
                "google_reference": f"G{gs_index}",
            }
            unified_docs.append(pseudo_doc)
            gs_ref = f"G{gs_index}"
            reference_mapping[gs_ref] = doc_index
            logger.debug(f"  [{gs_ref}] -> doc_index {doc_index}: {gs_doc.get('title', '')[:50]}...")

    logger.info(f"Unified document list created: {len(unified_docs)} total documents "
                f"({len(literature_docs)} literature + {len(ehr_answers)} EHR + {len(google_search_docs)} Google)")
    logger.info(f"Reference mapping: {list(reference_mapping.keys())}")

    return unified_docs, reference_mapping


def parse_reference_markers(text: str) -> List[str]:
    """
    Extract all [L#], [A#], and [G#] reference markers from text.

    Args:
        text: Text containing reference markers

    Returns:
        List of reference markers found (e.g., ["L1", "A2", "G1", "L3"])
    """
    # Pattern matches [L1], [A2], [G3], etc.
    pattern = r'\[([LAG]\d+)\]'
    markers = re.findall(pattern, text)
    return markers


def validate_references(
    text: str,
    reference_mapping: Dict[str, int]
) -> Dict[str, Any]:
    """
    Validate that all reference markers in text are valid.

    Args:
        text: Text to validate
        reference_mapping: Valid reference markers and their mappings

    Returns:
        Dictionary with validation results:
        - valid: Whether all references are valid
        - found_references: List of reference markers found
        - invalid_references: List of invalid reference markers
        - missing_documents: References that point to non-existent documents
    """
    found_refs = parse_reference_markers(text)
    invalid_refs = [ref for ref in found_refs if ref not in reference_mapping]

    validation = {
        "valid": len(invalid_refs) == 0,
        "found_references": found_refs,
        "invalid_references": invalid_refs,
        "reference_count": len(found_refs),
        "unique_reference_count": len(set(found_refs))
    }

    if invalid_refs:
        logger.warning(f"Found {len(invalid_refs)} invalid references: {invalid_refs}")

    return validation


def convert_references_to_doc_indices(
    reference_markers: List[str],
    reference_mapping: Dict[str, int]
) -> List[int]:
    """
    Convert reference markers ([L1], [A2], etc.) to document indices.

    Args:
        reference_markers: List of reference markers (e.g., ["L1", "E2"])
        reference_mapping: Maps reference markers to document indices

    Returns:
        List of document indices (e.g., [0, 5])
    """
    doc_indices = []
    for marker in reference_markers:
        if marker in reference_mapping:
            doc_indices.append(reference_mapping[marker])
        else:
            logger.warning(f"Reference marker {marker} not found in mapping")

    return doc_indices


def format_reference_list(
    unified_docs: List[Dict[str, Any]],
    reference_mapping: Dict[str, int],
    used_references: Optional[List[str]] = None
) -> str:
    """
    Format a reference list for display at the end of a summary.

    Args:
        unified_docs: List of all documents
        reference_mapping: Maps reference markers to document indices
        used_references: Optional list of actually used references (to filter)

    Returns:
        Formatted reference list string with separate sections for literature and EHR
    """
    output = []

    # Separate literature, EHR, and Google references
    lit_refs = []
    ehr_refs = []
    google_refs = []

    for ref_marker, doc_idx in sorted(reference_mapping.items(), key=lambda x: (x[0][0], int(x[0][1:]))):
        # Skip if we're filtering to used references only
        if used_references and ref_marker not in used_references:
            continue

        if ref_marker.startswith('L'):
            lit_refs.append((ref_marker, doc_idx))
        elif ref_marker.startswith('A'):
            ehr_refs.append((ref_marker, doc_idx))
        elif ref_marker.startswith('G'):
            google_refs.append((ref_marker, doc_idx))

    # Format literature references
    if lit_refs:
        output.append("=== Literature References ===")
        for ref_marker, doc_idx in lit_refs:
            if doc_idx < len(unified_docs):
                doc = unified_docs[doc_idx]
                title = doc.get('title', 'Untitled')
                pid = doc.get('pid', 'No ID')
                authors = doc.get('authors', '')

                # Format: [L1] Author et al. Title. PMID: 12345678
                if authors:
                    output.append(f"[{ref_marker}] {authors}. {title}. {pid}")
                else:
                    output.append(f"[{ref_marker}] {title}. {pid}")
        output.append("")  # Blank line

    # Format EHR references
    if ehr_refs:
        output.append("=== EHR/ASCENT Query References ===")
        for ref_marker, doc_idx in ehr_refs:
            if doc_idx < len(unified_docs):
                doc = unified_docs[doc_idx]
                question = doc.get('question', 'Unknown query')
                database = doc.get('database', 'Unknown database')
                answer = doc.get('answer', doc.get('text', ''))

                # Truncate long answers
                answer_preview = answer[:200] + "..." if len(answer) > 200 else answer

                output.append(f"[{ref_marker}] Query: \"{question}\"")
                output.append(f"     Database: {database}")
                output.append(f"     Answer: {answer_preview}")
                output.append("")  # Blank line between EHR references

    # Format Google Search references
    if google_refs:
        output.append("=== Web/Google Search References ===")
        for ref_marker, doc_idx in google_refs:
            if doc_idx < len(unified_docs):
                doc = unified_docs[doc_idx]
                title = doc.get('title', 'Web Source')
                url = doc.get('url', doc.get('pid', ''))
                output.append(f"[{ref_marker}] {title} - {url}")
        output.append("")

    return "\n".join(output)


def extract_claims_with_references(text: str) -> Dict[str, List[str]]:
    """
    Extract claims from text along with their reference markers.

    This function identifies sentences or statements that include reference
    markers and extracts both the claim text and the references.

    Args:
        text: Text containing claims with references

    Returns:
        Dictionary mapping claim text to list of reference markers
        Example: {
            "Diabetes affects 10% of adults": ["L1", "L2"],
            "Treatment improves outcomes": ["E1", "L3"]
        }
    """
    claims_with_refs = {}

    # Pattern to find text segments with references
    # Matches sentences or phrases ending with one or more [L#], [A#], or [G#] markers
    # Example: "Some claim text [L1][A2][G1]."
    claim_pattern = r'([^.!?\n]+(?:\[(?:L|A|G)\d+\])+[.!?]?)'

    for match in re.finditer(claim_pattern, text):
        claim_text = match.group(1)

        # Extract the claim without references for the key
        claim_clean = re.sub(r'\[(?:L|A|G)\d+\]', '', claim_text).strip()

        # Extract all reference markers from this claim
        refs = parse_reference_markers(claim_text)

        if refs and claim_clean:
            claims_with_refs[claim_clean] = refs

    logger.debug(f"Extracted {len(claims_with_refs)} claims with references from text")
    return claims_with_refs


def extract_summary_section(messages: List[Any]) -> Optional[str]:
    """
    Extract the final summary section from agent messages.

    Looks for content marked with "Summary" heading or similar indicators.

    Args:
        messages: List of agent messages (AIMessage, HumanMessage, etc.)

    Returns:
        Extracted summary text or None if not found
    """
    logger.debug(f"Searching for summary section in {len(messages)} messages")

    # Look through messages in reverse (most recent first)
    for msg in reversed(messages):
        # Check if message has content attribute
        content = getattr(msg, 'content', None)
        if not content:
            continue

        content_lower = content.lower()

        # Look for summary section markers
        if 'summary' in content_lower:
            # Try to extract just the summary section
            lines = content.split('\n')
            summary_lines = []
            in_summary = False

            for line in lines:
                line_lower = line.lower().strip()

                # Start capturing when we see "summary" heading
                # Look for: ## Summary, # Summary, ### Summary, **Summary**, Summary:, etc.
                if not in_summary and 'summary' in line_lower and (
                    line_lower.startswith('#') or
                    line_lower.startswith('*') or
                    line_lower.startswith('summary') or
                    line_lower.endswith('summary') or
                    'summary:' in line_lower
                ):
                    in_summary = True
                    # Don't include the heading itself, just mark that we're starting
                    continue

                # If we're in summary section, check for stopping conditions
                if in_summary:
                    # Stop if we hit another major section heading (###, ####, etc.)
                    # But only if it doesn't contain "summary"
                    if line_lower.startswith('###') and 'summary' not in line_lower:
                        # This is a subsection within the summary, keep going
                        summary_lines.append(line)
                        continue

                    if line_lower.startswith('##') and 'summary' not in line_lower:
                        # This is a new major section, stop
                        break

                    # Stop if we hit reference sections
                    if ('=== literature references ===' in line_lower or
                        '=== ehr' in line_lower or
                        'literature references' in line_lower):
                        # Don't include reference lists in the summary to be verified
                        break

                    # Capture all lines (including empty lines for formatting)
                    summary_lines.append(line)

            if summary_lines:
                summary = '\n'.join(summary_lines).strip()
                logger.info(f"Extracted summary section ({len(summary)} chars)")
                return summary

    logger.warning("No summary section found in messages")
    return None

