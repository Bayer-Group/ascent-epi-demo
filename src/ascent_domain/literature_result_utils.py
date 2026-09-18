"""Shared utilities for building and normalising literature search results.

Used by both the WebSocket pipeline (clues_ws_service) and the MCP tool
(get_contextual_literature) so that both surfaces return an identical,
fully-annotated representation of web search results.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Source type → citation prefix for reference normalisation.
SOURCE_PREFIXES: dict[str, str] = {
    "literature": "H",
    "web": "G",
}


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def build_sources(
    verified_lit: dict[str, Any],
    google_results: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Build a unified source list from literature + Google results.

    Each entry has the shape::

        {
            "doc_index": int,
            "title": str,
            "pid": str | None,          # e.g. "EMBASE:L46745164" — None for web
            "url": str | None,          # web URL, if any
            "abstract": str | None,     # None for web sources
            "source_type": "literature" | "web",
        }

    Args:
        verified_lit: The ``verified_literature_results`` dict from agent state,
            containing an ``all_answers`` list.
        google_results: The ``google_search_results`` list from agent state.

    Returns:
        Ordered list of deduplicated source dicts.
    """
    sources: list[dict[str, Any]] = []
    doc_idx = 0

    # --- Literature sources ---
    seen_pids: set[str] = set()
    for answer in verified_lit.get("all_answers", []):
        for claim in answer.get("verified_claims_result", []):
            for ref in claim.get("references", []):
                pid = ref.get("pid", "")
                if pid and pid not in seen_pids:
                    seen_pids.add(pid)
                    sources.append(
                        {
                            "doc_index": doc_idx,
                            "title": ref.get("title", "Untitled"),
                            "pid": pid,
                            "url": None,
                            "abstract": ref.get("abstract", ""),
                            "source_type": "literature",
                        }
                    )
                    doc_idx += 1

    # --- Web sources (Google) ---
    if google_results:
        for g_result in google_results:
            for src in g_result.get("sources", []):
                uri = src.get("uri") or src.get("url", "")
                title = src.get("title", uri)
                if uri:
                    sources.append(
                        {
                            "doc_index": doc_idx,
                            "title": title,
                            "pid": None,
                            "url": uri,
                            "abstract": None,
                            "source_type": "web",
                        }
                    )
                    doc_idx += 1

    return sources


# ---------------------------------------------------------------------------
# Reference normalisation
# ---------------------------------------------------------------------------


def normalize_literature_references(
    verified_lit: dict[str, Any],
    sources: list[dict[str, Any]],
) -> None:
    """Replace ``[N]`` citation markers in literature answers with ``[H#](url)`` links.

    Mutates ``verified_lit["all_answers"][*]["answer"]`` in-place so that
    downstream consumers receive pre-formatted Markdown.

    Args:
        verified_lit: The ``verified_literature_results`` dict (mutated in place).
        sources: The unified source list produced by :func:`build_sources`.
    """
    lit_sources = [s for s in sources if s["source_type"] == "literature"]
    pid_to_ref: dict[str, tuple[str, str]] = {}
    for idx, src in enumerate(lit_sources):
        pid = src.get("pid")
        if pid:
            pid_to_ref[pid] = (f"H{idx + 1}", src.get("url") or "")

    for answer_obj in verified_lit.get("all_answers", []):
        answer_text = answer_obj.get("answer", "")
        if not answer_text:
            continue

        # Build doc_index → (H-label, url) mapping for this answer's references
        idx_to_ref: dict[int, tuple[str, str]] = {}
        for claim in answer_obj.get("verified_claims_result", []):
            for ref in claim.get("references", []):
                pid = ref.get("pid", "")
                doc_idx = ref.get("doc_index")
                if pid in pid_to_ref and doc_idx is not None:
                    idx_to_ref[doc_idx] = pid_to_ref[pid]

        def _replace_ref(match: re.Match) -> str:  # noqa: B023
            n = int(match.group(1))
            d_idx = n - 1  # LLM uses 1-based, doc_index is 0-based
            if d_idx in idx_to_ref:
                label, url = idx_to_ref[d_idx]
                return f"[\\[{label}\\]]({url})" if url else f"[{label}]"
            return match.group(0)

        answer_obj["answer"] = re.sub(r"\[(\d+)\]", _replace_ref, answer_text)


def normalize_google_references(
    google_results: list[dict[str, Any]] | None,
    sources: list[dict[str, Any]],
) -> None:
    """Replace ``[N]`` citation markers in Google answers with ``[G#](url)`` links.

    Mutates ``google_results[*]["answer"]`` in-place.

    Args:
        google_results: The ``google_search_results`` list from agent state.
        sources: The unified source list produced by :func:`build_sources`.
    """
    if not google_results:
        return

    web_sources = [s for s in sources if s["source_type"] == "web"]
    offset = 0

    for g_result in google_results:
        local_count = sum(1 for s in g_result.get("sources", []) if (s.get("uri") or s.get("url", "")))
        base, offset = offset, offset + local_count

        answer_text = g_result.get("answer", "")
        if not answer_text:
            continue

        def _replace_ref(match: re.Match, base: int = base, local_count: int = local_count) -> str:
            n = int(match.group(1))
            if not 1 <= n <= local_count:
                return match.group(0)
            idx = base + n - 1
            if 0 <= idx < len(web_sources):
                label = f"G{idx + 1}"
                url = web_sources[idx].get("url") or ""
                return f"[\\[{label}\\]]({url})" if url else f"[{label}]"
            return match.group(0)

        # Only replace in the answer body, not in the embedded "**Sources:**" footer
        parts = answer_text.split("\n\n---\n\n**Sources:**")
        parts[0] = re.sub(r"\[(\d+)\]", _replace_ref, parts[0])
        g_result["answer"] = "\n\n---\n\n**Sources:**".join(parts)


# ---------------------------------------------------------------------------
# Google answer list builder
# ---------------------------------------------------------------------------


def build_google_answers(
    google_results: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """Extract structured Q&A pairs from Google search results.

    Strips the embedded ``**Sources:**`` footer from each answer because the
    unified ``sources`` list already provides all references.

    Args:
        google_results: The ``google_search_results`` list from agent state.

    Returns:
        List of ``{"question": str, "answer": str}`` dicts.
    """
    if not google_results:
        return []

    answers: list[dict[str, str]] = []
    for g in google_results:
        answer = g.get("answer", "")
        if not answer:
            continue
        answer = answer.split("\n\n---\n\n**Sources:**")[0].strip()
        answers.append({"question": g.get("question", ""), "answer": answer})
    return answers


# ---------------------------------------------------------------------------
# Publication formatting
# ---------------------------------------------------------------------------


def extract_source_db(pid: str) -> str:
    """Derive the source database name from a PID like ``EMBASE:L123`` or ``PUBMED:456``."""
    if pid and ":" in pid:
        return pid.split(":", 1)[0]
    return "Unknown"


def build_publications(literature_documents: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert raw agent literature documents into a deduplicated publication list.

    Each entry has the shape::

        {
            "title": str,
            "source": str,                  # e.g. "EMBASE", "PUBMED", "Unknown"
            "publication_year": str | int | None,
            "pid": str,
            "url": str | None,              # web URL, if any
            "authors": list[str],
            "abstract": str,
        }

    Args:
        literature_documents: The ``literature_documents`` list from agent state.

    Returns:
        Deduplicated list of publication dicts (deduplicated by PID).
    """
    if not literature_documents:
        return []

    seen_pids: set[str] = set()
    publications: list[dict[str, Any]] = []

    for doc in literature_documents:
        pid = doc.get("pid", "")
        if pid and pid in seen_pids:
            continue
        if pid:
            seen_pids.add(pid)

        publications.append(
            {
                "title": doc.get("title", "Untitled"),
                "source": extract_source_db(pid),
                "publication_year": doc.get("year") or doc.get("publication_year"),
                "pid": pid,
                "url": None,
                "authors": doc.get("authors", []),
                "abstract": doc.get("abstract", ""),
            }
        )

    return publications
