"""
Google Search Grounding Tool for the LiteratureAscentComparisonAgent.

Uses Gemini with Google Search grounding to answer questions with real-time
web information, regulatory updates, and general medical content not covered
by a dedicated literature database.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ascent_domain.config import get_domain_settings
from ascent_domain.omop.models.agents.langgraph.llm_utils import extract_text_from_content
from ascent_platform.llm.credentials import gemini_api_key
from ascent_platform.llm.http_options import gemini_http_options

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class WebSource(BaseModel):
    uri: str = Field(default="", description="URL of the source")
    title: str = Field(default="", description="Title of the source")

    def to_markdown_link(self) -> str:
        if not self.uri:
            return self.title or "Unknown Source"
        clean = self.title.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        return f"[{clean}]({self.uri})"


class GroundingChunk(BaseModel):
    web: WebSource = Field(default_factory=WebSource)


class TextSegment(BaseModel):
    start_index: int = Field(default=0)
    end_index: int = Field(default=0)
    text: str = Field(default="")


class GroundingSupport(BaseModel):
    segment: TextSegment = Field(default_factory=TextSegment)
    grounding_chunk_indices: List[int] = Field(default_factory=list)


class GroundingMetadata(BaseModel):
    web_search_queries: List[str] = Field(default_factory=list)
    grounding_chunks: List[GroundingChunk] = Field(default_factory=list)
    grounding_supports: List[GroundingSupport] = Field(default_factory=list)


class SearchResult(BaseModel):
    answer: str = Field(description="The answer text")
    search_queries: List[str] = Field(default_factory=list)
    sources: List[WebSource] = Field(default_factory=list)
    grounding_metadata: Optional[GroundingMetadata] = Field(default=None)
    model: str = Field(default="")
    question: str = Field(default="")
    status: str = Field(default="success")
    error: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Domain name helpers
# ---------------------------------------------------------------------------

_DOMAIN_NAMES: Dict[str, str] = {
    "pubmed.ncbi.nlm.nih.gov": "PubMed",
    "ncbi.nlm.nih.gov": "NCBI",
    "clinicaltrials.gov": "ClinicalTrials.gov",
    "fda.gov": "FDA",
    "ema.europa.eu": "EMA",
    "nejm.org": "NEJM",
    "nature.com": "Nature",
    "sciencedirect.com": "ScienceDirect",
    "springer.com": "Springer",
    "wiley.com": "Wiley",
    "medscape.com": "Medscape",
    "reuters.com": "Reuters",
    "healthline.com": "Healthline",
    "webmd.com": "WebMD",
    "mayoclinic.org": "Mayo Clinic",
    "nih.gov": "NIH",
    "cdc.gov": "CDC",
    "who.int": "WHO",
    "wikipedia.org": "Wikipedia",
    "drugs.com": "Drugs.com",
}


def _format_domain(domain: str) -> str:
    domain = domain.replace("www.", "")
    if domain in _DOMAIN_NAMES:
        return _DOMAIN_NAMES[domain]
    for key, name in _DOMAIN_NAMES.items():
        if key in domain or domain.endswith(key):
            return name
    parts = domain.split(".")
    return parts[-2].title() if len(parts) >= 2 else domain.title()


def generate_title_from_url(uri: str, original_title: str = "") -> str:
    """Generate a human-friendly title from a URL or its original title."""
    if not uri:
        return _format_domain(original_title) if original_title else "Unknown Source"

    if "vertexaisearch" in uri.lower() or "cloud.google.com" in uri.lower():
        return _format_domain(original_title) if original_title else "Google Search Result"

    if original_title and len(original_title) > 3 and " " not in original_title[:20]:
        pass  # fall through to URL parsing

    if original_title and not _is_garbage_title(original_title):
        parsed = urlparse(uri)
        domain = parsed.netloc.replace("www.", "")
        if original_title.lower() != domain.lower() and len(original_title) > len(domain):
            return original_title

    try:
        parsed = urlparse(uri)
        domain = parsed.netloc.replace("www.", "")
        path = unquote(parsed.path)
        parts = [p for p in path.strip("/").split("/") if p and not p.startswith("?")]
        friendly = _format_domain(domain)
        if parts:
            last = parts[-1].rsplit(".", 1)[0].replace("-", " ").replace("_", " ")
            if len(last) > 3:
                return f"{friendly}: {last.title()[:100]}"
        return friendly
    except Exception:
        return _format_domain(original_title) if original_title else uri[:50]


def _is_garbage_title(title: str) -> bool:
    if not title:
        return True
    if title.lower().startswith("vertexaisearch"):
        return True
    if len(title) > 20 and " " not in title and re.match(r"^[A-Za-z0-9_\-:]+$", title):
        return True
    return False


# ---------------------------------------------------------------------------
# Grounding metadata extraction
# ---------------------------------------------------------------------------


def extract_grounding_metadata(response: Any) -> GroundingMetadata:
    """Parse Gemini response into structured GroundingMetadata."""
    metadata = GroundingMetadata()
    try:
        if not hasattr(response, "candidates") or not response.candidates:
            return metadata
        gm = getattr(response.candidates[0], "grounding_metadata", None)
        if gm is None:
            return metadata

        if hasattr(gm, "web_search_queries") and gm.web_search_queries:
            metadata.web_search_queries = list(gm.web_search_queries)

        if hasattr(gm, "grounding_chunks") and gm.grounding_chunks:
            for chunk in gm.grounding_chunks:
                if hasattr(chunk, "web"):
                    uri = getattr(chunk.web, "uri", "") or ""
                    orig = getattr(chunk.web, "title", "") or ""
                    title = generate_title_from_url(uri, orig)
                    metadata.grounding_chunks.append(GroundingChunk(web=WebSource(uri=uri, title=title)))

        if hasattr(gm, "grounding_supports") and gm.grounding_supports:
            for sup in gm.grounding_supports:
                seg = TextSegment()
                if hasattr(sup, "segment"):
                    seg = TextSegment(
                        start_index=getattr(sup.segment, "start_index", 0) or 0,
                        end_index=getattr(sup.segment, "end_index", 0) or 0,
                        text=getattr(sup.segment, "text", "") or "",
                    )
                indices = list(sup.grounding_chunk_indices) if hasattr(sup, "grounding_chunk_indices") and sup.grounding_chunk_indices else []
                metadata.grounding_supports.append(GroundingSupport(segment=seg, grounding_chunk_indices=indices))
    except Exception as exc:
        logger.warning(f"Error extracting grounding metadata: {exc}")
    return metadata


# ---------------------------------------------------------------------------
# Citation helpers
# ---------------------------------------------------------------------------


def add_inline_citations_from_grounding(text: str, gm: GroundingMetadata) -> str:
    """Insert ``[N]`` citations at positions indicated by grounding_supports."""
    if not gm or not gm.grounding_supports:
        return text
    sorted_sups = sorted(gm.grounding_supports, key=lambda s: s.segment.end_index, reverse=True)
    result = text
    for sup in sorted_sups:
        end = sup.segment.end_index
        if sup.grounding_chunk_indices and 0 < end <= len(result):
            cites = "".join(f"[{i + 1}]" for i in sup.grounding_chunk_indices if i < len(gm.grounding_chunks))
            result = result[:end] + cites + result[end:]
    return result


def _remove_model_citations(text: str) -> str:
    """Remove ``[N]`` patterns the model may have generated on its own."""
    cleaned = re.sub(r"\[(\d+)]", "", text)
    cleaned = re.sub(r"\s*,\s*\.", ".", cleaned)
    cleaned = re.sub(r"\s+\.", ".", cleaned)
    return cleaned


def format_sources_section(text: str, sources: List[WebSource], gm: Optional[GroundingMetadata] = None) -> str:
    """Build a markdown sources section for actually cited sources."""
    if not sources:
        return ""
    used: set = set()
    if gm and gm.grounding_supports:
        for sup in gm.grounding_supports:
            for idx in sup.grounding_chunk_indices:
                used.add(idx)
    if not used:
        nums = sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", text)))
        used = {n - 1 for n in nums}
    if not used:
        used = set(range(min(len(sources), 6)))

    chunks = gm.grounding_chunks if gm else []
    lines, seen = [], set()
    for idx in sorted(used):
        src = chunks[idx].web if idx < len(chunks) else (sources[idx] if idx < len(sources) else None)
        if src and src.uri and src.uri not in seen:
            seen.add(src.uri)
            lines.append(f"[{idx + 1}] {src.to_markdown_link()}")
    if lines:
        return "\n\n---\n\n**Sources:**\n" + "\n".join(lines)
    return ""


# ---------------------------------------------------------------------------
# Core search function
# ---------------------------------------------------------------------------


async def _search_with_gemini_grounding(question: str, model_name: str) -> SearchResult:
    """Call Gemini with Google Search grounding and return a SearchResult."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("google-genai not installed. pip install google-genai")

    api_key = gemini_api_key()
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")

    logger.info(f"Google Search grounding: {question[:80]}...")

    client = genai.Client(api_key=api_key, http_options=gemini_http_options())
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool], temperature=1.0)

    system_prompt = (
        "You are a medical and scientific research expert. "
        "Search ONLY for peer-reviewed scientific publications, systematic reviews, "
        "meta-analyses, clinical trial results, and official sources (WHO, CDC, NIH, EMA). "
        "Prioritize results from PubMed, Google Scholar, Cochrane Library, and major medical journals. "
        "Do NOT use general health websites, forums, or non-academic sources. "
        "Provide a detailed, evidence-based answer with specific statistics, prevalence rates, "
        "and study findings. "
        "Do NOT add citation numbers yourself — they will be added automatically."
    )
    full_prompt = f"{system_prompt}\n\nQuestion: {question}"

    try:
        response = await client.aio.models.generate_content(model=model_name, contents=full_prompt, config=config)

        if hasattr(response, "text"):
            raw = response.text
        elif hasattr(response, "candidates") and response.candidates:
            parts = getattr(response.candidates[0].content, "parts", [])
            raw = "\n".join(getattr(p, "text", "") for p in parts)
        else:
            raw = str(response)

        text = extract_text_from_content(raw)
        gm = extract_grounding_metadata(response)
        unique_sources, seen = [], set()
        for ch in gm.grounding_chunks:
            if ch.web.uri and ch.web.uri not in seen:
                seen.add(ch.web.uri)
                unique_sources.append(ch.web)

        return SearchResult(
            answer=text,
            search_queries=gm.web_search_queries,
            sources=unique_sources,
            grounding_metadata=gm,
            model=model_name,
            question=question,
        )
    except Exception as exc:
        logger.error(f"Google Search error: {exc}")
        return SearchResult(
            answer=f"Error searching with Google: {exc}",
            model=model_name,
            question=question,
            status="error",
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------


@tool
async def google_search_grounding(question: str) -> str:
    """Search the web using Google Search for real-time information.

    Use this tool for recent developments, regulatory updates, drug approvals,
    clinical news, or general medical information not available in academic
    literature databases.

    Args:
        question: The question to search the web for.

    Returns:
        A JSON string with the answer, sources, and search queries.
    """
    model_name = get_domain_settings().GOOGLE_SEARCH_MODEL_NAME
    result = await _search_with_gemini_grounding(question, model_name)

    if result.status == "success" and result.grounding_metadata:
        clean = _remove_model_citations(result.answer)
        cited = add_inline_citations_from_grounding(clean, result.grounding_metadata)
        sources_md = format_sources_section(cited, result.sources, result.grounding_metadata)
        answer = cited + sources_md
    else:
        answer = result.answer

    response = {
        "question": question,
        "answer": answer,
        "sources": [{"title": s.title, "url": s.uri} for s in result.sources],
        "search_queries": result.search_queries,
        "status": result.status,
    }
    if result.error:
        response["error"] = result.error

    return json.dumps(response, indent=2)
