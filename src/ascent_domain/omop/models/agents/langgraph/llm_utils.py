"""
Shared LLM utilities for handling responses from different providers.

Provides helpers for extracting text from Gemini 3 content blocks and
normalising LLM responses across providers.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_text_from_content(content: Any) -> str:
    """Extract plain text from an LLM response.

    Handles:
    - Plain strings
    - Gemini 3 content blocks (list of ``{"type": "text", "text": "..."}`` dicts)
    - LangChain ``AIMessage`` objects
    - Objects with a ``.text`` attribute
    """
    if isinstance(content, str):
        return content

    # Gemini 3 content block list
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                # Skip thinking blocks
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)

    # LangChain message objects
    if hasattr(content, "content"):
        return extract_text_from_content(content.content)

    if hasattr(content, "text"):
        return str(content.text)

    return str(content) if content else ""

