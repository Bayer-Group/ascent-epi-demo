"""LLM connector contract.

Connectors are thin transport adapters: they turn a prompt into raw model text.
All medical-coding logic (filter prompt construction, include/exclude parsing,
grounded-search prompts) lives in the services layer — not here. This keeps the
provider layer reusable and makes the later AWS-decoupling swap mechanical.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMConnector(Protocol):
    """A provider that returns a JSON string for a prompt.

    Implementations return the raw JSON payload the model produced (as a string,
    to be parsed by the caller). This mirrors the historical handler contract
    where the model is asked for a JSON object and the first ``{...}`` span is
    returned verbatim.
    """

    async def complete_json(self, prompt: str) -> str:
        """Send ``prompt`` and return the model's JSON response as a string."""
        ...


@runtime_checkable
class GroundedSearchConnector(Protocol):
    """A provider that answers a prompt with live web-search grounding.

    ``schema`` is a pydantic model class constraining the response shape; the
    result is the parsed dict when the provider returns structured output, or
    the raw response text otherwise. Providers are selected via
    ``get_grounded_search_connector()`` — services never name a vendor.
    """

    async def grounded_search(self, prompt: str, schema: type | None = None) -> dict | str:
        """Run a web-search-grounded completion and return dict (parsed) or raw text."""
        ...
