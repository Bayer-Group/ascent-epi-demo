"""Drug-class expansion via a web-search-grounded LLM.

Ported from the old ``search_snowflake_drug.py``. Only ``expand_drug_class_names``
(and its ``DrugClassResult`` schema) is carried over here; the Snowflake search
helpers from the old module belong to a different module and are not needed by
this function. Transport is provider-agnostic — the connector comes from
:func:`get_grounded_search_connector` and the model from settings; no vendor or
model name is hardcoded here.
"""

# ruff: noqa: E501  -- few-shot prompt text is copied verbatim (long lines preserved)
from __future__ import annotations

import json
import logging

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

from ascent_medical_coder.connectors.llm.factory import get_grounded_search_connector

logger = logging.getLogger(__name__)


class DrugClassResult(BaseModel):
    """Schema for structured drug class expansion response."""

    result: list[str]


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def expand_drug_class_names(drug_class: str) -> list[str]:
    """Expand a drug class name into a list of active ingredient names
    using a web-search-grounded LLM.

    Args:
        drug_class: The drug class name to expand (e.g. "anticoagulants").

    Returns:
        A list of active ingredient names belonging to that drug class.
    """
    connector = get_grounded_search_connector()
    prompt = f"""
        Given an input text, your task is to return a complete list of active ingredients that are part of the drug class.

        drug class: This refers to name of group of medications and other compounds that have similar chemical structures, the same mechanism of action, and/or are used to treat the similar diseases.

        Here are a few examples:
        Input text: serotonin reuptake inhibitors
        result: ["fluoxetine", "sertraline", "paroxetine", "citalopram", "escitalopram", "fluvoxamine", "dapoxetine"]

        Input text: anticoagulants
        result: ["heparin", "warfarin", "rivaroxaban", "dabigatran", "apixaban", "edoxaban", "enoxaparin", "fondaparinux"]

        Input text: {drug_class}
    """
    response = await connector.grounded_search(prompt, schema=DrugClassResult)

    if isinstance(response, str):
        parsed = json.loads(response)
        result = parsed.get("result", [])
    elif isinstance(response, dict):
        result = response.get("result", [])
    else:
        logger.warning(f"Unexpected grounded-search response type: {type(response)}")
        return []

    # Handle case where result items might have commas (legacy format)
    if isinstance(result, list) and len(result) == 1 and "," in result[0]:
        result = [name.strip() for name in result[0].split(",")]

    return [name.strip() for name in result if name.strip()]
