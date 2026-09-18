"""LLM search fallback using web-search grounding.

When traditional vector search returns no results, this module asks a
web-search-grounded LLM for relevant medical concepts. The transport is
provider-agnostic: the connector comes from
:func:`get_grounded_search_connector` (selected by
``settings.llm.grounded_search_provider``) and the model from
``settings.llm.grounded_search_model`` — no vendor or model name is hardcoded
here. Prompt building and response normalization are unchanged from the legacy
``llm_search_fallback.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging

import pandas as pd

from ascent_medical_coder.connectors.llm.factory import get_grounded_search_connector
from ascent_medical_coder.core.settings import get_settings
from ascent_medical_coder.prompts.prompts_llm_fallback import get_prompt
from ascent_medical_coder.schemas.medical_concepts import MedicalConceptList

logger = logging.getLogger(__name__)


class LLMSearchFallback:
    """Web-search-grounded LLM fallback for medical-concept and drug searches.

    ``provider`` selects the grounded-search implementation (factory key);
    ``None`` uses the deployment default (``settings.llm.grounded_search_provider``).
    """

    def __init__(self, provider: str | None = None) -> None:
        resolved = provider or get_settings().llm.grounded_search_provider
        self._connector = get_grounded_search_connector(resolved)
        # Recorded on each returned concept; provider-derived, not hardcoded.
        self._source = resolved.title()

    async def __call_llm(self, prompt: str) -> pd.DataFrame:
        response = await self._connector.grounded_search(prompt, schema=MedicalConceptList)

        concepts = self._parse_response(response)
        df = pd.DataFrame(concepts)

        if df.empty:
            logger.warning("No results from grounded LLM search")
            return self._empty_dataframe()

        logger.info(f"LLM search fallback returned {len(df)} results")
        return df

    async def search_medical_concepts(
        self,
        search_text: str,
        domain_id: str | None = None,
        vocabulary: str | None = None,
        standard_concept: str | None = None,
        top_k: int = 10,
        custom_instructions: str | None = None,
    ) -> pd.DataFrame:
        """Search for general medical concepts via the grounded LLM.

        Returns a DataFrame with columns: CONCEPT_NAME, CONCEPT_ID, DOMAIN_ID,
        VOCABULARY_ID, STANDARD_CONCEPT, CONCEPT_CODE, SCORE, SOURCE.
        """
        logger.info(f"Initiating grounded LLM search fallback for query: {search_text}")

        try:
            prompt = get_prompt(
                prompt_type="medical_concept",
                search_text=search_text,
                domain_id=domain_id,
                vocabulary=vocabulary,
                standard_concept=standard_concept,
                top_k=top_k,
                custom_instructions=custom_instructions,
            )
            return await self.__call_llm(prompt=prompt)
        except Exception as e:
            logger.error(f"Error in LLM search fallback: {e!s}", exc_info=True)
            return self._empty_dataframe()

    async def search_drugs(
        self,
        search_text: str,
        is_drug_class: bool = False,
        vocabulary: str | None = None,
        standard_concept: str | None = None,
        top_k: int = 10,
        custom_instructions: str | None = None,
    ) -> pd.DataFrame:
        """Search for drug concepts via the grounded LLM."""
        logger.info(f"Initiating grounded LLM drug search fallback for query: {search_text}")

        try:
            prompt = get_prompt(
                prompt_type="drug_search",
                search_text=search_text,
                vocabulary=vocabulary,
                standard_concept=standard_concept,
                top_k=top_k,
                is_drug_class=is_drug_class,
                custom_instructions=custom_instructions,
            )
            return await self.__call_llm(prompt=prompt)
        except Exception as e:
            logger.error(f"Error in LLM drug search fallback: {e!s}", exc_info=True)
            return self._empty_dataframe()

    def _parse_response(self, response: dict | str) -> list[dict]:
        """Normalize the grounded-search response into concept dicts.

        The connector returns a parsed dict when the response schema was
        honored (``{"concepts": [...]}``), or raw text otherwise.
        """
        try:
            if isinstance(response, dict):
                concepts = response.get("concepts", [])
            else:
                concepts = json.loads(response.strip().strip("```json"))  # noqa: B005 -- preserves legacy strip behavior

            if not isinstance(concepts, list):
                logger.warning(f"Response is not a list, got: {type(concepts)}")
                return []

            normalized_concepts = []
            for concept in concepts:
                if not isinstance(concept, dict):
                    continue

                concept_id = concept.get("CONCEPT_ID")
                if concept_id is None or concept_id == 0:
                    hash_value = hashlib.md5(concept["CONCEPT_NAME"].encode("utf-8")).hexdigest()
                    # Take the first decimal digits from the integer version of the hash
                    concept_id = int(hash_value, 16) % 1000000000

                normalized = {
                    "CONCEPT_NAME": concept.get("CONCEPT_NAME", "Unknown"),
                    "CONCEPT_ID": concept_id,
                    "DOMAIN_ID": concept.get("DOMAIN_ID", "Drug"),
                    "VOCABULARY_ID": concept.get("VOCABULARY_ID", "RxNorm"),
                    "STANDARD_CONCEPT": concept.get("STANDARD_CONCEPT"),
                    "CONCEPT_CODE": concept.get("CONCEPT_CODE", ""),
                    "SCORE": float(concept.get("RELEVANCE_SCORE", concept.get("score", 0.5))),
                    "SOURCE": self._source,
                }
                normalized_concepts.append(normalized)

            return normalized_concepts

        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from grounded LLM response: {response}")
            return []
        except Exception as e:
            logger.error(f"Error normalizing concepts: {e}")
            return []

    @staticmethod
    def _empty_dataframe() -> pd.DataFrame:
        """Return an empty DataFrame with drug concept columns."""
        return pd.DataFrame(
            columns=[
                "CONCEPT_NAME",
                "CONCEPT_ID",
                "DOMAIN_ID",
                "VOCABULARY_ID",
                "STANDARD_CONCEPT",
                "CONCEPT_CODE",
                "RELEVANCE_SCORE",
            ]
        )
