import logging
from typing import Any

from ascent_platform.config.runtime import get_runtime_settings
from ascent_platform.external.ascent_client import AscentClient

logger = logging.getLogger(__name__)

# Payload keys carrying the clinical term a user asked about: the condition or
# drug someone is looking up, which must not be logged in clear text. Redacting
# the value rather than dropping the log keeps the request shape -- top_k,
# vocabulary, domain_ids, the flags -- which is what the log is actually for
# when a coder call misbehaves.
_CLINICAL_PAYLOAD_KEYS = frozenset({"query", "codes"})


def _loggable(payload: dict[str, Any]) -> dict[str, Any]:
    """The payload with clinical terms replaced by their size."""

    def _redact(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return f"<redacted {len(value)} items>"
        return f"<redacted {len(str(value))} chars>"

    return {key: _redact(value) if key in _CLINICAL_PAYLOAD_KEYS else value for key, value in payload.items()}


def unpack_concept(concept: dict) -> dict:
    """The concept fields every caller of the medical coder reads.

    Several clients consume this wire shape and differ only in what they do
    with the result: dicts here, dicts without IS_VALID in the OMOP client,
    Concept models in ConceptSearch. That difference belongs at the edge; the
    unpacking does not.
    """
    data = concept["CONCEPT_DATA"]
    return {
        "CONCEPT_ID": data["CONCEPT_ID"],
        "CONCEPT_NAME": data["CONCEPT_NAME"],
        "CONCEPT_CODE": data["CONCEPT_CODE"],
        "VOCABULARY_ID": data["VOCABULARY_ID"],
        "PATIENT_COUNT": data["PATIENT_COUNT"],
        "SIMILARITY_SCORE": concept["SCORE"],
    }


class MedicalCoder(AscentClient):
    def __init__(
        self,
        azure_client_id: str | None = None,
        azure_client_secret: str | None = None,
        azure_tenant_id: str | None = None,
        base_url: str | None = None,
    ) -> None:
        # Resolved on construction, not as import-time argument defaults, which
        # would read and validate the environment the moment this module is
        # imported.
        runtime = get_runtime_settings()
        super().__init__(
            base_url=base_url if base_url is not None else runtime.MEDICAL_CODER_BASE_URL,
            azure_client_id=azure_client_id if azure_client_id is not None else runtime.AZURE_CLIENT_ID,
            azure_client_secret=(azure_client_secret if azure_client_secret is not None else runtime.AZURE_CLIENT_SECRET),
            azure_tenant_id=azure_tenant_id,
        )

    async def get_medical_codes(self, query: str, **kwargs) -> dict[str, list[dict[str, Any]]]:
        """Fetches medical codes from Ascent API.

        Args:
            query (str): The search query for medical codes.
            **kwargs: Additional keyword arguments for the API request.

        Returns:
            dict[list[str], list[dict[str, Any]]]: A dictionary containing the query and a list of medical codes or drugs.
        """

        payload = {"query": query, **kwargs}
        logger.info("Payload: %s", _loggable(payload))
        resp = await self.query_ascent_api("get-medical-codes", payload=payload)

        if resp is None:
            logger.error("Medical codes for a %s-char query, domain %s, cannot be retrieved", len(query), kwargs.get("domain_id"))
            return {query: []}

        return self.construct_response(resp)

    async def get_medical_codes_reasoning(self, query: str, **kwargs) -> dict[str, Any]:
        """Fetches medical codes with LLM reasoning from Ascent API.

        Args:
            query: The search query for medical codes.
            **kwargs: Additional keyword arguments for the API request.

        Returns:
            Dictionary with 'results' (concept lists), 'llm_reasoning' (per-domain
            include/exclude rationale and counts), and 'lts_used' flag.
        """
        payload = {"query": query, **kwargs}
        logger.info("Reasoning payload: %s", _loggable(payload))
        resp = await self.query_ascent_api("get-medical-codes-reasoning", payload=payload)

        if resp is None:
            logger.error("Medical codes reasoning for a %s-char query cannot be retrieved", len(query))
            return {"results": {query: []}, "llm_reasoning": {}, "lts_used": False}

        return {
            "results": {k: self.extract_concept_data(v) for k, v in resp.get("results", {}).items()},
            "llm_reasoning": resp.get("llm_reasoning", {}),
            "lts_used": resp.get("lts_used", False),
        }

    def construct_response(self, resp: dict) -> dict[str, list[dict[str, Any]]]:
        """Constructs the response data based on the API response.

        Args:
            resp (dict): The response data from the API, where each key is a query and value is a list of concepts.

        Returns:
            dict[str, list[dict[str, Any]]]: A dictionary containing the queries and lists of medical codes or drugs.
        """
        # Initialize an empty dictionary to hold the restructured data
        restructured_data = {}

        # Iterate over each key in the response dictionary
        for key, concept_data in resp.items():
            # Apply the extract_concept_data method to each list of concepts
            restructured_data[key] = self.extract_concept_data(concept_data)

        return restructured_data

    async def get_patient_counts(
        self,
        codes: list[dict[str, str]],
        database: str,
        schema: str | None = None,
        standard_concept: str | None = None,
    ) -> dict:
        """Fetch patient counts for a list of (code, vocabulary_id) pairs.

        Works for both OMOP and non-OMOP databases. The medical coder service
        automatically detects whether a database is non-OMOP and routes the
        request to LLM-generated SQL when appropriate.

        Args:
            codes: List of dicts with "code" and "vocabulary_id" keys.
            database: Database name, optionally with schema (e.g. "SYNTHETIC_EHR_OMOP.CDM").
            schema: Optional explicit schema name (e.g. "CDM").
            standard_concept: If set, query standard_counts table; otherwise source_counts.
                Only applies to OMOP databases.

        Returns:
            dict: Response with results, database, codes_submitted, and codes_found.
        """
        payload: dict[str, Any] = {
            "codes": codes,
            "database": database,
        }
        if schema is not None:
            payload["schema"] = schema
        if standard_concept is not None:
            payload["standard_concept"] = standard_concept

        logger.info("Patient counts payload: %s", _loggable(payload))
        resp = await self.query_ascent_api("patient-counts", payload=payload)

        if resp is None:
            logger.error("Patient counts for database %s cannot be retrieved", database)
            return {"results": [], "database": database, "codes_submitted": len(codes), "codes_found": 0}

        return resp

    async def validate_codes(
        self,
        codes: list[dict[str, str]],
    ) -> dict:
        """Validate a list of medical codes against the ASCENT OMOP vocabulary.

        Args:
            codes: List of dicts with "code" and "vocabulary_id" keys.

        Returns:
            dict: Response with results, codes_submitted, codes_valid, codes_invalid.
        """
        payload = {"codes": codes}
        logger.info("Validate codes payload: %s", _loggable(payload))
        resp = await self.query_ascent_api("validate-codes", payload=payload)

        if resp is None:
            logger.error("Code validation request returned no response")
            return {"results": [], "codes_submitted": len(codes), "codes_valid": 0, "codes_invalid": len(codes)}

        return resp

    def extract_concept_data(self, concept_data: list) -> list[dict[str, str]]:
        """Extracts relevant data from the API response.

        Args:
            concept_data (list): The list of concepts.

        Returns:
            list[dict[str, str]]: A list of dictionaries containing concept information.
        """
        return [
            {
                **unpack_concept(concept),
                # OMOP/Athena validity; .get keeps compatibility with coder
                # versions that do not send the field (absent -> valid). This key
                # is the one difference between this client and the OMOP one,
                # which deliberately omits it.
                "IS_VALID": concept["CONCEPT_DATA"].get("IS_VALID", True),
            }
            for concept in concept_data
        ]
