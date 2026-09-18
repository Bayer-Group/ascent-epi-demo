import asyncio
import logging
from threading import Lock
from typing import List, Optional

from sqlalchemy import MetaData, Table

from ascent_domain.models.data_definitions import CodingSystem, Concept, MedicalConcept, MultiSearchConcept
from ascent_platform.config.runtime import get_runtime_settings
from ascent_platform.external import AscentClient
from ascent_platform.external.medical_coder import unpack_concept

logger = logging.getLogger(__name__)


class ConceptSearch(AscentClient):
    def __init__(
        self,
        azure_client_id: str,
        azure_client_secret: str,
        azure_tenant_id: str | None = None,
        base_url: str | None = None,
        metadata_cache: Optional[dict] = None,
        metadata_cache_lock: Optional[Lock] = None,
    ) -> None:
        super().__init__(
            base_url=base_url if base_url is not None else get_runtime_settings().MEDICAL_CODER_BASE_URL,
            azure_client_id=azure_client_id,
            azure_client_secret=azure_client_secret,
            azure_tenant_id=azure_tenant_id,
        )
        self.metadata_cache = metadata_cache or {}
        self.metadata_cache_lock = metadata_cache_lock or Lock()

    def get_table_metadata(self, engine, table_name: str) -> Optional[Table]:
        with self.metadata_cache_lock:
            if table_name not in self.metadata_cache:
                meta = MetaData()
                self.metadata_cache[table_name] = Table(table_name, meta, autoload_with=engine)
            return self.metadata_cache[table_name]

    async def search_concepts(
        self, query: str, domain_id: str = None, vocabulary_ids: list[str] | None = None, standard_concept=None, database=None
    ) -> List[Concept]:
        logger.info(f"search_concepts: query={query}, domain_id={domain_id}, vocabulary_ids={vocabulary_ids}")
        params = {
            "query": query,
            "domain_id": domain_id,
            "top_k": 1000,
            "cosine_similarity": 0.65,
            "vocabulary": vocabulary_ids if vocabulary_ids else [],
            "standard_concept": standard_concept,
            "database": database,
        }

        logger.info(f"search_concepts: params={params}")

        resp = await self.query_ascent_api("get-medical-codes", payload=params)
        mapped_concepts = self.construct_response(resp)
        return mapped_concepts

    async def bulk_concept_search(self, request) -> List[Concept]:
        logger.info(f"bulk_concept_search: request={request!r}")
        resp = await self.query_ascent_api("bulk-concept-search", payload=request.model_dump(mode="json"))
        if len(resp) > 0:
            return resp
        return []

    async def multi_search_concepts(
        self,
        entities: List[MultiSearchConcept],
        database: str = "SYNTHETIC_EHR_OMOP",
        coding_system: CodingSystem = "Standard",
    ) -> List[MedicalConcept]:
        sanitized_entities = {MultiSearchConcept(name=entity.name, domain_id=entity.domain_id) for entity in entities}
        logger.info(f"multi_search_concepts: entities={sanitized_entities}, database={database}, coding_system={coding_system}")

        codeLists = await self._build_codelist(sanitized_entities, coding_system, database)

        response = [MedicalConcept(name=entity.name, category=entity.domain_id.lower(), concepts=concepts) for entity, concepts in codeLists]
        return response

    def construct_response(self, resp: dict) -> list[Concept]:
        """Constructs the response data based on the API response.

        Args:
            resp (dict): The response data from the API, where each key is a query and value is a list of concepts.

        Returns:
            list[Concept]: A list of medical codes or drugs.
        """
        if not resp:
            return []  # Return an empty list if the response is empty

        # There is only one key in the response
        key = next(iter(resp))
        concept_data = resp[key]
        return self.extract_concept_data(concept_data)

    def extract_concept_data(self, concept_data: list) -> list[Concept]:
        """Extracts relevant data from the API response.

        Args:
            concept_data (list): The list of concepts.

        Returns:
            list[dict[str, str]]: A list of dictionaries containing concept information.
        """

        # Unpacked by the shared helper, then adapted to the model here -- the
        # type is the only thing this client does differently. SIMILARITY_SCORE
        # is not passed: Concept does not declare it, so pydantic dropped it
        # silently, and constructing it looked like it carried a score it never
        # did.
        concepts = [Concept(**{k: v for k, v in unpack_concept(c).items() if k in Concept.model_fields}) for c in concept_data]

        logger.info(f"Extracted concepts: {concepts}")

        return concepts

    async def _fetch_concept_results(
        self, entities: MultiSearchConcept, coding_system: str, database: str
    ) -> tuple[MultiSearchConcept, list[Concept]]:
        """
        Asynchronously search for a given concept and return a tuple with the original concept and the search result.

        Args:
            concept (Concept): The concept to search for.
            coding_system (str): The coding system to use ("Standard" or other).
            database (str): The database to search in.

        Returns:
            tuple[Concept, list[Concept]]: A tuple containing the original concept and the search result.
        """
        result = await self.search_concepts(
            query=entities.name,
            domain_id=entities.domain_id,
            standard_concept="S" if coding_system == "Standard" else None,
            database=database,
        )
        return entities, result

    async def _build_codelist(
        self, entities: List[MultiSearchConcept], coding_system: str, database: str
    ) -> List[tuple[MultiSearchConcept, list[Concept]]]:
        """
        Asynchronously build a code list for the given concepts.

        Args:
            concepts (List[Concept]): The list of concepts to search for.
            coding_system (str): The coding system to use ("Standard" or other).
            database (str): The database to search in.

        Returns:
            List[tuple[Concept, list[Concept]]]: A list of tuples containing the original concept and the search result.
        """
        tasks = [self._fetch_concept_results(concept, coding_system, database) for concept in entities]
        results = await asyncio.gather(*tasks)
        return results
