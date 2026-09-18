import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from snowflake.connector.cursor import SnowflakeCursor

from ascent_domain.omop.external_calls import MedicalCoder
from ascent_domain.omop.models.inference.rag import RAGProcessor
from ascent_domain.omop.schemas.constants import CodingType
from ascent_platform.config.runtime import settings
from ascent_platform.warehouse.session import execute_query

logger = logging.getLogger(__name__)


class MedicalSQLProcessor:
    def __init__(self, recommender=Optional[None], assistant=None, codes_cache=None):
        if not recommender:
            self.recommender = MedicalCoder(settings.AZURE_CLIENT_ID, settings.AZURE_CLIENT_SECRET)
        else:
            self.recommender = recommender

        self.assistant = assistant
        self.database_concepts = {}
        self.concept_not_found = []
        self.codes_cache = codes_cache  # Optional MedicalCodesCache instance

    def get_or_create_database_concepts(self, database_name):
        """
        Get or create the concept lists for a specific database name.
        Args:
            database_name (str): The name of the database.

        Returns:
            dict: A dictionary containing lists for 'condition', 'procedure', 'drug', 'measurement', and 'drug_class'.
        """
        if database_name not in self.database_concepts:
            self.database_concepts[database_name] = {
                "condition": [],
                "procedure": [],
                "drug": [],
                "measurement": [],
                "drug_class": [],
                "observation": [],
                "visit": [],
                "provider": [],
            }
        return self.database_concepts[database_name]

    def restructure_data(self, original_data):
        """
        Restructures the original data into a new format.

        :param original_data: A dictionary containing the original data structure
        :return: A list of dictionaries with the restructured data
        """
        categories = ["condition", "drug", "drug_class", "measurement", "observation", "procedure", "visit"]

        return [
            {"category": category, "name": name, "value": values}
            for category in categories
            for item in original_data.get(category, [])
            for name, values in item.items()
        ]

    def process_and_restructure(self, database_name):
        """
        Processes the database concepts and restructures the data.

        :param database_name: The name of the database to process
        :return: A dictionary with the restructured data
        """
        original_data = self.get_or_create_database_concepts(database_name)
        return self.restructure_data(original_data)

    @staticmethod
    async def get_dictionaries(db_session, domain_id):

        # Get cached result
        # cached_result = await cache.get(cache_key)
        # if cached_result is not None:
        #     return cached_result

        # Template SQL
        query_template = """
            SELECT DISTINCT c.VOCABULARY_ID
            FROM {table_name} AS t
            JOIN concept AS c ON t.{column_name} = c.concept_id
            WHERE c.VOCABULARY_ID IS NOT NULL;
        """
        # domain mapping
        # visit domain is missing, have a look
        domain_mapping = {
            "condition": ("CONDITION_OCCURRENCE", "condition_source_concept_id"),
            "drug": ("DRUG_EXPOSURE", "drug_source_concept_id"),
            "drug_class": ("DRUG_EXPOSURE", "drug_source_concept_id"),
            "procedure": ("PROCEDURE_OCCURRENCE", "procedure_source_concept_id"),
            "measurement": ("MEASUREMENT", "measurement_source_concept_id"),
            "observation": ("OBSERVATION", "observation_source_concept_id"),
            "visit": ('"VISIT_OCCURRENCE"', "visit_source_concept_id"),
        }

        # Check if the domain_id is valid.
        if domain_id not in domain_mapping:
            raise ValueError(f"Invalid domain_id: {domain_id}")

        # Get the table and column names.
        table_name, column_name = domain_mapping[domain_id]
        # Format
        query = query_template.format(table_name=table_name, column_name=column_name)
        # Execute the query.
        result, columns = await execute_query(db_session, query)
        # Return a list of strings.
        result_list = [row[0] for row in result if row[0] not in ("None", None)]
        # Cache the result
        # await cache.set(cache_key, result_list)

        return result_list

    async def get_replacement_value(
        self, category: str, name: str, preferred_coding_system: CodingType = CodingType.STANDARD_CODING, db_session: Optional[SnowflakeCursor] = None
    ) -> Optional[Dict[str, Any]]:
        database_concepts = self.get_or_create_database_concepts_for_category(category, db_session)
        concept = self.find_concept_by_name(database_concepts, name)
        is_drug_class = category == "drug_class"

        if concept:
            return concept[name] if is_drug_class else concept

        return await self.fetch_and_store_new_concept(category, name, preferred_coding_system, db_session)

    def get_or_create_database_concepts_for_category(self, category: str, db_session) -> List[Dict[str, Any]]:
        database_name = db_session.connection.database
        return self.get_or_create_database_concepts(database_name)[category]

    @staticmethod
    def find_concept_by_name(database_concepts: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
        name_lower = name.lower()
        for concept_dict in database_concepts:
            for key, value in concept_dict.items():
                if key.lower() == name_lower:
                    return {key: value}
        return None

    async def fetch_and_store_new_concept(self, category: str, name: str, preferred_coding_system, db_session) -> Dict[str, Any]:
        medical_coder_params = await self.build_medical_coder_params(category, name, preferred_coding_system, db_session)
        entity_codes = await self.get_codes(medical_coder_params)

        self.store_new_concept(name, category, entity_codes, db_session)
        return entity_codes

    async def build_medical_coder_params(self, category: str, name: str, preferred_coding_system, db_session) -> Dict[str, Any]:
        # Get the database name from the session
        database_name = db_session.connection.database

        # Set top_k based on category
        top_k = 10000 if category.lower() == "drug" else 500

        return {
            "query": name,
            "domain_ids": [category.title()],
            "top_k": top_k,
            "llm_filter": "chatgpt",
            "cosine_similarity": 0.65,
            "standard_concept": "S" if preferred_coding_system != CodingType.SOURCE_CODING else None,
            "vocabulary": await self.get_dictionaries(db_session, category) if preferred_coding_system == CodingType.SOURCE_CODING else None,
            "database": database_name,
        }

    def store_new_concept(self, name: str, category: str, entity_codes: Dict[str, Any], db_session):
        database_concepts = self.get_or_create_database_concepts_for_category(category, db_session)
        is_drug_class = category == "drug_class"

        if is_drug_class:
            drug_concepts = self.get_or_create_database_concepts_for_category("drug", db_session)
            drug_concepts_list = [{key: value} for key, value in entity_codes.items()]
            database_concepts.append({name: entity_codes})
            drug_concepts.extend(drug_concepts_list)
        else:
            database_concepts.append(entity_codes)

    async def get_codes(self, medical_coder_params):
        """
        Get medical codes, using cache if available.

        Args:
            medical_coder_params: Parameters for medical coder API call

        Returns:
            Dictionary of medical codes
        """
        cache_state = "ENABLED" if self.codes_cache is not None else "DISABLED"
        logger.info("[CODES] get_codes() query=%r cache=%s", medical_coder_params.get("query"), cache_state)

        # If cache is enabled, try to get from cache first
        if self.codes_cache is not None:
            # Extract parameters for cache lookup
            query = medical_coder_params.get("query")
            domain_ids = medical_coder_params.get("domain_ids", [])
            domain_id = domain_ids[0] if domain_ids else "unknown"
            coding_system = "Standard" if medical_coder_params.get("standard_concept") == "S" else "Source"
            database_name = medical_coder_params.get("database", "UNKNOWN")
            vocabulary = medical_coder_params.get("vocabulary")
            top_k = medical_coder_params.get("top_k", 500)
            llm_filter = medical_coder_params.get("llm_filter")
            cosine_similarity = medical_coder_params.get("cosine_similarity")

            # Try cache lookup
            cached_codes = self.codes_cache.get_codes(
                query=query,
                domain_id=domain_id,
                coding_system=coding_system,
                database_name=database_name,
                vocabulary=vocabulary,
                top_k=top_k,
                llm_filter=llm_filter,
                cosine_similarity=cosine_similarity,
            )

            if cached_codes is not None:
                logger.info(f"[CACHE HIT] Using cached codes for query='{query}', domain='{domain_id}'")
                return cached_codes
            else:
                logger.info(f"[CACHE MISS] Calling medical coder API for query='{query}', domain='{domain_id}'")

        # Cache miss or cache disabled - call API
        codes = await self.recommender.get_medical_codes(**medical_coder_params)

        # Store in cache if enabled AND codes are not empty
        # IMPORTANT: Do NOT cache empty results, so future runs can retry with the API
        if self.codes_cache is not None and codes is not None:
            query = medical_coder_params.get("query")
            domain_ids = medical_coder_params.get("domain_ids", [])
            domain_id = domain_ids[0] if domain_ids else "unknown"
            coding_system = "Standard" if medical_coder_params.get("standard_concept") == "S" else "Source"
            database_name = medical_coder_params.get("database", "UNKNOWN")
            vocabulary = medical_coder_params.get("vocabulary")
            top_k = medical_coder_params.get("top_k", 500)
            llm_filter = medical_coder_params.get("llm_filter")
            cosine_similarity = medical_coder_params.get("cosine_similarity")

            # Check if we actually got any codes
            num_codes = len(codes.get(query, []))

            if num_codes > 0:
                logger.info(f"[CACHE STORE] About to store codes in cache: query='{query}', domain='{domain_id}', num_codes={num_codes}")

                self.codes_cache.store_codes(
                    query=query,
                    domain_id=domain_id,
                    coding_system=coding_system,
                    database_name=database_name,
                    codes=codes,
                    vocabulary=vocabulary,
                    top_k=top_k,
                    llm_filter=llm_filter,
                    cosine_similarity=cosine_similarity,
                )
                logger.info(f"[CACHE STORE] Stored codes in cache for query='{query}', domain='{domain_id}'")
            else:
                logger.warning(f"[CACHE SKIP] NOT caching empty result for query='{query}', domain='{domain_id}' - future runs can retry")

        return codes

    async def get_ids_from_names(self, match, preferred_coding_system, db_session):
        category, name = match
        name_lower = name.lower()

        replacement_value = await self.get_replacement_value(
            category,
            name_lower,
            preferred_coding_system=preferred_coding_system,
            db_session=db_session,
        )

        if category == "drug_class":
            replacement_value = {name_lower: self.flatten_list(replacement_value)}

        return self.format_replacement_result(category, name_lower, replacement_value)

    @staticmethod
    def flatten_list(data: dict) -> list:
        """Flattens a dictionary of lists into a single list."""
        return [item for sublist in data.values() for item in sublist]

    def format_replacement_result(self, category, name, replacement_value):
        if len(replacement_value[name]) > 0 and "error" not in replacement_value[name]:
            concept_ids = [str(concept["CONCEPT_ID"]) for concept in replacement_value[name]]
            return ",".join(concept_ids)
        else:
            self.concept_not_found.append(name)
            raise ConceptNotFoundError(category, name)

    async def post_process_sql_query(
        self,
        sql_text: str,
        max_retries: int = 5,
        preferred_coding_system: CodingType = CodingType.STANDARD_CODING,
        rag_agent: Optional[RAGProcessor] = None,
        db_session: Optional[SnowflakeCursor] = None,
        custom_code_mappings: Optional[Dict[Tuple[str, str], str]] = None,
    ) -> str:
        """for docs on custom_code_mappings see get_codes_for_entity_type_value_pairs"""
        if not sql_text:
            logger.error("SQL text is empty.  Cannot process.")
            raise EmptySQLTextError("SQL text cannot be empty.")

        # # Pattern matches any text that follows after @ (e.g., from "[condition@atopic dermatitis]"
        pattern = r"\[([a-zA-Z_]+)@([a-zA-Z0-9_/\-\(\)\'\\ ]+)\]"
        attempts = 0
        current_sql = sql_text
        last_error = None

        # Handle SOURCE_CODING conversion once, before the loop
        if preferred_coding_system == CodingType.SOURCE_CODING:
            current_sql = self.replace_concept_id_to_source_concept_id(current_sql)
            logger.info("Replaced XXX_concept_id to XXX_source_concept_id for SOURCE_CODING")

        # Lowercase custom_code_mappings once, before the loop
        if custom_code_mappings:
            custom_code_mappings = {(entity_type, value.lower()): code for (entity_type, value), code in custom_code_mappings.items()}

        while attempts <= max_retries:
            try:
                matches = list(set(re.findall(pattern, str(current_sql))))

                # list of tuples with entity type and value
                # example: matches = [('condition', 'acute ischemic stroke')]
                modified_sql = await self.get_codes_for_entity_type_value_pairs(
                    matches,
                    current_sql,
                    preferred_coding_system=preferred_coding_system,
                    db_session=db_session,
                    custom_code_mappings=custom_code_mappings,
                )

                # Check if the MODIFIED SQL (after placeholder replacement) contains concept_name patterns
                # This is the self-healing check - if the LLM generated SQL with concept_name = '...',
                # we need to ask it to regenerate
                if not self.is_sql_for_concept_name_in(modified_sql):
                    # This is the successful execution - no invalid patterns found
                    return modified_sql

                logger.info("Generated query found containing a CONCEPT ==. Self-healing...")

                # Make the sql correct by asking LLM to regenerate
                # The corrected SQL becomes the new current_sql for the next iteration
                current_sql = await self.handle_invalid_sql(rag_agent)
                attempts += 1

            except Exception as e:
                last_error = e
                if attempts >= max_retries:
                    logger.error(f"Max retries ({max_retries}) exceeded: {str(e)}")
                    raise last_error
                logger.warning(f"Attempt {attempts + 1} failed: {str(e)}")
                attempts += 1

        # If we've exhausted all retries and haven't returned yet
        if last_error:
            raise last_error
        return current_sql

    async def get_codes_for_entity_type_value_pairs(
        self,
        matches: List[Tuple[str, str]],
        sql_text: str,
        preferred_coding_system: CodingType = CodingType.STANDARD_CODING,
        db_session: Optional[SnowflakeCursor] = None,
        custom_code_mappings: Optional[Dict[Tuple[str, str], str]] = None,
    ) -> str:
        """
        Process all regex matches and replace them in the SQL text.
        Allows for custom code mappings to override default lookups.

        Args:
            matches: List of tuples containing (entity_type, value) pairs
            sql_text: Original SQL query with placeholders
            preferred_coding_system: The coding system to use for lookups
            db_session: Database session for queries
            custom_code_mappings: Optional dictionary mapping (entity_type, value) to code strings
                                 Example:         custom_code_mappings = {
            ('condition', 'dysphagia'): '4310996,4159140',
            ('condition', 'atopic dermatitis'): '1112807, 1112808',
            ('procedure', 'appendectomy'): '2211444,2211445'
        }

        Returns:
            str: Modified SQL query with replacements
        """
        # Create a list to store coroutines for async lookups
        coroutines = []
        replacements = []
        custom_mapping_indices = []

        # Process each match
        for i, match in enumerate(matches):
            # Check if we have a custom mapping for this match
            if custom_code_mappings and match in custom_code_mappings:
                # Mark this index as having a custom mapping
                custom_mapping_indices.append(i)
                # Add a placeholder for this position
                coroutines.append(None)
                # Log the use of custom mapping (without all codes for cleaner logs)
                num_codes = len(custom_code_mappings[match].split(",")) if custom_code_mappings[match] else 0
                logger.info(f"Using custom mapping for {match[0]}@{match[1]} ({num_codes} concepts)")
            else:
                # Use the default lookup mechanism
                coroutines.append(
                    self.get_ids_from_names(
                        match,
                        preferred_coding_system=preferred_coding_system,
                        db_session=db_session,
                    )
                )

        # Execute all non-custom lookups asynchronously.
        # return_exceptions=True prevents one failed lookup (e.g. procedure@aflibercept not
        # found in this database) from aborting sibling coroutines that succeeded (e.g.
        # drug@aflibercept). Without this, asyncio.gather propagates the first exception
        # immediately, discarding all other in-flight results.
        raw_results = await asyncio.gather(
            *[coro for coro in coroutines if coro is not None],
            return_exceptions=True,
        )

        # Resolve per-result exceptions.
        # ConceptNotFoundError → "-1" so the placeholder produces no rows but doesn't abort.
        # Any other unexpected exception is re-raised immediately.
        non_custom_results = []
        for result in raw_results:
            if isinstance(result, ConceptNotFoundError):
                logger.warning(
                    f"Concept not found during parallel lookup: {result}. Placeholder will be replaced with -1 (returns no rows for this domain)."
                )
                non_custom_results.append("-1")
            elif isinstance(result, Exception):
                raise result
            else:
                non_custom_results.append(result)

        # Merge custom mappings with async results
        result_index = 0
        for i in range(len(matches)):
            if i in custom_mapping_indices:
                # Use the custom mapping
                match = matches[i]
                replacements.append(custom_code_mappings[match])
            else:
                # Use the result from async lookup
                replacements.append(non_custom_results[result_index])
                result_index += 1

        return self.replace_placeholders(matches, replacements, sql_text)

    @staticmethod
    def replace_placeholders(matches, replacements, sql_text):
        """
        Apply the replacements to the SQL text.
        """
        modified_sql = sql_text

        for match, replacement in zip(matches, replacements):
            placeholder = f"[{match[0]}@{match[1]}]"

            # A list of concept ids becomes a comma-separated string.
            if isinstance(replacement, list):
                replacement_str = ", ".join(str(concept_id) for concept_id in replacement)
            else:
                replacement_str = replacement

            # Warn if empty replacement (will result in empty set)
            if replacement_str == "":
                logging.warning(f"No IDs found for {placeholder}, query will return empty set")

            # parenthesis around the ids are already included
            modified_sql = modified_sql.replace(placeholder, replacement_str)

        return modified_sql

    async def handle_invalid_sql(self, rag: RAGProcessor):
        """
        Handle the case where the SQL does not meet the required criteria.

        This self-healing mechanism asks the LLM to regenerate the SQL without
        using 'concept_name =' patterns which are invalid in our system.

        The fix uses rag.default_assistant consistently for both adding the message
        and getting the response, maintaining proper conversation context.
        """
        prompt = """Your generated SQL query doesn't meet the requirements.
            Please correct the sql query based on the previously given instructions.
            [!IMPORTANT] Do not include conditions, such as 'WHERE concept_name IN ...' or  'WHERE concept_name = "..."'
            [!IMPORTANT] Do not return anything except the sql query"""

        # Check if rag and its default_assistant are available
        if rag is None or rag.default_assistant is None:
            logger.warning("RAG processor or default_assistant not available for self-healing. Raising error.")
            raise ValueError("Generated SQL contains invalid 'concept_name =' pattern and self-healing is not available.")

        # Use the SAME assistant for both adding the message and getting the response
        # This maintains proper conversation context
        logger.info("Attempting self-healing: asking LLM to regenerate SQL without concept_name patterns")
        rag.default_assistant.add_message(role="user", message=prompt)
        corrected_response = await rag.default_assistant.get_response()

        # Parse the corrected SQL from the response
        corrected_sql = self.parse_sql_from_response(corrected_response)
        logger.info("Self-healing successful: received corrected SQL")

        return corrected_sql

    @staticmethod
    def parse_sql_from_response(resp=""):
        if resp is None:
            resp = ""

        # Try standard patterns first
        pattern1 = r"(?:Snowflake )?SQL query:\s*\n\n(.*?);"
        # Pattern 2 should require closing ```
        pattern2 = r"(?:```sql|```)\s*([\s\S]*?)\s*```"

        # Fallback pattern - just look for content after ```sql
        pattern3 = r"```sql\s*([\s\S]*$)"

        match1 = re.search(pattern1, resp, re.DOTALL)
        match2 = re.search(pattern2, resp, re.DOTALL)
        match3 = re.search(pattern3, resp, re.DOTALL)

        if match2:
            logger.debug("Found properly formatted SQL with markdown tags")
            return match2.group(1)
        elif match1:
            logger.debug("Found SQL query with explicit 'SQL query:' prefix. Adding ; at the end")
            return match1.group(1) + ";"
        elif match3:
            logger.warning("Found SQL but missing proper closing tags - using fallback pattern")
            return match3.group(1)
        else:
            logger.error(f"No SQL found in response. Response length: {len(resp)}")
            raise NoSQLFoundError("No SQL code found in the response", response=resp)

    @staticmethod
    def parse_json_from_response(resp=""):
        pattern = r"(?:```json|```) ?\n([\s\S]+?)\n```"
        match = re.search(pattern, resp)
        if match:
            return match.group(1)
        else:
            logger.info("No JSON code found.")

    @staticmethod
    def save_string_to_file(text="", filename="log.txt"):
        with open(filename, "a") as text_file:
            text_file.write("\n")
            if text:
                text_file.write(text)

    @staticmethod
    def is_sql_for_concept_name_in(sql_text):
        pattern = r"\b(?:\w+\s*\(\s*)?concept_name\s*(?:\)\s*)?\s*(?:=|IN|LIKE|ILIKE)\s*(?:\(?\s*'[^']+'(?:\s*,\s*'[^']+')*\s*\)?|'%[^']+%')"
        match = re.search(pattern, str(sql_text), re.IGNORECASE)
        return bool(match)

    @staticmethod
    def replace_concept_id_to_source_concept_id(sql_text):
        # this does not include visit on purpose, since for visit we use the omop codes
        pattern = re.compile(r"(condition|drug|drug_class|procedure|measurement|observation)_concept_id", re.IGNORECASE)

        # Replacement function
        def replacement(match):
            concept_type = match.group(1)  # Extract the concept type (e.g., "condition", "drug", etc.)
            if match.group().islower():
                return f"{concept_type}_source_concept_id"
            else:
                return f"{concept_type.upper()}_SOURCE_CONCEPT_ID"

        # Apply the replacement
        return pattern.sub(replacement, sql_text)

    @staticmethod
    def extract_entities_from_sql(sql_text: str) -> List[Tuple[str, str]]:
        """
        Extract all unique (entity_type, value) pairs from SQL template placeholders.

        Args:
            sql_text: SQL query with placeholders like [condition@diabetes]

        Returns:
            List of unique (entity_type, value) tuples with lowercased values

        Example:
            >>> MedicalSQLProcessor.extract_entities_from_sql(
            ...     "SELECT * FROM t WHERE condition_concept_id IN ([condition@diabetes])"
            ... )
            [('condition', 'diabetes')]
        """
        pattern = r"\[([a-zA-Z_]+)@([a-zA-Z0-9_/\-\(\)\'\\ ]+)\]"
        matches = re.findall(pattern, str(sql_text))
        return list(set((entity_type, value.lower()) for entity_type, value in matches))

    async def batch_code_entities(
        self,
        entities: List[Tuple[str, str]],
        preferred_coding_system: CodingType,
        db_session: Optional[SnowflakeCursor] = None,
    ) -> Dict[Tuple[str, str], str]:
        """
        Code multiple entities in a single pass, returning a mapping.

        This is more efficient than calling post_process_sql_query multiple times
        as it reuses the same MedicalSQLProcessor instance and its caches.

        Args:
            entities: List of (entity_type, value) tuples to code
            preferred_coding_system: The coding system to use
            db_session: Optional database session

        Returns:
            Dict mapping (entity_type, value) to comma-separated concept ID strings

        Example:
            >>> processor = MedicalSQLProcessor()
            >>> mappings = await processor.batch_code_entities(
            ...     [('condition', 'diabetes'), ('drug', 'metformin')],
            ...     CodingType.STANDARD_CODING
            ... )
            >>> # Returns: {('condition', 'diabetes'): '201820,4182210', ('drug', 'metformin'): '1503297'}
        """
        code_mappings = {}

        async def _code_one(entity_type: str, value: str) -> tuple:
            """Code a single entity, returning ((type, value), concept_ids)."""
            try:
                concept_ids = await self.get_ids_from_names(
                    match=(entity_type, value),
                    preferred_coding_system=preferred_coding_system,
                    db_session=db_session,
                )
                logger.info(f"Coded entity {entity_type}@{value}: {len(concept_ids.split(',')) if concept_ids else 0} concepts")
                return (entity_type, value), concept_ids
            except ConceptNotFoundError as e:
                logger.warning(f"Could not code entity {entity_type}@{value}: {e}")
                return (entity_type, value), ""
            except Exception as e:
                logger.warning(f"Error coding entity {entity_type}@{value}: {e}")
                return (entity_type, value), ""

        results = await asyncio.gather(*[_code_one(et, val) for et, val in entities])

        for key, concept_ids in results:
            code_mappings[key] = concept_ids

        return code_mappings


class NoSQLFoundError(Exception):
    """Custom exception raised when SQL code cannot be found in a response."""

    def __init__(self, message="No SQL code found", response=None):
        self.message = message
        self.response = response
        super().__init__(self.message)

    def __str__(self):
        """Custom string representation of the error that includes the response"""
        error_msg = f"Error: {self.message}\n"
        if self.response:
            error_msg += f"Response content:\n{self.response}"
        return error_msg


class EmptySQLTextError(Exception):
    """Custom exception raised when the SQL text is empty."""

    pass


class ConceptNotFoundError(Exception):
    """Raised when no concepts are found for a given name."""

    def __init__(self, category: str, name: str):
        self.category = category
        self.name = name
        super().__init__(f"No concept IDs found for: {category}@{name}")
