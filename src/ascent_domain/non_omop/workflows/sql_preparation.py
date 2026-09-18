import logging
from typing import Any, Dict, List, Optional, Tuple

from ascent_domain.non_omop.decorators.retry_on_exception import retry_on_exception
from ascent_domain.non_omop.exceptions.sql_preparation import (
    MedicalConceptExtractionError,
    MetadataRetrievalError,
    SQLPreparationException,
    SQLPreparationLLMError,
    SQLPreparationValidationError,
    SQLRepairError,
)
from ascent_domain.non_omop.llm_provider import get_llm_service
from ascent_domain.non_omop.medical_coding.sql_postprocessing import get_medical_concepts_from_sql_async, merge_entity_references
from ascent_domain.non_omop.metadata.codes_with_dots import do_medical_entities_contain_dots
from ascent_domain.non_omop.metadata.peek_into_the_data import extract_metadata
from ascent_domain.non_omop.prompts.prompts_pipeline import (
    cohort_prompt_addon,
    get_prompt_cohort_sql_execution,
    get_prompt_sql_execution,
    get_prompt_sql_execution_with_wrong_placeholders,
    get_prompt_sql_preparation,
)
from ascent_domain.non_omop.schemas.cohort import CohortMetadata
from ascent_domain.non_omop.schemas.question_sanity_check import MedicalEntities
from ascent_domain.non_omop.schemas.question_to_analysis import QuestionToAnalysis
from ascent_domain.non_omop.schemas.sql_creation_simplified import (
    ExtractedConcepts,
    SQLCreationSimplified,
    SQLRepair,
)
from ascent_domain.non_omop.schemas.sql_preparation import SQLpreparation as SQLPreparationSchema
from ascent_domain.non_omop.workflows.utils import reformat_sql
from ascent_platform.llm.router import LLMRouter
from ascent_platform.warehouse.sql_executor import SqlExecutor

logger = logging.getLogger(__name__)


class SQLPreparation:
    """
    Service for preparing, generating, and repairing SQL queries using LLMs and metadata.

    This class orchestrates the process of:
      - Generating SQL preparation prompts
      - Executing LLM-based SQL generation
      - Extracting and validating medical concepts
      - Repairing SQL templates with missing placeholders
    """
    MAX_RETRIES = 3

    def __init__(self, model_name: str, executor: SqlExecutor | None = None, llm_service: LLMRouter | None = None) -> None:
        """
        Initialize the SQLPreparationService.

        Args:
            model_name (str): The name of the LLM model to use for SQL generation.
        """
        if executor is None:
            self.executor = SqlExecutor("ASCENT", "PUBLIC")
        else:
            self.executor = executor
        self.model_name = model_name
        self.llm_service = llm_service or get_llm_service()

    async def prepare_sql(
            self,
            database_name: str,
            database_schema: str,
            question_to_analysis: QuestionToAnalysis,
            medical_entities: MedicalEntities,
            tables_with_medical_coding_ontologies: List[Any],
            m_schema_string: str,
            metadata: Any,
            feedback: str | None = None,
            cohort_metadata: CohortMetadata | None = None,
            custom_instruction: str | None = None,
    ) -> SQLCreationSimplified:
        """
        Orchestrates the full SQL preparation pipeline:
          1. Fetches custom instructions
          2. Generates SQL preparation prompt
          3. Gets SQL preparation result from LLM
          4. Extracts relevant metadata
          5. Generates SQL execution prompt
          6. Gets SQL creation template from LLM
          7. Extracts medical concepts
          8. Repairs SQL if placeholders are missing

        Args:
            database_name (str): Name of the database.
            database_schema (str): Name of the schema within the database.
            question_to_analysis (QuestionToAnalysis): Parsed question analysis.
            medical_entities (MedicalEntities): Medical entities extracted from the question.
            tables_with_medical_coding_ontologies (List[Any]): List of tables with medical coding ontologies.
            m_schema_string (str): Schema string for the database.
            metadata (Any): Metadata for the database.
            feedback (str | None): Optional feedback for refining SQL generation.
            cohort_metadata (CohortMetadata | None): Optional cohort metadata for SQL generation.
            custom_instruction (str | None): Optional custom instruction for SQL generation.

        Returns:
            SQLCreationSimplified: The final, validated, and repaired SQL template.

        Raises:
            MetadataRetrievalError: If custom instructions or metadata extraction fails
            SQLPreparationLLMError: If LLM fails during SQL preparation or creation
            SQLPreparationValidationError: If SQL validation fails
            MedicalConceptExtractionError: If medical concept extraction fails
            SQLPreparationException: For any other unexpected errors

        Note:
            SQL repair failures are logged but do not raise exceptions. The method returns
            the best available SQL even if some placeholders remain unresolved.
        """
        try:
            # Step 1: Get custom instruction
            if not custom_instruction:
                try:
                    instruction = await self.get_custom_instructions(self.executor, database_name, database_schema)
                except Exception as e:
                    raise MetadataRetrievalError(
                        database_name=database_name,
                        database_schema=database_schema,
                        metadata_type="custom instructions",
                        error_details=f"Failed to retrieve custom instructions: {str(e)}",
                        original_exception=e,
                    )
            else:
                instruction = custom_instruction

            # Step 2: Generate the SQL preparation prompt
            prompt_sql_preparation = get_prompt_sql_preparation(
                question_to_analysis=question_to_analysis,
                medical_entities=medical_entities,
                tables_with_medical_coding_ontologies=tables_with_medical_coding_ontologies,
                m_schema_str=m_schema_string,
                custom_instruction=instruction
            )

            # Step 3: Fetch the SQL preparation result
            sql_cohort_preparation_json = (await self.get_sql_preparation(prompt_sql_preparation)).model_dump()

            # Step 4: Extract relevant metadata
            try:
                peek_into_db_for_relevant_tables_columns = extract_metadata(sql_cohort_preparation_json, metadata)
            except Exception as e:
                raise MetadataRetrievalError(
                    database_name=database_name,
                    database_schema=database_schema,
                    metadata_type="relevant tables and columns",
                    error_details=f"Failed to extract metadata: {str(e)}",
                    original_exception=e,
                )

            # Step 5: Generate the SQL execution prompt
            prompt_sql_execution = get_prompt_sql_execution(
                sql_cohort_preparation_json=sql_cohort_preparation_json,
                peek_into_db_for_relevant_tables_columns=peek_into_db_for_relevant_tables_columns,
                feedback=feedback
            )

            if cohort_metadata:
                prompt_sql_execution += cohort_prompt_addon(cohort_metadata=cohort_metadata)

            # Step 6: Get SQL creation template
            sql_response = await self.get_sql_creation_template(prompt_sql_execution)

            try:
                contains_dots = bool(do_medical_entities_contain_dots(sql_preparation=sql_response))
            except Exception as e:
                logger.error(f"Failed to detect if the codes contain dots: {str(e)}")
                contains_dots = False

            # Step 7: Extract medical concepts
            try:
                merged_sql, extracted_concepts = await self.get_medical_concepts(sql_response.sql_code_snippet, codes_with_dots=contains_dots, database=database_name)
                sql_response.sql_code_snippet = merged_sql
                sql_response.extracted_concepts = extracted_concepts
            except Exception as e:
                raise MedicalConceptExtractionError(
                    sql_snippet=sql_response.sql_code_snippet,
                    error_details=f"Failed to extract medical concepts: {str(e)}",
                    original_exception=e,
                )

            # Step 8: Detect if there are missing medical concepts and repair
            final_sql = await self.check_and_repair(sql_response=sql_response, sql_preparation_json=sql_cohort_preparation_json,
                                                    cohort_metadata=cohort_metadata)

            # Step 9: Reformat SQL (non-critical step)
            try:
                reformatted_query = reformat_sql(final_sql.sql_code_snippet).sql
                final_sql.sql_code_snippet = reformatted_query
            except Exception as e:
                logger.warning(f"Error during SQL reformatting: {repr(e)}. Using unformatted  SQL.")
                # Not critical - continue with unformatted SQL

            return final_sql

        except (MetadataRetrievalError, SQLPreparationLLMError, SQLPreparationValidationError,
                MedicalConceptExtractionError):
            # Re-raise custom exceptions as-is
            raise
        except Exception as e:
            # Catch any unexpected errors
            raise SQLPreparationException(
                message=f"Unexpected error during SQL preparation pipeline: {str(e)}",
                context={
                    "database_name": database_name,
                    "database_schema": database_schema,
                    "question": question_to_analysis.question if hasattr(question_to_analysis, 'question') else None,
                    "model_name": self.model_name,
                },
                original_exception=e,
            )

    @staticmethod
    async def get_custom_instructions(
            executor: Optional[SqlExecutor],
            database_name: str,
            database_schema: str
    ) -> str:
        """
        Query a custom instructions table for a given database name.

        Args:
            executor (Optional[SqlExecutor]): SQL executor instance. If None, a default is used.
            database_name (str): Name of the database.
            database_schema (str): Name of the schema (currently unused).

        Returns:
            str: The custom instruction string, or an empty string if not found.
        """
        if executor is None:
            executor = SqlExecutor("ASCENT", "PUBLIC")

        instruction_query = (
            f"select INSTRUCTION from ASCENT_CUSTOM_INSTRUCTIONS_NON_OMOP where database = '{database_name}'"
        )

        instruction = await executor.execute_sql_query(instruction_query)

        try:
            return instruction['INSTRUCTION'][0]
        except (KeyError, IndexError, TypeError):
            return ""

    @retry_on_exception(max_retries=MAX_RETRIES)
    async def get_sql_preparation(self, prompt_sql_preparation: str) -> SQLPreparationSchema:
        """
        Send the SQL preparation prompt to the LLM and parse the response.

        Args:
            prompt_sql_preparation (str): The prompt for SQL preparation.

        Returns:
            SQLPreparationSchema: Parsed SQL preparation response.

        Raises:
            SQLPreparationLLMError: If the LLM fails to generate SQL preparation
            SQLPreparationValidationError: If the response validation fails
        """
        try:
            logger.info(f"Attempting to get SQL cohort template from model '{self.model_name}'")
            raw_response = await self.llm_service.send_message(
                prompt_sql_preparation, schema=SQLPreparationSchema, model_name=self.model_name
            )

            if not raw_response:
                raise SQLPreparationLLMError(
                    model_name=self.model_name,
                    database_name="unknown",
                    database_schema="unknown",
                    error_details="Received an empty or null response from the model during SQL preparation",
                )

            if isinstance(raw_response, str):
                resp = SQLPreparationSchema.model_validate_json(raw_response)
            else:
                resp = SQLPreparationSchema.model_validate(raw_response)

            logger.info(f"Successfully received and parsed response from '{self.model_name}'")
            return resp

        except SQLPreparationLLMError:
            raise
        except Exception as e:
            logger.error(f"Failed to get SQL preparation from model '{self.model_name}': {str(e)}")
            if "validation" in str(e).lower() or "pydantic" in str(type(e).__name__.lower()):
                raise SQLPreparationValidationError(
                    validation_details=f"Failed to validate SQL preparation response: {str(e)}",
                    original_exception=e,
                )
            raise SQLPreparationLLMError(
                model_name=self.model_name,
                database_name="unknown",
                database_schema="unknown",
                error_details=str(e),
                original_exception=e,
            )

    @retry_on_exception(max_retries=MAX_RETRIES)
    async def get_sql_creation_template(self, prompt_sql_execution: str) -> SQLCreationSimplified:
        """
        Send the SQL execution prompt to the LLM and parse the response.

        Args:
            prompt_sql_execution (str): The prompt for SQL execution.

        Returns:
            SQLCreationSimplified: Parsed SQL creation template.

        Raises:
            SQLPreparationLLMError: If the LLM fails to generate SQL creation template
            SQLPreparationValidationError: If the response validation fails
        """
        try:
            logger.info(f"Attempting to get SQL creation template from model '{self.model_name}'")
            sql_cohort_creation_templated = await self.llm_service.send_message(
                prompt_sql_execution, schema=SQLCreationSimplified, model_name=self.model_name
            )

            if not sql_cohort_creation_templated:
                raise SQLPreparationLLMError(
                    model_name=self.model_name,
                    database_name="unknown",
                    database_schema="unknown",
                    error_details="Received an empty or null response from the model during SQL creation",
                )

            if isinstance(sql_cohort_creation_templated, str):
                result = SQLCreationSimplified.model_validate_json(sql_cohort_creation_templated)
            else:
                result = SQLCreationSimplified.model_validate(sql_cohort_creation_templated)

            logger.info(f"Successfully received and parsed SQL creation template from '{self.model_name}'")
            logger.info(f"SQL creation template: {result.sql_code_snippet}")
            return result

        except SQLPreparationLLMError:
            raise
        except Exception as e:
            logger.error(f"Failed to get SQL creation template from model '{self.model_name}': {str(e)}")
            if "validation" in str(e).lower() or "pydantic" in str(type(e).__name__.lower()):
                raise SQLPreparationValidationError(
                    validation_details=f"Failed to validate SQL creation template: {str(e)}",
                    original_exception=e,
                )
            raise SQLPreparationLLMError(
                model_name=self.model_name,
                database_name="unknown",
                database_schema="unknown",
                error_details=str(e),
                original_exception=e,
            )

    @retry_on_exception(max_retries=MAX_RETRIES)
    async def get_repaired_template(self, prompt_sql_execution: str) -> SQLRepair:
        """
        Send a repair prompt to the LLM and parse the repaired SQL template.

        Args:
            prompt_sql_execution (str): The prompt for repairing SQL.

        Returns:
            SQLRepair: Parsed repaired SQL template.

        Raises:
            SQLRepairError: If the LLM fails to repair the SQL template
        """
        try:
            logger.info(f"Attempting to get repaired SQL template from model '{self.model_name}'")
            sql_repaired_templated = await self.llm_service.send_message(
                prompt_sql_execution, schema=SQLRepair, model_name=self.model_name
            )

            if not sql_repaired_templated:
                raise SQLRepairError(
                    repair_attempt=1,
                    max_attempts=self.MAX_RETRIES,
                    error_details="Received an empty or null response from the model during SQL repair",
                )

            if isinstance(sql_repaired_templated, str):
                result = SQLRepair.model_validate_json(sql_repaired_templated)
            else:
                result = SQLRepair.model_validate(sql_repaired_templated)

            logger.info(f"Successfully received and parsed repaired SQL template from '{self.model_name}'")
            return result

        except SQLRepairError:
            raise
        except Exception as e:
            logger.error(f"Failed to get repaired SQL template from model '{self.model_name}': {str(e)}")
            raise SQLRepairError(
                repair_attempt=1,
                max_attempts=self.MAX_RETRIES,
                error_details=str(e),
                original_exception=e,
            )

    @staticmethod
    async def get_medical_concepts(sql_code_snippet: str, codes_with_dots: bool = False, top_k: int = 10000, database: str | None = None) -> tuple[str, ExtractedConcepts]:
        """Extract and merge medical entity references from a SQL code snippet.

        Returns the updated SQL (with combined multi-vocab placeholders replacing
        any individual per-vocab ones) together with the deduplicated list of
        extracted concepts — one entry per clinical concept regardless of how many
        coding systems it spans.

        Args:
            sql_code_snippet: The SQL code snippet to analyze.
            codes_with_dots: Flag indicating if codes with dots should be preserved.
                Used as the fallback when ``database`` has no recorded IS_WITH_DOTS flag.
            top_k: Max codes fetched per entity.
            database: Target non-OMOP database. When set and its IS_WITH_DOTS flag is
                recorded, that flag overrides ``codes_with_dots``.

        Returns:
            A 2-tuple ``(updated_sql, extracted_concepts)``.
        """
        entity_references = await get_medical_concepts_from_sql_async(sql_code_snippet, codes_with_dots, top_k=top_k, database=database)
        merged_sql, merged_refs = merge_entity_references(sql_code_snippet, list(entity_references))
        return merged_sql, ExtractedConcepts(entity_references=merged_refs)

    async def check_and_repair(
            self,
            sql_response: SQLCreationSimplified,
            sql_preparation_json: Dict[str, Any],
            max_retries: int = MAX_RETRIES,
            cohort_metadata: CohortMetadata | None = None,
    ) -> SQLCreationSimplified:
        """
        Repairs missing placeholders in SQL and retries until successful or retries are exhausted.

        This method attempts to repair SQL with missing placeholders but will always return
        a result, even if repair fails. Repair failures are logged but not raised as exceptions.

        Args:
            sql_response (SQLCreationSimplified): SQL template response.
            sql_preparation_json (dict): SQL preparation response (used for reconciling placeholders).
            max_retries (int): Maximum retry attempts (default is MAX_RETRIES).
            cohort_metadata (CohortMetadata | None): Optional cohort metadata for SQL generation.

        Returns:
            SQLCreationSimplified: The final SQL template. If all repair attempts fail,
                                   returns the best available SQL even with unresolved placeholders.

        Note:
            This method logs repair failures but does NOT raise exceptions. It returns the
            best result available to ensure the pipeline can continue.
        """
        attempt = 0
        repair_log = []  # Stores details of each repair attempt
        all_missing_concepts = []

        while attempt < max_retries:
            try:
                # Step 1: Check for missing placeholders
                missing_concepts, available_concepts = self.handle_wrong_placeholders(sql_response)

                if not missing_concepts:
                    logger.info("Successfully generated SQL without missing placeholders.")
                    if repair_log:
                        logger.info(f"Repair log: {repair_log}")

                    return sql_response

                # Step 2: Log missing placeholders and attempt a repair
                logger.info(f"Missing placeholders detected: {missing_concepts}. Repairing SQL prompt...")
                logger.info(f"Attempt {attempt + 1}/{max_retries}: Repairing SQL ...")

                # Update `repair_log` with details of the repair attempt
                repair_log.append({
                    "attempt": attempt + 1,
                    "missing_concepts": missing_concepts,
                    "previous_sql_snippet": sql_response.sql_code_snippet,
                })
                all_missing_concepts.extend(missing_concepts)

                # Step 3: Repair placeholders based on the new output
                repair_prompt = get_prompt_sql_execution_with_wrong_placeholders(
                    sql_preparation_json,
                    missing_concepts,
                    sql_response.sql_code_snippet,
                    available_concepts,
                    cohort_metadata
                )

                repaired_template = await self.get_repaired_template(repair_prompt)
                sql_response.sql_code_snippet = repaired_template.sql_code_snippet

            except SQLRepairError as e:
                # Log SQLRepairError but continue trying
                logger.warning(f"SQL repair error during attempt {attempt + 1}/{max_retries}: {e.message}")
                repair_log.append({
                    "attempt": attempt + 1,
                    "error": e.message,
                    "error_type": "SQLRepairError"
                })
            except Exception as e:
                # Capture any other exception and log it, include in repair log for debugging
                logger.warning(f"Error during SQL repair process: {repr(e)}. Retrying...")
                repair_log.append({
                    "attempt": attempt + 1,
                    "error": repr(e),
                    "error_type": type(e).__name__
                })

            attempt += 1

        # If retries are exhausted, log the failure and raise an exception
        logger.error(f"Failed to generate valid SQL after {max_retries} retries. Repair log: {repair_log}")
        return sql_response

    def handle_wrong_placeholders(
            self, resp: SQLCreationSimplified
    ) -> Tuple[List[str], List[str]]:
        """
        Identify missing and available placeholders in the SQL template.

        Args:
            resp (SQLCreationSimplified): The SQL template response.

        Returns:
            Tuple[List[str], List[str]]: (missing_concepts, available_concepts)
        """
        logger.info(f'extracted_concepts: {resp.extracted_concepts}')
        failed_to_get = []
        successful = []

        for entity_reference in resp.extracted_concepts.entity_references:
            if not entity_reference.concepts:
                failed_to_get.append(entity_reference.placeholder)
            else:
                successful.append(entity_reference.placeholder)

        return failed_to_get, successful

    async def prepare_cohort_sql(
            self,
            database_name: str,
            database_schema: str,
            inclusion_criteria: List[str],
            exclusion_criteria: List[str],
            index_date: str,
            question_to_analysis: QuestionToAnalysis,
            medical_entities: MedicalEntities,
            tables_with_medical_coding_ontologies: List[Any],
            m_schema_string: str,
            metadata: Any,
            custom_instruction: str | None = None,
            top_k: int = 10000,
    ) -> SQLCreationSimplified:
        """Prepare SQL for patient cohort creation (non-OMOP databases).

        Identical to :meth:`prepare_sql` except that step 5 uses
        :func:`get_prompt_cohort_sql_execution` which forces the LLM to
        produce ``SELECT DISTINCT ... AS SUBJECT_ID, ... AS INDEX_DATE``
        instead of analytics / count queries.

        Args:
            database_name: Snowflake database name.
            database_schema: Schema within the database.
            inclusion_criteria: List of inclusion criteria strings.
            exclusion_criteria: List of exclusion criteria strings.
            index_date: Index date definition.
            question_to_analysis: Parsed question analysis (used by step 2).
            medical_entities: Medical entities extracted from criteria.
            tables_with_medical_coding_ontologies: Tables with coding ontology metadata.
            m_schema_string: Machine-readable schema string.
            metadata: Database metadata (sample values, etc.).
            custom_instruction: Optional custom instruction override.

        Returns:
            SQLCreationSimplified with cohort SQL in ``sql_code_snippet``.
        """
        try:
            # Step 1: Get custom instruction
            if not custom_instruction:
                try:
                    instruction = await self.get_custom_instructions(self.executor, database_name, database_schema)
                except Exception as e:
                    raise MetadataRetrievalError(
                        database_name=database_name,
                        database_schema=database_schema,
                        metadata_type="custom instructions",
                        error_details=f"Failed to retrieve custom instructions: {str(e)}",
                        original_exception=e,
                    )
            else:
                instruction = custom_instruction

            # Step 2: Generate the SQL preparation prompt (same as prepare_sql)
            prompt_sql_preparation = get_prompt_sql_preparation(
                question_to_analysis=question_to_analysis,
                medical_entities=medical_entities,
                tables_with_medical_coding_ontologies=tables_with_medical_coding_ontologies,
                m_schema_str=m_schema_string,
                custom_instruction=instruction,
            )

            # Step 3: Fetch the SQL preparation result
            sql_cohort_preparation_json = (await self.get_sql_preparation(prompt_sql_preparation)).model_dump()

            # Step 4: Extract relevant metadata
            try:
                peek_into_db_for_relevant_tables_columns = extract_metadata(sql_cohort_preparation_json, metadata)
            except Exception as e:
                raise MetadataRetrievalError(
                    database_name=database_name,
                    database_schema=database_schema,
                    metadata_type="relevant tables and columns",
                    error_details=f"Failed to extract metadata: {str(e)}",
                    original_exception=e,
                )

            # Step 5: Generate the COHORT-SPECIFIC SQL execution prompt
            prompt_sql_execution = get_prompt_cohort_sql_execution(
                sql_cohort_preparation_json=sql_cohort_preparation_json,
                peek_into_db_for_relevant_tables_columns=peek_into_db_for_relevant_tables_columns,
                inclusion_criteria=inclusion_criteria,
                exclusion_criteria=exclusion_criteria,
                index_date=index_date,
                m_schema_string=m_schema_string
            )

            # Step 6: Get SQL creation template
            sql_response = await self.get_sql_creation_template(prompt_sql_execution)

            try:
                contains_dots = bool(do_medical_entities_contain_dots(sql_preparation=sql_response))
            except Exception as e:
                logger.error(f"Failed to detect if the codes contain dots: {str(e)}")
                contains_dots = False

            # Step 7: Extract medical concepts
            try:
                merged_sql, extracted_concepts = await self.get_medical_concepts(sql_response.sql_code_snippet, codes_with_dots=contains_dots, top_k=top_k, database=database_name)
                sql_response.sql_code_snippet = merged_sql
                sql_response.extracted_concepts = extracted_concepts
            except Exception as e:
                raise MedicalConceptExtractionError(
                    sql_snippet=sql_response.sql_code_snippet,
                    error_details=f"Failed to extract medical concepts: {str(e)}",
                    original_exception=e,
                )

            # Step 8: Detect and repair missing medical concepts
            final_sql = await self.check_and_repair(
                sql_response=sql_response,
                sql_preparation_json=sql_cohort_preparation_json,
            )

            # Step 9: Reformat SQL (non-critical)
            try:
                reformatted_query = reformat_sql(final_sql.sql_code_snippet).sql
                final_sql.sql_code_snippet = reformatted_query
            except Exception as e:
                logger.warning(f"Error during SQL reformatting: {repr(e)}. Using unformatted SQL.")

            return final_sql

        except (MetadataRetrievalError, SQLPreparationLLMError, SQLPreparationValidationError,
                MedicalConceptExtractionError):
            raise
        except Exception as e:
            raise SQLPreparationException(
                message=f"Unexpected error during cohort SQL preparation: {str(e)}",
                context={
                    "database_name": database_name,
                    "database_schema": database_schema,
                    "model_name": self.model_name,
                },
                original_exception=e,
            )
