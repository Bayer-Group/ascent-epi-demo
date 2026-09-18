import asyncio
import copy
import logging
import time
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from ascent_domain.omop.data.processing.sql_post_processor import MedicalSQLProcessor
from ascent_domain.omop.data.processing.utils_processing import lowercase_placeholder_values
from ascent_domain.omop.data.queries import result_query_builder
from ascent_domain.omop.external_calls import MedicalCoder
from ascent_domain.omop.models.inference.rag import RAGConfig, RAGProcessor
from ascent_domain.omop.models.inference.sql_explainer import SQLExplainer
from ascent_domain.omop.schemas.constants import CodingType
from ascent_domain.omop.schemas.data_definitions import CohortMetadata
from ascent_domain.omop.utils.mixing_temp import prepare_rwd_request
from ascent_platform.warehouse.session import get_db

CursorProvider = Callable[[str, Optional[str]], AbstractAsyncContextManager[Any]]


class QuestionAnsweringSystem:
    def __init__(
        self,
        base_dir: Path,
        log_folder: Path,
        querylib_file: Path,
        assistant_type: str = "gpt",
        model_name: str | None = None,
        snowflake_database: str = None,
        snowflake_database_schema: str = None,
        cursor_provider: Optional[CursorProvider] = None,
        settings=None,
        codes_cache_path: Optional[Path] = None,
    ):
        self.base_dir = base_dir
        self.log_folder = log_folder
        self.querylib_file = querylib_file
        self.assistant_type = assistant_type
        self.model_name = model_name
        self.snowflake_database = snowflake_database
        self.snowflake_database_schema = snowflake_database_schema
        self._cursor_provider = cursor_provider
        self.settings = settings

        self.config = RAGConfig(
            main_path=str(base_dir),
            log_folder=str(log_folder),
            assistant_type=assistant_type,
            model_name=model_name if model_name is not None else None,
        )

        self.rag_agent = RAGProcessor(self.config)
        self.rag_agent.querylib_manager.load_querylib(querylib_file=str(querylib_file))

        # Use settings passed in instead of importing directly
        self.recommender = MedicalCoder(self.settings.AZURE_CLIENT_ID, self.settings.AZURE_CLIENT_SECRET)

        # Initialize codes cache if path provided
        codes_cache = None
        if codes_cache_path is not None:
            from ascent_domain.omop.data.storage.medical_codes_cache import MedicalCodesCache

            codes_cache = MedicalCodesCache(codes_cache_path)
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Initialized medical codes cache: {codes_cache_path}")

        self.med_sql_processor = MedicalSQLProcessor(
            assistant=self.rag_agent.default_assistant, recommender=self.recommender, codes_cache=codes_cache
        )

        self.sql_explainer = SQLExplainer()
        self.logger = logging.getLogger(__name__)

    def _db_connection(self):
        """Return a Snowflake cursor context manager.

        When a ``cursor_provider`` is injected for user-scoped sessions, defer to it. Otherwise fall back to the
        package's machine-user pool (``get_db``) — this is the standalone path
        used when running the package outside the backend.
        """
        if self._cursor_provider is not None:
            return self._cursor_provider(
                self.snowflake_database,
                self.snowflake_database_schema,
            )
        return get_db(
            self.snowflake_database,
            self.snowflake_database_schema,
        )

    @classmethod
    async def initialize(cls, config: Dict[str, Any]) -> "QuestionAnsweringSystem":
        """
        Factory method to create and initialize a QuestionAnsweringSystem instance.

        Args:
            config: Configuration dictionary containing:
                - querylib_file: Path to query library (can be absolute or relative)
                - assistant: Dict with type and model
                - database: Snowflake database configuration
                - database_schema: Snowflake database schema configuration
                - settings: Settings object with Azure credentials
                - log_directory: Optional path for logs
                - codes_cache_path: Optional path to medical codes cache database

        Returns:
            Initialized QuestionAnsweringSystem instance
        """
        # Use base_dir from config if provided (for Docker), otherwise use package location
        if "base_dir" in config:
            base_dir = Path(config["base_dir"])
        else:
            base_dir = Path(__file__).resolve().parent.parent

        # Setup log folder
        if "log_directory" in config:
            log_folder = Path(config["log_directory"])
        else:
            log_folder = base_dir / "logs"  # default location
        log_folder.mkdir(parents=True, exist_ok=True)

        # Get database schema
        if "database_schema" in config:
            snowflake_database_schema = config["database_schema"]
        else:
            snowflake_database_schema = None

        # Get codes cache path if provided
        codes_cache_path = config.get("codes_cache_path")

        # Handle querylib_file as absolute path or relative to base_dir
        querylib_path = Path(config["querylib_file"])
        if querylib_path.is_absolute():
            querylib_file = querylib_path
        else:
            querylib_file = base_dir / config["querylib_file"]

        return cls(
            base_dir=base_dir,
            log_folder=log_folder,
            querylib_file=querylib_file,
            assistant_type=config.get("assistant", {}).get("type", "gpt"),
            model_name=config.get("assistant", {}).get("model"),
            snowflake_database=config.get("database"),
            snowflake_database_schema=snowflake_database_schema,
            settings=config.get("settings"),
            codes_cache_path=codes_cache_path,
        )

    # ==================== SINGLE QUESTION METHODS ====================

    async def generate_query_template(
        self,
        input_question: str,
        selected_databases: list[str] | None = None,
        drop_first: Optional[bool] = False,
        cohort_metadata: Optional[CohortMetadata] = None,
    ) -> Dict[str, Any]:
        """
        Generate the SQL query template from the input question.

        Args:
            input_question: The question to generate SQL for
            selected_databases: List of databases to consider
            drop_first: True if the first retrieved SQL from the query library should be dropped, False otherwise
                        Always use False when deploying, and use True when evaluating/benchmarking the models
            cohort_metadata: metadata of the table, applicable only if a cohort is being queried

        Returns:
            Dict containing the template query and generation metadata
        """
        selected_databases = ["SYNTHETIC_EHR_OMOP"] if selected_databases is None else selected_databases
        try:
            sql_query_result = await result_query_builder.generate_sql_query(
                user_input=input_question,
                return_filled_query=False,
                drop_first=drop_first,
                med_sql_processor=self.med_sql_processor,
                rag_agent=self.rag_agent,
                selected_databases=selected_databases,
                skip_db_upload=True,
                cohort_metadata=cohort_metadata,
            )

            # Lowercase the placeholder values in the template query
            template_query = lowercase_placeholder_values(sql_query_result["template_query"])

            return {"query_template": template_query, "generation_metadata": sql_query_result.get("metadata", {})}
        except Exception as e:
            self.logger.error(f"Error generating query template: {str(e)}")
            raise

    async def post_process_query(
        self,
        query_template: str,
        preferred_coding_system: CodingType = CodingType.STANDARD_CODING,
        custom_code_mappings: Optional[Dict[Tuple[str, str], str]] = None,
    ) -> str:
        """
        Post-process the query template to fill in codes and other specifics.

        Args:
            query_template: The template SQL query
            preferred_coding_system: The coding system to use
            custom_code_mappings: Optional dictionary mapping (entity_type, value) to code strings
                                 Example:         custom_code_mappings = {
            ('condition', 'dysphagia'): '4310996,4159140',
            ('condition', 'atopic dermatitis'): '1112807, 1112808',
            ('procedure', 'appendectomy'): '2211444,2211445'

        Returns:
            The processed SQL query ready for execution
        """
        try:
            async with self._db_connection() as db:
                query_filled = await self.med_sql_processor.post_process_sql_query(
                    query_template,
                    preferred_coding_system=preferred_coding_system,
                    rag_agent=self.rag_agent,
                    db_session=db,
                    custom_code_mappings=custom_code_mappings,
                )
                return query_filled
        except Exception as e:
            self.logger.error(f"Error post-processing query: {str(e)}")
            raise

    async def execute_query(
        self,
        query_filled: str,
        query_template: str,
        input_question: str,
        preferred_coding_system: str = "Standard",
        max_retries: int = 5,
    ) -> Tuple[pd.DataFrame, Any]:
        """
        Execute the processed SQL query and return results.

        Args:
            query_filled: The processed SQL query ready for execution
            query_template: The original template query
            input_question: The original question
            preferred_coding_system: The coding system used
            max_retries: Maximum number of query retry attempts

        Returns:
            Tuple of (DataFrame with results, RWD request object)
        """
        try:
            async with self._db_connection() as db:
                new_prompt = self.rag_agent.default_assistant.conversation

                rwd_request = prepare_rwd_request(
                    user_input=input_question,
                    query_filled_pred=query_filled,
                    query_template_pred=query_template,
                    preferred_coding_system=preferred_coding_system,
                    med_sql_processor=self.med_sql_processor,
                    rag=self.rag_agent,
                    prompt=new_prompt,
                )

                df = await rwd_request.run_query(
                    db=db,
                    max_retries=max_retries,
                    reset_conversation=False,
                )

                return df, rwd_request
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise

    async def generate_answer(
        self,
        rwd_request: Any,
        df: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Generate an answer based on the query results.

        Args:
            rwd_request: The RWD request object containing query context
            df: Optional DataFrame with query results

        Returns:
            Generated answer string
        """
        try:
            await rwd_request.get_answer(self.rag_agent.default_assistant_answers)
            return rwd_request.answer
        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            raise

    async def answer_question(
        self,
        input_question: str,
        drop_first: Optional[bool] = True,
        preferred_coding_system: str = "Standard",
        generate_answer: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a question end-to-end and return results.

        Args:
            input_question: The question to be answered
            preferred_coding_system: The coding system to use
            drop_first: True if the first retrieved SQL from the query library should be dropped, False otherwise
                Always use False when deploying, and use True when evaluating/benchmarking the models
            generate_answer: Whether to generate a natural language answer

        Returns:
            Dict containing results and metadata
        """
        start_time = time.time()

        try:
            # Generate query template
            template_result = await self.generate_query_template(input_question=input_question, drop_first=drop_first)
            query_template = template_result["query_template"]

            # Post-process query
            query_filled = await self.post_process_query(
                query_template,
                preferred_coding_system,
            )

            # Execute query
            df, rwd_request = await self.execute_query(query_filled, query_template, input_question, preferred_coding_system)

            # Generate answer if requested
            answer = None
            if generate_answer:
                answer = await self.generate_answer(rwd_request, df)

            execution_time = time.time() - start_time

            results = {
                "question": input_question,
                "answer": answer,
                "sql_template": query_template,
                "sql_filled": query_filled,
                "data": df.to_dict() if df is not None else None,
                "execution_time": execution_time,
                "generation_metadata": template_result.get("generation_metadata", {}),
            }

            self.log_results(results)
            return results

        except Exception as e:
            error_time = time.time() - start_time
            error_results = {"question": input_question, "error": str(e), "execution_time": error_time}
            self.logger.error(f"Error processing question: {str(e)}")
            return error_results

    async def get_query_explanation(self, sql_query: str, input_question: str = None) -> str:
        return await self.sql_explainer.get_explanation(sql_query, input_question)

    # ==================== MULTIPLE QUESTIONS METHODS ====================

    @classmethod
    async def create_qa_system_for_question(cls, config: Dict[str, Any]) -> "QuestionAnsweringSystem":
        """Create a fresh QA system instance for a single question to ensure clean conversation context."""
        return await cls.initialize(config)

        # In your qa.py file, inside the QuestionAnsweringSystem class

    def _create_isolated_worker(self) -> Tuple[RAGProcessor, MedicalSQLProcessor]:
        """
        Creates a lightweight, isolated set of "worker" components for a single parallel task.
        This performs a fast, in-memory copy and avoids all blocking I/O.
        """
        # This surgical copy pattern creates an isolated context without blocking.
        local_rag_agent = copy.copy(self.rag_agent)
        local_rag_agent.default_state = copy.copy(self.rag_agent.default_state)

        # Isolate both assistants to keep conversations separate
        local_rag_agent.default_state.assistant = copy.copy(self.rag_agent.default_state.assistant)
        local_rag_agent.default_state.assistant.conversation = []

        local_rag_agent.default_state.assistant_answers = copy.copy(self.rag_agent.default_state.assistant_answers)
        local_rag_agent.default_state.assistant_answers.conversation = []

        # Create a new processor that uses the isolated agent
        # IMPORTANT: Pass codes_cache to enable caching in parallel workers
        local_med_sql_processor = MedicalSQLProcessor(
            assistant=local_rag_agent.default_assistant,
            recommender=self.recommender,
            codes_cache=self.med_sql_processor.codes_cache,  # Pass cache from parent
        )

        return local_rag_agent, local_med_sql_processor

    async def _process_single_template_generation(
        self, question: str, drop_first: bool, use_rag: bool, selected_databases: list[str], cohort_metadata: Optional[Any]
    ) -> Dict[str, Any]:
        """
        Creates an isolated context using the central worker factory and awaits
        the now non-blocking RAG process.
        """
        # 1. Get a fresh, isolated worker. This is now a single, clean call.
        local_rag_agent, local_med_sql_processor = self._create_isolated_worker()

        # 2. Use the worker to do the job. The rest of the logic is unchanged.
        sql_query_result = await result_query_builder.generate_sql_query(
            user_input=question,
            return_filled_query=False,
            drop_first=drop_first,
            use_rag=use_rag,
            med_sql_processor=local_med_sql_processor,
            rag_agent=local_rag_agent,
            selected_databases=selected_databases,
            skip_db_upload=True,
            cohort_metadata=cohort_metadata,
        )

        template_query = lowercase_placeholder_values(sql_query_result["template_query"])

        return {"query_template": template_query, "generation_metadata": sql_query_result.get("metadata", {})}

    async def generate_query_templates(
        self,
        input_questions: Union[List[str], str],
        drop_first: Optional[bool] = False,
        use_rag: Optional[bool] = True,
        selected_databases: list[str] | None = None,
        cohort_metadata: Optional[CohortMetadata] = None,
        max_concurrency: int = 10,  # We can likely handle more concurrency now
    ) -> List[Dict[str, Any]]:
        """
        Generate SQL query templates from input questions with true, efficient parallelism.
        This method now uses the single, pre-initialized instance's resources.
        """
        if isinstance(input_questions, str):
            input_questions = [input_questions]
        selected_databases = ["SYNTHETIC_EHR_OMOP"] if selected_databases is None else selected_databases

        # A semaphore is still excellent practice for controlling API calls.
        semaphore = asyncio.Semaphore(max_concurrency)
        log_lock = asyncio.Lock()
        results = [None] * len(input_questions)

        async def process_with_semaphore(idx: int, question: str) -> None:
            """Wrapper to apply the semaphore to our processing method."""
            task_id = f"Task-{idx + 1}"
            # This async with block ensures only `max_concurrency` tasks run at once.
            async with semaphore:
                try:
                    async with log_lock:
                        self.logger.info(f"[{task_id}] STARTED Processing question {idx + 1}/{len(input_questions)}: '{question[:150]}...'")

                    start_time = time.time()

                    template_result = await self._process_single_template_generation(
                        question=question,
                        drop_first=drop_first,
                        use_rag=use_rag,
                        selected_databases=selected_databases,
                        cohort_metadata=cohort_metadata,
                    )

                    elapsed = time.time() - start_time
                    async with log_lock:
                        self.logger.info(f"[{task_id}] COMPLETED question {idx + 1} in {elapsed:.2f} seconds")

                    results[idx] = {"question": question, **template_result}

                except Exception as e:
                    async with log_lock:
                        self.logger.error(f"[{task_id}] Error for question {idx + 1}: {str(e)}")
                    results[idx] = {"question": question, "query_template": None, "error": str(e)}

        # This part remains the same - it correctly creates and runs the tasks.
        tasks = [asyncio.create_task(process_with_semaphore(i, q)) for i, q in enumerate(input_questions)]

        await asyncio.gather(*tasks)
        return results

    async def post_process_queries(
        self,
        template_results: List[Dict[str, Any]],
        preferred_coding_system: CodingType = CodingType.STANDARD_CODING,
        custom_code_mappings: Optional[Dict[Tuple[str, str], str]] = None,
        max_concurrency: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Post-process query templates in true parallel using isolated workers.
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        log_lock = asyncio.Lock()

        async def process_with_semaphore(template_result: Dict[str, Any]) -> Dict[str, Any]:
            if template_result.get("error") or not template_result.get("query_template"):
                return template_result

            async with semaphore:
                try:
                    # 1. Get a fresh, isolated worker. This is fast and non-blocking.
                    local_rag_agent, local_med_sql_processor = self._create_isolated_worker()

                    # 2. Use the worker to do the job.
                    async with self._db_connection() as db:
                        query_filled = await local_med_sql_processor.post_process_sql_query(
                            template_result["query_template"],
                            preferred_coding_system=preferred_coding_system,
                            rag_agent=local_rag_agent,
                            db_session=db,
                            custom_code_mappings=custom_code_mappings,
                        )
                    return {**template_result, "query_filled": query_filled}
                except Exception as e:
                    question = template_result.get("question", "N/A")
                    async with log_lock:
                        self.logger.error(f"Error post-processing query for question '{question}': {str(e)}")
                    return {**template_result, "query_filled": None, "error": str(e)}

        tasks = [process_with_semaphore(result) for result in template_results]
        processed_results = await asyncio.gather(*tasks)
        return processed_results

    async def execute_queries(
        self,
        processed_results: List[Dict[str, Any]],
        config: Dict[str, Any],
        preferred_coding_system: str = "Standard",
        max_retries: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Execute processed SQL queries and return results in TRUE parallel.
        Uses lightweight isolated workers instead of full QA system initialization.

        Args:
            processed_results: List of processed query results
            config: Configuration (not used - kept for API compatibility)
            preferred_coding_system: The coding system used
            max_retries: Maximum number of query retry attempts

        Returns:
            List of dicts with execution results
        """

        async def process_single_execution(processed_result: Dict[str, Any]) -> Dict[str, Any]:
            # Skip if no valid query
            if not processed_result.get("query_filled"):
                return {**processed_result, "dataframe": None, "rwd_request": None, "error": processed_result.get("error", "No query to execute")}

            try:
                # Use lightweight isolated worker (fast in-memory copy)
                local_rag_agent, local_med_sql_processor = self._create_isolated_worker()

                # Create database session for this worker
                async with self._db_connection() as db:
                    new_prompt = local_rag_agent.default_assistant.conversation

                    rwd_request = prepare_rwd_request(
                        user_input=processed_result["question"],
                        query_filled_pred=processed_result["query_filled"],
                        query_template_pred=processed_result["query_template"],
                        preferred_coding_system=preferred_coding_system,
                        med_sql_processor=local_med_sql_processor,
                        rag=local_rag_agent,
                        prompt=new_prompt,
                    )

                    df = await rwd_request.run_query(
                        db=db,
                        max_retries=max_retries,
                        reset_conversation=False,
                    )

                    return {**processed_result, "dataframe": df, "rwd_request": rwd_request}
            except Exception as e:
                self.logger.error(f"Error executing query for question '{processed_result['question']}': {str(e)}")
                return {**processed_result, "dataframe": None, "rwd_request": None, "error": str(e)}

        tasks = [process_single_execution(processed_result) for processed_result in processed_results]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred during gather
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Exception in query execution for question {i}: {str(result)}")
                final_results.append({**processed_results[i], "dataframe": None, "rwd_request": None, "error": str(result)})
            else:
                final_results.append(result)

        return final_results

    async def _process_single_answer_generation(self, rwd_request: Any) -> str:
        """
        Generates a single answer using an isolated context from the central worker factory.
        """
        # 1. Get a fresh, isolated worker.
        # We only need the agent here, so we use `_` for the processor.
        local_rag_agent, _ = self._create_isolated_worker()

        # 2. Use the isolated agent's 'assistant_answers' to generate the answer.
        await rwd_request.get_answer(local_rag_agent.default_assistant_answers)

        return rwd_request.answer

    async def generate_answers(self, execution_results: List[Dict[str, Any]], max_concurrency: int = 10) -> List[Dict[str, Any]]:
        """
        Generate answers based on query results in true parallel.
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        log_lock = asyncio.Lock()

        async def process_with_semaphore(execution_result: Dict[str, Any]) -> Dict[str, Any]:
            # If the previous steps failed, just pass the result through.
            if execution_result.get("error") or execution_result.get("rwd_request") is None:
                return execution_result

            async with semaphore:
                try:
                    # This helper method call is correct.
                    answer = await self._process_single_answer_generation(execution_result["rwd_request"])
                    return {**execution_result, "answer": answer}

                except Exception as e:
                    question = execution_result.get("question", "N/A")
                    async with log_lock:
                        self.logger.error(f"Error generating answer for question '{question}': {str(e)}")
                    return {**execution_result, "answer": None, "error": str(e)}

        tasks = [process_with_semaphore(result) for result in execution_results]
        final_results = await asyncio.gather(*tasks)

        return final_results

    async def generate_answers_simple(self, execution_results: List[Dict[str, Any]], max_concurrency: int = 10) -> List[str]:
        """
        Generate answers and return only the answer strings.
        """
        full_results = await self.generate_answers(execution_results, max_concurrency)

        answers = []
        for result in full_results:
            if result.get("answer"):
                answers.append(result["answer"])
            else:
                answers.append("")  # Empty string for missing answers

        return answers

    def log_results(self, result: dict) -> None:
        """Log the results of the question answering process"""
        self.logger.info(f"Question: {result.get('question', 'N/A')}")
        self.logger.debug(f"SQL template: {result.get('query_template', 'N/A')}")
        self.logger.debug(f"SQL filled: {result.get('query_filled', 'N/A')}")
        self.logger.info(f"Answer: {result.get('answer', 'N/A')}")
        self.logger.info(f"Total time: {result.get('execution_time', 0):.1f} s")
        if result.get("error"):
            self.logger.error(f"Error: {result['error']}")
        self.logger.info("-----------------")
