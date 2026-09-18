import logging

from ascent_domain.non_omop.llm_provider import get_llm_service
from ascent_domain.non_omop.prompts.prompts_pipeline import get_prompt_handle_invalid_sql
from ascent_domain.non_omop.schemas.sql_results import LLMOutputSQLHealing
from ascent_platform.llm.router import LLMRouter
from ascent_platform.warehouse.sql_executor import SqlExecutor

logger = logging.getLogger(__name__)


class SQLResults:
    """
    A service class for executing SQL queries and handling errors with self-healing capabilities.

    This class attempts to execute SQL queries and, if errors occur, tries to automatically
    fix them using LLM-based healing before retrying the execution.
    """

    def __init__(self, model_name: str, executor: SqlExecutor | None = None, llm_service: LLMRouter | None = None) -> None:
        """
        Initialize the SQLResults service.

        Args:
            model_name (str): The name of the LLM model to use for SQL generation and healing.
            executor (SqlExecutor | None, optional): SQL executor instance. If None, a default executor is created.
        """
        self.executor = executor
        self.model_name = model_name
        self.llm_service = llm_service or get_llm_service()

    async def handle_invalid_sql(self, query: str, error: str, m_schema: str) -> str | None:
        """
        Attempt to heal an invalid SQL query using LLM.

        This method sends the invalid query, error message, and schema metadata to an LLM
        to generate a corrected version of the query.

        Args:
            query (str): The invalid SQL query to heal
            error (str): The error message from the database
            m_schema (str): The schema metadata to help with healing

        Returns:
            str | None: The healed SQL query if healing was successful, or None if healing failed
        """
        prompt = get_prompt_handle_invalid_sql(query=query, error=error, m_schema=m_schema)

        prompt_response = await self.llm_service.send_message(prompt, model_name=self.model_name, schema=LLMOutputSQLHealing)

        if isinstance(prompt_response, str):
            healing_response = LLMOutputSQLHealing.model_validate_json(prompt_response)
        else:
            healing_response = LLMOutputSQLHealing.model_validate(prompt_response)

        return healing_response.healed_query
