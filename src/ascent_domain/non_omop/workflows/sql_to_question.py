import logging

from ascent_domain.non_omop.decorators.retry_on_exception import retry_on_exception
from ascent_domain.non_omop.exceptions.sql_to_question import (
    QuestionComparisonError,
    QuestionReconstructionError,
)
from ascent_domain.non_omop.llm_provider import get_llm_service
from ascent_domain.non_omop.prompts.prompt_questions_comparison import get_prompt_questions_comparison
from ascent_domain.non_omop.prompts.prompt_sql_to_question import get_prompt_sql_to_question
from ascent_domain.non_omop.schemas.sql_to_question import SQLToQuestionVerdict
from ascent_platform.llm.router import LLMRouter

logger = logging.getLogger(__name__)


class SQLToQuestion:
    """
    The SQLToQuestion class facilitates the transformation of SQL query previews into human-readable
    natural language questions and the comparison of such generated questions against an initial question.
    It leverages a large language model (LLM) service for prompt-based processing.

    This class is designed to enable easier interpretation and evaluation of SQL queries by translating
    technical query text into comprehensible questions. Additionally, it performs validation checks to
    compare reconstructed questions with the originally provided question and assess their similarity.
    """
    MAX_RETRIES = 3

    def __init__(self, model_name: str, llm_service: LLMRouter | None = None) -> None:
        """
        Initializes an instance of the class with a specified language model and sets up
        an LLM routing service using the provided API keys and fallback model name from the settings.

        Attributes:
            model_name (str): The name of the language model to be used.
        """
        self.model_name = model_name
        self.llm_service = llm_service or get_llm_service()

    @retry_on_exception(max_retries=MAX_RETRIES)
    async def _get_question_comparison(self, question_rebuilt: str, initial_question: str, cohort_used: bool) -> SQLToQuestionVerdict:
        """
        Retries the execution of the function when an exception is raised, up to the
        provided maximum retry count. This asynchronous method retrieves a verdict
        comparing an initial question to a rebuilt question.

        Parameters:
            question_rebuilt: str
                The rebuilt version of the question to compare.
            initial_question: str
                The original version of the question for comparison.
            bool cohort_used:
                Indicates whether a cohort was used in the SQL query.

        Returns:
            SQLToQuestionVerdict
                The verdict object derived from comparing the initial and rebuilt
                questions.`

        Raises:
            QuestionComparisonError: If the comparison fails
        """
        try:
            prompt_question_comparison = get_prompt_questions_comparison(original_question=initial_question, question_rebuilt=question_rebuilt,
                                                                         cohort_used=cohort_used)

            verdict = await self.llm_service.send_message(prompt=prompt_question_comparison, model_name=self.model_name, schema=SQLToQuestionVerdict)

            if isinstance(verdict, str):
                healing_response = SQLToQuestionVerdict.model_validate_json(verdict)
            else:
                healing_response = SQLToQuestionVerdict.model_validate(verdict)

            return healing_response
        except Exception as e:
            raise QuestionComparisonError(
                initial_question=initial_question,
                rebuilt_question=question_rebuilt,
                error_details=str(e),
                original_exception=e,
            )

    async def get_question_from_sql(self, query_preview: str, initial_question: str, cohort_used: bool = False) -> SQLToQuestionVerdict:
        """
        This method takes the SQL query preview and the initial question, constructs prompts for both creating
        a rebuilt question and comparing it to the original question, and sends them to an external service
        for processing. The final verdict on the similarity between the two questions is returned as a string.

        Parameters:
        query_preview: str
            The SQL query preview used to create the rebuilt question.
        initial_question: str
            The original question provided to compare against the rebuilt question.
        cohort_used: bool
            Indicates whether a cohort was used in the SQL query (default is False).
        Returns:
            SQLToQuestionVerdict
            The result of the comparison between the original and the rebuilt question.
        Raises:
            QuestionReconstructionError: If question reconstruction from SQL fails
            QuestionComparisonError: If the comparison fails
        """
        try:
            prompt_question_creation = get_prompt_sql_to_question(final_sql_query=query_preview)
            question_rebuilt = await self.llm_service.send_message(model_name=self.model_name, prompt=prompt_question_creation)
        except Exception as e:
            raise QuestionReconstructionError(
                query_preview=query_preview,
                error_details=str(e),
                original_exception=e,
            )

        return await self._get_question_comparison(question_rebuilt, initial_question, cohort_used)
