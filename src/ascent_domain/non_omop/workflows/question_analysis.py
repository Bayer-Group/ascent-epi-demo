from ascent_domain.non_omop.decorators.retry_on_exception import retry_on_exception
from ascent_domain.non_omop.exceptions.question_analysis import (
    QuestionAnalysisLLMError,
    QuestionAnalysisValidationError,
)
from ascent_domain.non_omop.llm_provider import get_llm_service
from ascent_domain.non_omop.prompts.prompts_pipeline import get_prompt_cohort_builder
from ascent_domain.non_omop.schemas.question_sanity_check import MedicalEntities
from ascent_domain.non_omop.schemas.question_to_analysis import QuestionToAnalysis
from ascent_platform.llm.router import LLMRouter


class QuestionAnalysis:
    """
    Service for running LLM-based question analysis for user questions.
    """
    MAX_RETRIES = 3

    def __init__(self, model_name: str, llm_service: LLMRouter = None):
        self.model_name = model_name
        self.llm_service = llm_service or get_llm_service()

    @retry_on_exception(max_retries=MAX_RETRIES)
    async def run_question_analysis(self, analytical_objective: str, question: str, medical_entities: MedicalEntities) -> QuestionToAnalysis:
        """
        Run the LLM-based question analysis for a given question and context.
        Args:
            analytical_objective: The analytical objective from the sanity check.
            question (str): The user question.
            medical_entities: Medical entities from the sanity check.
        Returns:
            QuestionToAnalysis: The validated question analysis result.
        Raises:
            QuestionAnalysisLLMError: If the LLM fails to process the request
            QuestionAnalysisValidationError: If the validation fails
        """
        try:
            prompt = get_prompt_cohort_builder(analytical_objective, question, medical_entities)
            response = await self.llm_service.send_message(
                prompt, schema=QuestionToAnalysis, model_name=self.model_name
            )
            if isinstance(response, str):
                return QuestionToAnalysis.model_validate_json(response)
            return QuestionToAnalysis.model_validate(response)
        except Exception as e:
            # Check if it's a validation error
            if "validation" in str(e).lower() or "pydantic" in str(type(e).__name__.lower()):
                raise QuestionAnalysisValidationError(
                    question=question,
                    analytical_objective=analytical_objective,
                    validation_details=str(e),
                    original_exception=e,
                )
            # Otherwise, it's an LLM error
            raise QuestionAnalysisLLMError(
                question=question,
                model_name=self.model_name,
                error_details=str(e),
                original_exception=e,
            )
