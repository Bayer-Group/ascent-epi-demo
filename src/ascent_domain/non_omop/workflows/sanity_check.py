from ascent_domain.non_omop.decorators.retry_on_exception import retry_on_exception
from ascent_domain.non_omop.exceptions.sanity_check import (
    SanityCheckLLMError,
    SanityCheckValidationError,
)
from ascent_domain.non_omop.llm_provider import get_llm_service
from ascent_domain.non_omop.prompts.prompts_pipeline import get_prompt_sanity_check
from ascent_domain.non_omop.schemas.question_sanity_check import LLMOutputSanityCheck
from ascent_platform.llm.router import LLMRouter


class SanityCheck:
    """
    Service for running LLM-based sanity checks on user questions.
    """
    MAX_RETRIES = 3

    def __init__(self, model_name: str, llm_service: LLMRouter = None):
        self.model_name = model_name
        self.llm_service = llm_service or get_llm_service()

    @retry_on_exception(max_retries=MAX_RETRIES)
    async def run_sanity_check(self, question: str) -> LLMOutputSanityCheck:
        """
        Run the LLM-based sanity check for a given question.
        Args:
            question (str): The user question to check.
        Returns:
            LLMOutputSanityCheck: The validated sanity check result.
        Raises:
            SanityCheckLLMError: If the LLM fails to process the request
            SanityCheckValidationError: If the validation fails
        """
        try:
            prompt = get_prompt_sanity_check(question)
            response = await self.llm_service.send_message(
                prompt, schema=LLMOutputSanityCheck, model_name=self.model_name
            )
            if isinstance(response, str):
                return LLMOutputSanityCheck.model_validate_json(response)
            return LLMOutputSanityCheck.model_validate(response)
        except Exception as e:
            # Check if it's a validation error
            if "validation" in str(e).lower() or "pydantic" in str(type(e).__name__.lower()):
                raise SanityCheckValidationError(
                    question=question,
                    validation_details=str(e),
                    original_exception=e,
                )
            # Otherwise, it's an LLM error
            raise SanityCheckLLMError(
                question=question,
                model_name=self.model_name,
                error_details=str(e),
                original_exception=e,
            )
