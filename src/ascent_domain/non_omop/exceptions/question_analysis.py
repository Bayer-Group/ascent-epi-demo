"""
Exceptions for the Question Analysis workflow step.
"""

from typing import Any, Dict, Optional

from .base import WorkflowException


class QuestionAnalysisException(WorkflowException):
    """Base exception for question analysis workflow errors."""

    def __init__(
            self,
            message: str,
            context: Optional[Dict[str, Any]] = None,
            original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            workflow_step="Question Analysis",
            context=context,
            original_exception=original_exception,
        )


class QuestionAnalysisValidationError(QuestionAnalysisException):
    """
    Raised when question analysis validation fails.
    
    Example: Unable to parse the analytical objective or medical entities.
    """

    def __init__(
            self,
            question: str,
            analytical_objective: Optional[str] = None,
            validation_details: Optional[str] = None,
            original_exception: Optional[Exception] = None,
    ):
        message = "Question analysis validation failed."

        super().__init__(
            message=message,
            context={
                "question": question,
                "analytical_objective": analytical_objective,
                "validation_details": validation_details,
            },
            original_exception=original_exception,
        )


class QuestionAnalysisLLMError(QuestionAnalysisException):
    """
    Raised when the LLM fails during question analysis.
    
    Example: API timeout, invalid response format, or model errors.
    """

    def __init__(
            self,
            question: str,
            model_name: str,
            error_details: Optional[str] = None,
            original_exception: Optional[Exception] = None,
    ):
        message = f"LLM model failed during question analysis."

        super().__init__(
            message=message,
            context={
                "question": question,
                "model_name": model_name,
                "error_details": error_details,
            },
            original_exception=original_exception,
        )
