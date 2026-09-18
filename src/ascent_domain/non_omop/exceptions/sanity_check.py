"""
Exceptions for the Sanity Check workflow step.
"""

from typing import Any, Dict, Optional

from .base import WorkflowException


class SanityCheckException(WorkflowException):
    """Base exception for sanity check workflow errors."""

    def __init__(
            self,
            message: str,
            context: Optional[Dict[str, Any]] = None,
            original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            workflow_step="Sanity Check",
            context=context,
            original_exception=original_exception,
        )


class SanityCheckValidationError(SanityCheckException):
    """
    Raised when the sanity check validation fails.
    
    Example: The question doesn't meet basic requirements for analysis.
    """

    def __init__(
            self,
            question: str,
            validation_details: Optional[str] = None,
            original_exception: Optional[Exception] = None,
    ):
        message = "The question failed sanity check validation."

        super().__init__(
            message=message,
            context={"question": question, "validation_details": validation_details},
            original_exception=original_exception,
        )


class SanityCheckLLMError(SanityCheckException):
    """
    Raised when the LLM fails during sanity check.
    
    Example: API timeout, invalid response format, or model errors.
    """

    def __init__(
            self,
            question: str,
            model_name: str,
            error_details: Optional[str] = None,
            original_exception: Optional[Exception] = None,
    ):
        message = f"LLM model failed during sanity check."

        super().__init__(
            message=message,
            context={
                "question": question,
                "model_name": model_name,
                "error_details": error_details,
            },
            original_exception=original_exception,
        )
