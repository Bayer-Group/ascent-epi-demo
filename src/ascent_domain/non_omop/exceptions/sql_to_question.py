"""
Exceptions for the SQL to Question workflow step.
"""

from typing import Any, Dict, Optional

from .base import WorkflowException


class SQLToQuestionException(WorkflowException):
    """Base exception for SQL to question workflow errors."""

    def __init__(
            self,
            message: str,
            context: Optional[Dict[str, Any]] = None,
            original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            workflow_step="SQL to Question",
            context=context,
            original_exception=original_exception,
        )


class QuestionComparisonError(SQLToQuestionException):
    """
    Raised when question comparison fails.
    
    Example: Unable to compare the rebuilt question with the original.
    """

    def __init__(
            self,
            initial_question: str,
            rebuilt_question: str,
            error_details: Optional[str] = None,
            original_exception: Optional[Exception] = None,
    ):
        message = "Failed to compare the rebuilt question with the original question."

        super().__init__(
            message=message,
            context={
                "initial_question": initial_question,
                "rebuilt_question": rebuilt_question,
                "error_details": error_details,
            },
            original_exception=original_exception,
        )


class QuestionReconstructionError(SQLToQuestionException):
    """
    Raised when question reconstruction from SQL fails.
    
    Example: LLM unable to generate a coherent question from the SQL.
    """

    def __init__(
            self,
            query_preview: str,
            error_details: Optional[str] = None,
            original_exception: Optional[Exception] = None,
    ):
        message = "Failed to reconstruct question from SQL."

        super().__init__(
            message=message,
            context={
                "query_preview": query_preview[:500] + "..." if len(query_preview) > 500 else query_preview,
                "error_details": error_details,
            },
            original_exception=original_exception,
        )
