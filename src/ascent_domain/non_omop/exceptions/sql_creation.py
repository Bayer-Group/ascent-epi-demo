"""
Exceptions for the SQL Creation workflow step.
"""

from typing import Any, Dict, List, Optional

from .base import WorkflowException


class SQLCreationException(WorkflowException):
    """Base exception for SQL creation workflow errors."""

    def __init__(
            self,
            message: str,
            context: Optional[Dict[str, Any]] = None,
            original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            workflow_step="SQL Creation",
            context=context,
            original_exception=original_exception,
        )


class PlaceholderMappingError(SQLCreationException):
    """
    Raised when placeholder mapping fails.
    
    Example: Unable to map entity reference codes to placeholders.
    """

    def __init__(
            self,
            placeholder: str,
            error_details: Optional[str] = None,
            original_exception: Optional[Exception] = None,
    ):
        message = f"Failed to map placeholder '{placeholder}'."

        super().__init__(
            message=message,
            context={
                "placeholder": placeholder,
                "error_details": error_details,
            },
            original_exception=original_exception,
        )


class EntityReferenceError(SQLCreationException):
    """
    Raised when entity reference processing fails.
    
    Example: Invalid entity reference format or missing required concepts.
    """

    def __init__(
            self,
            error_details: str,
            entity_references_count: Optional[int] = None,
            missing_concepts: Optional[List[str]] = None,
            original_exception: Optional[Exception] = None,
    ):
        message = f"Entity reference processing failed."

        context = {"error_details": error_details}
        if entity_references_count is not None:
            context["entity_references_count"] = str(entity_references_count)
        if missing_concepts:
            context["missing_concepts"] = str(missing_concepts)

        super().__init__(
            message=message,
            context=context,
            original_exception=original_exception,
        )
