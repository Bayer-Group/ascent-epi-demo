"""
Base exception classes for the Ascent Non-OMOP pipeline.

These provide structured error handling with human-readable messages
while maintaining full traceability for debugging.
"""

import logging
import traceback
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AscentBaseException(Exception):
    """
    Base exception for all Ascent Non-OMOP pipeline errors.

    This exception maintains full traceability while providing human-readable
    error messages. All custom exceptions should inherit from this class.

    Attributes:
        message: Human-readable error message
        workflow_step: The workflow step where the error occurred
        context: Additional context information
        original_exception: The original exception if this is wrapping another exception
        traceback_str: Full traceback string for debugging
    """

    def __init__(
            self,
            message: str,
            workflow_step: Optional[str] = None,
            context: Optional[Dict[str, Any]] = None,
            original_exception: Optional[Exception] = None,
    ):
        """
        Initialize the base exception.

        Args:
            message: Human-readable error message
            workflow_step: The workflow step where the error occurred
            context: Additional context information (e.g., question, model_name, etc.)
            original_exception: The original exception if wrapping another exception
        """
        self.message = message
        self.workflow_step = workflow_step
        self.context = context or {}
        self.original_exception = original_exception

        # Capture the full traceback for debugging
        # This is stored separately but __traceback__ will still be available
        # for backend's build_enhanced_traceback_from_exc function
        if original_exception:
            self.traceback_str = ''.join(traceback.format_exception(
                type(original_exception),
                original_exception,
                original_exception.__traceback__
            ))
        else:
            self.traceback_str = ''.join(traceback.format_stack())

        # Build the full message and call parent constructor
        # The __traceback__ attribute will be set when this exception is raised
        super().__init__(message)

        # Chain the original exception for proper exception chaining
        # This allows backend to walk through the full exception chain
        if original_exception:
            self.__cause__ = original_exception

        # Log the full error with traceback for backend debugging
        self._log_exception()

    def _log_exception(self):
        """Log the exception with full traceback for backend debugging."""
        logger.error(
            f"Exception in {self.workflow_step or 'Unknown Step'}: {self.message}",
            exc_info=self.original_exception,
            extra={
                "workflow_step": self.workflow_step,
                "context": self.context,
                "exception_type": type(self).__name__,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the exception to a dictionary for API responses.

        Returns:
            Dictionary with human-readable error information
        """
        return {
            "error_type": type(self).__name__,
            "message": self.message,
            "workflow_step": self.workflow_step,
            "context": self.context,
        }


class WorkflowException(AscentBaseException):
    """
    Base exception for workflow-related errors.
    
    This should be subclassed for specific workflow steps.
    """
    pass
