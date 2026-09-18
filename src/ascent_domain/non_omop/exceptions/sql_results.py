"""
Exceptions for the SQL Results workflow step.
"""

from typing import Any, Dict, Optional

from .base import WorkflowException


class SQLResultsException(WorkflowException):
    """Base exception for SQL results workflow errors."""

    def __init__(
            self,
            message: str,
            context: Optional[Dict[str, Any]] = None,
            original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            workflow_step="SQL Results",
            context=context,
            original_exception=original_exception,
        )


class SQLExecutionError(SQLResultsException):
    """
    Raised when SQL execution fails.
    
    Example: Syntax errors, invalid table references, or data type mismatches.
    """

    def __init__(
            self,
            query: str,
            database_name: str,
            database_schema: str,
            error_code: Optional[str] = None,
            error_details: Optional[str] = None,
            original_exception: Optional[Exception] = None,
    ):
        message = f"SQL execution failed on {database_name}.{database_schema}."

        super().__init__(
            message=message,
            context={
                "query": query[:500] + "..." if len(query) > 500 else query,
                "database_name": database_name,
                "database_schema": database_schema,
                "error_code": error_code,
                "error_details": error_details,
            },
            original_exception=original_exception,
        )


class SQLHealingError(SQLResultsException):
    """
    Raised when SQL healing/repair fails after execution errors.
    
    Example: LLM unable to fix the SQL after multiple attempts.
    """

    def __init__(
            self,
            query: str,
            attempt: int,
            max_retries: int,
            healing_details: Optional[str] = None,
            original_exception: Optional[Exception] = None,
    ):
        message = f"SQL healing failed after {attempt}/{max_retries} attempts."

        super().__init__(
            message=message,
            context={
                "query": query[:500] + "..." if len(query) > 500 else query,
                "attempt": attempt,
                "max_retries": max_retries,
                "healing_details": healing_details,
            },
            original_exception=original_exception,
        )


class DatabaseConnectionError(SQLResultsException):
    """
    Raised when database connection fails.
    
    Example: Network issues, authentication failures, or database unavailable.
    """

    def __init__(
            self,
            database_name: str,
            database_schema: str,
            error_details: Optional[str] = None,
            original_exception: Optional[Exception] = None,
    ):
        message = f"Failed to connect to database {database_name}.{database_schema}."

        super().__init__(
            message=message,
            context={
                "database_name": database_name,
                "database_schema": database_schema,
                "error_details": error_details,
            },
            original_exception=original_exception,
        )
