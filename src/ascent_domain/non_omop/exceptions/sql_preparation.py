"""
Exceptions for the SQL Preparation workflow step.
"""

from typing import Any, Dict, List, Optional

from .base import WorkflowException


class SQLPreparationException(WorkflowException):
    """Base exception for SQL preparation workflow errors."""

    def __init__(
            self,
            message: str,
            context: Optional[Dict[str, Any]] = None,
            original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            workflow_step="SQL Preparation",
            context=context,
            original_exception=original_exception,
        )


class SQLPreparationLLMError(SQLPreparationException):
    """
    Raised when the LLM fails during SQL preparation.
    
    Example: API timeout, invalid SQL generation, or model errors.
    """

    def __init__(
            self,
            model_name: str,
            database_name: str,
            database_schema: str,
            error_details: Optional[str] = None,
            original_exception: Optional[Exception] = None,
    ):
        message = f"LLM model failed during SQL preparation."

        super().__init__(
            message=message,
            context={
                "model_name": model_name,
                "database_name": database_name,
                "database_schema": database_schema,
                "error_details": error_details,
            },
            original_exception=original_exception,
        )


class SQLPreparationValidationError(SQLPreparationException):
    """
    Raised when SQL preparation validation fails.
    
    Example: Generated SQL doesn't match expected schema or has invalid placeholders.
    """

    def __init__(
            self,
            validation_details: str,
            sql_snippet: Optional[str] = None,
            original_exception: Optional[Exception] = None,
    ):
        message = f"SQL preparation validation failed."

        context = {"validation_details": validation_details}
        if sql_snippet:
            # Truncate SQL for logging if too long
            context["sql_snippet"] = sql_snippet[:500] + "..." if len(sql_snippet) > 500 else sql_snippet

        super().__init__(
            message=message,
            context=context,
            original_exception=original_exception,
        )


class MetadataRetrievalError(SQLPreparationException):
    """
    Raised when metadata retrieval fails.
    
    Example: Unable to fetch schema metadata, ontologies, or table information.
    """

    def __init__(
            self,
            database_name: str,
            database_schema: str,
            metadata_type: str,
            error_details: Optional[str] = None,
            original_exception: Optional[Exception] = None,
    ):
        message = f"Failed to retrieve metadata for {database_name}.{database_schema}."

        super().__init__(
            message=message,
            context={
                "database_name": database_name,
                "database_schema": database_schema,
                "metadata_type": metadata_type,
                "error_details": error_details,
            },
            original_exception=original_exception,
        )


class MedicalConceptExtractionError(SQLPreparationException):
    """
    Raised when medical concept extraction fails.
    
    Example: Unable to extract or validate medical concepts from SQL.
    """

    def __init__(
            self,
            sql_snippet: str,
            error_details: Optional[str] = None,
            original_exception: Optional[Exception] = None,
    ):
        message = "Failed to extract medical concepts from SQL."

        super().__init__(
            message=message,
            context={
                "sql_snippet": sql_snippet[:500] + "..." if len(sql_snippet) > 500 else sql_snippet,
                "error_details": error_details,
            },
            original_exception=original_exception,
        )


class SQLRepairError(SQLPreparationException):
    """
    Raised when SQL repair/healing fails.
    
    Example: Unable to fix missing placeholders or invalid SQL syntax.
    """

    def __init__(
            self,
            repair_attempt: int,
            max_attempts: int,
            error_details: Optional[str] = None,
            missing_placeholders: Optional[List[str]] = None,
            original_exception: Optional[Exception] = None,
    ):
        message = f"SQL repair failed after {repair_attempt}/{max_attempts} attempts."

        context = {
            "repair_attempt": repair_attempt,
            "max_attempts": max_attempts,
            "error_details": error_details,
        }
        if missing_placeholders:
            context["missing_placeholders"] = str(missing_placeholders)

        super().__init__(
            message=message,
            context=context,
            original_exception=original_exception,
        )
