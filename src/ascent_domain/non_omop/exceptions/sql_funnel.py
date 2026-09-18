"""
Exceptions for the SQL Funnel workflow step.
"""

from typing import Any, Dict, Optional

from .base import WorkflowException


class SQLFunnelException(WorkflowException):
    """Base exception for SQL funnel workflow errors."""
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            workflow_step="SQL Funnel",
            context=context,
            original_exception=original_exception,
        )


class FunnelGenerationError(SQLFunnelException):
    """
    Raised when funnel generation fails.
    
    Example: Unable to generate patient funnel from query.
    """
    
    def __init__(
        self,
        query_preview: str,
        database: str,
        schema: str,
        error_details: Optional[str] = None,
        original_exception: Optional[Exception] = None,
    ):
        message = f"Funnel generation failed for {database}.{schema}."
        
        super().__init__(
            message=message,
            context={
                "query_preview": query_preview[:500] + "..." if len(query_preview) > 500 else query_preview,
                "database": database,
                "schema": schema,
                "error_details": error_details,
            },
            original_exception=original_exception,
        )


class FunnelValidationError(SQLFunnelException):
    """
    Raised when funnel validation fails.
    
    Example: Invalid funnel structure or missing required fields.
    """
    
    def __init__(
        self,
        validation_details: str,
        funnel_data: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        message = f"Funnel validation failed."
        
        context = {"validation_details": validation_details}
        if funnel_data:
            context["funnel_data"] = str(funnel_data)[:500]
        
        super().__init__(
            message=message,
            context=context,
            original_exception=original_exception,
        )

