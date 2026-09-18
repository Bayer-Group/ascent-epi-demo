"""
Custom exceptions for the Ascent Non-OMOP pipeline.

This module provides a structured exception hierarchy that maintains full traceability
while providing human-readable error messages for each workflow step.
"""

from .base import (
    AscentBaseException,
    WorkflowException,
)
from .question_analysis import (
    QuestionAnalysisException,
    QuestionAnalysisLLMError,
    QuestionAnalysisValidationError,
)
from .sanity_check import (
    SanityCheckException,
    SanityCheckLLMError,
    SanityCheckValidationError,
)
from .sql_creation import (
    EntityReferenceError,
    PlaceholderMappingError,
    SQLCreationException,
)
from .sql_funnel import (
    FunnelGenerationError,
    FunnelValidationError,
    SQLFunnelException,
)
from .sql_preparation import (
    MedicalConceptExtractionError,
    MetadataRetrievalError,
    SQLPreparationException,
    SQLPreparationLLMError,
    SQLPreparationValidationError,
    SQLRepairError,
)
from .sql_results import (
    DatabaseConnectionError,
    SQLExecutionError,
    SQLHealingError,
    SQLResultsException,
)
from .sql_to_question import (
    QuestionComparisonError,
    QuestionReconstructionError,
    SQLToQuestionException,
)

__all__ = [
    # Base
    "AscentBaseException",
    "WorkflowException",

    # Sanity Check
    "SanityCheckException",
    "SanityCheckValidationError",
    "SanityCheckLLMError",

    # Question Analysis
    "QuestionAnalysisException",
    "QuestionAnalysisValidationError",
    "QuestionAnalysisLLMError",

    # SQL Preparation
    "SQLPreparationException",
    "SQLPreparationLLMError",
    "SQLPreparationValidationError",
    "MetadataRetrievalError",
    "MedicalConceptExtractionError",
    "SQLRepairError",

    # SQL Creation
    "SQLCreationException",
    "PlaceholderMappingError",
    "EntityReferenceError",

    # SQL Results
    "SQLResultsException",
    "SQLExecutionError",
    "SQLHealingError",
    "DatabaseConnectionError",

    # SQL Funnel
    "SQLFunnelException",
    "FunnelGenerationError",
    "FunnelValidationError",

    # SQL to Question
    "SQLToQuestionException",
    "QuestionComparisonError",
    "QuestionReconstructionError",
]
