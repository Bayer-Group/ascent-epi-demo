"""
Uncertainty Quantification Module

CLUES (Conditional Language Uncertainty via Entropy and Schur) framework
for decomposing semantic uncertainty into ambiguity and instability.

Split API Functions (for backend integration):
    1. generate_question_interpretations()    → returns primary + alt interpretations
    2. generate_query_templates()          → returns all templates (1 + N×M)
    3. extract_medical_entities()          → returns unique entities from templates
    4. code_medical_entities()             → returns code mappings (SINGLE PASS)
       ← SPLIT POINT: User can modify code_mappings here
    5. fill_query_templates()              → fills templates with codes
    6. get_query_explanation()       → explanation for primary query
    7. execute_queries()             → executes all queries
    8. generate_all_answers()            → generates all answers
    9. compute_clues_metrics()             → computes H(I), H(R), H(R|I)
    10. generate_combined_answer()   → combined answer from all

Convenience Function:
    answer_question_with_clues()           → runs full pipeline
"""

from .clues_api import (
    # Convenience function
    answer_question_with_clues,
    code_medical_entities,
    compute_clues_metrics,
    execute_queries,
    extract_medical_entities,
    fill_query_templates,
    generate_all_answers,
    generate_combined_answer,
    generate_query_templates,
    # Split API functions
    generate_question_interpretations,
    get_query_explanation,
)
from .uncertainty_config import (
    DEFAULT_THRESHOLDS,
    UncertaintyRegime,
    UncertaintyResult,
    UncertaintyThresholds,
)
from .uncertainty_processing import extract_clues_metrics

__all__ = [
    # Config
    "UncertaintyRegime",
    "UncertaintyThresholds",
    "UncertaintyResult",
    "DEFAULT_THRESHOLDS",
    # Convenience function
    "answer_question_with_clues",
    # Split API functions
    "generate_question_interpretations",
    "generate_query_templates",
    "extract_medical_entities",
    "code_medical_entities",
    "fill_query_templates",
    "get_query_explanation",
    "execute_queries",
    "generate_all_answers",
    "compute_clues_metrics",
    "generate_combined_answer",
    # Metrics
    "extract_clues_metrics",
]
