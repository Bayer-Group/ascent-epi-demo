"""
Pydantic schemas for uncertainty and disambiguation models.

This module contains schema definitions used across disambiguation and alternative_interpretations
to avoid circular imports.
"""

from typing import List

from pydantic import BaseModel, Field


class DisambiguationResponse(BaseModel):
    """Schema for the complete disambiguation analysis"""

    original_question: str = Field(..., description="The original question that was analyzed")
    is_ambiguous: bool = Field(..., description="Whether the question contains meaningful ambiguities")
    ambiguity_score: float = Field(..., description="Overall ambiguity score from 0 (unambiguous) to 1 (highly ambiguous)")
    final_interpretations: List[str] = Field(
        ...,
        description="Final list of distinct question interpretations that represent all meaningful combinations of ambiguity resolutions",
    )


class ExploreMeaningsResponse(BaseModel):
    """
    Schema for 'Explore Meanings' prompt - Stage 1 uncertainty estimation.

    This prompt is designed to generate diverse semantic interpretations for
    measuring aleatoric uncertainty H(I). It explores the full breadth of
    potential meanings without domain constraints.
    """

    original_question: str = Field(..., description="The original question analyzed")
    interpretations: List[str] = Field(
        ...,
        description=(
            "Diverse semantic interpretations exploring different possible meanings. "
            "Each interpretation should resolve ambiguities differently to capture "
            "the interpretation space."
        ),
    )
