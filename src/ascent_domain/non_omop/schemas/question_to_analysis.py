from typing import List

from pydantic import BaseModel


class CohortDefinition(BaseModel):
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    index_date: str

class AnalyticalSteps(BaseModel):
    steps: List[str]

class Visualization(BaseModel):
    visualization_type: str
    description: str

class QuestionToAnalysis(BaseModel):
    original_question: str
    cohort_definition: CohortDefinition
    analytical_steps: AnalyticalSteps
    suggested_visualizations: List[Visualization]