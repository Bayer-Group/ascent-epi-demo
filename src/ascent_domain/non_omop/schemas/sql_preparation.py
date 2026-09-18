from typing import List

from pydantic import BaseModel, Field


class TableAttribute(BaseModel):
    table_name: str
    columns_needed: List[str]


class MedicalCoding(BaseModel):
    medical_entity: str
    domain_id: str
    codes_with_dots: bool
    coding_systems: List[str]
    notes: str


# class SqlStep(BaseModel):
#     step_number: int
#     step_name: str
#     description: str
#     sql_cte_name: Optional[str] = None
#     sql_code_snippet: Optional[str] = None


class KeyConsideration(BaseModel):
    consideration: str
    recommendation: str


class NextStep(BaseModel):
    step_order: int
    description: str


class SQLpreparation(BaseModel):
    original_question: str = Field(description="The original question posed by the user")
    goal: str = Field(description="The overarching analysis goal")
    relevant_tables: List[TableAttribute] = Field(
        description="List of database tables and relevant attributes identified")
    medical_coding: List[MedicalCoding] = Field(description="Medical entities, their domain and associated coding standards used")
    # sql_preparation_steps: List[SqlStep] = Field(
    #     description="List of steps with explanations used to prepare and execute SQL query")
    key_considerations: List[KeyConsideration] = Field(
        description="Important considerations and potential refinements identified")
    next_steps: List[NextStep] = Field(description="Action items for further analysis")
