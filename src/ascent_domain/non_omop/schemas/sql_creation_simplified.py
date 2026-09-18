from typing import List, Optional

from pydantic import BaseModel, Field


class MedicalConcept(BaseModel):
    CONCEPT_ID: int
    CONCEPT_CODE: str
    CONCEPT_NAME: str
    DOMAIN_ID: str
    VOCABULARY_ID: str
    STANDARD_CONCEPT: Optional[str]
    PATIENT_COUNT: Optional[int]


class MedicalConceptSimplified(BaseModel):
    CONCEPT_CODE: str


class EntityReferenceSimplified(BaseModel):
    concepts: List[MedicalConceptSimplified] = Field(description="List of concepts")
    placeholder: str = Field(description="Placeholder for the entity")


class EntityReference(BaseModel):
    entity: str = Field(description="Name of the entity")
    entity_name: str = Field(description="Name of the medical concept")
    coding_systems: str = Field(description="Coding system ontologies")
    codes_with_dots: bool = Field(description="Indicates if codes contain dots")
    concepts: List[MedicalConcept] = Field(description="List of concepts")
    codes_str: str = Field(
        description="A string containing all the codes connected to this entity joined with a comma"
    )
    placeholder: str = Field(description="Placeholder for the entity")



class ExtractedConcepts(BaseModel):
    entity_references: List[EntityReference] = Field(
        description="References to entities used in the SQL"
    )


class SQLCreationSimplified(BaseModel):
    description: str = Field(
        description="Description and explanation of the entire SQL logic."
    )
    limitations: str = Field(
        description="Describe what the SQL code does not cover referring to the initial question."
    )
    cte_names: List[str] = Field(
        description="List of Common Table Expression (CTE) names used in the SQL query."
    )
    sql_code_snippet: str = Field(
        description="The complete SQL query as a single code snippet."
    )
    extracted_concepts: Optional[ExtractedConcepts] = Field(
        description="Medical concept list to be used in the SQL"
    )


class SQLRepair(BaseModel):
    sql_code_snippet: str = Field(
        description="The complete SQL query as a single code snippet."
    )
