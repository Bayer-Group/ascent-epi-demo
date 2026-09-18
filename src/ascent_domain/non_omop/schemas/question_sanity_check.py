from typing import List, Optional

from pydantic import BaseModel, Field


class SanityCheck(BaseModel):
    is_valid: bool = Field(..., description="Whether the query is valid")
    issues: List[str] = Field(default_factory=list, description="List of issues if any")
    suggested_reformulation: Optional[str] = Field(None, description="Suggested reformulation if needed")

class IntentClassification(BaseModel):
    primary_intent: str = Field(..., description="Main intent category")
    secondary_intent: Optional[str] = Field(None, description="Secondary intent if applicable")
    analytical_objective: str = Field(..., description="Brief description of query goal")

class MedicalEntity(BaseModel):
    mention: str = Field(..., description="Text as mentioned")
    normalized_term: str = Field(..., description="Standardized term")

class TemporalPoint(BaseModel):
    mention: str = Field(..., description="The temporal point as mentioned")
    normalized_value: Optional[str] = Field(None, description="Normalized representation")
    type: str = Field(..., description="Type of time point (date, age, event)")

class TemporalPeriod(BaseModel):
    start: Optional[TemporalPoint] = Field(None, description="Start of time period")
    end: Optional[TemporalPoint] = Field(None, description="End of time period")
    duration: Optional[str] = Field(None, description="Duration if specified")
    description: str = Field(..., description="Description of the time period")

class TemporalRelationship(BaseModel):
    entity1: str = Field(..., description="First entity in relationship")
    entity2: str = Field(..., description="Second entity in relationship")
    relationship_type: str = Field(..., description="Type of relationship (before, after, during)")
    description: str = Field(..., description="Description of the temporal relationship")

class FrequencyPattern(BaseModel):
    mention: str = Field(..., description="Frequency as mentioned")
    normalized_pattern: Optional[str] = Field(None, description="Normalized representation")
    description: str = Field(..., description="Description of the frequency pattern")

class TemporalElements(BaseModel):
    time_points: List[TemporalPoint] = Field(default_factory=list)
    time_periods: List[TemporalPeriod] = Field(default_factory=list)
    temporal_relationships: List[TemporalRelationship] = Field(default_factory=list)
    frequency_patterns: List[FrequencyPattern] = Field(default_factory=list)

class DatabaseMappingHints(BaseModel):
    suggested_tables: List[str] = Field(default_factory=list)
    key_join_fields: List[str] = Field(default_factory=list)
    required_filters: List[str] = Field(default_factory=list)

class MedicalEntities(BaseModel):
    conditions: List[MedicalEntity] = Field(default_factory=list)
    demographics: List[MedicalEntity] = Field(default_factory=list)
    providers_facilities: List[MedicalEntity] = Field(default_factory=list)
    measurements: List[MedicalEntity] = Field(default_factory=list)
    Drug: List[MedicalEntity] = Field(default_factory=list)
    Drug_classes: List[MedicalEntity] = Field(default_factory=list)
    Observation: List[MedicalEntity] = Field(default_factory=list)
    Device: List[MedicalEntity] = Field(default_factory=list)
    Condition: List[MedicalEntity] = Field(default_factory=list)
    Procedure: List[MedicalEntity] = Field(default_factory=list)
    Geography: List[MedicalEntity] = Field(default_factory=list)
    Spec_Anatomic_Site: List[MedicalEntity] = Field(default_factory=list, alias="Spec Anatomic Site")
    Unit: List[MedicalEntity] = Field(default_factory=list)
    Specimen: List[MedicalEntity] = Field(default_factory=list)
    Provider: List[MedicalEntity] = Field(default_factory=list)
    Language: List[MedicalEntity] = Field(default_factory=list)
    Race: List[MedicalEntity] = Field(default_factory=list)
    Visit: List[MedicalEntity] = Field(default_factory=list)
    Revenue_Code: List[MedicalEntity] = Field(default_factory=list, alias="Revenue Code")
    Relationship: List[MedicalEntity] = Field(default_factory=list)
    Route: List[MedicalEntity] = Field(default_factory=list)
    Currency: List[MedicalEntity] = Field(default_factory=list)
    Payer: List[MedicalEntity] = Field(default_factory=list)
    Cost: List[MedicalEntity] = Field(default_factory=list)
    Condition_Status: List[MedicalEntity] = Field(default_factory=list, alias="Condition Status")
    Episode: List[MedicalEntity] = Field(default_factory=list)
    Gender: List[MedicalEntity] = Field(default_factory=list)
    Plan_Stop_Reason: List[MedicalEntity] = Field(default_factory=list, alias="Plan Stop Reason")
    Plan: List[MedicalEntity] = Field(default_factory=list)
    Meas_Value_Operator: List[MedicalEntity] = Field(default_factory=list, alias="Meas Value Operator")
    Sponsor: List[MedicalEntity] = Field(default_factory=list)
    Ethnicity: List[MedicalEntity] = Field(default_factory=list)

class QueryAnalysis(BaseModel):
    original_query: str = Field(..., description="User's original question")
    sanity_check: SanityCheck
    intent_classification: IntentClassification
    medical_entities: MedicalEntities
    temporal_elements: TemporalElements
    database_mapping_hints: DatabaseMappingHints

class LLMOutputSanityCheck(BaseModel):
    query_analysis: QueryAnalysis
