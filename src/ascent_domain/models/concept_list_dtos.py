from datetime import UTC, datetime
from typing import List, Literal, TypeAlias

from pydantic import UUID4, ConfigDict, Field

from ascent_domain.models.data_definitions import AscentBaseModel, ConceptCategory, PatchModel, StandardConcept


class Concepts(AscentBaseModel):
    type: Literal["concepts"] = "concepts"
    database: str
    concepts: List[StandardConcept]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "type": "concepts",
                    "database": "SYNTHETIC_EHR_OMOP",
                    "concepts": [
                        {
                            "CONCEPT_ID": 200461,
                            "CONCEPT_NAME": "Endometriosis of uterus",
                            "CONCEPT_CODE": "76376003",
                            "VOCABULARY_ID": "SNOMED",
                            "PATIENT_COUNT": 250746,
                            "SIMILARITY_SCORE": 0.9108418,
                        },
                        {
                            "CONCEPT_ID": 4132140,
                            "CONCEPT_NAME": "Endometriosis of pelvis",
                            "CONCEPT_CODE": "26681001",
                            "VOCABULARY_ID": "SNOMED",
                            "PATIENT_COUNT": 0,
                            "SIMILARITY_SCORE": 0.9009241,
                        },
                    ],
                }
            ]
        }
    )


class ConceptIDs(AscentBaseModel):
    type: Literal["concept_ids"] = "concept_ids"
    database: str | None = None
    concept_ids: List[int]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "type": "concept_ids",
                    "database": "SYNTHETIC_EHR_OMOP",
                    "concept_ids": [200461, 4132140],
                }
            ]
        }
    )


class ConceptCodes(AscentBaseModel):
    type: Literal["concept_codes"] = "concept_codes"
    database: str | None = None
    concept_codes: List[str]
    vocabulary_id: str

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "type": "concept_codes",
                    "database": "SYNTHETIC_EHR_OMOP",
                    "concept_codes": ["76376003", "26681001"],
                    "vocabulary_id": "SNOMED",
                }
            ]
        }
    )


AnyConcepts: TypeAlias = Concepts | ConceptIDs | ConceptCodes


class ConceptListRevisionRequestDTO(AscentBaseModel):
    name: str = Field(min_length=1, json_schema_extra={"examples": ["endometriosis"]})
    description: str | None = None
    concepts: Concepts


class ConceptListRevisionDTO(AscentBaseModel):
    id: UUID4 = Field()
    name: str = Field(min_length=1)
    created: datetime = Field(default_factory=lambda: datetime.now(UTC))
    database: str | None = None
    description: str | None = None
    concepts: List[StandardConcept]


class ConceptListMetaDTO(AscentBaseModel):
    id: UUID4 = Field()
    name: str = Field(min_length=1)
    category: ConceptCategory
    description: str | None = None


class ConceptListDTO(ConceptListMetaDTO):
    revisions: List[ConceptListRevisionDTO]


class ConceptListCreateRequestDTO(AscentBaseModel):
    name: str = Field(min_length=1)
    description: str | None = None
    category: ConceptCategory
    revision: ConceptListRevisionRequestDTO | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "endometriosis",
                    "category": "condition",
                }
            ]
        },
    )


class ConceptListPatchRequestDTO(PatchModel):
    description: str | None = None

    model_config = ConfigDict(json_schema_extra={"examples": [{"description": "New description"}]})
