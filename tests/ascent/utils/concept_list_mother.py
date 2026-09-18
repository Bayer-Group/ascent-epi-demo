from uuid import uuid4

from pydantic import UUID4

from ascent_domain.models.concept_list_dtos import ConceptListDTO, ConceptListMetaDTO, ConceptListRevisionDTO
from ascent_domain.models.data_definitions import ConceptCategory, MedicalConcept, StandardConcept

DEFAULT_CONCEPTS = [
    StandardConcept(
        concept_id=200461,
        concept_name="Endometriosis of uterus",
        concept_code="76376003",
        vocabulary_id="SNOMED",
        patient_count=250746,
        similarity_score=0.9108418,
    ),
    StandardConcept(
        concept_id=4132140,
        concept_name="Endometriosis of pelvis",
        concept_code="26681001",
        vocabulary_id="SNOMED",
        patient_count=0,
        similarity_score=0.9009241,
    ),
]


def default_medical_concept(
    name: str = "endometriosis", category: ConceptCategory = "condition", concepts: list[StandardConcept] | None = None
) -> MedicalConcept:
    return MedicalConcept(
        name=name,
        category=category,
        concepts=concepts or DEFAULT_CONCEPTS,
    )


def default_concept_list_revision(
    id: UUID4 | None = None, name: str | None = None, description: str | None = None, concepts: list[StandardConcept] | None = None
) -> ConceptListRevisionDTO:
    return ConceptListRevisionDTO(
        id=id or uuid4(),
        name=name or "endometriosis",
        database="SYNTHETIC_EHR_OMOP",
        description=description or "Any kind of endometriosis",
        concepts=concepts or DEFAULT_CONCEPTS,
    )


def default_concept_list_meta(
    id: UUID4 | None = None,
    name: str | None = None,
    category: ConceptCategory = "condition",
    description: str = "Description",
) -> ConceptListMetaDTO:
    return ConceptListMetaDTO(
        id=id or uuid4(),
        name=name or "endometriosis",
        category=category,
        description=description,
    )


def default_concept_list(
    id: UUID4 | None = None,
    name: str | None = None,
    category: ConceptCategory = "condition",
    description: str = "Description",
    revisions: list[ConceptListRevisionDTO] | None = None,
) -> ConceptListDTO:
    return ConceptListDTO(
        id=id or uuid4(),
        name=name or "endometriosis",
        category=category,
        description=description,
        revisions=revisions or [default_concept_list_revision()],
    )
