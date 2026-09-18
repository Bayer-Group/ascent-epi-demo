import re
import uuid
from datetime import UTC, datetime
from enum import Enum
from itertools import chain
from typing import Annotated, Any, Dict, Generic, Iterable, List, Literal, Optional, Set, TypeVar, Union, get_args
from uuid import uuid4

import pandas as pd
from pydantic import UUID4, BaseModel, ConfigDict, Field, HttpUrl, TypeAdapter, field_validator
from sqlalchemy import JSON, UUID, Boolean, Column, DateTime, Float, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.orm import relationship

from ascent_domain.models.base_model import AscentBaseModel
from ascent_platform.db.postgresql_session import Base

QUERIES_ID = "queries.id"


FLOW = Literal["OMOP", "NON-OMOP"]


# Orm Objects
class Query(Base):
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String)
    user_email = Column(String)
    user_input = Column(String)
    interpretation = Column(String, nullable=True)  # applicable for search with uncertainty_estimation
    interpretation_index = Column(Integer, nullable=True)  # applicable for search with uncertainty_estimation
    answer = Column(String)
    query_template = Column(String)
    query_filled_preview = Column(String)
    query_filled_search = Column(String)
    database_name = Column(String)
    created_date = Column(DateTime(timezone=True), server_default=func.now())
    cohort_id = Column(UUID, ForeignKey("cohort_descriptor.id", name="queries_cohort_id_fkey", ondelete="CASCADE"), nullable=True)
    flow = Column(String, default="OMOP")
    flow_session_id = Column(UUID, ForeignKey("flow_session.id", name="flow_session_fkey", ondelete="CASCADE"), nullable=True)
    uncertainty_estimation_data = Column(
        UUID, ForeignKey("uncertainty_estimation_data.id", name="uncertainty_estimation_data_id_fkey", ondelete="CASCADE"), nullable=True
    )  # applicable for search with uncertainty_estimation


class FlowSession(Base):
    __tablename__ = "flow_session"

    id = Column(UUID, primary_key=True, default=uuid.uuid4, index=True)
    flow = Column(String, default="OMOP")


class UncertaintyEstimationData(Base):
    __tablename__ = "uncertainty_estimation_data"

    id = Column(UUID, primary_key=True, default=uuid.uuid4, index=True)
    session_id = Column(String)
    flow_session_id = Column(UUID, ForeignKey("flow_session.id", name="flow_session_fkey", ondelete="CASCADE"), nullable=True)
    user_email = Column(String)
    created_date = Column(DateTime(timezone=True), server_default=func.now())
    database_name = Column(String)
    user_input = Column(String)
    final_answer = Column(String)
    matrices = Column(JSON)
    entropies = Column(JSON)
    contributions = Column(JSON)
    uncertainty_estimate = Column(String)
    timestamp = Column(String)
    question_index = Column(Integer)


class NonOmopLog(Base):
    __tablename__ = "non_omop_analytics"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String)
    created_date = Column(DateTime(timezone=True), server_default=func.now())
    database = Column(String, nullable=True)
    question = Column(String, nullable=True)
    sql_explanation = Column(String, nullable=True)
    sql_query = Column(String, nullable=True)
    exec_time_analysis = Column(Float, nullable=True)
    exec_time_sql = Column(Float, nullable=True)
    sql_result = Column(String, nullable=True)
    success = Column(Boolean, nullable=True)
    trace_id = Column(String, nullable=True)


class QueryResult(Base):
    __tablename__ = "query_results"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey(QUERIES_ID, ondelete="CASCADE"))
    database_name = Column(String)
    database_data = Column(JSON)

    query = relationship("Query")


class MedicalConcepts(Base):
    __tablename__ = "medical_concepts"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey(QUERIES_ID, ondelete="CASCADE"))
    concept_name = Column(String)
    domain_id = Column(String)
    vocabulary = Column(JSON)
    top_k = Column(Integer)
    llm_filter = Column(String)
    cosine_similarity = Column(Numeric)
    standard_concept = Column(String)
    database = Column(String)
    concepts = Column(JSON)

    query = relationship("Query")


class ExecutedQuery(Base):
    __tablename__ = "executed_queries"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey(QUERIES_ID, ondelete="CASCADE"))
    database_name = Column(String)
    sql_text = Column(String)

    query = relationship("Query")


class ApplicationError(Base):
    __tablename__ = "application_errors"

    id = Column(Integer, primary_key=True, index=True)
    message = Column(String)
    user_id = Column(String)
    created_date = Column(DateTime(timezone=True), server_default=func.now())


class SQLSelfHealingLog(Base):
    __tablename__ = "sql_self_healing_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(String)
    session_id = Column(String, nullable=True)
    user_input = Column(String)
    query_template = Column(Text)
    initial_query = Column(Text)
    final_query = Column(Text)
    error_message = Column(Text)
    attempts = Column(Integer)
    max_retries_reached = Column(Boolean)
    success = Column(Boolean)
    created_date = Column(DateTime(timezone=True), server_default=func.now())


class UserInfo(Base):
    __tablename__ = "user_analytics"

    id = Column(Integer, primary_key=True, index=True)
    action = Column(String)
    page = Column(String)
    data = Column(JSON)
    user_id = Column(String)
    session_id = Column(String)
    created_date = Column(DateTime(timezone=True), server_default=func.now())


class ResultsFeedback(Base):
    __tablename__ = "results_feedback"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=True)
    email = Column(String, nullable=True)
    database = Column(String, nullable=True)
    contact_permission = Column(String, nullable=True)
    message = Column(String, nullable=True)
    sql = Column(String, nullable=True)
    created_date = Column(DateTime(timezone=True), server_default=func.now())


class PersonalizedQuestion(Base):
    __tablename__ = "personalized_questions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String)
    questions = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PreferredCodingSystem(Base):
    __tablename__ = "preferred_coding_system"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String)
    preferred_coding_system = Column(String, default="Standard")


class SelectedDatabase(Base):
    __tablename__ = "selected_databases"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String)
    databases = Column(JSON, default=["SYNTHETIC_EHR_OMOP", "SYNTHETIC_EHR_OMOP", "SYNTHETIC_EHR_OMOP"])


class DBCohortDescriptor(Base):
    __tablename__ = "cohort_descriptor"

    # Shared fields
    id = Column(UUID, primary_key=True, index=True)
    name = Column(String, nullable=True)
    created = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    state = Column(String, nullable=False)
    description = Column(String, nullable=True)
    origin = Column(String, nullable=False)
    database = Column(String, nullable=False)
    database_schema = Column(String, nullable=True)
    table = Column(String, nullable=False)
    size = Column(Integer, nullable=False)
    attributes = Column(JSON, nullable=False)

    # User-cohort fields
    owner_id = Column(String, nullable=True)
    criteria = Column(JSON, nullable=True)
    coding_system = Column(String, nullable=True)
    user_concepts = Column(JSON, nullable=True)
    query_template = Column(String, nullable=True)
    query = Column(String, nullable=True)
    explanation = Column(String, nullable=True)
    covariates = Column(JSON, nullable=True)
    funnel = relationship("DBFunnelCriterion", backref="cohort", cascade="all, delete, delete-orphan")
    access = relationship("DBCohortAccess", backref="cohort", cascade="all, delete, delete-orphan")

    # Study-cohort fields
    study_id = Column(UUID, ForeignKey("study_descriptor.id"), nullable=True)


_COHORT_ROLE_ORDER = {"viewer": 0, "editor": 1, "owner": 2}


def role_at_least(role: "str | None", required: str) -> bool:
    """True when ``role`` grants at least the ``required`` cohort access level."""
    return role is not None and _COHORT_ROLE_ORDER[role] >= _COHORT_ROLE_ORDER[required]


class DBCohortAccess(Base):
    __tablename__ = "cohort_access"
    __table_args__ = (UniqueConstraint("cohort_id", "user_email", name="_cohort_access_cohort_user_uc"),)

    id = Column(UUID, primary_key=True, index=True)
    cohort_id = Column(UUID, ForeignKey("cohort_descriptor.id", ondelete="CASCADE"), nullable=False, index=True)
    user_email = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)
    granted_by = Column(String, nullable=True)
    granted_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class DBFunnelCriterion(Base):
    __tablename__ = "funnel_criterion"
    __table_args__ = (UniqueConstraint("cohort_id", "index", name="_cohort_id_index_uc"),)

    id = Column(UUID, primary_key=True, index=True)
    index = Column(Integer, nullable=False)
    description = Column(String, nullable=False)
    query_template = Column(String, nullable=False)
    query = Column(String, nullable=False)
    type = Column(String, nullable=False)
    size = Column(Integer, nullable=False)
    cohort_id = Column(UUID, ForeignKey("cohort_descriptor.id", ondelete="CASCADE"), nullable=False)


class DBStudyDescriptor(Base):
    __tablename__ = "study_descriptor"

    id = Column(UUID, primary_key=True, index=True)
    ext_id = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    protocol_link = Column(String, nullable=False)
    cohorts = relationship("DBCohortDescriptor", backref="study")


class DBConceptListRevision(Base):
    __tablename__ = "concept_list_revision"
    __table_args__ = (UniqueConstraint("concept_list_id", "name", name="_concept_list_id_name_uc"),)

    id = Column(UUID, primary_key=True, index=True)
    concept_list_id = Column(UUID, ForeignKey("concept_list.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    created = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    database = Column(String, nullable=True)
    description = Column(String, nullable=True)
    concepts = Column(JSON, nullable=False)


class DBConceptList(Base):
    __tablename__ = "concept_list"
    __table_args__ = (UniqueConstraint("name", "category", "owner_id", name="_name_category_owner_id_uc"),)

    id = Column(UUID, primary_key=True, index=True)
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    owner_id = Column(String, nullable=False)
    description = Column(String, nullable=True)
    revisions = relationship("DBConceptListRevision", backref="concept_list", cascade="all, delete, delete-orphan")


# Model objects


class AscentBaseModelFrozen(BaseModel):
    model_config = ConfigDict(validate_assignment=True, from_attributes=True, extra="ignore", populate_by_name=True, frozen=True)


class PatchModel(AscentBaseModel):
    model_config = ConfigDict(validate_assignment=True, from_attributes=True)

    def apply(self, instance: Any, **kwargs) -> Set[str]:
        patch = self.model_dump(exclude_unset=True, mode="json") | kwargs
        for key, value in patch.items():
            setattr(instance, key, value)

        return set(patch.keys())


T = TypeVar("T", bound=AscentBaseModel)


class PaginatedResponse(AscentBaseModel, Generic[T]):
    items: List[T]
    page: int = Field(ge=1, description="Current page number")
    page_size: int = Field(ge=1, description="Maximum items per page")
    total: int = Field(ge=0, description="Total number of items available")


Deployment = Literal["local", "dev", "prod"]


class VersionInfo(AscentBaseModel):
    deployment: Deployment | str = Field(..., description="Deployment environment (local, dev, prod etc.)")
    version: str | None = Field(..., description="Version (commit hash) of the release")
    ascent_ai_version: str | None = Field(..., description="Ascent AI Package version")
    ascent_non_omop_version: str | None = Field(..., description="Ascent non OMOP Package version")
    openai_api_version: str | None = Field(..., description="OpenAI API version")

    @property
    def version_short(self) -> str | None:
        return self.version[:8] if self.version else None

    @field_validator("version")
    def _validate_version(cls, value: str | None) -> str | None:
        if value:
            return re.sub(r"[^0-9a-fA-F]", "", value).lower()
        return None


class ColumnInfo(BaseModel):
    name: str
    type: str


class TableInfo(AscentBaseModel):
    name: str
    size: int = Field(ge=0)
    size_bytes: int = Field(ge=0)
    columns: List[ColumnInfo]


CovariateSource = Literal["inline", "extended"]


class CovariateInfo(BaseModel):
    name: str
    type: str
    source: CovariateSource
    definition: str | None = None
    created: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class StandardConcept(AscentBaseModel):
    concept_id: int = Field(alias="CONCEPT_ID")
    concept_name: str = Field(alias="CONCEPT_NAME")
    concept_code: str | None = Field(alias="CONCEPT_CODE")
    vocabulary_id: str = Field(alias="VOCABULARY_ID")
    patient_count: int | None = Field(None, ge=0, alias="PATIENT_COUNT")
    similarity_score: float | None = Field(None, ge=0.0, le=1.0, alias="SIMILARITY_SCORE")

    @field_validator("similarity_score", mode="before")
    def _validate_similarity_score(cls, score) -> float | None:
        # cast value and make sure it's within boundaries
        if score is not None:
            return max(0.0, min(float(score), 1.0))
        return None


ConceptCategory = str
ConceptLiteral = Literal["observation", "condition", "procedure", "measurement", "drug"]
CONCEPT_CATEGORIES: List[ConceptLiteral] = list(get_args(ConceptLiteral))


class MedicalConcept(AscentBaseModel):
    name: str = Field(min_length=1)
    category: ConceptCategory
    concepts: List[StandardConcept]
    return_date: str = "first"

    @classmethod
    def from_medical_coder(cls, data: Dict[str, Any]) -> "MedicalConcept":
        concepts = data["value"]

        if isinstance(concepts, dict):
            # for drug class concepts, value is not a list but a dictionary of lists
            concepts = list(chain(*concepts.values()))

        unique_concepts = {d["CONCEPT_ID"]: d for d in concepts}
        concepts = list(unique_concepts.values())

        return cls(name=data["name"], category=data["category"], concepts=concepts)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "endometriosis",
                    "category": "condition",
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
    }


ExplorerConcepts = Annotated[
    Dict[str, str],
    Field(
        description="User defined medical concepts to be used for query substitution. "
        "The name of the medical concept is used as key while the value is a comma separated string of concept IDs. ",
        json_schema_extra={"examples": [{"endometriosis": "200461,4132140"}]},
    ),
]


UserConcepts = Annotated[
    List[MedicalConcept],
    Field(
        description="User defined medical concepts to be used for query substitution. "
        "Also accepts concept list revision IDs which are used to load stored concepts from database. "
        "Concepts and revision IDs can be mixed freely."
    ),
]


class Criteria(AscentBaseModel):
    include: List[str] = Field(default_factory=list)
    exclude: List[str] = Field(default_factory=list)
    index_date: str | None = Field(None, description="Index date definition")

    @field_validator("index_date", mode="before")
    def _validate_index_date(cls, value) -> str | None:
        if isinstance(value, Iterable) and not isinstance(value, str):
            return ", ".join(map(str, value))
        return str(value) if value else None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "include": [
                        "Women must have at least two diagnoses of endometriosis between 2010 and 2019",
                        "Women must be aged 16–45 years at the date of first diagnosis of endometriosis",
                        "Patient must have received at least one diagnosis of endometriosis",
                    ],
                    "exclude": [],
                    "index_date": "the date of first diagnosis of endometriosis",
                }
            ]
        }
    }


CriteriaType = Literal["include", "exclude"]


class FunnelCriterion(AscentBaseModel):
    id: UUID4 = Field(default_factory=uuid4)
    index: int = Field(ge=0)
    description: str
    query_template: str
    query: str
    type: CriteriaType
    size: int = Field(ge=0, description="Number of patients remaining in a cohort after this criterion has been applied.")


CohortState = Literal[
    # The cohort has been created but was not explicitly saved by the user.
    # If the cohort remains in this state for too long it may be deleted by the garbage collector.
    "pending",
    # Healthy state of the cohort.
    "active",
    # The corresponding cohort table cannot be found.
    # We hide it from the user for now and try to recover it at a later point.
    "orphaned",
]
CohortOrigin = Literal["user", "study"]
CohortRole = Literal["owner", "editor", "viewer"]


class CohortDescriptorBase(AscentBaseModel):
    id: UUID4 = Field(default_factory=uuid4)
    name: str | None = None
    created: datetime = Field(default_factory=lambda: datetime.now(UTC))
    state: CohortState = "pending"
    description: str | None = None
    origin: CohortOrigin
    database: str
    database_schema: str | None = None
    table: str
    size: int = Field(ge=0)
    attributes: List[ColumnInfo] = Field(min_length=1)

    @property
    def id_column(self) -> ColumnInfo:
        for attribute in self.attributes:
            if attribute.name.lower() in {"subject_id", "person_id", "patient_id"}:
                return attribute
        raise ValueError(f"Cohort table {self.table} has no ID column")


class StudyCohortDescriptor(CohortDescriptorBase):
    origin: Literal["study"] = "study"
    study_id: UUID4


CodingSystem = Literal["Standard", "Source"]


class UserCohortDescriptor(CohortDescriptorBase):
    origin: Literal["user"] = "user"
    owner_id: str
    criteria: Criteria | None = None
    coding_system: CodingSystem | None = "Standard"
    user_concepts: List[MedicalConcept] | None = None
    query_template: str | None = None
    query: str | None = None
    funnel: List[FunnelCriterion] = Field(default_factory=list)
    explanation: str | None = None
    covariates: List[CovariateInfo] | None = None


CohortDescriptor = Annotated[
    Union[
        StudyCohortDescriptor,
        UserCohortDescriptor,
    ],
    Field(discriminator="origin"),
]
CohortDescriptorAdapter = TypeAdapter(CohortDescriptor)


class UserCohortDescriptorPatchPublic(PatchModel):
    name: str | None = None
    description: str | None = None


class UserCohortDescriptorPatch(UserCohortDescriptorPatchPublic):
    name: str | None = None
    state: CohortState | None = None
    description: str | None = None
    size: int | None = None
    attributes: List[ColumnInfo] | None = None
    covariates: List[CovariateInfo] | None = None


class StudyDescriptor(AscentBaseModel):
    id: UUID4 = Field(default_factory=uuid4)
    ext_id: str
    name: str
    description: str | None = None
    protocol_link: HttpUrl
    cohorts: List[StudyCohortDescriptor] = Field(min_length=1)


class User(BaseModel):
    email: str | None = None
    name: str | None = None
    family_name: str | None = None
    given_name: str | None = None
    sub: str | None = None
    picture: str | None = None
    exp: int | None = None
    iss: str | None = None
    preferred_username: str | None = None
    groups: list[str] | None = None


class Concept(BaseModel):
    CONCEPT_ID: int
    CONCEPT_NAME: str
    CONCEPT_CODE: str
    VOCABULARY_ID: str
    PATIENT_COUNT: Optional[int] = None


class MultiSearchConcept(AscentBaseModelFrozen):
    name: str
    domain_id: str

    @field_validator("domain_id", "name", mode="after")
    def _validate_domain_id(cls, value: str) -> str:
        if isinstance(value, str):
            return value.capitalize()
        return value.capitalize()


class SearchType(str, Enum):
    CONCEPT_ID = "concept_id"
    CONCEPT_CODE = "concept_code"
    CONCEPT_NAME = "concept_name"


class BulkConceptSearchRequest(BaseModel):
    query: str | None = None
    database: str | None = None
    search_type: SearchType


class CohortMetadata(BaseModel):
    id: UUID4
    table_name: str
    snowflake_table_ref: str
    columns: list[str]

    @classmethod
    def from_cohort_descriptor(cls, cohort: CohortDescriptorBase) -> "CohortMetadata":
        return cls(
            id=cohort.id,
            table_name=cohort.table,
            snowflake_table_ref=f"ASCENT.ASCENT_COHORTS.{cohort.table}",
            columns=[a.name for a in cohort.attributes],
        )


class UserQuestionsResponse(BaseModel):
    id: int = 0
    user_id: str
    created_date: str
    search_text: str
    total_rows: int


class UserSearch(AscentBaseModel):
    id: int = 0
    created: datetime
    query: str
    cohort_id: UUID4 | None = None


class Frame(AscentBaseModel):
    columns: List[str]
    index: List[Any] = Field(default_factory=list)
    data: List[List[Any]] = Field(default_factory=list)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "Frame":
        return cls.model_validate(df.to_dict(orient="split"))

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.data, columns=self.columns, index=self.index)
