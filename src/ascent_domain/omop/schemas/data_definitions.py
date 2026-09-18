from datetime import UTC, datetime
from typing import List, Literal, Optional
from uuid import UUID, uuid4

from pydantic import UUID4, BaseModel, ConfigDict, Field


class AscentBaseModel(BaseModel):
    model_config = ConfigDict(validate_assignment=True, from_attributes=True, extra="ignore", populate_by_name=True)


class ColumnInfo(BaseModel):
    name: str
    type: str


CohortOrigin = Literal["user", "study"]


class CohortDescriptorBase(AscentBaseModel):
    id: UUID4 = Field(default_factory=uuid4)
    name: str | None = None
    created: datetime = Field(default_factory=lambda: datetime.now(UTC))
    description: str | None = None
    origin: CohortOrigin
    database: str
    table: str
    size: int = Field(ge=0)
    attributes: List[ColumnInfo] = Field(min_length=1)


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


class Query(BaseModel):
    id: Optional[int] = None
    user_input: Optional[str] = None
    gpt_answer: str
    initial_prompt: str
    text_sql_template: str
    df_recs_list_out: str
    question_masked: str
    created_date: datetime = datetime.now()
    cohort_id: Optional[UUID] = None


