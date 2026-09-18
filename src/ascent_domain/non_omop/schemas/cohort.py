from datetime import UTC, datetime
from typing import List, Literal
from uuid import uuid4

from pydantic import UUID4, BaseModel, Field


class ColumnInfo(BaseModel):
    name: str
    type: str


CohortOrigin = Literal["user", "study"]


class CohortDescriptorBase(BaseModel):
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
