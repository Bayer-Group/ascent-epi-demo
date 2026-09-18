from typing import List, Tuple

from ascent_domain.models.base_model import AscentBaseModel


class DatabasePairOption(AscentBaseModel):
    database_name: str
    database_schemas: List[str]


class MetaDataColumn(AscentBaseModel):
    name: str
    data_type: str
    nullable: bool
    distinct_count: int
    null_count: int
    empty_count: int
    top_values: List[Tuple[str, int | str]] | None = None
    description: str | None = None


class MetaDataTables(AscentBaseModel):
    name: str
    row_count: int
    columns_count: int
    columns: List[MetaDataColumn]


class MetaDataResponse(AscentBaseModel):
    database_name: str
    schema_name: str
    tables: List[MetaDataTables]
    tables_count: int
