"""FastAPI wiring for the domain's AI services.

Only the dependency aliases live here. What they build moved to
``ascent_domain.services``: the factories are domain logic, and while they sat
in this module every domain function that named one of these aliases as a
parameter type was importing the application package to do it.

``make_search_service`` exists rather than ``Depends(SearchService)`` so the
query-library sub-dependency is still declared to FastAPI. Passing the class
directly would make ``query_library`` look like a query parameter and add it to
the OpenAPI schema.
"""

from typing import Annotated

from fastapi import Depends

from ascent_domain.omop.data.queries.query_library import QueryLibrary
from ascent_domain.omop.models.inference.criteria_to_counts import CriteriaProcessor
from ascent_domain.omop.models.inference.table_one import Table1
from ascent_domain.services import (
    SearchService as _SearchService,
)
from ascent_domain.services import (
    load_query_library,
    make_criteria_processor,
    make_table_one,
)


def make_search_service(query_library: QueryLibrary = Depends(load_query_library)) -> _SearchService:
    return _SearchService(query_library)


def make_criteria_service(_query_library: QueryLibrary = Depends(load_query_library)) -> CriteriaProcessor:
    return make_criteria_processor()


def make_table_one_service(_query_library: QueryLibrary = Depends(load_query_library)) -> Table1:
    return make_table_one()


SearchService = Annotated[_SearchService, Depends(make_search_service)]
CriteriaService = Annotated[CriteriaProcessor, Depends(make_criteria_service)]
TableOneService = Annotated[Table1, Depends(make_table_one_service)]

__all__ = [
    "CriteriaService",
    "SearchService",
    "TableOneService",
    "load_query_library",
    "make_criteria_service",
    "make_search_service",
    "make_table_one_service",
]
