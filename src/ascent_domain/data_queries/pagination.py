from typing import Callable, Tuple, Type, TypeVar

from sqlalchemy import ColumnElement, Select, func, select

from ascent_domain.models.data_definitions import AscentBaseModel, PaginatedResponse
from ascent_platform.db.postgresql_session import AsyncSessionLocal, Base

T = TypeVar("T", bound=AscentBaseModel)
U = TypeVar("U", bound=Base)


async def _get_total_count(db: AsyncSessionLocal, base_query) -> int:
    query = select(func.count()).select_from(base_query.subquery())
    result = await db.scalars(query)
    return result.one()


async def paginate_query(
    db: AsyncSessionLocal,
    query: Select[Tuple[U]],
    *,
    on_response: Type[T] | Callable[[U], T],
    page: int,
    page_size: int,
    order_by: ColumnElement | str | None = None,
) -> PaginatedResponse[T]:
    offset = page_size * (page - 1)
    total = await _get_total_count(db, query)

    # Apply sorting, limit, and offset
    if order_by is not None:
        query = query.order_by(order_by)
    query = query.offset(offset).limit(page_size)

    # Execute the query
    result = await db.scalars(query)

    # Build response
    _on_response = on_response.model_validate if isinstance(on_response, type) and issubclass(on_response, AscentBaseModel) else on_response
    items = list(map(_on_response, result.all()))
    return PaginatedResponse(items=items, page=page, page_size=page_size, total=total)
