import logging
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import AsyncIterator, List, Set

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
from duckdb import Error as WarehouseError
from snowflake.connector import SnowflakeConnection
from snowflake.connector.cursor import SnowflakeCursor
from sqlalchemy import delete, or_, select, update
from sqlalchemy.orm import selectinload

from ascent_domain.errors import AccessDeniedError, ConflictError, InvalidRequestError, NotFoundError
from ascent_domain.models.data_definitions import (
    CohortDescriptor,
    CohortDescriptorAdapter,
    CohortDescriptorBase,
    CohortOrigin,
    CohortRole,
    ColumnInfo,
    DBCohortAccess,
    DBCohortDescriptor,
    DBFunnelCriterion,
    DBStudyDescriptor,
    FunnelCriterion,
    StudyDescriptor,
    TableInfo,
    User,
    UserCohortDescriptorPatch,
    role_at_least,
)
from ascent_platform.config.runtime import get_runtime_settings
from ascent_platform.db.postgresql_session import AsyncSessionLocal
from ascent_platform.warehouse.db_type import is_omop_database
from ascent_platform.warehouse.session import execute_query, get_db

logger = logging.getLogger(__name__)

USER_COHORT_TABLE_INFIX = "_USR_"
OWNED_COHORT_TABLE_PREFIX = f"{get_runtime_settings().RELEASE_VERSION.upper()}{USER_COHORT_TABLE_INFIX}"

# Minimum age before garbage collection may touch a cohort. Cohort creation
# writes the Snowflake table BEFORE committing the descriptor, so a GC sweep
# in that window would see the fresh table as unreferenced and drop it (and,
# conversely, could flag a fresh descriptor whose table it read just too early
# as orphaned). Anything younger than this is left alone.
GC_MIN_COHORT_AGE = timedelta(hours=24)

QUERY_USER_COHORT_TABLES = rf"""
SELECT TABLE_NAME
FROM information_schema.tables
WHERE TABLE_TYPE = 'BASE TABLE'
  AND UPPER(TABLE_SCHEMA) = 'ASCENT_COHORTS'
  AND TABLE_NAME LIKE '%{USER_COHORT_TABLE_INFIX}%'
ORDER BY TABLE_NAME;
""".strip()

QUERY_USER_COHORT_TABLES_OWNED = rf"""
SELECT TABLE_NAME
FROM information_schema.tables
WHERE TABLE_TYPE = 'BASE TABLE'
  AND UPPER(TABLE_SCHEMA) = 'ASCENT_COHORTS'
  AND TABLE_NAME LIKE '{OWNED_COHORT_TABLE_PREFIX}%'
  AND CREATED < DATEADD('hour', -{int(GC_MIN_COHORT_AGE.total_seconds() // 3600)}, CURRENT_TIMESTAMP())
ORDER BY TABLE_NAME;
""".strip()

_TYPE_LOOKUP = {
    "subject_id": "NUMBER",
    "person_id": "NUMBER",
    "index_date": "DATE",
}
_NAME_LOOKUP = {
    "subject_id": "SUBJECT_ID",
    "person_id": "SUBJECT_ID",
    "index_date": "INDEX_DATE",
}

_RESERVED_COLUMNS = {"SUBJECT_ID", "INDEX_DATE"}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

TEMP_COHORT_TABLE_PREFIX = "TMP_COHORT_"

# Snowflake error text raised when a session without ASCENT grants (i.e. any
# SSO user) references the cohort table. Used to refuse "healed" results that
# merely dropped the cohort from the query.
ASCENT_NOT_AUTHORIZED_MARKER = "'ASCENT' does not exist or not authorized"


def _sanitize_identifier(name: str) -> str:
    """Uppercase a covariate column name, rejecting anything that isn't a plain
    SQL identifier — covariate names are interpolated into DDL, so this guards
    against identifier injection."""
    candidate = str(name).strip().upper()
    if not _IDENTIFIER_RE.match(candidate):
        raise ValueError(f"Invalid covariate column name {name!r}: must match [A-Za-z_][A-Za-z0-9_]*")
    return candidate


def _subject_id_type(values: "pd.Series") -> str:
    """NUMBER when the patient identifier is numeric, VARCHAR when it is not.

    OMOP person_id is numeric and so are the claims patient ids upstream, so
    this was hardcoded to NUMBER. The synthetic source layer keys on MRN
    ("RBM0000001"), and a hardcoded NUMBER fails the insert with
    'Could not convert string "RBM0000001" to DECIMAL(38,0)' -- after the
    table has already been created. Deciding from the data keeps both kinds
    of source working.
    """
    import pandas as pd

    if pd.api.types.is_numeric_dtype(values):
        return "NUMBER"
    coerced = pd.to_numeric(values, errors="coerce")
    return "NUMBER" if coerced.notna().all() and len(values) else "VARCHAR"


def _infer_sf_type(series: pd.Series) -> str:
    """Map a pandas Series to a Snowflake column type for a covariate column."""
    values = series.dropna()
    if values.empty:
        return "VARCHAR"
    if ptypes.is_bool_dtype(values):
        return "BOOLEAN"
    if ptypes.is_integer_dtype(values):
        return "NUMBER"
    if ptypes.is_float_dtype(values):
        return "FLOAT"
    if ptypes.is_datetime64_any_dtype(values):
        return "DATE"
    coerced = pd.to_numeric(values, errors="coerce")
    if coerced.notna().all():
        return "NUMBER" if (coerced == coerced.round()).all() else "FLOAT"
    return "VARCHAR"


def _normalize_date_series(series: pd.Series) -> pd.Series:
    # Snowflake is picky about the input format, see `SHOW PARAMETERS LIKE 'DATE_INPUT_FORMAT'`.
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d").replace({pd.NA: None, pd.NaT: None, np.nan: None})


async def _write_cohort_table(
    ascent_db: SnowflakeCursor,
    table_name: str,
    cohort: pd.DataFrame,
    spec: List[tuple],
    create_or_replace: bool = False,
) -> None:
    """Create (or replace) a cohort table and bulk-insert ``cohort``.

    ``spec`` is an ordered list of ``(df_column, sf_name, sf_type)``. Only the
    sanitized ``sf_name``/``sf_type`` are interpolated into the DDL; all values
    are parameterized.
    """
    for df_col, _, sf_type in spec:
        if sf_type == "DATE":
            cohort[df_col] = _normalize_date_series(cohort[df_col])

    df_columns = [df_col for df_col, _, _ in spec]
    column_types = ", ".join(f"{sf_name} {sf_type}" for _, sf_name, sf_type in spec)
    column_names = ", ".join(sf_name for _, sf_name, _ in spec)
    placeholders = ", ".join("?" for _ in spec)

    verb = "CREATE OR REPLACE TABLE" if create_or_replace else "CREATE TABLE"
    query_create = rf"""{verb} ASCENT_COHORTS.{table_name} ({column_types});"""
    query_insert = rf"""INSERT INTO ASCENT_COHORTS.{table_name} ({column_names}) VALUES ({placeholders});"""
    logger.debug(f"Table query: {query_create!r}")
    logger.debug(f"Insert query: {query_insert!r}")

    await execute_query(ascent_db, query_create)
    if not cohort.empty:
        await execute_query(ascent_db, query_insert, params=list(cohort[df_columns].itertuples(index=False)), bulk=True)
    else:
        logger.info("Cohort is empty")


def _build_covariate_spec(cohort: pd.DataFrame, covariate_columns: List[str], reserved: Set[str]) -> List[tuple]:
    spec: List[tuple] = []
    seen = set(reserved)
    for col in covariate_columns:
        sf_name = _sanitize_identifier(col)
        if sf_name in seen:
            raise ValueError(f"Covariate column {col!r} collides with an existing column {sf_name!r}")
        seen.add(sf_name)
        spec.append((col, sf_name, _infer_sf_type(cohort[col])))
    return spec


class CohortNotFoundError(NotFoundError):
    def __init__(self, *cohort_ids: uuid.UUID) -> None:
        super().__init__(
            detail={
                "error": "Not found",
                "cohort_ids": list(map(str, cohort_ids)),
            },
        )


class CohortAccessDeniedError(AccessDeniedError):
    def __init__(self, cohort_id: uuid.UUID, required: CohortRole) -> None:
        super().__init__(
            detail={
                "error": f"This action requires '{required}' access to the cohort.",
                "cohort_id": str(cohort_id),
            },
        )


class CohortNotQueryableError(ConflictError):
    def __init__(self, cohort_id: uuid.UUID, state: str) -> None:
        super().__init__(
            detail={
                "error": (f"Cohort is in state '{state}' and cannot be queried. Its backing table is no longer available; recreate the cohort."),
                "cohort_id": str(cohort_id),
            },
        )


class CohortTableMissingError(ConflictError):
    def __init__(self, cohort_id: uuid.UUID, table: str) -> None:
        super().__init__(
            detail={
                "error": (
                    f"The cohort's backing table {table!r} no longer exists in Snowflake "
                    "(it was likely garbage-collected). The cohort cannot be queried; recreate it."
                ),
                "cohort_id": str(cohort_id),
            },
        )


class CohortDemographicsUnsupportedError(InvalidRequestError):
    def __init__(self, cohort_id: uuid.UUID, database: str) -> None:
        super().__init__(
            detail={
                "error": (
                    f"Demographics (Table One) is only available for cohorts on OMOP databases; "
                    f"this cohort was built on the non-OMOP database {database!r}."
                ),
                "cohort_id": str(cohort_id),
            },
        )


QUERYABLE_COHORT_STATES: frozenset = frozenset({"active", "pending"})


def ensure_cohort_queryable(descriptor) -> None:
    """Reject cohorts whose backing table is known to be gone (state 'orphaned')
    before any Snowflake query references it — a clean 409 instead of a raw
    SQL compilation error."""
    if descriptor.state not in QUERYABLE_COHORT_STATES:
        raise CohortNotQueryableError(descriptor.id, descriptor.state)


def ensure_cohort_demographics_supported(descriptor) -> None:
    """Reject non-OMOP cohorts before any Table One SQL reaches Snowflake — the
    demographics queries reference OMOP-CDM tables (PERSON, CONCEPT, ...), which
    only exist in OMOP databases."""
    if not is_omop_database(descriptor.database):
        raise CohortDemographicsUnsupportedError(descriptor.id, descriptor.database)


def resolve_cohort_role(db_descriptor: DBCohortDescriptor, user_email: str | None) -> CohortRole | None:
    """The current user's effective role on a cohort, or ``None`` if no access.

    Owner is the immutable creator (``owner_id``); ``cohort_access`` holds
    editor/viewer grants; study/public cohorts are viewable by everyone.
    """
    if user_email and db_descriptor.owner_id == user_email:
        return "owner"
    for grant in db_descriptor.access:
        if grant.user_email == user_email:
            return grant.role
    if db_descriptor.origin == "study" or db_descriptor.owner_id is None:
        return "viewer"
    return None


def _cohort_visible_filter(user_email: str | None):
    return or_(
        DBCohortDescriptor.owner_id == user_email,
        DBCohortDescriptor.owner_id.is_(None),
        DBCohortDescriptor.origin == "study",
        DBCohortDescriptor.id.in_(select(DBCohortAccess.cohort_id).where(DBCohortAccess.user_email == user_email)),
    )


async def _load_cohort_for_user(
    app_db: AsyncSessionLocal, user: User, cohort_id: uuid.UUID, required: CohortRole
) -> tuple[DBCohortDescriptor, CohortRole]:
    """Load a cohort ORM row and enforce the user has at least ``required`` access."""
    query = (
        select(DBCohortDescriptor)
        .where(DBCohortDescriptor.id == cohort_id)
        .options(selectinload(DBCohortDescriptor.funnel), selectinload(DBCohortDescriptor.access))
    )
    db_descriptor = (await app_db.scalars(query)).one_or_none()
    if db_descriptor is None:
        raise CohortNotFoundError(cohort_id)
    role = resolve_cohort_role(db_descriptor, user.email)
    if role is None:
        raise CohortNotFoundError(cohort_id)
    if not role_at_least(role, required):
        raise CohortAccessDeniedError(cohort_id, required)
    return db_descriptor, role


async def authorize_cohort(
    app_db: AsyncSessionLocal, user: User, cohort_id: uuid.UUID, required: CohortRole = "viewer"
) -> tuple[CohortDescriptor, CohortRole]:
    """Return ``(descriptor, role)`` if the user has at least ``required`` access."""
    db_descriptor, role = await _load_cohort_for_user(app_db, user, cohort_id, required)
    return CohortDescriptorAdapter.validate_python(db_descriptor), role


async def grant_cohort_access(app_db: AsyncSessionLocal, current_user: User, cohort_id: uuid.UUID, recipient_email: str, role: CohortRole) -> None:
    """Owner-only: grant (or update) editor/viewer access for ``recipient_email``."""
    if role not in ("editor", "viewer"):
        raise ValueError("Granted role must be 'editor' or 'viewer'.")
    await _load_cohort_for_user(app_db, current_user, cohort_id, "owner")

    existing = (
        await app_db.scalars(select(DBCohortAccess).where(DBCohortAccess.cohort_id == cohort_id, DBCohortAccess.user_email == recipient_email))
    ).one_or_none()
    if existing is not None:
        existing.role = role
        existing.granted_by = current_user.email
    else:
        app_db.add(
            DBCohortAccess(
                id=uuid.uuid4(),
                cohort_id=cohort_id,
                user_email=recipient_email,
                role=role,
                granted_by=current_user.email,
            )
        )
    await app_db.commit()


async def revoke_cohort_access(app_db: AsyncSessionLocal, current_user: User, cohort_id: uuid.UUID, recipient_email: str) -> None:
    """Owner-only: remove an editor/viewer grant."""
    await _load_cohort_for_user(app_db, current_user, cohort_id, "owner")
    await app_db.execute(delete(DBCohortAccess).where(DBCohortAccess.cohort_id == cohort_id, DBCohortAccess.user_email == recipient_email))
    await app_db.commit()


async def _get_cohort_table_info(ascent_db: SnowflakeCursor, table_name: str) -> TableInfo:
    # ROW_COUNT and BYTES are Snowflake columns on information_schema.tables;
    # the standard view has neither. Count the rows instead, and report no
    # byte size rather than inventing one -- nothing downstream reads it
    # except the cohort listing, which shows it for information.
    # TableInfo.size_bytes is a required int, so report 0 rather than NULL.
    query_stats = rf"""SELECT COUNT(*) AS ROW_COUNT, 0 AS BYTES
                       FROM ASCENT.ASCENT_COHORTS.{table_name};"""
    query_columns = rf"""SELECT COLUMN_NAME, DATA_TYPE
                         FROM ASCENT.information_schema.columns
                         WHERE UPPER(TABLE_CATALOG) = 'ASCENT'
                           AND UPPER(TABLE_SCHEMA) = 'ASCENT_COHORTS'
                           AND UPPER(TABLE_NAME) = UPPER('{table_name}')
                         ORDER BY COLUMN_NAME;"""

    ((size, size_bytes),), _ = await execute_query(ascent_db, query_stats)
    columns, _ = await execute_query(ascent_db, query_columns)
    columns = [ColumnInfo(name=n, type=t) for (n, t) in columns]
    table_info = TableInfo(
        name=table_name,
        size=size,
        size_bytes=size_bytes,
        columns=columns,
    )

    return table_info


def default_cohort_name(database: str, criteria_include: list[str] | None = None) -> str:
    """A readable fallback name for cohorts created without one (deterministic, no LLM)."""
    today = datetime.now().strftime("%Y-%m-%d")
    if criteria_include and criteria_include[0].strip():
        return f"{criteria_include[0].strip()[:60]} — {today}"
    pretty = database.replace("_OMOP", "").replace("_", " ").title()
    return f"{pretty} cohort — {today}"


async def get_cohort_descriptor(app_db: AsyncSessionLocal, user: User, cohort_id: uuid.UUID) -> CohortDescriptor:
    db_descriptor, _ = await _load_cohort_for_user(app_db, user, cohort_id, "viewer")
    return CohortDescriptorAdapter.validate_python(db_descriptor)


async def _query_visible_descriptors(
    app_db: AsyncSessionLocal, current_user: User, restrict_origin: CohortOrigin | None, include_pending: bool
) -> List[DBCohortDescriptor]:
    filter_states = {"pending", "active"} if include_pending else {"active"}

    query = (
        select(DBCohortDescriptor)
        .options(selectinload(DBCohortDescriptor.funnel), selectinload(DBCohortDescriptor.access))
        .where(DBCohortDescriptor.state.in_(filter_states))
        .where(_cohort_visible_filter(current_user.email))
    )
    if restrict_origin:
        query = query.where(DBCohortDescriptor.origin == restrict_origin)

    return list((await app_db.scalars(query)).all())


async def get_cohort_descriptors(
    app_db: AsyncSessionLocal, current_user: User, restrict_origin: CohortOrigin | None = None, include_pending: bool = True
) -> List[CohortDescriptor]:
    db_descriptors = await _query_visible_descriptors(app_db, current_user, restrict_origin, include_pending)
    return [CohortDescriptorAdapter.validate_python(d) for d in db_descriptors]


async def get_cohort_descriptors_with_roles(
    app_db: AsyncSessionLocal, current_user: User, restrict_origin: CohortOrigin | None = None, include_pending: bool = True
) -> List[tuple[CohortDescriptor, CohortRole]]:
    """Like ``get_cohort_descriptors`` but pairs each cohort with the user's role."""
    db_descriptors = await _query_visible_descriptors(app_db, current_user, restrict_origin, include_pending)
    return [(CohortDescriptorAdapter.validate_python(d), resolve_cohort_role(d, current_user.email)) for d in db_descriptors]


async def update_cohort_descriptor(
    app_db: AsyncSessionLocal,
    current_user: User,
    cohort_id: uuid.UUID,
    patch: UserCohortDescriptorPatch,
) -> CohortDescriptor:
    logger.info(f"Update cohort {cohort_id}")

    db_descriptor, _ = await _load_cohort_for_user(app_db, current_user, cohort_id, "editor")

    patch.apply(db_descriptor)
    await app_db.commit()

    descriptor = CohortDescriptorAdapter.validate_python(db_descriptor)
    return descriptor


async def update_cohort_descriptor_funnel(
    app_db: AsyncSessionLocal,
    current_user: User,
    cohort_id: uuid.UUID,
    funnel: List[FunnelCriterion],
) -> CohortDescriptor:
    logger.info(f"Update cohort {cohort_id} funnel")

    # Authorize first (editor+) so we never mutate a cohort the user can't edit.
    db_descriptor, _ = await _load_cohort_for_user(app_db, current_user, cohort_id, "editor")

    # NOTE: Explicitly deleting the funnel should not be necessary as SQL Alchemy does it automatically.
    #   However, there seem to be cases where this does not work as expected.
    logger.debug("Delete existing funnel")
    query_funnel = delete(DBFunnelCriterion).where(DBFunnelCriterion.cohort_id == cohort_id)
    await app_db.execute(query_funnel)
    await app_db.commit()
    await app_db.refresh(db_descriptor)

    logger.debug("Update cohort funnel")
    db_descriptor.funnel = [DBFunnelCriterion(cohort_id=cohort_id, **fc.model_dump()) for fc in funnel]
    await app_db.commit()

    descriptor = CohortDescriptorAdapter.validate_python(db_descriptor)
    return descriptor


async def delete_cohort_descriptors(
    app_db: AsyncSessionLocal,
    current_user: User,
    cohort_ids: Set[uuid.UUID],
) -> Set[str]:
    # Only the creator (owner) may delete; authorize every id before deleting any.
    db_descriptors = [(await _load_cohort_for_user(app_db, current_user, cohort_id, "owner"))[0] for cohort_id in cohort_ids]

    for descriptor in db_descriptors:
        await app_db.delete(descriptor)
    await app_db.commit()

    return {d.table for d in db_descriptors}


async def get_studies(app_db: AsyncSessionLocal) -> List[StudyDescriptor]:
    query = select(DBStudyDescriptor).options(
        selectinload(DBStudyDescriptor.cohorts),
    )
    response = await app_db.scalars(query)
    db_descriptors = response.all()

    descriptors = list(map(StudyDescriptor.model_validate, db_descriptors))
    return descriptors


async def fetch_cohort_dataframe(rwd_db: SnowflakeCursor, query: str) -> pd.DataFrame:
    """Run the cohort-defining SELECT on the SSO RWD cursor and return rows.

    Result is fed into ``store_cohort`` on a machine-user ASCENT cursor so the
    RWD read and the ASCENT write run with the correct auth contexts."""
    rows, columns = await execute_query(rwd_db, query)
    return pd.DataFrame(rows, columns=columns)


async def store_cohort(
    cohort_id: uuid.UUID,
    ascent_db: SnowflakeCursor,
    cohort: pd.DataFrame,
    covariate_columns: List[str] | None = None,
) -> TableInfo:
    cohort_name = f"{OWNED_COHORT_TABLE_PREFIX}{cohort_id.hex.upper()}"
    base_columns = [c for c in cohort.columns if c in _NAME_LOOKUP]
    covariate_columns = covariate_columns or []

    accounted = set(base_columns) | set(covariate_columns)
    if columns_unexpected := {c for c in cohort.columns if c not in accounted}:
        logger.warning(f"Found unexpected columns in cohort dataframe: {columns_unexpected}")

    # The subject id's type comes from the data, not the lookup: OMOP
    # person_id is numeric, but a non-OMOP source can key on a string MRN.
    spec = [
        (
            c,
            _NAME_LOOKUP[c],
            _subject_id_type(cohort[c]) if _NAME_LOOKUP[c] == "SUBJECT_ID" else _TYPE_LOOKUP[c],
        )
        for c in base_columns
    ]
    base_sf_names = {sf_name for _, sf_name, _ in spec}
    spec += _build_covariate_spec(cohort, covariate_columns, reserved=base_sf_names)

    logger.info(f"Store cohort {cohort_id} with {cohort.shape[0]} records")
    await _write_cohort_table(ascent_db, cohort_name, cohort, spec)

    return await _get_cohort_table_info(ascent_db, cohort_name)


async def replace_cohort_columns(cohort_id: uuid.UUID, ascent_db: SnowflakeCursor, cohort: pd.DataFrame) -> TableInfo:
    """Rebuild an existing cohort table from a dataframe whose columns are already
    in final Snowflake names (``SUBJECT_ID`` / ``INDEX_DATE`` plus covariate columns).

    Used when extending a cohort with additional per-patient covariate columns.
    """
    cohort_name = f"{OWNED_COHORT_TABLE_PREFIX}{cohort_id.hex.upper()}"
    spec: List[tuple] = []
    for col in cohort.columns:
        upper = str(col).upper()
        if upper == "SUBJECT_ID":
            spec.append((col, "SUBJECT_ID", _subject_id_type(cohort[col])))
        elif upper == "INDEX_DATE":
            spec.append((col, "INDEX_DATE", "DATE"))
        else:
            spec.append((col, _sanitize_identifier(col), _infer_sf_type(cohort[col])))

    logger.info(f"Replace cohort {cohort_id} table with {len(spec)} columns, {cohort.shape[0]} records")

    # Stage, then swap. Writing in place is CREATE OR REPLACE followed by a
    # separate INSERT: when the insert fails the original table is already
    # gone, so the cohort ends up empty and wrongly typed while its descriptor
    # still advertises the old size. That turns a rejected covariate into
    # silent data loss, which is how a 90-patient cohort became 0 rows.
    staging = f"{cohort_name}_STAGING"
    await _write_cohort_table(ascent_db, staging, cohort, spec, create_or_replace=True)
    try:
        await execute_query(ascent_db, rf"""DROP TABLE IF EXISTS ASCENT_COHORTS.{cohort_name};""")
        await execute_query(ascent_db, rf"""ALTER TABLE ASCENT_COHORTS.{staging} RENAME TO {cohort_name};""")
    except Exception:
        await execute_query(ascent_db, rf"""DROP TABLE IF EXISTS ASCENT_COHORTS.{staging};""")
        raise

    return await _get_cohort_table_info(ascent_db, cohort_name)


def rewrite_cohort_table_ref(sql: str, cohort_table: str, replacement: str) -> str:
    """Point cohort references in ``sql`` at ``replacement`` (a session temp table).

    Matches ``ASCENT.ASCENT_COHORTS.<table>`` (the canonical form emitted via
    ``CohortMetadata.snowflake_table_ref``) with the ``ASCENT.`` qualifier
    optional, case-insensitively.
    """
    table = _sanitize_identifier(cohort_table)
    pattern = re.compile(rf"\b(?:ASCENT\s*\.\s*)?ASCENT_COHORTS\s*\.\s*{table}\b", re.IGNORECASE)
    return pattern.sub(replacement, sql)


def restore_cohort_table_ref(sql: str, cohort_table: str, temp_table: str) -> str:
    """Undo ``rewrite_cohort_table_ref`` so user-facing SQL shows the canonical
    ``ASCENT.ASCENT_COHORTS.<table>`` reference instead of a session temp name."""
    pattern = re.compile(rf"\b{re.escape(temp_table)}\b", re.IGNORECASE)
    return pattern.sub(f"ASCENT.ASCENT_COHORTS.{_sanitize_identifier(cohort_table)}", sql)


def healed_away_cohort_access(result) -> bool:
    """True when SQL self-healing "fixed" an ASCENT authorization failure.

    A healed query that could not read the cohort table did not answer the
    cohort-scoped question — its result must be discarded, not returned as if
    it were cohort data.
    """
    return bool(
        getattr(result, "self_healing_attempted", False)
        and getattr(result, "original_error_message", None)
        and ASCENT_NOT_AUTHORIZED_MARKER in result.original_error_message
    )


def _temp_table_spec(cohort_df: pd.DataFrame, descriptor: CohortDescriptorBase) -> List[tuple]:
    """Ordered ``(df_column, sf_name, sf_type)`` for every cohort attribute.

    Types come from the descriptor (mirrors ASCENT information_schema); names
    and types are re-validated before DDL interpolation.
    """
    by_lower = {str(c).lower(): c for c in cohort_df.columns}
    spec: List[tuple] = []
    for attribute in descriptor.attributes:
        df_col = by_lower.get(attribute.name.lower())
        if df_col is None:
            raise ValueError(f"Cohort table {descriptor.table!r} is missing expected column {attribute.name!r}")
        sf_type = str(attribute.type).strip().upper()
        if not _IDENTIFIER_RE.match(sf_type):
            raise ValueError(f"Unsupported column type {attribute.type!r} for cohort column {attribute.name!r}")
        spec.append((df_col, _sanitize_identifier(attribute.name), sf_type))
    return spec


@asynccontextmanager
async def cohort_temp_table(connection: SnowflakeConnection, descriptor: CohortDescriptorBase) -> AsyncIterator[str]:
    """Copy the persisted cohort into a session-scoped TEMPORARY table on ``connection``.

    Cohorts live in ``ASCENT.ASCENT_COHORTS`` where only the machine user has
    grants, while cohort-scoped question SQL runs on the user's SSO session
    against the RWD database — no single principal can read both. The machine
    user reads the cohort rows here and they are re-inserted as a TEMPORARY
    table in the caller's session, so generated SQL can join them without any
    ASCENT grants. The table is invisible to other sessions and dropped on
    exit (and at session end regardless), so nothing accumulates.

    Every statement using the yielded name must run on ``connection`` —
    temp tables do not exist outside their session.
    """
    source_table = _sanitize_identifier(descriptor.table)
    try:
        async with get_db("ASCENT") as ascent_db:
            cohort_df = await fetch_cohort_dataframe(ascent_db, rf"SELECT * FROM ASCENT_COHORTS.{source_table}")
    except WarehouseError as exc:
        if "does not exist" in str(exc):
            raise CohortTableMissingError(descriptor.id, descriptor.table) from exc
        raise

    spec = _temp_table_spec(cohort_df, descriptor)
    for df_col, _, sf_type in spec:
        if sf_type == "DATE":
            cohort_df[df_col] = _normalize_date_series(cohort_df[df_col])
    # Force plain Python values (None instead of NaN/NaT) for qmark binding.
    cohort_df = cohort_df.astype(object).where(pd.notna(cohort_df), None)

    df_columns = [df_col for df_col, _, _ in spec]
    column_types = ", ".join(f"{sf_name} {sf_type}" for _, sf_name, sf_type in spec)
    column_names = ", ".join(sf_name for _, sf_name, _ in spec)
    placeholders = ", ".join("?" for _ in spec)
    # Deterministic per cohort; OR REPLACE covers a lingering copy when the
    # session comes from the pooled machine-user connections.
    temp_table = f"{TEMP_COHORT_TABLE_PREFIX}{descriptor.id.hex.upper()}"

    cursor = connection.cursor()
    try:
        logger.info(f"Materialize cohort {descriptor.id} ({cohort_df.shape[0]} rows) as temp table {temp_table}")
        await execute_query(cursor, rf"CREATE OR REPLACE TEMPORARY TABLE {temp_table} ({column_types});")
        if not cohort_df.empty:
            await execute_query(
                cursor,
                rf"INSERT INTO {temp_table} ({column_names}) VALUES ({placeholders});",
                params=list(cohort_df[df_columns].itertuples(index=False)),
                bulk=True,
            )
        yield temp_table
    finally:
        try:
            await execute_query(cursor, rf"DROP TABLE IF EXISTS {temp_table};")
        except Exception:
            logger.warning(f"Failed to drop temp cohort table {temp_table}; session close will reclaim it")
        cursor.close()


async def store_cohort_descriptor(app_db: AsyncSessionLocal, descriptor: CohortDescriptor):
    data = descriptor.model_dump()
    data["funnel"] = [DBFunnelCriterion(**fc) for fc in data.pop("funnel", [])]
    db_descriptor = DBCohortDescriptor(**descriptor.model_dump())
    app_db.add(db_descriptor)
    await app_db.commit()


async def delete_abandoned_cohort_descriptors(app_db: AsyncSessionLocal):
    """
    Find cohort descriptors that have been in pending state for too long (>24h), i.e. not saved by their creator, and delete them.
    Corresponding cohort tables will be removed by the garbage collection.
    """
    logger.info("Delete abandoned cohort descriptors")

    query = delete(DBCohortDescriptor).where(
        DBCohortDescriptor.origin == "user",
        DBCohortDescriptor.state == "pending",
        DBCohortDescriptor.created < (datetime.now() - timedelta(days=1)),
    )
    result = await app_db.execute(query)
    await app_db.commit()

    logger.info(f"Deleted {result.rowcount} abandoned cohort descriptors")


async def flag_orphaned_cohort_descriptors(ascent_db: SnowflakeCursor, app_db: AsyncSessionLocal):
    """
    Find cohort descriptors that don't have a corresponding cohort table and flag them as orphaned.
    """
    logger.info("Flag orphaned cohort descriptors")

    # Query all user generate cohort tables
    user_cohort_tables_cursor, _ = await execute_query(ascent_db, QUERY_USER_COHORT_TABLES)
    user_cohort_tables = [t for (t,) in user_cohort_tables_cursor]

    # Fresh descriptors are skipped: their table may have been created after
    # the snapshot above was taken (see GC_MIN_COHORT_AGE).
    query = (
        update(DBCohortDescriptor)
        .where(
            DBCohortDescriptor.origin == "user",
            DBCohortDescriptor.table.notin_(user_cohort_tables),
            DBCohortDescriptor.created < (datetime.now() - GC_MIN_COHORT_AGE),
        )
        .values(state="orphaned")
    )
    result = await app_db.execute(query)
    await app_db.commit()

    logger.info(f"Flagged {result.rowcount} cohort descriptors as orphaned")


async def delete_orphaned_cohort_tables(ascent_db: SnowflakeCursor, app_db: AsyncSessionLocal):
    """
    Delete orphaned cohort tables.
    A cohort table is deleted when

    - it is owned by this instance of ascent, i.e. its name starts with the corresponding prefix (e.g. "PROD_USR_", "DEV_USR_" etc.)
    - no descriptor references the table
    """
    logger.info("Find and delete orphaned cohorts")

    # Query user generate cohort tables this instance of Ascent owns
    user_cohort_tables_owned_cursor, _ = await execute_query(ascent_db, QUERY_USER_COHORT_TABLES_OWNED)
    user_cohort_tables_owned = {t for (t,) in user_cohort_tables_owned_cursor}

    # Query user generated cohort descriptors
    query = select(DBCohortDescriptor.table).where(DBCohortDescriptor.origin == "user")
    cursor = await app_db.execute(query)
    user_cohort_tables_referenced = {table for (table,) in cursor}

    # Determine orphans
    orphaned_tables = {table for table in user_cohort_tables_owned if table not in user_cohort_tables_referenced}

    # Delete orphaned tables
    for table in orphaned_tables:
        logger.debug(f"Remove orphaned cohort table {table!r}")
        query_drop_table = rf"DROP TABLE ASCENT.ASCENT_COHORTS.{table};"
        await execute_query(ascent_db, query_drop_table)

    logger.info(f"Deleted {len(orphaned_tables)} orphaned cohort tables")
