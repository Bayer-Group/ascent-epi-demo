import logging
from datetime import UTC, datetime
from typing import Dict, List, Set, Tuple
from uuid import uuid4

from pydantic import UUID4, TypeAdapter
from snowflake.connector.cursor import SnowflakeCursor
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ascent_domain.config import get_domain_settings
from ascent_domain.errors import AccessDeniedError, InvalidRequestError, NotFoundError
from ascent_domain.external.concept_search import ConceptSearch
from ascent_domain.models.concept_list_dtos import (
    AnyConcepts,
    ConceptCodes,
    ConceptIDs,
    ConceptListCreateRequestDTO,
    ConceptListDTO,
    ConceptListMetaDTO,
    ConceptListPatchRequestDTO,
    ConceptListRevisionDTO,
    ConceptListRevisionRequestDTO,
    Concepts,
)
from ascent_domain.models.data_definitions import (
    CONCEPT_CATEGORIES,
    BulkConceptSearchRequest,
    ConceptCategory,
    DBConceptList,
    DBConceptListRevision,
    ExplorerConcepts,
    MedicalConcept,
    SearchType,
    StandardConcept,
    User,
)
from ascent_platform.config.runtime import get_runtime_settings
from ascent_platform.warehouse.session import execute_query

logger = logging.getLogger(__name__)
concept_search = ConceptSearch(get_domain_settings().AZURE_CLIENT_ID, get_domain_settings().AZURE_CLIENT_SECRET)

# RWD database used to look up reference OMOP concepts for the /validate
# endpoint. Read from settings rather than spelled as a literal in the two
# queries below.
#
# This is interpolated as a SQL identifier, not bound: it is deployment
# configuration, never caller input. The values that do come from callers are
# bound -- see _load_concept_from_codes.
_CONCEPT_REFERENCE_DB = get_runtime_settings().CONCEPT_LOOKUP_DATABASE


class _NotFoundError(NotFoundError):
    def __init__(self, *ids: UUID4) -> None:
        super().__init__(
            detail={
                "error": "Not found",
                "ids": list(map(str, ids)),
            },
        )


class ConceptListNotFound(_NotFoundError): ...


class ConceptListRevisionNotFound(_NotFoundError): ...


class _ExistsError(AccessDeniedError):
    def __init__(self, message: str) -> None:
        super().__init__(
            detail={
                "error": "Item already exists",
                "message": message,
            },
        )


class ConceptListExists(_ExistsError): ...


class ConceptListRevisionExists(_ExistsError): ...


class _IsUsedError(AccessDeniedError):
    def __init__(self, message: str) -> None:
        super().__init__(
            detail={
                "error": "Item is referenced by another object",
                "message": message,
            },
        )


class ConceptListIsUsed(_IsUsedError): ...


class ConceptListRevisionIsUsed(_IsUsedError): ...


class _UnknownError(InvalidRequestError):
    def __init__(self, message: str) -> None:
        super().__init__(
            detail={
                "error": "Item already exists",
                "message": message,
            },
        )


class UnknownConceptIDs(InvalidRequestError):
    def __init__(self, concept_ids: Set[int]) -> None:
        super().__init__(
            detail={
                "error": "Unknown concept IDs",
                "concept_ids": sorted(concept_ids),
            },
        )


class UnknownConceptCodes(InvalidRequestError):
    def __init__(self, concept_codes: Set[str], vocabulary_id: str) -> None:
        super().__init__(
            detail={"error": "Unknown concept codes", "concept_codes": sorted(concept_codes), "vocabulary_id": vocabulary_id},
        )


StandardConceptList = TypeAdapter(List[StandardConcept])


async def create_concept_list(request: ConceptListCreateRequestDTO, user: User, app_db: AsyncSession) -> ConceptListDTO:
    """Creates a concept list in the application DB"""
    db_concept_list_revisions = []
    if revision := request.revision:
        db_concept_list_revisions.append(
            DBConceptListRevision(
                id=uuid4(),
                name=revision.name,
                created=datetime.now(UTC),
                database=revision.concepts.database or "",
                description=request.description,
                concepts=[concepts.model_dump(mode="json") for concepts in revision.concepts.concepts],
            )
        )

    db_concept_list = DBConceptList(
        id=uuid4(),
        name=request.name,
        category=request.category,
        description=request.description,
        owner_id=user.email,
        revisions=db_concept_list_revisions,
    )

    logger.info(f"Storing cohort list to app DB: '{db_concept_list}'")
    try:
        app_db.add(db_concept_list)
        await app_db.commit()
    except IntegrityError:
        await app_db.rollback()
        raise ConceptListExists(request.name)

    return ConceptListDTO.model_validate(db_concept_list)


async def retrieve_concept_lists(
    user: User, app_db: AsyncSession, name: str | None = None, category: ConceptCategory | None = None
) -> list[ConceptListMetaDTO]:
    """Retrieves all concept list for a user from the application DB"""
    query = select(DBConceptList).where(DBConceptList.owner_id == user.email)

    if name:
        logger.debug(f"Filter for names that contain {name} (case insensitive)")
        query = query.where(DBConceptList.name.ilike(f"%{name}%"))

    if category:
        logger.debug(f"Filter for category {category}")
        query = query.where(DBConceptList.category == category)

    logger.info("Retrieving all concept lists that match filter criteria")
    response = await app_db.scalars(query)
    db_concept_lists = response.all()

    concept_lists = list(map(ConceptListMetaDTO.model_validate, db_concept_lists))
    return concept_lists


async def retrieve_concept_list(concept_list_id: UUID4, user: User, app_db: AsyncSession) -> ConceptListDTO:
    """Retrieves a specific concept list belonging to a user from the application DB"""
    query = (
        select(DBConceptList)
        .options(selectinload(DBConceptList.revisions))
        .where(DBConceptList.id == concept_list_id, DBConceptList.owner_id == user.email)
    )
    logger.info("Retrieving concept list from DB")
    response = await app_db.scalars(query)
    try:
        db_concept_list = response.one()
    except NoResultFound:
        raise ConceptListNotFound(f"Unknown Concept list ID: '{concept_list_id}'")

    concept_list = ConceptListDTO.model_validate(db_concept_list)
    return concept_list


async def update_concept_list(concept_list_id: UUID4, patch: ConceptListPatchRequestDTO, user: User, app_db: AsyncSession) -> ConceptListMetaDTO:
    query = select(DBConceptList).where(DBConceptList.id == concept_list_id, DBConceptList.owner_id == user.email)
    logger.info("Retrieving concept list from DB")
    response = await app_db.scalars(query)
    try:
        db_concept_list = response.one()
    except NoResultFound:
        raise ConceptListNotFound(f"Unknown Concept list ID: '{concept_list_id}'")

    logger.info(f"Applying patch: '{patch}'")
    patch.apply(db_concept_list)
    await app_db.commit()

    concept_list = ConceptListMetaDTO.model_validate(db_concept_list)
    return concept_list


async def delete_concept_lists(concept_list_ids: Set[UUID4], user: User, app_db: AsyncSession) -> None:
    query = select(DBConceptList).where(DBConceptList.id.in_(concept_list_ids), DBConceptList.owner_id == user.email)

    logger.info("Retrieving concept lists from DB")
    response = await app_db.scalars(query)
    db_concept_lists = response.all()

    if ids_not_found := concept_list_ids - {d.id for d in db_concept_lists}:
        raise ConceptListNotFound(*ids_not_found)

    logger.info(f"Deleting concept lists {concept_list_ids}")
    try:
        for db_concept_list in db_concept_lists:
            await app_db.delete(db_concept_list)
        await app_db.commit()
    except IntegrityError:
        await app_db.rollback()
        raise ConceptListIsUsed("Cannot delete concept list/s because of a reference from another object")


async def _load_concepts_from_ids(rwd_db: SnowflakeCursor, concept_ids: Set[int]) -> List[StandardConcept]:  # noqa: F821
    if not concept_ids:
        return []

    # Bound, not interpolated: these ids arrive on a request payload, and the
    # caller names them *_unverified. The early return above is what keeps an
    # empty set from rendering ``IN ()``, invalid in either dialect.
    ordered_ids = sorted(concept_ids)
    placeholders = ", ".join("?" for _ in ordered_ids)
    query = rf"""
    SELECT CONCEPT_ID, CONCEPT_NAME, CONCEPT_CODE, VOCABULARY_ID
    FROM {_CONCEPT_REFERENCE_DB}.concept
    WHERE CONCEPT_ID IN ({placeholders});
    """

    records, _ = await execute_query(rwd_db, query, params=list(ordered_ids))
    concepts = [StandardConcept(concept_id=cid, concept_name=name, concept_code=code, vocabulary_id=vid) for cid, name, code, vid in records]

    if unknown_concept_ids := concept_ids - {c.concept_id for c in concepts}:
        raise UnknownConceptIDs(unknown_concept_ids)

    return concepts


async def _load_concepts_from_ids_medical_coder(database: str, concept_ids: Set[int]) -> List[StandardConcept]:
    request = BulkConceptSearchRequest(query=",".join(map(str, sorted(concept_ids))), database=database, search_type=SearchType.CONCEPT_ID)
    response = await concept_search.bulk_concept_search(request)
    concepts = StandardConceptList.validate_python(response)

    if unknown_concept_ids := concept_ids - {c.concept_id for c in concepts}:
        raise UnknownConceptIDs(unknown_concept_ids)

    return concepts


async def _load_concept_from_codes(rwd_db: SnowflakeCursor, concept_codes: Set[str], vocabulary_id: str) -> List[StandardConcept]:
    if not concept_codes:
        return []

    # Bound, not interpolated. Both the codes and the vocabulary id come
    # straight off a request payload, and quoting them into the statement made
    # any value containing a quote a way to rewrite the query. An empty set also
    # rendered ``IN ()``, which is not valid SQL in either dialect.
    ordered_codes = sorted(concept_codes)
    placeholders = ", ".join("?" for _ in ordered_codes)
    query = rf"""
    SELECT CONCEPT_ID, CONCEPT_NAME, CONCEPT_CODE, VOCABULARY_ID
    FROM {_CONCEPT_REFERENCE_DB}.concept
    WHERE CONCEPT_CODE IN ({placeholders})
      AND VOCABULARY_ID = ?;
    """

    records, _ = await execute_query(rwd_db, query, params=[*ordered_codes, vocabulary_id])
    concepts = [StandardConcept(concept_id=cid, concept_name=name, concept_code=code, vocabulary_id=vid) for cid, name, code, vid in records]

    if unknown_concept_codes := concept_codes - {c.concept_code for c in concepts}:
        raise UnknownConceptCodes(unknown_concept_codes, vocabulary_id)

    return concepts


async def _load_concepts_from_codes_medical_coder(database: str, concept_codes: Set[str], vocabulary_id: str) -> List[StandardConcept]:
    query = ",".join(map(str, sorted(concept_codes)))
    request = BulkConceptSearchRequest(query=query, database=database, search_type=SearchType.CONCEPT_CODE)
    response = await concept_search.bulk_concept_search(request)
    concepts = StandardConceptList.validate_python(response)

    if unknown_concept_codes := concept_codes - {c.concept_code for c in concepts}:
        raise UnknownConceptCodes(unknown_concept_codes, vocabulary_id)

    return concepts


async def _load_concepts_from_concepts(rwd_db: SnowflakeCursor, concepts: List[StandardConcept]) -> List[StandardConcept]:
    concepts_ref = await _load_concepts_from_ids(rwd_db, {c.concept_id for c in concepts})
    concepts_ref_lookup = {c.concept_id: c for c in concepts_ref}

    excl = {"patient_count", "similarity_score"}
    for concept in concepts:
        concept_ref = concepts_ref_lookup[concept.concept_id]
        if concept.model_dump(exclude=excl) != concept_ref.model_dump(exclude=excl):
            logger.warning(f"Concept {concept!r} differs from reference concept {concept_ref!r}")

    return concepts


async def validate_concepts(concepts: AnyConcepts, rwd_db: SnowflakeCursor) -> Tuple[str | None, List[StandardConcept]]:
    match concepts:
        case Concepts(database=database, concepts=concepts_unverified):
            logger.info("Use user provided concepts")
            validated_concepts = await _load_concepts_from_concepts(rwd_db, concepts_unverified)
            return database, validated_concepts

        case ConceptIDs(database=None, concept_ids=concept_ids_unverified):
            logger.info("Derive concepts from given IDs")
            validate_concepts = await _load_concepts_from_ids(rwd_db, set(concept_ids_unverified))
            return None, validate_concepts

        case ConceptIDs(database=database, concept_ids=concept_ids_unverified):
            logger.info("Derive concepts from given IDs using Medical Coder")
            validate_concepts = await _load_concepts_from_ids_medical_coder(database or "", set(concept_ids_unverified))
            return database, validate_concepts

        case ConceptCodes(database=None, concept_codes=concept_codes_unverified, vocabulary_id=vocabulary_id):
            logger.info("Derive concepts from given concept codes and vocabulary")
            validate_concepts = await _load_concept_from_codes(rwd_db, set(concept_codes_unverified), vocabulary_id)
            return None, validate_concepts

        case ConceptCodes(database=database, concept_codes=concept_codes_unverified, vocabulary_id=vocabulary_id):
            logger.info("Derive concepts from given concept codes and vocabulary using Medical Coder")
            validate_concepts = await _load_concepts_from_codes_medical_coder(database or "", set(concept_codes_unverified), vocabulary_id)
            return database, validate_concepts

        case _:
            raise RuntimeError("Unreachable")


async def create_concept_list_revision(
    concept_list_id: UUID4, request: ConceptListRevisionRequestDTO, app_db: AsyncSession
) -> ConceptListRevisionDTO:
    db_concept_list_revision = DBConceptListRevision(
        id=uuid4(),
        concept_list_id=concept_list_id,
        name=request.name,
        created=datetime.now(UTC),
        database=request.concepts.database,
        description=request.description,
        concepts=[c.model_dump(mode="json") for c in request.concepts.concepts],
    )

    logger.info("Write concept list revision to DB")
    try:
        app_db.add(db_concept_list_revision)
        await app_db.commit()
    except IntegrityError:
        await app_db.rollback()
        raise ConceptListRevisionExists(request.name)

    return ConceptListRevisionDTO.model_validate(db_concept_list_revision)


async def retrieve_concept_list_revision(
    concept_list_id: UUID4, concept_list_revision_id: UUID4, user: User, app_db: AsyncSession
) -> ConceptListRevisionDTO:
    """Retrieves a specific concept list revision belonging to a user from the application DB"""
    query = (
        select(DBConceptListRevision)
        .join(DBConceptList, DBConceptListRevision.concept_list_id == DBConceptList.id)
        .where(
            DBConceptListRevision.id == concept_list_revision_id,
            DBConceptList.id == concept_list_id,
            DBConceptList.owner_id == user.email,
        )
    )

    logger.info("Retrieving concept list revision from DB")
    response = await app_db.scalars(query)
    try:
        db_concept_list = response.one()
    except NoResultFound:
        raise ConceptListRevisionNotFound(f"Unknown Concept List Revision: '{concept_list_revision_id}'")

    concept_list_revision = ConceptListRevisionDTO.model_validate(db_concept_list)
    return concept_list_revision


async def delete_concept_list_revisions(concept_list_id: UUID4, concept_list_revision_ids: Set[UUID4], user: User, app_db: AsyncSession):
    query = (
        select(DBConceptListRevision)
        .join(DBConceptList, DBConceptListRevision.concept_list_id == DBConceptList.id)
        .where(
            DBConceptListRevision.id.in_(concept_list_revision_ids),
            DBConceptList.id == concept_list_id,
            DBConceptList.owner_id == user.email,
        )
    )

    logger.info("Retrieving concept list revisions from DB")
    response = await app_db.scalars(query)
    db_concept_list_revisions = response.all()

    if ids_not_found := concept_list_revision_ids - {r.id for r in db_concept_list_revisions}:
        raise ConceptListRevisionNotFound(*ids_not_found)

    logger.info(f"Deleting concept list revisions {concept_list_revision_ids}")
    try:
        for db_concept_list_revision in db_concept_list_revisions:
            await app_db.delete(db_concept_list_revision)
        await app_db.commit()
    except IntegrityError:
        await app_db.rollback()
        raise ConceptListRevisionIsUsed("Cannot delete concept list revisions because of references from another object")


async def _resolve_user_concept(concept_list_revision_id: UUID4, user: User, app_db: AsyncSession) -> MedicalConcept:
    query = (
        select(DBConceptList.name, DBConceptList.category, DBConceptListRevision.concepts)
        .join(DBConceptList, DBConceptListRevision.concept_list_id == DBConceptList.id)
        .where(
            DBConceptListRevision.id == concept_list_revision_id,
            DBConceptList.owner_id == user.email,
        )
    )

    logger.info("Retrieving concept list revision from DB")
    response = await app_db.execute(query)
    try:
        cl_name, cl_category, cl_rev_concepts = response.one()
    except NoResultFound:
        raise ConceptListRevisionNotFound(f"Unknown Concept List Revision: '{concept_list_revision_id}'")

    concept = MedicalConcept(
        name=cl_name,
        category=cl_category,
        concepts=cl_rev_concepts,
    )

    return concept


async def resolve_user_concepts(user_concepts: List[MedicalConcept], user: User, app_db: AsyncSession) -> List[MedicalConcept]:
    """
    Take a potentially mixed list of medical concepts and concept list revision IDs and turn it into a list of medical concepts.
    Medical concepts are just passed as they are while IDs are used to find the corresponding concept list revision in the database.
    """
    return [mc if isinstance(mc, MedicalConcept) else await _resolve_user_concept(mc, user, app_db) for mc in user_concepts]


def user_concepts_to_explorer_concepts(user_concepts: List[MedicalConcept]) -> ExplorerConcepts:
    """
    Convert a list of medical concepts into the explorer concept format.
    """
    return {mc.name: ",".join(str(c.concept_id) for c in sorted(mc.concepts, key=lambda c: c.concept_id)) for mc in user_concepts}


async def resolve_user_concepts_to_explorer_concepts(
    user_concepts: List[MedicalConcept | UUID4], user: User, app_db: AsyncSession
) -> ExplorerConcepts:
    """Take a potentially mixed list of medical concepts and concept list revision IDs and turn it into an explorer concept dictionary."""
    resolved_concepts = await resolve_user_concepts(user_concepts, user, app_db)
    explorer_concepts = user_concepts_to_explorer_concepts(resolved_concepts)
    return explorer_concepts


def transform_to_custom_code_mappings(concepts: List[MedicalConcept]) -> Dict[Tuple[str, str], str]:
    custom_code_mappings = {}
    for concept in concepts:
        category_name_tuple = (
            (concept.category.lower(), concept.name.lower()) if concept.category in CONCEPT_CATEGORIES else ("drug_class", concept.category.lower())
        )
        concept_ids = ",".join([str(standard_concept.concept_id) for standard_concept in concept.concepts])
        if category_name_tuple in custom_code_mappings:
            custom_code_mappings[category_name_tuple] += f",{concept_ids}"
        else:
            custom_code_mappings[category_name_tuple] = concept_ids

    return custom_code_mappings
