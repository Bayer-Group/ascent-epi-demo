from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, NoResultFound

from ascent_domain import concept_lists as service
from ascent_domain.concept_lists import StandardConceptList
from ascent_domain.models.concept_list_dtos import (
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
    BulkConceptSearchRequest,
    DBConceptList,
    DBConceptListRevision,
    MedicalConcept,
    SearchType,
    StandardConcept,
)
from tests.ascent.utils.concept_list_mother import DEFAULT_CONCEPTS


def assert_query_equal(expected_query, query):
    assert str(expected_query.compile()) == str(query.compile())


@pytest.fixture
def mock_concepts_json():
    return StandardConceptList.dump_python(DEFAULT_CONCEPTS, mode="json")


@pytest.fixture
def mock_cl_id() -> UUID:
    return uuid4()


@pytest.fixture
def mock_cl_revision_id() -> UUID:
    return uuid4()


@pytest.fixture
def mock_cl_revision(mock_cl_id, mock_cl_revision_id, mock_concepts_json) -> DBConceptListRevision:
    return DBConceptListRevision(
        id=mock_cl_revision_id,
        concept_list_id=mock_cl_id,
        name="ConceptRevision Name",
        created=datetime.now(UTC),
        database="SYNTHETIC_EHR_OMOP",
        concepts=mock_concepts_json,
    )


@pytest.fixture
def mock_cl_revision_dto(mock_cl_revision) -> ConceptListRevisionDTO:
    return ConceptListRevisionDTO.model_validate(mock_cl_revision)


@pytest.fixture
def mock_cl(mock_cl_id, mock_cl_revision) -> DBConceptList:
    concept_list = DBConceptList(
        id=mock_cl_id,
        name="Concept Name",
        category="condition",
    )
    concept_list.revisions = [mock_cl_revision]
    return concept_list


@pytest.fixture
def mock_cl_meta_dto(mock_cl) -> ConceptListMetaDTO:
    return ConceptListMetaDTO.model_validate(mock_cl)


@pytest.fixture
def mock_cl_dto(mock_cl) -> ConceptListDTO:
    return ConceptListDTO.model_validate(mock_cl)


@pytest.fixture
def mock_cl_medical_concept(mock_cl, mock_cl_revision) -> MedicalConcept:
    return MedicalConcept(
        name=mock_cl.name,
        category=mock_cl.category,
        concepts=mock_cl_revision.concepts,
    )


async def test_create_concept_list(mock_app_db, mock_user, mock_cl_dto):
    # arrange
    request = ConceptListCreateRequestDTO(name=mock_cl_dto.name, category=mock_cl_dto.category)

    # act
    concept_list = await service.create_concept_list(request, mock_user, mock_app_db)

    # assert
    assert concept_list.id is not None
    assert concept_list.name == request.name
    assert concept_list.category == request.category


@patch("ascent_domain.concept_lists.validate_concepts", new_callable=AsyncMock)
async def test_create_concept_list_with_revision(mock_validate_concepts, mock_app_db, mock_user, mock_cl, mock_cl_revision, mock_cl_dto):
    # arrange
    concepts = Concepts(database=mock_cl_revision.database, concepts=mock_cl_revision.concepts)
    revision_request = ConceptListRevisionRequestDTO(name=mock_cl_revision.name, description=mock_cl_revision.description, concepts=concepts)
    request = ConceptListCreateRequestDTO(name=mock_cl_dto.name, category=mock_cl_dto.category, revision=revision_request)

    (concept_list_revision_ref,) = mock_cl_dto.revisions
    mock_validate_concepts.return_value = (concepts.database, concepts.concepts)

    # act
    concept_list = await service.create_concept_list(request, mock_user, mock_app_db)
    (concept_list_revision,) = concept_list.revisions

    # assert
    assert concept_list.id is not None
    assert concept_list.name == request.name
    assert concept_list.category == request.category

    assert concept_list_revision.name == concept_list_revision_ref.name
    assert concept_list_revision.database == concept_list_revision_ref.database
    assert concept_list_revision.description == concept_list_revision_ref.description
    assert concept_list_revision.concepts == concept_list_revision_ref.concepts


async def test_create_concept_list_exists(mock_app_db, mock_user, mock_cl, mock_cl_dto):
    # arrange
    request = ConceptListCreateRequestDTO(name=mock_cl_dto.name, category=mock_cl_dto.category)
    mock_app_db.add.side_effect = IntegrityError(None, None, Exception())

    # act
    with pytest.raises(service.ConceptListExists):
        await service.create_concept_list(request, mock_user, mock_app_db)


async def test_retrieve_concept_lists(mock_app_db, mock_user, mock_cl, mock_cl_meta_dto):
    # arrange
    expected_query = select(DBConceptList).where(DBConceptList.owner_id == mock_user.email)
    mock_app_db.scalars.return_value.all.return_value = [mock_cl]

    # act
    all_concept_lists = await service.retrieve_concept_lists(mock_user, mock_app_db)
    actual_query = mock_app_db.scalars.call_args[0][0]

    # assert
    assert all_concept_lists == [mock_cl_meta_dto]
    assert_query_equal(expected_query, actual_query)


async def test_retrieve_concept_lists_filter_name(mock_app_db, mock_user):
    # arrange
    expected_query = select(DBConceptList).where(DBConceptList.owner_id == mock_user.email).where(DBConceptList.name.ilike("%test%"))
    mock_app_db.scalars.return_value.all.return_value = []

    # act
    all_concept_lists = await service.retrieve_concept_lists(mock_user, mock_app_db, name="test")
    actual_query = mock_app_db.scalars.call_args[0][0]

    # assert
    assert all_concept_lists == []
    assert_query_equal(expected_query, actual_query)


async def test_retrieve_concept_lists_filter_category(mock_app_db, mock_user):
    # arrange
    expected_query = select(DBConceptList).where(DBConceptList.owner_id == mock_user.email).where(DBConceptList.category == "drug")
    mock_app_db.scalars.return_value.all.return_value = []

    # act
    all_concept_lists = await service.retrieve_concept_lists(mock_user, mock_app_db, category="drug")
    actual_query = mock_app_db.scalars.call_args[0][0]

    # assert
    assert all_concept_lists == []
    assert_query_equal(expected_query, actual_query)


async def test_retrieve_single_concept_list(mock_app_db, mock_user, mock_cl, mock_cl_dto):
    # arrange
    mock_app_db.scalars.return_value.one.return_value = mock_cl

    # act
    concept_list = await service.retrieve_concept_list(mock_cl.id, mock_user, mock_app_db)

    # assert
    assert concept_list == mock_cl_dto


async def test_retrieve_single_concept_list_with_invalid_id(mock_app_db, mock_user, mock_cl_id):
    # arrange
    mock_app_db.scalars.return_value.one.side_effect = NoResultFound

    # act
    with pytest.raises(service.ConceptListNotFound):
        await service.retrieve_concept_list(mock_cl_id, mock_user, mock_app_db)


async def test_patch_concept_list(mock_app_db, mock_user, mock_cl_id, mock_cl, mock_cl_meta_dto, mock_cl_dto):
    # arrange
    request = ConceptListPatchRequestDTO(description="New description")
    _concept_list = mock_cl_meta_dto.model_copy()
    _concept_list.description = "New description"
    mock_app_db.scalars.return_value.one.return_value = mock_cl

    # act
    concept_list = await service.update_concept_list(mock_cl_id, request, mock_user, mock_app_db)

    # assert
    assert concept_list == _concept_list


async def test_patch_concept_list_with_invalid_id(mock_app_db, mock_user, mock_cl_id):
    # arrange
    mock_app_db.scalars.return_value.one.side_effect = NoResultFound

    # act
    with pytest.raises(service.ConceptListNotFound):
        await service.update_concept_list(mock_cl_id, MagicMock(), mock_user, mock_app_db)


async def test_delete_concept_list(mock_app_db, mock_user, mock_cl_id, mock_cl):
    # arrange
    mock_app_db.scalars.return_value.all.return_value = [mock_cl]

    # act
    await service.delete_concept_lists({mock_cl_id}, mock_user, mock_app_db)

    # assert
    mock_app_db.delete.assert_called_once_with(mock_cl)


async def test_delete_single_concept_list_with_invalid_id(mock_app_db, mock_user, mock_cl_id):
    # arrange
    mock_app_db.scalars.return_value.one.side_effect = NoResultFound

    # act
    with pytest.raises(service.ConceptListNotFound):
        await service.delete_concept_lists({mock_cl_id}, mock_user, mock_app_db)


async def test_delete_single_concept_list_referenced(mock_app_db, mock_user, mock_cl_id, mock_cl):
    # arrange
    mock_app_db.scalars.return_value.all.return_value = [mock_cl]
    mock_app_db.delete.side_effect = IntegrityError(None, None, Exception())

    # act
    with pytest.raises(service.ConceptListIsUsed):
        await service.delete_concept_lists({mock_cl_id}, mock_user, mock_app_db)


@patch("ascent_domain.concept_lists._load_concepts_from_ids", new_callable=AsyncMock)
async def test_load_concepts_from_concepts(mock_load_concepts_from_ids, mock_rwd_db, mock_cl_revision):
    # arrange
    concept_ids = {c["concept_id"] for c in mock_cl_revision.concepts}
    expected_concepts = list(map(StandardConcept.model_validate, mock_cl_revision.concepts))
    mock_load_concepts_from_ids.return_value = expected_concepts

    # act
    concepts = await service._load_concepts_from_concepts(mock_rwd_db, expected_concepts)

    # assert
    mock_load_concepts_from_ids.assert_called_once_with(mock_rwd_db, concept_ids)
    assert concepts == expected_concepts


@patch("ascent_domain.concept_lists.execute_query", new_callable=AsyncMock)
async def test_load_concepts_from_ids(mock_execute_query, mock_rwd_db, mock_cl_revision):
    # arrange
    concept_ids = {c["concept_id"] for c in mock_cl_revision.concepts}
    ordered_ids = sorted(concept_ids)
    placeholders = ", ".join("?" for _ in ordered_ids)
    records = [(c["concept_id"], c["concept_name"], c["concept_code"], c["vocabulary_id"]) for c in mock_cl_revision.concepts]
    mock_execute_query.return_value = (records, None)
    expected_concepts = [StandardConcept(concept_id=cid, concept_name=name, concept_code=code, vocabulary_id=vid) for cid, name, code, vid in records]

    # act
    concepts = await service._load_concepts_from_ids(mock_rwd_db, concept_ids)

    # assert
    mock_execute_query.assert_called_once_with(
        mock_rwd_db,
        rf"""
    SELECT CONCEPT_ID, CONCEPT_NAME, CONCEPT_CODE, VOCABULARY_ID
    FROM SYNTHETIC_EHR_OMOP.concept
    WHERE CONCEPT_ID IN ({placeholders});
    """,
        params=ordered_ids,
    )
    assert concepts == expected_concepts


@patch("ascent_domain.concept_lists.execute_query", new_callable=AsyncMock)
async def test_load_concepts_from_ids_unknown(mock_execute_query, mock_rwd_db, mock_cl_revision):
    # arrange
    concept_ids = {c["concept_id"] for c in mock_cl_revision.concepts}
    records = [(c["concept_id"], c["concept_name"], c["concept_code"], c["vocabulary_id"]) for c in mock_cl_revision.concepts]
    mock_execute_query.return_value = (records[:1], None)

    # act
    with pytest.raises(service.UnknownConceptIDs):
        await service._load_concepts_from_ids(mock_rwd_db, concept_ids)


@patch("ascent_domain.concept_lists.concept_search.bulk_concept_search", new_callable=AsyncMock)
async def test_load_concepts_from_ids_medical_coder(mock_bulk_concept_search, mock_rwd_db, mock_cl_revision):
    # arrange
    concept_ids = {c["concept_id"] for c in mock_cl_revision.concepts}
    concept_ids_str = ",".join(map(str, sorted(concept_ids)))
    mock_bulk_concept_search.return_value = mock_cl_revision.concepts
    expected_request = BulkConceptSearchRequest(query=concept_ids_str, database=mock_cl_revision.database, search_type=SearchType.CONCEPT_ID)
    expected_concepts = list(map(StandardConcept.model_validate, mock_cl_revision.concepts))

    # act
    concepts = await service._load_concepts_from_ids_medical_coder(mock_cl_revision.database, concept_ids)

    # assert
    mock_bulk_concept_search.assert_called_once_with(expected_request)
    assert concepts == expected_concepts


@patch("ascent_domain.concept_lists.concept_search.bulk_concept_search", new_callable=AsyncMock)
async def test_load_concepts_from_ids_medical_coder_unknown(mock_bulk_concept_search, mock_rwd_db, mock_cl_revision):
    # arrange
    concept_ids = {c["concept_id"] for c in mock_cl_revision.concepts}
    mock_bulk_concept_search.return_value = mock_cl_revision.concepts[:1]

    # act
    with pytest.raises(service.UnknownConceptIDs):
        await service._load_concepts_from_ids_medical_coder(mock_cl_revision.database, concept_ids)


@patch("ascent_domain.concept_lists.execute_query", new_callable=AsyncMock)
async def test_load_concepts_from_codes(mock_execute_query, mock_rwd_db, mock_cl_revision):
    # arrange
    concept_codes = {c["concept_code"] for c in mock_cl_revision.concepts}
    ordered_codes = sorted(concept_codes)
    placeholders = ", ".join("?" for _ in ordered_codes)
    records = [(c["concept_id"], c["concept_name"], c["concept_code"], c["vocabulary_id"]) for c in mock_cl_revision.concepts]
    mock_execute_query.return_value = (records, None)
    expected_concepts = [StandardConcept(concept_id=cid, concept_name=name, concept_code=code, vocabulary_id=vid) for cid, name, code, vid in records]

    # act
    concepts = await service._load_concept_from_codes(mock_rwd_db, concept_codes, vocabulary_id="SNOMED")

    # assert
    mock_execute_query.assert_called_once_with(
        mock_rwd_db,
        rf"""
    SELECT CONCEPT_ID, CONCEPT_NAME, CONCEPT_CODE, VOCABULARY_ID
    FROM SYNTHETIC_EHR_OMOP.concept
    WHERE CONCEPT_CODE IN ({placeholders})
      AND VOCABULARY_ID = ?;
    """,
        params=[*ordered_codes, "SNOMED"],
    )
    assert concepts == expected_concepts


@patch("ascent_domain.concept_lists.execute_query", new_callable=AsyncMock)
async def test_load_concepts_from_codes_unknown(mock_execute_query, mock_rwd_db, mock_cl_revision):
    # arrange
    concept_codes = {c["concept_code"] for c in mock_cl_revision.concepts}
    records = [(c["concept_id"], c["concept_name"], c["concept_code"], c["vocabulary_id"]) for c in mock_cl_revision.concepts]
    mock_execute_query.return_value = (records[:1], None)

    # act
    with pytest.raises(service.UnknownConceptCodes):
        await service._load_concept_from_codes(mock_rwd_db, concept_codes, vocabulary_id="SNOMED")


@patch("ascent_domain.concept_lists.concept_search.bulk_concept_search", new_callable=AsyncMock)
async def test_load_concepts_from_codes_medical_coder(mock_bulk_concept_search, mock_rwd_db, mock_cl_revision):
    # arrange
    concept_codes = {c["concept_code"] for c in mock_cl_revision.concepts}
    concept_codes_str = ",".join(map(str, sorted(concept_codes)))
    mock_bulk_concept_search.return_value = mock_cl_revision.concepts
    expected_request = BulkConceptSearchRequest(query=concept_codes_str, database=mock_cl_revision.database, search_type=SearchType.CONCEPT_CODE)
    expected_concepts = list(map(StandardConcept.model_validate, mock_cl_revision.concepts))

    # act
    concepts = await service._load_concepts_from_codes_medical_coder(mock_cl_revision.database, concept_codes, vocabulary_id="SNOMED")

    # assert
    mock_bulk_concept_search.assert_called_once_with(expected_request)
    assert concepts == expected_concepts


@patch("ascent_domain.concept_lists.concept_search.bulk_concept_search", new_callable=AsyncMock)
async def test_load_concepts_from_codes_medical_coder_unknown(mock_bulk_concept_search, mock_rwd_db, mock_cl_revision):
    # arrange
    concept_codes = {c["concept_code"] for c in mock_cl_revision.concepts}
    mock_bulk_concept_search.return_value = mock_cl_revision.concepts[:1]

    # act
    with pytest.raises(service.UnknownConceptCodes):
        await service._load_concepts_from_codes_medical_coder(mock_cl_revision.database, concept_codes, vocabulary_id="SNOMED")


@patch("ascent_domain.concept_lists._load_concepts_from_concepts", new_callable=AsyncMock)
async def test_validate_concepts(mock_load_concepts_from_concepts, mock_rwd_db, mock_cl_revision):
    # arrange
    request = Concepts(database=mock_cl_revision.database, concepts=mock_cl_revision.concepts)
    mock_load_concepts_from_concepts.return_value = mock_cl_revision.concepts

    # act
    database, concepts = await service.validate_concepts(request, mock_rwd_db)

    # assert
    mock_load_concepts_from_concepts.assert_called_once_with(mock_rwd_db, request.concepts)
    assert database == mock_cl_revision.database
    assert concepts == mock_cl_revision.concepts


@patch("ascent_domain.concept_lists._load_concepts_from_ids", new_callable=AsyncMock)
async def test_validate_concepts_ids(mock_load_concepts_from_concepts, mock_rwd_db, mock_cl_revision):
    # arrange
    request = ConceptIDs(concept_ids=[c["concept_id"] for c in mock_cl_revision.concepts])
    mock_load_concepts_from_concepts.return_value = mock_cl_revision.concepts

    # act
    database, concepts = await service.validate_concepts(request, mock_rwd_db)

    # assert
    mock_load_concepts_from_concepts.assert_called_once_with(mock_rwd_db, set(request.concept_ids))
    assert database is None
    assert concepts == mock_cl_revision.concepts


@patch("ascent_domain.concept_lists._load_concepts_from_ids_medical_coder", new_callable=AsyncMock)
async def test_validate_concepts_ids_database(mock_load_concepts_from_ids_medical_coder, mock_rwd_db, mock_cl_revision):
    # arrange
    request = ConceptIDs(concept_ids=[c["concept_id"] for c in mock_cl_revision.concepts], database=mock_cl_revision.database)
    mock_load_concepts_from_ids_medical_coder.return_value = mock_cl_revision.concepts

    # act
    database, concepts = await service.validate_concepts(request, mock_rwd_db)

    # assert
    mock_load_concepts_from_ids_medical_coder.assert_called_once_with(mock_cl_revision.database, set(request.concept_ids))
    assert database == mock_cl_revision.database
    assert concepts == mock_cl_revision.concepts


@patch("ascent_domain.concept_lists._load_concept_from_codes", new_callable=AsyncMock)
async def test_validate_concepts_codes(mock_load_concepts_from_concepts, mock_rwd_db, mock_cl_revision):
    # arrange
    request = ConceptCodes(concept_codes=[c["concept_code"] for c in mock_cl_revision.concepts], vocabulary_id="SNOMED")
    mock_load_concepts_from_concepts.return_value = mock_cl_revision.concepts

    # act
    database, concepts = await service.validate_concepts(request, mock_rwd_db)

    # assert
    mock_load_concepts_from_concepts.assert_called_once_with(mock_rwd_db, set(request.concept_codes), "SNOMED")
    assert database is None
    assert concepts == mock_cl_revision.concepts


@patch("ascent_domain.concept_lists._load_concepts_from_codes_medical_coder", new_callable=AsyncMock)
async def test_validate_concepts_codes_database(mock_load_concepts_from_codes_medical_coder, mock_rwd_db, mock_cl_revision):
    # arrange
    request = ConceptCodes(
        concept_codes=[c["concept_code"] for c in mock_cl_revision.concepts], vocabulary_id="SNOMED", database=mock_cl_revision.database
    )
    mock_load_concepts_from_codes_medical_coder.return_value = mock_cl_revision.concepts

    # act
    database, concepts = await service.validate_concepts(request, mock_rwd_db)

    # assert
    mock_load_concepts_from_codes_medical_coder.assert_called_once_with(mock_cl_revision.database, set(request.concept_codes), "SNOMED")
    assert database == mock_cl_revision.database
    assert concepts == mock_cl_revision.concepts


@patch("ascent_domain.concept_lists.validate_concepts", new_callable=AsyncMock)
async def test_create_concept_list_revision(mock_validate_concepts, mock_app_db, mock_cl_revision, mock_cl_id):
    # arrange
    concepts = Concepts(database=mock_cl_revision.database, concepts=mock_cl_revision.concepts)
    mock_validate_concepts.return_value = (None, concepts.concepts)
    request = ConceptListRevisionRequestDTO(name=mock_cl_revision.name, description=mock_cl_revision.description, concepts=concepts)

    # act
    concept_list = await service.create_concept_list_revision(mock_cl_id, request, mock_app_db)

    # assert
    assert concept_list.id is not None
    assert concept_list.name == request.name
    assert concept_list.description == request.description
    assert concept_list.concepts == concepts.concepts


@patch("ascent_domain.concept_lists.validate_concepts", new_callable=AsyncMock)
async def test_create_concept_list_revision_exists(mock_validate_concepts, mock_app_db, mock_cl_revision, mock_cl_id):
    # arrange
    request = ConceptListRevisionRequestDTO(
        name=mock_cl_revision.name,
        description=mock_cl_revision.description,
        concepts=Concepts(database=mock_cl_revision.database, concepts=mock_cl_revision.concepts),
    )
    mock_validate_concepts.return_value = (None, request.concepts.concepts)
    mock_app_db.add.side_effect = IntegrityError(None, None, Exception())

    # act
    with pytest.raises(service.ConceptListRevisionExists):
        await service.create_concept_list_revision(mock_cl_id, request, mock_app_db)


async def test_retrieve_concept_list_revision(mock_app_db, mock_user, mock_cl_revision, mock_cl_id, mock_cl_revision_dto):
    # arrange
    mock_app_db.scalars.return_value.one.return_value = mock_cl_revision

    # act
    concept_list_revision = await service.retrieve_concept_list_revision(mock_cl_id, mock_cl_revision.id, mock_user, mock_app_db)

    # assert
    assert concept_list_revision == mock_cl_revision_dto


async def test_retrieve_concept_list_revision_invalid_id(mock_app_db, mock_user, mock_cl_revision_id, mock_cl_id):
    # arrange
    mock_app_db.scalars.return_value.one.side_effect = NoResultFound

    # act
    with pytest.raises(service.ConceptListRevisionNotFound):
        await service.retrieve_concept_list_revision(mock_cl_id, mock_cl_revision_id, mock_user, mock_app_db)


async def test_delete_concept_list_revision(mock_app_db, mock_user, mock_cl_revision_id, mock_cl_revision, mock_cl_id):
    # arrange
    mock_app_db.scalars.return_value.all.return_value = [mock_cl_revision]

    # act
    await service.delete_concept_list_revisions(mock_cl_id, {mock_cl_revision_id}, mock_user, mock_app_db)

    # assert
    mock_app_db.delete.assert_called_once_with(mock_cl_revision)


async def test_delete_single_concept_list_revision_with_invalid_id(mock_app_db, mock_user, mock_cl_revision_id, mock_cl_id):
    # arrange
    mock_app_db.scalars.return_value.one.side_effect = NoResultFound

    # act
    with pytest.raises(service.ConceptListRevisionNotFound):
        await service.delete_concept_list_revisions(mock_cl_id, {mock_cl_revision_id}, mock_user, mock_app_db)


async def test_delete_single_concept_list_revision_referenced(mock_app_db, mock_user, mock_cl_revision_id, mock_cl_revision, mock_cl_id):
    # arrange
    mock_app_db.scalars.return_value.all.return_value = [mock_cl_revision]
    mock_app_db.delete.side_effect = IntegrityError(None, None, Exception())

    # act
    with pytest.raises(service.ConceptListRevisionIsUsed):
        await service.delete_concept_list_revisions(mock_cl_id, {mock_cl_revision_id}, mock_user, mock_app_db)


async def test_resolve_user_concept(mock_app_db, mock_user, mock_cl_revision, mock_cl, mock_cl_medical_concept):
    # arrange
    mock_app_db.scalars.return_value.one.return_value = (mock_cl.name, mock_cl.category, mock_cl_revision.concepts)

    # act
    medical_concept = await service._resolve_user_concept(mock_cl_revision.id, mock_user, mock_app_db)

    # assert
    assert medical_concept == mock_cl_medical_concept


async def test_resolve_user_concept_invalid_id(mock_app_db, mock_user, mock_cl_revision_id, mock_cl_id):
    # arrange
    mock_app_db.scalars.return_value.one.side_effect = NoResultFound

    # act
    with pytest.raises(service.ConceptListRevisionNotFound):
        await service._resolve_user_concept(mock_cl_revision_id, mock_user, mock_app_db)


@patch("ascent_domain.concept_lists._resolve_user_concept", new_callable=AsyncMock)
async def test_resolve_user_concepts_id(mock_resolve_user_concept, mock_app_db, mock_user, mock_cl_medical_concept):
    # arrange
    concept_list_revision_id = uuid4()
    mock_resolve_user_concept.return_value = mock_cl_medical_concept

    # act
    (medical_concept,) = await service.resolve_user_concepts([concept_list_revision_id], mock_user, mock_app_db)

    # assert
    mock_resolve_user_concept.assert_called_once_with(concept_list_revision_id, mock_user, mock_app_db)
    assert medical_concept == mock_cl_medical_concept


@patch("ascent_domain.concept_lists._resolve_user_concept", new_callable=AsyncMock)
async def test_resolve_user_concepts(mock_resolve_user_concept, mock_app_db, mock_user, mock_cl_medical_concept):
    # act
    (medical_concept,) = await service.resolve_user_concepts([mock_cl_medical_concept], mock_user, mock_app_db)

    # assert
    mock_resolve_user_concept.assert_not_called()
    assert medical_concept == mock_cl_medical_concept


def test_user_concepts_to_explorer_concepts(mock_cl_medical_concept):
    # act
    explorer_concepts = service.user_concepts_to_explorer_concepts([mock_cl_medical_concept])

    # assert
    assert explorer_concepts == {"Concept Name": "200461,4132140"}


@patch("ascent_domain.concept_lists.resolve_user_concepts", new_callable=AsyncMock)
@patch("ascent_domain.concept_lists.user_concepts_to_explorer_concepts")
async def test_resolve_user_concepts_to_explorer_concepts(
    mock_user_concepts_to_explorer_concepts, mock_resolve_user_concepts, mock_app_db, mock_user, mock_cl_medical_concept
):
    # arrange
    concept_list_revision_id = uuid4()
    explorer_concepts_ref = {"Concept Name": "200461,4132140"}
    mock_resolve_user_concepts.return_value = [mock_cl_medical_concept]
    mock_user_concepts_to_explorer_concepts.return_value = explorer_concepts_ref

    # act
    explorer_concepts = await service.resolve_user_concepts_to_explorer_concepts([concept_list_revision_id], mock_user, mock_app_db)

    # assert
    mock_resolve_user_concepts.assert_called_once_with([concept_list_revision_id], mock_user, mock_app_db)
    mock_user_concepts_to_explorer_concepts.assert_called_once_with([mock_cl_medical_concept])
    assert explorer_concepts == explorer_concepts_ref


# ---------------------------------------------------------------------------
# Concept lookups bind their inputs.
#
# concept_codes and vocabulary_id arrive on a request payload, and the caller
# names them *_unverified. Quoted into the statement, any value containing a
# quote could rewrite the query. These assert on the property that prevents
# that: the values reach the driver as parameters and never appear in the SQL
# text.
# ---------------------------------------------------------------------------

HOSTILE_CODE = "x' OR '1'='1"
HOSTILE_VOCAB = "SNOMED'; DROP TABLE concept; --"


@patch("ascent_domain.concept_lists.execute_query", new_callable=AsyncMock)
async def test_codes_and_vocabulary_are_bound_not_interpolated(mock_execute_query, mock_rwd_db):
    mock_execute_query.return_value = ([(1, "n", HOSTILE_CODE, HOSTILE_VOCAB)], None)

    await service._load_concept_from_codes(mock_rwd_db, {HOSTILE_CODE}, vocabulary_id=HOSTILE_VOCAB)

    sql = mock_execute_query.call_args.args[1]
    params = mock_execute_query.call_args.kwargs["params"]

    assert HOSTILE_CODE not in sql, "concept code reached the SQL text"
    assert HOSTILE_VOCAB not in sql, "vocabulary id reached the SQL text"
    assert "DROP TABLE" not in sql
    assert params == [HOSTILE_CODE, HOSTILE_VOCAB], "values must arrive as bound parameters"


@patch("ascent_domain.concept_lists.execute_query", new_callable=AsyncMock)
async def test_concept_ids_are_bound_not_interpolated(mock_execute_query, mock_rwd_db):
    mock_execute_query.return_value = ([(7, "n", "c", "v")], None)

    await service._load_concepts_from_ids(mock_rwd_db, {7})

    sql = mock_execute_query.call_args.args[1]
    assert "IN (?)" in sql
    assert mock_execute_query.call_args.kwargs["params"] == [7]


@patch("ascent_domain.concept_lists.execute_query", new_callable=AsyncMock)
async def test_empty_input_does_not_emit_invalid_in_clause(mock_execute_query, mock_rwd_db):
    """``IN ()`` is a syntax error; the old string-join produced exactly that."""
    assert await service._load_concept_from_codes(mock_rwd_db, set(), vocabulary_id="SNOMED") == []
    assert await service._load_concepts_from_ids(mock_rwd_db, set()) == []
    mock_execute_query.assert_not_called()
