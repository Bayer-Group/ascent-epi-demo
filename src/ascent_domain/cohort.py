import logging
import uuid
from typing import List, TypedDict

from pydantic import UUID4

from ascent_domain.concept_lists import resolve_user_concepts, transform_to_custom_code_mappings
from ascent_domain.data_queries.cohort import (
    authorize_cohort,
    fetch_cohort_dataframe,
    grant_cohort_access,
    store_cohort,
    store_cohort_descriptor,
    update_cohort_descriptor_funnel,
)
from ascent_domain.errors import BadRequestError
from ascent_domain.models.data_definitions import (
    CodingSystem,
    CohortDescriptor,
    CohortRole,
    Criteria,
    CriteriaType,
    FunnelCriterion,
    MedicalConcept,
    User,
    UserCohortDescriptor,
    UserConcepts,
)
from ascent_domain.omop.data.processing.sql_post_processor import ConceptNotFoundError
from ascent_domain.omop.models.inference.criteria_to_counts import CriteriaProcessor
from ascent_platform.db.postgresql_session import AsyncSessionLocal
from ascent_platform.identity import TokenFetcher
from ascent_platform.warehouse.session import get_db, get_db_schema
from ascent_platform.warehouse.user_session import get_db_as_user

logger = logging.getLogger(__name__)


class CohortPreview(TypedDict):
    concepts: List[MedicalConcept]
    query_template: str
    query: str


async def prepare_criteria(user_input: str, service: CriteriaProcessor) -> Criteria:
    prepared_criteria = await service.text_to_criteria(user_input)
    return Criteria.model_validate(prepared_criteria)


async def preview_cohort_query(
    database: str, database_schema: str | None, coding_system: CodingSystem, criteria: Criteria, service: CriteriaProcessor
) -> CohortPreview:
    logger.info(f"preview_cohort: database={database}, coding_system={coding_system}, criteria={criteria}")

    attempts = 5
    missing_concept_mappings = {}  # on this step we will allow missing concepts and ask the user to fill these later manually
    for i in range(attempts):
        try:
            (_, query_template, query, *_) = await service.criteria_to_data(
                criteria_dict=criteria.model_dump(),
                preferred_coding_system=coding_system,
                snowflake_database=database,
                snowflake_database_schema=database_schema,
                main_path=None,
                log_folder=None,
                querylib_file=None,
                verify=False,
                run_query=False,
                custom_code_mappings=missing_concept_mappings,
            )
            break
        except ConceptNotFoundError as e:
            if i + 1 < attempts:
                missing_concept_mappings[(e.category, e.name)] = ""
            else:
                raise

    logger.info(f"preview_cohort: query_template={query_template}, query={query}")
    concepts_raw = service.find_medical_concepts(query_template=query_template or "", database=database)

    concepts = []
    for raw_item in concepts_raw:
        if raw_item["category"] == "drug_class":
            # special handling for drug_class, as it's a nested dictionary
            for drug_name, drug_concepts in raw_item["value"].items():
                concepts.append(MedicalConcept(name=drug_name, category=raw_item["name"], concepts=drug_concepts))
        else:
            # use from_medical_coder for non-drug_class categories
            concepts.append(MedicalConcept.from_medical_coder(raw_item))

    return {"query_template": query_template or "", "query": query, "concepts": concepts}


async def create_cohort(
    database: str,
    database_schema: str | None,
    coding_system: CodingSystem,
    criteria: Criteria,
    user_concepts: UserConcepts | None,
    token_fetcher: TokenFetcher | None,
    app_db: AsyncSessionLocal,
    user: User,
    service: CriteriaProcessor,
) -> CohortDescriptor:
    if concepts := user_concepts:
        user_concepts = await resolve_user_concepts(concepts, user, app_db)
        explorer_concepts = transform_to_custom_code_mappings(user_concepts)
        logger.info("Consider user-defined concepts for {} for cohort generation".format({c.name for c in user_concepts}))
    else:
        user_concepts = None
        explorer_concepts = None

    # query cohort
    (message, query_template, query, *_) = await service.criteria_to_data(
        criteria_dict=criteria.model_dump(),
        preferred_coding_system=coding_system,
        snowflake_database=database,
        snowflake_database_schema=database_schema,
        main_path=None,
        log_folder=None,
        querylib_file=None,
        custom_code_mappings=explorer_concepts,
        verify=False,
        run_query=False,
    )

    if query_template is not None:
        query_explanation = await service.get_query_explanation(query_template, criteria.model_dump())

    # Persist cohort: SELECT runs as the SSO user against the RWD DB; the
    # resulting patient list is written to ASCENT.ASCENT_COHORTS via the
    # machine-user pool. SSO users have no grants on ASCENT.
    # token_fetcher=None signals the caller is a machine user (cannot OBO);
    # fall back to the machine-user pool for the RWD read too.
    logger.info("Store cohort")
    cohort_id = uuid.uuid4()
    if token_fetcher is None:
        async with get_db(database, database_schema) as rwd_db:
            cohort_df = await fetch_cohort_dataframe(rwd_db, query)
    else:
        sf_token = await token_fetcher()
        async with get_db_as_user(database, sf_token, database_schema) as rwd_db:
            cohort_df = await fetch_cohort_dataframe(rwd_db, query)
    async with get_db("ASCENT") as ascent_db:
        table_info = await store_cohort(cohort_id=cohort_id, ascent_db=ascent_db, cohort=cohort_df)

    logger.info("Store cohort descriptor")
    if database_schema is None:
        database_schema = await get_db_schema(database)

    descriptor = UserCohortDescriptor(
        # Shared fields
        id=cohort_id,
        state="pending",
        description=f"NOTE: {message}" if message else None,
        database=database,
        database_schema=database_schema,
        table=table_info.name,
        size=table_info.size,
        attributes=table_info.columns,
        # User-cohort fields
        owner_id=user.email if user.email else "",
        criteria=criteria,
        coding_system=coding_system,
        user_concepts=user_concepts,
        query_template=query_template,
        query=query,
        explanation=query_explanation if query_template else None,
    )
    await store_cohort_descriptor(app_db=app_db, descriptor=descriptor)

    return descriptor


def _compose_funnel_criteria(
    descriptions: List[str],
    query_templates: List[str],
    queries: List[str],
    types: List[CriteriaType],
    sizes: List[int],
) -> List[FunnelCriterion]:
    """Convert split data into funnel criteria objects"""
    return [
        FunnelCriterion(
            index=i,
            description=description,
            query_template=query_template,
            query=query,
            type=type_,
            size=size,
        )
        for i, (query_template, query, type_, description, size) in enumerate(zip(query_templates, queries, types, descriptions, sizes))
    ]


async def create_cohort_funnel(
    descriptor: CohortDescriptor,
    app_db: AsyncSessionLocal,
    user: User,
    service: CriteriaProcessor,
) -> List[FunnelCriterion]:
    explorer_concepts = transform_to_custom_code_mappings(descriptor.user_concepts) if descriptor.user_concepts else None

    # Collect funnel data
    query_templates, queries, cohorts, criteria, descriptions = await service.split_query_by_criteria(
        preferred_coding_system=descriptor.coding_system,
        snowflake_database=descriptor.database,
        snowflake_database_schema=descriptor.database_schema,
        query_template_pred=descriptor.query_template,
        criteria_dict=descriptor.criteria.model_dump(),
        custom_code_mappings=explorer_concepts,
    )
    sizes, _ = service.get_patient_funnel_from_dfs(output_dfs=cohorts, criteria_list=criteria)

    # Update the cohort descriptor
    funnel = _compose_funnel_criteria(descriptions=descriptions, query_templates=query_templates, queries=queries, types=criteria, sizes=sizes)
    await update_cohort_descriptor_funnel(app_db, user, descriptor.id, funnel)

    return funnel


class CannotShareStudyCohortsException(BadRequestError):
    def __init__(self):
        super().__init__(detail={"error": "Study cohorts cannot be shared"})


async def share_cohort(
    cohort_id: UUID4,
    recipient_email: str,
    app_db: AsyncSessionLocal,
    current_user: User,
    role: CohortRole = "viewer",
) -> None:
    """Grant another user access to a cohort, preserving the original ownership.

    Only the owner (creator) may share. The recipient gets ``editor`` or
    ``viewer`` access via a ``cohort_access`` grant; the cohort itself (and its
    creator) is unchanged. Re-sharing updates the recipient's role.

    Raises:
        CannotShareStudyCohortsException: the cohort is part of a study and cannot be shared
    """
    descriptor, _ = await authorize_cohort(app_db, current_user, cohort_id, "owner")

    if descriptor.origin == "study":
        raise CannotShareStudyCohortsException()

    await grant_cohort_access(app_db, current_user, cohort_id, recipient_email, role)
