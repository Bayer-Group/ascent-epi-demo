"""Construction of the AI services the domain runs on.

These were in ``ascent/services/ai_services.py``, where each one was defined as
``Annotated[X, Depends(factory)]``. Domain functions then took those aliases as
parameter types -- so a cohort rule's signature named a FastAPI dependency,
even though the functions are not routes and FastAPI never resolves them (the
routers declare the alias themselves and pass the value in).

The plain factories live here; ``ascent/services/ai_services.py`` keeps the
Annotated aliases, which are transport wiring and belong to the app.

The six-argument CriteriaProcessor construction was written out four times --
here, tools_v1_cohort, generate_cohort_attrition and _cohort_helpers -- each
reading the same four assistant_config keys. A model-policy change had to be
made in four places or the surfaces would silently disagree about which model
extracts criteria.
"""

from __future__ import annotations

import logging
from functools import cache

from ascent_domain.config import get_domain_settings
from ascent_domain.model_policy import assistant_config
from ascent_domain.omop.data.queries.query_library import QueryLibrary
from ascent_domain.omop.models.inference.criteria_to_counts import CriteriaProcessor
from ascent_domain.omop.models.inference.qa import QuestionAnsweringSystem
from ascent_domain.omop.models.inference.rag import QueryLibraryManager
from ascent_domain.omop.models.inference.table_one import Table1
from ascent_platform.config.runtime import get_settings
from ascent_platform.identity import TokenFetcher, ci_service_account_mode
from ascent_platform.warehouse.session import get_db_schema
from ascent_platform.warehouse.user_session import make_user_cursor_provider

logger = logging.getLogger(__name__)


@cache
def load_query_library() -> QueryLibrary:
    """The retrieval corpus of worked example queries.

    The library is optional and may be absent.

    A missing library degrades retrieval rather than stopping the service: the
    OMOP question path loses its few-shot examples and gets worse at SQL, but
    every other route -- discovery, execution, the non-OMOP pipeline, the
    coder -- is unaffected. Raising here would take all of them down for a
    missing optional corpus, and the FastAPI dependency wiring means the
    failure would surface on unrelated endpoints.
    """
    querylib_file = str(get_domain_settings().QUERY_LIBRARY)
    logger.info(f"Loading query library from {querylib_file!r}")
    try:
        return QueryLibraryManager.get_instance().load_querylib(querylib_file=querylib_file)
    except FileNotFoundError:
        logger.warning(
            "No query library at %s. OMOP question answering will run without "
            "retrieved examples and produce weaker SQL. Everything else is "
            "unaffected.",
            querylib_file,
        )
        return QueryLibrary()


def make_criteria_processor() -> CriteriaProcessor:
    """The one place model policy becomes a CriteriaProcessor."""
    return CriteriaProcessor(
        text2sql_assistant_type=assistant_config["text2sql"]["type"],
        mask_assistant_type=assistant_config["mask"]["type"],
        text_to_criteria_assistant_type=assistant_config["text_to_criteria"]["type"],
        text2sql_assistant_model_name=assistant_config["text2sql"]["model"],
        mask_assistant_model_name=assistant_config["mask"]["model"],
        text_to_criteria_assistant_model_name=assistant_config["text_to_criteria"]["model"],
    )


def make_table_one() -> Table1:
    return Table1()


class SearchService:
    """Builds a QuestionAnsweringSystem for a given database.

    Was ``_SearchService``; the underscore only existed because the public name
    was taken by the Annotated alias next to it.
    """

    def __init__(self, query_library: QueryLibrary | None = None):
        settings = get_domain_settings()
        self._root_dir = settings.PROJECT_ROOT
        self._log_dir = self._root_dir / "logs"
        self._query_library = query_library
        self._query_library_file = settings.QUERY_LIBRARY

    async def search_client(
        self,
        database: str | None = None,
        token_fetcher: TokenFetcher | None = None,
        cursor_provider=None,
    ) -> QuestionAnsweringSystem:
        schema = None
        if database:
            db_schema_pair = database.upper().split(".")
            database = db_schema_pair[0]
            schema = db_schema_pair[1] if len(db_schema_pair) > 1 else None
            if schema is None:
                schema = await get_db_schema(database=database)

        # SSO is only required when the QA system will actually query Snowflake.
        # Callers that build a QA service purely for LLM-side work (visualization,
        # query explanation, uncertainty analysis) pass ``database=None`` and never
        # invoke ``_db_connection()`` — no fetcher is needed there. ``ASCENT`` is
        # always machine-user (metadata DB), so it doesn't require SSO either.
        # A caller-supplied ``cursor_provider`` (e.g. a session-pinned provider
        # for cohort-scoped queries, see ``user_cursor_session``) takes
        # precedence over the standard per-call SSO provider.
        if cursor_provider is None and database and database.upper() != "ASCENT":
            if token_fetcher is None:
                if not ci_service_account_mode():
                    raise RuntimeError(
                        "search_client requires a Snowflake token_fetcher when a user RWD database "
                        "is specified: SSO is mandatory inside the backend for user RWD queries; "
                        "machine-user fallback is not allowed."
                    )
            else:
                cursor_provider = make_user_cursor_provider(token_fetcher)

        return QuestionAnsweringSystem(
            base_dir=self._root_dir,
            log_folder=self._log_dir,
            querylib_file=self._query_library_file,
            assistant_type=assistant_config["text2sql"]["type"],
            model_name=assistant_config["text2sql"]["model"],
            snowflake_database=database,
            snowflake_database_schema=schema if schema else None,
            cursor_provider=cursor_provider,
            settings=get_settings(),
        )
