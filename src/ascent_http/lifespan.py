import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from ascent_domain.garbage_collect_orphans_task import garbage_collect_cohorts
from ascent_domain.personalized_questions_task import generate_personalized_questions
from ascent_http.services.ai_services import load_query_library
from ascent_http.settings import require_app_config, settings
from ascent_platform.llm.executor import shutdown_executor as shutdown_llm_executor
from ascent_platform.warehouse.session import initialize_connection_pool

scheduler = AsyncIOScheduler()
logger = logging.getLogger(__name__)


class FilterDocs(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/docs" not in record.getMessage()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.debug(f"Use cache at {settings.REDIS_HOST!r} in {settings.CACHE!r} mode")

    # No OpenID discovery and no JWKS to warm: this deployment does not
    # authenticate.

    # Warm the JWKS cache before serving. A cold worker that takes a burst of
    # concurrent MCP `initialize` calls would otherwise have every one of them
    # racing to fetch the same signing keys.

    scheduler.start()

    # Schedule for immediate execution
    scheduler.add_job(load_query_library, "date")
    scheduler.add_job(initialize_connection_pool, "date")
    scheduler.add_job(generate_personalized_questions, "date")
    if settings.RELEASE_VERSION != "local":
        scheduler.add_job(garbage_collect_cohorts, "date")
    # No study-metadata sync: it read from services that do not exist here.

    # Schedule daily jobs (around midnight)
    scheduler.add_job(generate_personalized_questions, "cron", hour=0, minute=0)
    scheduler.add_job(garbage_collect_cohorts, "cron", hour=0, minute=15)

    try:
        yield
    finally:
        scheduler.shutdown()
        await asyncio.to_thread(shutdown_llm_executor)


@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    """Combined lifespan manager for FastAPI and all MCP apps."""
    # Configuration the API cannot run without. Asserted at startup rather than
    # declared required on Settings: the platform shares that class, and
    # importing the platform must not require a database password. Checking here
    # also names everything missing at once, rather than one field per run.
    require_app_config()

    # Import here to avoid circular import
    from ascent_mcp.app_experimental import mcp_experimental_app
    from ascent_mcp.app_ui import mcp_ui_app
    from ascent_mcp.app_v1 import mcp_v1_app

    async with lifespan(app):
        async with mcp_v1_app.lifespan(app):
            async with mcp_experimental_app.lifespan(app):
                async with mcp_ui_app.lifespan(app):
                    yield
