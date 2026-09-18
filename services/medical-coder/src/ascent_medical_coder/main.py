"""FastAPI application factory.

Mirrors the current service's app wiring — same title/version, OpenAPI/docs URLs,
Swagger OAuth, CORS, 500 handler, optional profiling, lifespan, the open system
routes, the five Security-gated business routers, and the self-authenticating
WebSocket — but on the clean ``src``-layout package (no ``sys.path`` hack).
"""

# ruff: noqa: E402 -- load_dotenv() must run BEFORE the app imports below:
# settings singletons and connector modules read env at import time.
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Security
from starlette.middleware.cors import CORSMiddleware

from ascent_medical_coder.api.routers import coding, system
from ascent_medical_coder.connectors.embeddings.factory import warm_embedders
from ascent_medical_coder.core.errors import add_profiling_middleware, register_error_handlers
from ascent_medical_coder.core.logging import configure_logging
from ascent_medical_coder.core.security import azure_scheme
from ascent_medical_coder.core.settings import get_settings
from ascent_medical_coder.db.session import dispose_engine, init_models_once
from ascent_medical_coder.routes import business_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_models_once()
    if get_settings().service.warm_embedders_on_startup:
        # Load embedding models off the event loop at boot (slow start, fast first request).
        await asyncio.to_thread(warm_embedders)
    yield
    await dispose_engine()


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    prefix = f"/{settings.service.service_path}"

    app = FastAPI(
        title="Ascent Medical Coder Service",
        description="API server to enable medical coding.",
        version=settings.service.release_version,
        openapi_url=f"{prefix}/openapi.json",
        docs_url=f"{prefix}/docs",
        redoc_url=None,
        swagger_ui_oauth2_redirect_url=f"{prefix}/oauth2-redirect",
        swagger_ui_init_oauth={
            "usePkceWithAuthorizationCodeGrant": True,
            "clientId": settings.auth.client_id,
        },
        swagger_ui_parameters={"syntaxHighlight": False},
        lifespan=lifespan,
    )

    register_error_handlers(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.service.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    add_profiling_middleware(app)

    # Open system routes (health/version/test): no Security.
    app.include_router(system.router, prefix=prefix)

    app.include_router(business_router, prefix=prefix, dependencies=[Security(azure_scheme)])

    # WebSocket authenticates itself via the ?token= query param.
    app.websocket(f"{prefix}/ws/get-medical-codes-reasoning")(
        coding.websocket_get_medical_codes_reasoning
    )

    return app


app = create_app()

__all__ = ["app", "create_app"]
