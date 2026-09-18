"""Infrastructure configuration the platform reads for itself.

The platform layer needs values -- Postgres host, Redis host, the cache
backend -- and reads them here rather than reaching up to the app package,
which the one-way dependency rule forbids it from importing.

Constructed lazily: importing a module should not read and validate the
environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ascent_platform.config.paths import PROJECT_ROOT as _PROJECT_ROOT


class Settings(BaseSettings):
    """Infrastructure endpoints. Optional, because the platform must stay
    importable in contexts that use only part of it (a test touching Redis
    should not require Postgres credentials)."""

    model_config = SettingsConfigDict(env_file=(".env", ".env.local"), extra="ignore")

    # Defaults match the postgres service in docker-compose, so the stack
    # starts with no .env at all: the database is part of the shipped stack.
    DB_HOST: Optional[str] = "localhost"
    DB_PORT: Optional[int] = 5432
    DB_NAME: Optional[str] = "ascent"
    DB_APP_USR: Optional[str] = "ascent"
    DB_APP_PWD: Optional[str] = "ascent"

    CACHE: str = "default"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    CACHE_REDIS_MAX_CONNECTIONS: int = 50
    CACHE_REDIS_SOCKET_TIMEOUT: float = 5.0

    # Local warehouse: the data ships with the repo, so it takes a directory
    # and a scale rather than an account, a warehouse and a role.
    # The single local identity. There is no authentication; this only
    # decides who demo data is attributed to.
    # Optional shared bearer. Unset (the default) means no authentication at
    # all, which is correct for a localhost stack and wrong for anything else.
    # Setting it requires the token on every HTTP and MCP request.
    API_AUTH_TOKEN: Optional[str] = None

    WAREHOUSE_EXECUTOR_WORKERS: int = 20
    WAREHOUSE_DATA_DIR: str = "data/synthetic"
    WAREHOUSE_WORK_DIR: str = ".warehouse"
    # DATA_SCALE is the documented name -- it is what .env.template, the README
    # and docker-compose all use. Both spellings are accepted, so the from-source
    # path honours DATA_SCALE the same way compose does.
    WAREHOUSE_SCALE: str = Field(default="1k", validation_alias=AliasChoices("WAREHOUSE_SCALE", "DATA_SCALE"))

    # Outbound HTTP budgets
    MEDICAL_CODER_CONNECT_TIMEOUT: float = 5.0
    MEDICAL_CODER_READ_TIMEOUT: float = 300.0
    MEDICAL_CODER_TOTAL_TIMEOUT: float = 600.0
    MEDICAL_CODER_TOKEN_HTTP_TIMEOUT: float = 10.0
    # The coder that ships in this stack. docker-compose sets this too; the
    # default matters for anyone running the app outside compose.
    MEDICAL_CODER_BASE_URL: str = "http://medical-coder:8000/v1/medical-coder"
    # Size of the dedicated bulkhead AscentClient uses for the blocking MSAL
    # token call, so it never lands on the default executor.
    ASCENT_CLIENT_TOKEN_WORKERS: int = 4
    # Per-request fan-out over entities in the non-OMOP coder path. Each entity
    # burns a worker via asyncio.to_thread; unbounded gather across N concurrent
    # requests demanded N x M threads and starved the pool.
    MEDICAL_CODER_ENTITY_CONCURRENCY: int = 5

    # LLM
    GEMINI_HTTP_TIMEOUT_SECONDS: float = 120.0
    VERTEX_AI_LOCATION: str = "us-central1"
    LLM_MAX_CONCURRENT_CALLS: int = 8
    # Dedicated bulkhead for provider SDKs that expose only synchronous APIs.
    # These calls must not occupy the event loop or Python's shared executor.
    # No queued backlog: calls above this capacity fail immediately.
    LLM_BLOCKING_EXECUTOR_WORKERS: int = Field(default=8, ge=1)
    GEMINI_API_KEY: Optional[str] = None
    # Alternative spelling of the same credential, accepted second. Resolution
    # order lives in ascent_platform.llm.credentials, which is the only place
    # that should read either of these.
    GOOGLE_API_KEY: Optional[str] = None
    GOOGLE_CLOUD_PROJECT: Optional[str] = None

    # Azure OpenAI, for the assistant client family.
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_KEY_GPT4O: Optional[str] = None
    OPENAI_API_KEY_O1: Optional[str] = None
    OPENAI_API_BASE: Optional[str] = None
    OPENAI_API_VERSION: Optional[str] = None
    MODEL_NAME: Optional[str] = None

    # Per-provider keys for the remaining assistant clients.
    AZURE_INFERENCE_KEY_DEEPSEEK_R1_EAST_US2: Optional[str] = None
    # Azure AI Foundry serverless endpoints are per-deployment, so there is no
    # sensible default.
    AZURE_INFERENCE_ENDPOINT_DEEPSEEK_R1: Optional[str] = None
    MISTRAL_API_KEY: Optional[str] = None
    # Claude direct, as an alternative to the Bedrock-backed claude_sonnet.
    ANTHROPIC_API_KEY: Optional[str] = None

    # Call sites name a provider directly ("gpt", "claude_sonnet"). Inside a
    # deployment where all of them are configured that reads as a role; with a
    # single key configured it is a hard dependency on a provider the operator
    # may not have. These make the mapping explicit.
    #   LLM_PROVIDER_ALIASES='{"gpt":"gemini","claude_sonnet":"gemini"}'
    LLM_PROVIDER_ALIASES: Optional[dict[str, str]] = None

    @field_validator("LLM_PROVIDER_ALIASES", mode="before")
    @classmethod
    def _empty_alias_map_is_unset(cls, value):
        """Treat an empty value as unset.

        docker compose substitutes ``${LLM_PROVIDER_ALIASES:-}`` to an empty
        string when the variable is not in .env, and pydantic rejects "" for a
        dict field -- so leaving an optional setting alone crashed startup.
        """
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return value

    LLM_FALLBACK_PROVIDER: Optional[str] = None
    # The router's fallback when a model fails.
    FALLBACK_MODEL_NAME: str = "gemini-3.7-flash"

    # Azure AD, for the on-behalf-of exchange that turns a user's bearer into a
    # Snowflake-scoped token.
    # MSAL_* are accepted as aliases: they name the same app registration, so a
    # deployment setting either spelling works.
    AZURE_CLIENT_ID: Optional[str] = Field(default=None, validation_alias=AliasChoices("AZURE_CLIENT_ID", "MSAL_CLIENT_ID"))
    AZURE_CLIENT_SECRET: Optional[str] = Field(default=None, validation_alias=AliasChoices("AZURE_CLIENT_SECRET", "MSAL_CLIENT_SECRET"))
    AZURE_TENANT_ID: Optional[str] = None
    # "" not None: MSAL compares scope strings internally, so a None here makes
    # acquire_token_on_behalf_of raise TypeError deep inside the library.
    SNOWFLAKE_OAUTH_SCOPE: str = ""
    SNOWFLAKE_OBO_EXECUTOR_WORKERS: int = 8
    SNOWFLAKE_OBO_HTTP_TIMEOUT_SECONDS: float = 10.0
    SNOWFLAKE_OBO_TOTAL_TIMEOUT_SECONDS: float = 30.0
    SNOWFLAKE_ROLE: Optional[str] = None
    # The warehouse every per-user OAuth connection runs in.
    USER_WAREHOUSE: str = "COMPUTE_WH"

    # Deployment identity. Read here as well as in the app's Settings because
    # platform code branches on them (local vs deployed behaviour) and must not
    # import the app to find out.
    RELEASE_VERSION: str = "local"
    APP_VERSION: Optional[str] = None
    CI_SERVICE_ACCOUNT_MODE: Optional[str] = None
    GITHUB_ACTIONS: Optional[str] = None
    DEBUG: bool = False

    # AWS X-Ray tracing and CloudWatch EMF metrics. Off by default: the shipped
    # stack runs no collector, so the exporter has nowhere to send a span and
    # the metrics are printed to stdout instead, one JSON document per tool
    # call. Enable it only where something is listening.
    TELEMETRY_ENABLED: bool = False

    # The origins the shipped stack actually serves: LibreChat on 3080. Not
    # "*", because main.py pairs this with allow_credentials=True, and Starlette
    # answers a wildcard-plus-credentials policy by echoing whatever Origin
    # asked. Any page in the browser could then call the MCP endpoints on
    # localhost and read the replies. Override for a different front end.
    CORS_ORIGINS: str = "http://localhost:3080,http://127.0.0.1:3080"
    PROJECT_ROOT: Path = _PROJECT_ROOT
    BASE_DIR: Path = _PROJECT_ROOT

    # Logging. Read by setup_logging(); nothing else in the process may call
    # it, see the note on that method.
    LOG_LEVEL: str = "INFO"
    LOG_BASE_DIR: Optional[str] = None
    LOG_FORMAT: str = "%(asctime)s [%(levelname)s] - %(name)s: %(message)s"
    LOG_FILE_FORMAT: str = "%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s"
    LOG_DATE_FORMAT: str = "%H:%M:%S"
    LOG_SUBDIRS: bool = True
    LOG_SUBDIR_FORMAT: str = "%Y/%m/%d"
    USE_STRUCTURED_LOGGING: bool = False
    SILENT_LOGGERS: dict[str, str] = {
        "snowflake.connector": "WARNING",
        "sentence_transformers.SentenceTransformer": "WARNING",
        "ascent_platform.external.ascent_client": "WARNING",
        "google_genai.models": "WARNING",
        "httpx": "WARNING",
    }

    # Model policy that is configuration rather than domain decision: which
    # model serves which non-OMOP step. (The OMOP assistant_config lives in
    # ascent_domain.model_policy, which is a domain decision, not a knob.)
    # The model GeminiClient uses when a caller does not name one.
    GEMINI_DEFAULT_MODEL_NAME: str = "gemini-3.8-flash"
    CONTEXT_AGENT_MODEL_NAME: str = "gemini-3.1-pro-preview"
    # Model behind google_search_grounding.
    GOOGLE_SEARCH_MODEL_NAME: str = "gemini-3-flash-preview"
    SEMANTIC_SEARCH_MODEL_NAME: str = "gemini-2.5-flash"
    O4_MINI_MODEL_NAME: str = "o4-mini_ascent"
    NON_OMOP_QUESTION_SANITY_CHECK_MODEL_NAME: str = "gemini-3.7-flash"
    NON_OMOP_QUESTION_ANALYSIS_MODEL_NAME: str = "gemini-3.7-flash"
    NON_OMOP_SQL_PREPARATION_MODEL_NAME: str = "gemini-3.7-flash"
    NON_OMOP_SQL_RESULTS_HEALING_MODEL_NAME: str = "gemini-3.6-flash"
    NON_OMOP_SQL_TO_QUESTION_MODEL_NAME: str = "gemini-3.6-flash"

    # OMOP query library. The validator is why this cannot be a plain field:
    # unset means "newest querylib_*.db beside the checkout", and reading it as
    # None silently loads an empty library instead of failing.
    QUERY_LIBRARY: Optional[Path] = Path("data/querylib/querylib_public.db")
    CONCEPT_LOOKUP_DATABASE: str = "SYNTHETIC_EHR_OMOP"
    SNOWFLAKE_DATABASE: Optional[str] = None
    SNOWFLAKE_DATABASE_SCHEMA: Optional[str] = None

    # Azure OpenAI east-US deployment, used by the non-OMOP clients.
    OPENAI_EAST_US_BASE: Optional[str] = None
    OPENAI_EAST_US_KEY: Optional[str] = None
    OPENAI_EAST_US_GPT4O_MODEL_NAME: Optional[str] = None
    OPENAI_EAST_US_GPT4O_API_VERSION: Optional[str] = None

    # The second east-US deployment, which the o-series chat clients name
    # directly. Both fall back to the generic OPENAI_API_BASE / OPENAI_API_KEY
    # when unset.
    OPENAI_EAST_US_2_BASE: Optional[str] = None
    OPENAI_EAST_US_2_KEY: Optional[str] = None

    @field_validator("QUERY_LIBRARY", mode="before")
    @classmethod
    def _resolve_query_library(cls, value):
        if value:
            return value
        try:
            return max(_PROJECT_ROOT.glob("querylib_*.db"))
        except ValueError:
            return None

    @computed_field
    @property
    def BASE_URL(self) -> str:
        """Base URL for this deployment.

        Only used to advertise the MCP resource in OAuth discovery metadata,
        which a local client resolves at localhost.
        """
        return "http://localhost:8000"


_runtime: Settings | None = None


def get_settings() -> Settings:
    global _runtime
    if _runtime is None:
        _runtime = Settings()
    return _runtime


def set_settings(settings: Settings | None) -> None:
    """Override the runtime config -- the injection point for the app or tests."""
    global _runtime
    _runtime = settings


class _LazySettings:
    """Defers construction to first attribute access, so that importing a
    module does not read and validate the environment."""

    def __getattr__(self, name):
        return getattr(get_settings(), name)

    def __repr__(self) -> str:
        return f"<lazy {'unloaded' if _runtime is None else 'loaded'} Settings>"


# The process-wide lazy proxy. One instance, so `from ... import settings`
# resolves to the same object everywhere instead of each package building its
# own.
settings = _LazySettings()


# Aliases, so call sites can use either name.
RuntimeSettings = Settings
get_runtime_settings = get_settings
set_runtime_settings = set_settings


# What this deployment genuinely cannot start without. Empty, because
# everything it needs ships with the stack: the data is in the repo and the
# database defaults to the compose service.
#
# The one thing that cannot ship is a model. Every pipeline ends in an LLM
# call, so starting without one produces a service that accepts requests and
# fails every one of them -- worse than refusing to boot with a message that
# says which key to set. That check is below.
_APP_REQUIRED: tuple[str, ...] = ()


def require_app_config(settings: Settings | None = None) -> None:
    """Fail startup if the API is missing configuration it cannot run without.

    Call from the app entrypoint, not at import: a library import must not
    require a database password.

    Checks ``model_fields_set``, not truthiness, because required fields carry
    defaults. CORS_ORIGINS is why: it has a default and main.py calls
    ``.split(",")`` on it, so it can never be None or empty. "Was it actually
    supplied" is the question that has to be asked.
    """
    settings = settings or get_settings()
    supplied = settings.model_fields_set
    missing = [name for name in _APP_REQUIRED if name not in supplied or not getattr(settings, name, None)]
    # Ask the provider registry what is actually usable rather than checking a
    # hand-written list of key names, which drifts: a key that no provider
    # reads would pass startup and then fail every call.
    from ascent_platform.llm.availability import available_providers, requirements_summary

    if not available_providers():
        raise RuntimeError("No LLM provider is configured, and every pipeline ends in a model call. Configure one of:\n" + requirements_summary())
    if missing:
        raise RuntimeError("Missing required configuration: " + ", ".join(missing) + ". Set them in the environment or .env.")
