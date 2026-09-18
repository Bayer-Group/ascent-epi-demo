"""Typed application settings.

Every field binds to an exact environment-variable name via
``validation_alias``, grouped into nested models by concern.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import AliasChoices, BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _empty_to_false(v: object) -> object:
    """Map an empty/whitespace env value to ``False``.

    The legacy ``os.getenv`` config treated ``FOO=`` (empty) as falsy (or
    ``!= "true"``); a typed bool field would otherwise reject the empty string.
    Non-empty values pass through for pydantic's normal bool parsing.
    """
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return False
    return v


# bool env var that tolerates an empty value (empty -> False, matching old behavior)
EnvBool = Annotated[bool, BeforeValidator(_empty_to_false)]

_MODEL_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
    case_sensitive=False,
)


class ServiceSettings(BaseSettings):
    """Service / app-level configuration."""

    model_config = _MODEL_CONFIG

    debug: EnvBool = Field(default=False, validation_alias="DEBUG")
    release_version: str = Field(default="local", validation_alias="RELEASE_VERSION")
    api_version: str | None = Field(default=None, validation_alias="API_VERSION")
    service_path: str = Field(default="v1/medical-coder", validation_alias="SERVICE_PATH")
    cors_origins: str = Field(default="*", validation_alias="CORS_ORIGINS")
    profile_requests: EnvBool = Field(default=False, validation_alias="PROFILE_REQUESTS")
    # Where profiles go. Without a bucket they are kept in the container.
    profile_s3_bucket: str | None = Field(default=None, validation_alias="PROFILE_S3_BUCKET")
    profile_s3_prefix: str = Field(default="profiles", validation_alias="PROFILE_S3_PREFIX")
    # Eagerly load embedding models at startup (slow boot, fast first request); disable locally.
    warm_embedders_on_startup: EnvBool = Field(default=True, validation_alias="WARM_EMBEDDERS_ON_STARTUP")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_deployed(self) -> bool:
        return self.release_version in {"dev", "qa", "prod", "main"}


class AzureAuthSettings(BaseSettings):
    """Azure AD OAuth2 (bearer) settings for API auth."""

    model_config = _MODEL_CONFIG

    client_id: str | None = Field(default=None, validation_alias="AZURE_CLIENT_ID")
    tenant_id: str | None = Field(default=None, validation_alias="AZURE_TENANT_ID")

    @property
    def jwks_url(self) -> str:
        return f"https://login.microsoftonline.com/{self.tenant_id}/discovery/v2.0/keys"

    @property
    def issuer(self) -> str:
        return f"https://login.microsoftonline.com/{self.tenant_id}/v2.0"


class QdrantSettings(BaseSettings):
    """Qdrant vector store connection + collection names per encoder."""

    model_config = _MODEL_CONFIG

    host: str | None = Field(default=None, validation_alias="QDRANT_HOST")
    port: int = Field(default=6333, validation_alias="QDRANT_PORT")
    timeout: int = 180
    # Backpressure: process-wide cap so fan-out queues instead of stampeding the single Qdrant instance.
    max_concurrent_searches: int = Field(default=10, validation_alias="QDRANT_MAX_CONCURRENT_SEARCHES")

    bge_collection: str = Field(
        default="concepts-bge-hybrid-new",
        validation_alias="BGE_EMBEDDING_COLLECTION_QDRANT",
    )
    gemini_collection: str = Field(
        default="concepts-gemini-hybrid-new",
        validation_alias="GEMINI_EMBEDDING_COLLECTION_QDRANT",
    )
    biolord_collection: str = Field(
        default="concepts-biolord-hybrid-new",
        validation_alias="BIOLORD_EMBEDDING_COLLECTION_QDRANT",
    )


class PostgresSettings(BaseSettings):
    """User-facing Postgres app DB (cache + analytics). Optional — the cache /
    analytics layers no-op when ``host`` is unset."""

    model_config = _MODEL_CONFIG

    app_usr: str | None = Field(default=None, validation_alias="USR_DB_APP_USR")
    app_pwd: str | None = Field(default=None, validation_alias="USR_DB_APP_PWD")
    host: str | None = Field(default=None, validation_alias="USR_DB_HOST")
    port: str | None = Field(default=None, validation_alias="USR_DB_PORT")
    name: str | None = Field(default=None, validation_alias="USR_DB_NAME")

    @property
    def configured(self) -> bool:
        return bool(self.host)

    @property
    def async_url(self) -> str:
        return f"postgresql+asyncpg://{self.app_usr}:{self.app_pwd}@{self.host}:{self.port}/{self.name}"


class LLMSettings(BaseSettings):
    """LLM provider credentials.

    Only the providers actually reached by the coding pipeline are kept:
    Azure OpenAI (East-US-2 for chat, East-US for embeddings), Gemini
    (Developer API key), and Anthropic via Bedrock (ambient AWS credentials —
    no explicit fields).
    """

    model_config = _MODEL_CONFIG

    azure_openai_chat_endpoint: str | None = Field(default=None, validation_alias="AZURE_OPENAI_ENDPOINT_EAST2_US")
    azure_openai_chat_key: str | None = Field(default=None, validation_alias="AZURE_OPENAI_KEY_EAST2_US")
    azure_openai_chat_api_version: str | None = Field(
        default=None, validation_alias="AZURE_OPENAI_EAST_US_2_API_VERSION"
    )

    azure_openai_embed_endpoint: str | None = Field(default=None, validation_alias="AZURE_OPENAI_ENDPOINT_EAST_US")
    azure_openai_embed_key: str | None = Field(default=None, validation_alias="AZURE_OPENAI_KEY_EAST_US")
    azure_openai_embed_api_version: str | None = Field(
        default=None, validation_alias="AZURE_OPENAI_EAST_US_GPT4O_API_VERSION"
    )

    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )
    gemini_project: str | None = Field(default=None, validation_alias="GOOGLE_CLOUD_PROJECT")

    # Model names: single source of truth, all env-overridable. Nothing outside this class hardcodes a model id.

    # Literal-typed so a typo'd env value fails loudly at startup instead of flowing downstream.
    default_llm_filter: Literal["gemini", "chatgpt", "haiku"] = Field(
        default="gemini", validation_alias="DEFAULT_LLM_FILTER"
    )

    openai_chat_model: str = Field(default="gpt-5.2_ascent", validation_alias="AZURE_OPENAI_CHAT_MODEL")
    gemini_model: str = Field(default="gemini-3-flash-preview", validation_alias="GEMINI_MODEL_NAME")
    anthropic_model_id: str = Field(
        default="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        validation_alias="BEDROCK_ANTHROPIC_MODEL_ID",
    )

    grounded_search_provider: str = Field(default="gemini", validation_alias="GROUNDED_SEARCH_PROVIDER")
    grounded_search_model: str = Field(default="gemini-3-flash-preview", validation_alias="GROUNDED_SEARCH_MODEL_NAME")
    # Low temperature: medical-code search is determinism-sensitive (legacy fallback ran at 0.1).
    grounded_search_temperature: float = Field(default=0.1, validation_alias="GROUNDED_SEARCH_TEMPERATURE")

    non_omop_model: str = Field(default="gemini-3-flash-preview", validation_alias="NON_OMOP_MODEL_NAME")

    azure_embed_model: str = Field(default="text-embedding-3-large_ascent", validation_alias="AZURE_OPENAI_EMBED_MODEL")
    gemini_embed_model: str = Field(default="gemini-embedding-001", validation_alias="GEMINI_EMBED_MODEL")
    sap_embed_model: str = Field(
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        validation_alias="SAP_EMBED_MODEL",
    )
    bge_embed_model: str = Field(default="BAAI/bge-base-en-v1.5", validation_alias="BGE_EMBED_MODEL")
    biolord_embed_model: str = Field(default="FremyCompany/BioLORD-2023", validation_alias="BIOLORD_EMBED_MODEL")


class Settings(BaseSettings):
    """Root settings aggregating every concern-scoped group."""

    model_config = _MODEL_CONFIG

    service: ServiceSettings = Field(default_factory=ServiceSettings)
    auth: AzureAuthSettings = Field(default_factory=AzureAuthSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached)."""
    return Settings()
