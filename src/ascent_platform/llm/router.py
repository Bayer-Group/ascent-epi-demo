"""
LLM Router - Unified interface for routing requests to various LLM providers.

This module provides a registry-based router that supports:
- Multiple LLM providers (Gemini, Azure OpenAI, etc.)
- Automatic fallback on failure
- Lazy client instantiation with caching
- Easy extensibility via the ModelRegistry
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel

from ascent_platform.config.runtime import get_runtime_settings
from ascent_platform.llm.clients.azure_chat import OmodelsClient
from ascent_platform.llm.clients.base import BaseLLMClient
from ascent_platform.llm.clients.gemini import GeminiClient
from ascent_platform.llm.credentials import gemini_api_key

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Configuration for a single LLM model.

    Attributes:
        client_class: The client class to instantiate (must inherit from BaseLLMClient).
        params: Default parameters passed to the client's send() method.
        api_key_env: Optional environment variable name for the API key.
    """

    client_class: Type[BaseLLMClient]
    params: Dict[str, Any] = field(default_factory=dict)
    api_key_env: Optional[str] = None


class ModelRegistry:
    """
    A registry for managing LLM model configurations.

    Supports runtime registration of new models without code changes to the router.

    Example:
        registry = ModelRegistry()
        registry.register("my-custom-model", ModelConfig(
            client_class=GeminiClient,
            params={"thinking_budget": 8192},
            api_key_env="MY_CUSTOM_API_KEY"
        ))
    """

    def __init__(self) -> None:
        self._configs: Dict[str, ModelConfig] = {}

    def register(self, model_name: str, config: ModelConfig) -> None:
        """Register a new model configuration."""
        self._configs[model_name] = config
        logger.debug(f"Registered model: {model_name}")

    def get(self, model_name: str) -> ModelConfig:
        """Get configuration for a model. Raises KeyError if not found."""
        if model_name not in self._configs:
            raise KeyError(f"Model '{model_name}' is not registered.")
        return self._configs[model_name]

    def has(self, model_name: str) -> bool:
        """Check if a model is registered."""
        return model_name in self._configs

    def list_models(self) -> list[str]:
        """Return a list of all registered model names."""
        return list(self._configs.keys())


# Default registry with pre-configured models
DEFAULT_REGISTRY = ModelRegistry()

# Gemini models
DEFAULT_REGISTRY.register(
    "gemini-2.5-flash-lite",
    ModelConfig(
        client_class=GeminiClient,
        params={},
        api_key_env="GEMINI_API_KEY",
    ),
)
DEFAULT_REGISTRY.register(
    "gemini-flash-lite-latest",
    ModelConfig(
        client_class=GeminiClient,
        params={},
        api_key_env="GEMINI_API_KEY",
    ),
)
DEFAULT_REGISTRY.register(
    "gemini-3-flash-preview",
    ModelConfig(
        client_class=GeminiClient,
        params={"thinkingLevel": "HIGH"},
        api_key_env="GEMINI_API_KEY",
    ),
)
DEFAULT_REGISTRY.register(
    "gemini-3.6-flash",
    ModelConfig(
        client_class=GeminiClient,
        params={"thinkingLevel": "HIGH"},
        api_key_env="GEMINI_API_KEY",
    ),
)
DEFAULT_REGISTRY.register(
    "gemini-3.7-flash",
    ModelConfig(
        client_class=GeminiClient,
        params={"thinkingLevel": "HIGH"},
        api_key_env="GEMINI_API_KEY",
    ),
)
DEFAULT_REGISTRY.register(
    "gemini-3.8-flash",
    ModelConfig(
        client_class=GeminiClient,
        params={"thinkingLevel": "HIGH"},
        api_key_env="GEMINI_API_KEY",
    ),
)
DEFAULT_REGISTRY.register(
    "gemini-3.1-flash-lite-preview",
    ModelConfig(
        client_class=GeminiClient,
        params={"thinkingLevel": "HIGH"},
        api_key_env="GEMINI_API_KEY",
    ),
)
DEFAULT_REGISTRY.register(
    "gemini-flash-latest",
    ModelConfig(
        client_class=GeminiClient,
        params={},
        api_key_env="GEMINI_API_KEY",
    ),
)
DEFAULT_REGISTRY.register(
    "gemini-3-pro-preview",
    ModelConfig(
        client_class=GeminiClient,
        params={"thinkingLevel": "HIGH"},
        api_key_env="GEMINI_API_KEY",
    ),
)
DEFAULT_REGISTRY.register(
    "gemini-3.1-pro-preview",
    ModelConfig(
        client_class=GeminiClient,
        params={"thinkingLevel": "HIGH"},
    ),
)

# Azure OpenAI models
DEFAULT_REGISTRY.register(
    "gpt-5_ascent",
    ModelConfig(
        client_class=OmodelsClient,
        params={},
    ),
)
DEFAULT_REGISTRY.register(
    "gpt-5-mini_ascent",
    ModelConfig(
        client_class=OmodelsClient,
        params={},
    ),
)
DEFAULT_REGISTRY.register(
    "o4-mini_ascent",
    ModelConfig(
        client_class=OmodelsClient,
        params={},
    ),
)


# Process-wide cap on concurrent LLM calls (per event loop, so tests with
# their own loops don't collide). Tunable via LLM_MAX_CONCURRENT_CALLS.
_LLM_MAX_CONCURRENT_CALLS = get_runtime_settings().LLM_MAX_CONCURRENT_CALLS
_llm_semaphores: "Dict[Any, asyncio.Semaphore]" = {}


def _get_llm_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    semaphore = _llm_semaphores.get(loop)
    if semaphore is None:
        semaphore = asyncio.Semaphore(max(1, _LLM_MAX_CONCURRENT_CALLS))
        _llm_semaphores[loop] = semaphore
    return semaphore


class LLMRouter:
    """
    Routes LLM requests to appropriate providers with automatic fallback.

    Features:
        - Lazy client instantiation with caching
        - Automatic fallback to a secondary model on failure
        - Registry-based configuration for easy extensibility
        - Async interface with thread pool execution for sync clients

    Example:
        router = LLMRouter(fallback_model_name="gemini-2.5-flash")
        response = await router.send_message("Hello!", model_name="gemini-2.5-pro")
    """

    def __init__(
        self,
        fallback_model_name: str,
        *,
        gemini_api_key: Optional[str] = None,
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        """
        Initialize the LLM router.

        Args:
            fallback_model_name: Model to use when the primary model fails.
            gemini_api_key: Optional API key for Gemini models. If not provided,
                           falls back to GEMINI_API_KEY or GOOGLE_API_KEY env vars.
            registry: Optional custom model registry. Uses DEFAULT_REGISTRY if not provided.
        """
        self._registry = registry or DEFAULT_REGISTRY
        self._clients: Dict[str, BaseLLMClient] = {}
        self._gemini_api_key = gemini_api_key
        self.fallback_model_name = fallback_model_name

        if not self._registry.has(fallback_model_name):
            raise ValueError(f"Fallback model '{fallback_model_name}' is not registered.")

        logger.info(f"LLMRouter initialized with fallback: {fallback_model_name}")

    def _resolve_api_key(self, config: ModelConfig) -> Optional[str]:
        """Resolve API key from explicit value, config env var, or common env vars."""
        if self._gemini_api_key and config.client_class is GeminiClient:
            return self._gemini_api_key

        if config.api_key_env:
            key = os.getenv(config.api_key_env)
            if key:
                return key

        # Fallback to the shared resolver for Gemini, which owns the
        # GEMINI_API_KEY / GOOGLE_API_KEY precedence for every caller.
        if config.client_class is GeminiClient:
            return gemini_api_key()

        return None

    def _create_client(self, model_name: str) -> BaseLLMClient:
        """Create a new client instance for the given model."""
        config = self._registry.get(model_name)

        init_kwargs: Dict[str, Any] = {"model_name": model_name}

        # Handle API key for clients that need it
        api_key = self._resolve_api_key(config)
        if api_key and config.client_class is GeminiClient:
            init_kwargs["api_key"] = api_key

        client = config.client_class(**init_kwargs)
        logger.info(f"Created client for model: {model_name}")
        return client

    def _get_client(self, model_name: str) -> BaseLLMClient:
        """Get or create a cached client instance for the given model."""
        if model_name not in self._clients:
            self._clients[model_name] = self._create_client(model_name)
        return self._clients[model_name]

    def _get_params(self, model_name: str) -> Dict[str, Any]:
        """Get the default parameters for a model."""
        config = self._registry.get(model_name)
        return config.params.copy()

    async def send_message(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        """
        Send a prompt to the specified model asynchronously.

        Falls back to the configured fallback model if the primary model fails.

        Args:
            prompt: The prompt to send to the model.
            model_name: The name of the model to use. If None, uses the fallback model.
            schema: Optional Pydantic model for structured output.
            **kwargs: Additional parameters passed to the client's send method.

        Returns:
            The model's response as a string or dict (if schema is provided).

        Raises:
            ConnectionError: If both primary and fallback models fail.
        """
        # Use fallback model if no model_name is provided
        if model_name is None:
            model_name = self.fallback_model_name

        async def _execute(name: str) -> Union[str, Dict[str, Any]]:
            client = self._get_client(name)
            params = self._get_params(name)
            params.update(kwargs)

            # Backpressure: bound concurrent in-flight LLM calls. Unbounded
            # fan-out trips provider rate limits. Safe only because each client
            # call is itself timeout-bounded.
            #
            # No run_in_executor any more: the clients are natively async, so
            # this no longer occupies a DEFAULT-pool worker for the duration of
            # the call plus up to 15s of retry backoff. That pool is shared with
            # every other blocking call in the process, and the LLM path has no
            # bulkhead of its own.
            async with _get_llm_semaphore():
                return await client.send(prompt, schema=schema, **params)

        try:
            logger.info(f"Sending prompt to primary model: {model_name}")
            response = await _execute(model_name)
            logger.info(f"Received response from {model_name}")
            return response

        except Exception as primary_error:
            logger.warning(f"Primary model '{model_name}' failed: {primary_error}", exc_info=True)

            if model_name == self.fallback_model_name:
                raise ConnectionError(f"Primary model '{model_name}' failed and is also the fallback.") from primary_error

            logger.info(f"Falling back to: {self.fallback_model_name}")

            try:
                response = await _execute(self.fallback_model_name)
                logger.info(f"Received response from fallback: {self.fallback_model_name}")
                return response

            except Exception as fallback_error:
                logger.error(f"Fallback model '{self.fallback_model_name}' also failed: {fallback_error}", exc_info=True)
                raise ConnectionError("Both primary and fallback LLM services failed.") from fallback_error

    def clear_cache(self, model_name: Optional[str] = None) -> None:
        """
        Clear cached client instances.

        Args:
            model_name: If provided, only clear the cache for this model.
                       Otherwise, clear all cached clients.
        """
        if model_name:
            self._clients.pop(model_name, None)
            logger.debug(f"Cleared cache for model: {model_name}")
        else:
            self._clients.clear()
            logger.debug("Cleared all client caches")
