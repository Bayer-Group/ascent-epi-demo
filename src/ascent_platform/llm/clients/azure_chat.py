"""
Azure chat clients for OpenAI-based models via AzureOpenAI, with BaseLLMClient for retries.
"""

import json
import logging
from typing import Any, Dict, Optional, Type, Union

from openai import AsyncAzureOpenAI
from pydantic import BaseModel

from .base import BaseLLMClient

# Set up logger
logger = logging.getLogger(__name__)


def _resolve_setting(name: str, generic: str) -> Optional[str]:
    """Resolve a per-deployment setting, falling back to the generic one.

    The fallback is what makes a single-provider install work: these clients
    name specific Azure deployments (OPENAI_EAST_US_2_BASE and friends), which
    is right for a deployment that runs several, and a dead end for anyone who
    configured one Azure endpoint and expected the product to use it.

    Both names are read off Settings rather than the environment. They used to
    be fetched with a bare os.getenv, which is how OPENAI_EAST_US_2_BASE came to
    be required by the code while appearing in no Settings class and no
    .env.template -- undiscoverable until the first non-OMOP question failed.
    """
    from ascent_platform.config.runtime import get_runtime_settings

    settings = get_runtime_settings()
    for key in (name, generic):
        value = getattr(settings, key, None)
        if value:
            return value
    return None


class AzureOpenAIClient(BaseLLMClient):
    """
    Base class for Azure OpenAI chat completion clients.
    Supports structured output via .parse() and reasoning effort.
    """

    def __init__(
        self,
        *,
        endpoint_env: str,
        key_env: str,
        api_version: str,
        model_name: str,
        instructions: str = "You are a helpful assistant",
        reasoning_effort: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        logger.info(f"Initializing AzureOpenAIClient for model='{model_name}'")

        endpoint = _resolve_setting(endpoint_env, "OPENAI_API_BASE")
        if not endpoint:
            logger.error("Neither '%s' nor OPENAI_API_BASE is set.", endpoint_env)
            raise ValueError(
                f"No Azure OpenAI endpoint configured for this model. Set {endpoint_env}, "
                "or OPENAI_API_BASE to use one deployment for every Azure model."
            )
        logger.debug(f"Resolved Azure endpoint from '{endpoint_env}': {endpoint}")

        api_key = _resolve_setting(key_env, "OPENAI_API_KEY")
        if not api_key:
            logger.error("Neither '%s' nor OPENAI_API_KEY is set.", key_env)
            raise ValueError(
                f"No Azure OpenAI API key configured for this model. Set {key_env}, or OPENAI_API_KEY to use one key for every Azure model."
            )
        logger.debug(f"API key for '{key_env}' successfully resolved (value not logged for security).")

        # Initialize client
        try:
            self.client = AsyncAzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )
            logger.info(f"AsyncAzureOpenAI client initialized | API Version: {api_version}")
        except Exception as e:
            logger.error(f"Failed to create AsyncAzureOpenAI client: {type(e).__name__}: {e}", exc_info=True)
            raise

        self.model_name = model_name
        self.instructions = instructions
        self.reasoning_effort = reasoning_effort

        logger.debug(f"System instructions set: {repr(self.instructions)}")
        if reasoning_effort:
            logger.info(f"Reasoning effort configured: {reasoning_effort}")
        else:
            logger.debug("No reasoning effort set.")

        logger.info("AzureOpenAIClient initialization completed.")

    async def _send(
        self,
        prompt: str,
        *,
        schema: Optional[Union[str, Type[BaseModel]]] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        """
        Send a prompt to the Azure OpenAI model with optional schema enforcement.
        """
        logger.info(f"Entering _send() with model='{self.model_name}'")
        logger.debug(f"Input prompt length: {len(prompt)} characters")
        logger.debug(f"Prompt preview (first 200 chars): {repr(prompt[:200])}{'...' if len(prompt) > 200 else ''}")

        if schema:
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                logger.info(f"Structured output requested via Pydantic schema: {schema.__name__}")
            else:
                logger.info(f"Response format set to: {schema}")
        else:
            logger.debug("No schema provided; response will be unstructured.")

        # Build messages
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]
        logger.debug(f"Constructed {len(messages)} messages for API call.")

        # Build parameters
        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }
        logger.debug(f"Base params: model='{self.model_name}'")

        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
            logger.debug(f"Added reasoning_effort='{self.reasoning_effort}' to request.")

        if schema:
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                params["response_format"] = schema
                logger.debug(f"Pydantic model '{schema.__name__}' set as response_format.")
            else:
                params["response_format"] = {"type": schema}
                logger.debug(f"Response format set to type='{schema}'")

        # Add any additional kwargs (e.g., temperature, max_tokens)
        extra_params = {k: v for k, v in kwargs.items() if k not in params}
        if extra_params:
            params.update(extra_params)
            logger.debug(f"Applied additional parameters: {list(extra_params.keys())}")

        # Log full request (safe for logging)
        safe_params = params.copy()
        if "api_key" in safe_params:
            safe_params["api_key"] = "<redacted>"
        if "azure_endpoint" in safe_params:
            safe_params["azure_endpoint"] = "<redacted>"

        try:
            logger.debug("Final request payload (before API call):")
            logger.debug(json.dumps(safe_params, indent=2, default=str, ensure_ascii=False))
        except Exception:
            logger.debug("Could not serialize request payload for logging.")

        # Make API call
        logger.info("Sending request to Azure OpenAI API...")
        try:
            completion = await self.client.beta.chat.completions.parse(**params)
            logger.info("Received successful response from Azure OpenAI.")
        except Exception as e:
            logger.error(f"Azure OpenAI API call failed | Model: {self.model_name} | Error: {type(e).__name__}: {e}", exc_info=True)
            raise

        # Handle response
        if not completion.choices:
            logger.warning("No choices returned in completion response.")
            raise ValueError("No choices returned from the model.")

        choice = completion.choices[0]
        message = choice.message

        if hasattr(message, "parsed") and message.parsed:
            logger.info("Response was successfully parsed into structured format.")
            try:
                parsed_data = message.parsed.model_dump()
                logger.debug(f"Parsed response keys: {list(parsed_data.keys())}")
                json_output = message.parsed.model_dump_json(indent=2)
                logger.debug("Returning structured JSON response.")
                return json_output
            except Exception as e:
                logger.error(f"Failed to dump parsed response to JSON: {e}", exc_info=True)
                raise
        else:
            logger.warning("No parsed content found; falling back to message content.")
            content = message.content or ""
            logger.debug(f"Raw content length: {len(content)} | Preview: {repr(content[:200])}")
            return content


class OmodelsClient(AzureOpenAIClient):
    """
    Client for O3/O4 Mini models on Azure OpenAI.
    Uses high reasoning effort by default.
    """

    def __init__(
        self,
        instructions: str = "You are a helpful assistant",
        model_name: str = "o4-mini_ascent",
        reasoning_effort: str = "high",
        **kwargs: Any,
    ) -> None:
        logger.info("Initializing OmodelsClient")
        logger.debug(f"Model name: {model_name}, Reasoning effort: {reasoning_effort}")
        logger.debug(f"Custom instructions: {repr(instructions)}")

        super().__init__(
            endpoint_env="OPENAI_EAST_US_2_BASE",
            key_env="OPENAI_EAST_US_2_KEY",
            api_version="2024-12-01-preview",
            model_name=model_name,
            instructions=instructions,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )
        logger.info("OmodelsClient initialized successfully.")


class GPT4oMiniClient(AzureOpenAIClient):
    """
    Client for GPT4o Mini models on Azure OpenAI.
    """

    def __init__(
        self,
        instructions: str = "You are a helpful assistant",
        model_name: str = "gpt4o-mini_ascent",
        **kwargs: Any,
    ) -> None:
        logger.info("Initializing GPT4oMiniClient")
        logger.debug(f"Using model: {model_name}")
        logger.debug(f"Instructions: {repr(instructions)}")

        super().__init__(
            endpoint_env="OPENAI_EAST_US_2_BASE",
            key_env="OPENAI_EAST_US_2_KEY",
            api_version="2024-12-01-preview",
            model_name=model_name,
            instructions=instructions,
            reasoning_effort=None,  # GPT-4o Mini may not support this
            **kwargs,
        )
        logger.info("GPT4oMiniClient initialized successfully.")
