import json
import logging
from typing import Any, Dict, Optional, Type, Union

from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from pydantic import BaseModel, ValidationError

from ascent_platform.config.runtime import get_runtime_settings
from ascent_platform.llm.credentials import gemini_api_key, google_cloud_project

from .base import BaseLLMClient

# Use a module-level logger
logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    """
    Client for Google Gemini models via the google-genai SDK.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model_name: Optional[str] = None,
        # Bounds a stalled call, which otherwise hangs the pipeline.
        # 120s comfortably covers a healthy flash generate; env-tunable.
        default_timeout: float | None = None,
        use_vertex_ai: bool = False,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # None means "use the configured default" throughout: the values live in
        # RuntimeSettings, not in default arguments reading os.environ.
        settings = get_runtime_settings()
        self.model_name = model_name or settings.GEMINI_DEFAULT_MODEL_NAME
        self.default_timeout = settings.GEMINI_HTTP_TIMEOUT_SECONDS if default_timeout is None else default_timeout
        location = location or settings.VERTEX_AI_LOCATION

        logger.info(f"Initializing GeminiClient for model='{self.model_name}' with timeout={self.default_timeout}s")

        if use_vertex_ai:
            self._init_vertex_ai_client(project_id, location)
        else:
            self._init_google_ai_client(api_key)

        logger.info("GeminiClient initialization complete.")

    def _init_vertex_ai_client(self, project_id: Optional[str], location: str) -> None:
        """Initializes the client for Vertex AI."""
        logger.info("Using Vertex AI configuration.")
        resolved_project_id = google_cloud_project(project_id)
        if not resolved_project_id:
            raise ValueError("project_id is required for Vertex AI. Pass it as an argument or set the GOOGLE_CLOUD_PROJECT environment variable.")
        logger.debug(f"Vertex AI project='{resolved_project_id}', location='{location}'")

        # Configure HTTP timeout to prevent server disconnects on long requests
        # timeout is in milliseconds for HttpOptions
        timeout_ms = int(self.default_timeout * 1000)
        http_options = HttpOptions(timeout=timeout_ms)

        self.client = genai.Client(
            vertexai=True,
            project=resolved_project_id,
            location=location,
            http_options=http_options,
        )

    def _init_google_ai_client(self, api_key: Optional[str]) -> None:
        """Initializes the client for the Google AI (Developer) API."""
        logger.info("Using Google AI (Developer API) configuration.")
        # Was GOOGLE_API_KEY-first here and GEMINI_API_KEY-first in the router,
        # for one credential. The shared resolver settles it: GEMINI_API_KEY wins.
        resolved_api_key = gemini_api_key(api_key)
        if not resolved_api_key:
            raise ValueError("API key is required. Pass it as an argument or set the GOOGLE_API_KEY/GEMINI_API_KEY environment variable.")

        # Configure HTTP timeout to prevent server disconnects on long requests
        # timeout is in milliseconds for HttpOptions
        timeout_ms = int(self.default_timeout * 1000)
        http_options = HttpOptions(timeout=timeout_ms)

        self.client = genai.Client(api_key=resolved_api_key, http_options=http_options)
        logger.debug(f"Client initialized using API key with timeout={self.default_timeout}s ({timeout_ms}ms).")

    def _is_pro_model(self) -> bool:
        """Checks if the model name suggests it supports `include_thoughts`."""
        return "pro" in self.model_name.lower() or "thinking" in self.model_name.lower()

    def _prepare_config(
        self,
        schema: Optional[Type[BaseModel]] = None,
        thinking_budget: Optional[int] = None,
        thinking_level: Optional[str] = None,
        include_thoughts: bool = False,
        **kwargs: Any,
    ) -> types.GenerateContentConfig:
        """Prepares the GenerateContentConfig object from various parameters."""

        # --- FIX: Sanitize kwargs ---
        # If 'thinkingLevel' (camelCase) is passed, it lands in kwargs.
        # We must extract it and remove it from kwargs to prevent Pydantic validation errors.
        if "thinkingLevel" in kwargs:
            val = kwargs.pop("thinkingLevel")
            if thinking_level is None:
                thinking_level = val

        if "thinkingBudget" in kwargs:
            val = kwargs.pop("thinkingBudget")
            if thinking_budget is None:
                thinking_budget = val
        # ----------------------------

        config_dict = {
            "temperature": 1,
            "max_output_tokens": 65536,
            **kwargs,
        }

        # Apply schema if provided
        if schema:
            logger.info(f"Applying response schema: {schema.__name__}")
            config_dict["response_schema"] = schema
            config_dict["response_mime_type"] = "application/json"

        # Apply thinking configuration
        thinking_config_args = {}

        # 1. Handle include_thoughts
        if include_thoughts:
            logger.info("Enabling 'include_thoughts'.")
            thinking_config_args["include_thoughts"] = True

        # 2. Handle Gemini 2.5 Flash Thinking (Token Budget) - deprecated in newer SDK versions
        if thinking_budget is not None:
            logger.debug(f"thinking_budget parameter provided ({thinking_budget}), but may not be supported by current SDK version.")
            # Skip thinking_budget as it's not supported in google-genai >= 1.50

        # 3. Handle Gemini 3.0 Pro (Thinking Level)
        if thinking_level is not None:
            logger.info(f"Setting 'thinking_level' to {thinking_level}.")
            thinking_config_args["thinking_level"] = thinking_level

        # Construct ThinkingConfig if any parameters were set
        if thinking_config_args:
            try:
                # We strictly pass only valid params to ThinkingConfig
                config_dict["thinking_config"] = types.ThinkingConfig(**thinking_config_args)
            except ValidationError:
                logger.warning(f"ThinkingConfig not supported with args: {thinking_config_args}. Proceeding without thinking configuration.")
                # Continue without thinking config if it fails

        try:
            # Now config_dict is clean of 'thinkingLevel', so this won't crash
            config = types.GenerateContentConfig(**config_dict)

            # Logging helper
            config_for_logging = config.model_dump(exclude_none=True)
            if "response_schema" in config_for_logging:
                schema_repr = getattr(config_for_logging["response_schema"], "__name__", "UnknownSchema")
                config_for_logging["response_schema"] = f"<Pydantic Schema: {schema_repr}>"
            logger.debug(f"Final GenerateContentConfig:\n{json.dumps(config_for_logging, indent=2)}")

            return config
        except Exception as e:
            logger.error(f"Failed to construct GenerateContentConfig: {e}", exc_info=True)
            raise

    async def _execute_non_streaming_request(self, prompt: str, config: types.GenerateContentConfig) -> genai.types.GenerateContentResponse:
        """Executes a non-streaming API call and returns the raw response.

        ``client.aio`` is the SDK's async namespace. The sync equivalent used to
        run in the default thread pool via the router; awaiting the SDK removes
        the executor hop and the thread it occupied for the whole call.
        """
        logger.info(f"Sending non-streaming request to model: '{self.model_name}'")
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            usage = self.get_usage_metadata(response)
            if usage:
                logger.info(
                    f"Token usage - Prompt: {usage.get('prompt_tokens', 'N/A')}, "
                    f"Output: {usage.get('output_tokens', 'N/A')}, "
                    f"Thinking: {usage.get('thinking_tokens', 'N/A')}"
                )
            return response
        except Exception as e:
            logger.error(f"Gemini API call failed: {type(e).__name__}: {e}", exc_info=True)
            raise

    def _process_response(
        self, response: genai.types.GenerateContentResponse, schema: Optional[Type[BaseModel]] = None, include_thoughts: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """Processes a raw API response into the desired user-facing format."""
        if schema and hasattr(response, "parsed") and response.parsed:
            logger.info(f"Response successfully parsed into schema: {schema.__name__}")
            return response.parsed.model_dump()

        if include_thoughts and self._is_pro_model():
            logger.info("Extracting thinking content from Pro/Thinking model response.")
            return self._extract_thinking_from_response(response)

        logger.debug("Returning raw text from response.")
        return response.text

    async def _send(
        self,
        prompt: str,
        *,
        schema: Optional[Type[BaseModel]] = None,
        timeout: Optional[float] = None,
        thinking_budget: Optional[int] = None,
        thinking_level: Optional[str] = None,
        include_thoughts: bool = False,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        """Sends a prompt to the Gemini API and returns the processed response."""
        logger.debug(f"Received _send request for model '{self.model_name}'.")
        config = self._prepare_config(
            schema=schema, thinking_budget=thinking_budget, thinking_level=thinking_level, include_thoughts=include_thoughts, **kwargs
        )
        raw_response = await self._execute_non_streaming_request(prompt, config)
        return self._process_response(raw_response, schema, include_thoughts)

    async def send_with_usage(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Sends a prompt and returns both the processed response and usage metadata.
        """
        logger.info(f"Executing send_with_usage for model '{self.model_name}'.")
        schema = kwargs.get("schema")
        include_thoughts = kwargs.get("include_thoughts", False)

        config = self._prepare_config(**kwargs)
        raw_response = await self._execute_non_streaming_request(prompt, config)
        processed_response = self._process_response(raw_response, schema, include_thoughts)
        usage = self.get_usage_metadata(raw_response)

        return {"response": processed_response, "usage": usage}

    def _extract_thinking_from_response(self, response: genai.types.GenerateContentResponse) -> Dict[str, str]:
        """Extracts thinking and regular content from a response."""
        thinking_parts, response_parts = [], []

        # Guard against empty or invalid response structures
        if not (response.candidates and response.candidates[0].content and response.candidates[0].content.parts):
            logger.warning("Response has no content parts to extract thinking from. Returning full text.")
            return {"thinking": "", "response": response.text, "full_text": response.text}

        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                thinking_parts.append(part.text)
            else:
                response_parts.append(part.text)

        thinking_content = "".join(thinking_parts)
        response_content = "".join(response_parts)

        if not thinking_content:
            logger.warning("`include_thoughts=True` but no thought parts were found.")
            # If no thinking is found, the response content is the full text.
            response_content = response.text

        return {
            "thinking": thinking_content,
            "response": response_content,
            "full_text": response.text,
        }

    async def stream_thinking(
        self,
        prompt: str,
        *,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """
        Streams a response with the thinking process visible.
        """
        logger.info(f"Starting stream_thinking for model='{self.model_name}'")

        # `include_thoughts` is mandatory for this method
        config = self._prepare_config(include_thoughts=True, **kwargs)

        response_chunks = await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        thinking_buffer, response_buffer = [], []
        is_thinking_phase = True

        if show_progress:
            print("🤔 Model is thinking...\n" + "-" * 50)

        async for chunk in response_chunks:
            # Safely access parts to avoid errors on empty chunks
            parts = chunk.candidates[0].content.parts if (chunk.candidates and chunk.candidates[0].content) else []

            for part in parts:
                if not part.text:
                    continue

                if part.thought:
                    thinking_buffer.append(part.text)
                    if show_progress:
                        print(part.text, end="", flush=True)
                else:
                    # First time we see a non-thought part, print the separator
                    if show_progress and is_thinking_phase:
                        print("\n" + "-" * 20 + " RESPONSE " + "-" * 20)
                        is_thinking_phase = False

                    response_buffer.append(part.text)
                    if show_progress:
                        print(part.text, end="", flush=True)

        if show_progress:
            print("\n" + "-" * 50 + "\n✅ Streaming complete!")

        full_thinking = "".join(thinking_buffer).strip()
        full_response = "".join(response_buffer).strip()

        return {"thinking": full_thinking, "response": full_response, "full_text": (full_thinking + " " + full_response).strip()}

    def get_usage_metadata(self, response: genai.types.GenerateContentResponse) -> Optional[Dict[str, int]]:
        """Extracts usage metadata, including thinking token counts."""
        if not hasattr(response, "usage_metadata"):
            logger.warning("Response object lacks 'usage_metadata'.")
            return None

        metadata = response.usage_metadata
        usage_info = {
            "prompt_tokens": getattr(metadata, "prompt_token_count", 0),
            "output_tokens": getattr(metadata, "candidates_token_count", 0),
            "total_tokens": getattr(metadata, "total_token_count", 0),
        }
        if hasattr(metadata, "thoughts_token_count"):
            usage_info["thinking_tokens"] = metadata.thoughts_token_count

        return usage_info
