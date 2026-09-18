"""Gemini (Google) connector.

Thin transport over the google-genai SDK, implementing both
:class:`~ascent_medical_coder.connectors.llm.base.LLMConnector` and
:class:`~ascent_medical_coder.connectors.llm.base.GroundedSearchConnector`.

Model names are never hardcoded here — defaults come from
``settings.llm.gemini_model`` / ``settings.llm.grounded_search_model``, so the
deployed models are switchable purely via environment. Defaults to the
Developer API (``genai.Client(api_key=...)``); the Vertex AI path is kept
optional and off by default.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from google import genai
from google.genai import types

from ascent_medical_coder.core.settings import get_settings

logger = logging.getLogger(__name__)

# Per-family generation configs, selected by model-name marker; newer families inherit the "3" config.
_FAMILY_GENERATION_CONFIG: dict[str, dict[str, Any]] = {
    "legacy": {
        "temperature": 0.1,
        "top_p": 0.95,
        "response_mime_type": "application/json",
    },
    "3": {
        "temperature": 1,
        "top_p": 0.9,
        "response_mime_type": "application/json",
        "thinking_level": "high",
    },
}


def _select_generation_config(model_name: str) -> dict[str, Any]:
    """Return the generation config for a model's family ("2.x" legacy vs 3+)."""
    if "2." in model_name:
        return _FAMILY_GENERATION_CONFIG["legacy"].copy()
    return _FAMILY_GENERATION_CONFIG["3"].copy()


class GeminiConnector:
    """Transport-only Gemini connector (LLMConnector + GroundedSearchConnector)."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        use_vertex_ai: bool = False,
        project_id: str | None = None,
        location: str = "us-central1",
    ) -> None:
        self.use_vertex_ai = use_vertex_ai
        self.location = location
        settings = get_settings().llm
        self.default_model = settings.gemini_model
        self.default_grounded_model = settings.grounded_search_model
        resolved_project = project_id or settings.gemini_project

        if use_vertex_ai:
            try:
                self._init_vertex_ai_client(resolved_project, location)
            except Exception as e:
                logger.warning(
                    "Vertex AI initialization failed: %s. Falling back to Google AI "
                    "(Developer API). To use Vertex AI, pass project_id or set "
                    "GOOGLE_CLOUD_PROJECT.",
                    e,
                )
                self.use_vertex_ai = False
                self._init_google_ai_client(api_key or settings.gemini_api_key)
        else:
            self._init_google_ai_client(api_key or settings.gemini_api_key)

        logger.info("GeminiConnector initialization complete (use_vertex_ai=%s).", self.use_vertex_ai)

    def _init_google_ai_client(self, api_key: str | None) -> None:
        """Initialize the client for the Google AI (Developer) API."""
        resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "API key is required. Pass it as an argument or set the "
                "GOOGLE_API_KEY/GEMINI_API_KEY environment variable."
            )
        self.client = genai.Client(api_key=resolved_api_key)
        logger.debug("Gemini client initialized using API key.")

    def _init_vertex_ai_client(self, project_id: str | None, location: str) -> None:
        """Initialize the client for Vertex AI."""
        resolved_project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not resolved_project_id:
            raise ValueError(
                "project_id is required for Vertex AI. Pass it as an argument or "
                "set the GOOGLE_CLOUD_PROJECT environment variable."
            )
        self.client = genai.Client(vertexai=True, project=resolved_project_id, location=location)

    async def complete_json(self, prompt: str, model_name: str | None = None) -> str:
        """Generate a JSON response (model defaults to ``settings.llm.gemini_model``)."""
        model_key = model_name or self.default_model
        generation_config = _select_generation_config(model_key)

        try:
            config_kwargs: dict[str, Any] = {
                "temperature": generation_config.get("temperature", 0.1),
                "top_p": generation_config.get("top_p", 0.95),
                "response_mime_type": generation_config.get("response_mime_type", "application/json"),
            }
            # thinking_level must ride in ThinkingConfig; the SDK rejects it as a bare kwarg.
            if thinking_level := generation_config.get("thinking_level"):
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
            config = types.GenerateContentConfig(**config_kwargs)
            response = await self.client.aio.models.generate_content(
                model=model_key,
                contents=prompt,
                config=config,
            )
            return self._process_response(response)
        except Exception as e:
            logger.error("Async generation failed via Google AI client: %s", e)
            raise

    async def grounded_search(
        self,
        prompt: str,
        schema: Any = None,
        model_name: str | None = None,
    ) -> dict[str, Any] | str:
        """Generate with Google Search grounding (model defaults to
        ``settings.llm.grounded_search_model``)."""
        model_key = model_name or self.default_grounded_model

        try:
            grounding_tool = types.Tool(google_search=types.GoogleSearch())

            # Determinism-sensitive: use the dedicated low grounded temperature, not the chat-family config.
            config_dict: dict[str, Any] = {
                "temperature": get_settings().llm.grounded_search_temperature,
                "max_output_tokens": 65536,
            }
            if schema:
                logger.info("Applying response schema: %s", schema.__name__)
                config_dict["response_schema"] = schema
            config_dict["tools"] = [grounding_tool]

            config = types.GenerateContentConfig(**config_dict)

            response = await self.client.aio.models.generate_content(
                model=model_key,
                contents=prompt,
                config=config,
            )
            return self._process_response(response, schema=schema)
        except Exception as e:
            logger.warning(
                "Google Search grounding failed (%s). Falling back to regular generation "
                "without grounding.",
                e,
            )
            return await self.complete_json(prompt, model_key)

    def _process_response(
        self,
        response: types.GenerateContentResponse,
        schema: Any = None,
    ) -> dict[str, Any] | str:
        """Process a raw API response into the desired user-facing format."""
        if schema and hasattr(response, "parsed") and response.parsed:
            logger.info("Response successfully parsed into schema: %s", schema.__name__)
            return response.parsed.model_dump()
        logger.debug("Returning raw text from response.")
        return response.text
