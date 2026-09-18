"""Azure OpenAI (GPT) connector.

Thin transport over ``AsyncAzureOpenAI``. Ported from the old
``OpenAIChatAssistant`` (``assistants/gpt_assistant.py``) as wired by
``ChatGPTHandler``: same GPT-5 vs classic request branches, same seed/penalties,
same JSON response format. Endpoint/key/version come from settings (chat vars).
"""

from __future__ import annotations

import logging

from openai import AsyncAzureOpenAI

from ascent_medical_coder.core.settings import get_settings

logger = logging.getLogger(__name__)


class OpenAIConnector:
    """Transport-only Azure OpenAI connector implementing :class:`LLMConnector`."""

    def __init__(
        self,
        endpoint: str | None = None,
        key: str | None = None,
        version: str | None = None,
        model_name: str | None = None,
    ) -> None:
        settings = get_settings().llm
        self.client = AsyncAzureOpenAI(
            azure_endpoint=endpoint or settings.azure_openai_chat_endpoint,
            api_key=key or settings.azure_openai_chat_key,
            api_version=version or settings.azure_openai_chat_api_version,
        )
        self.model_name = model_name or settings.openai_chat_model
        logger.info("OpenAIConnector initialized with model: %s", self.model_name)

    async def complete_json(self, prompt: str) -> str:
        """Generate a JSON response, mirroring ``generate_response(prompt, json_format=True)``."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt},
        ]

        if "5" in self.model_name:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                seed=1234,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                response_format={"type": "json_object"},
            )
        else:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=4096,
                top_p=0.7,
                seed=1234,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                response_format={"type": "json_object"},
            )

        return response.choices[0].message.content
