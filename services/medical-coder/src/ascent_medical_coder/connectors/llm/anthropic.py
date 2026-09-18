"""Anthropic (Claude) connector.

Ships as an interface (:class:`AnthropicConnector`) plus a Bedrock
implementation. The Bedrock path preserves the exact request/response behavior
of the old ``BedrockAnthropicHaikuAssistant`` so filter outputs stay identical.

Open-sourcing note: to drop AWS, add a ``DirectAnthropicConnector`` implementing
the same interface with the first-party SDK::

    from anthropic import Anthropic  # or AnthropicBedrockMantle
    client = Anthropic()
    msg = client.messages.create(
        model="claude-haiku-4-5", max_tokens=65536, system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(b.text for b in msg.content if b.type == "text")

No service-layer change is needed for the swap — only the factory binding.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Protocol

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from ascent_medical_coder.core.settings import get_settings

logger = logging.getLogger(__name__)

ANTHROPIC_VERSION = "bedrock-2023-05-31"
# Claude Haiku 4.5's output cap is 64000; the legacy 2**16 (65536) exceeded it.
DEFAULT_MAX_TOKENS = 64000


class AnthropicConnector(Protocol):
    """Transport-only Claude connector — see :class:`LLMConnector`."""

    async def complete_json(self, prompt: str) -> str: ...


class BedrockAnthropicConnector:
    """Claude Haiku via Amazon Bedrock ``invoke_model``.

    Uses ambient AWS credentials/region (no explicit config), matching current
    behavior. Returns the first ``{...}`` JSON span found in the completion.
    """

    def __init__(
        self,
        model_id: str | None = None,
        *,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> None:
        self.model_id = model_id or get_settings().llm.anthropic_model_id
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self._client = boto3.client(service_name="bedrock-runtime")

    async def complete_json(self, prompt: str, *, max_tokens: int | None = None) -> str:
        payload: dict = {
            "anthropic_version": ANTHROPIC_VERSION,
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p

        body_json = json.dumps(payload)
        raw = await self._invoke(body_json)

        text = ""
        content_list = raw.get("content")
        if isinstance(content_list, list) and content_list:
            first = content_list[0]
            if isinstance(first, dict):
                text = first.get("text", "") or first.get("content", "") or ""

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON found in response.")
        return match.group(0)

    async def _invoke(self, body_json: str) -> dict:
        def _sync() -> dict:
            try:
                response = self._client.invoke_model(
                    body=body_json,
                    modelId=self.model_id,
                    contentType="application/json",
                )
            except (ClientError, BotoCoreError) as e:
                logger.error("Bedrock invoke failed: %s", e)
                raise
            return json.loads(response.get("body").read())

        return await asyncio.to_thread(_sync)
