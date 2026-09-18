"""Remote (API-backed) embedders.

Ports the Azure OpenAI and Gemini branches of the old ``EmbeddingsGenerator``.
Model ids, dimensions, and Gemini's ``task_type`` are preserved verbatim. The
Gemini backend keeps the old AWS-credential-refresh-on-expiry retry path used
when running under an ECS task with rotating credentials.
"""

from __future__ import annotations

import asyncio
import logging
import os

from ascent_medical_coder.core.settings import get_settings

logger = logging.getLogger(__name__)

_AZURE_EMBED_DIM = 3072

GEMINI_EMBEDDING_SIZE = 768


class GeminiEmbedder:
    """Google Gemini embedder (dim 768, ``SEMANTIC_SIMILARITY`` task type).

    Model from ``settings.llm.gemini_embed_model`` unless overridden.
    """

    def __init__(self, model_name: str | None = None) -> None:
        from google import genai

        s = get_settings().llm
        resolved_api_key = s.gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or s.gemini_embed_model
        self.client = genai.Client(api_key=resolved_api_key)

    @staticmethod
    def _set_aws_credentials() -> None:
        import boto3
        from google.auth.aws import environment_vars

        task_credentials = boto3.Session().get_credentials().get_frozen_credentials()
        os.environ[environment_vars.AWS_ACCESS_KEY_ID] = task_credentials.access_key
        os.environ[environment_vars.AWS_SECRET_ACCESS_KEY] = task_credentials.secret_key
        if task_credentials.token:
            os.environ[environment_vars.AWS_SESSION_TOKEN] = task_credentials.token

    def _run(self, contents):
        from google.genai import types

        return self.client.models.embed_content(
            model=self.model_name,
            contents=contents,
            config=types.EmbedContentConfig(
                output_dimensionality=GEMINI_EMBEDDING_SIZE,
                task_type="SEMANTIC_SIMILARITY",
            ),
        )

    async def _embed(self, contents):
        from google.auth.aws import environment_vars

        try:
            return await asyncio.to_thread(self._run, contents)
        except Exception as e:
            logger.error("Error generating embedding with Gemini; credentials expired: %s", e)
            for var in (
                environment_vars.AWS_ACCESS_KEY_ID,
                environment_vars.AWS_SECRET_ACCESS_KEY,
                environment_vars.AWS_SESSION_TOKEN,
            ):
                os.environ.pop(var, None)
            self._set_aws_credentials()
            return await asyncio.to_thread(self._run, contents)

    async def encode(self, text: str) -> list[float]:
        response = await self._embed(text)
        return response.embeddings[0].values

    async def encode_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self._embed(texts)
        return [list(e.values) for e in response.embeddings]

    def dimension(self) -> int:
        return GEMINI_EMBEDDING_SIZE
