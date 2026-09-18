import asyncio
import json
import logging
import time
import warnings
from abc import abstractmethod
from datetime import datetime, timezone
from functools import cache, partial, wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Concatenate,
    Dict,
    Optional,
    ParamSpec,
    Type,
    TypeVar,
    Union,
)

import google.genai as genai
import google.genai.errors as genai_errors
from pydantic import BaseModel

# Suppress known google-genai bug: https://github.com/googleapis/python-genai/issues/1989
warnings.filterwarnings("ignore", message="Inheritance class AiohttpClientSession from ClientSession is discouraged")

import aiohttp
import boto3
import openai
import tiktoken
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ServiceRequestError, ServiceResponseError
from botocore.config import Config as BotoConfig
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)
from openai import AsyncAzureOpenAI

from ascent_platform.config.runtime import get_runtime_settings
from ascent_platform.llm.credentials import gemini_api_key
from ascent_platform.llm.executor import run_blocking_llm_call as _run_blocking_llm_call
from ascent_platform.llm.http_options import gemini_http_options

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")

boto_config = BotoConfig(
    connect_timeout=10,
    read_timeout=180,
    # Application retry policy owns the complete attempt budget. Without this,
    # Botocore silently retries inside every decorated application attempt.
    retries={"total_max_attempts": 1, "mode": "standard"},
)


@cache
def _import_vertexai_generative_model():
    from vertexai.preview.generative_models import GenerativeModel

    return GenerativeModel


# The errors worth another attempt. Every provider SDK wraps its transport
# failures in its own exception type, and none of them subclass the builtins:
# retrying on ``ConnectionError``/``TimeoutError`` alone catches a raw
# asyncio timeout and nothing a provider actually raises, so a rate limit or a
# 502 is fatal on first sight.
TRANSIENT_LLM_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
    genai_errors.ServerError,
    aiohttp.ClientConnectionError,
    aiohttp.ServerTimeoutError,
    ServiceRequestError,
    ServiceResponseError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)

_BEDROCK_TRANSIENT_CODES = {
    "InternalFailure",
    "InternalServerException",
    "ModelNotReadyException",
    "ModelTimeoutException",
    "ServiceUnavailableException",
    "ThrottlingException",
    "TooManyRequestsException",
}


def is_transient_llm_error(error: Exception) -> bool:
    """Return whether a provider failure is safe to retry.

    Provider base exception classes are too broad. In particular, Botocore's
    ``ClientError`` represents both throttling and permanent failures such as
    access denial, while Azure's ``HttpResponseError`` does the same for HTTP
    status codes. Classify those two families by their provider details.
    """
    if isinstance(error, ClientError):
        response = error.response or {}
        code = str(response.get("Error", {}).get("Code", ""))
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return code in _BEDROCK_TRANSIENT_CODES or status == 429 or (isinstance(status, int) and status >= 500)

    if isinstance(error, HttpResponseError):
        status = getattr(error, "status_code", None)
        return status == 429 or (isinstance(status, int) and 500 <= status < 600)
    if isinstance(error, aiohttp.ClientResponseError):
        return error.status == 429 or 500 <= error.status < 600

    return isinstance(error, (*TRANSIENT_LLM_ERRORS, *_anthropic_transient_errors()))


@cache
def _anthropic_transient_errors() -> tuple[type[Exception], ...]:
    # Keep this SDK optional and lazy, just like AnthropicAssistant.client.
    try:
        from anthropic import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
    except ImportError:
        return ()
    return APIConnectionError, APITimeoutError, InternalServerError, RateLimitError


RetryFilter = Union[
    type[Exception],
    tuple[type[Exception], ...],
    Callable[[Exception], bool],
]


def _matches_retry_filter(error: Exception, retry_on: RetryFilter | None) -> bool:
    if retry_on is None:
        return True
    if isinstance(retry_on, type) or isinstance(retry_on, tuple):
        return isinstance(error, retry_on)
    return retry_on(error)


class RetryError(Exception):
    """Base class for retry-related errors"""

    pass


class EmptyResponseError(Exception):
    """Exception raised when an empty response is received from the model."""

    pass


class InvalidJsonResponseError(Exception):
    """Exception raised when the model returns invalid JSON that doesn't match the expected schema."""

    pass


class RetryMixin:
    """Base class providing retry functionality"""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor

    @staticmethod
    def calculate_sleep_time(attempt: int, base_delay: float, backoff_factor: float) -> float:
        return base_delay * (backoff_factor**attempt)

    async def handle_retry(
        self,
        err: Exception,
        attempt: int,
        max_retries: int,
        base_delay: float,
        backoff_factor: float,
    ) -> bool:
        """
        Default retry handling logic with exponential backoff.

        Args:
            err: The exception that triggered the retry
            attempt: Current attempt number (0-based)
            max_retries: Maximum number of retries
            base_delay: Initial delay between retries
            backoff_factor: Multiplicative factor for backoff

        Returns:
            bool: True if should retry, False otherwise
        """
        if attempt < max_retries - 1:
            sleep_time = self.calculate_sleep_time(attempt, base_delay, backoff_factor)
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(err)}. Retrying in {sleep_time:.2f} seconds...")
            await asyncio.sleep(sleep_time)
            return True
        logger.error(f"All {max_retries} retry attempts failed. Last error: {str(err)}")
        return False


async def _run_with_retry(
    call: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    base_delay: float,
    backoff_factor: float,
    retry_on: RetryFilter | None,
    handle_retry: Callable[..., Awaitable[bool]] | None = None,
) -> T:
    """Shared attempt engine; max_retries means total attempts, including the first."""
    if max_retries < 1 or base_delay < 0 or backoff_factor < 0:
        raise ValueError("Retry attempts must be positive and delays/factors non-negative")
    handler = handle_retry or RetryMixin().handle_retry
    for attempt in range(max_retries):
        try:
            return await call()
        except Exception as error:
            if not _matches_retry_filter(error, retry_on):
                raise
            # The handler owns the backoff. Never invoke it after the last
            # attempt: even a custom handler cannot extend the attempt budget.
            if attempt == max_retries - 1:
                raise
            if not await handler(error, attempt, max_retries, base_delay, backoff_factor):
                raise
    raise RetryError("Unexpected end of retry loop")


def async_retry(
    max_retries: Optional[int] = None,
    base_delay: Optional[float] = None,
    backoff_factor: Optional[float] = None,
    retry_on: RetryFilter | None = None,
) -> Callable[[Callable[Concatenate[Any, P], Awaitable[T]]], Callable[Concatenate[Any, P], Awaitable[T]]]:
    """Retry a method, using instance settings only when an override is None."""

    def decorator(func: Callable[Concatenate[Any, P], Awaitable[T]]) -> Callable[Concatenate[Any, P], Awaitable[T]]:
        @wraps(func)
        async def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> T:
            return await _run_with_retry(
                partial(func, self, *args, **kwargs),
                max_retries=getattr(self, "max_retries", 3) if max_retries is None else max_retries,
                base_delay=getattr(self, "base_delay", 1.0) if base_delay is None else base_delay,
                backoff_factor=getattr(self, "backoff_factor", 2.0) if backoff_factor is None else backoff_factor,
                retry_on=retry_on,
                handle_retry=getattr(self, "handle_retry", None),
            )

        return wrapper

    return decorator


def async_retry_fn(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retry_on: RetryFilter | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Standalone-function adapter for the same retry engine used by methods."""

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await _run_with_retry(
                partial(func, *args, **kwargs),
                max_retries=max_retries,
                base_delay=base_delay,
                backoff_factor=backoff_factor,
                retry_on=retry_on,
            )

        return wrapper

    return decorator


class BaseAssistant(RetryMixin):
    def __init__(self, system_message: Optional[dict] = None):
        super().__init__(max_retries=5, base_delay=1, backoff_factor=1.5)

        self.system_message = system_message or {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
        self.conversation = [self.system_message]

    @abstractmethod
    async def get_response(
        self,
        prompt: Optional[str] = None,
        temperature=0,
        json_format: Optional[bool] = False,
        quiet: Optional[bool] = False,
        response_schema: Optional[Type[BaseModel]] = None,
    ):
        pass

    def reset_conversation(self):
        self.conversation = [self.system_message]

    def add_message(self, role, message):
        self.conversation.append({"role": role, "content": message})


# Define a registry for assistant creators
assistant_registry: Dict[str, Type[BaseAssistant]] = {}


# A decorator function for registering assistant creators
def register_assistant(assistant_type: str):
    def decorator(cls: Type[BaseAssistant]):
        assistant_registry[assistant_type] = cls
        return cls

    return decorator


@register_assistant("gpt")
class GPTAssistant(BaseAssistant):
    model_name: str
    api_key: str
    api_base: str
    api_version: str
    max_response_tokens: int
    client: AsyncAzureOpenAI

    # No endpoint default: it is tenant-specific, and a wrong one silently
    # sends prompts to somebody else's resource. Configure AZURE_OPENAI_ENDPOINT
    # (or OPENAI_API_BASE) and the deployment name for your own deployment.
    MODEL_NAME = "gpt-4o"
    OPENAI_API_VERSION = "2024-02-01"

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        super().__init__()

        # Precedence: explicit argument, then configuration, then the class
        # default. The order matters -- the class constants are non-empty, so
        # consulting them before settings would make the settings branch
        # unreachable and silently discard OPENAI_API_BASE, OPENAI_API_VERSION
        # and MODEL_NAME.
        settings = get_runtime_settings()
        self.model_name = model_name or settings.MODEL_NAME or self.MODEL_NAME
        self.api_key = api_key or settings.OPENAI_API_KEY_GPT4O or settings.OPENAI_API_KEY
        self.api_base = api_base or settings.OPENAI_API_BASE
        self.api_version = api_version or settings.OPENAI_API_VERSION or self.OPENAI_API_VERSION

        if not self.api_base:
            raise ValueError(
                "Azure OpenAI endpoint is not configured. Set OPENAI_API_BASE to your own resource, e.g. https://<your-resource>.openai.azure.com/"
            )

        # If you need to override the default system message
        self.system_message = {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
        self.conversation = [self.system_message]  # Reset conversation with new system message

        self.max_response_tokens = 16384
        self.token_limit = 8192 * 4
        self.conversation = [self.system_message]
        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.api_base,
        )

    @staticmethod
    def num_tokens_from_messages(messages):
        encoding = tiktoken.encoding_for_model("gpt-4-32k")
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    def add_message(self, role, message):
        self.conversation.append({"role": role, "content": message})
        self.manage_conversation_length()

    def manage_conversation_length(self):
        conv_history_tokens = self.num_tokens_from_messages(self.conversation)

        while conv_history_tokens + self.max_response_tokens >= self.token_limit:
            del self.conversation[1]
            conv_history_tokens = self.num_tokens_from_messages(self.conversation)

    @async_retry(retry_on=TRANSIENT_LLM_ERRORS)
    async def get_response(
        self,
        prompt: Optional[str] = None,
        temperature=0,
        json_format: Optional[bool] = False,
    ):
        """
        Sends a prompt to Gpt and retrieves the response.

        This method formats the prompt as required, sends it to Gpt, and handles retries in case of failures.

        Parameters:
        - prompt (Optional[str]): The input prompt to send to Gpt. If None, the existing conversation is used.
        - json_format (Optional[bool]): A flag indicating whether the response should be formatted as JSON. If True,
          the response will include a system message and be structured as a JSON object.

        Returns:
        - str: The content of the message from Gpt's response.

        Note:
        - If `json_format` is True, the response will be a JSON object; otherwise, it will be a plain text string.
        """
        messages = self.prepare_messages(prompt, json_format)
        response = await self.make_api_call(messages, temperature, json_format)
        return self.process_response(response, prompt)

    def prepare_messages(self, prompt, json_format):
        if json_format:
            return [
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to output JSON.",
                },
                {"role": "user", "content": prompt},
            ]
        return [{"role": "user", "content": prompt}] if prompt is not None else self.conversation

    async def make_api_call(self, messages, temperature, json_format):
        """Template method that uses _get_api_params"""
        params = self._get_api_params(messages, temperature, json_format)
        return await self.client.chat.completions.create(**params)

    def _get_api_params(self, messages, temperature, json_format):
        """Returns the API parameters for standard GPT models"""
        params = {
            "model": self.model_name,
            "temperature": temperature,
            "messages": messages,
            "max_tokens": self.max_response_tokens,
            "timeout": 600,
        }
        if json_format:
            params["response_format"] = {"type": "json_object"}
        return params

    def process_response(self, response, prompt):
        if prompt is None:
            self.add_message(role="assistant", message=response.choices[0].message.content)
        return response.choices[0].message.content


@register_assistant("gpt-o1")
class GPTo1Assistant(GPTAssistant):
    """
    GPTo1Assistant inherits from GPTAssistant to handle the OpenAI O1 preview model.
    The main difference is in the API parameters - O1 uses 'max_completion_tokens'
    instead of 'max_tokens' which is used by other GPT models.
    """

    DEFAULT_MODEL = "o1"
    DEFAULT_API_VERSION = "2024-12-01-preview"

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        # Use class defaults before falling back to settings
        model_name = model_name or self.DEFAULT_MODEL
        api_version = api_version or self.DEFAULT_API_VERSION
        api_base = api_base or get_runtime_settings().OPENAI_API_BASE
        api_key = api_key or get_runtime_settings().OPENAI_API_KEY_O1

        # Initialize the parent class with all necessary parameters
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
        )

    # TO DO: clean up temperature since it is not needed
    def _get_api_params(self, messages, temperature, json_format):
        """Returns the API parameters for O1 model"""
        params = {
            "model": self.model_name,
            # "temperature": temperature,   #not available
            "messages": messages,
            "max_completion_tokens": self.max_response_tokens,  # Only difference
            "timeout": 600,
        }
        if json_format:
            params["response_format"] = {"type": "json_object"}
        return params

    def prepare_messages(self, prompt, json_format):
        """Override to handle the fact that O1 doesn't support system messages"""
        if json_format:
            # For JSON format, include the instruction in the user message
            return [
                {
                    "role": "user",
                    "content": "You are a helpful assistant designed to output JSON. " + prompt,
                }
            ]

        if prompt is not None:
            # For new prompts, combine system message content with the prompt
            system_content = self.system_message["content"]
            return [{"role": "user", "content": f"{system_content}\n\n{prompt}"}]

        # For conversation history, filter out system messages and convert them
        messages = []
        for msg in self.conversation:
            if msg["role"] == "system":
                # Convert system message to user message
                messages.append({"role": "user", "content": msg["content"]})
            else:
                messages.append(msg)
        return messages


async def _collect_stream_text(stream, label: str, started_at: float) -> str:
    """Consume Gemini streams identically across normal and fallback paths."""
    parts: list[str] = []
    async for chunk in stream:
        if chunk.text:
            parts.append(chunk.text)
            elapsed = time.time() - started_at
            if len(parts) == 1:
                logger.info("[%s] First chunk received! TTFT: %.2fs", label, elapsed)
            if len(parts) % 10 == 0:
                logger.info("[%s] Chunk %s received, elapsed: %.2fs", label, len(parts), elapsed)
    logger.info("[%s] Complete! Total chunks: %s, Total time: %.2fs", label, len(parts), time.time() - started_at)
    return "".join(parts)


@register_assistant("gemini")
class GeminiAssistant(BaseAssistant):
    # Names the setting rather than the model: the value lives in
    # RuntimeSettings, and this is how the seam test sees that a default exists.
    DEFAULT_MODEL_SETTING = "GEMINI_DEFAULT_MODEL_NAME"

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        max_output_tokens: int = 128000,
        temperature: float = 1.0,
        top_p: float = 0.8,
    ):
        super().__init__()
        # Resolved from settings rather than a class constant, and coerced from
        # an explicit None as well as a missing argument: callers pass
        # model_name=None to mean "use the default", and Gemini answers an empty
        # model id with 404 {'message': '', 'status': 'Not Found'}.
        model_name = model_name or get_runtime_settings().GEMINI_DEFAULT_MODEL_NAME

        # Shared resolver: this read GEMINI_API_KEY only, while the platform's
        # two Gemini sites also accepted GOOGLE_API_KEY, in opposite orders.
        self.api_key = gemini_api_key(api_key)
        if not self.api_key:
            raise ValueError("Gemini API key not found")

        # Initialize the client with API key
        self.client = genai.Client(api_key=self.api_key, http_options=gemini_http_options())

        # Initialize model and parameters
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Initialize conversation history
        self.conversation = []

    @async_retry(retry_on=(*TRANSIENT_LLM_ERRORS, EmptyResponseError, InvalidJsonResponseError))
    async def get_response(
        self,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        json_format: Optional[bool] = False,
        quiet: Optional[bool] = False,
        response_schema: Optional[Type[BaseModel]] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Dict[str, Any], BaseModel]:
        """
        Get a response from the Gemini model.

        Args:
            prompt: The input prompt
            temperature: Temperature for response generation
            json_format: Whether to return response in JSON format
            quiet: Return log is response correctly received
            response_schema: Optional Pydantic model to structure the response
            stream: Whether to use streaming (keeps connection alive for long responses)
            **kwargs: Additional parameters

        Returns:
            Union[str, Dict[str, Any], BaseModel]: The generated response
        """
        try:
            if prompt is None:
                prompt = str(self.conversation)
                logger.info("No prompt passed, using conversation history.")

            # Add JSON format instruction if needed
            if json_format or response_schema:
                # Enhance the prompt with more specific JSON formatting instructions
                json_instruction = (
                    "Your response must be valid, properly formatted JSON that can be parsed by Python's json.loads(). "
                    "Ensure all quotes are properly escaped and all brackets/braces are balanced. "
                    "Do not include any explanatory text outside the JSON structure."
                )
                prompt = f"{json_instruction}\n\n{prompt}"

            # Prepare config dictionary with generation parameters
            config = {
                "max_output_tokens": self.max_output_tokens,
                "temperature": (temperature if temperature is not None else self.temperature),
                "top_p": self.top_p,
            }

            # Add response schema to config if present
            if response_schema:
                # Convert Pydantic model to JSON schema
                schema_dict = response_schema.model_json_schema()
                config["response_mime_type"] = "application/json"
                config["response_schema"] = schema_dict

            # Generate response - streaming or non-streaming
            if stream:
                # Use streaming to keep connection alive for long responses
                # This prevents timeout disconnections by receiving chunks continuously
                stream_start_time = time.time()
                logger.info(f"[STREAMING] Starting stream request to {self.model_name}...")

                used_fallback = False  # Track if we used unstructured fallback

                try:
                    response_stream = await self.client.aio.models.generate_content_stream(model=self.model_name, contents=prompt, config=config)

                    response_text = await _collect_stream_text(response_stream, "STREAMING", stream_start_time)

                except Exception as stream_error:
                    # Check if this is a timeout/disconnect error and we have a schema to fallback
                    error_msg = str(stream_error).lower()
                    is_timeout = "disconnected" in error_msg or "timeout" in error_msg or "server" in error_msg

                    if is_timeout and response_schema:
                        elapsed = time.time() - stream_start_time
                        logger.warning(f"[STREAMING] Structured output timed out after {elapsed:.2f}s: {stream_error}")
                        logger.info("[FALLBACK] Retrying WITHOUT structured output schema (will parse manually)...")

                        used_fallback = True

                        # Create config WITHOUT response_schema
                        fallback_config = {
                            "max_output_tokens": self.max_output_tokens,
                            "temperature": temperature if temperature is not None else self.temperature,
                            "top_p": self.top_p,
                        }

                        # Add JSON instruction and schema to prompt for fallback
                        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
                        fallback_prompt = f"""{prompt}

IMPORTANT: You MUST return your response as a valid JSON object matching this exact schema:
```json
{schema_json}
```

Return ONLY the JSON object, no other text before or after it."""

                        fallback_start = time.time()
                        logger.info("[FALLBACK] Starting unstructured stream request...")

                        try:
                            response_stream = await self.client.aio.models.generate_content_stream(
                                model=self.model_name, contents=fallback_prompt, config=fallback_config
                            )

                            response_text = await _collect_stream_text(response_stream, "FALLBACK", fallback_start)

                        except Exception as fallback_error:
                            # Second fallback failed - try with reduced thinking for gemini-3-pro
                            fallback_error_msg = str(fallback_error).lower()
                            is_fallback_timeout = any(token in fallback_error_msg for token in ("disconnected", "timeout", "server"))
                            is_gemini_3_pro = "gemini-3-pro" in self.model_name.lower()

                            if is_fallback_timeout and is_gemini_3_pro:
                                elapsed = time.time() - fallback_start
                                logger.warning(f"[FALLBACK] Unstructured output also timed out after {elapsed:.2f}s: {fallback_error}")
                                logger.warning(f"[LOW-THINKING] ⚠️ REDUCING THINKING LEVEL TO 'low' for {self.model_name}")
                                logger.warning("[LOW-THINKING] This may reduce response quality but should complete faster")

                                # Import types for GenerateContentConfig with ThinkingConfig
                                from google.genai import types as genai_types

                                # Create GenerateContentConfig with LOW thinking level + STRUCTURED OUTPUT
                                # Per Google docs: thinking_config=types.ThinkingConfig(thinking_level="low")
                                # Low thinking is much faster, so we can use structured output directly
                                low_thinking_config_dict = {
                                    "max_output_tokens": self.max_output_tokens,
                                    "temperature": temperature if temperature is not None else self.temperature,
                                    "top_p": self.top_p,
                                    "thinking_config": genai_types.ThinkingConfig(thinking_level="low"),
                                }

                                # Add structured output schema if we have one
                                if response_schema:
                                    low_thinking_config_dict["response_mime_type"] = "application/json"
                                    low_thinking_config_dict["response_schema"] = response_schema.model_json_schema()
                                    logger.info("[LOW-THINKING] Using structured output with low thinking")

                                low_thinking_config = genai_types.GenerateContentConfig(**low_thinking_config_dict)

                                low_thinking_start = time.time()
                                logger.info("[LOW-THINKING] Starting stream request with thinking_level='low'...")

                                response_stream = await self.client.aio.models.generate_content_stream(
                                    model=self.model_name, contents=prompt, config=low_thinking_config
                                )

                                response_text = await _collect_stream_text(response_stream, "LOW-THINKING", low_thinking_start)
                                # Mark that we used low thinking with structured output (not fallback)
                                used_fallback = False
                            else:
                                # Not gemini-3-pro or not a timeout - re-raise
                                raise
                    else:
                        # Not a timeout or no schema to fallback - re-raise original error
                        raise
            else:
                # Non-streaming: single request/response
                response = await self.client.aio.models.generate_content(model=self.model_name, contents=prompt, config=config)
                response_text = response.text
                used_fallback = False

            # Check for empty response and raise custom exception to trigger retry
            if not response_text or response_text.strip() == "":
                raise EmptyResponseError("Empty response received from Gemini")

            # Store in conversation history
            self.add_message("user", prompt)
            self.add_message("assistant", response_text)

            if not quiet:
                mode_str = " (streaming)" if stream else ""
                if stream and used_fallback:
                    mode_str = " (streaming+fallback)"
                logger.info(
                    f"Successfully generated response with model: {self.model_name}{mode_str}, "
                    f"timestamp: {datetime.now(timezone.utc).strftime('%Y.%m.%d %H:%M')}"
                )

            # Process response based on format requirements
            if response_schema:
                try:
                    # If we used fallback (unstructured), make a second call to extract structured data
                    if stream and used_fallback:
                        logger.info("[FALLBACK] Making structured extraction call from unstructured response...")

                        # Create config WITH response_schema for the extraction call
                        extraction_config = {
                            "max_output_tokens": self.max_output_tokens,
                            "temperature": 0,  # Use 0 for deterministic extraction
                            "top_p": self.top_p,
                            "response_mime_type": "application/json",
                            "response_schema": response_schema.model_json_schema(),
                        }

                        extraction_prompt = f"""Extract the structured data from the following text and return it as a valid JSON object.

TEXT TO EXTRACT FROM:
{response_text}

Return ONLY the JSON object matching the required schema. Do not include any other text."""

                        # Make non-streaming call for extraction (should be fast since input is small)
                        extraction_response = await self.client.aio.models.generate_content(
                            model=self.model_name, contents=extraction_prompt, config=extraction_config
                        )

                        extraction_text = extraction_response.text
                        logger.info("[FALLBACK] Structured extraction complete")

                        # Parse the structured extraction response
                        parsed_data = json.loads(extraction_text)
                        return response_schema(**parsed_data)
                    else:
                        # Direct parsing for structured output (non-fallback case)
                        parsed_data = json.loads(response_text)
                        return response_schema(**parsed_data)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse response as schema: {e}")
                    # Instead of returning the raw text, raise an exception to trigger retry
                    raise InvalidJsonResponseError(f"Invalid JSON response for schema: {e}")
            elif json_format:
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse response as JSON: {e}")
                    raise InvalidJsonResponseError(f"Invalid JSON format: {e}")

            return response_text

        except EmptyResponseError as e:
            # Log the empty response error before re-raising
            logger.warning(f"Received empty response from Gemini API, triggering retry: {str(e)}")
            # Re-raise to be caught by the retry decorator
            raise

        except InvalidJsonResponseError as e:
            # Log the JSON parsing error before re-raising
            logger.warning(f"Received malformed JSON from Gemini API, triggering retry: {str(e)}")
            # Re-raise to be caught by the retry decorator
            raise

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation = []

    def add_message(self, role: str, message: str):
        """Add a message to the conversation history"""
        self.conversation.append({"role": role, "content": message})


@register_assistant("bedrock_llama3")
class BedrockLlama3Assistant(BaseAssistant):
    DEFAULT_MODEL = "us.meta.llama3-1-70b-instruct-v1:0"

    # all the ones below do not work

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_gen_len: int = 4096,
        temperature: float = 0.1,
        top_p: float = 0.9,
        region_name: str = "us-east-1",
        system_message: Optional[dict] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        backoff_factor: float = 2.0,
    ):
        super().__init__(system_message)
        self._client = None
        self.region_name = region_name
        self.model_id = model_name or self.DEFAULT_MODEL
        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.top_p = top_p
        # Update retry parameters
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.max_tokens_limit = 128000

    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name=self.region_name, config=boto_config)
        return self._client

    @staticmethod
    def create_prompt(user_message: str) -> str:
        prompt = f"""
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        {user_message}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """
        return prompt

    def create_request(self, prompt, max_gen_len=None, temperature=None, top_p=None):
        return {
            "prompt": prompt,
            "max_gen_len": self.max_gen_len if max_gen_len is None else max_gen_len,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p if top_p is None else top_p,
        }

    def invoke_model(self, request, stream=True) -> str:
        """Invoke and fully consume the synchronous Bedrock response.

        ``get_response`` runs this whole method in the dedicated blocking-LLM
        executor. Keeping stream iteration here prevents a synchronous body
        iterator from leaking back onto the event loop.
        """
        if not stream:
            response = self.client.invoke_model(
                body=json.dumps(request),
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response["body"].read())
            return response_body.get("generation", "")

        response = self.client.invoke_model_with_response_stream(
            body=json.dumps(request),
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json",
        )
        full_response = []
        for chunk in response.get("body"):
            chunk_data = json.loads(chunk["chunk"]["bytes"].decode())
            if "generation" in chunk_data:
                full_response.append(chunk_data["generation"])
        return "".join(full_response)

    @async_retry(retry_on=is_transient_llm_error)
    async def get_response(
        self,
        prompt: Optional[str] = None,
        max_gen_len: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        json_format: Optional[bool] = False,
        stream: bool = True,
        **kwargs,
    ):
        if prompt is None:
            prompt = str(self.conversation)
            logger.info("No prompt passed, using conversation.")

        if kwargs:
            logger.warning(f"The following kwargs were passed but are not used: {', '.join(kwargs.keys())}")

        formatted_prompt = self.create_prompt(prompt)
        request = self.create_request(
            formatted_prompt,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        response_text = await _run_blocking_llm_call(self.invoke_model, request, stream=stream)

        logger.info(
            f"Successful {self.model_id} response model: {self.model_id}, utc-timestamp: {datetime.now(timezone.utc).strftime('%Y.%m.%d %H:%M')}"
        )
        logger.debug(f"message:{str(prompt)}, response-content: {response_text}")

        if json_format:
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse response as JSON")
                return response_text

        return response_text


class BedrockConverseAssistant(BaseAssistant):
    """Shared implementation for models exposed through Bedrock Converse."""

    DEFAULT_MODEL: str

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_tokens: int = 32000,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        region_name: str = "us-east-1",
    ):
        self._client = None
        self.region_name = region_name
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_message = {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
        self.conversation = [self.system_message]

        super().__init__()

    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name=self.region_name, config=boto_config)
        return self._client

    def create_messages(self, prompt: str) -> list:
        """Create messages in the format expected by the converse API."""
        return [{"role": "user", "content": [{"text": prompt}]}]

    def create_inference_config(self) -> dict:
        """Create inference configuration for the converse API."""
        config: dict[str, int | float] = {
            "maxTokens": self.max_tokens,
        }

        # Only include temperature if it's not None
        if self.temperature is not None:
            config["temperature"] = self.temperature

        # Only include topP if it's not None
        if self.top_p is not None:
            config["topP"] = self.top_p

        return config

    def invoke_model(self, messages: list, inference_config: dict) -> str:
        """Send request to AWS Bedrock using the converse API."""
        response = self.client.converse(
            modelId=self.model_name,
            messages=messages,
            inferenceConfig=inference_config,
        )

        # Extract the text from the response
        # Response structure: {"output": {"message": {"content": [...]}}}
        # GPT-OSS 120B may return multiple content items:
        #   - {"reasoningContent": {"reasoningText": {"text": "..."}}} - reasoning/thinking
        #   - {"text": "..."} - actual response text
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])

        # Look for the actual text response (not reasoningContent)
        for item in content:
            if "text" in item:
                return item.get("text", "")

        # Fallback: if no direct text, try first item
        if content and len(content) > 0:
            return content[0].get("text", "")

        return ""

    @async_retry(retry_on=is_transient_llm_error)
    async def get_response(self, prompt: Optional[str] = None, **kwargs):
        if prompt is None:
            prompt = str(self.conversation)
            logger.info("No prompt passed, using conversation.")

        if kwargs:
            logger.warning(f"The following kwargs were passed but are not used: {', '.join(kwargs.keys())}")

        messages = self.create_messages(prompt)
        inference_config = self.create_inference_config()
        response_text = await _run_blocking_llm_call(self.invoke_model, messages, inference_config)

        logger.info(
            f"Successful {self.model_name} response model: {self.model_name}, utc-timestamp: {datetime.now(timezone.utc).strftime('%Y.%m.%d %H:%M')}"
        )
        logger.debug(f"message:{str(prompt)}, response-content: {response_text}")

        return response_text


@register_assistant("bedrock_gptoss")
class BedrockGPTOSSAssistant(BedrockConverseAssistant):
    """GPT-OSS through the shared Bedrock Converse transport."""

    DEFAULT_MODEL = "openai.gpt-oss-120b-1:0"


@register_assistant("bedrock_qwen")
class BedrockQwenAssistant(BedrockConverseAssistant):
    """Qwen through the shared Bedrock Converse transport."""

    DEFAULT_MODEL = "qwen.qwen3-vl-235b-a22b"


@register_assistant("bedrock_kimi")
class BedrockKimiAssistant(BedrockConverseAssistant):
    """Kimi through the shared Bedrock Converse transport."""

    DEFAULT_MODEL = "moonshot.kimi-k2-thinking"


@register_assistant("claude_sonnet")
class ClaudeSonnetAssistant(BaseAssistant):
    DEFAULT_MODEL = "us.anthropic.claude-opus-4-6-v1"
    FALLBACK_MODELS = [
        "global.anthropic.claude-opus-4-5-20251101-v1:0",
        "anthropic.claude-sonnet-4-6",
    ]

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_tokens: int = 32000,
        temperature: float = 0.0,
        top_p: Optional[str] = None,  # claude 4.5 cannot take top_p and temperature as input
        region_name: str = "us-east-1",
    ):
        # Initialize a boto3 client for the AWS Bedrock.
        self._client = None
        self.region_name = region_name
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_message = {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
        self.conversation = [self.system_message]

        super().__init__()

    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name=self.region_name, config=boto_config)
        return self._client

    def create_request(self, prompt):
        # Create a request dictionary.
        request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "system": "You are a helpful assistant",
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }

        # Only include temperature if it's not None
        if self.temperature is not None:
            request["temperature"] = self.temperature

        # Only include top_p if it's not None
        # Note: Claude 4.5 models cannot accept both temperature and top_p
        if self.top_p is not None:
            request["top_p"] = self.top_p

        return request

    def invoke_model(self, request):
        # Send the request to aws bedrock with fallback on capacity errors.
        models_to_try = [self.model_name] + self.FALLBACK_MODELS
        last_exception = None
        for model_id in models_to_try:
            try:
                response = self.client.invoke_model(body=json.dumps(request), modelId=model_id)
                model_response = json.loads(response.get("body").read())
                if model_id != self.model_name:
                    logger.warning(f"Fallback succeeded with model: {model_id}")
                return model_response.get("content")[0].get("text")
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code in ("ServiceUnavailableException", "ThrottlingException"):
                    logger.warning(f"Model {model_id} unavailable ({error_code}), trying next fallback...")
                    last_exception = e
                    continue
                raise
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("No Bedrock model was attempted")

    @async_retry(retry_on=is_transient_llm_error)
    async def get_response(self, prompt: Optional[str] = None, **kwargs):
        if prompt is None:
            prompt = str(self.conversation)
            logger.info("No prompt passed, using conversation.")

        if kwargs:
            logger.warning(f"The following kwargs were passed but are not used: {', '.join(kwargs.keys())}")

        request = self.create_request(prompt)
        response_text = await _run_blocking_llm_call(self.invoke_model, request)

        logger.info(
            f"Successful {self.model_name} response model: {self.model_name}, utc-timestamp: {datetime.now(timezone.utc).strftime('%Y.%m.%d %H:%M')}"
        )
        logger.debug(f"message:{str(prompt)}, response-content: {response_text}")

        return response_text


@register_assistant("anthropic")
class AnthropicAssistant(BaseAssistant):
    """Claude through Anthropic's own API, rather than Bedrock.

    ``claude_sonnet`` reaches Claude through AWS Bedrock, which means an AWS
    account with Bedrock model access. That is a reasonable ask inside a
    company that already runs on AWS and an unreasonable one for somebody who
    just wants to try this with the API key they already have.

    Both are registered, so the choice is configuration rather than a code
    change: set ANTHROPIC_API_KEY for this one, or AWS credentials for the
    Bedrock one. Requests for ``claude_sonnet`` on a deployment with only an
    Anthropic key resolve here automatically.

    The request shape is nearly identical -- Bedrock speaks Anthropic's
    messages format -- so this differs mainly in transport and in taking a
    plain model id rather than a Bedrock inference-profile arn.
    """

    DEFAULT_MODEL = "claude-sonnet-4-5"

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = 0,
        max_tokens: int = 8192,
        system_message: Optional[dict] = None,
    ):
        super().__init__(system_message)
        settings = get_runtime_settings()
        self.model_name = model_name or self.DEFAULT_MODEL
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY, or use the Bedrock-backed 'claude_sonnet' assistant with AWS credentials."
            )

    @property
    def client(self):
        # Imported lazily so the package is only needed by deployments that
        # actually use this provider.
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.api_key, max_retries=0)
        return self._client

    @async_retry(retry_on=is_transient_llm_error)
    async def get_response(self, prompt: Optional[str] = None, **kwargs):
        if prompt is None:
            prompt = str(self.conversation)
            logger.info("No prompt passed, using conversation.")

        message = await self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature if self.temperature is not None else 0,
            system=(self.system_message or {}).get("content", "You are a helpful assistant"),
            messages=[{"role": "user", "content": prompt}],
        )
        # Return the text, matching every other assistant: callers get a string
        # and parse JSON themselves when they asked for it.
        return "".join(block.text for block in message.content if block.type == "text")


@register_assistant("bedrock_mistral")
class BedrockMistralAssistant(BaseAssistant):
    DEFAULT_MODEL = "mistral.mistral-large-2402-v1:0"

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_tokens: int = 32000,
        temperature: float = 0.5,
        top_p: float = 0.9,
        top_k: int = 50,
        region_name: str = "us-east-1",
        system_message: Optional[dict] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        backoff_factor: float = 2.0,
    ):
        super().__init__(system_message)
        self._client = None
        self.region_name = region_name
        self.model_id = model_name or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        # Update retry parameters
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor

    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name=self.region_name, config=boto_config)
        return self._client

    @staticmethod
    def create_prompt(user_message: str) -> str:
        prompt = f"<s>[INST] {user_message} [/INST]"
        return prompt

    def create_request(self, prompt, max_tokens=None, temperature=None, top_p=None, top_k=None):
        return {
            "prompt": prompt,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p if top_p is None else top_p,
            "top_k": self.top_k if top_k is None else top_k,
        }

    def invoke_model(self, request):
        response = self.client.invoke_model(
            body=json.dumps(request),
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response["body"].read())
        return response_body.get("generation", "")

    @async_retry(retry_on=is_transient_llm_error)
    async def get_response(
        self,
        prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        json_format: Optional[bool] = False,
        **kwargs,
    ):
        if prompt is None:
            prompt = str(self.conversation)
            logger.info("No prompt passed, using conversation.")

        if kwargs:
            logger.warning(f"The following kwargs were passed but are not used: {', '.join(kwargs.keys())}")

        formatted_prompt = self.create_prompt(prompt)
        request = self.create_request(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        response_text = await _run_blocking_llm_call(self.invoke_model, request)

        logger.info(
            f"Successful {self.model_id} response model: {self.model_id}, utc-timestamp: {datetime.now(timezone.utc).strftime('%Y.%m.%d %H:%M')}"
        )
        logger.debug(f"message:{str(prompt)}, response-content: {response_text}")

        if json_format:
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse response as JSON")
                return response_text

        return response_text


@register_assistant("deepseek")
class DeepSeekR1Assistant(BaseAssistant):
    DEFAULT_MODEL = "DeepSeek-R1"
    DEFAULT_ENDPOINT = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0,
        system_message: Optional[dict] = None,
        stream: bool = True,  # Added streaming parameter
    ):
        super().__init__(system_message)

        settings = get_runtime_settings()
        self.model_name = model_name or self.DEFAULT_MODEL
        self.endpoint = endpoint or settings.AZURE_INFERENCE_ENDPOINT_DEEPSEEK_R1
        self.api_key = api_key or settings.AZURE_INFERENCE_KEY_DEEPSEEK_R1_EAST_US2
        self.stream = stream  # Store streaming preference

        if not self.api_key:
            raise ValueError("DeepSeek R1 API key not found")
        if not self.endpoint:
            raise ValueError(
                "DeepSeek R1 endpoint is not configured. Azure AI Foundry endpoints are "
                "per-deployment; set AZURE_INFERENCE_ENDPOINT_DEEPSEEK_R1 to your own, "
                "e.g. https://<your-deployment>.<region>.models.ai.azure.com/"
            )

        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = ChatCompletionsClient(endpoint=self.endpoint, credential=AzureKeyCredential(self.api_key))
        return self._client

    @async_retry(retry_on=is_transient_llm_error)
    async def get_response(
        self,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        json_format: Optional[bool] = False,
        **kwargs,
    ) -> str:
        try:
            if prompt is None:
                prompt = str(self.conversation)
                logger.info("No prompt passed, using conversation history.")

            messages = [
                SystemMessage(content=self.system_message["content"]),
                UserMessage(content=prompt),
            ]

            if json_format:
                messages[0].content += " Provide your response in JSON format."

            response_text = await _run_blocking_llm_call(
                self._complete_response,
                messages,
                temperature if temperature is not None else self.temperature,
            )

            # Store in conversation history
            self.add_message("user", prompt)
            self.add_message("assistant", response_text)

            logger.info(
                f"Successfully generated response with model: {self.model_name}, timestamp: {datetime.now(timezone.utc).strftime('%Y.%m.%d %H:%M')}"
            )

            if json_format:
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse response as JSON")
                    return response_text

            return response_text

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def _complete_response(self, messages, temperature: float) -> str:
        """Run and consume the synchronous Azure inference response."""
        response = self.client.complete(
            messages=messages,
            model=self.model_name,
            temperature=temperature,
            max_tokens=self.max_tokens,
            stream=self.stream,
        )
        if not self.stream:
            if not response.choices:
                raise ValueError("Empty response received from DeepSeek")
            return response.choices[0].message.content

        parts = []
        for chunk in response:
            choices = getattr(chunk, "choices", None)
            delta = getattr(choices[0], "delta", None) if choices else None
            content = getattr(delta, "content", None)
            if content:
                parts.append(content)
        if not parts:
            raise ValueError("Empty streaming response received from DeepSeek")
        return "".join(parts)

    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation = [self.system_message]
        if self._client:
            self._client.close()
            self._client = None


# preferred way to connect to mistral is via bedrock
@register_assistant("mistral")
class MistralAssistant(BaseAssistant):
    # Every other assistant defaults its model, and create_assistant builds
    # them with no arguments. Without a default here a deployment configured
    # only with MISTRAL_API_KEY passed startup and then raised TypeError on
    # the first call -- the provider resolver had legitimately chosen it.
    DEFAULT_MODEL = "mistral-large-latest"

    def __init__(self, model_name: Optional[str] = None, mistral_api_key=None):
        super().__init__()
        model_name = model_name or self.DEFAULT_MODEL
        if mistral_api_key is None:
            self.mistral_api_key = get_runtime_settings().MISTRAL_API_KEY
        else:
            self.mistral_api_key = mistral_api_key
        self.url = "https://api.mistral.ai/v1/chat/completions"
        self.model_name = model_name

    @async_retry(retry_on=is_transient_llm_error)
    async def get_response(
        self,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        json_format: Optional[bool] = False,
        quiet: Optional[bool] = False,
        response_schema: Optional[Type[BaseModel]] = None,
        *,
        message: Optional[str] = None,
    ) -> Union[str, Dict[str, Any], BaseModel]:
        """Implement the common contract; keep message= as a legacy alias."""
        if message is not None:
            if prompt is not None:
                raise ValueError("Pass either prompt or message, not both")
            prompt = message
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.mistral_api_key}",
        }

        messages = list(self.conversation)
        if prompt is not None:
            messages.append({"role": "user", "content": prompt})
        data: dict[str, Any] = {"model": self.model_name, "messages": messages}
        if temperature is not None:
            data["temperature"] = temperature
        if response_schema:
            data["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": response_schema.__name__, "schema": response_schema.model_json_schema()},
            }
        elif json_format:
            data["response_format"] = {"type": "json_object"}

        timeout = aiohttp.ClientTimeout(total=get_runtime_settings().GEMINI_HTTP_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(trust_env=True, timeout=timeout) as session:
            async with session.post(self.url, headers=headers, json=data) as response:
                response.raise_for_status()
                payload = await response.json()

        choices = payload.get("choices")
        text = choices[0].get("message", {}).get("content") if choices else None
        if not isinstance(text, str) or not text.strip():
            raise EmptyResponseError("Empty response received from Mistral")
        if response_schema:
            result = response_schema.model_validate_json(text)
        elif json_format:
            result = json.loads(text)
        else:
            result = text
        # Only record successful responses, never HTTP error bodies or failed parses.
        if prompt is not None:
            self.add_message("user", prompt)
        self.add_message("assistant", text)
        if not quiet:
            logger.info("Successful Mistral response from %s", self.model_name)
        return result


def create_assistant(assistant_type: str, **kwargs) -> Optional[BaseAssistant]:
    """
    :param assistant_type: Either gpt, gemini, linguist, mistral
    :param kwargs: Optional parameters model_name, api_key, api_base, api_version for different model usage from gpt
    :return:
    """
    from ascent_platform.llm.availability import resolve

    # Resolve the requested provider against what this deployment can reach.
    # Returning the request unchanged when it is configured, so nothing moves
    # for a deployment that has every provider set up.
    resolved = resolve(assistant_type) or assistant_type

    if resolved != assistant_type:
        # Everything here is provider-specific. Call sites pin these alongside
        # the provider -- the OMOP RAG path asks for claude_sonnet WITH
        # "us.anthropic.claude-opus-4-6-v1"; the personalized-questions job
        # asks for gpt WITH an Azure api_base -- so carrying them across a
        # substitution either points the substitute at a model it does not
        # have, or hands it an argument it does not accept at all
        # (TypeError: GeminiAssistant.__init__() got an unexpected keyword
        # argument 'api_base', which failed that job on every run). Dropping
        # them lets the substitute use its own configuration.
        provider_specific = {"model_name", "api_key", "api_base", "api_version"}
        carried = sorted(provider_specific & kwargs.keys())
        if carried:
            logging.info(
                "Dropping %s: %r was substituted with %r, and those belong to the provider that was asked for.",
                ", ".join(carried),
                assistant_type,
                resolved,
            )
            kwargs = {k: v for k, v in kwargs.items() if k not in provider_specific}

    assistant_cls = assistant_registry.get(resolved)
    if not assistant_cls:
        logging.exception(f"Assistant type '{resolved}' is not registered.")
        return None
    return assistant_cls(**kwargs)


# GPT_fallback_model_list was here: a module-level list that CONSTRUCTED a
# ClaudeSonnetAssistant and a BedrockLlama3Assistant -- and therefore their
# boto3/SDK handles -- at import time, on every import of this module. Nothing
# read it. Removed rather than made lazy, because a fallback chain nobody
# consults is not a fallback chain.
