from __future__ import annotations

import asyncio
import io
import time
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError, ConnectionClosedError, ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError

from ascent_platform.llm import executor
from ascent_platform.llm.clients import assistants
from ascent_platform.llm.clients.assistants import (
    BedrockGPTOSSAssistant,
    BedrockKimiAssistant,
    BedrockLlama3Assistant,
    BedrockMistralAssistant,
    BedrockQwenAssistant,
    DeepSeekR1Assistant,
    is_transient_llm_error,
)


class _SlowBedrock:
    def invoke_model(self, **kwargs):
        time.sleep(0.08)
        return {"body": io.BytesIO(b'{"generation":"ok"}')}

    def invoke_model_with_response_stream(self, **kwargs):
        def stream():
            time.sleep(0.08)
            yield {"chunk": {"bytes": b'{"generation":"ok"}'}}

        return {"body": stream()}


class _SlowAzure:
    def complete(self, **kwargs):
        if kwargs["stream"]:

            def stream():
                time.sleep(0.08)
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))])

            return stream()

        time.sleep(0.08)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])


async def _assert_event_loop_remains_responsive(call):
    started = asyncio.Event()
    released = asyncio.Event()

    async def other_task():
        started.set()
        await asyncio.sleep(0.02)
        released.set()

    task = asyncio.create_task(other_task())
    await started.wait()
    result = await call()
    assert released.is_set(), "synchronous provider work blocked the event loop"
    await task
    assert result == "ok"


@pytest.mark.parametrize("stream", [False, True])
async def test_llama_request_and_stream_consumption_do_not_block(stream):
    client = BedrockLlama3Assistant()
    client._client = _SlowBedrock()
    await _assert_event_loop_remains_responsive(lambda: client.get_response("probe", stream=stream))


async def test_mistral_request_does_not_block():
    client = BedrockMistralAssistant()
    client._client = _SlowBedrock()
    await _assert_event_loop_remains_responsive(lambda: client.get_response("probe"))


@pytest.mark.parametrize("stream", [False, True])
async def test_deepseek_request_and_stream_consumption_do_not_block(stream):
    client = DeepSeekR1Assistant(api_key="dummy", endpoint="https://example.invalid", stream=stream)
    client._client = _SlowAzure()
    await _assert_event_loop_remains_responsive(lambda: client.get_response("probe"))


class _FailingBedrock:
    def __init__(self, error: Exception):
        self.error = error
        self.calls = 0

    def invoke_model(self, **kwargs):
        self.calls += 1
        raise self.error


@pytest.fixture
def no_retry_delay(monkeypatch):
    async def no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(assistants.asyncio, "sleep", no_sleep)


@pytest.mark.parametrize("provider", [BedrockLlama3Assistant, BedrockMistralAssistant])
@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("temporary"),
        EndpointConnectionError(endpoint_url="https://example.invalid"),
        ReadTimeoutError(endpoint_url="https://example.invalid"),
        ConnectTimeoutError(endpoint_url="https://example.invalid"),
        ConnectionClosedError(endpoint_url="https://example.invalid"),
    ],
)
async def test_transient_errors_are_retried_once_per_configured_attempt(provider, error, no_retry_delay):
    client = provider(max_retries=3)
    transport = _FailingBedrock(error)
    client._client = transport

    kwargs = {"stream": False} if provider is BedrockLlama3Assistant else {}
    with pytest.raises(type(error)):
        await client.get_response("probe", **kwargs)

    assert transport.calls == 3


@pytest.mark.parametrize(
    "error",
    [
        ValueError("bad request"),
        ClientError(
            {
                "Error": {"Code": "AccessDeniedException", "Message": "denied"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "InvokeModel",
        ),
    ],
)
@pytest.mark.parametrize("provider", [BedrockLlama3Assistant, BedrockMistralAssistant])
async def test_permanent_errors_are_not_retried(provider, error, no_retry_delay):
    client = provider(max_retries=3)
    transport = _FailingBedrock(error)
    client._client = transport

    kwargs = {"stream": False} if provider is BedrockLlama3Assistant else {}
    with pytest.raises(type(error)):
        await client.get_response("probe", **kwargs)

    assert transport.calls == 1


def test_bedrock_error_codes_are_classified_by_retryability():
    throttled = ClientError(
        {
            "Error": {"Code": "ThrottlingException", "Message": "slow down"},
            "ResponseMetadata": {"HTTPStatusCode": 429},
        },
        "InvokeModel",
    )
    denied = ClientError(
        {
            "Error": {"Code": "AccessDeniedException", "Message": "denied"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        },
        "InvokeModel",
    )

    assert is_transient_llm_error(throttled)
    assert not is_transient_llm_error(denied)


@pytest.mark.parametrize("provider,default", [(BedrockLlama3Assistant, 0.1), (BedrockMistralAssistant, 0.5)])
def test_explicit_zero_temperature_is_preserved(provider, default):
    client = provider()
    assert client.create_request("probe")["temperature"] == default
    assert client.create_request("probe", temperature=0.0)["temperature"] == 0.0


def test_bedrock_converse_models_share_one_implementation():
    for provider in (BedrockGPTOSSAssistant, BedrockQwenAssistant, BedrockKimiAssistant):
        assert "get_response" not in provider.__dict__
        assert "invoke_model" not in provider.__dict__


def test_blocking_provider_executor_is_process_wide_and_bounded():
    first = executor.get_executor()
    second = executor.get_executor()
    assert first is second
    assert first._max_workers == assistants.get_runtime_settings().LLM_BLOCKING_EXECUTOR_WORKERS


def test_application_retry_policy_owns_the_bedrock_attempt_budget():
    assert assistants.boto_config.retries["total_max_attempts"] == 1
