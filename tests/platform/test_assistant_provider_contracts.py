from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import anthropic
import httpx
import pytest
from pydantic import BaseModel

from ascent_platform.llm.clients import assistants
from ascent_platform.llm.clients.assistants import AnthropicAssistant, MistralAssistant, is_transient_llm_error


class _Response:
    def __init__(self, status=200, text="ok"):
        self.status = status
        self.content = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=SimpleNamespace(real_url="https://example.invalid"),
                history=(),
                status=self.status,
                message="provider error",
            )

    async def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


@pytest.fixture
def mistral_transport(monkeypatch):
    requests = []
    responses = []

    class Session:
        def __init__(self, **kwargs):
            assert kwargs["timeout"].total > 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def post(self, url, **kwargs):
            requests.append(kwargs["json"])
            return responses.pop(0)

    monkeypatch.setattr(assistants.aiohttp, "ClientSession", Session)
    return requests, responses


async def test_mistral_supports_common_call_and_conversation_contract(mistral_transport):
    requests, responses = mistral_transport
    responses.extend([_Response(), _Response(text="second")])
    client = MistralAssistant(mistral_api_key="dummy")
    client.add_message("user", "context")
    assert await client.get_response(prompt="probe", temperature=0, quiet=True) == "ok"
    assert requests[0]["temperature"] == 0
    assert requests[0]["messages"][-2:] == [
        {"role": "user", "content": "context"},
        {"role": "user", "content": "probe"},
    ]
    assert await client.get_response() == "second"
    assert requests[1]["messages"][-1] == {"role": "assistant", "content": "ok"}
    client.reset_conversation()
    assert client.conversation == [client.system_message]


async def test_mistral_preserves_legacy_message_keyword(mistral_transport):
    requests, responses = mistral_transport
    responses.append(_Response())
    client = MistralAssistant(mistral_api_key="dummy")
    assert await client.get_response(message="legacy") == "ok"
    assert requests[0]["messages"][-1]["content"] == "legacy"
    with pytest.raises(ValueError, match="not both"):
        await client.get_response(prompt="new", message="legacy")


async def test_mistral_structured_response_contract(mistral_transport):
    class Answer(BaseModel):
        count: int

    requests, responses = mistral_transport
    responses.extend([_Response(text='{"count": 2}'), _Response(text='{"count": 3}')])
    client = MistralAssistant(mistral_api_key="dummy")
    assert await client.get_response("probe", json_format=True) == {"count": 2}
    result = await client.get_response(prompt="probe", response_schema=Answer, quiet=True)
    assert isinstance(result, Answer)
    assert result.count == 3
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert requests[1]["response_format"]["json_schema"]["schema"] == Answer.model_json_schema()


@pytest.mark.parametrize("status,attempts", [(400, 1), (401, 1), (403, 1), (429, 3), (502, 3)])
async def test_mistral_http_failures_are_not_successful_answers(mistral_transport, status, attempts):
    requests, responses = mistral_transport
    responses.extend(_Response(status=status) for _ in range(attempts))
    client = MistralAssistant(mistral_api_key="dummy")
    client.max_retries = 3
    client.base_delay = 0
    with pytest.raises(aiohttp.ClientResponseError) as error:
        await client.get_response(prompt="probe")
    assert error.value.status == status
    assert len(requests) == attempts
    assert client.conversation == [client.system_message]


@pytest.mark.parametrize("status,attempts", [(400, 1), (401, 1), (403, 1), (429, 3), (503, 3)])
async def test_anthropic_sdk_errors_follow_application_policy(status, attempts):
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.invalid"))
    error_class = {
        400: anthropic.BadRequestError,
        401: anthropic.AuthenticationError,
        403: anthropic.PermissionDeniedError,
        429: anthropic.RateLimitError,
        503: anthropic.InternalServerError,
    }[status]
    error = error_class("provider error", response=response, body=None)
    create = AsyncMock(side_effect=error)
    client = AnthropicAssistant(api_key="dummy")
    client._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    client.max_retries = 3
    client.base_delay = 0
    with pytest.raises(error_class):
        await client.get_response("probe")
    assert create.await_count == attempts


@pytest.mark.parametrize("error_type", [anthropic.APIConnectionError, anthropic.APITimeoutError])
def test_anthropic_transport_errors_are_retryable(error_type):
    assert is_transient_llm_error(error_type(request=httpx.Request("POST", "https://example.invalid")))


def test_anthropic_sdk_does_not_add_hidden_attempts(monkeypatch):
    constructor = []
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda **kwargs: constructor.append(kwargs) or object())
    client = AnthropicAssistant(api_key="dummy")
    assert client.client is not None
    assert constructor == [{"api_key": "dummy", "max_retries": 0}]
