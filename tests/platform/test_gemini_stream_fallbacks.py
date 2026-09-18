from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.genai.types import ThinkingLevel
from pydantic import BaseModel

from ascent_platform.llm.clients import assistants


async def _chunks(*texts):
    for text in texts:
        if isinstance(text, Exception):
            raise text
        yield SimpleNamespace(text=text)


async def test_stream_collector_ignores_empty_chunks_and_preserves_order():
    assert await assistants._collect_stream_text(_chunks(None, "a", "", "b"), "TEST", 0) == "ab"
    assert await assistants._collect_stream_text(_chunks(), "TEST", 0) == ""


async def test_stream_collector_preserves_transport_failure():
    with pytest.raises(ConnectionError, match="disconnected"):
        await assistants._collect_stream_text(_chunks("partial", ConnectionError("disconnected")), "TEST", 0)


@pytest.mark.parametrize("failures", [0, 1, 2])
async def test_normal_unstructured_and_low_thinking_fallbacks(monkeypatch, failures):
    class Answer(BaseModel):
        count: int

    stream = AsyncMock(side_effect=[TimeoutError("server timeout")] * failures + [_chunks(None, '{"count":', " 2}")])
    extraction = AsyncMock(return_value=SimpleNamespace(text='{"count": 2}'))
    models = SimpleNamespace(generate_content_stream=stream, generate_content=extraction)
    monkeypatch.setattr(assistants.genai, "Client", lambda **kwargs: SimpleNamespace(aio=SimpleNamespace(models=models)))
    client = assistants.GeminiAssistant(model_name="gemini-3-pro", api_key="dummy")
    result = await client.get_response("probe", stream=True, response_schema=Answer, quiet=True)
    assert isinstance(result, Answer)
    assert result.count == 2
    assert stream.await_count == failures + 1
    assert extraction.await_count == (1 if failures == 1 else 0)
    assert "response_schema" in stream.call_args_list[0].kwargs["config"]
    if failures >= 1:
        assert "response_schema" not in stream.call_args_list[1].kwargs["config"]
    if failures == 2:
        assert stream.call_args_list[2].kwargs["config"].thinking_config.thinking_level == ThinkingLevel.LOW
