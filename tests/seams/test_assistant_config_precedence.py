"""Configuration must be able to override a provider client's class defaults.

``GPTAssistant`` used to resolve its endpoint, deployment and api-version from
class constants *before* consulting settings. Because those constants are
non-empty, the settings branch was unreachable: ``OPENAI_API_BASE``,
``OPENAI_API_VERSION`` and ``MODEL_NAME`` were read from the environment and
then thrown away.

That went unnoticed because ``.env.template`` sets the same values the
constants already carry, so nothing misbehaved -- but it meant the client was
pinned to a testing instance and could not be repointed without a code change.

These tests fix the precedence in place: argument, then configuration, then
class default.
"""

from __future__ import annotations

from unittest import mock

import pytest

from ascent_platform.llm.clients import assistants


class _Settings:
    """Only the fields this construction path reads."""

    MODEL_NAME = None
    OPENAI_API_KEY_GPT4O = "key"
    OPENAI_API_KEY = None
    OPENAI_API_BASE = None
    OPENAI_API_VERSION = None


def _build(**overrides):
    settings = _Settings()
    for name, value in overrides.items():
        setattr(settings, name, value)
    with mock.patch.object(assistants, "get_runtime_settings", return_value=settings), \
         mock.patch.object(assistants, "AsyncAzureOpenAI", lambda **kwargs: object()):
        return assistants.GPTAssistant()


def test_settings_override_the_class_defaults():
    client = _build(
        OPENAI_API_BASE="https://mine.openai.azure.com/",
        MODEL_NAME="my-deployment",
        OPENAI_API_VERSION="2025-01-01",
    )
    assert client.api_base == "https://mine.openai.azure.com/"
    assert client.model_name == "my-deployment"
    assert client.api_version == "2025-01-01"


def test_an_unconfigured_endpoint_is_refused():
    """No endpoint default exists here, and that is deliberate.

    Upstream keeps a class-level endpoint so an unconfigured deployment
    behaves as it always has. This tree cannot: the default it inherited was
    one specific tenant's Azure resource, so a user who set a key but no
    endpoint would have sent their prompts to somebody else's deployment.
    Refusing to construct is the safe failure.
    """
    with pytest.raises(ValueError, match="endpoint is not configured"):
        _build()


@pytest.mark.parametrize("field,argument,configured", [
    ("api_base", "https://arg.openai.azure.com/", "https://cfg.openai.azure.com/"),
    ("model_name", "arg-deployment", "cfg-deployment"),
    ("api_version", "2030-01-01", "2029-01-01"),
])
def test_an_explicit_argument_beats_configuration(field, argument, configured):
    settings_field = {
        "api_base": "OPENAI_API_BASE",
        "model_name": "MODEL_NAME",
        "api_version": "OPENAI_API_VERSION",
    }[field]
    settings = _Settings()
    # An endpoint is always required here, so supply one for the cases that
    # are not themselves exercising api_base.
    settings.OPENAI_API_BASE = "https://cfg.openai.azure.com/"
    setattr(settings, settings_field, configured)
    with mock.patch.object(assistants, "get_runtime_settings", return_value=settings), \
         mock.patch.object(assistants, "AsyncAzureOpenAI", lambda **kwargs: object()):
        client = assistants.GPTAssistant(**{field: argument})
    assert getattr(client, field) == argument
