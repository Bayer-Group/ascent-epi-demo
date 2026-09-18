"""A retryable provider error must actually be retried.

Every provider SDK wraps its transport failures in its own exception type, and
none of them subclass the builtins. Listing ``ConnectionError`` and
``TimeoutError`` in ``retry_on`` therefore matches a raw asyncio timeout and
nothing a provider raises, so a rate limit or a 502 ends the call on first
sight -- with the retry decorator still in place, reading as if it were covered.
"""

from __future__ import annotations

import google.genai.errors as genai_errors
import openai
import pytest

from ascent_platform.llm.clients.assistants import TRANSIENT_LLM_ERRORS


@pytest.mark.parametrize(
    "exc_type",
    [
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.InternalServerError,
        genai_errors.ServerError,
        ConnectionError,
        TimeoutError,
    ],
)
def test_transient_failures_are_retryable(exc_type):
    assert issubclass(exc_type, TRANSIENT_LLM_ERRORS)


@pytest.mark.parametrize(
    "exc_type",
    [
        openai.BadRequestError,
        openai.AuthenticationError,
        openai.PermissionDeniedError,
        openai.NotFoundError,
        ValueError,
    ],
)
def test_permanent_failures_are_not_retried(exc_type):
    """Retrying these burns the backoff and fails anyway."""
    assert not issubclass(exc_type, TRANSIENT_LLM_ERRORS)


def test_asyncio_timeouts_are_still_covered():
    import asyncio

    assert issubclass(asyncio.TimeoutError, TRANSIENT_LLM_ERRORS)


def test_no_client_still_lists_only_the_builtins():
    """The shape of the original defect, pinned so it cannot come back."""
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "src/ascent_platform/llm/clients/assistants.py"
    assert "retry_on=(ConnectionError, TimeoutError)" not in src.read_text()
