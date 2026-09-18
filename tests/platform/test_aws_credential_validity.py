"""Bedrock availability turns on whether AWS accepts the credential.

Resolving a credential and having a working one are different things, and the
difference decides whether a request for ``claude_sonnet`` is served by Bedrock
or substituted. A stale key that still resolves used to report every Bedrock
provider as configured, so nothing substituted and every call failed at Bedrock
after five retries -- absent credentials degraded gracefully while invalid ones
took the pipeline down.
"""

from __future__ import annotations

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from ascent_platform.llm import availability


@pytest.fixture(autouse=True)
def _clear_cache():
    availability.reset_aws_credentials_cache()
    yield
    availability.reset_aws_credentials_cache()


class _Session:
    """Minimal stand-in for a botocore session."""

    def __init__(self, credentials=object(), sts_raises=None):
        self._credentials = credentials
        self._sts_raises = sts_raises
        self.sts_calls = 0

    def get_credentials(self):
        return self._credentials

    def create_client(self, name, **kwargs):
        assert name == "sts"
        session = self

        class _Client:
            def get_caller_identity(self):
                session.sts_calls += 1
                if session._sts_raises is not None:
                    raise session._sts_raises
                return {"Account": "123456789012"}

        return _Client()


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "GetCallerIdentity")


@pytest.fixture
def patched(monkeypatch):
    def _install(session):
        import botocore.session

        monkeypatch.setattr(botocore.session, "get_session", lambda: session)
        return session

    return _install


def test_credentials_that_sts_accepts_are_available(patched):
    session = patched(_Session())
    assert availability._aws_credentials_available() is True
    assert session.sts_calls == 1


def test_absent_credentials_are_unavailable_without_calling_sts(patched):
    session = patched(_Session(credentials=None))
    assert availability._aws_credentials_available() is False
    assert session.sts_calls == 0, "no credential to check means no reason to call STS"


@pytest.mark.parametrize(
    "code",
    ["InvalidClientTokenId", "UnrecognizedClientException", "ExpiredToken", "SignatureDoesNotMatch"],
)
def test_credentials_sts_rejects_are_unavailable(patched, code):
    """The bug this exists for: resolvable but dead keys must not look configured."""
    patched(_Session(sts_raises=_client_error(code)))
    assert availability._aws_credentials_available() is False


def test_an_unreachable_sts_does_not_condemn_the_credential(patched):
    """A network failure leaves the credential unproven, not disproven.

    Treating unproven as invalid would disable Bedrock on a blip in a
    deployment where it works.
    """
    patched(_Session(sts_raises=EndpointConnectionError(endpoint_url="https://sts.amazonaws.com/")))
    assert availability._aws_credentials_available() is True


def test_an_unrelated_sts_error_does_not_condemn_the_credential(patched):
    patched(_Session(sts_raises=_client_error("ServiceUnavailable")))
    assert availability._aws_credentials_available() is True


def test_the_answer_is_cached_for_the_process(patched):
    session = patched(_Session())
    for _ in range(6):  # once per Bedrock provider on the startup path
        availability._aws_credentials_available()
    assert session.sts_calls == 1, "the probe is a network call; it must not repeat per provider"


def test_the_probe_is_bounded_and_unretried(patched, monkeypatch):
    """A hanging probe on the startup path would cost more than the answer."""
    captured = {}

    class _Recording(_Session):
        def create_client(self, name, **kwargs):
            captured.update(kwargs)
            return super().create_client(name, **kwargs)

    patched(_Recording())
    availability._aws_credentials_available()

    config = captured["config"]
    assert config.connect_timeout is not None and config.connect_timeout <= 5
    assert config.read_timeout is not None and config.read_timeout <= 5
    assert config.retries == {"max_attempts": 0}


def test_rejected_credentials_make_claude_sonnet_resolve_elsewhere(patched, monkeypatch):
    """The point of the whole check, stated end to end."""
    patched(_Session(sts_raises=_client_error("InvalidClientTokenId")))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used-for-calls")
    availability._warned.discard("claude_sonnet")

    assert availability.resolve("claude_sonnet") == "gemini"
