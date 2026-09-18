"""Which providers this deployment can actually reach, and what to use instead.

Call sites ask for a provider by name -- ``create_assistant("gpt")``,
``"claude_sonnet"``, ``"gemini"`` -- chosen when the code was written because
that model suited that task. In the original internal deployment every one of
those is configured, so the names are effectively roles and the hardcoding is
invisible.

Outside, it breaks immediately. A user sets one key, and the eight call sites
that name a different provider fail: set ``GEMINI_API_KEY`` and the
personalised-questions job still asks for Azure OpenAI, gets an unconfigured
endpoint, and dies. Whichever single key you bring, parts of the product stop
working -- and they fail at the call site, far from the configuration that
caused it.

So a requested provider is resolved against what is configured:

1. an explicit alias, if the deployment set one -- ``LLM_PROVIDER_ALIASES``
   is the predictable, say-what-you-mean option;
2. the request itself, when that provider has credentials;
3. otherwise the configured fallback, or the sole configured provider.

Step 3 is a real behaviour change and is logged once per substitution: quietly
answering with a different model than the code asked for is the kind of thing
that should appear in a log, not just in a docstring.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# What each registered provider needs before it can be constructed. Each entry
# is a tuple of requirement groups: every group must be satisfied, and a group
# is satisfied by any one of its names. Endpoint-and-key providers need both,
# which a flat any-of list cannot express -- "gpt" used to be declared as
# OPENAI_API_BASE alone, so setting an endpoint with no key reported the
# provider available and then failed on the first call.
_REQUIREMENTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "gpt": (("OPENAI_API_BASE",), ("OPENAI_API_KEY_GPT4O", "OPENAI_API_KEY")),
    "gpt-o1": (("OPENAI_API_BASE",), ("OPENAI_API_KEY_O1", "OPENAI_API_KEY")),
    "gemini": (("GEMINI_API_KEY", "GOOGLE_API_KEY"),),
    "mistral": (("MISTRAL_API_KEY",),),
    "anthropic": (("ANTHROPIC_API_KEY",),),
    "deepseek": (
        ("AZURE_INFERENCE_ENDPOINT_DEEPSEEK_R1",),
        ("AZURE_INFERENCE_KEY_DEEPSEEK_R1_EAST_US2",),
    ),
}

# Bedrock is not a key-in-the-environment provider: boto3 resolves credentials
# from env vars, a shared profile, SSO, or an instance/task role, and the
# clients default region_name themselves. Listing AWS_REGION as a requirement
# therefore checked the one thing that does not matter -- and since AWS_REGION
# is set in most container images, all six Bedrock providers reported available
# on a box with no credentials at all, which let require_app_config() pass with
# nothing usable. Ask boto3 instead.
_BEDROCK_PROVIDERS = frozenset(
    {
        "bedrock_llama3",
        "bedrock_gptoss",
        "bedrock_qwen",
        "bedrock_kimi",
        "bedrock_mistral",
        "claude_sonnet",
    }
)

_aws_credentials: Optional[bool] = None


# Error codes STS returns when the credentials themselves are the problem.
# GetCallerIdentity needs no IAM permission, so a rejection here is about the
# credential, never about what it is allowed to do.
_INVALID_CREDENTIAL_CODES = frozenset(
    {
        "InvalidClientTokenId",
        "UnrecognizedClientException",
        "ExpiredToken",
        "ExpiredTokenException",
        "InvalidAccessKeyId",
        "SignatureDoesNotMatch",
        "AuthFailure",
    }
)


def _aws_credentials_available() -> bool:
    """Whether boto3 can resolve credentials that AWS actually accepts.

    Resolving is not the same as working, and the difference is not cosmetic:
    a stale key still resolves, so every Bedrock provider reported itself
    configured, no substitution happened, and each call failed at Bedrock after
    five retries. Absent credentials degraded gracefully to another provider
    while invalid ones took the pipeline down -- the wrong way round.

    So the resolved credential is put to AWS once, via GetCallerIdentity, which
    requires no IAM permission and therefore answers about the credential
    rather than about its grants.

    An unreachable STS is not a rejection. A network failure leaves the
    credential unproven, and treating unproven as invalid would disable Bedrock
    on a blip in a deployment where it works; only an explicit auth error
    counts against it.

    Cached because the probe is a network call and this is asked once per
    Bedrock provider -- six times -- on the startup path.
    """
    global _aws_credentials
    if _aws_credentials is not None:
        return _aws_credentials

    try:
        import botocore.session
    except Exception:  # noqa: BLE001 - botocore absent
        _aws_credentials = False
        return _aws_credentials

    session = botocore.session.get_session()
    try:
        resolved = session.get_credentials() is not None
    except Exception:  # noqa: BLE001 - the resolver itself failed
        resolved = False
    if not resolved:
        _aws_credentials = False
        return _aws_credentials

    try:
        from botocore.config import Config
        from botocore.exceptions import ClientError

        # Bounded and un-retried: this runs on the startup path, and a hanging
        # probe would cost more than the answer is worth.
        try:
            sts = session.create_client(
                "sts",
                config=Config(connect_timeout=3, read_timeout=3, retries={"max_attempts": 0}),
            )
        except ValueError as exc:
            # A client that cannot be constructed -- an empty region, which is
            # what compose passes when AWS_REGION is unset -- is a
            # configuration answer, not an unreachable endpoint. Bedrock will
            # not work either, so report it as unconfigured and let another
            # provider take the call.
            logger.warning("Cannot build an STS client (%s); treating Bedrock providers as unconfigured.", exc)
            _aws_credentials = False
            return _aws_credentials
        sts.get_caller_identity()
        _aws_credentials = True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in _INVALID_CREDENTIAL_CODES:
            logger.warning(
                "AWS credentials resolved but were rejected by STS (%s); treating Bedrock "
                "providers as unconfigured so requests for them fall back to a working provider.",
                code,
            )
            _aws_credentials = False
        else:
            logger.warning("Could not verify AWS credentials with STS (%s); assuming they work.", code)
            _aws_credentials = True
    except Exception as exc:  # noqa: BLE001 - network, endpoint or clock problems
        logger.warning("Could not reach STS to verify AWS credentials (%s); assuming they work.", type(exc).__name__)
        _aws_credentials = True

    return _aws_credentials


def reset_aws_credentials_cache() -> None:
    """Test seam; also lets a process pick up credentials that arrived late."""
    global _aws_credentials
    _aws_credentials = None


# Preference order when substituting, best general-purpose first.
_PREFERENCE = (
    "gemini",
    "anthropic",
    "gpt",
    "claude_sonnet",
    "mistral",
    "deepseek",
    "bedrock_llama3",
)

_warned: set[str] = set()


def _configured(name: str) -> bool:
    """Whether *name* has the configuration it needs to be constructed."""
    import os

    from ascent_platform.config.runtime import get_runtime_settings

    if name in _BEDROCK_PROVIDERS:
        return _aws_credentials_available()

    groups = _REQUIREMENTS.get(name)
    if groups is None:
        return True  # unknown provider: let the registry decide
    settings = get_runtime_settings()
    return all(any(getattr(settings, key, None) or os.environ.get(key) for key in group) for group in groups)


def available_providers() -> list[str]:
    """Every registered provider this deployment can construct."""
    from ascent_platform.llm.clients.assistants import assistant_registry

    return [name for name in assistant_registry if _configured(name)]


def resolve(assistant_type: str) -> Optional[str]:
    """The provider to actually use for a request for *assistant_type*."""
    from ascent_platform.config.runtime import get_runtime_settings

    settings = get_runtime_settings()

    alias = (settings.LLM_PROVIDER_ALIASES or {}).get(assistant_type)
    if alias:
        return alias

    if _configured(assistant_type):
        return assistant_type

    fallback = settings.LLM_FALLBACK_PROVIDER
    if fallback and _configured(fallback):
        chosen = fallback
    else:
        usable = available_providers()
        chosen = next((p for p in _PREFERENCE if p in usable), None) or (usable[0] if usable else None)

    if chosen is None:
        return None

    if assistant_type not in _warned:
        _warned.add(assistant_type)
        logger.warning(
            "'%s' is not configured; using '%s' instead. Set LLM_PROVIDER_ALIASES to choose deliberately, or configure '%s'.",
            assistant_type,
            chosen,
            assistant_type,
        )
    return chosen


def requirements_summary() -> str:
    """What each provider needs, for the startup error.

    Generated from the same table the resolver uses, so the message cannot
    describe a provider that does not exist or omit one that does.
    """
    known = tuple(_REQUIREMENTS) + tuple(sorted(_BEDROCK_PROVIDERS))
    ordered = _PREFERENCE + tuple(n for n in known if n not in _PREFERENCE)

    lines = []
    for name in ordered:
        if name in _BEDROCK_PROVIDERS:
            requirement = "AWS credentials resolvable by boto3 (env, profile, SSO or instance role)"
        else:
            groups = _REQUIREMENTS.get(name)
            if not groups:
                continue
            requirement = " and ".join(
                # Parenthesise a choice so "A and B or C" cannot be misread as
                # offering C on its own when C is an alternative to B.
                f"({' or '.join(group)})" if len(group) > 1 and len(groups) > 1 else " or ".join(group)
                for group in groups
            )
        lines.append(f"  {name:<16} {requirement}")
    return "\n".join(lines)
