"""The injection point for provider clients.

The provider clients -- Azure OpenAI, Gemini, five Bedrock models, DeepSeek,
Mistral -- are infrastructure: SDK handles, credentials, retry, token counting.
They know nothing about cohorts, criteria or OMOP, and this is where callers
ask for one.

``create_assistant`` keeps its existing contract, including the wart: an
unregistered type logs and returns ``None`` rather than raising. Changing it is
a behaviour change for every call site at once.
"""

from __future__ import annotations

from typing import Dict, Type

from ascent_platform.llm.clients.assistants import (
    BaseAssistant,
    assistant_registry,
    create_assistant,
    register_assistant,
)

__all__ = [
    "BaseAssistant",
    "assistant_registry",
    "available_assistant_types",
    "create_assistant",
    "register_assistant",
]


def available_assistant_types() -> Dict[str, Type[BaseAssistant]]:
    """The registered provider types, as a copy.

    Handing out the registry itself let a caller mutate the shared mapping;
    the model-policy test reads this to check that every configured type
    resolves.
    """
    return dict(assistant_registry)
