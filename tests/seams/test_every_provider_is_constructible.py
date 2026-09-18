"""Every registered provider must be constructible with no arguments.

``create_assistant(assistant_type)`` builds a client with whatever kwargs the
caller happened to pass, which is often none. On top of that the availability
resolver substitutes providers, and a substitution deliberately drops the
caller's ``model_name`` because model names belong to the provider that was
asked for.

So any provider can be asked to build itself from nothing. ``MistralAssistant``
could not -- ``model_name`` was a required positional argument -- and the
failure mode was bad: a deployment configured only with ``MISTRAL_API_KEY``
passed startup, the resolver correctly chose the one provider available, and
the first call raised ``TypeError``. Nothing in that chain was wrong except the
constructor.

This pins the property for every provider in the registry rather than for the
one that happened to break, since the registry is where new providers arrive.
"""

from __future__ import annotations

import inspect

import pytest

from ascent_platform.llm.clients.assistants import assistant_registry


def _required_arguments(cls) -> list[str]:
    signature = inspect.signature(cls.__init__)
    return [
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.default is inspect.Parameter.empty
        and parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
    ]


@pytest.mark.parametrize("assistant_type", sorted(assistant_registry))
def test_constructor_needs_no_arguments(assistant_type: str) -> None:
    """A provider must not demand arguments create_assistant cannot supply."""
    required = _required_arguments(assistant_registry[assistant_type])
    assert not required, (
        f"{assistant_type} requires {required}. create_assistant() builds "
        f"providers with no arguments, and provider substitution drops "
        f"model_name, so this raises TypeError at the first call rather than "
        f"at configuration time. Give the argument a default."
    )


@pytest.mark.parametrize("assistant_type", sorted(assistant_registry))
def test_declares_a_default_model(assistant_type: str) -> None:
    """A provider must know which model to use when it is not told.

    Substitution drops the requested model name, so the substitute has to
    supply its own; otherwise it inherits ``None`` and fails at the API call
    with a message about an empty model id.
    """
    cls = assistant_registry[assistant_type]
    defaults = [getattr(cls, name, None) for name in ("DEFAULT_MODEL", "MODEL_NAME", "DEFAULT_MODEL_NAME")]
    # Or it names a settings field and resolves the value at construction, so
    # the model is configurable rather than compiled in.
    setting = getattr(cls, "DEFAULT_MODEL_SETTING", None)
    if setting:
        from ascent_platform.config.runtime import RuntimeSettings

        assert setting in RuntimeSettings.model_fields, f"{assistant_type} names {setting}, which is not a setting"
        defaults.append(RuntimeSettings.model_fields[setting].default)
    signature = inspect.signature(cls.__init__)
    model_parameter = signature.parameters.get("model_name")
    from_signature = model_parameter is not None and model_parameter.default not in (inspect.Parameter.empty, None)
    assert any(defaults) or from_signature, (
        f"{assistant_type} has no default model. It will be constructed with model_name=None whenever it is used as a substitute."
    )
