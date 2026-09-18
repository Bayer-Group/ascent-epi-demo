"""The configured Gemini model has to be the one that gets used.

GEMINI_DEFAULT_MODEL_NAME governed only GeminiClient. GeminiAssistant -- the
class create_assistant builds, and therefore the one every CLUES and
disambiguation path runs on -- fell back to a hardcoded class constant, and ten
call sites passed a hardcoded model name of their own on top of that. Changing
the setting moved neither. These pin that it now does.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from ascent_platform.config.runtime import RuntimeSettings
from ascent_platform.llm.clients import assistants

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"


@pytest.fixture
def gemini(monkeypatch):
    """A GeminiAssistant built without touching the network."""
    monkeypatch.setattr(assistants, "gemini_api_key", lambda *_a, **_k: "test-key")
    monkeypatch.setattr(assistants.genai, "Client", lambda **_k: object())
    return assistants.GeminiAssistant


def test_unspecified_model_comes_from_settings(gemini, monkeypatch):
    monkeypatch.setattr(assistants, "get_runtime_settings", lambda: RuntimeSettings(_env_file=None, GEMINI_DEFAULT_MODEL_NAME="gemini-9.9-test"))
    assert gemini().model_name == "gemini-9.9-test"


def test_explicit_none_also_comes_from_settings(gemini, monkeypatch):
    """Callers pass model_name=None to mean "the default".

    A plain argument default does not cover this, and Gemini answers an empty
    model id with a 404 that reads like a missing model rather than a missing
    argument.
    """
    monkeypatch.setattr(assistants, "get_runtime_settings", lambda: RuntimeSettings(_env_file=None, GEMINI_DEFAULT_MODEL_NAME="gemini-9.9-test"))
    assert gemini(model_name=None).model_name == "gemini-9.9-test"


def test_an_explicit_model_still_wins(gemini, monkeypatch):
    monkeypatch.setattr(assistants, "get_runtime_settings", lambda: RuntimeSettings(_env_file=None, GEMINI_DEFAULT_MODEL_NAME="gemini-9.9-test"))
    assert gemini(model_name="gemini-1.0-explicit").model_name == "gemini-1.0-explicit"


def test_the_setting_is_what_ships():
    assert RuntimeSettings.model_fields["GEMINI_DEFAULT_MODEL_NAME"].default == "gemini-3.8-flash"


def test_no_model_name_default_is_a_hardcoded_gemini_model():
    """A ratchet on *defaults* specifically.

    Not on every literal: the registry in llm/router.py is keyed by model name,
    and one branch in assistants.py tests `"gemini-3-pro" in model_name` to pick
    a thinking level. Those are data and behaviour. What shadows the setting is
    a default -- an argument default, or a {"model": ...} entry standing in for
    one -- because it reaches the client as an explicit value and the `or`
    fallback never runs.

    The failure this prevents is silent: the pipeline keeps working, on a model
    nobody selected, and the setting that was supposed to select it reads as
    though it had.
    """
    model_keys = {"model", "model_name"}
    pattern = re.compile(r"gemini-[0-9].*")

    def is_model_literal(node) -> bool:
        return isinstance(node, ast.Constant) and isinstance(node.value, str) and bool(pattern.fullmatch(node.value))

    offenders = []
    for path in SRC.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                pairs = list(zip(args.args[-len(args.defaults) :] if args.defaults else [], args.defaults))
                pairs += [(k, d) for k, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
                for arg, default in pairs:
                    if arg.arg in model_keys and is_model_literal(default):
                        offenders.append(f"{path.relative_to(SRC)}:{default.lineno} {arg.arg}={default.value}")
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value in model_keys and is_model_literal(value):
                        offenders.append(f"{path.relative_to(SRC)}:{value.lineno} {{{key.value!r}: {value.value!r}}}")

    assert not offenders, "these defaults shadow GEMINI_DEFAULT_MODEL_NAME:\n  " + "\n  ".join(offenders)
