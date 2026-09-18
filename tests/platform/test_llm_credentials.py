"""One credential, one resolution order.

Three sites resolved the Gemini key three ways, and two of them disagreed on
precedence (GEMINI_API_KEY-first in the router, GOOGLE_API_KEY-first in the
client). Nothing failed, because GOOGLE_API_KEY is unset everywhere -- so the
disagreement was invisible right up until someone set both.

These pin the settled order and, more importantly, that all three call sites
now go through it.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from ascent_platform.config.runtime import RuntimeSettings
from ascent_platform.llm import credentials

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"


@pytest.fixture
def settings(monkeypatch):
    """Swap the process-wide RuntimeSettings for one built from given values.

    ``_env_file=None`` and the delenv calls are both required: pydantic-settings
    reads the .env file AND the process environment, so without them a developer
    with a real GEMINI_API_KEY sees "assert <their real key> is None" -- the
    absence cases would never actually test absence.
    """
    def _apply(**values):
        for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(
            credentials,
            "get_runtime_settings",
            lambda: RuntimeSettings(_env_file=None, **values),
        )
    return _apply


def test_explicit_argument_wins(settings):
    settings(GEMINI_API_KEY="from-env")
    assert credentials.gemini_api_key("explicit") == "explicit"


def test_gemini_api_key_beats_google_api_key(settings):
    """The settled order. GEMINI_API_KEY is what deployments actually set."""
    settings(GEMINI_API_KEY="gemini", GOOGLE_API_KEY="google")
    assert credentials.gemini_api_key() == "gemini"


def test_google_api_key_is_accepted_when_it_is_the_only_one(settings):
    settings(GOOGLE_API_KEY="google")
    assert credentials.gemini_api_key() == "google"


def test_none_when_neither_is_set(settings):
    settings()
    assert credentials.gemini_api_key() is None


def test_project_id_falls_back_to_settings(settings):
    settings(GOOGLE_CLOUD_PROJECT="proj")
    assert credentials.google_cloud_project() == "proj"
    assert credentials.google_cloud_project("explicit") == "explicit"


# --------------------------------------------------------------------------
# The part that actually prevents the regression
# --------------------------------------------------------------------------

_CREDENTIAL_VARS = {"GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT"}

_CALL_SITES = [
    "ascent_platform/llm/router.py",
    "ascent_platform/llm/clients/gemini.py",
    "ascent_platform/llm/clients/assistants.py",
]


@pytest.mark.parametrize("relpath", _CALL_SITES)
def test_no_call_site_reads_the_credential_itself(relpath):
    """Resolution order only stays single if nobody re-implements it.

    AST rather than grep: the docstrings in these files legitimately name the
    variables while explaining why they must not be read here.
    """
    tree = ast.parse((SRC / relpath).read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "attr", None)
        if name not in {"getenv", "get"}:
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and arg.value in _CREDENTIAL_VARS:
                offenders.append(f"{relpath}:{node.lineno} reads {arg.value}")
    assert not offenders, (
        f"{offenders} -- use ascent_platform.llm.credentials so there is one "
        f"precedence, not one per call site."
    )


def test_the_scan_would_catch_a_reintroduced_read(tmp_path):
    """All three files being clean and the detector being broken look the same."""
    probe = tmp_path / "probe.py"
    probe.write_text("import os\nk = os.getenv('GEMINI_API_KEY')\n")
    tree = ast.parse(probe.read_text())
    found = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", None) == "getenv"
        and any(isinstance(a, ast.Constant) and a.value in _CREDENTIAL_VARS for a in n.args)
    ]
    assert found, "detector missed a direct os.getenv('GEMINI_API_KEY')"


def test_router_gemini_fallback_actually_calls_the_resolver():
    """This branch had no coverage, which is how a NameError survived the whole
    suite: the import for `gemini_api_key` failed to land in router.py, ruff
    caught it, 965 tests did not. Exercise the branch, not just the resolver.
    """
    from dataclasses import replace

    from ascent_platform.llm.clients.gemini import GeminiClient
    from ascent_platform.llm.router import DEFAULT_REGISTRY, LLMRouter

    model = next(
        m for m in DEFAULT_REGISTRY.list_models()
        if DEFAULT_REGISTRY.get(m).client_class is GeminiClient
    )
    # No explicit key on the router and no api_key_env on the config, so the
    # only way to a value is through the shared resolver.
    config = replace(DEFAULT_REGISTRY.get(model), api_key_env=None)
    router = LLMRouter(fallback_model_name=model)
    router._gemini_api_key = None

    import ascent_platform.llm.router as router_module

    original = router_module.gemini_api_key
    try:
        router_module.gemini_api_key = lambda explicit=None: "resolved-by-shared"
        assert router._resolve_api_key(config) == "resolved-by-shared"
    finally:
        router_module.gemini_api_key = original
