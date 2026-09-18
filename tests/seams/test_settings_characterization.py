"""Characterization tests for the Settings object.

Pin what exists, warts included, so a change to configuration can be shown to
preserve behaviour.

The subtle one is precedence. With more than one loader, the one that runs
`load_dotenv(..., override=True)` first wins for any variable present in both
file lists, because a later call without `override` cannot overwrite what is
already set — an accident of import order, not a decision, and one that can
silently flip which .env applies in a deployed container. A single loader owns
this, and the precedence it implements is asserted behaviourally below.
"""

import os

import ascent_domain.models.data_definitions  # noqa: F401  (import-cycle order)


def test_exactly_one_settings_class_exists():
    """One class, many names.

    Separate settings classes duplicate fields, and duplicated fields disagree
    on type and default. Every settings name resolves to the same class.
    """
    from ascent_domain.config import DomainSettings
    from ascent_http.settings import Settings as BackendSettings
    from ascent_platform.config.runtime import RuntimeSettings, Settings
    from ascent_platform.config.runtime import Settings as AiSettings

    assert BackendSettings is Settings
    assert AiSettings is Settings
    assert DomainSettings is Settings
    assert RuntimeSettings is Settings


def test_only_one_basesettings_subclass_is_declared_in_src():
    """The names being aliases is not enough; a new class must not appear."""
    import ast
    import pathlib as _pathlib

    src = _pathlib.Path(__file__).resolve().parents[2] / "src"
    declarations = []
    for path in src.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef) and any(
                getattr(b, "id", getattr(b, "attr", "")) in {"BaseSettings", "PlatformSettings"} for b in node.bases
            ):
                declarations.append(f"{path.relative_to(src)}:{node.lineno} {node.name}")
    assert len(declarations) == 1, f"expected exactly one settings class, found {len(declarations)}: {declarations}"


def test_the_two_formerly_undeclared_keys_are_declared():
    """Under extra="allow" an undeclared key resolves only when it happens to
    be set, so an unset OPENAI_API_KEY_GPT4O or OPENAI_API_KEY_O1 raises
    AttributeError inside a client constructor instead of failing validation.
    Declaring them is what lets extra be "ignore".
    """
    from ascent_platform.config.runtime import Settings

    for name in ("OPENAI_API_KEY_GPT4O", "OPENAI_API_KEY_O1"):
        assert name in Settings.model_fields, f"{name} is undeclared again"
    assert Settings.model_config.get("extra") == "ignore"


def test_non_omop_config_no_longer_requires_a_key_at_import():
    """Insisting on credentials is the application's job, not the platform's.

    A settings class that declares GEMINI_API_KEY required and instantiates at
    import makes every non-OMOP workflow unimportable without the key. There is
    no such module: the pipeline is ascent_domain.non_omop and its settings live
    in RuntimeSettings, where they are optional.
    """
    import importlib

    import pytest

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("ascent_non_omop.settings")

    from ascent_platform.config.runtime import RuntimeSettings

    assert not RuntimeSettings.model_fields["GEMINI_API_KEY"].is_required()
    # That a default exists at all is the property under test: the setting has
    # to be usable without being supplied. The exact model is not -- pinning the
    # string here would break a test about import-time credential requirements
    # on every model bump.
    fallback = RuntimeSettings.model_fields["FALLBACK_MODEL_NAME"]
    assert not fallback.is_required()
    assert isinstance(fallback.default, str) and fallback.default


def test_env_precedence_override_pass_beats_gap_fill_pass(tmp_path, monkeypatch):
    """Behavioural version of the precedence pin: files in the override pass
    win over anything already set, and the gap-fill pass only supplies what is
    missing.
    """
    from ascent_platform.config import env as envmod

    (tmp_path / ".env").write_text("PRECEDENCE_PROBE=from-env-file\n")
    (tmp_path / ".env.local").write_text("PRECEDENCE_PROBE=from-env-local\n")

    monkeypatch.setenv("ENV_LOCAL_PATH", str(tmp_path))
    monkeypatch.setenv("PRECEDENCE_PROBE", "preexisting")
    monkeypatch.setattr(envmod, "_loaded_bases", set())
    monkeypatch.chdir(tmp_path)

    envmod.load_environment(force=True)

    # .env.local is later in the override list, so it wins; and the override
    # pass beats the value that was already in os.environ.
    assert os.environ["PRECEDENCE_PROBE"] == "from-env-local"


def test_load_environment_is_idempotent_per_base(tmp_path, monkeypatch):
    """A second call for the same base must not re-read files."""
    from ascent_platform.config import env as envmod

    (tmp_path / ".env").write_text("IDEMPOTENCY_PROBE=first\n")
    monkeypatch.setenv("ENV_LOCAL_PATH", str(tmp_path))
    monkeypatch.setattr(envmod, "_loaded_bases", set())
    monkeypatch.chdir(tmp_path)

    envmod.load_environment()
    (tmp_path / ".env").write_text("IDEMPOTENCY_PROBE=second\n")
    envmod.load_environment()

    assert os.environ["IDEMPOTENCY_PROBE"] == "first", "second call re-read the files"


def test_constructing_ascent_ai_settings_does_not_touch_root_logging():
    """Constructing Settings must be inert.

    A setup_logging() call in Settings.__init__ would run
    logging.basicConfig(force=True) and attach a FileHandler to the ROOT logger
    merely because the object was built, tearing down the QueueHandler from
    ascent_http.util.logging -- the one that keeps log writes off the event loop
    -- at whatever moment some module happens to import first.
    """
    import logging

    from ascent_platform.config.runtime import Settings as AiSettings

    assert not hasattr(AiSettings, "setup_logging")

    root = logging.getLogger()
    before = list(root.handlers)
    AiSettings()
    assert list(root.handlers) == before, "constructing Settings changed the root logger's handlers again"


def test_settings_are_not_constructed_at_import():
    """The module-level `settings` is a lazy proxy, so importing the package
    does not read and validate the environment."""
    from ascent_http.settings import settings as backend_settings
    from ascent_platform.config.runtime import _LazySettings
    from ascent_platform.config.runtime import settings as ai_settings

    assert isinstance(backend_settings, _LazySettings)
    assert type(ai_settings).__name__ == "_LazySettings"
    # and it still behaves like the real thing
    assert isinstance(backend_settings.RELEASE_VERSION, str)


def test_backend_settings_project_root_resolves_to_the_repo():
    """PROJECT_ROOT is computed by counting parents up from settings.py, so it
    breaks if the package moves depth.
    """
    from ascent_http.settings import PROJECT_ROOT

    assert (PROJECT_ROOT / "pyproject.toml").exists(), f"PROJECT_ROOT no longer points at the repo root: {PROJECT_ROOT}"


def test_non_omop_model_defaults_are_all_registered():
    """The settings/registry contract — a mismatch here degrades silently."""
    from ascent_http.settings import settings
    from ascent_platform.llm.router import DEFAULT_REGISTRY

    assert DEFAULT_REGISTRY.has(settings.NON_OMOP_SQL_PREPARATION_MODEL_NAME)
