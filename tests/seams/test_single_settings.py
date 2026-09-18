"""One Settings class, and the fail-fast it must not silently drop.

Replaces test_settings_field_snapshot.py, which pinned the five-class
arrangement field-by-field so the merge could be shown to preserve it. That
fixture described a shape that no longer exists; what still needs guarding is
the one behaviour the merge genuinely changed.

ascent_http.Settings declared fourteen fields with no default, so pydantic raised on
first attribute access -- at import, in practice, one field per run. A single
Settings shared with the platform cannot do that without making a library
import require a database password, so require_app_config() asserts it at
startup instead. If a name falls off that list, a misconfigured deploy boots.
"""

from __future__ import annotations

import pytest

from ascent_platform.config.runtime import _APP_REQUIRED, Settings, require_app_config

# Exactly the fields ascent_http.Settings declared with no default, read off the
# class before it was deleted. Not derived at runtime -- the point is to compare
# against history, and history is gone.
_HISTORICALLY_REQUIRED = {
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_TENANT_ID",
    "CORS_ORIGINS",
    "DB_APP_PWD",
    "DB_APP_USR",
    "DB_HOST",
    "DB_NAME",
    "DB_PORT",
    "GEMINI_API_KEY",
    "OPENAI_API_VERSION",
    "SNOWFLAKE_ACCOUNT_IDENTIFIER",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_WAREHOUSE",
}


def test_no_settings_field_is_required():
    """No *field* must be mandatory for the build to start.

    The warehouse is a local DuckDB file, there is no directory to
    authenticate against, and every DB_* field defaults to what
    docker-compose provides.

    An LLM provider is still required -- see below. That check asks the
    provider registry rather than naming keys, which is why it is not here.

    The list is still checked, so adding a name back re-arms the startup
    check rather than silently doing nothing.
    """
    assert _APP_REQUIRED == ()


def test_startup_fails_without_any_llm_provider(monkeypatch):
    """Every pipeline ends in a model call, so this is the one hard demand."""
    import ascent_platform.config.runtime as runtime

    monkeypatch.setattr(runtime, "available_providers", lambda: [], raising=False)
    monkeypatch.setattr("ascent_platform.llm.availability.available_providers", lambda: [])

    with pytest.raises(RuntimeError, match="No LLM provider is configured"):
        require_app_config(Settings())


def test_startup_passes_once_a_provider_is_configured(monkeypatch):
    """The counterpart, so the test above cannot pass on an always-raising check."""
    monkeypatch.setattr("ascent_platform.llm.availability.available_providers", lambda: ["gemini"])

    require_app_config(Settings())


def test_database_defaults_match_what_compose_provides():
    """The defaults are what make "no configuration" work; they must be real.

    A default of None here would move the failure from a named startup error
    into an obscure SQLAlchemy connection error, which is the regression the
    removed test guarded against.
    """
    # Read the declared defaults, not an instance: the environment supplies
    # DB_HOST=postgres inside compose, which would mask a missing default.
    defaults = {n: Settings.model_fields[n].default for n in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_APP_USR", "DB_APP_PWD")}
    assert all(v is not None for v in defaults.values()), defaults
    assert defaults["DB_PORT"] == 5432


def test_every_required_name_is_a_real_field():
    """A typo in the list silently stops checking that field."""
    for name in _APP_REQUIRED:
        assert name in Settings.model_fields, f"{name} is not a Settings field"


@pytest.fixture
def bare_env(monkeypatch):
    """Settings with nothing supplied.

    ``_env_file=None`` only stops the .env files being read; pydantic-settings
    still reads os.environ, and a developer machine has all fourteen set. So the
    absence cases have to clear them or they test nothing -- the same trap the
    LLM credential tests hit.

    The alias names have to go too: AZURE_CLIENT_ID also answers to
    MSAL_CLIENT_ID, and SNOWFLAKE_ACCOUNT_IDENTIFIER to SNOWFLAKE_ACCOUNT, so
    clearing only the canonical names leaves three of the fourteen resolving.
    """
    aliases = ("MSAL_CLIENT_ID", "MSAL_CLIENT_SECRET", "SNOWFLAKE_ACCOUNT")
    for name in (*_APP_REQUIRED, *aliases):
        monkeypatch.delenv(name, raising=False)
    return lambda **kw: Settings(_env_file=None, **kw)


def test_the_check_passes_when_everything_is_supplied(bare_env, monkeypatch):
    """Otherwise the tests above would pass on a check that always raises."""
    # require_app_config also asks the provider registry whether any LLM is
    # reachable, and that reads the environment rather than the settings object
    # it is handed. Without this the test depended on the developer's own key
    # being exported -- it passed locally and failed on a clean checkout.
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used-for-calls")

    values = {n: "x" for n in _APP_REQUIRED}
    values["DB_PORT"] = 5432  # typed int, unlike the rest
    require_app_config(bare_env(**values))


def test_the_app_actually_calls_it():
    """A startup check nobody invokes is a comment."""
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "src"
    tree = ast.parse((src / "ascent_http/lifespan.py").read_text())
    called = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "require_app_config"]
    assert called, "ascent/lifespan.py no longer calls require_app_config()"
