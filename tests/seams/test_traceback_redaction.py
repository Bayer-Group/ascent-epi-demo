"""The 500 handler returns this traceback to the CALLER and publishes it to SNS.

build_enhanced_traceback_from_exc dumps repr() of every argument on every
project frame. A frame holding a bearer token, key or password as a named
argument would therefore hand that credential to anyone who can read the
response or the notification email.

Two things make this test fiddly, both worth knowing:

* The frame filter is a SUBSTRING TEST ON THE ABSOLUTE FILENAME against
  ``PROJECT_MODULES``. A probe defined in <stdin> matches nothing, captures no
  frame, and passes while asserting nothing -- hence
  ``test_the_probe_actually_captures_a_project_frame``.
* That same filter matches the CHECKOUT DIRECTORY, so this test file is itself
  "project code" and any argument of a helper here would be dumped too. The
  probe source is therefore handed over through a module global rather than as
  a function argument -- otherwise the test fails on the harness's own frame,
  which has nothing to do with redaction.
"""

import ast
import pathlib

import pytest

import ascent_domain.models.data_definitions  # noqa: F401  (import-cycle order)

_PROBE_SRC = ""
_PROBE_FILENAME = "/app/src/ascent_platform/snowflake/session.py"


def _traceback_for(entry: str) -> str:
    """Compile _PROBE_SRC under a project-looking path and capture its traceback.

    Takes no argument that could carry the secret; see the module docstring.
    """
    from ascent_http.util.error import build_enhanced_traceback_from_exc

    namespace: dict = {}
    exec(compile(_PROBE_SRC, _PROBE_FILENAME, "exec"), namespace)
    try:
        namespace[entry]()
    except RuntimeError as exc:
        return build_enhanced_traceback_from_exc(exc)
    raise AssertionError("the probe did not raise")


def _run(src: str, entry: str = "drive") -> str:
    global _PROBE_SRC
    _PROBE_SRC = src
    return _traceback_for(entry)


def _probe(arg_name: str, value: str = "SUPER_SECRET_VALUE") -> str:
    return f"def probe({arg_name}):\n    raise RuntimeError('boom')\ndef drive():\n    probe({value!r})\n"


def test_the_probe_actually_captures_a_project_frame():
    """Guards the test itself: if the filter stops matching, the assertions
    below would all pass vacuously."""
    assert "probe" in _run(_probe("oauth_token"))


@pytest.mark.parametrize("arg_name", ["oauth_token", "api_key", "private_key_pem_b64", "password", "client_secret"])
def test_sensitive_argument_names_are_redacted(arg_name):
    tb = _run(_probe(arg_name))
    assert "SUPER_SECRET_VALUE" not in tb, f"{arg_name} leaked its value into the traceback"
    assert "<redacted>" in tb


def test_non_sensitive_arguments_survive_so_the_traceback_stays_useful():
    tb = _run(_probe("database_name", "SYNTHETIC_CLAIMS"))
    assert "'SYNTHETIC_CLAIMS'" in tb


def test_huge_values_are_truncated():
    src = "def probe(rows):\n    raise RuntimeError('boom')\ndef drive():\n    probe(['x' * 50] * 200)\n"
    assert "truncated" in _run(src)


def test_redaction_is_applied_at_every_argument_position():
    """AST, not grep: the dumper renders positional, *args and **kwargs
    separately, and an unguarded branch would leak just as effectively."""
    source = (pathlib.Path(__file__).resolve().parents[2] / "src/ascent_http/util/error.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "build_enhanced_traceback_from_exc")
    renders = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_render_arg"]
    assert len(renders) == 3, f"expected _render_arg at all 3 argument positions, found {len(renders)}"
