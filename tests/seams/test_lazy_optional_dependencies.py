"""Dependencies a library imports on our behalf, at call time.

An AST scan of our source cannot see these: nothing in this repo does
``import tabulate``, yet ``DataFrame.to_markdown()`` fails without it. A
dependency list derived from such a scan drops tabulate as unused and the image
ships without it. The unit suite stays green because it mocks the LLM path; the
failure surfaces only in e2e, as ``ImportError: Missing optional dependency
'tabulate'`` while building a prompt -- i.e. a production bug on the OMOP QA
path, not a test-only one.

So the check runs the other way round: find the CALL SITES, then assert the
module each one needs is importable. Add a row whenever a pandas (or similar)
optional feature starts being used.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"

# (attribute or function called, module pandas imports for it, why)
LAZY_DEPENDENCIES = [
    ("to_markdown", "tabulate", "pandas renders markdown tables through tabulate"),
    ("read_excel", "openpyxl", "pandas reads .xlsx through openpyxl"),
    ("to_excel", "openpyxl", "pandas writes .xlsx through openpyxl"),
    ("to_parquet", "pyarrow", "pandas writes parquet through pyarrow"),
    ("read_parquet", "pyarrow", "pandas reads parquet through pyarrow"),
]


def _call_sites(name: str) -> list[str]:
    """Every call of ``.name(...)`` or ``name(...)`` in src/, as path:line."""
    sites = []
    for path in SRC.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = getattr(func, "attr", None) or getattr(func, "id", None)
            if called == name:
                sites.append(f"{path.relative_to(SRC)}:{node.lineno}")
    return sites


@pytest.mark.parametrize("call,module,why", LAZY_DEPENDENCIES, ids=[d[0] for d in LAZY_DEPENDENCIES])
def test_lazy_dependency_is_installed_wherever_it_is_used(call, module, why):
    sites = _call_sites(call)
    if not sites:
        pytest.skip(f"nothing calls {call}() yet")
    assert importlib.util.find_spec(module) is not None, (
        f"{call}() is called at {sites} but {module!r} is not installed -- {why}. "
        f"Add it to pyproject dependencies; an import scan will never find it, "
        f"because our code does not import it."
    )


def test_the_known_offender_is_still_covered():
    """tabulate is the one an import scan is most likely to drop. If that call
    site disappears the row above starts skipping, and the gap could return
    unnoticed alongside a new to_markdown() call."""
    assert _call_sites("to_markdown"), "no to_markdown() call sites found -- either the code changed and this row can go, or the detector is broken"
    import tabulate  # noqa: F401


def test_the_detector_finds_a_call_it_should():
    """A call-site scan that silently matches nothing would make every row above
    skip rather than fail."""
    assert _call_sites("read_sql"), "detector found no read_sql() calls in src/"
