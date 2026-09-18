"""No module may use a name it never binds.

A name that is used but never imported or assigned surfaces only at RUNTIME --
typically as a NameError inside an except clause, which turns into an error
message on the socket rather than a raised exception, and often on a branch the
tests do not reach.

ruff's F821 catches most of this, but it is advisory here and the repo carries
a large pre-existing backlog, so a new one is easy to miss in the noise. This
is scoped to the packages that must stay clean.
"""

from __future__ import annotations

import ast
import builtins
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"

# Packages held to this rule. Not the whole tree: the OMOP research code has
# its own backlog and is not what this guards.
WATCHED = ["ascent_platform", "ascent_http"]


def _unbound(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text())
    # module-level dunders exist without being bound in the source
    bound = set(dir(builtins)) | {
        "__file__",
        "__name__",
        "__doc__",
        "__package__",
        "__spec__",
        "__loader__",
        "__builtins__",
        "__path__",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            bound |= {a.asname or a.name for a in node.names}
        elif isinstance(node, ast.Import):
            bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Global):
            bound |= set(node.names)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            # `case [database_name, schema_name]:` binds both names
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return sorted(used - bound)


def _files() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for pkg in WATCHED:
        d = SRC / pkg
        assert d.is_dir(), f"{pkg} no longer exists; update WATCHED"
        out += [p for p in d.rglob("*.py") if "__pycache__" not in str(p)]
    return sorted(out)


@pytest.mark.parametrize("path", _files(), ids=lambda p: str(p.relative_to(SRC)))
def test_every_name_is_bound(path):
    missing = _unbound(path)
    assert not missing, (
        f"{path.relative_to(SRC)} uses names it never binds: {missing}. "
        f"These fail at runtime, often inside an except clause where they become "
        f"an error response rather than a crash."
    )


def test_the_detector_finds_an_unbound_name(tmp_path):
    """A clean tree and a broken detector look the same."""
    probe = tmp_path / "probe.py"
    probe.write_text("def f():\n    return SomethingNeverImported()\n")
    assert _unbound(probe) == ["SomethingNeverImported"]
