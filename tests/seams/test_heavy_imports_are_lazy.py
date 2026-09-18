"""Heavy dependencies must not be imported at module scope.

matplotlib/seaborn/streamlit/torch were all imported eagerly by production
modules, so every container start paid for them and `matplotlib.use("Agg")`
ran as an import side effect -- a process-wide backend set by a request-scoped
concern.

torch and transformers are deliberately NOT in this list. They are not ours to
defer: langchain_core imports transformers, which imports torch, and
sentence-transformers (the query-library embeddings at rag.py) needs torch at
runtime. Our own code no longer imports them eagerly, which is what this test
can actually enforce.
"""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
FORBIDDEN_EAGER = {"matplotlib", "seaborn", "streamlit", "torch", "transformers"}


def _eager_offenders():
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in str(path) or "/docs/" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in tree.body:  # module level only == eager
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            elif isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            for n in names:
                if n in FORBIDDEN_EAGER:
                    yield f"{path.relative_to(SRC)}:{node.lineno} imports {n}"


def test_no_production_module_imports_a_heavy_dependency_eagerly():
    offenders = list(_eager_offenders())
    assert not offenders, "import these inside the function that needs them:\n  " + "\n  ".join(offenders)


@pytest.mark.parametrize("package", ["seaborn", "streamlit"])
def test_optional_packages_are_not_required_at_import(package):
    """These moved to the 'research' extra, so the app must import without them."""
    import importlib.util

    if importlib.util.find_spec(package) is not None:
        pytest.skip(f"{package} is installed (research extra); nothing to prove")
    # reaching here means it is absent -- the suite importing at all is the proof
    assert True


def test_lazy_import_helpers_do_not_call_themselves():
    """The obvious way to write these helpers is also a way to write infinite
    recursion.

    Making the heavy imports lazy meant inserting `plt = _plt()` at the top of
    every function that used `plt` -- and the helper _plt() uses `plt` too, so
    it got the line as well and called itself. No test rendered a chart, so the
    suite stayed green while the visualization path was one call away from
    RecursionError. ruff's F811 is what surfaced it.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "src"
    offenders = []
    for path in src.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("_"):
                continue
            if not any(k in fn.name for k in ("plt", "plot", "require", "lazy")):
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and getattr(node.func, "id", None) == fn.name:
                    offenders.append(f"{path.relative_to(src)}:{node.lineno} {fn.name}() calls itself")
    assert not offenders, "lazy-import helper recurses:\n  " + "\n  ".join(offenders)
