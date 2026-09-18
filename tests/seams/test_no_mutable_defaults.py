from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_source_functions_do_not_use_mutable_collection_defaults():
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            defaults = [*node.args.defaults, *node.args.kw_defaults]
            if any(isinstance(default, (ast.List, ast.Dict, ast.Set)) for default in defaults):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} {node.name}")

    assert not offenders, "mutable function defaults share state across calls:\n  " + "\n  ".join(offenders)
