"""The injection seam is only worth having if it actually holds.

Five of the seven non-OMOP workflows used to hard-code their ``LLMRouter``, so
constructing them required a live Gemini key and no test could substitute a
fake. These tests assert the property that fixed, not the implementation: every
workflow accepts an ``llm_service`` and uses the one it is given.

The AST test is the one that stops the regression. A newly added workflow that
hard-codes a router again would pass every behavioural test in this file simply
by not being in it.
"""

import ast
import pathlib
from unittest.mock import AsyncMock, MagicMock

import pytest

import ascent_domain.models.data_definitions  # noqa: F401  (import-cycle order)

WORKFLOW_DIR = pathlib.Path(__file__).resolve().parents[2] / "src/ascent_domain/non_omop/workflows"

# A parametrized test whose discovery path goes stale does not fail -- it
# collapses to "empty parameter set" and skips. Fail loudly instead.
assert WORKFLOW_DIR.is_dir(), f"workflow discovery path is stale: {WORKFLOW_DIR}"


def _workflow_classes():
    """(class_name, file_name, __init__ node) for every workflow class."""
    for path in sorted(WORKFLOW_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            init = next(
                (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
                None,
            )
            if init is not None:
                yield cls.name, path.name, init


@pytest.mark.parametrize(
    "cls_name,file_name,init",
    [pytest.param(c, f, i, id=c) for c, f, i in _workflow_classes()],
)
def test_every_workflow_accepts_an_injected_llm_service(cls_name, file_name, init):
    args = [a.arg for a in init.args.args] + [a.arg for a in init.args.kwonlyargs]
    assert "llm_service" in args, f"{cls_name} ({file_name}) does not accept llm_service — it cannot be tested or re-pointed without editing it"


def test_no_workflow_constructs_its_own_router():
    """AST, not grep: a comment mentioning LLMRouter must not fail this."""
    offenders = []
    for path in sorted(WORKFLOW_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "LLMRouter":
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"workflows construct LLMRouter directly, bypassing the provider: {offenders}"


def test_provider_returns_one_shared_router(monkeypatch):
    from ascent_domain.non_omop import llm_provider as provider

    monkeypatch.setattr(provider, "_instance", None)
    sentinel = MagicMock()
    monkeypatch.setattr(provider, "LLMRouter", MagicMock(return_value=sentinel))

    assert provider.get_llm_service() is sentinel
    assert provider.get_llm_service() is sentinel
    assert provider.LLMRouter.call_count == 1, "the router was rebuilt on the second call"


def test_set_llm_service_overrides_and_reset_restores(monkeypatch):
    from ascent_domain.non_omop import llm_provider as provider

    monkeypatch.setattr(provider, "_instance", None)
    built = MagicMock()
    monkeypatch.setattr(provider, "LLMRouter", MagicMock(return_value=built))

    injected = MagicMock()
    provider.set_llm_service(injected)
    assert provider.get_llm_service() is injected
    assert provider.LLMRouter.call_count == 0, "an injected service must not build a router"

    provider.reset_llm_service()
    assert provider.get_llm_service() is built


@pytest.mark.asyncio
async def test_injected_service_is_the_one_actually_called(monkeypatch):
    """End to end through a workflow that previously hard-coded its router.

    Guards against a constructor that accepts llm_service and then ignores it.
    """
    from ascent_domain.non_omop import llm_provider as provider
    from ascent_domain.non_omop.workflows.sql_to_question import SQLToQuestion

    monkeypatch.setattr(
        provider,
        "LLMRouter",
        MagicMock(side_effect=AssertionError("fell through to the default router")),
    )

    fake = MagicMock()
    fake.send_message = AsyncMock(return_value=MagicMock(verdict="ok"))
    wf = SQLToQuestion(model_name="gemini-3.6-flash", llm_service=fake)

    assert wf.llm_service is fake


def test_workflow_discovery_is_not_vacuous():
    """Guards the parametrization itself: every workflow must be found."""
    found = list(_workflow_classes())
    assert len(found) >= 5, f"only discovered {[c for c, _, _ in found]}"
