"""The platform layer must not depend upward.

The whole point of ascent_platform is a one-way dependency: app -> domain ->
platform. If platform imports back into the app or domain packages, the
layering is decorative — the cycle is back and the shared code is no longer
shareable.

It is enforced rather than documented.
"""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
PLATFORM = SRC / "ascent_platform"
FORBIDDEN = ("ascent_http", "ascent_mcp", "ascent_domain")


def _imported_roots(path: pathlib.Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", sorted(PLATFORM.rglob("*.py")), ids=lambda p: str(p.relative_to(PLATFORM)))
def test_platform_does_not_import_upward(path):
    offenders = _imported_roots(path) & set(FORBIDDEN)
    assert not offenders, f"{path.relative_to(SRC)} imports {sorted(offenders)} — the platform layer must not depend on the app or domain packages"


def test_platform_packages_are_present():
    """Guards against the layer being left half-built."""
    for sub in ("config", "warehouse", "cache", "http"):
        assert (PLATFORM / sub / "__init__.py").exists(), f"ascent_platform/{sub} is missing"


def test_warehouse_executor_is_shared_not_duplicated():
    """One bulkhead process-wide. Two pools of 20 would defeat the bound."""
    import ascent_domain.models.data_definitions  # noqa: F401  (import-cycle order)
    from ascent_platform.warehouse.executor import get_executor

    assert get_executor() is get_executor()


def test_no_warehouse_work_runs_on_the_default_executor():
    """`run_in_executor(None, ...)` puts blocking driver calls on Python's
    default pool — a starvation failure. The warehouse layer may not.
    """
    for rel in ("ascent_platform/warehouse/session.py", "ascent_platform/warehouse/executor.py", "ascent_platform/warehouse/sql_executor.py"):
        tree = ast.parse((SRC / rel).read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run_in_executor"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value is None
            ):
                pytest.fail(f"{rel}:{node.lineno} runs warehouse work on the default executor")


def test_llm_connectors_live_in_the_platform_not_in_a_domain_library():
    """LLM connectors belong in the platform.

    The router and the client classes read their configuration from the
    environment and know nothing about cohorts, questions or SQL. Inside a
    domain library, nothing can reach a Gemini client without importing the
    whole non-OMOP package — which is how a second router grows elsewhere. So
    fail loudly if one of those directories appears.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "src"
    for gone in ("ascent_non_omop/llm", "ascent_non_omop/llm_refactored"):
        assert not (src / gone).exists(), f"{gone} is back — LLM connectors belong in ascent_platform/llm/"

    assert (src / "ascent_platform/llm/router.py").exists()
    assert (src / "ascent_platform/llm/clients/gemini.py").exists()


def test_connector_helpers_have_exactly_one_definition_each():
    """Sharing is only worth anything if there are no copies.

    A second definition reappearing means a call site quietly stopped sharing,
    and the copies drift — one keeps ``total=None`` long after the other is
    bounded.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "src"
    watched = {
        "MEDICAL_CODER_READ_TIMEOUT": 1,
        "MEDICAL_CODER_TOTAL_TIMEOUT": 1,
        "MEDICAL_CODER_CONNECT_TIMEOUT": 1,
    }
    found = {k: [] for k in watched}
    for p in src.rglob("*.py"):
        if "__pycache__" in str(p):
            continue
        for node in ast.parse(p.read_text()).body:
            names = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            for n in names:
                if n in watched:
                    found[n].append(str(p.relative_to(src)))
    for name, expected in watched.items():
        assert len(found[name]) == expected, f"{name} is defined {len(found[name])}x, expected {expected}: {found[name]}"


def test_snowflake_work_never_uses_the_default_executor():
    """``run_in_executor(None, ...)`` on a Snowflake call is a starvation failure.

    The default executor is min(32, cpu+4) and shared with every other blocking
    call in the process; a handful of slow queries exhaust it and unrelated work
    queues behind them. LLM calls are exempt -- they have their own bound and
    must not share the Snowflake pool.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "src"
    offenders = []
    for p in src.rglob("*.py"):
        if "__pycache__" in str(p) or "llm" in str(p):
            continue
        text = p.read_text()
        if "snowflake" not in text.lower():
            continue
        for node in ast.walk(ast.parse(text)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run_in_executor"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value is None
            ):
                offenders.append(f"{p.relative_to(src)}:{node.lineno}")
    assert not offenders, f"Snowflake work on the default executor: {offenders}"


def test_domain_layer_holds_no_transport_concern():
    """ascent_domain must not know how it is delivered.

    Two surfaces sit above it -- REST routers and MCP tools -- and without this
    rule each reaches straight into services, data access and infrastructure. A
    module that cannot obey it is not domain logic yet; it is still holding a
    transport concern, which is exactly why search/cohort/concept_lists have
    NOT moved: they raise HTTPException from business paths.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "src"
    forbidden = {"fastapi", "starlette", "fastmcp", "mcp", "ascent_mcp"}
    offenders = []
    for path in (src / "ascent_domain").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            mods = []
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                mods = [node.module]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for m in mods:
                if m.split(".")[0] in forbidden:
                    offenders.append(f"{path.relative_to(src)}:{node.lineno} imports {m}")
    assert not offenders, "transport leaked into the domain layer:\n  " + "\n  ".join(offenders)


def test_no_call_site_still_reaches_the_moved_services_through_ascent():
    """Callers import ascent_domain directly.

    A compatibility shim under ascent_http.services is a trap: one that copies
    globals instead of rebinding sys.modules silently breaks every
    mock.patch("ascent_http.services.X..."), and the symptom is a test hanging
    on a live connection rather than failing.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2]
    moved = [
        "literature_agent_service",
        "observe_decide",
        "router",
        "table_one",
        "personalized_questions_task",
        "garbage_collect_orphans_task",
    ]
    # The shim check applies to every name -- a shim appearing is the
    # regression this guards -- but only these have a domain module to assert.
    still_in_the_domain = {"literature_agent_service", "personalized_questions_task", "garbage_collect_orphans_task"}
    services = root / "src/ascent/services"
    for name in moved:
        assert not (services / f"{name}.py").exists(), f"shim {name}.py came back"
        if name in still_in_the_domain:
            assert (root / f"src/ascent_domain/{name}.py").exists(), f"{name} missing from the domain"

    stale = []
    for path in list((root / "src").rglob("*.py")) + list((root / "tests").rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        text = path.read_text()
        for name in moved:
            if re.search(rf"ascent\.services\.{name}\b", text) or re.search(rf"from ascent\.services import [^\n]*\b{name}\b", text):
                stale.append(f"{path.relative_to(root)} -> {name}")
    assert not stale, "still reaching moved modules through ascent_http.services:\n  " + "\n  ".join(stale)


# Edges the domain layer has into the app. Empty, which makes it a rule rather
# than a budget: ascent_domain does not import ascent_http.
#
# Do not add an entry to make a move compile. Where the common cases belong:
#   data_definitions  the models are domain-owned -- move them down
#   external_calls    split by whether the client knows a DTO
#   services          the factory is domain, the Depends alias is transport
#   settings          model policy is a domain decision, not configuration
_DOMAIN_UPWARD_BUDGET: dict[str, int] = {}


def _domain_upward_edges():
    import ast
    import collections
    import pathlib

    counts = collections.Counter()
    root = pathlib.Path(__file__).resolve().parents[2] / "src/ascent_domain"
    for path in root.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            mods = []
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                mods = [node.module]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for m in mods:
                if m.startswith("ascent_http."):
                    counts[".".join(m.split(".")[:2])] += 1
    return counts


def test_domain_upward_edges_do_not_grow():
    """With an empty budget this is simply: the domain imports no app module."""
    actual = _domain_upward_edges()
    grown = {k: (v, _DOMAIN_UPWARD_BUDGET.get(k, 0)) for k, v in actual.items() if v > _DOMAIN_UPWARD_BUDGET.get(k, 0)}
    assert not grown, f"ascent_domain gained imports of the app layer {{as (now, budget)}}: {grown}. Move what it needs down instead."


def test_domain_upward_budget_is_tightened_when_edges_are_removed():
    """Kept for when an entry is added back: a budget left above reality stops
    ratcheting, and the entry would then never be paid down."""
    actual = _domain_upward_edges()
    slack = {k: (actual.get(k, 0), v) for k, v in _DOMAIN_UPWARD_BUDGET.items() if actual.get(k, 0) < v}
    assert not slack, f"edges were removed; lower the budget {{as (now, budget)}}: {slack}"


def test_infrastructure_packages_never_import_the_domain():
    """Infrastructure must not point up.

    The package list is asserted to exist. rglob over a missing directory
    yields nothing and passes, so a renamed or deleted package would silently
    turn this into a test of nothing.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src"
    # ascent_platform is the only infrastructure package. ascent_domain.omop
    # names cohorts, criteria, OMOP and patients throughout, which makes it
    # domain code.
    packages = ("ascent_platform",)
    missing = [p for p in packages if not (root / p).is_dir()]
    assert not missing, f"this test scans packages that no longer exist: {missing}"

    offenders = []
    for pkg in packages:
        for path in (root / pkg).rglob("*.py"):
            if "__pycache__" in str(path):
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                mods = []
                if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    mods = [node.module]
                elif isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                for m in mods:
                    if m.split(".")[0] in {"ascent_domain", "ascent_http", "ascent_mcp"}:
                        offenders.append(f"{path.relative_to(root)}:{node.lineno} -> {m}")
    assert not offenders, "infrastructure imports upward:\n  " + "\n  ".join(offenders)


# os.getenv sites still present in ascent_platform, with today's count. The rule
# is that configuration comes from RuntimeSettings/PlatformSettings, not from
# the environment read per-module: scattered getenv hides what a component
# needs, defeats typing and defaults, and lets one value acquire two names.
_PLATFORM_GETENV_BUDGET = {
    "config/env.py": 1,  # the env-file loader itself; bootstraps before settings exist
    "llm/router.py": 1,  # ModelConfig.api_key_env names the variable at
    # runtime, so this one cannot be a static field
    # azure_chat.py is deliberately absent: it resolves the endpoint and key
    # through Settings, so its budget is 0.
    "cache/redis_tls.py": 1,
    "llm/availability.py": 1,  # provider requirements include AWS_ACCESS_KEY_ID and
    # friends, which are not settings fields: boto3 reads
    # them from the environment itself, so a provider can
    # be usable with nothing on the settings object.
}


def test_platform_does_not_read_the_environment_directly():
    """A ratchet: no new os.getenv in the platform, and budgets must shrink."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2] / "src/ascent_platform"
    pat = re.compile(r"os\.(?:getenv\(|environ\.get\(|environ\[)")
    actual: dict[str, int] = {}
    for path in root.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        n = len(pat.findall(path.read_text()))
        if n:
            actual[str(path.relative_to(root))] = n

    grown = {k: (v, _PLATFORM_GETENV_BUDGET.get(k, 0)) for k, v in actual.items() if v > _PLATFORM_GETENV_BUDGET.get(k, 0)}
    assert not grown, f"new direct environment reads {{as (now, budget)}}: {grown}"

    slack = {k: (actual.get(k, 0), v) for k, v in _PLATFORM_GETENV_BUDGET.items() if actual.get(k, 0) < v}
    assert not slack, f"reads were removed; lower the budget {{as (now, budget)}}: {slack}"


def test_the_upward_edge_scan_would_notice_an_edge():
    """The rule above passes when the budget is empty AND when the scan is
    broken. An empty result set proves nothing on its own, so check the
    detector against a module that really does import upward."""
    import ast
    import collections
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "src"
    counts = collections.Counter()
    for path in (src / "ascent_mcp").rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            mods = []
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                mods = [node.module]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for m in mods:
                if m.startswith("ascent_http."):
                    counts[".".join(m.split(".")[:2])] += 1
    assert counts, (
        "the scan found no ascent_http.* imports in ascent_mcp, which is where the "
        "app surface legitimately has them -- the detector is broken, so "
        "test_domain_upward_edges_do_not_grow is passing vacuously"
    )


# The domain reads configuration through DomainSettings / RuntimeSettings, not
# from the environment per-module. Same rule as the platform ratchet above and
# the same reasoning: a scattered getenv hides what a component needs, defeats
# typing and defaults, and lets one value acquire two spellings.
#
# Empty, so this is a rule rather than a budget.
_DOMAIN_GETENV_BUDGET: dict[str, int] = {}


def _domain_getenv_sites(_root=None) -> dict[str, int]:
    """AST, not grep: a regex counts a commented-out ``os.getenv`` as a
    violation."""
    import ast
    import pathlib

    def _is_env_read(node) -> bool:
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "getenv" and getattr(f.value, "id", "") == "os":
                return True
            if isinstance(f, ast.Attribute) and f.attr == "get" and isinstance(f.value, ast.Attribute):
                return f.value.attr == "environ" and getattr(f.value.value, "id", "") == "os"
        if isinstance(node, ast.Subscript):
            # A WRITE is not a configuration read. `os.environ["MPLBACKEND"] = "Agg"`
            # forces a matplotlib backend; counting it as config would tell the
            # caller to "add the field to RuntimeSettings", which is nonsense.
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                return False
            v = node.value
            return isinstance(v, ast.Attribute) and v.attr == "environ" and getattr(v.value, "id", "") == "os"
        return False

    root = _root or pathlib.Path(__file__).resolve().parents[2] / "src/ascent_domain"
    found: dict[str, int] = {}
    for path in root.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        n = sum(1 for node in ast.walk(ast.parse(path.read_text())) if _is_env_read(node))
        if n:
            found[str(path.relative_to(root))] = n
    return found


def test_domain_does_not_read_the_environment_directly():
    actual = _domain_getenv_sites()
    grown = {k: (v, _DOMAIN_GETENV_BUDGET.get(k, 0)) for k, v in actual.items() if v > _DOMAIN_GETENV_BUDGET.get(k, 0)}
    assert not grown, (
        f"the domain reads the environment directly {{as (now, budget)}}: {grown}. Add the field to DomainSettings or RuntimeSettings instead."
    )
    slack = {k: (actual.get(k, 0), v) for k, v in _DOMAIN_GETENV_BUDGET.items() if actual.get(k, 0) < v}
    assert not slack, f"reads were removed; lower the budget {{as (now, budget)}}: {slack}"


def test_the_getenv_scan_would_notice_a_read(tmp_path):
    """An empty budget and a broken detector look identical, so run the real
    detector over a file that definitely reads the environment."""
    (tmp_path / "probe.py").write_text(
        "import os\nA = os.getenv('X')\nB = os.environ.get('Y')\nC = os.environ['Z']\n# os.getenv('COMMENTED') -- must not count\n"
    )
    assert _domain_getenv_sites(tmp_path) == {"probe.py": 3}
