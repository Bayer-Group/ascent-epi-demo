"""Test directories the default run skips must still resolve their imports.

`pytest tests --ignore=tests/e2e` is how this suite is run everywhere, and
tests/load is only executed by the locust workflow. So a path sweep can move a
module out from under either of them and nothing goes red -- which is what
happened: tests/e2e/conftest.py kept importing ``ascent.db.postgresql_session``
through the whole platform extraction, and the break surfaced by hand at merge
time rather than in CI.

These tests are not run here -- they need a live app, a database and a deployed
environment. Collection is the part that can rot silently, so collection is
what is checked, in a subprocess because a failed collection is an exception at
import time, not a return value.

The first version of this file checked ``importlib.util.find_spec`` on each
imported module instead, and did NOT catch the real bug: ``from ascent.db
import postgresql_session`` names the package ``ascent.db``, which still
exists, while the submodule it actually wants does not. Hence the subprocess.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Suites the default `pytest tests --ignore=tests/e2e` run does not import.
# tests/load holds locust files and collects zero tests; that is fine and
# expected -- what matters is that collection does not ERROR.
# Empty here: tests/e2e and tests/load both needed a deployed environment and a
# live database, so neither shipped with the open-source build. The guard stays
# rather than being deleted -- add a suite back and it starts checking again.
EXCLUDED_SUITES: list[str] = []


@pytest.mark.parametrize("suite", EXCLUDED_SUITES)
def test_suite_still_collects(suite):
    path = ROOT / suite
    assert path.is_dir(), f"{suite} no longer exists; drop it from EXCLUDED_SUITES"

    env = {
        **dict(__import__("os").environ),
        "PYTHONPATH": str(ROOT / "src"),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(path), "--collect-only", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    # 0 = collected, 5 = collected nothing (tests/load). Anything else is an error.
    assert proc.returncode in (0, 5), (
        f"{suite} no longer collects — a refactor moved something out from under a "
        f"suite the default run skips.\n"
        f"exit={proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-2000:]}"
    )


def test_the_check_would_notice_a_moved_module(tmp_path):
    """A collect that always passes is worth nothing. Point the same subprocess
    at a file importing a module that does not exist."""
    (tmp_path / "test_probe.py").write_text(
        "from ascent_http.util import a_module_that_does_not_exist  # noqa\n"
        "def test_x():\n    pass\n"
    )
    env = {
        **dict(__import__("os").environ),
        "PYTHONPATH": str(ROOT / "src"),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path), "--collect-only", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode not in (0, 5), (
        "collection succeeded on a module that does not exist — this check is inert"
    )
