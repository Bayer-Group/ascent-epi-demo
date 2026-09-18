"""No module may reference a name it never defines or imports.

Two of these were live at once. ``databases.py`` called ``logger.exception``
without importing logging, so the error handler raised NameError and destroyed
the exception it was reporting. ``patient_counts.py`` called
``get_snowflake_connector``, a helper deleted in the move to the local
warehouse, on a path the patient-count workflow awaits.

Both are invisible to normal testing: they live in branches only taken when
something else has already gone wrong, or in code no test exercises. A linter
sees them immediately, which is the point of asserting it here rather than
hoping someone runs one.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _ruff() -> str | None:
    """ruff ships as a binary, not an importable module.

    It is a declared dev dependency, so it is normally on PATH or beside the
    interpreter. Looked up rather than assumed: an earlier version of this test
    invoked ``python -m ruff``, which does not exist, so the check skipped
    itself and would have passed with the bug still in the tree.
    """
    return shutil.which("ruff") or next(
        (str(p) for p in (Path(sys.prefix) / "bin" / "ruff", Path(sys.executable).parent / "ruff") if p.exists()),
        None,
    )


def test_no_undefined_names_in_src():
    ruff = _ruff()
    if ruff is None:
        pytest.skip("ruff is not installed in this environment")

    result = subprocess.run(
        [ruff, "check", "src", "--select", "F821", "--output-format=concise"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return

    findings = [ln for ln in result.stdout.splitlines() if "F821" in ln]
    assert not findings, (
        "undefined names in src/ -- these raise NameError at runtime, and the ones "
        "in exception handlers replace the original failure with this one:\n  "
        + "\n  ".join(findings)
    )
