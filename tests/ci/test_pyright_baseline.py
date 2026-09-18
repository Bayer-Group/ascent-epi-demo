"""Exercise the actual baseline shell step, including an all-new-files PR."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(("has_parent", "has_existing_file"), [(False, False), (True, False), (True, True)])
def test_added_file_errors_are_never_subtracted_as_baseline(tmp_path, has_parent, has_existing_file):
    def run(*args, **kwargs):
        return subprocess.run(args, cwd=tmp_path, text=True, capture_output=True, check=True, **kwargs)

    run("git", "init", "-q")
    identity = ("git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid")
    if has_parent:
        (tmp_path / "README").write_text("baseline")
        if has_existing_file:
            (tmp_path / "existing.py").write_text("OLD_ERROR")
        run("git", "add", ".")
        run(*identity, "commit", "-qm", "baseline")
        base = run("git", "rev-parse", "HEAD").stdout.strip()
    else:
        base = run("git", "hash-object", "-t", "tree", "-w", "--stdin", input="").stdout.strip()
    (tmp_path / "added.py").write_text("NEW_ERROR")
    run("git", "add", ".")
    run(*identity, "commit", "-qm", "head")

    # Deterministic fake Pyright: the workflow must pass it only baseline files.
    # The real ratchet consumes these reports exactly as it does Pyright's JSON.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "files = sys.argv[sys.argv.index('--outputjson') + 1:]\n"
        "diagnostics = [{'file': str(pathlib.Path(f).resolve()), 'severity': 'error',\n"
        "                'message': pathlib.Path(f).read_text(), 'rule': 'test-rule'} for f in files]\n"
        "print(json.dumps({'generalDiagnostics': diagnostics}))\n"
    )
    uv.chmod(0o755)
    files = ["existing.py", "added.py"] if has_existing_file else ["added.py"]
    head = run(str(uv), "run", "pyright", "--outputjson", *files).stdout
    (tmp_path / "head.json").write_text(head)

    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    step = next(s for s in workflow["jobs"]["types"]["steps"] if s.get("name") == "pyright, on the merge base")
    script = step["run"].replace("${{ steps.changed.outputs.base }}", base)
    script = script.replace("${{ steps.changed.outputs.files }}", " ".join(files))
    run("bash", "-eu", "-c", script, env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"})

    diagnostics = json.loads((tmp_path / "base.json").read_text())["generalDiagnostics"]
    assert all(Path(d["file"]).name != "added.py" for d in diagnostics)
    assert (tmp_path / "added.py").read_text() == "NEW_ERROR"
    result = subprocess.run(
        [sys.executable, str(ROOT / "ci/pyright_ratchet.py"), "base.json", "head.json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "1 new type error(s)" in result.stdout
    assert "NEW_ERROR" in result.stdout
