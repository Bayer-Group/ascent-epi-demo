"""Run the actual changed-file workflow steps against small Git histories."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True).stdout.strip()


def _write(repo, name, content="value = 1\n"):
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _commit(repo):
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    _git(tmp_path, "config", "core.hooksPath", "/dev/null")
    _write(tmp_path, "src/package/app.py")
    _write(tmp_path, "tests/test_app.py")
    _write(tmp_path, "README.md", "Fixture\n")
    _commit(tmp_path)
    return tmp_path


def _changed(repo, job, base_ref="HEAD~1", *, check=True):
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    step = next(s for s in workflow["jobs"][job]["steps"] if s.get("id") == "changed")
    # Explicit bash opts into GitHub Actions' -eo pipefail behavior. A failed
    # diff must not be swallowed by the trailing tr command in its pipeline.
    assert step["shell"] == "bash"
    script = step["run"].replace("${{ github.event.pull_request.base.sha || 'HEAD~1' }}", base_ref)
    assert "${{" not in script
    output = repo / ".git" / "test-step-output"
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        cwd=repo,
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        text=True,
        capture_output=True,
        check=check,
    )
    values = dict(line.split("=", 1) for line in output.read_text().splitlines()) if output.exists() else {}
    return result, values


@pytest.mark.parametrize("job", ["lint", "types"])
def test_root_commit_checks_python_files_against_empty_tree(repo, job):
    _, output = _changed(repo, job)
    expected = {"src/package/app.py"}
    if job == "lint":
        expected.add("tests/test_app.py")
    assert set(output["files"].split()) == expected
    assert _git(repo, "cat-file", "-t", output["base"]) == "tree"
    assert _git(repo, "ls-tree", output["base"]) == ""


@pytest.mark.parametrize("job", ["lint", "types"])
def test_later_push_checks_only_added_and_modified_files(repo, job):
    _write(repo, "src/package/removed.py")
    _commit(repo)
    parent = _git(repo, "rev-parse", "HEAD")
    _write(repo, "src/package/app.py", "value = 2\n")
    _write(repo, "src/package/added.py")
    (repo / "src/package/removed.py").unlink()
    _commit(repo)
    _, output = _changed(repo, job)
    assert output["base"] == parent
    assert set(output["files"].split()) == {"src/package/app.py", "src/package/added.py"}


@pytest.mark.parametrize("job", ["lint", "types"])
def test_diverged_pull_request_uses_actual_merge_base(repo, job):
    common = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-qb", "target")
    _write(repo, "src/package/app.py", "value = 2\n")
    _write(repo, "src/package/target_only.py")
    target = _commit(repo)
    _git(repo, "checkout", "-qb", "feature", common)
    _write(repo, "src/package/feature_only.py")
    _commit(repo)
    _, output = _changed(repo, job, target)
    assert output["base"] == common
    assert set(output["files"].split()) == {"src/package/feature_only.py"}


@pytest.mark.parametrize("job", ["lint", "types"])
def test_invalid_pull_request_base_fails_instead_of_skipping_checks(repo, job):
    result, output = _changed(repo, job, "missing-pr-base", check=False)
    assert result.returncode != 0
    assert "files" not in output
