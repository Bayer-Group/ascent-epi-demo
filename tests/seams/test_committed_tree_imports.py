"""The COMMITTED tree must import, not just the working tree.

Twice on this branch a required source change was lost because it shared a file
with an uncommitted local override:

  ascent_platform/config/runtime.py  carries a local port override, and the
      shared `settings` proxy went with it -- CI could not collect at all.
  ascent_mcp/authentication.py       carries local Azure credentials, and the
      ascent -> ascent_http import went with it -- every ascent_mcp test
      errored on import.

Both were invisible locally, because the working tree had the fix. Every other
check in this suite reads the working tree, so none of them could see it.

This reads what git has, not what is on disk.
"""

from __future__ import annotations

import ast
import pathlib
import re
import subprocess

import pytest

# Files known to carry uncommitted local overrides. The point is not that these
# are special -- it is that ANY file can be, so the check covers the whole tree.
_PACKAGES = ("ascent_http", "ascent_mcp", "ascent_domain", "ascent_platform")


def _in_git_checkout() -> bool:
    """Whether git history is reachable from here.

    It is not inside the Docker image, or in a tree exported with
    `git archive` -- both of which the verification flow uses. The check is
    meaningful only where there is history to read, so it skips rather than
    fails there; CI runs on a real checkout.

    The image has no history *and no git binary*, so the call itself raises
    FileNotFoundError rather than returning non-zero -- which crashed
    collection for the whole suite instead of skipping this module.
    """
    try:
        r = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
    except (FileNotFoundError, OSError):
        return False
    return r.returncode == 0 and r.stdout.strip() == "true"


pytestmark = pytest.mark.skipif(not _in_git_checkout(), reason="no git history here (exported tree or image)")


def _committed(path: str) -> str | None:
    r = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def _tracked_python_files() -> list[str]:
    r = subprocess.run(["git", "ls-files", "src/*.py"], capture_output=True, text=True, check=True)
    return [p for p in r.stdout.split("\n") if p.endswith(".py")]


def test_no_committed_file_imports_a_package_that_does_not_exist():
    """A rename that misses a file shows up here, not in CI."""
    files = _tracked_python_files()
    assert files, "no tracked python files found -- this check is inert"

    known = set(_PACKAGES)
    offenders = []
    for path in files:
        source = _committed(path)
        if source is None:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            offenders.append(f"{path}: does not parse as committed ({exc})")
            continue
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                mods = [node.module]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for m in mods:
                root = m.split(".")[0]
                # only first-party roots: an "ascent*" package that is not one
                # of the four does not exist any more
                if root.startswith("ascent") and root not in known:
                    offenders.append(f"{path}:{node.lineno} imports {m}")
    assert not offenders, (
        "the COMMITTED tree imports packages that do not exist:\n  "
        + "\n  ".join(offenders)
        + "\n\nThis usually means a rename was unstaged along with a local override."
    )


def test_the_check_reads_git_not_the_working_tree():
    """Otherwise it would duplicate every other test in this suite.

    Proven by the one case where git and the disk cannot agree: a file that
    exists on disk and has never been committed. If ``_committed`` were reading
    the working tree it would return the contents; reading git, it returns None.

    The credential check this used to carry is now its own test below, applied
    to the whole tree rather than to one file -- a secret is no more welcome in
    any other module than in this one.
    """
    probe = pathlib.Path("src/ascent_mcp/_committed_tree_probe.py")
    probe.write_text("# untracked probe, deleted below\n")
    try:
        assert probe.exists(), "the probe must be on disk for this to prove anything"
        assert _committed(str(probe)) is None, (
            "_committed returned contents for a file git has never seen -- it is reading the working tree, so the check above is inert"
        )
    finally:
        probe.unlink(missing_ok=True)

    # And the positive half: a tracked file does come back.
    assert _committed("src/ascent_mcp/authentication.py") is not None


def test_no_committed_file_hardcodes_a_credential():
    """No secret reaches the committed tree, whichever file it is in.

    Matched by shape rather than by content, so that asserting a secret is
    absent does not require writing one down -- an earlier version of this
    suite embedded a fragment of the real credential to prove the credential
    was gone, which put it in front of every secret scanner.
    """
    pat = re.compile(
        r"""(client_secret|api_key|password|secret_key|token)\s*[:=]\s*["']([^"'{}$<>\s]{12,})["']""",
        re.IGNORECASE,
    )
    # Placeholders and obvious non-secrets. All-zero and "dummy" values are used
    # deliberately elsewhere in the tree precisely so they read as inert.
    benign = re.compile(r"^(0+|x+|dummy|changeme|placeholder|your[-_])", re.IGNORECASE)

    offenders = []
    for path in _tracked_python_files():
        source = _committed(path)
        if source is None:
            continue
        for lineno, line in enumerate(source.split("\n"), 1):
            m = pat.search(line)
            if m and not benign.match(m.group(2)):
                offenders.append(f"{path}:{lineno} {m.group(1)}=...")
    assert not offenders, "a hardcoded credential reached the committed tree:\n  " + "\n  ".join(offenders)


@pytest.mark.parametrize("pkg", _PACKAGES)
def test_each_package_is_tracked(pkg):
    r = subprocess.run(["git", "ls-files", f"src/{pkg}/"], capture_output=True, text=True, check=True)
    assert r.stdout.strip(), f"src/{pkg} has no tracked files"
