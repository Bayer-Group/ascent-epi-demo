"""Single owner of environment-file loading.

Loading happens in two passes: one set of files loaded with ``override=True``,
which beats anything already in ``os.environ``, then a second set loaded
without override, which can only fill gaps. Both are idempotent per base
directory.

The ordering is deliberate and load-bearing; see
``tests/seams/test_settings_characterization.py``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# The repo root: .../src/ascent_platform/config/env.py -> up four.
REPO_ROOT = Path(__file__).resolve().parents[3]

# Bases already loaded. Keyed by path rather than a bare flag, so a second
# call with a *different* base is still honoured.
_loaded_bases: set[Path] = set()


def base_dir() -> Path:
    """Directory the env files are resolved against.

    ``ENV_LOCAL_PATH`` wins when set (the container sets it to ``/app``), so
    deployments keep controlling this explicitly.
    """
    configured = os.environ.get("ENV_LOCAL_PATH")
    return Path(configured) if configured else REPO_ROOT


def _override_files(base: Path) -> list[Path]:
    """Loaded with override=True — these beat anything already in os.environ."""
    return [base / ".env", base / ".env.local", base / ".env.dev", Path(".env"), Path(".env.local")]


def _fill_gap_files(base: Path) -> list[Path]:
    """Loaded with override=False — these only fill variables not already set."""
    return [base / ".env", base / ".env.local"]


def load_environment(*, force: bool = False) -> Path:
    """Load env files once, override pass first, gap-fill pass second.

    Returns the base directory used, so callers can log or assert on it.
    """
    base = base_dir().resolve()
    if base in _loaded_bases and not force:
        return base

    # ENV_LOCAL_PATH is exported for consumers that read it directly.
    os.environ.setdefault("ENV_LOCAL_PATH", str(base.resolve()))

    for path in _override_files(base):
        if path.exists():
            load_dotenv(path, override=True)
            logger.debug("env loaded (override): %s", path)

    for path in _fill_gap_files(base):
        if path.exists():
            load_dotenv(path)
            logger.debug("env loaded (gap-fill): %s", path)

    _loaded_bases.add(base)
    return base
