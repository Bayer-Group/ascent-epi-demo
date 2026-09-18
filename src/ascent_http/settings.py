"""The application's view of configuration -- the same object as everyone else's.

``Settings`` and the shared ``settings`` instance live in
``ascent_platform.config.runtime``; this module re-exports them and performs the
app-level environment setup.

Required configuration is validated by ``require_app_config()`` at startup
rather than by declaring fields without defaults, which would make importing the
library require a database password. The startup check names every missing
variable at once.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ascent_platform.config import load_environment
from ascent_platform.config.paths import PROJECT_ROOT
from ascent_platform.config.runtime import (
    Settings,
    get_settings,
    require_app_config,
    set_settings,
    settings,
)

logger = logging.getLogger(__name__)

# Several modules and the alembic env read this, and it is the anchor the
# query-library glob resolves against.
os.environ["ENV_LOCAL_PATH"] = f"{str(Path(PROJECT_ROOT).resolve())}{os.environ.get('ENV_PATH', '')}"

# One shared loader owns which env files are read and in what precedence.
load_environment()


__all__ = [
    "PROJECT_ROOT",
    "Settings",
    "get_settings",
    "require_app_config",
    "set_settings",
    "settings",
]
