"""
OMOP data-science package: cohort generation, question answering and RAG over
OMOP-standardised data.
"""

# Set matplotlib backend BEFORE any imports to prevent tkinter threading errors
# This must be at the very top of the package to ensure it's set before any module imports matplotlib
import os

os.environ["MPLBACKEND"] = "Agg"  # Force Agg backend (non-interactive) to prevent tkinter issues

# This package ships inside the application rather than as its own
# distribution, so there is no package metadata to read a version from; report
# the application version instead.
from ascent_platform.config.runtime import get_settings  # noqa: E402

__version__ = get_settings().APP_VERSION or "vendored"

__all__ = ["__version__"]
