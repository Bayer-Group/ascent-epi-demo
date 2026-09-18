"""Where the checkout is, for code that must not ask the app.

``ascent_http.settings`` computes PROJECT_ROOT as ``Path(__file__).parent ** 3`` and
several packages read it from there, which is one more reason a lower layer
ends up importing the application. The anchor itself belongs to no layer, so it
lives here.

Depth is counted from this file, so it differs from the expression in
ascent_http.settings; tests/seams assert the two resolve to the same directory.
"""

from __future__ import annotations

from pathlib import Path

# src/ascent_platform/config/paths.py -> config -> ascent_platform -> src -> root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
