"""Switching DATA_SCALE must actually switch the dataset.

Two silent failures live here, and both answer questions against the wrong
data while reporting success:

1. A built filename with no scale in it. Moving from 1k to 10k finds the
   existing file, logs "already built", and serves 1,000 patients under a 10k
   banner.
2. ``DATA_SCALE`` is the name in .env.template, the README and compose, while
   the settings field is ``WAREHOUSE_SCALE``. If only compose translates
   between them, the switch works under Docker and does nothing on the
   documented from-source path.
"""

from pathlib import Path

import pytest

from ascent_platform.config.runtime import Settings
from ascent_platform.warehouse.bootstrap import DATABASES, warehouse_path


@pytest.mark.parametrize("db", DATABASES, ids=lambda d: d.name)
def test_the_built_path_names_the_scale(db):
    one = warehouse_path(Path("/w"), db, "1k")
    ten = warehouse_path(Path("/w"), db, "10k")

    assert one != ten, "1k and 10k must not collide on one filename"
    assert "1k" in one.name and "10k" in ten.name


def test_both_scales_can_coexist():
    """Switching back must not force a rebuild of what is already there."""
    paths = {warehouse_path(Path("/w"), db, scale) for db in DATABASES for scale in ("1k", "10k")}

    assert len(paths) == len(DATABASES) * 2


def test_data_scale_is_accepted_as_the_documented_name(monkeypatch):
    monkeypatch.delenv("WAREHOUSE_SCALE", raising=False)
    monkeypatch.setenv("DATA_SCALE", "10k")

    assert Settings().WAREHOUSE_SCALE == "10k"


def test_an_explicit_warehouse_scale_wins(monkeypatch):
    """Compose sets WAREHOUSE_SCALE directly; it must not be overridden."""
    monkeypatch.setenv("WAREHOUSE_SCALE", "1k")
    monkeypatch.setenv("DATA_SCALE", "10k")

    assert Settings().WAREHOUSE_SCALE == "1k"


def test_the_archive_name_tracks_the_scale():
    """The scale reaches the filename the loader looks for, not just the log."""
    for db in DATABASES:
        assert db.archive.format(scale="10k").endswith("_10k.db.zip")
        assert db.archive.format(scale="1k").endswith("_1k.db.zip")
