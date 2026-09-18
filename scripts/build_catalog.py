"""CLI for the catalog generator.

The implementation lives in ``ascent_platform.warehouse.catalog_build`` so the
warehouse bootstrap can call it too -- the container generates a missing
catalog at startup rather than requiring this script to have been run.

    python scripts/build_catalog.py
    python scripts/build_catalog.py --scale 10k
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ascent_platform.warehouse.catalog_build import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
