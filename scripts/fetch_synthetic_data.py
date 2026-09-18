"""Download a synthetic dataset that is too large to ship in the repository.

The 1k dataset is committed, so a clone runs with no download at all. Larger
scales are not: ``omop_synthetic_10k.db.zip`` alone is 104 MiB, past GitHub's
100 MiB hard limit for a single file, so committing it would make the
repository unpushable.

Usage:

    python scripts/fetch_synthetic_data.py 10k
    python scripts/fetch_synthetic_data.py 10k --base-url https://example.org/ascent

Then point the stack at it:

    DATA_SCALE=10k docker compose up

The base URL must serve the files at their plain names, e.g.
``<base>/omop_synthetic_10k.db.zip``. Set ASCENT_DATA_BASE_URL to avoid
passing --base-url every time.

Downloads are checksum-verified against ``data/synthetic/SHA256SUMS`` when an
entry exists, and a file that already matches is left alone -- so re-running
this is cheap and safe.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"
SUMS_FILE = DATA_DIR / "SHA256SUMS"

# One entry per file a scale needs. The warehouse builder looks for the two
# archives by these exact names (see ascent_platform.warehouse.bootstrap);
# the ground truth is what the numbers are checked against.
FILES = (
    "omop_synthetic_{scale}.db.zip",
    "source_synthetic_{scale}.db.zip",
    "ground_truth_{scale}.json",
)


def _expected_sums() -> dict[str, str]:
    if not SUMS_FILE.exists():
        return {}
    sums: dict[str, str] = {}
    for line in SUMS_FILE.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1]] = parts[0]
    return sums


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path) -> None:
    """Stream to a temporary file, then move into place.

    Writing directly to the target leaves a truncated file behind when the
    connection drops -- and a truncated archive is indistinguishable from a
    real one until the build fails deep inside zipfile.
    """
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {target.name}: {done >> 20}/{total >> 20} MiB ({pct:.0f}%)",
                          end="", flush=True)
        print()
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(target)


def fetch(scale: str, base_url: str) -> int:
    sums = _expected_sums()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for template in FILES:
        name = template.format(scale=scale)
        target = DATA_DIR / name
        expected = sums.get(name)

        if target.exists():
            if expected and _sha256(target) == expected:
                print(f"  {name}: already present and verified")
                continue
            if expected:
                print(f"  {name}: present but checksum differs — re-downloading")
            else:
                print(f"  {name}: already present (no checksum on record)")
                continue

        url = f"{base_url.rstrip('/')}/{name}"
        print(f"  {name}: fetching from {url}")
        try:
            _download(url, target)
        except urllib.error.HTTPError as err:
            print(f"\n{name}: HTTP {err.code} from {url}", file=sys.stderr)
            return 1
        except urllib.error.URLError as err:
            print(f"\n{name}: could not reach {url} ({err.reason})", file=sys.stderr)
            return 1

        if expected:
            actual = _sha256(target)
            if actual != expected:
                target.unlink(missing_ok=True)
                print(f"\n{name}: checksum mismatch\n  expected {expected}\n  got      {actual}",
                      file=sys.stderr)
                return 1
            print(f"  {name}: checksum OK")

    print(f"\nDone. Start the stack with:  DATA_SCALE={scale} docker compose up")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scale", nargs="?", default="10k",
                        help="dataset scale to fetch (default: 10k)")
    parser.add_argument("--base-url", default=os.environ.get("ASCENT_DATA_BASE_URL", ""),
                        help="where the files are published; or set ASCENT_DATA_BASE_URL")
    args = parser.parse_args()

    if args.scale == "1k":
        print("The 1k dataset ships in the repository — nothing to fetch.")
        return 0

    if not args.base_url:
        print(
            "No download location configured.\n\n"
            "The larger datasets are not committed (a 104 MiB file cannot be pushed\n"
            "to GitHub) and no public URL is baked into this script. Pass where they\n"
            "are published:\n\n"
            "    python scripts/fetch_synthetic_data.py "
            f"{args.scale} --base-url https://example.org/ascent\n\n"
            "or set ASCENT_DATA_BASE_URL. Expected filenames at that location:\n"
            + "".join(f"    {t.format(scale=args.scale)}\n" for t in FILES),
            file=sys.stderr,
        )
        return 2

    return fetch(args.scale, args.base_url)


if __name__ == "__main__":
    raise SystemExit(main())
