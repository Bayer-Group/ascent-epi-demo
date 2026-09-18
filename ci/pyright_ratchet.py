"""Fail only on type errors a change introduces.

pyright is run twice over the same files, in the same tree and the same
virtualenv, with only the file contents differing: once as the branch leaves
them and once as the merge base had them. Anything present in both is
pre-existing and is reported but not enforced.

Errors are compared without line and column numbers, so inserting a line above
an untouched error does not present it as new.
"""

import json
import sys
from collections import Counter


def load(path: str) -> Counter:
    with open(path) as fh:
        report = json.load(fh)
    return Counter(
        (d.get("file", ""), d.get("rule") or "", " ".join(d.get("message", "").split()))
        for d in report.get("generalDiagnostics", [])
        if d.get("severity") == "error"
    )


def main() -> int:
    base_path, head_path = sys.argv[1], sys.argv[2]
    base, head = load(base_path), load(head_path)

    introduced = head - base
    resolved = base - head

    if resolved:
        print(f"resolved {sum(resolved.values())} pre-existing error(s)")
    carried = sum((head & base).values())
    if carried:
        print(f"carrying {carried} pre-existing error(s) in these files, not enforced")

    if not introduced:
        print("no new type errors")
        return 0

    print(f"\n{sum(introduced.values())} new type error(s):\n")
    for (file, rule, message), count in sorted(introduced.items()):
        suffix = f"  [{rule}]" if rule else ""
        for _ in range(count):
            print(f"  {file}: {message}{suffix}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
