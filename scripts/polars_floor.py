"""The floor of the polars range ``pyproject.toml`` declares.

Usage: python3 scripts/polars_floor.py      prints ``version=1.34.0``

``release.yml``'s ``floor-polars`` leg and the canary's monthly job of the
same name append that line to ``$GITHUB_OUTPUT``, then install exactly that
py-polars and its runtime package and run the suite on it. Read from the
declaration, so raising the floor moves both legs with it. A dependency line
this cannot read as ``polars>=X.Y.Z,<N`` exits naming it, rather than print a
version the leg would then install. Standard library only, so it runs on the
runner's own Python before any environment is built.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: The one form the leg can act on: a floor and a major ceiling.
RANGE = re.compile(r"polars>=(\d+\.\d+\.\d+),<\d+")


def version(root: Path = REPO) -> str:
    """The floor of the polars dependency in ``root``'s ``pyproject.toml``."""
    meta = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    found = [d for d in meta.get("dependencies", []) if re.match(r"polars(?![\w-])", d)]
    if len(found) != 1:
        sys.exit(f"pyproject.toml declares {len(found)} polars dependencies, not one: {found}")
    (line,) = found
    m = RANGE.fullmatch(line.replace(" ", ""))
    if not m:
        sys.exit(f"pyproject.toml's polars dependency {line!r} is not of the form polars>=X.Y.Z,<N")
    return m.group(1)


def main() -> int:
    print(f"version={version()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
