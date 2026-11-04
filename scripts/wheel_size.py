"""The built wheels against the README's install *size* row.

Usage: python3 scripts/wheel_size.py [--readme README.md] [--notes FILE] WHEEL...

The README states one range for every platform's wheel, with no version:
``| size | 25 to 30 MB to download and 71 to 99 MB installed, by platform |``.
A wheel's download is its file's size, and what it installs is the sum of
its members' unpacked sizes, each in megabytes of 10**6 bytes, rounded to
the whole number the row states. Every wheel a release builds must fall
inside both ranges, or the release stops before anything is uploaded: the
row is corrected in the change that moved the size, not after the release
that shipped it. ``--notes`` writes each wheel's exact sizes as a Markdown
table, for the GitHub release's notes. Exits 1 naming each wheel outside
the row. Standard library only, so it runs before any environment is built.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MB = 10**6

#: The README's size row, and the four numbers it states.
ROW = re.compile(
    r"^\| size \| (\d+) to (\d+) MB to download and (\d+) to (\d+) MB installed, "
    r"by platform \|$",
    re.M,
)


@dataclass(frozen=True)
class Size:
    """One wheel: its name, its file's bytes and its members' unpacked bytes."""

    name: str
    download: int
    installed: int


def measure(path: Path) -> Size:
    with zipfile.ZipFile(path) as z:
        installed = sum(info.file_size for info in z.infolist())
    return Size(path.name, path.stat().st_size, installed)


def stated(readme: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """The README's two ranges, download and installed, in whole MB."""
    found = ROW.findall(readme)
    if len(found) != 1:
        raise ValueError(
            f"README.md has {len(found)} size rows of the form "
            "`| size | A to B MB to download and C to D MB installed, by platform |`"
        )
    a, b, c, d = (int(n) for n in found[0])
    if not (a <= b and c <= d):
        raise ValueError(f"README.md's size row runs backwards: {found[0]}")
    return (a, b), (c, d)


def outside(sizes: list[Size], ranges: tuple[tuple[int, int], tuple[int, int]]) -> list[str]:
    """Each wheel whose download or installed size, rounded to the whole MB
    the row states, falls outside the row's range."""
    out = []
    for s in sizes:
        for what, value, (lo, hi) in (
            ("download", s.download, ranges[0]),
            ("installed", s.installed, ranges[1]),
        ):
            mb = round(value / MB)
            if not lo <= mb <= hi:
                out.append(
                    f"{s.name}: {value:,} bytes {what} is {mb} MB, outside the README's "
                    f"{lo} to {hi} MB; correct the README's size row"
                )
    return out


def notes(sizes: list[Size]) -> str:
    """The exact sizes, as the GitHub release's notes give them."""
    lines = [
        "### Wheel sizes",
        "",
        "Each wheel's download, and what it installs (the sum of its files, unpacked).",
        "",
        "| wheel | download | installed |",
        "|---|---|---|",
    ]
    for s in sorted(sizes, key=lambda s: s.name):
        lines.append(
            f"| `{s.name}` | {s.download:,} bytes ({s.download / MB:.1f} MB) "
            f"| {s.installed:,} bytes ({s.installed / MB:.1f} MB) |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("wheels", type=Path, nargs="+")
    ap.add_argument("--readme", type=Path, default=REPO / "README.md")
    ap.add_argument("--notes", type=Path, help="write the exact sizes here, as Markdown")
    args = ap.parse_args(argv)
    sizes = [measure(w) for w in args.wheels]
    ranges = stated(args.readme.read_text(encoding="utf-8"))
    table = notes(sizes)
    print(table)
    if args.notes is not None:
        args.notes.write_text(table, encoding="utf-8")
    bad = outside(sizes, ranges)
    for b in bad:
        print(f"::error::{b}")
    if not bad:
        (a, b), (c, d) = ranges
        print(f"every wheel within the README's {a} to {b} MB and {c} to {d} MB")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
