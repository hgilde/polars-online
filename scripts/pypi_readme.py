"""README.md for PyPI: each relative link made absolute, on GitHub, at one ref.

    python scripts/pypi_readme.py --ref v0.10.1     # rewrites README.md in place
    python scripts/pypi_readme.py --ref main --check  # says what it would change, writes nothing

PyPI shows README.md as the project page. There an in-page link (`#section`)
works, but a link to another file (`docs/PLAN.md`) resolves against pypi.org
and is a 404. The release workflow runs this on its own checkout, before
maturin reads README.md into the package metadata, so each version's page
links to the files as they were at that version's tag. The README in the
repository keeps its relative links, which are the right ones on GitHub.

Left alone: absolute links, in-page anchors, mail links, and anything inside
a fenced code block or an inline code span. A link to a file the checkout
does not have is an error here, rather than a 404 on PyPI. The repository's
address is `pyproject.toml`'s `[project.urls] Repository`. Standard library
only, so any runner's Python will do.
"""

from __future__ import annotations

import argparse
import posixpath
import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
#: `](target)`, the inline link form; the README uses no other.
LINK = re.compile(r"\]\(([^)\s]+)\)")
KEEP = ("http://", "https://", "mailto:", "#")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def repository(root: Path = REPO) -> str:
    meta = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return meta["project"]["urls"]["Repository"].rstrip("/")


def rewrite(text: str, base: str, ref: str, root: Path) -> tuple[str, list[str], list[str]]:
    """`text` with each relative link absolute, the targets it changed, and
    the targets `root` does not have."""
    changed: list[str] = []
    missing: list[str] = []

    def absolute(m: re.Match[str]) -> str:
        target = m.group(1)
        if target.startswith(KEEP):
            return m.group(0)
        path, hash_, anchor = target.partition("#")
        clean = posixpath.normpath(path)
        local = root / clean
        if clean.startswith("..") or not local.exists():
            missing.append(target)
            return m.group(0)
        kind = "tree" if local.is_dir() else "blob"
        changed.append(target)
        return f"]({base}/{kind}/{ref}/{clean}{hash_}{anchor})"

    out: list[str] = []
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        opened = FENCE.match(line)
        if opened:
            marker = opened.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            out.append(line)
            continue
        if fence is not None:
            out.append(line)
            continue
        # Outside code spans only: the even pieces between backticks.
        pieces = line.split("`")
        pieces[::2] = [LINK.sub(absolute, piece) for piece in pieces[::2]]
        out.append("`".join(pieces))
    return "".join(out), changed, missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", required=True, help="the tag or branch the links point at")
    ap.add_argument("--check", action="store_true", help="write nothing")
    args = ap.parse_args()
    readme = REPO / "README.md"
    text, changed, missing = rewrite(
        readme.read_text(encoding="utf-8"), repository(), args.ref, REPO
    )
    for target in missing:
        print(f"README.md links to {target!r}, which this checkout does not have")
    if missing:
        return 1
    print(f"{len(changed)} links now point at {repository()} at {args.ref}")
    if not args.check:
        readme.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
