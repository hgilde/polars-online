"""Every link the package prints lands on a page and a heading that exist.

A message names its long form by a stable address a terminal shows in full,
``https://github.com/hgilde/polars-online/blob/main/docs/<page>.md#<anchor>``
(docs/WRITING.md §11). A released wheel keeps printing that address for as
long as it is installed, so the heading it names must stay. This reads every
such address in the Rust sources and the Python package, and holds each to a
file in the repository and, where it names one, to a heading of that file,
by GitHub's own anchors (``scripts/doc_structure.py``).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "doc_structure", REPO / "scripts" / "doc_structure.py"
)
assert _spec is not None and _spec.loader is not None
doc_structure = importlib.util.module_from_spec(_spec)
sys.modules["doc_structure"] = doc_structure
_spec.loader.exec_module(doc_structure)

#: A link into the repository's main branch, as code writes one: a Rust
#: string may continue it on the next line after a backslash.
LINK = re.compile(
    r"https://github\.com/hgilde/polars-online/blob/main/((?:[\w./-]|\\\s*)+?\.md)"
    r"(?:#((?:[\w-]|\\\s*)+))?"
)


def _sources() -> list[Path]:
    rust = sorted((REPO / "crates").glob("*/src/**/*.rs"))
    python = sorted((REPO / "python" / "polars_online").rglob("*.py"))
    return rust + python


def _links() -> list[tuple[str, str, str]]:
    found = []
    for path in _sources():
        for m in LINK.finditer(path.read_text(encoding="utf-8")):
            page, anchor = (re.sub(r"\\\s*", "", g or "") for g in m.groups())
            found.append((path.relative_to(REPO).as_posix(), page, anchor))
    return found


def test_the_links_are_found():
    """The step-back error's and the readiness warning's, at least."""
    pages = {(page, anchor) for _, page, anchor in _links()}
    assert ("docs/how-to/put-rows-in-clock-order.md", "fix-a-chunk-the-bank-refused") in pages
    assert ("docs/DIAGNOSTICS.md", "is-it-ready") in pages


def test_every_link_lands_on_a_heading_that_exists():
    problems = []
    for where, page, anchor in _links():
        target = REPO / page
        if not target.is_file():
            problems.append(f"{where}: {page} does not exist")
            continue
        if anchor:
            doc = doc_structure.read(target.read_text(encoding="utf-8"), page)
            if anchor not in doc.anchors:
                problems.append(f"{where}: {page} has no heading #{anchor}")
    assert not problems, "\n".join(problems)
