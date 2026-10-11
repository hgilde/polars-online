"""WRITING's step-5 structure checks (docs/WRITING.md §6), over every
Markdown file git tracks, and the checker's rules pinned on small documents.

Every expectation below is what GitHub's renderer did with the same text
on 2026-09-27 (``POST /markdown``, ``mode=markdown``), not what the checker
says: the anchors it wrote, the cells each row kept, and which blocks it
made a table, a paragraph or code.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

from doc_recipes import shown_outputs

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "doc_structure", REPO / "scripts" / "doc_structure.py"
)
assert _spec and _spec.loader
doc_structure = importlib.util.module_from_spec(_spec)
# Registered first: a dataclass looks its module up while it is built.
sys.modules["doc_structure"] = doc_structure
_spec.loader.exec_module(doc_structure)


def test_every_tracked_document_keeps_its_structure():
    names = sorted(f for f in doc_structure.tracked(REPO) if f.endswith(".md"))
    problems, counts = doc_structure.check(REPO, names)
    assert not problems, "\n".join(problems)
    # A check that examined nothing has not passed (WRITING §6's traps).
    for what in (
        "headings",
        "tables",
        "table rows",
        "in-page links",
        "links to other files",
        "anchors in other files",
    ):
        assert counts[what] > 0, (what, counts)
    assert counts["files"] == len(names) > 30


def test_every_document_under_docs_has_a_row_in_the_index():
    """`docs/README.md` is the map of `docs/`: each document a guide or a
    record, with what it is for. Nothing held it to the files, and two logs,
    PHRASING and README-ITERATIONS, sat under Guides (review 2026-10-06,
    DB14). Every Markdown file in `docs/` and `docs/records/` has a row that
    links it, every row links a file that exists, and the two logs are
    records."""
    import re

    docs = REPO / "docs"
    index = (docs / "README.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| \[[^\]]+\]\(([^)#]+)\)", index, re.M)
    files = {
        p.relative_to(docs).as_posix() for d in (docs, docs / "records") for p in d.glob("*.md")
    } - {"README.md"}
    assert len(files) > 30, files
    assert sorted(files - set(rows)) == [], "documents with no row in docs/README.md"
    assert [r for r in rows if not (docs / r).exists()] == [], "rows that link no file"
    records = index.split("\n## Records\n", 1)[1].split("\n## ", 1)[0]
    for log in ("PHRASING.md", "README-ITERATIONS.md"):
        assert re.search(rf"^\| \[{log}\]\(([\w/]*){log}\)", records, re.M), log


def test_the_pages_the_readme_moved_out_are_guides_to_using_it():
    """The README's model entries, its windows and `po.gbdt`'s section moved
    to pages of their own (README-ITERATIONS, I8: C1 to C3). Each is a guide
    a user reads, so `docs/README.md` lists it under *Using the library*,
    where a record's row would mislead, and the README links it."""
    index = (REPO / "docs" / "README.md").read_text(encoding="utf-8")
    using = index.split("| **Using the library** | |", 1)[1].split("| **Changing", 1)[0]
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for page in ("MODELS.md", "WINDOWS.md", "GBDT.md"):
        assert f"| [{page}]({page}) |" in using, page
        assert f"](docs/{page}" in readme, page


def test_anchors_are_githubs():
    doc = doc_structure.read(
        "# A\n\n# A\n\n# A-1\n\n## A-1\n\n"
        "# `po.spec.ew_ridge(...)` and **bold** — with [a link](x.md)\n\n"
        "# Cost at 10⁴ features × 50² targets, ε ≤ 1\n\n"
        '<a id="kept"></a>\n\n# kept\n',
        "x.md",
    )
    assert doc.anchors == {
        "a",
        "a-1",
        "a-1-1",
        "a-1-2",
        "pospecew_ridge-and-bold--with-a-link",
        "cost-at-10-features--50-targets-ε--1",
        "kept",
    }
    assert doc.headings == 7


def _problems(text: str) -> list[str]:
    return doc_structure.read(text, "x.md").problems


def test_a_row_of_the_wrong_width_is_found():
    found = _problems(
        "| a | b |\n|---|---|\n| 1 | 2 |\n| 1 | 2 | 3 |\n| 1 |\n| wrapped\n| `x | y` | 2 |\n"
    )
    assert found == [
        "x.md:4: a table row of 3 cells under a header of 2: GitHub drops the extra cells",
        "x.md:5: a table row of 1 cell under a header of 2: GitHub pads it with empty cells",
        "x.md:6: a table row of 1 cell under a header of 2: GitHub pads it with empty cells",
        # A pipe splits a cell inside a code span too: GitHub showed "`x",
        # "y`" and dropped the 2.
        "x.md:7: a table row of 3 cells under a header of 2: GitHub drops the extra cells",
    ]


def test_an_escaped_pipe_is_no_cell():
    doc = doc_structure.read("| a | b |\n|---|---|\n| x \\| y | 2 |\n", "x.md")
    assert (doc.problems, doc.tables, doc.rows) == ([], 1, 1)


def test_a_table_github_does_not_render_is_found():
    text = (
        "- item text\n\n      | a | b |\n      |---|---|\n      | 1 | 2 |\n\n"
        "- item text\n\n    | c | d |\n    |---|---|\n    | 1 | 2 |\n\n"
        "| e | f |\n|---|---|---|\n| 1 | 2 |\n"
    )
    doc = doc_structure.read(text, "x.md")
    assert doc.problems == [
        "x.md:3: a table GitHub renders as code: it sits four or more spaces past the "
        "text it belongs to",
        "x.md:13: a table GitHub renders as text: its delimiter row is not its header's width",
    ]
    # The table four spaces into its item is one, and nothing is wrong with it.
    assert (doc.tables, doc.rows) == (1, 1)


def test_code_holds_no_links_and_no_headings():
    doc = doc_structure.read(
        "```\n# not a heading\n[x](#nowhere)\n```\n\n`[y](#nowhere)`\n\n    [z](#nowhere)\n",
        "x.md",
    )
    assert (doc.headings, doc.links, doc.problems) == (0, [], [])


def test_links_land_on_headings_in_this_file_and_others(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "a.md").write_text(
        "# Top\n\n"
        "[ok](#top) [gone](#nowhere) [self]() [anchor](#kept) [page](#)\n\n"
        '<a id="kept"></a>\n\n'
        "[there](docs/b.md#there) [missing](docs/b.md#gone) [moved](docs/c.md)\n"
        "[encoded](docs/b.md#%CE%B5-in-a-heading) [dir](docs) [site](https://example.com/#x)\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "b.md").write_text(
        "# There\n\n# ε in a heading\n\n[back](../a.md#top) [up](../../a.md)\n", encoding="utf-8"
    )
    known = {"a.md", "docs/b.md", "docs", "."}
    problems, counts = doc_structure.check(tmp_path, ["a.md", "docs/b.md"], known=known)
    assert problems == [
        "a.md:3: #nowhere is no heading of this file",
        "a.md:7: #gone is no heading of docs/b.md",
        "a.md:7: docs/c.md names no file git tracks",
        "docs/b.md:5: ../../a.md names no file git tracks",
    ]
    assert counts["in-page links"] == 5
    assert counts["anchors in other files"] == 4


def _readme_sections() -> list[dict]:
    """The README's headings in order, each with its level, its anchor, the
    links of its own text (before its first subheading, code apart), and the
    fenced blocks of its whole section, subsections included, as (info,
    line) pairs. Read as GitHub reads the file, so a link inside a code
    block or a code span is no link."""
    text = (REPO / "README.md").read_text(encoding="utf-8")
    tokens = doc_structure.PARSER.parse(text)
    sections: list[dict] = []
    taken: set[str] = set()
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open":
            title = doc_structure.plain(tokens[i + 1])
            anchor = doc_structure.slug(title)
            assert anchor not in taken, f"the README repeats the heading {title!r}"
            taken.add(anchor)
            sections.append({"level": int(tok.tag[1]), "anchor": anchor, "links": [], "own": []})
        elif sections and tok.type == "inline" and tokens[i - 1].type != "heading_open":
            for child in tok.children or []:
                if child.type == "link_open":
                    sections[-1]["links"].append(child.attrs["href"])
        elif sections and tok.type == "fence" and tok.map:
            sections[-1]["own"].append(tok.map[0] + 1)
    # A section's blocks are its own and every subsection's.
    for k, sec in enumerate(sections):
        sec["blocks"] = list(sec["own"])
        for sub in sections[k + 1 :]:
            if sub["level"] <= sec["level"]:
                break
            sec["blocks"] += sub["own"]
    return sections


def _guides(group: str) -> list[str]:
    """The guides `docs/README.md` lists under one bold group of its table."""
    index = (REPO / "docs" / "README.md").read_text(encoding="utf-8")
    rows = index.split("\n## Guides\n", 1)[1].split("\n## ", 1)[0]
    part = rows.split(f"| **{group}** | |", 1)[1].split("| **", 1)[0]
    return re.findall(r"^\| \[[^\]]+\]\(([^)#]+)\)", part, re.M)


def test_every_guide_has_a_readme_section():
    """A casual reader finds a feature by scanning the README, so every
    guide to using the library has a README section of its own (the user,
    2026-10-10: "features with their own doc pages have a section in the
    main readme so a casual user can find"). For each guide under *Using the
    library* in `docs/README.md`, but VALIDATION, which is generated and
    only says where a default came from, some README heading at level 2 or
    3 links the guide as the first link of its own text, that heading is in
    the contents table, and its section shows the capability with a recipe:
    a python block, or for the command line a shell block, followed by a
    ```text block of what it prints, which the README's recipe tests hold.
    Each guide for changing the library or writing its docs is linked from
    the README at least once."""
    readme = REPO / "README.md"
    text = readme.read_text(encoding="utf-8")
    contents = text.split("\n## ", 1)[0]
    recipes = {line for kind in ("python", "sh") for line, _, _ in shown_outputs(readme, kind)}
    sections = _readme_sections()
    using = [g for g in _guides("Using the library") if g != "VALIDATION.md"]
    assert len(using) >= 11, using
    problems = []
    for guide in using:
        owners = [
            s
            for s in sections
            if s["level"] in (2, 3) and s["links"] and s["links"][0] == f"docs/{guide}"
        ]
        listed = [s for s in owners if f"](#{s['anchor']})" in contents]
        shown = [s for s in listed if recipes & set(s["blocks"])]
        if not owners:
            problems.append(f"{guide}: no README section links it first")
        elif not listed:
            problems.append(f"{guide}: {[s['anchor'] for s in owners]} not in the contents table")
        elif not shown:
            problems.append(f"{guide}: {[s['anchor'] for s in listed]} hold no recipe")
    for group in ("Changing the library", "Writing the documentation"):
        for guide in _guides(group):
            if f"](docs/{guide}" not in text:
                problems.append(f"{guide}: the README never links it")
    assert not problems, "\n".join(problems)
