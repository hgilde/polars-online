"""The structure checks of a documentation pass (docs/WRITING.md §6, step 5).

Usage: uv run python scripts/doc_structure.py [FILE.md ...]

With no files, every Markdown file git tracks. The four checks WRITING's
table names, each as GitHub renders the file:

1. every table row has its header's cell count. GitHub pads a short row
   and drops a long row's extra cells without a word, and a row wrapped
   onto a second line becomes a row of one cell. A table GitHub does not
   render at all -- indented into a code block, or with a delimiter row of
   the wrong width -- is reported too;
2. every in-page link lands on a heading of its own file;
3. every anchor another file uses survives: ``other.md#x`` lands on a
   heading of ``other.md``;
4. a link to moved text points where it went: a relative link names a file
   git tracks, so a link left behind by a move fails here.

The file is parsed as CommonMark with GitHub's tables (markdown-it-py), so
lists, code blocks and HTML are read as GitHub reads them. Two things are
this script's own: a row's cells are counted on its source line, split as
GFM splits it, because the parser pads and trims a row to its header; and
an anchor is GitHub's -- the heading's rendered text, lowercased, with
punctuation, symbols and superscripts dropped and each space made a hyphen,
and a heading repeated in its file numbered ``-1``, ``-2`` -- or an
``<a id>`` / ``<a name>``. Both were checked against GitHub's own rendering
of every file here on 2026-09-27 (``POST /markdown``, ``mode=markdown``):
the same 1,066 anchors, tables and rows.

Prints each problem as ``path:line: what``, then how much each check
examined, since a check that examined nothing has not passed (WRITING §6,
"a check that finds nothing"). Exits 1 on any problem.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

from markdown_it import MarkdownIt
from markdown_it.token import Token

REPO = Path(__file__).resolve().parents[1]
PARSER = MarkdownIt("commonmark", {"html": True}).enable("table")

DELIMITER_CELL = re.compile(r"^:?-+:?$")
HTML_ANCHOR = re.compile(r"<a\s+(?:[^>]*\s)?(?:id|name)=\"([^\"]+)\"", re.I)
HTML_HREF = re.compile(r"<a\s+(?:[^>]*\s)?href=\"([^\"]+)\"", re.I)
SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.I)


def slug(text: str) -> str:
    """GitHub's anchor for a heading whose rendered text is ``text``:
    lowercase; keep letters, marks, decimal digits, ``-``, ``_`` and
    spaces, and drop the rest, a superscript ``²`` among them; make each
    space a hyphen."""
    kept = [
        c
        for c in text.strip().lower()
        if c in "-_ "
        or unicodedata.category(c) in ("Nd", "Nl")
        or unicodedata.category(c)[0] in "LM"
    ]
    return "".join(kept).replace(" ", "-")


def plain(inline: Token) -> str:
    """An inline token's text as GitHub renders it: a code span keeps its
    content, a link its text; HTML, images and emphasis markers go."""
    out = []
    for child in inline.children or []:
        if child.type in ("text", "code_inline"):
            out.append(child.content)
        elif child.type in ("softbreak", "hardbreak"):
            out.append(" ")
    return "".join(out)


def cells(row: str) -> list[str]:
    """A table row's cells as GFM splits them: on every ``|`` not escaped
    with a backslash -- inside a code span too -- after one leading and one
    trailing pipe."""
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|") and not row.endswith("\\|"):
        row = row[:-1]
    return re.split(r"(?<!\\)\|", row)


def _is_delimiter(line: str) -> bool:
    return "-" in line and all(DELIMITER_CELL.match(c.strip()) for c in cells(line))


@dataclass
class Doc:
    """One Markdown file, read the way GitHub reads it."""

    anchors: set[str] = field(default_factory=set)
    headings: int = 0
    # (line number, link target) for every link.
    links: list[tuple[int, str]] = field(default_factory=list)
    tables: int = 0
    rows: int = 0
    problems: list[str] = field(default_factory=list)


def read(text: str, rel: str) -> Doc:
    doc = Doc()
    lines = text.splitlines()
    # github-slugger's bookkeeping: every slug it has handed out, and how
    # many times each base has been numbered. An `<a id>` takes no part.
    taken: dict[str, int] = {}

    def heading(title: str) -> None:
        base = anchor = slug(title)
        while anchor in taken:
            taken[base] += 1
            anchor = f"{base}-{taken[base]}"
        taken[anchor] = 0
        doc.anchors.add(anchor)
        doc.headings += 1

    def line_of(span: list[int] | None, target: str) -> int:
        """The source line a link sits on, 1-based: the first line of its
        block that holds the target as written."""
        if not span:
            return 0
        for n in range(span[0], span[1]):
            if target in lines[n] or unquote(target) in lines[n]:
                return n + 1
        return span[0] + 1

    def html(content: str, span: list[int] | None) -> None:
        doc.anchors.update(HTML_ANCHOR.findall(content))
        for href in HTML_HREF.findall(content):
            doc.links.append((line_of(span, href), href))

    def lookalike(body: list[str], first: int, rendered_as: str) -> None:
        """A header and a delimiter row that GitHub does not render as a
        table."""
        for k in range(len(body) - 1):
            if "|" in body[k] and "|" in body[k + 1] and _is_delimiter(body[k + 1]):
                doc.problems.append(
                    f"{rel}:{first + k + 1}: a table GitHub renders as {rendered_as}"
                )

    tokens = PARSER.parse(text)
    in_head, width, block = False, 0, None
    for i, tok in enumerate(tokens):
        if tok.map:
            block = tok.map
        if tok.type == "heading_open":
            heading(plain(tokens[i + 1]))
        elif tok.type == "thead_open":
            in_head = True
        elif tok.type == "thead_close":
            in_head = False
        elif tok.type == "tr_open" and tok.map:
            n = tok.map[0]
            got = len(cells(lines[n]))
            if in_head:
                doc.tables += 1
                width = got
            else:
                doc.rows += 1
                if got != width:
                    what = "drops the extra cells" if got > width else "pads it with empty cells"
                    doc.problems.append(
                        f"{rel}:{n + 1}: a table row of {got} cell{'s' * (got != 1)} "
                        f"under a header of {width}: GitHub {what}"
                    )
        elif tok.type == "code_block" and tok.map:
            lookalike(
                lines[tok.map[0] : tok.map[1]],
                tok.map[0],
                "code: it sits four or more spaces past the text it belongs to",
            )
        elif tok.type == "paragraph_open" and tok.map:
            body = lines[tok.map[0] : tok.map[1]]
            lookalike(body, tok.map[0], "text: its delimiter row is not its header's width")
        elif tok.type == "html_block":
            html(tok.content, tok.map)
        elif tok.type == "inline":
            for child in tok.children or []:
                if child.type == "link_open":
                    href = str(child.attrs.get("href", ""))
                    doc.links.append((line_of(block, href), href))
                elif child.type == "image":
                    src = str(child.attrs.get("src", ""))
                    doc.links.append((line_of(block, src), src))
                elif child.type == "html_inline":
                    html(child.content, block)
    return doc


def tracked(root: Path) -> set[str]:
    """Every file git tracks under ``root``, and every directory holding one."""
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True, check=True
    ).stdout
    files = {f for f in out.split("\0") if f}
    dirs = {Path(f).parent.as_posix() for f in files}
    for d in list(dirs):
        dirs.update(p.as_posix() for p in Path(d).parents)
    return files | dirs


def _normalise(path: str) -> str:
    parts: list[str] = []
    for part in path.split("/"):
        if part in ("", "."):
            continue
        if part == ".." and parts and parts[-1] != "..":
            parts.pop()
        else:
            parts.append(part)
    return "/".join(parts) or "."


def check(
    root: Path, names: list[str], known: set[str] | None = None
) -> tuple[list[str], Counter[str]]:
    """Every problem in ``names`` (paths relative to ``root``), and how much
    each check examined. ``known`` is the set of paths a link may name,
    git's by default."""
    known = tracked(root) if known is None else known
    docs: dict[str, Doc] = {}

    def doc_for(rel: str) -> Doc:
        if rel not in docs:
            docs[rel] = read((root / rel).read_text(encoding="utf-8"), rel)
        return docs[rel]

    problems: list[str] = []
    counts: Counter[str] = Counter()
    for rel in names:
        d = doc_for(rel)
        problems += d.problems
        counts["files"] += 1
        counts["headings"] += d.headings
        counts["tables"] += d.tables
        counts["table rows"] += d.rows
        for n, target in d.links:
            if SCHEME.match(target) or target.startswith("//"):
                continue
            path_part, _, anchor = target.partition("#")
            anchor = unquote(anchor)
            if not path_part:
                counts["in-page links"] += 1
                if anchor and anchor not in d.anchors:
                    problems.append(f"{rel}:{n}: #{anchor} is no heading of this file")
                continue
            counts["links to other files"] += 1
            path_part = unquote(path_part)
            if path_part.startswith("/"):
                name = _normalise(path_part)
            else:
                name = _normalise(f"{Path(rel).parent.as_posix()}/{path_part}")
            if name not in known:
                problems.append(f"{rel}:{n}: {path_part} names no file git tracks")
                continue
            if anchor and name.endswith(".md"):
                counts["anchors in other files"] += 1
                if anchor not in doc_for(name).anchors:
                    problems.append(f"{rel}:{n}: #{anchor} is no heading of {name}")
    return problems, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", help="Markdown files; every tracked one when none")
    args = parser.parse_args()
    names = [Path(f).resolve().relative_to(REPO).as_posix() for f in args.files] or sorted(
        f for f in tracked(REPO) if f.endswith(".md")
    )
    problems, counts = check(REPO, names)
    for p in problems:
        print(p)
    print(", ".join(f"{v} {k}" for k, v in counts.items()))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
