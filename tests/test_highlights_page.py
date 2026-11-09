"""``docs/HIGHLIGHTS.md``: the five tasks of the highlights page, held to the code.

Each task on the page makes up its own data with a planted answer, writes
it to parquet files and scans them, and prints what the library found
(docs/WRITING.md §3, "A capability is shown by a recipe"). So the page's
python blocks run here in order, in one namespace and one directory, as a
reader runs them, and a block followed by a ```text block is held to it
with ``tests/doc_recipes.py``'s ``differences``. The page highlights the
Python API alone (the user, 2026-11-08: "we are not highlighting the
command line tool only the python api"), which a test holds too. The
README's section on the page links it (PLAN task 365).
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

from doc_recipes import differences, fenced_blocks

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "HIGHLIGHTS.md"


def test_every_block_runs_in_order_and_prints_what_the_page_shows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the blocks write their files where they run
    ns: dict[str, object] = {"__name__": "__recipe__"}
    blocks = fenced_blocks(PAGE)
    problems: list[str] = []
    ran = 0
    for j, (kind, line, code) in enumerate(blocks):
        if kind != "python":
            continue
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(code, f"HIGHLIGHTS.md:{line}", "exec"), ns)
        ran += 1
        after = blocks[j + 1] if j + 1 < len(blocks) else None
        if after is not None and after[0] == "text":
            problems += [
                f"HIGHLIGHTS.md:{line}:\n{p}" for p in differences(after[2], buf.getvalue())
            ]
    assert ran >= 6, ran  # five tasks, one of them in two blocks
    assert not problems, "\n\n".join(problems)


def test_the_page_shows_the_python_api_alone():
    """No shell block, and no run of the ``polars-online`` program."""
    blocks = fenced_blocks(PAGE)
    assert {kind for kind, _, _ in blocks} <= {"python", "text"}, {k for k, _, _ in blocks}
    text = PAGE.read_text(encoding="utf-8")
    for word in ("polars-online --", "command line", "command-line", "TOML"):
        assert word not in text, word


def test_every_capability_row_links_a_step_and_a_guide():
    """The index at the end names, for each capability, the task that shows
    it and the guide that covers it: two links a row."""
    text = PAGE.read_text(encoding="utf-8")
    index = text.split("\n## Capabilities\n", 1)[1]
    rows = [
        r for r in index.splitlines() if r.startswith("| ") and not r.startswith("| capability")
    ]
    assert len(rows) >= 19, len(rows)
    for row in rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert "](#" in cells[2], row
        assert "](" in cells[3] and "](#" not in cells[3], row


def test_the_readme_section_links_the_page():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    section = readme.split("\n## What you can do with it\n", 1)[1].split("\n## ", 1)[0]
    assert "](docs/HIGHLIGHTS.md)" in section
    assert "](docs/HIGHLIGHTS.md#capabilities)" in section
