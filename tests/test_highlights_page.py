"""``docs/HIGHLIGHTS.md``: the five tasks of the highlights page, held to the code.

Each task on the page makes up its own data with a planted answer, writes
it to parquet files and scans them, and prints what the library found
beside the answer (docs/WRITING.md §3, "A capability is shown by a
recipe"). So the page's blocks run here in order, in one directory, as a
reader runs them: a python block in one namespace, and a ```sh block
through the built ``polars-online`` on the files the blocks before it
wrote. A block followed by a ```text block is held to it with
``tests/doc_recipes.py``'s ``differences``. The README's section on the
page links it (PLAN task 365).
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
from pathlib import Path

from doc_recipes import differences, fenced_blocks
from test_production_hardening import _as_a_terminal_shows

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "HIGHLIGHTS.md"


def test_every_block_runs_in_order_and_prints_what_the_page_shows(
    tmp_path, monkeypatch, online_cli
):
    monkeypatch.chdir(tmp_path)  # the blocks write their files where they run
    monkeypatch.setenv("PATH", f"{online_cli.parent}{os.pathsep}{os.environ['PATH']}")
    ns: dict[str, object] = {"__name__": "__recipe__"}
    blocks = fenced_blocks(PAGE)
    problems: list[str] = []
    ran = 0
    for j, (kind, line, code) in enumerate(blocks):
        if kind == "python":
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(code, f"HIGHLIGHTS.md:{line}", "exec"), ns)
            printed = buf.getvalue()
        elif kind == "sh":
            printed = ""
            for command in (c for c in code.splitlines() if c.strip()):
                res = subprocess.run(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                out = res.stdout.decode("utf-8")
                assert res.returncode == 0, f"HIGHLIGHTS.md:{line}: {command}\n{out}"
                printed += out
            printed = _as_a_terminal_shows(printed)
        else:
            continue
        ran += 1
        after = blocks[j + 1] if j + 1 < len(blocks) else None
        if after is not None and after[0] == "text":
            problems += [f"HIGHLIGHTS.md:{line}:\n{p}" for p in differences(after[2], printed)]
    assert ran >= 7, ran  # five tasks, one of them in three blocks
    assert not problems, "\n\n".join(problems)


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
