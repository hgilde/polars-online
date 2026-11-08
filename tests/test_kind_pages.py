"""The pages of the documentation rewrite, held to the code (PLAN task 338).

Every page under ``docs/how-to/``, ``docs/explanation/`` and
``docs/tutorials/`` is a page of one of WRITING §9's kinds, and its examples
are claims like any other (docs/WRITING.md §3). So each page's python blocks
run here in order, in the namespace the README's *Example data* builds (a
page names that as a prerequisite and links it), as a reader runs them: a
block reads what an earlier block of the page built. A block followed by a
```text block is held to it with ``tests/doc_recipes.py``'s
``differences``, a line that starts with ``time`` held to its words alone
(``untimed``). Each ```sh block followed by a ```text block runs through the
built ``polars-online`` in a directory of its own, as the runner guide's do.

A page is found by its directory, so a page a later stage adds is held from
the change that adds it.
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

from doc_recipes import differences, fenced_blocks, shown_outputs, untimed
from test_production_hardening import _as_a_terminal_shows, _readme_namespace

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
KINDS = ("how-to", "explanation", "tutorials")
PAGES = sorted(p for kind in KINDS for p in (REPO / "docs" / kind).glob("*.md"))


def _rel(page: Path) -> str:
    return page.relative_to(REPO).as_posix()


def test_the_pages_are_found():
    """The pages stage 1 of the rewrite writes, at least."""
    found = {_rel(p) for p in PAGES}
    for want in (
        "docs/how-to/put-rows-in-clock-order.md",
        "docs/how-to/resume-from-a-saved-state.md",
        "docs/explanation/out-of-sample.md",
    ):
        assert want in found, (want, found)


@pytest.mark.parametrize("page", PAGES, ids=_rel)
def test_a_pages_python_runs_in_order_and_prints_what_it_shows(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a block writes its files where it runs
    ns: dict[str, object] = {"__name__": "__recipe__", **_readme_namespace(tmp_path)}
    blocks = fenced_blocks(page)
    problems: list[str] = []
    for j, (kind, line, code) in enumerate(blocks):
        if kind != "python":
            continue
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            # A block inside a numbered step is indented with the step.
            exec(compile(textwrap.dedent(code), f"{_rel(page)}:{line}", "exec"), ns)
        after = blocks[j + 1] if j + 1 < len(blocks) else None
        if after is not None and after[0] == "text":
            # A line that reports a time is the machine's: its words are held, as the README's are.
            shown = untimed(textwrap.dedent(after[2]))
            printed = untimed(buf.getvalue())
            problems += [f"{_rel(page)}:{line}:\n{p}" for p in differences(shown, printed)]
    assert not problems, "\n\n".join(problems)


SHELL = [
    pytest.param(page, line, code, shown, id=f"{_rel(page)}:L{line}")
    for page in PAGES
    for line, code, shown in shown_outputs(page, "sh")
]


@pytest.mark.parametrize(("page", "line", "code", "shown"), SHELL)
def test_a_pages_shell_recipe_prints_what_it_shows(
    page, line, code, shown, tmp_path, monkeypatch, online_cli
):
    monkeypatch.setenv("PATH", f"{online_cli.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.chdir(tmp_path)
    _readme_namespace(tmp_path)  # writes job.toml, run.state and the inputs
    printed = ""
    for command in (c for c in code.splitlines() if c.strip()):
        res = subprocess.run(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
        )
        out = res.stdout.decode("utf-8")
        assert res.returncode == 0, f"{_rel(page)}:{line}: {command}\n{out}"
        printed += out
    problems = differences(shown, _as_a_terminal_shows(printed))
    assert not problems, f"{_rel(page)}:{line}:\n" + "\n".join(problems)


@pytest.mark.parametrize("page", PAGES, ids=_rel)
def test_a_page_declares_its_kind_on_its_first_line(page):
    first = page.read_text(encoding="utf-8").splitlines()[0]
    kind = {"how-to": "how-to", "explanation": "explanation", "tutorials": "tutorial"}[
        page.parent.name
    ]
    assert first == f"<!-- kind: {kind} -->", (_rel(page), first)
