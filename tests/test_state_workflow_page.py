"""``docs/STATE-WORKFLOW.md``: the state workflow's guide, held to the code.

The guide part of the page, everything before its dated record (*The
research behind it*), shows each capability by a recipe: a complete program
that generates its data, saves it to parquet and reads it back, followed by
a ```text block of what it prints. Each runs here alone, in an empty
namespace and a directory of its own, and what it prints is held to the page
by ``tests/doc_recipes.py``'s comparison: the same lines and words, and each
number within two units of its last printed digit (docs/WRITING.md §3,
"Every code block runs"). The record keeps its code as it was written.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_recipes import differences, fenced_blocks, run, shown_outputs

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "STATE-WORKFLOW.md"

#: The line the dated record starts on: the guide is everything above it.
RECORD = (
    PAGE.read_text(encoding="utf-8").split("\n").index("## The research behind it, 2026-09-03") + 1
)

#: The guide's recipes: each python block above the record, with what it prints.
RECIPES = [r for r in shown_outputs(PAGE) if r[0] < RECORD]


def test_every_python_block_of_the_guide_is_a_recipe():
    """The four steps, a query's writes, resuming on an overlap, and what a
    saved bank carries: each block shows what it prints."""
    guide = [line for kind, line, _ in fenced_blocks(PAGE) if kind == "python" and line < RECORD]
    assert guide == [line for line, _, _ in RECIPES], guide
    assert len(RECIPES) >= 4, len(RECIPES)


@pytest.mark.parametrize(
    ("line", "code", "shown"),
    [pytest.param(*r, id=f"STATE-WORKFLOW.md:L{r[0]}") for r in RECIPES],
)
def test_a_recipe_prints_what_the_page_shows(line, code, shown, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a recipe writes its parquet and state files where it runs
    problems = differences(shown, run(code, PAGE))
    assert not problems, f"STATE-WORKFLOW.md:{line}:\n" + "\n".join(problems)
