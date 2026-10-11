"""``docs/PERFORMANCE.md``: the recipe of its guide part, held to the code.

The page's guide part opens with a recipe that times each part of a run,
followed by a ```text block of what it prints. It runs here in a directory
of its own and is held to that block by ``tests/doc_recipes.py``'s
comparison (docs/WRITING.md §3, "Every code block runs"). Its lines report
wall times, which are the machine's, so each is held to its words and its
numbers are left free (``untimed``). The measurement log's own blocks are
records, run by no test, so only a block followed by a ```text block is
held (``shown_outputs``).
"""

from __future__ import annotations

from pathlib import Path

from doc_recipes import TIMED, differences, run, shown_outputs, untimed

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "PERFORMANCE.md"

RECIPES = shown_outputs(PAGE)


def test_the_guide_part_has_its_recipe():
    head = PAGE.read_text(encoding="utf-8").split("\n## The first pass", 1)[0]
    assert len(RECIPES) >= 1
    assert all(code in head for _, code, _ in RECIPES), "a recipe outside the guide part"


def test_a_recipe_prints_what_the_page_shows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the recipe writes its parquet file where it runs
    problems = []
    for line, code, shown in RECIPES:
        problems += [
            f"PERFORMANCE.md:{line}:\n{p}"
            for p in differences(untimed(shown), untimed(run(code, PAGE)))
        ]
    assert not problems, "\n\n".join(problems)


def test_the_page_times_only_what_a_machine_decides():
    """Every line the recipe prints is a timed one, and each reports a rate:
    a result printed on a timed line would escape the comparison."""
    for _, _, shown in RECIPES:
        for text in shown.splitlines():
            assert text.startswith(TIMED) and text.endswith("rows a second"), text
