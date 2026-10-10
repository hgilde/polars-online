"""``docs/GRAM.md`` (docs/PLAN.md task 237): wide fits from Grams, held to
the code.

Every recipe on the page runs here and prints what the page shows under it,
so a recipe and the prose that reads its output cannot drift from the library
(docs/WRITING.md §3, "Every code block runs"). The comparison is
``tests/doc_recipes.py``'s, which ``tests/test_diagnostics_page.py`` and
``tests/test_data_quality.py`` use too: the same lines and words, and each
number within two units of its last printed digit. A line that reports a wall
time is the machine's, so it is held to its words and its numbers are left
free. Every function of ``po.gram`` is on the page, and the README, the
document map and ``llms.txt`` link it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import polars_online as po
from doc_recipes import NUMBER, differences, recipes, run

TIER = "mixed"

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "GRAM.md"

RECIPES = recipes(PAGE)

#: A printed line that starts so reports a wall time.
TIMED = "time"


def _untimed(text: str) -> str:
    """``text`` with the numbers of each timed line replaced, so the line is
    held to its words alone."""
    return "\n".join(
        NUMBER.sub("#", line) if line.startswith(TIMED) else line for line in text.splitlines()
    )


def _param(line: int, code: str, shown: str) -> object:
    marks = []
    if "gram_threads=threads" in code:
        marks.append(
            pytest.mark.extended(reason="threads: a 2,000-column Gram learned on 1 and 8 (1.2 s)")
        )
    return pytest.param(line, code, shown, id=f"GRAM.md:L{line}", marks=marks)


def test_the_page_has_its_recipes():
    """The opening example and a recipe for each task: twelve."""
    assert len(RECIPES) >= 12, len(RECIPES)


@pytest.mark.parametrize(("line", "code", "shown"), [_param(*r) for r in RECIPES])
def test_a_recipe_prints_what_the_page_shows(line, code, shown, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a recipe writes its parquet files where it runs
    problems = differences(_untimed(shown), _untimed(run(code, PAGE)))
    assert not problems, f"GRAM.md:{line}:\n" + "\n".join(problems)


def test_a_timed_line_is_held_to_its_words_alone():
    def differ(shown: str, printed: str) -> bool:
        return bool(differences(_untimed(shown), _untimed(printed)))

    shown = "time: 0.20 s by merge, 0.02 s by solve_subsets\nrows 217"
    assert not differ(shown, "time: 1.75 s by merge, 0.3 s by solve_subsets\nrows 217")
    assert differ(shown, "time: 1.75 s by merges, 0.3 s by solve_subsets\nrows 217")
    assert differ(shown, "time: 0.20 s by merge, 0.02 s by solve_subsets\nrows 230")
    # Only a line that starts with the word: a number elsewhere is held.
    assert differ("the time is 0.20 s", "the time is 9.99 s")


def test_the_page_times_only_what_a_machine_decides():
    """Two timed lines, and every other number held: a recipe that printed
    its results on a timed line would escape the comparison."""
    timed = [
        line for _, _, shown in RECIPES for line in shown.splitlines() if line.startswith(TIMED)
    ]
    assert len(timed) == 2, timed
    assert all(" s " in line or line.endswith(" as fast") for line in timed), timed


def test_every_po_gram_function_is_on_the_page():
    """A function added to ``po.gram`` and not to the page fails here, so the
    page stays the guide to all of them."""
    text = PAGE.read_text(encoding="utf-8")
    functions = [name for name in po.gram.__all__ if callable(getattr(po.gram, name))]
    assert len(functions) >= 12, functions
    missing = sorted(name for name in functions if f"po.gram.{name}" not in text)
    assert not missing, f"po.gram functions the page does not name: {missing}"


def test_the_readme_the_map_and_llms_txt_link_the_page():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    start = readme.index("\n### The running sums behind a fit")
    section = readme[start : readme.index("\n### ", start + 1)]
    assert "docs/GRAM.md" in section
    assert "](GRAM.md)" in (REPO / "docs" / "README.md").read_text(encoding="utf-8")
    assert "docs/GRAM.md" in (REPO / "llms.txt").read_text(encoding="utf-8")
