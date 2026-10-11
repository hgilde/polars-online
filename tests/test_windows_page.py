"""``docs/WINDOWS.md``: the window operators' guide, held to the code.

Every python block on the page is a recipe: a complete program whose printed
output the ```text block under it shows. Each reads only what the README's
*Example data* block builds (``trades``, ``df`` and the parquet files it
writes) or builds its own frame, so each runs here alone, in a directory of
its own, in the namespace that block builds as written. What it prints is
held to the page by ``tests/doc_recipes.py``'s comparison: the same lines and
words, and each number within two units of its last printed digit
(docs/WRITING.md §3, "Every code block runs").
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_recipes import differences, recipes, run
from test_production_hardening import _example_data

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "WINDOWS.md"

RECIPES = recipes(PAGE)


def test_the_page_has_its_recipes():
    """One for each claim the page shows working: the operators against
    Polars, the keywords, `closed`, the refused formulas, two calls, `partial`,
    a VWAP, clock order, a window target, z-scores and resuming."""
    assert len(RECIPES) >= 11, len(RECIPES)


@pytest.mark.parametrize(
    ("line", "code", "shown"), [pytest.param(*r, id=f"WINDOWS.md:L{r[0]}") for r in RECIPES]
)
def test_a_recipe_prints_what_the_page_shows(line, code, shown, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # Example data writes its files here, and so does the recipe
    printed = run(code, PAGE, _example_data(tmp_path))
    problems = differences(shown, printed)
    assert not problems, f"WINDOWS.md:{line}:\n" + "\n".join(problems)
