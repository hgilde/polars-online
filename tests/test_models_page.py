"""``docs/MODELS.md``'s recipes, held to what they print.

Each model's entry shows the model working on a planted answer: a python
block that generates or reads its data and prints what the model found,
followed by a ```text block of that output. A recipe here runs alone, in a
directory of its own, in the namespace the README's *Example data* block
builds, as the README's own recipes do (``tests/test_production_hardening.py``,
``test_a_readme_recipe_prints_what_it_shows``), so a recipe may read ``df``
or make its own stream. What it prints is held by ``tests/doc_recipes.py``'s
comparison: the same lines and words, and each number within two units of its
last printed digit. The page's other blocks, the parameter tours, only have
to run, which ``tests/test_production_hardening.py`` sees to.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_recipes import differences, shown_outputs
from doc_recipes import run as run_recipe
from test_production_hardening import _example_data

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "MODELS.md"

RECIPES = shown_outputs(PAGE)


def test_the_page_has_its_recipes():
    """A recipe for each of the twelve entries outside the linear family at
    least, when this was written."""
    assert len(RECIPES) >= 12, [line for line, _, _ in RECIPES]


@pytest.mark.parametrize(
    ("line", "code", "shown"),
    [pytest.param(*r, id=f"MODELS.md:L{r[0]}") for r in RECIPES],
)
def test_a_recipe_prints_what_the_page_shows(line, code, shown, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a recipe writes its parquet files where it runs
    printed = run_recipe(code, Path(f"MODELS.md:{line}"), _example_data(tmp_path))
    problems = differences(shown, printed)
    assert not problems, f"MODELS.md:{line}:\n" + "\n".join(problems)
