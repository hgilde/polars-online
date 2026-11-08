"""``docs/CALIBRATION.md`` (docs/PLAN.md task 336): choosing a model's
settings from the stream's own errors, held to the code.

Every python block on the page is a recipe: a complete program that writes
its own data and prints what the ```text block under it shows, held by
``tests/doc_recipes.py``'s comparison (the same lines and words, each number
within two units of its last printed digit; docs/WRITING.md §3). The README,
the document map and ``llms.txt`` link the page.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_recipes import differences, recipes, run

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "CALIBRATION.md"

RECIPES = recipes(PAGE)


def test_the_page_has_its_recipes():
    """The opening half-life, the hand form beside the call, and one recipe for
    each of the six settings. The optimism of the best over the same rows moved
    to docs/explanation/grids-and-instances.md (task 338 stage 5b), whose
    recipe tests/test_kind_pages.py runs."""
    assert len(RECIPES) >= 8, len(RECIPES)


@pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
@pytest.mark.parametrize(
    ("line", "code", "shown"), [pytest.param(*r, id=f"CALIBRATION.md:L{r[0]}") for r in RECIPES]
)
def test_a_recipe_prints_what_the_page_shows(line, code, shown, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a recipe writes its parquet file where it runs
    problems = differences(shown, run(code, PAGE))
    assert not problems, f"CALIBRATION.md:{line}:\n" + "\n".join(problems)


def test_the_readme_the_map_and_llms_txt_link_the_page():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    start = readme.index("\n### Calibrating a model")
    section = readme[start : readme.index("\n## ", start + 1)]
    assert "docs/CALIBRATION.md" in section
    assert "po.eval.grid_search" in section
    assert "](CALIBRATION.md)" in (REPO / "docs" / "README.md").read_text(encoding="utf-8")
    assert "docs/CALIBRATION.md" in (REPO / "llms.txt").read_text(encoding="utf-8")
