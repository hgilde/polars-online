"""``docs/GBDT.md``: ``po.gbdt``'s guide, its recipes held to the code.

Every recipe on the page writes its own files, runs here in a directory of
its own, and prints what the page shows under it, so a recipe and the prose
that reads its output cannot drift from the library or from the two
libraries it builds for (docs/WRITING.md §3, "Every code block runs"). The
comparison is ``tests/doc_recipes.py``'s: the same lines and words, and each
number within two units of its last printed digit. Both public functions of
``po.gbdt`` are on the page, and the README's section on it links it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import polars_online as po
from doc_recipes import differences, recipes, run

TIER = "mixed"

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "GBDT.md"

RECIPES = recipes(PAGE)


def _param(line: int, code: str, shown: str) -> object:
    marks = []
    if "lightgbm_dataset" in code and "xgboost_matrix" in code:
        marks.append(pytest.mark.extended(reason="both libraries, 50 rounds each (1 s)"))
    return pytest.param(line, code, shown, id=f"GBDT.md:L{line}", marks=marks)


def test_the_page_has_its_recipes():
    """The opening example, the planted cuts, the two engines, a categorical
    feature and LightGBM's binning parameters: five."""
    assert len(RECIPES) >= 5, len(RECIPES)


@pytest.mark.parametrize(("line", "code", "shown"), [_param(*r) for r in RECIPES])
def test_a_recipe_prints_what_the_page_shows(line, code, shown, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a recipe writes its parquet files where it runs
    problems = differences(shown, run(code, PAGE))
    assert not problems, f"GBDT.md:{line}:\n" + "\n".join(problems)


def test_every_po_gbdt_function_is_on_the_page():
    text = PAGE.read_text(encoding="utf-8")
    missing = [name for name in po.gbdt.__all__ if f"po.gbdt.{name}" not in text]
    assert not missing, f"po.gbdt functions the page does not name: {missing}"


def test_the_readme_section_links_the_page():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    start = readme.index("\n### Training XGBoost and LightGBM on files larger than memory\n")
    section = readme[start : readme.index("\n## ", start + 1)]
    assert "](docs/GBDT.md)" in section
