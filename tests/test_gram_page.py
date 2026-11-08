"""``docs/GRAM.md`` (docs/PLAN.md task 237): wide fits from Grams, held to
the code.

The page is an explanation (task 338 stage 9b): how a Gram stands in for the
rows, with a table from each task to its how-to. Its recipes moved to those
how-tos, whose blocks ``tests/test_kind_pages.py`` runs and holds to what
they print, so here the page is held to naming every function of ``po.gram``
(with the how-tos), to having no recipe left that no test would run, and to
the links that lead to it.
"""

from __future__ import annotations

from pathlib import Path

import polars_online as po
from doc_recipes import recipes

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "GRAM.md"
HOW_TO = REPO / "docs" / "how-to"


def test_the_page_runs_no_recipe_of_its_own():
    """Every recipe is on a how-to, where test_kind_pages runs it; a recipe
    added back to the explanation would run nowhere."""
    assert recipes(PAGE) == []


def test_the_task_table_links_a_how_to_for_each_task():
    text = PAGE.read_text(encoding="utf-8")
    for page in (
        "fit-weekly-grams-in-one-pass",
        "pool-grams",
        "fit-a-ridge-from-a-gram",
        "check-a-fit-is-determined",
        "choose-columns-on-the-lasso-path",
        "select-columns-stably-over-weeks",
        "fit-per-block-coefficients",
        "read-grams-from-saved-files",
        "store-a-compact-gram",
        "fit-hundreds-of-features",
        "fit-huber-on-a-gram",
    ):
        assert f"how-to/{page}.md" in text, page
        assert (HOW_TO / f"{page}.md").exists(), page


def test_every_po_gram_function_is_on_the_page():
    """A function added to ``po.gram`` and not to the page fails here, so the
    page, with the how-tos its tasks moved to, stays the guide to all of them."""
    text = "\n".join(
        p.read_text(encoding="utf-8")
        for p in [PAGE, *sorted((REPO / "docs" / "how-to").glob("*.md"))]
    )
    functions = [name for name in po.gram.__all__ if callable(getattr(po.gram, name))]
    assert len(functions) >= 12, functions
    missing = sorted(name for name in functions if f"po.gram.{name}" not in text)
    assert not missing, f"po.gram functions the page does not name: {missing}"


def test_the_readme_the_how_to_the_map_and_llms_txt_link_the_page():
    """The README's section on fitting from one pass moved to a how-to
    (task 338 stage 8), which leads to the page; the README keeps a link."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "docs/GRAM.md" in readme
    how_to = REPO / "docs" / "how-to" / "fit-weekly-grams-in-one-pass.md"
    assert "](../GRAM.md)" in how_to.read_text(encoding="utf-8")
    assert "](GRAM.md)" in (REPO / "docs" / "README.md").read_text(encoding="utf-8")
    assert "docs/GRAM.md" in (REPO / "llms.txt").read_text(encoding="utf-8")
