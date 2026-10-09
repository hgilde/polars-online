"""``docs/DIAGNOSTICS.md`` (docs/PLAN.md task 222): is the model working,
held to the code.

Every recipe on the page runs here, and prints what the page shows under it,
so a recipe and the prose that reads its output cannot drift from the library
(docs/WRITING.md §3, "Every code block runs"). The comparison is
``tests/doc_recipes.py``'s, which ``tests/test_data_issues.py`` uses too:
the same lines and words, each number within two units of its last printed
digit, and a number past a million held to its sign and size only. Every
diagnostic switch a spec takes is on the page, and the README, the document
map and ``llms.txt`` link it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import polars_online as po
from doc_recipes import differences, recipes, run

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "DIAGNOSTICS.md"


RECIPES = recipes(PAGE)


def test_the_page_has_its_recipes():
    """The opening example, every switch at once, and a recipe for each
    question, the worked example and the simulated data: eighteen."""
    assert len(RECIPES) >= 18, len(RECIPES)


@pytest.mark.parametrize(
    ("line", "code", "shown"), RECIPES, ids=[f"DIAGNOSTICS.md:L{line}" for line, _, _ in RECIPES]
)
def test_a_recipe_prints_what_the_page_shows(line, code, shown, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a recipe writes its parquet file where it runs
    problems = differences(shown, run(code, PAGE))
    assert not problems, f"DIAGNOSTICS.md:{line}:\n" + "\n".join(problems)


def _switches() -> set[str]:
    """Every diagnostic switch a linear model's spec carries: its ``emit_*``
    keys, ``conformal`` and ``resid_quantiles``."""
    spec = po.spec.ewridge("m", targets=["y"], features=["x"], half_life=1.0)
    return {k for k in spec if k.startswith("emit_")} | {"conformal", "resid_quantiles"}


def test_every_switch_is_on_the_page():
    """A switch added to the specs and not to the page fails here, so the page
    stays the one place that answers "is the model working?"."""
    text = PAGE.read_text(encoding="utf-8")
    switches = _switches()
    assert len(switches) >= 19, switches
    missing = sorted(s for s in switches if f"`{s}`" not in text and f"{s}=" not in text)
    assert not missing, f"switches the page does not name: {missing}"


def test_every_switch_runs_in_the_every_switch_recipe():
    """*Every switch at once* turns each switch on, so its output lists every
    field the switches write."""
    block = next(code for _, code, _ in RECIPES if 'po.spec.ewridge(\n    "every"' in code)
    missing = sorted(
        s for s in _switches() - {"emit_selected", "emit_averaged"} if f"{s}=" not in block
    )
    assert not missing, missing


def test_the_readme_the_map_and_llms_txt_link_the_page():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    start = readme.index("\n## Diagnostics, selection and evaluation")
    section = readme[start : readme.index("\n## ", start + 1)]
    assert "docs/DIAGNOSTICS.md" in section
    assert "](DIAGNOSTICS.md)" in (REPO / "docs" / "README.md").read_text(encoding="utf-8")
    assert "docs/DIAGNOSTICS.md" in (REPO / "llms.txt").read_text(encoding="utf-8")
    assert "DIAGNOSTICS.md" in (REPO / "docs" / "DATA-ISSUES.md").read_text(encoding="utf-8")
