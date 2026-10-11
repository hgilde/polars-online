"""``docs/MODELS.md``: each model's recipe, held to what it prints.

A python block on the page followed by a ```text block is a recipe: it runs
here and prints what the page shows under it, so an entry's numbers cannot
drift from the library (docs/WRITING.md §3, "Every code block runs"). The
comparison is ``tests/doc_recipes.py``'s: the same lines and words, and each
number within two units of its last printed digit. The page's other blocks,
which show a model's parameters, run in ``tests/test_production_hardening.py``
with the README's.

Each recipe runs alone, in a directory of its own, in the namespace the
README's *Example data* block builds there, as the README's recipes do. So a
recipe may read ``df`` or ``trades`` (and say so in the line above it), or
make its own data and save it to parquet, as the linear models' recipes do.
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path

import pytest

from doc_recipes import differences, run, shown_outputs

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "MODELS.md"

RECIPES = shown_outputs(PAGE)

#: The linear models, each with a recipe in its entry.
LINEAR = ["ewridge", "rls", "lasso", "kalman", "huber", "sgd", "pa", "ftrl", "holt"]


def _example_data(here: Path) -> dict[str, object]:
    """What the README's *Example data* block builds, run as written in
    ``here``, where it writes its parquet files."""
    text = (REPO / "README.md").read_text(encoding="utf-8")
    start = text.index("\n### Example data\n")
    section = text[start : text.index("\n## ", start)]
    (block,) = re.findall(r"```python\n(.*?)\n```", section, re.S)
    ns: dict[str, object] = {"__name__": "__readme__"}
    with contextlib.chdir(here):
        exec(compile(block, "README.md: Example data", "exec"), ns)
    return ns


def _entry(line: int) -> str:
    """The model heading a page line sits under: its first backticked name."""
    lines = PAGE.read_text(encoding="utf-8").splitlines()[:line]
    heading = next(h for h in reversed(lines) if h.startswith("### "))
    return heading.split("`")[1]


def test_every_linear_model_has_a_recipe():
    """One recipe in each linear model's entry; `huber` and `quantile` share
    one entry, and its recipe runs both."""
    have = {_entry(line) for line, _, _ in RECIPES}
    missing = [m for m in LINEAR if m not in have]
    assert not missing, missing


@pytest.mark.parametrize(
    ("line", "code", "shown"), RECIPES, ids=[f"MODELS.md:L{line}" for line, _, _ in RECIPES]
)
def test_a_recipe_prints_what_the_page_shows(line, code, shown, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a recipe writes its parquet files where it runs
    printed = run(code, PAGE, _example_data(tmp_path))
    problems = differences(shown, printed)
    assert not problems, f"MODELS.md:{line}:\n" + "\n".join(problems)
