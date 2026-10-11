"""Every model entry of ``docs/MODELS.md`` outside the linear family shows the
model working: a python block followed by a ```text block of what it prints.

``tests/test_models_page.py`` runs every such block of the page and holds it
to its output. This test holds the other side: that each entry of the three
families (moments and correlation, clustering and classification, sequential
tests and regimes) has one, so an entry cannot drop its recipe and pass.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from doc_recipes import shown_outputs

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "MODELS.md"

#: The entries of the three families, by the model each heading names.
MODELS = [
    "ew_cov",
    "marginal",
    "deco",
    "rcov",
    "audit",
    "kmeans",
    "micro",
    "ew_class",
    "seqtest",
    "corrchange",
    "bocpd",
    "hmm",
]


def _entries() -> dict[str, tuple[int, int]]:
    """Each entry's first and last line on the page, 1-based."""
    lines = PAGE.read_text(encoding="utf-8").splitlines()
    starts = [
        (i + 1, m.group(1))
        for i, line in enumerate(lines)
        if (m := re.match(r"^### `(\w+)`", line))
    ]
    family = [i + 1 for i, line in enumerate(lines) if line.startswith("## ")]
    out = {}
    for (start, name), nxt in zip(starts, [*starts[1:], (len(lines) + 1, "")], strict=True):
        end = min([nxt[0], *(f for f in family if f > start)]) - 1
        out[name] = (start, end)
    return out


@pytest.mark.parametrize("model", MODELS)
def test_each_entry_has_a_recipe(model):
    start, end = _entries()[model]
    recipes = [line for line, _, _ in shown_outputs(PAGE) if start < line <= end]
    assert recipes, f"`{model}` (MODELS.md lines {start}-{end}) shows no recipe with its output"
