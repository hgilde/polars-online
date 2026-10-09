"""``docs/DATA-ISSUES.md`` (docs/PLAN.md task 224): the page each finding of
``ModelBank.check`` links to, held to the code.

Every recipe on the page runs here, and prints what the page shows under it,
so a recipe and the prose that reads its output cannot drift from the library
(docs/WRITING.md §3, "Every code block runs"). Every code ``check`` can give
has a section anchored by its name, every finding's message ends with the
link to it, and the docstring lists the same codes.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import pytest

import check_streams as cs
import polars_online as po
from doc_recipes import differences, recipes, run
from polars_online import _check

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "DATA-ISSUES.md"


RECIPES = recipes(PAGE)


def test_the_page_has_its_recipes():
    """One recipe for the opening example and one for each problem."""
    assert len(RECIPES) >= 12, len(RECIPES)


@pytest.mark.parametrize(
    ("line", "code", "shown"), RECIPES, ids=[f"DATA-ISSUES.md:L{line}" for line, _, _ in RECIPES]
)
def test_a_recipe_prints_what_the_page_shows(line, code, shown, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a recipe writes its parquet file where it runs
    problems = differences(shown, run(code, PAGE))
    assert not problems, f"DATA-ISSUES.md:{line}:\n" + "\n".join(problems)


def test_the_comparison_holds_words_digits_and_blow_ups():
    assert not differences("R² 0.578", "R² 0.579")
    assert differences("R² 0.578", "R² 0.581")
    assert differences("R² 0.578", "R2 0.578")
    assert not differences("x1 0.0464", "x1 0.0465")
    assert not differences("slope 5.4e+13", "slope 9.1e+12")
    assert differences("slope 5.4e+13", "slope -5.4e+13")
    assert differences("slope 5.4e+13", "slope 0.481")
    assert differences("rows 217", "rows 220")
    assert differences("a\nb", "a")


def _anchors() -> set[str]:
    text = PAGE.read_text(encoding="utf-8")
    return set(re.findall(r'<a id="([^"]+)"></a>', text)) | {
        re.sub(r"[^\w\- ]", "", h).strip().lower().replace(" ", "-")
        for h in re.findall(r"^#+ (.+)$", text, re.M)
    }


def test_every_code_has_a_section_on_the_page():
    """A finding's message links to ``DATA-ISSUES.md#<code>``, so each code
    is an anchor there: an explicit one, or a heading whose text is the code."""
    missing = sorted(_check.CODES - _anchors())
    assert not missing, f"codes with no anchor on docs/DATA-ISSUES.md: {missing}"


def test_the_docstring_lists_the_same_codes():
    doc = po.ModelBank.check.__doc__ or ""
    listed = set(re.findall(r"\* - ``(\w+)``\n\s+- (?:error|warning|info)", doc))
    assert listed == _check.CODES, listed ^ _check.CODES


def _every_planted_finding() -> pl.DataFrame:
    frames = []
    for make, _, _, _ in cs.PLANTED.values():
        frames.append(cs.findings(*make(0)))
    for make, _, _ in cs.AUDIT_PLANTED.values():
        frames.append(cs.audit_findings(make(0)))
    return pl.concat(frames)


def test_every_message_ends_with_the_link_to_its_code():
    found = _every_planted_finding()
    assert len(set(found["code"])) >= 20, set(found["code"])
    assert set(found["code"]) <= _check.CODES, set(found["code"]) - _check.CODES
    for code, message in found.select("code", "message").iter_rows():
        assert message.endswith(f". See {_check.DATA_ISSUES}#{code}"), message


def test_the_link_is_the_page_on_main():
    assert _check.DATA_ISSUES.endswith("/blob/main/docs/DATA-ISSUES.md")
    assert PAGE.is_file()


def test_the_readme_links_the_page_beside_check_and_audit():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for anchor in ("**`check()` reads these tables", "#### `audit` — what a stream's columns hold"):
        start = readme.index(anchor)
        section = readme[start : readme.index("\n#", start + len(anchor))]
        assert "docs/DATA-ISSUES.md" in section, anchor
