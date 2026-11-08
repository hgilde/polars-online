"""``docs/STATE-WORKFLOW.md``: the saved state's reference, held to the code.

The page shows its guarantees by recipes: each a complete program that
generates its data, saves it to parquet and reads it back, followed by a
```text block of what it prints. Each runs here alone, in an empty
namespace and a directory of its own, and what it prints is held to the page
by ``tests/doc_recipes.py``'s comparison: the same lines and words, and each
number within two units of its last printed digit (docs/WRITING.md §3,
"Every code block runs"). The dated research that decided the design moved
to ``docs/records/STATE-WORKFLOW-RESEARCH.md`` (task 338, stage 10). What the
page's four-step and overlap recipes showed is shown by the tutorial, the
how-tos and the explanation that hold those tasks, whose own tests run them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_recipes import differences, fenced_blocks, run, shown_outputs

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "STATE-WORKFLOW.md"

#: The page's recipes: each python block, with what it prints.
RECIPES = shown_outputs(PAGE)


def test_every_python_block_of_the_page_is_a_recipe():
    """What a saved bank carries, and a query's writes: each block shows
    what it prints."""
    blocks = [line for kind, line, _ in fenced_blocks(PAGE) if kind == "python"]
    assert blocks == [line for line, _, _ in RECIPES], blocks
    assert len(RECIPES) >= 2, len(RECIPES)


@pytest.mark.parametrize(
    ("line", "code", "shown"),
    [pytest.param(*r, id=f"STATE-WORKFLOW.md:L{r[0]}") for r in RECIPES],
)
def test_a_recipe_prints_what_the_page_shows(line, code, shown, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a recipe writes its parquet and state files where it runs
    problems = differences(shown, run(code, PAGE))
    assert not problems, f"STATE-WORKFLOW.md:{line}:\n" + "\n".join(problems)


def test_the_windows_state_version_is_the_one_the_code_writes():
    """The guide and VERSIONS.md each name the window run's state
    version, and both had stayed at 7 while the code wrote 9 (review round 9,
    K-3): a reader checking a saved state's header against the guide would
    call a 9 foreign. Both are read against the constant the code writes."""
    import re

    source = (REPO / "crates/online-polars/src/windows_frame.rs").read_text(encoding="utf-8")
    (written,) = re.findall(r"const WINDOWS_VERSION: u32 = (\d+);", source)
    guide = re.findall(r"The windows state is version (\d+)", PAGE.read_text(encoding="utf-8"))
    record = re.findall(
        r"The window run's state reached version (\d+)",
        (REPO / "docs/VERSIONS.md").read_text(encoding="utf-8"),
    )
    assert guide == [written] and record == [written], (written, guide, record)
