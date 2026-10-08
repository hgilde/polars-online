"""`docs/VALIDATION.md` must still be what the code produces.

The defaults this library ships -- the solve cadence, `standardize`
per model, the elastic-net ratio, Kalman's `share_p` -- were chosen from the
measurements in that document. If the code moves and the document does not,
the defaults are justified by numbers that are no longer true, and nothing
would say so: it is generated once and committed.

Cheap enough to check every run (the script takes ~0.4s against the cached
data). Skips when the public dataset cannot be fetched, like the other tests
that use it.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

import child
from data import VALIDATION_DATES, public_intraday_or_skip

TIER = "extended"
pytestmark = pytest.mark.extended(
    reason="a document's experiments: scripts/validate.py over the downloaded intraday days"
)

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "VALIDATION.md"

#: Wall-clock timings vary run to run and say nothing about correctness.
_TIMING = re.compile(r"\d+\.\d+s")


def _normalize(text: str) -> list[str]:
    return [_TIMING.sub("<time>s", line) for line in text.strip().splitlines()]


@pytest.fixture(scope="module")
def regenerated():
    # Skips offline, and warms the cache for every day the script reads, so
    # a download the retries could not save is a skip here rather than a
    # crash in the subprocess (or its silent fallback to synthetic data).
    public_intraday_or_skip(VALIDATION_DATES)
    res = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "validate.py")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=child.env(),
        cwd=str(REPO),
        check=False,
    )
    assert res.returncode == 0, res.stderr
    return res.stdout


@pytest.mark.pins
def test_the_committed_document_is_what_the_code_produces(regenerated):
    """Marked `pins` because the document's header records the polars version
    it was generated with, so it can only match under the pinned one. The
    canary deselects it: an unpinned polars would fail it on that line alone,
    and a canary that cries wolf is worse than none (`polars-canary.yml`)."""
    want = _normalize(DOC.read_text(encoding="utf-8"))
    got = _normalize(regenerated)
    if want == got:
        return
    # Report the first divergence rather than a wall of diff.
    for i, (a, b) in enumerate(zip(want, got, strict=False)):
        if a != b:
            pytest.fail(
                f"docs/VALIDATION.md is stale at line {i + 1}.\n"
                f"  committed: {a}\n"
                f"  produced:  {b}\n"
                "Regenerate with: uv run python scripts/validate.py > docs/VALIDATION.md"
            )
    pytest.fail(
        f"docs/VALIDATION.md has {len(want)} lines, the script produced {len(got)}. "
        "Regenerate with: uv run python scripts/validate.py > docs/VALIDATION.md"
    )


def test_it_still_measures_the_defaults_it_claims_to(regenerated):
    """A guard on the guard: if an experiment is dropped from the script, the
    comparison above would pass on a document that no longer justifies
    anything."""
    for heading in [
        "Solve schedule",
        "`standardize` default",
        "Elastic net",
        "Kalman `share_p`",
        "Models at matched settings",
        # Task 212: the protocol every section runs under, and the recipe
        # the README cites the numbers of.
        "How the runs are set",
        "Features as z-scores",
    ]:
        assert heading in regenerated, f"validate.py no longer measures: {heading}"


def _table(text: str, after: str) -> list[list[str]]:
    """The cells of the first markdown table after the line holding
    ``after``, its header and rule dropped."""
    lines = text[text.index(after) :].splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("|"))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def test_the_readme_quotes_section_6_as_it_stands():
    """The README's *Features in units of their spread* quotes section 6's
    R² for four models, each target, raw and z-scored (task 212). Read from
    the committed document, so a regeneration that moves a number fails here
    until the README is brought along: the README rounds to four places."""
    rows = _table(DOC.read_text(encoding="utf-8"), "## 6. Features as z-scores")
    r2 = {(spec, target, features): float(r) for spec, target, features, _, r, *_ in rows}
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    quoted = _table(readme, "**On minute returns, every model lost less with z-scored features.**")
    names = {
        "`ewridge`, at its default solve cadence": "ewridge",
        "`ewridge`, solved every row": "ewridge_every_row",
        "`rls`": "rls",
        "`kalman`": "kalman",
    }
    assert [row[0] for row in quoted] == list(names)
    columns = [("y0", "raw"), ("y0", "z-scores"), ("y1", "raw"), ("y1", "z-scores")]
    for model, *cells in quoted:
        for (target, features), cell in zip(columns, cells, strict=True):
            want = round(r2[(names[model], target, features)], 4)
            assert float(cell) == want, (model, target, features, cell, want)
