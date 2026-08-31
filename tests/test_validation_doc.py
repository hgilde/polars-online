"""`docs/VALIDATION.md` must still be what the code produces.

The defaults this library ships -- `solve_every = halflife/50`, `standardize`
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

from data import public_intraday_or_skip

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "VALIDATION.md"

#: Wall-clock timings vary run to run and say nothing about correctness.
_TIMING = re.compile(r"\d+\.\d+s")


def _normalize(text: str) -> list[str]:
    return [_TIMING.sub("<time>s", line) for line in text.strip().splitlines()]


@pytest.fixture(scope="module")
def regenerated():
    public_intraday_or_skip()  # skips offline, and warms the cache
    res = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "validate.py")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(REPO),
        check=False,
    )
    assert res.returncode == 0, res.stderr
    return res.stdout


def test_the_committed_document_is_what_the_code_produces(regenerated):
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
    ]:
        assert heading in regenerated, f"validate.py no longer measures: {heading}"
