"""`docs/OUTPUTS.md` must still be what the models write.

The outputs *are* the product of a model, and until task 64 the only record
of them per model was a test fixture. The document is generated from
`po.spec.output_fields` on a canonical spec, with the meanings written in
`scripts/outputs_doc.py`; this regenerates it and fails on any difference,
so a new field cannot ship undocumented and a removed one cannot linger.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "OUTPUTS.md"
SCRIPT = REPO / "scripts" / "outputs_doc.py"


def test_the_document_is_what_the_generator_writes():
    got = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True, cwd=REPO
    ).stdout
    want = DOC.read_text(encoding="utf-8")
    if got != want:
        first = next(
            (
                i
                for i, (a, b) in enumerate(zip(got.splitlines(), want.splitlines(), strict=False))
                if a != b
            ),
            min(len(got.splitlines()), len(want.splitlines())),
        )
        raise AssertionError(
            "docs/OUTPUTS.md is stale. Regenerate it with\n"
            "    uv run python scripts/outputs_doc.py > docs/OUTPUTS.md\n"
            f"first difference at line {first + 1}:\n"
            f"  committed: {(want.splitlines() + [''])[first]!r}\n"
            f"  generated: {(got.splitlines() + [''])[first]!r}"
        )


def test_every_model_has_a_section():
    sys.path.insert(0, str(REPO / "tests"))
    from test_model_registry import MINIMAL

    text = DOC.read_text(encoding="utf-8")
    for name in MINIMAL:
        assert f"\n## `{name}`\n" in text, f"docs/OUTPUTS.md has no section for {name}"


def test_no_field_is_left_undocumented():
    """The generator marks an unmapped field rather than omitting it, so the
    document itself is the check."""
    assert "**undocumented**" not in DOC.read_text(encoding="utf-8")
