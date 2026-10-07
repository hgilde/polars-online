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

import child

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "OUTPUTS.md"
SCRIPT = REPO / "scripts" / "outputs_doc.py"


def test_the_document_is_what_the_generator_writes():
    got = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        cwd=REPO,
        env=child.env(),
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


def _table(section: str) -> tuple[str, list[list[str]]]:
    """The header and the rows of the first field table in ``section``."""
    lines = section.splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith("| field |"))
    rows = []
    for line in lines[at + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in line.strip().strip("|").split(" | ")])
    return lines[at], rows


def test_every_field_gives_its_dtype_and_where_else_it_is_null():
    """Review 2026-10-06 (DA14): the page gave no field a dtype, and said when
    a field is null for a few only. Each table gives the dtype
    ``po.spec.output_index`` declares, and fills the *also null* column for
    every field, so no reader has to find out where a field is null by
    running it."""
    import polars_online as po

    sys.path.insert(0, str(REPO / "tests"))
    from test_model_registry import MINIMAL, _build

    text = DOC.read_text(encoding="utf-8")
    sections = {
        name: (text.split(f"\n## `{name}`\n", 1)[1].split("\n## ", 1)[0], _build(name))
        for name in MINIMAL
    }
    shared = text.split("\n### Fields most models write\n", 1)[1].split("\n### ", 1)[0]
    sections["the shared fields"] = (shared, _build("ewridge"))
    for name, (section, spec) in sections.items():
        header, rows = _table(section)
        assert header == "| field | dtype | what it holds | also null |", (name, header)
        declared = po.spec.output_index(spec)
        dtypes = dict(zip(declared["field"], declared["dtype"], strict=True))
        for row in rows:
            assert len(row) == 4 and all(row), (name, row)
            field, dtype = row[0].strip("`"), row[1].strip("`")
            assert dtypes[field] == dtype, (name, field, dtype, dtypes[field])
