"""`docs/REGIMES.md` against the script that measures it.

Every number in REGIMES.md comes from `scripts/regime_experiments.py`. Task
159 re-ran section 8 and updated its table, and the prose beside it kept the
old 0.151 where the table read 0.150 (task 160, SC2). The experiments behind
sections 1 and 5 to 9 take under half a minute each, so they run here:
every body row of their output tables must be a row of the section's
tables, and every three-decimal figure in the section's prose must be one
of the output's. Sections 6 and 9 joined in task 192, after their prose had
drifted from their tables (review 2026-10-06, DB13). Blockquotes are the
dated notes that keep a section's earlier figures on purpose, and are left
out.

macOS only: the document is generated there, and another platform's libm may
move a figure in its third decimal, which an exact comparison would read as a
stale document. The CI's macOS legs run it.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import sys
from pathlib import Path

import pytest

TIER = "extended"

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "REGIMES.md"

#: A number, not the second end of a range (`8-58`) or a decimal's tail.
NUM = re.compile(r"(?<![\d.])-?\d+(?:\.\d+)?")
SEPARATOR = re.compile(r"^\|[-| :]+\|$")

#: Each section and the experiments its tables come from.
SECTIONS = {
    "1": ["recovery", "switch"],
    "5": ["delay"],
    "6": ["epps"],
    "7": ["deco"],
    "8": ["rcov"],
    "9": ["sequential"],
}

pytestmark = [
    pytest.mark.skipif(
        sys.platform != "darwin",
        reason="REGIMES.md is generated on macOS; another libm may move a third decimal",
    ),
    pytest.mark.extended(
        reason="a document's experiments: scripts/regime_experiments.py, 32 s (section 9 23 s)"
    ),
]


def _experiments():
    spec = importlib.util.spec_from_file_location(
        "regime_experiments", REPO / "scripts" / "regime_experiments.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EXPERIMENTS


def _run(name: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _experiments()[name]()
    return buf.getvalue()


def _sections(text: str) -> dict[str, str]:
    parts = re.split(r"(?m)^## (\d+)\. ", text)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def _body_rows(text: str) -> list[list[str]]:
    """Each table body row's numbers, the label cell left out. A header row
    is the one just above a separator, and can hold numbers of its own."""
    lines = text.splitlines()
    rows = []
    for i, line in enumerate(lines):
        if not line.startswith("|") or SEPARATOR.match(line.strip()):
            continue
        if i + 1 < len(lines) and SEPARATOR.match(lines[i + 1].strip()):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        numbers = [n for c in cells[1:] for n in NUM.findall(c)]
        if numbers:
            rows.append(numbers)
    return rows


def _prose(text: str) -> str:
    keep, fenced = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        if not fenced and not line.startswith(("|", ">")):
            keep.append(line)
    return "\n".join(keep)


@pytest.mark.parametrize("section", sorted(SECTIONS))
def test_a_section_reads_what_its_experiments_print(section):
    body = _sections(DOC.read_text(encoding="utf-8").replace("−", "-"))[section]
    printed = "".join(_run(name) for name in SECTIONS[section])
    documented = _body_rows(body)
    # The output may carry trailing columns the document leaves out (rcov's
    # block and row counts), so a documented row is a prefix of the printed.
    for row in _body_rows(printed):
        assert any(len(d) >= 2 and row[: len(d)] == d for d in documented), (
            f"section {section}: printed row {row} is in none of the section's tables"
        )
    figures = {n.lstrip("-") for n in NUM.findall(printed)}
    stale = [
        n
        for n in re.findall(r"(?<![\d.])-?\d+\.\d{3}\b", _prose(body))
        if n.lstrip("-") not in figures
    ]
    assert not stale, f"section {section}: prose figures the experiments do not print: {stale}"
