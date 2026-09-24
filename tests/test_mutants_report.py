"""`scripts/mutants_report.py` and `scripts/mutants_equivalent.toml`: the
report the mutation-testing jobs print, and the mutants it does not count
(docs/TESTING.md T-D4)."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("mutants_report", REPO / "scripts/mutants_report.py")
assert _spec and _spec.loader
report_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report_mod)


def test_every_equivalent_still_finds_its_line():
    """An entry lapses when its line changes, and this says so: a lapsed
    entry would let a real survivor on a rewritten line be reported, which
    is the safe side, but the list should not carry dead entries."""
    entries = report_mod.load_equivalents()
    assert entries, "the list is empty"
    for e in entries:
        assert set(e) == {"file", "function", "mutation", "line_has", "why"}, e
        assert re.fullmatch(r"replace .+ with .+", e["mutation"]), e["mutation"]
        source = (REPO / e["file"]).read_text(encoding="utf-8")
        assert e["line_has"] in source, f"{e['file']}: no line has {e['line_has']!r}"
        assert f"fn {e['function'].rsplit('::', 1)[-1]}(" in source, e["function"]


def _run(tmp_path: Path, missed: list[tuple[str, int, str]]) -> Path:
    """A mutants.out with a caught mutant and the given missed ones."""

    def outcome(line: int, what: str, summary: str) -> dict:
        return {
            "scenario": {
                "Mutant": {
                    "name": f"src/m.rs:{line}:9: {what} in f",
                    "file": "src/m.rs",
                    "function": {"function_name": "f"},
                    "span": {"start": {"line": line, "column": 9}},
                }
            },
            "summary": summary,
        }

    run = tmp_path / "mutants.out"
    run.mkdir()
    outcomes = [{"scenario": "Baseline", "summary": "Success"}]
    outcomes += [outcome(1, "replace + with -", "CaughtMutant")]
    outcomes += [outcome(line, what, "MissedMutant") for _, line, what in missed]
    (run / "outcomes.json").write_text(json.dumps({"outcomes": outcomes}), encoding="utf-8")
    return run


EQUIVALENT = {
    "file": "src/m.rs",
    "function": "f",
    "mutation": "replace > with >=",
    "line_has": "if w > 0.0 {",
    "why": "w is positive here",
}


def test_an_equivalent_is_not_counted_and_follows_its_line(tmp_path):
    (tmp_path / "src").mkdir()
    # The line the entry names has moved from 2 to 4, and a second missed
    # mutant sits on a line no entry names.
    (tmp_path / "src/m.rs").write_text("a\nb\nc\n    if w > 0.0 {\nlet x = y < z;\n")
    run = _run(tmp_path, [("", 4, "replace > with >="), ("", 5, "replace < with <=")])
    text, missed = report_mod.report([run], root=tmp_path, equivalents=[EQUIVALENT])
    assert missed == 1
    assert "3 mutants: 1 caught, 1 survived, 1 equivalent" in text
    assert "src/m.rs:5:9" in text.split("### Equivalent")[0]
    assert "w is positive here" in text


def test_an_equivalent_lapses_when_its_line_changes(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/m.rs").write_text("a\n    if w >= 1.0 {\n")
    run = _run(tmp_path, [("", 2, "replace > with >=")])
    _, missed = report_mod.report([run], root=tmp_path, equivalents=[EQUIVALENT])
    assert missed == 1


def test_a_change_with_no_mutable_code_is_reported_as_such(tmp_path):
    empty = tmp_path / "mutants.out"
    empty.mkdir()
    text, missed = report_mod.report([empty], root=tmp_path, equivalents=[])
    assert missed == 0 and "No mutants" in text
