"""`scripts/mutants_report.py` and `scripts/mutants_equivalent.toml`: the
report the mutation-testing jobs print, and the mutants it does not count
(docs/TESTING.md T-D4)."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
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


def _listed(run: Path, n: int) -> None:
    """The run's list of the mutants it was given, as the jobs write it."""
    lines = "".join(f"src/m.rs:{i}:9: replace + with - in f\n" for i in range(n))
    (run / "listed.txt").write_text(lines, encoding="utf-8")


def test_a_run_that_stopped_early_is_incomplete(tmp_path):
    """A shard stopped at its time limit tests fewer mutants than it was
    given, and its outcomes stop where it stopped. The report counted what
    it found, so a shard cut short read as a smaller clean one (task 155)."""
    run = _run(tmp_path, [("", 2, "replace > with >=")])
    _listed(run, 5)
    assert report_mod.incomplete([run]) == ["mutants.out: tested 2 of the 5 mutants it was given"]
    text, _ = report_mod.report([run], root=tmp_path, equivalents=[])
    assert "### Incomplete" in text and "tested 2 of the 5" in text


def test_a_run_that_sent_nothing_is_missing(tmp_path):
    """A shard killed outright uploads nothing: the report knows how many
    runs to expect, and a path that is not a directory is not one."""
    run = _run(tmp_path, [])
    _listed(run, 1)
    assert report_mod.incomplete([run, tmp_path / "absent"], expect_runs=3) == [
        "2 of the 3 runs reported nothing"
    ]


def test_completeness_and_survivors_set_the_exit_apart(tmp_path):
    """The weekly report fails on an incomplete pass, never on a survivor;
    the job on a change's lines fails on both."""
    run = _run(tmp_path, [("", 2, "replace < with <=")])
    _listed(run, 2)
    assert report_mod.incomplete([run], expect_runs=1) == []
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_STEP_SUMMARY"}

    def exit_code(*flags: str) -> int:
        script = str(REPO / "scripts/mutants_report.py")
        cmd = [sys.executable, script, str(run), *flags]
        return subprocess.run(cmd, capture_output=True, env=env, check=False).returncode

    assert exit_code("--expect-runs", "1", "--fail-on-incomplete") == 0
    assert exit_code("--fail-on-missed") == 1
    _listed(run, 3)
    assert exit_code("--fail-on-incomplete") == 1
    assert exit_code() == 0
