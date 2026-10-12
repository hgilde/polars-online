"""The budget file and its rules (``budgets.toml``, ``scripts/budgets.py``).

A budget is a fixed number every push is measured against, so a run of
small regressions adds up against one number instead of each passing
against the last. These hold the file to the repository (the cases the
bench names, the harness the lock resolves, the image the workflow runs
on), the rules to what they say -- over fails, under fails naming the
number to tighten to, a measuring run fails nothing -- and every raise in
the file's history to a ``[[raised]]`` entry with a reason and the
CHANGELOG *Performance* entry it quotes.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

TIER = "essential"
REPO = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # a dataclass looks its module up there
    spec.loader.exec_module(module)
    return module


budgets = _load("budgets")
DOC = budgets.load()
WORKFLOW = yaml.safe_load((REPO / ".github/workflows/budgets.yml").read_text(encoding="utf-8"))
WORKFLOW["on"] = WORKFLOW.pop(True, WORKFLOW.get("on"))

#: A file with every kind, small enough to read in a test.
SMALL = """
[meta]
toolchain = "1.99.0"
runner = "ubuntu-24.04"
gungraun = "0.20.0"
rows = 1000
release = "1.0.0"
drift_signoff = 0.10

[meta.tolerance]
instructions = 0.02
memory = 0.02
rss = 0.15

[instructions.a]
what = "case a"
budget = 1000000
release = 1000000

[instructions.b]
what = "case b"

[memory.ewridge]
pred_y = { budget = 8.125 }
"abs_resid_q0.1_y" = { budget = 8.125, release = 8.0 }

[rss.streaming]
linux = { budget = 100.0 }

# --- raises and sign-offs
"""


def small(extra: str = "") -> dict:
    return tomllib.loads(SMALL + extra)


# --- the file against the repository -------------------------------------------


def test_the_file_is_well_formed_and_every_raise_quotes_the_changelog():
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    assert budgets.check_file(DOC, changelog, budgets.bench_cases()) == []


def test_every_case_the_bench_names_has_an_entry():
    """The bench against the crate's lists is `crates/online-bench`'s own
    test (`the_bench_names_every_case_and_nothing_else`)."""
    cases = budgets.bench_cases()
    assert len(cases) == len(set(cases)) >= 14
    assert set(DOC["instructions"]) == set(cases)


def test_the_pins_are_the_repositorys():
    """The harness the lock resolves, the rows a case counts, the image the
    workflow runs on, the compiler it builds with: each is part of a count,
    so the file names each and these hold it to the repository."""
    meta = DOC["meta"]
    assert meta["gungraun"] == budgets.runner_version()
    lib = (REPO / "crates/online-bench/src/lib.rs").read_text(encoding="utf-8")
    (rows,) = re.findall(r"pub const ROWS: usize = ([\d_]+);", lib)
    assert meta["rows"] == int(rows.replace("_", ""))
    job = WORKFLOW["jobs"]["instructions"]
    assert job["runs-on"] == meta["runner"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", meta["toolchain"])
    runs = " ".join(str(s.get("run", "")) for s in job["steps"])
    # Read from the file, never spelled out in the workflow.
    assert "scripts/budgets.py toolchain" in runs and meta["toolchain"] not in runs
    assert "scripts/budgets.py runner-version" in runs
    cargo = tomllib.loads((REPO / "crates/online-bench/Cargo.toml").read_text(encoding="utf-8"))
    dev = cargo["target"]["cfg(unix)"]["dev-dependencies"]
    assert dev["gungraun"] == f"={meta['gungraun']}"


def test_the_workflow_gates_every_push_and_pull_request_and_a_release_calls_it():
    on = WORKFLOW["on"]
    assert {"push", "pull_request", "workflow_dispatch", "workflow_call"} <= set(on)
    assert on["workflow_dispatch"]["inputs"]["measure"]["default"] is False
    job = WORKFLOW["jobs"]["instructions"]
    steps = job["steps"]
    assert steps[0]["with"]["fetch-depth"] == 0, "the raise rule reads the whole history"
    runs = [str(s.get("run", "")) for s in steps]
    assert any("budgets.py raises" in r for r in runs)
    (count,) = [r for r in runs if "cargo bench" in r]
    assert "for run in first second" in count and "--output-format=json" in count
    (check,) = [r for r in runs if "budgets.py check" in r]
    assert "first.json second.json" in check
    assert not any(s.get("continue-on-error") for s in steps)
    release = yaml.safe_load((REPO / ".github/workflows/release.yml").read_text(encoding="utf-8"))
    call = release["jobs"]["budgets"]
    assert call["uses"] == "./.github/workflows/budgets.yml"
    assert call["with"]["release-check"] == "${{ needs.version.outputs.version }}"
    assert "budgets" in release["jobs"]["publish"]["needs"]


def test_this_files_history_explains_every_raise():
    """The rule itself, over this checkout's history of the file (one commit
    in CI's shallow clone; the budgets workflow fetches it all)."""
    versions, _ = budgets.history()
    working = (REPO / "budgets.toml").read_text(encoding="utf-8")
    assert budgets.check_history(versions, working) == []


# --- the rules ---------------------------------------------------------------------


def test_a_measurement_past_its_budget_fails_and_says_how_to_raise_it():
    v = budgets.judge("instructions.a", 1_030_000, {"budget": 1_000_000}, 0.02)
    assert v.status == "over" and v.failed
    assert "+3.00% over its budget of 1,000,000" in v.message
    assert "[[raised]]" in v.message and "CHANGELOG" in v.message


def test_a_measurement_under_its_budget_fails_naming_the_number_to_tighten_to():
    v = budgets.judge("instructions.a", 970_000, {"budget": 1_000_000}, 0.02)
    assert v.status == "under" and v.failed
    assert "tighten the budget to 970000" in v.message
    m = budgets.judge("memory.ewridge.pred_y", 8.00001, {"budget": 8.5}, 0.02)
    # Rounded up, so the new budget is never below what it holds.
    assert "tighten the budget to 8.0001" in m.message


@pytest.mark.parametrize("measured", [980_000, 1_000_000, 1_020_000])
def test_within_the_tolerance_either_way_passes(measured):
    v = budgets.judge("instructions.a", measured, {"budget": 1_000_000}, 0.02)
    assert v.status == "ok" and not v.failed


def test_a_case_with_no_budget_is_measured_and_fails_nothing():
    v = budgets.judge("instructions.b", 123_456, {"what": "b"}, 0.02)
    assert v.status == "unmeasured" and not v.failed
    assert "set budget = 123456" in v.message


def test_a_measuring_run_fails_nothing_and_reports_everything():
    doc = small()
    verdicts = budgets.judge_all(
        doc, {"instructions.a": 2_000_000, "memory.ewridge.pred_y": 1.0}, measure=True
    )
    assert not any(v.failed for v in verdicts)
    assert all(v.message.startswith("measuring:") for v in verdicts)
    table = budgets.report(doc, verdicts)
    assert "instructions.a" in table and "memory.ewridge.pred_y" in table


def test_each_kind_has_its_own_tolerance():
    doc = small()
    rss = budgets.judge_all(doc, {"rss.streaming.linux": 110.0})
    assert rss[0].status == "ok"  # 10% within the 15% of resident memory
    mem = budgets.judge_all(doc, {"memory.ewridge.pred_y": 8.125 * 1.1})
    assert mem[0].status == "over"  # 10% past the 2% of bytes


def test_drift_since_the_release_past_the_signoff_needs_a_signoff():
    doc = small()
    verdicts = budgets.judge_all(doc, {"instructions.a": 1_015_000, "memory.ewridge.pred_y": 8.0})
    assert budgets.signoff_failures(doc, verdicts) == []
    # A case that crept 2% a change for six changes, each within its budget
    # once raised: 12% past the release.
    doc = small()
    doc["instructions"]["a"]["budget"] = 1_120_000
    verdicts = budgets.judge_all(doc, {"instructions.a": 1_120_000})
    (failure,) = budgets.signoff_failures(doc, verdicts, "1.1.0")
    assert "+12.0% since 1.0.0" in failure
    signed = small(
        '[[signoff]]\nkey = "instructions.a"\nversion = "1.1.0"\nreason = "the new prior"\n'
    )
    signed["instructions"]["a"]["budget"] = 1_120_000
    assert budgets.signoff_failures(signed, verdicts, "1.1.0") == []
    # A sign-off is for one release.
    assert budgets.signoff_failures(signed, verdicts, "1.2.0") == [failure]


def test_two_runs_of_one_binary_must_agree_well_inside_the_tolerance():
    doc = small()
    a = {"instructions.a": 1_000_000.0}
    assert budgets.repeat_failures(doc, a, {"instructions.a": 1_004_000.0}) == []
    (bad,) = budgets.repeat_failures(doc, a, {"instructions.a": 1_006_000.0})
    assert "differ by 0.600%" in bad
    assert budgets.repeat_failures(doc, a, {}) == ["instructions.a: in one run only"]


# --- the raise rule ----------------------------------------------------------------

RAISE = """
[[raised]]
key = "instructions.a"
from = 1000000
to = 1100000
date = 2026-10-12
reason = "the exact clock on every row costs this"
changelog = "exact clock stamps"
"""


def _raised(budget: int, extra: str = "") -> str:
    return SMALL.replace("budget = 1000000", f"budget = {budget}") + extra


def test_a_raise_without_an_entry_is_refused():
    (error,) = budgets.check_history([("a" * 40, SMALL)], _raised(1_100_000))
    assert "instructions.a: 1000000 -> 1100000 with no [[raised]] entry" in error


def test_a_raise_with_its_entry_passes_and_a_tightening_needs_none():
    assert budgets.check_history([("a" * 40, SMALL)], _raised(1_100_000, RAISE)) == []
    assert budgets.check_history([("a" * 40, SMALL)], _raised(900_000)) == []


def test_an_entry_must_name_the_numbers_of_the_raise():
    other = RAISE.replace("to = 1100000", "to = 1050000")
    (error,) = budgets.check_history([("a" * 40, SMALL)], _raised(1_100_000, other))
    assert "1000000 -> 1100000" in error


def test_a_raise_committed_without_its_entry_is_answered_by_a_later_one_and_deleting_it_fails():
    """Entries accumulate: a raise is answered by an entry in the file as it
    stands, so a commit that forgot one is repaired by the next, and
    deleting the entry brings the raise back."""
    versions = [("a" * 40, SMALL), ("b" * 40, _raised(1_100_000))]
    assert budgets.check_history(versions, _raised(1_100_000, RAISE)) == []
    assert len(budgets.check_history(versions, _raised(1_100_000))) == 1


def test_a_wider_tolerance_is_a_raise():
    wider = SMALL.replace("instructions = 0.02", "instructions = 0.05")
    (error,) = budgets.check_history([("a" * 40, SMALL)], wider)
    assert "tolerance.instructions: 0.02 -> 0.05" in error


def test_an_entry_must_quote_a_performance_entry_of_the_changelog():
    changelog = (
        "## [Unreleased]\n\n### Fixed\n\n- exact clock stamps, a fix\n\n"
        "### Performance\n\n- **Exact clock\n  stamps on every row** cost 3%.\n\n## [0.13.0]\n"
    )
    doc = small(RAISE)
    doc["instructions"]["a"]["budget"] = 1_100_000
    assert budgets.check_file(doc, changelog, ["a", "b"]) == [], "quoted across a line break"
    elsewhere = changelog.replace("### Performance", "### Changed")
    (error,) = budgets.check_file(doc, elsewhere, ["a", "b"])
    assert "no *Performance* entry containing 'exact clock stamps'" in error


def test_an_entry_says_why_and_when():
    doc = small(RAISE.replace("the exact clock on every row costs this", "slower"))
    doc["raised"][0]["date"] = "2026-10-12"
    errors = budgets.check_file(doc, "### Performance\n\nexact clock stamps\n", ["a", "b"])
    assert any("the reason says too little" in e for e in errors), errors
    assert any("not a TOML date" in e for e in errors), errors
    assert isinstance(small(RAISE)["raised"][0]["date"], dt.date)


def test_the_history_is_read_from_git(tmp_path):
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    for budget in (1_000_000, 1_100_000):
        (tmp_path / "budgets.toml").write_text(_raised(budget), encoding="utf-8")
        git("add", "budgets.toml")
        git("commit", "-q", "-m", str(budget))
    versions, shallow = budgets.history(repo=tmp_path)
    assert not shallow and len(versions) == 2
    (error,) = budgets.check_history(versions, None)
    assert "1000000 -> 1100000" in error


# --- reading the harness, and writing the file ----------------------------------------


def _summary(case: str, ir: int, old: int | None = None) -> dict:
    """A summary as Gungraun 0.20 writes it (`gungraun_runner::summary::model`,
    schema 7): the Callgrind profile's total `Ir`, new and, after a first
    run, old."""
    values = {"new": ir} if old is None else {"new": ir, "old": old}
    return {
        "version": "7",
        "kind": "LibraryBenchmark",
        "id": case,
        "function_name": "model",
        "module_path": "instructions::hot_paths::model",
        "profiles": [
            {
                "tool": "Callgrind",
                "data": {
                    "parts": [],
                    "total": {
                        "metrics": {"Ir": {"values": values}, "L1hits": {"values": {"new": 1}}},
                        "regressions": [],
                    },
                },
            }
        ],
    }


def test_gungrauns_json_is_read_one_count_per_case():
    lines = "\n".join(json.dumps(_summary(c, n)) for c, n in (("a", 1000), ("b", 2000)))
    assert budgets.parse_gungraun(lines) == {"a": 1000, "b": 2000}
    pretty = json.dumps(_summary("a", 7, old=6), indent=2) + "\n" + json.dumps(_summary("b", 8))
    assert budgets.parse_gungraun(pretty) == {"a": 7, "b": 8}
    with pytest.raises(ValueError, match="two summaries"):
        budgets.parse_gungraun(lines + "\n" + json.dumps(_summary("a", 1)))
    none = _summary("a", 1)
    none["profiles"] = []
    with pytest.raises(ValueError, match="0 Callgrind profiles"):
        budgets.parse_gungraun(json.dumps(none))


def test_apply_writes_new_and_tightened_budgets_and_never_raises_one():
    doc = small()
    verdicts = budgets.judge_all(
        doc,
        {
            "instructions.a": 1_500_000,  # over: left for a stated raise
            "instructions.b": 42_000,  # new
            "memory.ewridge.pred_y": 7.0,  # under
            "memory.ewridge.abs_resid_q0.1_y": 8.125,  # within
            "memory.lasso.coef": 198.205,  # a new spec
            "rss.streaming.darwin": 300.5,  # a new platform
        },
    )
    text, done = budgets.apply(SMALL, verdicts)
    after = tomllib.loads(text)
    assert after["instructions"]["a"]["budget"] == 1_000_000
    assert after["instructions"]["b"]["budget"] == 42_000
    assert after["memory"]["ewridge"]["pred_y"]["budget"] == 7.0
    assert after["memory"]["ewridge"]["abs_resid_q0.1_y"] == {"budget": 8.125, "release": 8.0}
    assert after["memory"]["lasso"]["coef"]["budget"] == 198.205
    assert after["rss"]["streaming"] == {"linux": {"budget": 100.0}, "darwin": {"budget": 300.5}}
    assert len(done) == 4
    # New tables go before the raises, which close the file.
    assert text.index("[memory.lasso]") < text.index(budgets.RAISES_MARK)
    assert budgets.check_history([("a" * 40, SMALL)], text) == []


def test_apply_records_a_releases_counts():
    doc = small()
    verdicts = budgets.judge_all(doc, {"instructions.a": 990_000, "instructions.b": 5})
    text, _ = budgets.apply(SMALL, verdicts, release="1.1.0")
    after = tomllib.loads(text)
    assert after["meta"]["release"] == "1.1.0"
    assert after["instructions"]["a"] == {"what": "case a", "budget": 1_000_000, "release": 990_000}
    assert after["instructions"]["b"]["release"] == 5


def test_check_end_to_end(tmp_path, monkeypatch, capsys):
    """The command the workflow runs: a count under its budget fails,
    naming the number; a measuring run of the same counts passes; the job
    summary gets the table either way."""
    path = tmp_path / "budgets.toml"
    path.write_text(SMALL, encoding="utf-8")
    monkeypatch.setattr(budgets, "BUDGETS", path)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.delenv(budgets.MEASURE_VARIABLE, raising=False)
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    for f in (first, second):
        f.write_text("\n".join(json.dumps(_summary(c, n)) for c, n in (("a", 900_000), ("b", 7))))
    measured = tmp_path / "measured.json"
    assert budgets.main(["check", str(first), str(second), "--measured", str(measured)]) == 1
    out = capsys.readouterr().out
    assert "::error::instructions.a: 900,000 is -10.00% under its budget" in out
    assert "tighten the budget to 900000" in out
    assert "::warning::instructions.b: no budget yet" in out
    assert json.loads(measured.read_text()) == {"instructions.a": 900_000, "instructions.b": 7}
    assert "| instructions.a | 1,000,000 | 900,000 | -10.00% | -10.00% |" in summary.read_text()
    assert budgets.main(["check", str(first), "--measure"]) == 0
    assert budgets.main(["apply", str(measured)]) == 0
    assert tomllib.loads(path.read_text())["instructions"]["a"]["budget"] == 900_000
    assert budgets.main(["check", str(first), str(second)]) == 0


def test_measuring_is_asked_for_by_name(monkeypatch):
    monkeypatch.setenv(budgets.MEASURE_VARIABLE, "measure")
    assert budgets.measuring()
    monkeypatch.setenv(budgets.MEASURE_VARIABLE, "1")
    assert not budgets.measuring()
