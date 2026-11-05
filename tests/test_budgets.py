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
drift_signoff = 0.10

[meta.tolerance]
instructions = 0.02
memory = 0.02
rss = 0.15

[instructions.a]
what = "case a"
budget = 1000000

[instructions.b]
what = "case b"

[memory.ewridge]
pred_y = { budget = 8.125 }
"abs_resid_q0.1_y" = { budget = 8.125 }

[rss.streaming]
linux = { budget = 100.0 }

# --- raises and sign-offs
"""


def small(extra: str = "") -> dict:
    return tomllib.loads(SMALL + extra)


SHA = "c36c03e32538cd02f57db783282f65fb4c0802fe"


def baseline(version: str = "1.0.0", **measured: float) -> object:
    """The last release's measurements: ``instructions.a`` at 1,000,000 and
    ``memory.ewridge.abs_resid_q0.1_y`` at 8.0 unless given."""
    values = {"instructions.a": 1_000_000.0, "memory.ewridge.abs_resid_q0.1_y": 8.0}
    values.update({k.replace("__", "."): v for k, v in measured.items()})
    return budgets.Baseline(version, SHA, values)


# --- the file against the repository -------------------------------------------


def test_the_file_is_well_formed_and_every_raise_quotes_the_changelog():
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    assert budgets.check_file(DOC, changelog, budgets.bench_cases(), budgets.scale_cells()) == []


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
    then = baseline()
    measured = {"instructions.a": 1_015_000, "memory.ewridge.pred_y": 8.0}
    verdicts = budgets.judge_all(doc, measured, baseline=then)
    assert budgets.signoff_failures(doc, verdicts, released=then.version) == []
    # A case that crept 2% a change for six changes, each within its budget
    # once raised: 12% past the release.
    doc = small()
    doc["instructions"]["a"]["budget"] = 1_120_000
    verdicts = budgets.judge_all(doc, {"instructions.a": 1_120_000}, baseline=then)
    (failure,) = budgets.signoff_failures(doc, verdicts, "1.1.0", then.version)
    assert "+12.0% since 1.0.0" in failure
    signed = small(
        '[[signoff]]\nkey = "instructions.a"\nversion = "1.1.0"\nreason = "the new prior"\n'
    )
    signed["instructions"]["a"]["budget"] = 1_120_000
    assert budgets.signoff_failures(signed, verdicts, "1.1.0", then.version) == []
    # A sign-off is for one release.
    assert budgets.signoff_failures(signed, verdicts, "1.2.0", then.version) == [failure]
    # With no baseline there is no drift, and nothing to sign off.
    verdicts = budgets.judge_all(doc, {"instructions.a": 1_120_000})
    assert verdicts[0].drift is None
    assert budgets.signoff_failures(doc, verdicts, "1.1.0") == []


def test_on_a_push_only_a_signoff_for_a_coming_release_counts():
    """On a push the drift since the release is a warning, silenced by a
    sign-off -- once by one of any version, a past release's too (review 8,
    E-6). Only a sign-off above the baseline's release speaks for the
    release to come."""

    def signed(version: str) -> dict:
        doc = small(
            f'[[signoff]]\nkey = "instructions.a"\nversion = "{version}"\nreason = "the prior"\n'
        )
        doc["instructions"]["a"]["budget"] = 1_120_000
        return doc

    then = baseline()
    verdicts = budgets.judge_all(signed("1.1.0"), {"instructions.a": 1_120_000}, baseline=then)
    assert budgets.signoff_failures(signed("1.1.0"), verdicts, released="1.0.0") == []
    assert budgets.signoff_failures(signed("1.0.10"), verdicts, released="1.0.0") == []
    for stale in ("0.9.0", "1.0.0"):  # the baseline's release is 1.0.0
        (failure,) = budgets.signoff_failures(signed(stale), verdicts, released="1.0.0")
        assert "+12.0% since 1.0.0" in failure


def test_a_signoff_names_a_budget():
    changelog = "### Performance\n\nexact clock stamps\n"
    entry = '[[signoff]]\nkey = "{}"\nversion = "1.1.0"\nreason = "the prior"\n'
    assert budgets.check_file(small(entry.format("instructions.a")), changelog, ["a", "b"]) == []
    (error,) = budgets.check_file(small(entry.format("instructions.nope")), changelog, ["a", "b"])
    assert "[[signoff]] #1 (instructions.nope): no such budget" in error


@pytest.mark.parametrize(
    ("measured", "status"),
    [(93.0, "ok"), (92.0, "under"), (129.0, "ok"), (131.0, "over"), (100.0, "ok")],
)
def test_a_resident_memory_budget_tightens_past_a_quarter_of_its_tolerance(measured, status):
    """Tightened only past the whole tolerance, an RSS budget settled
    anywhere up to m/(1-t) of the run that set it, and a later run passed up
    to m(1+t)/(1-t): +86% at 30% (review 8, E-4). Tightened past a quarter
    of it, the ceiling is m(1+t)/(1-t/4), +41%; a run over it still fails
    only past the whole tolerance, since two runs of one build read 14-18%
    apart."""
    doc = small()
    doc["meta"]["tolerance"]["rss"] = 0.30
    (v,) = budgets.judge_all(doc, {"rss.streaming.linux": measured})
    assert v.status == status, v.message
    if status == "under":
        assert f"tighten the budget to {measured}" in v.message
        text, done = budgets.apply(SMALL, [v])
        assert tomllib.loads(text)["rss"]["streaming"]["linux"]["budget"] == measured, done
    # The instruction and memory budgets keep the whole tolerance either way.
    (i,) = budgets.judge_all(doc, {"instructions.a": 1_000_000 * (1 - 0.019)})
    assert i.status == "ok"


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


def test_a_budget_deleted_then_returned_higher_is_a_raise():
    """Each version was compared with the next only, so a key missing from
    one version was in neither side of the comparison: deleting a budget in
    one commit and writing it back higher in the next raised it with no
    entry (review 8, E-2). The last number seen for each key is carried
    through the history instead."""
    gone = SMALL.replace('[instructions.a]\nwhat = "case a"\nbudget = 1000000\n', "")
    assert "instructions.a" not in budgets.entries(tomllib.loads(gone))
    versions = [("a" * 40, SMALL), ("b" * 40, gone)]
    (error,) = budgets.check_history(versions, _raised(5_000_000))
    assert "instructions.a: 1000000 -> 5000000 with no [[raised]] entry" in error
    raise_ = RAISE.replace("to = 1100000", "to = 5000000")
    assert budgets.check_history(versions, _raised(5_000_000, raise_)) == []
    # Back at or below the last number it had: no raise.
    assert budgets.check_history(versions, _raised(1_000_000)) == []
    assert budgets.check_history(versions, _raised(900_000)) == []
    # Tightened, deleted, then back at the old number: a raise from the tighter.
    versions = [("a" * 40, SMALL), ("b" * 40, _raised(900_000)), ("c" * 40, gone)]
    (error,) = budgets.check_history(versions, SMALL)
    assert "instructions.a: 900000 -> 1000000" in error


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
    assert after["memory"]["ewridge"]["abs_resid_q0.1_y"] == {"budget": 8.125}
    assert after["memory"]["lasso"]["coef"]["budget"] == 198.205
    assert after["rss"]["streaming"] == {"linux": {"budget": 100.0}, "darwin": {"budget": 300.5}}
    assert len(done) == 4
    # New tables go before the raises, which close the file.
    assert text.index("[memory.lasso]") < text.index(budgets.RAISES_MARK)
    assert budgets.check_history([("a" * 40, SMALL)], text) == []


def test_a_release_number_in_the_file_is_refused():
    """The last release's numbers are its baseline asset, not lines here: a
    `release` beside a budget, or in `[meta]`, is what an old file or an old
    habit leaves, and would read as a baseline nobody uses."""
    changelog = "### Performance\n\nexact clock stamps\n"
    stale = SMALL.replace("budget = 1000000\n", "budget = 1000000\nrelease = 990000\n")
    stale = stale.replace("rows = 1000\n", 'rows = 1000\nrelease = "1.0.0"\n')
    errors = budgets.check_file(tomllib.loads(stale), changelog, ["a", "b"])
    assert len(errors) == 2, errors
    assert any(e.startswith("[meta] names a release") for e in errors), errors
    assert any(e.startswith("instructions.a has a `release` number") for e in errors), errors


def test_a_baseline_merges_a_releases_measurements_and_refuses_a_hole(tmp_path):
    """The release job's `baseline` merges the run's instruction counts and
    each platform's memory into one asset: every budgeted quantity must be
    in it, two files may not disagree on a key, and what it writes its own
    reader takes."""
    doc = small()
    counts = {"instructions.a": 990_000.0, "instructions.b": 5.0}
    memory = {"memory.ewridge.pred_y": 8.125, "memory.ewridge.abs_resid_q0.1_y": 8.0}
    rss = {"rss.streaming.linux": 99.5}
    out = budgets.make_baseline("1.1.0", SHA, [counts, memory, memory, rss], doc)
    assert out["version"] == "1.1.0" and out["sha"] == SHA
    assert out["measured"] == {**counts, **memory, **rss}
    parsed = budgets.parse_baseline(json.dumps(out))
    assert parsed == budgets.Baseline("1.1.0", SHA, {**counts, **memory, **rss})
    with pytest.raises(ValueError, match=r"miss 1 budgets: \['rss.streaming.linux'\]"):
        budgets.make_baseline("1.1.0", SHA, [counts, memory], doc)
    with pytest.raises(ValueError, match="memory.ewridge.pred_y: two measurements disagree"):
        budgets.make_baseline("1.1.0", SHA, [counts, memory, rss, {"memory.ewridge.pred_y": 9.0}])


@pytest.mark.parametrize(
    ("doc", "match"),
    [
        ({"version": "1.1.0", "sha": SHA}, "holds"),
        ({"version": "v1.1.0", "sha": SHA, "measured": {"instructions.a": 1}}, "not a release"),
        ({"version": "1.1.0", "sha": "abc", "measured": {"instructions.a": 1}}, "not a commit"),
        ({"version": "1.1.0", "sha": SHA, "measured": {}}, "measures nothing"),
        ({"version": "1.1.0", "sha": SHA, "measured": {"speed.a": 1}}, "no kind of budget"),
        ({"version": "1.1.0", "sha": SHA, "measured": {"instructions.a": 0}}, "not a positive"),
        ({"version": "1.1.0", "sha": SHA, "measured": {"instructions.a": True}}, "not a positive"),
    ],
)
def test_a_malformed_baseline_is_an_error_not_a_missing_one(doc, match):
    """A file that is read and wrong fails every run, CI or not: only a
    baseline that cannot be read at all is reported and passed over."""
    with pytest.raises(ValueError, match=match):
        budgets.parse_baseline(json.dumps(doc))
    with pytest.raises(ValueError, match="not JSON"):
        budgets.parse_baseline("<html>")


def test_the_baseline_is_read_from_a_file_a_directory_or_the_variable(tmp_path, monkeypatch):
    """Locally and in tests the baseline is a path: the asset itself, or the
    directory a `gh release download` left it in; the variable stands in for
    `--baseline`. A path that is not there is `NoBaseline`, which a plain
    run reports and passes over. By default it is the newest release's
    asset, by a URL that names no version."""
    then = baseline("1.1.0")
    body = {"version": then.version, "sha": then.sha, "measured": then.measured}
    (tmp_path / budgets.BASELINE_ASSET).write_text(json.dumps(body), encoding="utf-8")
    assert budgets.read_baseline(tmp_path) == then
    assert budgets.read_baseline(tmp_path / budgets.BASELINE_ASSET) == then
    monkeypatch.setenv(budgets.BASELINE_VARIABLE, str(tmp_path))
    assert budgets.read_baseline() == then
    with pytest.raises(budgets.NoBaseline):
        budgets.read_baseline(tmp_path / "nowhere")
    reason = budgets.baseline_or_reason(tmp_path / "nowhere")
    assert isinstance(reason, str) and "could not read the last release's baseline" in reason
    assert budgets.BASELINE_URL == (
        "https://github.com/hgilde/polars-online/releases/latest/download/budgets-baseline.json"
    )


def test_the_report_says_which_release_the_drift_is_from_or_why_none():
    doc = small()
    then = baseline("1.1.0")
    verdicts = budgets.judge_all(doc, {"instructions.a": 1_010_000}, baseline=then)
    table = budgets.report(doc, verdicts, baseline=then)
    assert "drift since release 1.1.0 (its budgets-baseline.json)" in table
    assert "| instructions.a | 1,000,000 | 1,010,000 | +1.00% | +1.00% |" in table
    table = budgets.report(doc, budgets.judge_all(doc, {"instructions.a": 1_010_000}), baseline="x")
    assert "no drift: x" in table
    assert "| instructions.a | 1,000,000 | 1,010,000 | +1.00% |  |" in table


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
    then = baseline()
    asset = tmp_path / budgets.BASELINE_ASSET
    asset.write_text(
        json.dumps({"version": then.version, "sha": then.sha, "measured": then.measured}),
        encoding="utf-8",
    )
    monkeypatch.setenv(budgets.BASELINE_VARIABLE, str(asset))
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    for f in (first, second):
        f.write_text(
            "\n".join(json.dumps(_summary(c, n)) for c, n in (("a", 900_000), ("b", 7))),
            encoding="utf-8",
        )
    measured = tmp_path / "measured.json"
    assert budgets.main(["check", str(first), str(second), "--measured", str(measured)]) == 1
    out = capsys.readouterr().out
    assert "::error::instructions.a: 900,000 is -10.00% under its budget" in out
    assert "tighten the budget to 900000" in out
    assert "::warning::instructions.b: no budget yet" in out
    assert json.loads(measured.read_text(encoding="utf-8")) == {
        "instructions.a": 900_000,
        "instructions.b": 7,
    }
    assert "| instructions.a | 1,000,000 | 900,000 | -10.00% | -10.00% |" in summary.read_text(
        encoding="utf-8"
    )
    assert budgets.main(["check", str(first), "--measure"]) == 0
    assert budgets.main(["apply", str(measured)]) == 0
    assert tomllib.loads(path.read_text(encoding="utf-8"))["instructions"]["a"]["budget"] == 900_000
    assert budgets.main(["check", str(first), str(second)]) == 0


def test_a_run_that_cannot_read_the_baseline_fails_only_when_told_to(tmp_path, monkeypatch, capsys):
    """Off CI, `check` without the network reports that it read no baseline
    and holds the budgets as ever; CI's job passes `--require-baseline`,
    and a release check needs it always, so a lost asset cannot quietly
    turn the drift and its sign-off off. With the baseline, a release check
    fails on a case past the sign-off with no `[[signoff]]` for it."""
    path = tmp_path / "budgets.toml"
    path.write_text(SMALL, encoding="utf-8")
    monkeypatch.setattr(budgets, "BUDGETS", path)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv(budgets.MEASURE_VARIABLE, raising=False)
    monkeypatch.setenv(budgets.BASELINE_VARIABLE, str(tmp_path / "nowhere"))
    run = tmp_path / "run.json"
    run.write_text(
        "\n".join(json.dumps(_summary(c, n)) for c, n in (("a", 1_000_000), ("b", 7))),
        encoding="utf-8",
    )
    assert budgets.main(["check", str(run)]) == 0
    out = capsys.readouterr().out
    assert "::warning::could not read the last release's baseline" in out
    assert budgets.main(["check", str(run), "--require-baseline"]) == 1
    assert "::error::could not read the last release's baseline" in capsys.readouterr().out
    assert budgets.main(["check", str(run), "--release-check", "1.1.0"]) == 1
    capsys.readouterr()
    # Read: 12% past the release's own count needs a sign-off for 1.1.0.
    then = baseline("1.0.0", instructions__a=893_000.0)
    asset = tmp_path / budgets.BASELINE_ASSET
    body = {"version": then.version, "sha": then.sha, "measured": then.measured}
    asset.write_text(json.dumps(body), encoding="utf-8")
    args = ["check", str(run), "--baseline", str(tmp_path), "--require-baseline"]
    assert budgets.main(args) == 0
    assert "::warning::instructions.a: +12.0% since 1.0.0" in capsys.readouterr().out
    assert budgets.main([*args, "--release-check", "1.1.0"]) == 1
    assert "::error::instructions.a: +12.0% since 1.0.0" in capsys.readouterr().out
    measured = tmp_path / "measured.json"
    measured.write_text(json.dumps({"instructions.a": 1_000_000}), encoding="utf-8")
    check = ["release-check", str(measured), "--version", "1.1.0", "--baseline", str(asset)]
    assert budgets.main(check) == 1
    path.write_text(
        SMALL + '[[signoff]]\nkey = "instructions.a"\nversion = "1.1.0"\nreason = "a prior"\n',
        encoding="utf-8",
    )
    assert budgets.main([*args, "--release-check", "1.1.0"]) == 0
    assert budgets.main(check) == 0
    nowhere = ["release-check", str(measured), "--version", "1.1.0", "--baseline", "/nowhere"]
    assert budgets.main(nowhere) == 1


def test_baseline_writes_the_asset_the_release_attaches(tmp_path, monkeypatch, capsys):
    path = tmp_path / "budgets.toml"
    path.write_text(SMALL, encoding="utf-8")
    monkeypatch.setattr(budgets, "BUDGETS", path)
    parts = {
        "measured.json": {"instructions.a": 990_000, "instructions.b": 5},
        "memory.json": {"memory.ewridge.pred_y": 8.125, "memory.ewridge.abs_resid_q0.1_y": 8.0},
        "rss-linux.json": {"rss.streaming.linux": 99.5},
    }
    for name, values in parts.items():
        (tmp_path / name).write_text(json.dumps(values), encoding="utf-8")
    out = tmp_path / "asset" / budgets.BASELINE_ASSET
    out.parent.mkdir()
    files = [str(tmp_path / n) for n in parts]
    args = ["baseline", *files, "--version", "1.1.0", "--sha", SHA, "--out", str(out)]
    assert budgets.main(args) == 0
    assert "1.1.0 at c36c03e325, 2 instructions, 2 memory, 1 rss" in capsys.readouterr().out
    assert budgets.read_baseline(out.parent).measured["rss.streaming.linux"] == 99.5
    with pytest.raises(ValueError, match="miss 1 budgets"):
        budgets.main(["baseline", *files[:2], "--version", "1.1.0", "--sha", SHA, "--out", "x"])


def test_measuring_is_asked_for_by_name(monkeypatch):
    monkeypatch.setenv(budgets.MEASURE_VARIABLE, "measure")
    assert budgets.measuring()
    monkeypatch.setenv(budgets.MEASURE_VARIABLE, "1")
    assert not budgets.measuring()


# --- the scaling table: a ratio within one run -----------------------------------

SCALING = """
[scaling]
tolerance = 0.10

[scaling.cells."bank.groups_seen"]
what = "a row over the groups behind it"
growth = "1"
small = 250
large = 2000

[scaling.cells."ewridge.k"]
what = "the Gram's update"
growth = "n^2"
small = 4
large = 32
"""


def test_a_growth_bounds_the_ratio_of_two_sizes():
    assert budgets.growth_bound("1", 4, 32) == 1.0
    assert budgets.growth_bound("n", 4, 32) == 8.0
    assert budgets.growth_bound("n^2", 4, 32) == 64.0
    assert budgets.growth_bound("log n", 10, 100) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="one of"):
        budgets.growth_bound("n log n", 4, 32)


def test_the_bench_names_each_scaling_cell_at_both_sizes():
    """A cell is ``scale_<surface>_by_<dimension>_small`` and ``..._large``,
    and is no budgeted case; a cell at one size alone is no cell."""
    bench = "\n".join(
        f"#[bench::{c}(args = (), setup = s)]"
        for c in (
            "a",
            budgets.scale_case("bank.groups_seen", "small"),
            budgets.scale_case("bank.groups_seen", "large"),
            budgets.scale_case("ewridge.k", "small"),
        )
    )
    assert budgets.scale_case("bank.groups_seen", "large") == "scale_bank_by_groups_seen_large"
    assert budgets.bench_cases(bench) == ["a"]
    assert budgets.scale_cells(bench) == ["bank.groups_seen"]


def test_the_repositorys_scaling_table_is_the_benchs():
    assert set(DOC["scaling"]["cells"]) == set(budgets.scale_cells())
    assert len(budgets.scale_cells()) >= 20


def test_a_scaling_table_that_disagrees_with_the_bench_is_refused():
    doc = small(SCALING)
    assert budgets.check_scaling_file(doc, ["bank.groups_seen", "ewridge.k"]) == []
    (error,) = budgets.check_scaling_file(doc, ["bank.groups_seen"])
    assert "in the file only ['ewridge.k']" in error
    bad = small(SCALING.replace('growth = "n^2"', 'growth = "k^2"'))
    assert any("one of" in e for e in budgets.check_scaling_file(bad, None))
    flat = small(SCALING.replace("large = 32", "large = 4"))
    assert any("small < large" in e for e in budgets.check_scaling_file(flat, None))


@pytest.mark.parametrize(
    ("large", "failed"),
    [(1_000_000, False), (1_099_000, False), (1_200_000, True), (8_000_000, True)],
)
def test_a_constant_cell_fails_past_its_tolerance(large, failed):
    doc = small(SCALING)
    counts = {
        budgets.scale_case("bank.groups_seen", "small"): 1_000_000,
        budgets.scale_case("bank.groups_seen", "large"): large,
        budgets.scale_case("ewridge.k", "small"): 1_000_000,
        budgets.scale_case("ewridge.k", "large"): 20_000_000,
    }
    scaled, missing = budgets.judge_scaling(doc, counts)
    assert missing == []
    by = {v.cell: v for v in scaled}
    assert by["bank.groups_seen"].failed is failed
    assert not by["ewridge.k"].failed  # 20x, within 64x
    assert ("past" in by["bank.groups_seen"].message) is failed


def test_a_cell_the_run_missed_fails():
    doc = small(SCALING)
    counts = {
        budgets.scale_case("ewridge.k", "small"): 1,
        budgets.scale_case("ewridge.k", "large"): 2,
    }
    scaled, missing = budgets.judge_scaling(doc, counts)
    assert [v.cell for v in scaled] == ["ewridge.k"]
    assert missing == ["scaling.bank.groups_seen: the run did not count both sizes"]


def test_check_holds_the_scaling_cells_even_in_a_measuring_run(tmp_path, monkeypatch, capsys):
    """A ratio needs no budget, so a measuring run holds it too: a wrong
    growth is a defect, not a number to re-measure. The scaling cases are
    never budgets, and never in the measurements ``apply`` reads."""
    path = tmp_path / "budgets.toml"
    path.write_text(SMALL + SCALING, encoding="utf-8")
    monkeypatch.setattr(budgets, "BUDGETS", path)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv(budgets.MEASURE_VARIABLE, raising=False)
    asset = tmp_path / budgets.BASELINE_ASSET
    then = baseline()
    asset.write_text(
        json.dumps({"version": then.version, "sha": then.sha, "measured": then.measured}),
        encoding="utf-8",
    )
    monkeypatch.setenv(budgets.BASELINE_VARIABLE, str(asset))
    run = tmp_path / "run.json"
    counts = [
        ("a", 1_000_000),
        ("b", 7),
        (budgets.scale_case("bank.groups_seen", "small"), 1_000_000),
        (budgets.scale_case("bank.groups_seen", "large"), 8_000_000),
        (budgets.scale_case("ewridge.k", "small"), 1_000_000),
        (budgets.scale_case("ewridge.k", "large"), 20_000_000),
    ]
    run.write_text("\n".join(json.dumps(_summary(c, n)) for c, n in counts), encoding="utf-8")
    measured = tmp_path / "measured.json"
    assert budgets.main(["check", str(run), "--measure", "--measured", str(measured)]) == 1
    out = capsys.readouterr().out
    assert "::error::scaling.bank.groups_seen: 8.000x from the small size to the large" in out
    assert "| ewridge.k | 1,000,000 | 20,000,000 | 20.000x | 70.400x |  |" in out
    assert set(json.loads(measured.read_text(encoding="utf-8"))) == {
        "instructions.a",
        "instructions.b",
    }
