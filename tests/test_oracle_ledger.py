"""The oracle ledger and the gate's hash guard (``oracles.toml``,
``oracle_hashes.toml``, ``tests/ledger.py``, ``scripts/oracles.py``).

The heavy grids against a third-party oracle run on CI. The ledger holds
each case's error against its oracle and only tightens; these hold the file
to its rules (past the ledger fails, under it fails naming the number,
every raise answered by a ``[[raised]]`` entry quoting the CHANGELOG), the
registrations to the modules, and -- the guard -- this library's outputs on
each group's inputs to the hash the last passing oracle run left. While a
hash holds, the comparison still holds: the oracle's answer is fixed by its
pinned version, which is checked beside it.
"""

from __future__ import annotations

import ast
import datetime as dt
import importlib.util
import json
import os
import sys
import tomllib
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

import ledger

TIER = "essential"
REPO = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # a dataclass looks its module up there
    spec.loader.exec_module(module)
    return module


oracles = _load("oracles")
DOC = oracles.load()
HASHES = tomllib.loads(oracles.HASHES.read_text(encoding="utf-8"))
REGISTERED = ledger.registered_modules()
GROUP_IDS = sorted(ledger.group_id(m, n) for m, names in REGISTERED.items() for n in names)
MODULE_OF = {ledger.group_id(m, n): m for m, names in REGISTERED.items() for n in names}

#: Set to ``enforce`` or ``report`` to override where the guard fails.
MODE_VARIABLE = "POLARS_ONLINE_ORACLE_HASHES"


def enforced() -> bool:
    """Where a moved hash fails: on the platform the hashes were written on,
    off CI. CI runs the oracles themselves on every push, and the hashes are
    the gate's early word, so elsewhere a moved hash is reported, not
    failed: what the CI legs report says whether one set serves them all."""
    mode = os.environ.get(MODE_VARIABLE)
    if mode in ("enforce", "report"):
        return mode == "enforce"
    return not os.environ.get("CI") and ledger.platform_tag() == HASHES["meta"]["platform"]


# --- the guard -----------------------------------------------------------------


@pytest.mark.pins
class TestTheHashGuard:
    """Each group's outputs, quantized, with its tests' source, against the
    committed hash, and its oracle libraries against the ledger's versions.
    A moved hash names the command that re-runs exactly the moved groups'
    oracle grids and, if they pass, records them."""

    @pytest.mark.parametrize("gid", GROUP_IDS)
    def test_our_outputs_on_the_cases_inputs_hold(self, gid):
        group = ledger.load(MODULE_OF[gid])[gid]
        entry = DOC.get("group", {}).get(gid, {})
        problems = []
        try:
            got = ledger.digest(group, entry.get("tests", []))
        except KeyError as e:
            # A test the ledger names is gone, renamed or removed: the group
            # must run again, which records the tests it has now.
            got = None
            problems.append(f"a test the ledger names is gone ({e})")
        want = HASHES.get("hashes", {}).get(gid)
        installed = ledger.versions(group.oracles)
        if want is None:
            problems.append("no hash yet: a new group")
        elif got is not None and got != want:
            problems.append("this library's outputs on its inputs, or its tests, changed")
        if installed != entry.get("oracles", {}):
            problems.append(
                f"its oracles are {installed}, the ledger measured {entry.get('oracles', {})}"
            )
        cases = [c for c in DOC.get("case", {}) if c.rsplit(".", 1)[0] == gid]
        if not cases:
            problems.append("no case in the ledger")
        out = os.environ.get(oracles.HASHES_OUT)
        if out:
            with open(out, "a", encoding="utf-8") as fh:
                line = {
                    "group": gid,
                    "module": MODULE_OF[gid],
                    "hash": got,
                    "holds": not problems,
                    "platform": ledger.platform_tag(),
                }
                fh.write(json.dumps(line) + "\n")
        if not problems:
            return
        message = (
            f"{gid}: {'; '.join(problems)}. Its oracle comparison must run again before "
            f"the ledger can vouch for it: `uv run python scripts/oracles.py rerun {gid}` "
            "(or `rerun` alone for every moved group) runs its oracle grids, and records the "
            "ledger and the hash if they pass."
        )
        if enforced():
            pytest.fail(message)
        warnings.warn(f"{message} (reported, not failed, on {ledger.platform_tag()})", stacklevel=1)
        path = os.environ.get("GITHUB_STEP_SUMMARY")
        if path:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(f"- oracle hash on {ledger.platform_tag()}: {message}\n")


# --- the registrations against the modules ---------------------------------------


def _is_grid_mark(node: ast.AST) -> bool:
    f = node.func if isinstance(node, ast.Call) else node
    return isinstance(f, ast.Attribute) and f.attr == "oracle_grid"


def _grid_tests(tree: ast.Module) -> list[ast.FunctionDef]:
    """The test functions a module marks for the grid tier: on the function,
    its class, or one of its parameters -- directly, or through a
    module-level name the decorator reaches (``marks=_GRID``, or a function
    that builds the parameters with it)."""
    out = []
    top: dict[str, list[ast.stmt]] = {}
    for stmt in tree.body:
        for name in ledger._defines(stmt):
            top.setdefault(name, []).append(stmt)

    def marks(node: ast.AST, seen: set[str]) -> bool:
        for n in ast.walk(node):
            if _is_grid_mark(n):
                return True
            if isinstance(n, ast.Name) and n.id in top and n.id not in seen:
                seen.add(n.id)
                if any(marks(s, seen) for s in top[n.id]):
                    return True
        return False

    def visit(body: list[ast.stmt], inherited: bool) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, inherited or any(_is_grid_mark(d) for d in node.decorator_list))
            elif (
                isinstance(node, ast.FunctionDef)
                and node.name.startswith("test")
                and (inherited or any(marks(d, set()) for d in node.decorator_list))
            ):
                out.append(node)

    module_mark = any(
        _is_grid_mark(n)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", "") == "pytestmark" for t in node.targets)
        for n in ast.walk(node.value)
    )
    visit(tree.body, module_mark)
    return out


def _records(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "record"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "ledger"
        for n in ast.walk(node)
    )


def _trees() -> dict[str, ast.Module]:
    return {
        p.stem: ast.parse(p.read_text(encoding="utf-8"))
        for p in sorted((REPO / "tests").glob("test_*.py"))
    }


def test_every_grid_of_a_registered_module_records():
    """A module that registers records every grid it holds, directly or
    through a module-level helper, so a grid cannot be left out of the
    ledger beside its neighbours."""
    trees = _trees()
    bare = []
    for module in REGISTERED:
        tree = trees[module]
        helpers = {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and _records(n)}
        for test in _grid_tests(tree):
            calls = {
                n.func.id
                for n in ast.walk(test)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            if not (_records(test) or calls & helpers):
                bare.append(f"{module}::{test.name}")
    assert not bare, f"a grid of a registered module records nothing: {bare}"


def test_every_module_with_a_grid_registers_or_is_pending():
    trees = _trees()
    with_grids = {m for m, t in trees.items() if _grid_tests(t)}
    unaccounted = sorted(with_grids - set(REGISTERED) - set(ledger.PENDING))
    assert not unaccounted, (
        f"modules with oracle grids that neither register with the ledger nor are listed in "
        f"ledger.PENDING: {unaccounted}"
    )
    stale = sorted(m for m in ledger.PENDING if m in REGISTERED or m not in with_grids)
    assert not stale, f"ledger.PENDING lists modules that register or hold no grid: {stale}"


def test_the_ledger_the_hashes_and_the_guards_name_the_same_groups():
    ledger_groups = set(DOC.get("group", {}))
    hashed = set(HASHES.get("hashes", {}))
    guards = set(GROUP_IDS)
    assert ledger_groups <= guards, f"in the ledger with no guard: {sorted(ledger_groups - guards)}"
    assert hashed <= guards, f"hashed with no guard: {sorted(hashed - guards)}"
    assert len(GROUP_IDS) == len(set(GROUP_IDS)), "a group registered twice"


def test_the_ledger_is_well_formed_and_every_raise_quotes_the_changelog():
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    assert oracles.check_file(DOC, changelog) == []
    assert HASHES["meta"]["platform"]


# --- the rules -------------------------------------------------------------------

SMALL = """
[meta]
tolerance = 0.02
platform = "darwin-arm64"

[group."m.g"]
tests = []
oracles = { "scikit-learn" = "1.9.1" }
dataset = ""

[case."m.g.err"]
ideal = 0.0
worst = 0.1
p95 = 0.05
n = 50

[case."m.g.ari"]
ideal = 1.0
worst = 0.9
p95 = 0.95
n = 50

[case."m.g.tiny"]
ideal = 0.0
worst = 1e-14
p95 = 1e-15
n = 50
floor = 1e-12

# --- raises
"""


def _m(case: str, worst: float, p95: float, ideal: float = 0.0, n: int = 50) -> dict:
    return {
        "case": case,
        "group": "m.g",
        "worst": worst,
        "p95": p95,
        "n": n,
        "ideal": ideal,
        "floor": 0.0,
        "test": "",
        "oracles": {"scikit-learn": "1.9.1"},
        "dataset": "",
    }


def _status(doc: dict, *measured: dict) -> dict[str, str]:
    cases = {m["case"]: m for m in measured}
    groups = {"m.g": {"oracles": measured[0]["oracles"], "dataset": "", "tests": []}}
    return {v.key: v.status for v in oracles.judge_all(doc, cases, groups)}


class TestTheRules:
    doc = tomllib.loads(SMALL)

    def test_a_worse_error_fails_and_a_better_one_asks_to_tighten(self):
        s = _status(self.doc, _m("m.g.err", 0.103, 0.05))
        assert s == {"m.g.err.worst": "over", "m.g.err.p95": "ok"}
        s = _status(self.doc, _m("m.g.err", 0.1019, 0.0485))
        assert s == {"m.g.err.worst": "ok", "m.g.err.p95": "under"}

    def test_a_score_is_judged_by_its_gap_from_its_ideal(self):
        # ARI 0.9 is a gap of 0.1: 0.897 is 3% worse, 0.9025 2.5% better.
        s = _status(self.doc, _m("m.g.ari", 0.897, 0.95, ideal=1.0))
        assert s["m.g.ari.worst"] == "over"
        s = _status(self.doc, _m("m.g.ari", 0.9025, 0.95, ideal=1.0))
        assert s["m.g.ari.worst"] == "under"

    def test_a_gap_under_its_floor_is_round_off(self):
        s = _status(self.doc, _m("m.g.tiny", 9e-13, 1e-16))
        assert set(s.values()) == {"ok"}
        assert _status(self.doc, _m("m.g.tiny", 2e-12, 1e-16))["m.g.tiny.worst"] == "over"

    def test_a_new_case_fails_nothing_and_a_moved_oracle_fails(self):
        m = _m("m.g.other", 0.5, 0.4)
        assert _status(self.doc, m) == {"m.g.other": "new"}
        m = _m("m.g.err", 0.1, 0.05)
        m["oracles"] = {"scikit-learn": "1.9.2"}
        assert _status(self.doc, m)["m.g.oracles"] == "version"

    def test_a_case_the_run_did_not_measure_fails(self):
        cases = {"m.g.err": _m("m.g.err", 0.1, 0.05)}
        missed = {v.key for v in oracles.missing(self.doc, cases)}
        assert missed == {"m.g.ari", "m.g.tiny"}

    def test_apply_writes_new_and_tightened_numbers_and_never_a_raise(self):
        measured = [_m("m.g.err", 0.2, 0.04), _m("m.g.new", 0.3, 0.2)]
        cases = {m["case"]: m for m in measured}
        groups = {"m.g": {"oracles": {"scikit-learn": "1.9.2"}, "dataset": "", "tests": ["t"]}}
        verdicts = oracles.judge_all(self.doc, cases, groups)
        text, done = oracles.apply(SMALL, self.doc, cases, groups, verdicts)
        new = tomllib.loads(text)
        assert new["case"]["m.g.err"]["worst"] == 0.1, "a raise needs a reason, not apply"
        assert new["case"]["m.g.err"]["p95"] == 0.04
        assert new["case"]["m.g.new"] == {"ideal": 0.0, "worst": 0.3, "p95": 0.2, "n": 50}
        assert new["group"]["m.g"]["oracles"] == {"scikit-learn": "1.9.2"}
        assert new["group"]["m.g"]["tests"] == ["t"]
        # A test renamed since: the run's tests replace the ledger's.
        renamed = SMALL.replace("tests = []", 'tests = ["gone"]')
        text, _ = oracles.apply(renamed, tomllib.loads(renamed), cases, groups, verdicts)
        assert tomllib.loads(text)["group"]["m.g"]["tests"] == ["t"]
        assert text.index('[case."m.g.new"]') < text.index("# --- raises")

    def test_every_raise_in_the_history_needs_an_entry(self):
        raised = SMALL.replace("worst = 0.1\n", "worst = 0.2\n")
        errors = oracles.check_history([("a" * 40, SMALL)], raised)
        assert len(errors) == 1 and "m.g.err.worst" in errors[0]
        entry = (
            '\n[[raised]]\nkey = "m.g.err.worst"\nfrom = 0.1\nto = 0.2\ndate = 2026-10-26\n'
            'reason = "the price of a change worth having"\nchangelog = "worth having"\n'
        )
        assert oracles.check_history([("a" * 40, SMALL)], raised + entry) == []
        # A score raises when it moves away from its ideal: 0.9 -> 0.85.
        worse = SMALL.replace("worst = 0.9\n", "worst = 0.85\n")
        assert "m.g.ari.worst" in oracles.check_history([("a" * 40, SMALL)], worse)[0]
        better = SMALL.replace("worst = 0.9\n", "worst = 0.92\n")
        assert oracles.check_history([("a" * 40, SMALL)], better) == []
        doc = tomllib.loads(raised + entry)
        assert oracles.check_file(doc, "- the price of a change worth having") == []
        assert oracles.check_file(doc, "nothing")[0].endswith("'worth having'")
        assert isinstance(doc["raised"][0]["date"], dt.date)

    def test_a_count_is_held_exactly_however_large(self):
        # 2% of a delay of 333 rows is 6.7 rows; a count holds to the row.
        count = '[case."m.g.delay"]\nideal = 0.0\nworst = 333.0\np95 = 277.0\nn = 200\n'
        doc = tomllib.loads(
            SMALL.replace("# --- raises", count + "tolerance = 0.0\n\n# --- raises")
        )
        assert _status(doc, _m("m.g.delay", 334.0, 277.0))["m.g.delay.worst"] == "over"
        assert _status(doc, _m("m.g.delay", 332.0, 277.0))["m.g.delay.worst"] == "under"
        assert set(_status(doc, _m("m.g.delay", 333.0, 277.0)).values()) == {"ok"}
        loose = tomllib.loads(SMALL.replace("# --- raises", count + "\n# --- raises"))
        assert _status(loose, _m("m.g.delay", 339.0, 277.0))["m.g.delay.worst"] == "ok"
        # The test's mark holds it before the ledger carries it, and `apply`
        # writes it: on a new case, and on one the test now says is a count.
        marked = {**_m("m.g.delay", 334.0, 277.0), "tolerance": 0.0}
        assert _status(loose, marked)["m.g.delay.worst"] == "over"
        cases = {
            "m.g.new": {**_m("m.g.new", 70.0, 60.0), "tolerance": 0.0},
            "m.g.err": {**_m("m.g.err", 0.1, 0.05), "tolerance": 0.0},
        }
        groups = {"m.g": {"oracles": {"scikit-learn": "1.9.1"}, "dataset": "", "tests": []}}
        verdicts = oracles.judge_all(self.doc, cases, groups)
        text, _ = oracles.apply(SMALL, self.doc, cases, groups, verdicts)
        new = tomllib.loads(text)["case"]
        assert new["m.g.new"]["tolerance"] == 0.0 and new["m.g.err"]["tolerance"] == 0.0
        assert oracles.check_history([("a" * 40, SMALL)], text) == [], "a tightening"

    def test_records_are_the_worst_seed_and_the_nearest_rank_p95(self):
        values = np.r_[np.arange(1, 101) / 100.0]
        worst, p95, n = ledger.summarize(values)
        assert (worst, p95, n) == (1.0, 0.95, 100)
        worst, p95, _ = ledger.summarize(values, ideal=1.0)
        assert (worst, p95) == (0.01, 0.06)
        with pytest.raises(ValueError, match="not finite"):
            ledger.summarize([0.1, np.nan])


def _check(tmp_path, *lines: dict, args: tuple[str, ...] = ()) -> tuple[int, str]:
    """``oracles.py check`` on records holding ``lines``, against SMALL."""
    import subprocess

    (tmp_path / "ledger.toml").write_text(SMALL, encoding="utf-8")
    records = tmp_path / "records"
    records.mkdir(exist_ok=True)
    (records / "records-1.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
    )
    res = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "oracles.py"),
            "check",
            str(records),
            "--ledger",
            str(tmp_path / "ledger.toml"),
            *args,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return res.returncode, res.stdout


class TestTheCheckCommand:
    """The command CI's ledger job runs, end to end on records as
    ``ledger.record`` writes them: a case 3% worse than the ledger fails,
    one 3% better fails naming the number to tighten to, and one within 2%
    passes."""

    rest = (_m("m.g.ari", 0.9, 0.95, ideal=1.0), _m("m.g.tiny", 1e-14, 1e-15))

    def test_within_the_tolerance_passes(self, tmp_path):
        code, out = _check(tmp_path, _m("m.g.err", 0.101, 0.0495), *self.rest)
        assert code == 0, out

    def test_a_worse_case_fails(self, tmp_path):
        code, out = _check(tmp_path, _m("m.g.err", 0.103, 0.05), *self.rest)
        assert code == 1 and "::error::m.g.err.worst: 0.103" in out, out
        assert "past the ledger's 0.1" in out

    def test_a_better_case_fails_asking_to_tighten(self, tmp_path):
        code, out = _check(tmp_path, _m("m.g.err", 0.097, 0.05), *self.rest)
        assert code == 1 and "tighten it to 0.097 (`apply`)" in out, out

    def test_a_case_a_complete_run_left_out_fails_and_so_does_an_incomplete_run(self, tmp_path):
        one = _m("m.g.err", 0.1, 0.05)
        code, out = _check(tmp_path, one, args=("--expect-shards", "1"))
        assert code == 1 and "0 of 1 shards finished" in out, out
        assert "m.g.ari" not in out, "an incomplete run is not judged on what it left out"
        (tmp_path / "records" / "done").touch()
        code, out = _check(tmp_path, one, args=("--expect-shards", "1"))
        assert code == 1 and "::error::m.g.ari: in the ledger, and the run did not" in out, out


class TestTheHash:
    def test_the_last_bits_do_not_move_it_and_a_real_change_does(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(10_000) * 10.0 ** rng.uniform(-6, 6, 10_000)
        base = ledger.canonical(x).hexdigest()
        # Every value moved by up to 1e-12 relative, as another libm's last
        # bits can: the chance that one of 10,000 straddles a 24-bit
        # rounding boundary is about 10,000 x 1e-12 / 6e-8, 1.7e-1 at most
        # here -- so the seed is one where none does, and the 1e-15 moves
        # the usual differences make never do.
        for scale in (1e-15, 1e-13):
            moved = x * (1 + scale * rng.uniform(-1, 1, x.size))
            assert ledger.canonical(moved).hexdigest() == base, scale
        changed = x.copy()
        changed[17] *= 1 + 1e-6
        assert ledger.canonical(changed).hexdigest() != base
        assert ledger.canonical(x, bits=52).hexdigest() != base

    def test_round_off_about_zero_and_minus_zero_hash_as_zero(self):
        a = np.array([1.0, 3e-17, -0.0, 2.0])
        b = np.array([1.0, -2e-17, 0.0, 2.0])
        assert ledger.canonical(a).hexdigest() == ledger.canonical(b).hexdigest()

    def test_a_value_at_a_power_of_two_does_not_straddle(self):
        # 1.0 and the doubles either side of it round to the same 24 bits.
        a = np.array([1.0, 0.5])
        b = np.array([np.nextafter(1.0, 0.0), np.nextafter(0.5, 1.0)])
        assert ledger.canonical(a).hexdigest() == ledger.canonical(b).hexdigest()

    def test_frames_and_nested_outputs_hash_by_kind_shape_and_nulls(self):
        f = pl.DataFrame({"a": [1.0, None, 3.0], "b": [1, 2, 3]})
        g = pl.DataFrame({"a": [1.0, 0.0, 3.0], "b": [1, 2, 3]})
        assert ledger.canonical(f).hexdigest() != ledger.canonical(g).hexdigest()
        assert (
            ledger.canonical([1.0, 2.0]).hexdigest() != ledger.canonical([[1.0], [2.0]]).hexdigest()
        )
        assert ledger.canonical({"x": 1}).hexdigest() == ledger.canonical({"x": 1}).hexdigest()

    def test_a_tests_source_counts_and_its_docstring_does_not(self, tmp_path, monkeypatch):
        src = 'def test_a():\n    """Prose."""\n    assert 1 < 2\n'
        (tmp_path / "t.py").write_text(src, encoding="utf-8")
        monkeypatch.setattr(ledger, "REPO", tmp_path)
        a = ledger.source_of("t.py::test_a[x]")
        (tmp_path / "t.py").write_text(src.replace("Prose.", "Other prose."), encoding="utf-8")
        assert ledger.source_of("t.py::test_a") == a
        (tmp_path / "t.py").write_text(src.replace("1 < 2", "1 < 3"), encoding="utf-8")
        assert ledger.source_of("t.py::test_a") != a


# --- the workflow ------------------------------------------------------------------


def test_the_oracle_workflow_records_every_shard_and_checks_the_ledger():
    wf = yaml.safe_load((REPO / ".github/workflows/oracles.yml").read_text(encoding="utf-8"))
    on = wf.get(True, wf.get("on"))
    assert "schedule" in on, "the weekly full sweep"
    # Every push to main runs every case, so a bump of an oracle library or
    # a dataset's loader re-runs everything on the push that carries it.
    assert on["push"] == {"branches": ["main"]}, on["push"]
    assert "github.event_name" in wf["concurrency"]["group"], "a push cannot cancel the sweep"
    cache = [s for s in wf["jobs"]["grid"]["steps"] if "actions/cache" in str(s.get("uses", ""))]
    assert [s.get("if") for s in cache] == ["github.event_name != 'schedule'"], (
        "the weekly sweep downloads the datasets afresh"
    )
    grid = wf["jobs"]["grid"]
    assert grid["env"][oracles.OUT_VARIABLE]
    n = len(grid["strategy"]["matrix"]["shard"])
    uploads = [s for s in grid["steps"] if "upload-artifact" in str(s.get("uses", ""))]
    assert len(uploads) == 1 and uploads[0]["if"] == "always()"
    job = wf["jobs"]["ledger"]
    assert job["needs"] == "grid" and "always()" in job["if"]
    assert job["steps"][0]["with"]["fetch-depth"] == 0, "the raise rule reads the whole history"
    runs = " ".join(str(s.get("run", "")) for s in job["steps"])
    assert "oracles.py raises" in runs
    assert f"oracles.py check records --expect-shards {n} --measured measured.json" in runs
    assert "uv " not in runs, "the check needs no project environment"
