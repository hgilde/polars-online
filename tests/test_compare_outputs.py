"""`scripts/compare_outputs.py`: two builds' outputs, how far each column moves and what flips.

The tool is a dev instrument for judging a change that moves output bits, so
what is held here is that it tells the truth about small movements: builds
that are the same report no movement at all, and a one-ulp change in one
column of one model reports that column and no other, with the movement it
is. The second build is a copy of the installed package whose
`fit_predict_batches` nudges one output field by one ulp, run under the same
interpreter through the tool's ``--b-path``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import polars_online

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


co = _load("compare_outputs")

#: Appended to the copy's ``__init__``: the ridge's ``pred_y0`` one ulp up, nulls kept.
NUDGE = """

def _nudge() -> None:
    import numpy as np
    import polars as pl

    from ._bank import ModelBank

    orig = ModelBank.fit_predict_batches

    def patched(self, *args, **kwargs):
        for out in orig(self, *args, **kwargs):
            fields = out["smoke_ridge"].struct.unnest()
            col = fields["pred_y0"]
            up = pl.Series(np.nextafter(col.fill_null(0.0).to_numpy(), np.inf))
            fields = fields.with_columns(
                pl.when(col.is_null()).then(None).otherwise(pl.lit(up)).alias("pred_y0")
            )
            yield out.with_columns(pl.struct(fields).alias("smoke_ridge"))

    ModelBank.fit_predict_batches = patched


_nudge()
"""


def _copy_of_package(dest: Path, nudge: bool) -> Path:
    """``dest/polars_online``: the installed package, its native library linked
    (copied where a link is not allowed), and the nudge appended if asked."""
    src = Path(polars_online.__file__).parent
    pkg = dest / "polars_online"
    pkg.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_dir():
            continue
        if f.suffix in (".so", ".pyd", ".dylib"):
            try:
                os.symlink(f, pkg / f.name)
                continue
            except OSError:
                pass
        shutil.copy2(f, pkg / f.name)
    if nudge:
        with open(pkg / "__init__.py", "a", encoding="utf-8") as fh:
            fh.write(NUDGE)
    return dest


def _run(tmp_path: Path, b_path: Path, *extra: str) -> tuple[int, dict]:
    out = tmp_path / "out"
    shutil.rmtree(out, ignore_errors=True)
    code = co.main(
        [
            "--a", sys.executable, "--b", sys.executable, "--b-path", str(b_path),
            "--builtin", "--rows-scale", "0.3", "--time-rounds", "0",
            "--out", str(out), "--work", str(tmp_path / "work"),
            "--require-identical", *extra,
        ]
    )  # fmt: skip
    return code, json.loads((out / "report.json").read_text(encoding="utf-8"))


def test_identical_builds_report_nothing_moved(tmp_path, capsys):
    code, report = _run(tmp_path, _copy_of_package(tmp_path / "same", nudge=False))
    capsys.readouterr()
    assert code == 0
    assert len(report["cases"]) == 2
    for case in report["cases"]:
        s = case["summary"]
        assert s["float_columns"] > 8, "the columns were compared"
        assert s["float_columns_moved"] == 0 and s["max_rel"] == 0.0
        assert s["rows_any_differ"] == 0
        assert not s["flipped_columns"] and not s["decision_flips"]
        assert s["nulls_vanish"] == 0 and s["nulls_appear"] == 0
        assert case["state"] == "same"


def test_one_ulp_in_one_column_is_that_column_alone(tmp_path, capsys):
    code, report = _run(
        tmp_path, _copy_of_package(tmp_path / "nudged", nudge=True), "--only", "smoke/k4-g3-c500"
    )
    capsys.readouterr()
    assert code == 1
    (case,) = report["cases"]
    moved = [c for c, st in case["columns"].items() if st.get("rows_differing")]
    assert moved == ["smoke_ridge.pred_y0"]
    st = case["columns"]["smoke_ridge.pred_y0"]
    assert 0 < st["max_rel"] <= 2.0**-52 * 1.01, "one ulp is at most 2**-52 relative"
    assert st["median_rel"] <= st["p999_rel"] <= st["max_rel"]
    assert st["nulls_vanish"] == 0 and st["nulls_appear"] == 0
    assert case["summary"]["worst_column"] == "smoke_ridge.pred_y0"
    assert case["summary"]["decision_flips"] == {} and case["summary"]["flipped_columns"] == {}


def test_abba_time_and_the_tables(tmp_path, capsys):
    b = _copy_of_package(tmp_path / "same", nudge=False)
    _run(tmp_path, b, "--only", "smoke/k4-g3-save", "--time-rounds", "2")
    capsys.readouterr()
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    t = report["cases"][0]["timing"]
    assert t["rounds"] == 2 and all(len(v) == 2 for v in t["seconds"].values())
    assert t["min"][0] <= t["median"][0] and t["ratio_min"] > 0
    md = (tmp_path / "out" / "item-smoke.md").read_text(encoding="utf-8")
    assert "smoke/k4-g3-save" in md and "1e-12" in md, "the floor is printed"


def test_a_build_is_not_compared_with_itself(tmp_path):
    with pytest.raises(SystemExit, match="nothing would be compared"):
        co.main(
            [
                "--a", sys.executable, "--b", sys.executable, "--builtin", "--rows-scale", "0.2",
                "--only", "smoke/k4-g3-save", "--time-rounds", "0",
                "--out", str(tmp_path / "out"), "--work", str(tmp_path / "work"),
            ]
        )  # fmt: skip


def _stats(a, b, floor=1e-12):
    st = co.column_stats(pl.Series(a), pl.Series(b), floor)
    st.pop("_rows")
    return st


def test_movement_is_relative_with_a_floor_and_counts_what_appears():
    one = 1.0 + 2.0**-52
    st = _stats([1.0, 2.0, None, 5.0, float("nan")], [one, 2.0, 3.0, None, float("nan")])
    assert st["kind"] == "float" and st["rows_differing"] == 3
    assert st["max_rel"] == pytest.approx(2.0**-52 / one)
    assert st["nulls_appear"] == 1 and st["nulls_vanish"] == 1
    assert st["nan_appear"] == 0 and st["nan_vanish"] == 0, "NaN against NaN is the same"
    # Near zero the floor decides: 1e-20 against 3e-20 is a movement of 2e-20 / 1e-12.
    assert _stats([1e-20], [3e-20], floor=1e-12)["max_rel"] == pytest.approx(2e-8)
    assert _stats([1e-20], [3e-20], floor=1e-30)["max_rel"] == pytest.approx(2 / 3)
    assert _stats([0.0], [-0.0])["rows_differing"] == 1, "a sign of zero is a bit that differs"
    assert _stats([0.0], [-0.0])["max_rel"] == 0.0


def test_a_list_is_compared_by_element_and_a_length_change_is_named():
    a = [[1.0, 2.0], [3.0, 4.0], [5.0], None]
    b = [[1.0, 2.0], [3.0, 4.0 * (1 + 2.0**-50)], [5.0, 6.0], None]
    st = _stats(a, b)
    assert st["kind"] == "list" and st["shape_rows"] == 1
    assert st["rows_differing"] == 2, "the element that moved and the row whose length did"
    assert st["max_rel"] == pytest.approx(2.0**-50, rel=0.01)


def test_other_columns_report_the_rows_that_flip():
    assert _stats([True, False, True], [True, True, True])["rows_differing"] == 1
    assert _stats([1, 2, 3], [1, 2, 4])["rows_differing"] == 1
    assert _stats(["a", None, "b"], ["a", "b", "b"])["rows_differing"] == 1
    old = pl.Series(["x", "y"], dtype=pl.Enum(["x", "y"]))
    new = pl.Series(["x", "z"], dtype=pl.Enum(["x", "z"]))
    st = co.column_stats(old, new, 1e-12)
    assert st["kind"] == "exact" and st["rows_differing"] == 1


def test_a_declared_decision_flips_where_it_crosses_its_threshold():
    cols_a = {"m.stat": pl.Series([0.9, 1.1, 2.0, None]), "m.p": pl.Series([0.2, 0.2, 0.9, 0.5]),
              "m.q": pl.Series([0.8, 0.8, 0.1, 0.4])}  # fmt: skip
    cols_b = {"m.stat": pl.Series([1.001, 1.1, 0.99, None]), "m.p": pl.Series([0.2, 0.9, 0.9, 0.5]),
              "m.q": pl.Series([0.8, 0.1, 0.1, 0.4])}  # fmt: skip
    names = list(cols_a)
    (t,) = co._decision_flips(
        {"name": "alarm", "kind": "threshold", "column": "m.st*", "value": 1.0, "op": ">"},
        cols_a.__getitem__, cols_b.__getitem__, names,
    )  # fmt: skip
    assert t["flips"] == 2 and t["rows"] == 4
    (arg,) = co._decision_flips(
        {"name": "label", "kind": "argmax", "columns": ["m.p", "m.q"]},
        cols_a.__getitem__, cols_b.__getitem__, names,
    )  # fmt: skip
    assert arg["flips"] == 1


def test_cases_are_files_of_dicts_found_by_a_folder(tmp_path):
    (tmp_path / "one.py").write_text(
        "def cases():\n    return [{'name': 'one/a', 'specs': lambda po, ctx: [], 'rows': 100}]\n",
        encoding="utf-8",
    )
    (tmp_path / "two.py").write_text(
        "CASES = [{'name': 'two/a', 'item': 'b', 'windows': lambda po, ctx: {}, 'k': 20}]\n",
        encoding="utf-8",
    )
    cases = co.load_cases([tmp_path], builtin=False)
    assert [(c["name"], c["item"], c["k"], c["chunk"]) for c in cases] == [
        ("one/a", "one", 4, 1000),
        ("two/a", "b", 20, 1000),
    ]
    assert [c["name"] for c in co.select(cases, ["two"], [])] == ["two/a"]
    assert [c["name"] for c in co.select(cases, [], ["one"])] == ["one/a"]
    assert co.scaled(cases[0], 0.01)["rows"] == 50, "a scaled stream keeps a floor of rows"


def test_inputs_are_the_same_bits_from_the_same_case():
    case = {**co.CASE_DEFAULTS, "name": "x", "rows": 2000, "groups": 5, "extras": ["hit", "lab"]}
    a, b = co.make_input(case), co.make_input(case)
    assert a.equals(b) and co.data_key(case) == co.data_key(dict(case))
    assert {"t", "w", "g", "x0", "x3", "y0", "hit", "lab"} <= set(a.columns)
    assert a["g"].n_unique() <= 5 and np.all(np.diff(a["t"].to_numpy()) > 0)
