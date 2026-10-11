"""`scripts/compare_release.py` and its workload, `scripts/release_probe.py`:
the release step that compares every output with the last release's, bit
for bit (docs/RELEASE-READINESS.md, "Cutting a release")."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import polars as pl

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = _load("compare_release")
probe = _load("release_probe")


def test_bits_are_compared_not_values():
    same = compare.same
    assert same(pl.Series([1.0, None]), pl.Series([1.0, None]))
    assert not same(pl.Series([0.0]), pl.Series([-0.0])), "-0.0 is not 0.0 here"
    assert not same(pl.Series([1.0, None]), pl.Series([1.0, 2.0])), "a null is not a value"
    assert not same(pl.Series([1.0]), pl.Series([1.0 + 2**-52])), "one unit in the last place"
    lists = pl.Series([[1.0, 2.0], None, []])
    assert same(lists, lists.clone())
    assert not same(lists, pl.Series([[1.0, 2.0], None, [0.0]]))
    assert not same(pl.Series([1.0]), pl.Series([1], dtype=pl.Int64)), "the dtype is compared"


def test_a_dtype_change_is_named_not_called_bits():
    """A field whose values are all equal but whose dtype moved -- an
    integer widened, an Enum's categories renamed -- was reported as "bits
    differ (e.g. -0.0 against 0.0)", which sent the reader after a sign of
    zero that was not there (release 1.0's comparison: `kmeans.cluster`,
    every `withheld_reason`)."""
    old = pl.Series([1, 0, None], dtype=pl.Int32)
    new = pl.Series([1, 0, None], dtype=pl.Int64)
    assert not compare.same(old, new)
    assert compare.first_difference(old, new) == "dtype Int32 -> Int64, every value equal"
    old_e = pl.Series(["a", None], dtype=pl.Enum(["a", "below_min_periods"]))
    new_e = pl.Series(["a", None], dtype=pl.Enum(["a", "below_min_weight"]))
    text = compare.first_difference(old_e, new_e)
    assert "bits differ" not in text, text
    assert "gone ['below_min_periods']" in text and "new ['below_min_weight']" in text, text
    # A row that differs is still named first, with the dtype beside it.
    assert compare.first_difference(old, pl.Series([1, 2, None])).startswith("row 1: 0 -> 2")
    # And a sign of zero, whose dtype is the same, keeps its own message.
    assert compare.first_difference(pl.Series([0.0]), pl.Series([-0.0])).startswith("bits differ")
    # An Array against a List of the same numbers is not a dtype change.
    arr = pl.Series([[1.0, 2.0]], dtype=pl.Array(pl.Float64, 2))
    assert compare.same(arr, pl.Series([[1.0, 2.0]]))


def test_this_build_runs_the_whole_workload(tmp_path):
    """A spec this build refuses would drop out of every comparison unseen:
    the workload must keep up with the API."""
    out, manifest = tmp_path / "out.parquet", tmp_path / "m.json"
    probe.main(str(out), str(manifest))
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    assert meta["refused"] == {}, meta["refused"]
    assert meta["ran"] == [name for name, *_ in probe.WORKLOAD]
    frame = pl.read_parquet(out)
    assert frame.height == probe.stream().height and frame.width > 250


def test_a_spec_this_build_refuses_is_a_difference_not_a_silence():
    """`compare` took only the specs both builds ran and never read this
    build's refusals, so a spec the release ran and this build refuses
    dropped out, and the verdict said "identical", exit 0 (task 160, SC1).
    The gate's workload test above catches the probe refusing at HEAD; the
    script's own verdict must not depend on it."""
    meta = {"file": "a", "version": "0.0.1", "ran": ["ridge", "rls"], "refused": {}}
    new_meta = {
        "file": "b",
        "version": "0.0.2",
        "ran": ["ridge"],
        "refused": {"rls": "RuntimeError: boom"},
    }
    frame = pl.DataFrame({"ridge.pred": [1.0, 2.0]})
    old = frame.with_columns(pl.lit(3.0).alias("rls.pred"))
    out: list[str] = []
    code = compare.compare("0.0.1", old, meta, frame, new_meta, out)
    text = "\n".join(out)
    assert code == 1, text
    assert "rls" in text and "RuntimeError: boom" in text, text
    assert "identical" not in text, text
