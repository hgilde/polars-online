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


def _sgd_on(loss: str):
    return lambda b, kw: b == "sgd" and kw.get("loss") == loss


def _kalman_unsized(b: str, kw: dict) -> bool:
    """`kalman` standardized with no `obs_var`, no intercept and a lowered
    `min_weight`: rows past `min_weight` before a target's prior is sized
    (review round 9, B-1, the path 1.2.0 predicted 0.0 on). With an
    intercept the first row sizes it, diffuse, and no row waits (task 278)."""
    return (b == "kalman" and kw.get("standardize") is True and "obs_var" not in kw) and (
        "min_weight" in kw and kw.get("fit_intercept") is False
    )


#: The paths the release comparison must reach, each by the keyword that
#: takes it (review round 9, J-2: the workload ran 135 of the 346 field names
#: the fullest workload writes, and none of these). A spec of the workload
#: must satisfy each.
PATHS = {
    "a temporal clock": lambda b, kw: kw.get("clock") == "ts",
    "a duration parameter": lambda b, kw: isinstance(kw.get("half_life"), str),
    "gram_block_rows": lambda b, kw: "gram_block_rows" in kw,
    "gram_threads": lambda b, kw: "gram_threads" in kw,
    "target_gaps pairwise": lambda b, kw: kw.get("target_gaps") == "pairwise",
    "a formula target": lambda b, kw: (
        any(isinstance(t, probe._Formula) for t in kw["targets"]) if "targets" in kw else False
    ),
    "feature_sets": lambda b, kw: "feature_sets" in kw,
    "coef_prior": lambda b, kw: "coef_prior" in kw,
    "revert_half_life": lambda b, kw: "revert_half_life" in kw,
    "share_p": lambda b, kw: kw.get("share_p") is True,
    "kalman q and obs_var": lambda b, kw: "q" in kw and "obs_var" in kw,
    "kalman standardized": lambda b, kw: b == "kalman" and kw.get("standardize") is True,
    "kalman's prior sized past min_weight": _kalman_unsized,
    "drift_action reset": lambda b, kw: kw.get("drift_action") == "reset",
    "max_rows_between_solves": lambda b, kw: "max_rows_between_solves" in kw,
    "max_rows_between_coefs": lambda b, kw: "max_rows_between_coefs" in kw,
    "restart_after_step_back": lambda b, kw: "restart_after_step_back" in kw,
    "a lam accumulator": lambda b, kw: "lam" in kw,
    "closed both": lambda b, kw: kw.get("closed") == "both",
    "a lasso window": lambda b, kw: b == "lasso" and "window_size" in kw,
    "an ew_class window": lambda b, kw: b == "ew_class" and "window_size" in kw,
    "marginal window_lags": lambda b, kw: kw.get("window_lags") is True,
    "marginal bin_edges": lambda b, kw: "bin_edges" in kw,
    "group_close session": lambda b, kw: kw.get("group_close") == "session",
    "ftrl squared": lambda b, kw: b == "ftrl" and kw.get("loss") == "squared",
    "pa1": lambda b, kw: b == "pa" and kw.get("mode") == "pa1",
    "pa2": lambda b, kw: b == "pa" and kw.get("mode") == "pa2",
    "sgd adagrad": lambda b, kw: b == "sgd" and kw.get("schedule") == "adagrad",
    **{
        f"sgd {loss}": _sgd_on(loss)
        for loss in ("huber", "quantile", "epsilon_insensitive", "poisson", "logistic")
    },
}


def test_the_workload_reaches_every_field_name_and_every_named_path():
    """The release comparison is read as comparing every output: it holds
    the fullest workload's field names (the memory budgets', every output of
    every model switched on), all of them, and every path in `PATHS`."""
    import polars_online as po
    import test_budgets_memory as memory

    probed: set[str] = set()
    for name, builder, kw, _ in probe.WORKLOAD:
        probed |= set(po.spec.output_fields(probe.build(builder, name, kw)))
    fullest: set[str] = set()
    for name, builder, kw, _ in memory.WORKLOAD:
        fullest |= set(po.spec.output_fields(getattr(po.spec, builder)(name, **kw)))
    assert len(fullest) > 300, len(fullest)
    assert fullest <= probed, sorted(fullest - probed)
    missed = [
        p
        for p, reached in PATHS.items()
        if not any(reached(b, kw) for _, b, kw, _ in probe.WORKLOAD)
    ]
    assert not missed, missed


def test_the_kalman_unsized_prior_path_is_reached_not_only_named():
    """A spec matching `PATHS`' entry for `kalman`'s unsized prior must reach
    it on the probe's stream: at least one row withheld as
    `below_warm_rows`, the rows past `min_weight` before the prior is sized
    (review 9's R4 §6: `kalman_standardized` reported none, so the release
    comparison never compared that path's rows)."""
    import polars_online as po

    specs = [(n, b, kw) for n, b, kw, _ in probe.WORKLOAD if _kalman_unsized(b, kw)]
    assert specs
    df = probe.stream()
    for name, builder, kw in specs:
        out = po.ModelBank([probe.build(builder, name, kw)]).fit_predict(df).unnest(name)
        warm = out.filter(pl.col("withheld_reason") == "below_warm_rows")
        assert warm.height >= 1, (name, out["withheld_reason"].value_counts().to_dicts())
        # Nothing is predicted there: null, not the 0.0 of an unsized prior.
        assert warm["pred_y"].is_null().all(), warm
