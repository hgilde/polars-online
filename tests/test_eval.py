"""Task 12: the evaluation harness (docs/PLAN.md section 8)."""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from data import synthetic


def _fitted(n_groups=2, n_rows=300, **kw):
    df, _ = synthetic(seed=51, n_groups=n_groups, n_rows=n_rows, k=3, null_frac=0.0)
    opts = dict(
        targets=["y0"],
        features=["x0", "x1", "x2"],
        halflife=200.0,
        clock="t",
        max_dclock=50.0,
        group="group",
        min_periods=10.0,
    )
    opts.update(kw)
    return po.ModelBank([po.spec.ewridge("m", **opts)]).fit_predict(df)


def test_metrics_shape_and_values():
    out = _fitted()
    m = po.eval.metrics(out, "m", by=["group"], targets=["y0"])
    assert set(m["group"]) == {"g0", "g1"}
    assert m["slot"].unique().to_list() == ["pred_y0"]
    # synthetic data is genuinely predictable, so R^2 and IC must be positive
    assert (m["r2"] > 0.3).all()
    assert (m["ic"] > 0.5).all()
    assert ((m["hit_rate"] >= 0) & (m["hit_rate"] <= 1)).all()


def test_metrics_drops_nulls_not_rows():
    # Warmup rows have null predictions; they must not enter the counts.
    # (min_periods must stay below the saturation level of n_eff, which for
    # halflife 200 and ~10-unit clock deltas is around 29.)
    out = _fitted(min_periods=20.0)
    m = po.eval.metrics(out, "m", targets=["y0"])
    n_finite = out["m"].struct.field("pred_y0").is_not_null().sum()
    assert m["n"][0] == n_finite


def test_r2_matches_a_manual_computation():
    out = _fitted(n_groups=1)
    m = po.eval.metrics(out, "m", targets=["y0"])
    pred = out["m"].struct.field("pred_y0").to_numpy().astype(float)
    y = out["y0"].to_numpy().astype(float)
    ok = np.isfinite(pred) & np.isfinite(y)
    r2 = 1 - ((y[ok] - pred[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
    assert abs(m["r2"][0] - r2) < 1e-12


def test_rolling_windows_partition_the_clock():
    out = _fitted(n_groups=1, n_rows=600)
    r = po.eval.rolling_metrics(out, "m", clock="t", window=800.0, targets=["y0"], min_obs=5)
    assert r.height > 1
    starts = r["window_start"].to_numpy()
    assert (np.diff(starts) == 800.0).all()
    assert (starts % 800.0 == 0).all()


def test_compare_specs_stacks_with_a_spec_column():
    df, _ = synthetic(seed=52, n_groups=1, n_rows=250, k=3, null_frac=0.0)
    common = dict(targets=["y0"], features=["x0", "x1", "x2"], halflife=200.0, min_periods=10.0)
    out = po.ModelBank(
        [
            po.spec.ewridge("a", ridge=1e-6, **common),
            po.spec.ewridge("b", ridge=10.0, **common),
        ]
    ).fit_predict(df)
    cmp = po.eval.compare_specs(out, ["a", "b"], targets=["y0"])
    assert set(cmp["spec"]) == {"a", "b"}
    # heavy ridge must fit worse on data with a real signal
    mse = dict(zip(cmp["spec"], cmp["mse"], strict=True))
    assert mse["a"] < mse["b"]


def test_grid_slots_resolve_to_their_target():
    df, _ = synthetic(seed=53, n_groups=1, n_rows=250, k=2, n_targets=2, null_frac=0.0)
    spec = po.spec.ewridge(
        "m",
        targets=["y0", "y1"],
        features=["x0", "x1"],
        ridge=[1e-6, 1.0],
        halflife=200.0,
        min_periods=10.0,
    )
    out = po.ModelBank([spec]).fit_predict(df)
    m = po.eval.metrics(out, "m", targets=["y0", "y1"])
    assert m.height == 4  # 2 targets x 2 ridge values
    assert set(m["target"]) == {"y0", "y1"}


def test_unpack_is_long_form():
    out = _fitted(n_groups=1, n_rows=100)
    long = po.eval.unpack(out, "m", targets=["y0"])
    assert long.height == out.height  # one slot
    assert {"slot", "target", "pred", "y"} <= set(long.columns)


def test_rejects_non_struct_column():
    out = _fitted(n_groups=1, n_rows=50)
    with pytest.raises(TypeError, match="not a model-output struct"):
        po.eval.metrics(out, "x0")


def test_noise_target_gives_no_edge():
    rng = np.random.default_rng(3)
    n = 3000
    df = pl.DataFrame(
        {"x0": rng.standard_normal(n), "x1": rng.standard_normal(n), "y0": rng.standard_normal(n)}
    )
    spec = po.spec.ewridge(
        "m", targets=["y0"], features=["x0", "x1"], halflife=300.0, min_periods=20.0
    )
    out = po.ModelBank([spec]).fit_predict(df)
    m = po.eval.metrics(out, "m", targets=["y0"])
    assert abs(m["ic"][0]) < 0.06
    assert abs(m["hit_rate"][0] - 0.5) < 0.05


def test_target_named_like_an_output_column_does_not_collide():
    # A target column literally called "y" collided with unpack()'s own "y"
    # output; reserved names are dropped from the passthrough columns instead.
    df, _ = synthetic(seed=54, n_groups=1, n_rows=150, k=2, null_frac=0.0)
    df = df.rename({"y0": "y"})
    spec = po.spec.ewridge(
        "m", targets=["y"], features=["x0", "x1"], halflife=200.0, min_periods=10.0
    )
    out = po.ModelBank([spec]).fit_predict(df)
    m = po.eval.metrics(out, "m", targets=["y"])
    assert m.height == 1
    assert m["target"][0] == "y"
    long = po.eval.unpack(out, "m", targets=["y"])
    assert long.columns.count("y") == 1
