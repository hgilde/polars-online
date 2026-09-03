"""Task 10: lasso path + online lambda selection (docs/PLAN.md section 4.3)."""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from data import synthetic


def _spec(path, **kw):
    defaults = dict(
        targets=["y0"],
        features=["x0", "x1", "x2"],
        lasso_path=path,
        halflife=1e9,
        max_rows_between_solves=1,
        min_periods=10.0,
    )
    defaults.update(kw)
    return po.spec.lasso("m", **defaults)


def test_output_fields_include_lam_selected():
    spec = _spec([1.0, 0.1, 0.0])
    fields = po.spec.output_fields(spec)
    assert "lam_selected_y0" in fields
    assert "pred_y0__l1" in fields and "pred_y0__l0" in fields


def test_zero_penalty_matches_ewridge():
    # lambda = 0 lasso == unpenalized least squares == ew_ridge with tiny ridge.
    df, _ = synthetic(seed=31, n_groups=1, n_rows=300, k=3, null_frac=0.0)
    common = dict(
        targets=["y0"],
        features=["x0", "x1", "x2"],
        halflife=1e9,
        max_rows_between_solves=1,
        min_periods=10.0,
    )
    a = po.ModelBank([po.spec.lasso("m", lasso_path=[0.0], **common)]).fit_predict(df)
    b = po.ModelBank([po.spec.ewridge("m", ridge=1e-12, standardize=True, **common)]).fit_predict(
        df
    )
    # lasso fields are always suffixed by their path point
    pa = a["m"].struct.field("pred_y0__l0").to_numpy().astype(float)
    pb = b["m"].struct.field("pred_y0").to_numpy().astype(float)
    m = np.isfinite(pa) & np.isfinite(pb)
    assert m.sum() > 200
    assert np.max(np.abs(pa[m] - pb[m])) < 1e-6


def test_heavy_penalty_predicts_the_mean():
    df, _ = synthetic(seed=32, n_groups=1, n_rows=200, k=3, null_frac=0.0)
    out = po.ModelBank([_spec([1e6, 0.0])]).fit_predict(df)
    # All slopes zeroed => prediction is the running EW mean of y, so it must be
    # constant across rows with very different x.
    pred = out["m"].struct.field("pred_y0__l1000000").to_numpy().astype(float)
    finite = pred[np.isfinite(pred)]
    assert finite.std() < 0.5 * df["y0"].std()


def test_lam_selected_is_on_the_path():
    df, _ = synthetic(seed=33, n_groups=2, n_rows=250, k=3, null_frac=0.0)
    path = [1.0, 0.3, 0.1, 0.0]
    out = po.ModelBank([_spec(path, group="group", halflife=200.0)]).fit_predict(df)
    sel = out["m"].struct.field("lam_selected_y0").to_numpy().astype(float)
    assert set(np.unique(sel[np.isfinite(sel)])).issubset(set(path))


def test_selection_prefers_penalty_when_features_are_noise():
    # y is pure noise: no feature helps, so a penalty that zeroes everything
    # must beat lambda = 0 on out-of-sample error most of the time.
    rng = np.random.default_rng(5)
    n = 3000
    df = pl.DataFrame(
        {
            "x0": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "y0": rng.standard_normal(n),
        }
    )
    out = po.ModelBank([_spec([10.0, 0.0], halflife=300.0)]).fit_predict(df)
    sel = out["m"].struct.field("lam_selected_y0").to_numpy().astype(float)
    sel = sel[np.isfinite(sel)]
    assert (sel == 10.0).mean() > 0.8


def test_chunk_invariance():
    df, _ = synthetic(seed=34, n_groups=2, n_rows=180, k=3, null_frac=0.0)
    spec = _spec([0.5, 0.0], group="group", halflife=200.0)
    one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")

    bank = po.ModelBank([spec])
    many = (
        pl.concat([bank.fit_predict(df.slice(i, 25)) for i in range(0, df.height, 25)])
        .select("m")
        .unnest("m")
    )
    keep = [c for c in one.columns if not c.startswith("coef")]
    assert one.select(keep).equals(many.select(keep), null_equal=True)


def test_expression_equals_bank():
    df, _ = synthetic(seed=34, n_groups=2, n_rows=180, k=3, null_frac=0.0)
    spec = _spec([0.5, 0.0], group="group", halflife=200.0)
    one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
    keep = [c for c in one.columns if not c.startswith("coef")]
    expr = df.select(
        pl.col("y0")
        .online.lasso(
            features=["x0", "x1", "x2"],
            lasso_path=[0.5, 0.0],
            halflife=200.0,
            max_rows_between_solves=1,
            min_periods=10.0,
        )
        .over("group")
    ).unnest("y0")
    assert one.select(keep).equals(expr.select(keep), null_equal=True)


def test_path_must_be_decreasing():
    with pytest.raises(ValueError, match="decreasing"):
        _spec([0.1, 1.0])
